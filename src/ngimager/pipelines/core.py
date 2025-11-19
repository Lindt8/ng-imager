'''
The CLI is implemented via Typer, so the script behaves like a simple one-argument command:

    - argument = path to TOML config file

The pipeline will:

    - Load the TOML config
    - Detect the adapter (PHITS, ROOT, HDF5 restart)
    - Shape/validate hits → events
    - Build cones (now neutron + gamma depending on run.neutrons, run.gammas)
    - Run SBP imaging
    - Write unified HDF5 output

You can always show help via:

    - `python -m ngimager.pipelines.core --help`

Example run commands from project root:

python -m ngimager.pipelines.core path/to/config.toml

python -m ngimager.pipelines.core examples/configs/phits_usrdef_simple.toml

python -m ngimager.pipelines.core .\examples\configs\phits_usrdef_simple.toml


'''
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Literal, Optional, Dict
import typer

import numpy as np

from ngimager.config.load import load_config
from ngimager.config.schemas import Config, RunCfg
from ngimager.geometry.plane import Plane
from ngimager.io.adapters import make_adapter
from ngimager.io.lm_store import (
    write_init,
    write_summed,
    write_cones,
    write_lm_indices,
    write_events_hits,
)
from ngimager.io.lut import build_lut_registry
from ngimager.imaging.sbp import reconstruct_sbp
from ngimager.physics.cones import build_cone_from_neutron, build_cone_from_gamma
from ngimager.physics.events import NeutronEvent, GammaEvent, Event
from ngimager.physics.energy_strategies import make_energy_strategy
from ngimager.physics.priors import make_prior, Prior
from ngimager.filters.shapers import shape_events_for_cones, ShapeConfig
from ngimager.filters.to_typed_events import shaped_to_typed_events
from ngimager.filters.hit_filters import apply_hit_filters, is_reconstructable
from ngimager.vis.hdf import save_summed_png


def _iter_source_events(cfg: Config) -> Iterable[Event]:
    """
    Unified event source for real data.

    For now, all supported source types use the configured adapter to read
    events from cfg.io.input_path. Synthetic / toy sources have been removed
    from the production pipeline; use dedicated dev/test scripts instead.

    - cfg.io.adapter.kind selects ROOT vs PHITS-style adapters.
    - cfg.io.input_path is passed to the adapter for real data.
    """
    adapter_cfg: Dict[str, object] = dict(cfg.io.adapter)

    det_cfg = getattr(cfg, "detectors", None)
    if det_cfg is not None:
        mat_map = getattr(det_cfg, "material_map", None)
        if mat_map and "material_map" not in adapter_cfg:
            adapter_cfg["material_map"] = mat_map

        default_mat = getattr(det_cfg, "default_material", None)
        if default_mat and "default_material" not in adapter_cfg:
            adapter_cfg["default_material"] = default_mat

    adapter = make_adapter(adapter_cfg)
    return adapter.iter_events(str(cfg.io.input_path))


def _build_cones_from_events(
    cfg: Config,
    events: Sequence[Event],
    plane: Plane,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn events into cone geometry arrays for SBP.

    Returns
    -------
    cone_ids, apex_xyz_cm, axis_xyz, theta_rad
    """
    lut_registry = build_lut_registry(cfg.energy.lut_paths)
    energy_model = make_energy_strategy(cfg.energy, lut_registry=lut_registry)
    # Prior for cone selection (especially important for gammas).
    # make_prior will interpret cfg.prior and may fall back to the imaging
    # plane center if the user hasn't supplied an explicit point.
    prior = make_prior(cfg.prior.model_dump(), plane)
    # prior = make_prior(cfg.prior.dict(), plane) # In case of older Pydantic/Config semantics, fall back to dict()


    cones = []
    for j, ev in enumerate(events):
        # enforce time ordering & sanity without crashing the whole run
        try:
            ev = ev.ordered()  # returns same type
            ev.validate(strict=False)
        except Exception as exc:
            if j < 5 and cfg.run.diagnostics_level >= 2:
                print(f"[cones] Skipping event {j} during ordered/validate: {exc}")
            continue

        # ---- Species-aware cone building ----
        # Respect run.neutrons / run.gammas toggles and choose the
        # appropriate cone builder. For now each event yields at most
        # one cone.
        if isinstance(ev, NeutronEvent):
            if not cfg.run.neutrons:
                continue
            species = "n"
        elif isinstance(ev, GammaEvent):
            if not cfg.run.gammas:
                continue
            species = "g"
        else:
            # Unknown event type – ignore for now
            continue

        try:
            if species == "n":
                # Neutrons: build proton vs carbon recoil hypotheses and, when
                # a plane/prior are available, select the most plausible one
                # using the same Δ = |φ − θ| scoring as for gammas.
                # The [energy].force_proton_recoils flag bypasses this and
                # treats all recoils as protons.
                cone = build_cone_from_neutron(
                    ev,
                    energy_model,
                    plane=plane,
                    prior=prior,
                    force_proton=cfg.energy.force_proton_recoils,
                )
            else:
                # Gammas: Compton-based cone construction (uses Hit.L
                # as deposited energy; energy_model is passed for future
                # gamma LUT support even if not used now).
                # Gamma cone construction with full permutation search,
                # axis-towards-plane test, and Δ = |φ - θ| prior scoring.
                cone = build_cone_from_gamma(ev, energy_model, plane=plane, prior=prior)
        except Exception as exc:
            if j < 5 and cfg.run.diagnostics_level >= 2:
                print(f"[cones] Failed to build cone from event {j}: {exc}")
            continue

        cones.append(cone)

        # Fast-mode: optional conservative cap on number of cones
        if cfg.run.fast and (cfg.run.max_cones is not None):
            if len(cones) >= cfg.run.max_cones:
                if cfg.run.diagnostics_level >= 1:
                    print(f"[cones] Reached max_cones={cfg.run.max_cones}, stopping cone build.")
                break

    if not cones:
        if cfg.run.diagnostics_level >= 1:
            print("[cones] No cones were successfully built.")
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    cone_ids = np.arange(len(cones), dtype=np.uint32)
    apex_xyz_cm = np.stack([c.apex for c in cones], axis=0).astype(np.float32)
    axis_xyz = np.stack([c.dir for c in cones], axis=0).astype(np.float32)
    theta_rad = np.array([c.theta for c in cones], dtype=np.float32)
    if cfg.run.diagnostics_level >= 1:
        print(f"[cones] Built {len(cones)} cones from {len(events)} events")
    return cone_ids, apex_xyz_cm, axis_xyz, theta_rad


def run_pipeline(
    cfg_path: str,
    *,
    fast: Optional[bool] = None,
    list_mode: Optional[bool] = None,
    neutrons: Optional[bool] = None,
    gammas: Optional[bool] = None,
) -> Path:
    """
    Orchestrate the full pipeline from a TOML config file.

    CLI flags (--fast/--list/--neutrons/--no-neutrons/--gammas/--no-gammas)
    override the corresponding [run] fields when not None.

    Parameters
    ----------
    cfg_path : str
        Path to TOML configuration file.

    Returns
    -------
    Path to written HDF5 file.
    """
    #cfg_path = str(cfg_path)
    cfg = load_config(cfg_path)

    # ---- apply CLI overrides on top of TOML ----
    if fast is not None:
        cfg.run.fast = fast
    if list_mode is not None:
        cfg.run.list = list_mode
    if neutrons is not None:
        cfg.run.neutrons = neutrons
    if gammas is not None:
        cfg.run.gammas = gammas

    # Conveniences
    diag_level = cfg.run.diagnostics_level
    verbose = diag_level >= 2

    # Basic logging
    if diag_level >= 1:
        print(f"[run] config = {cfg_path}")
        print(f"[run] neutrons={cfg.run.neutrons} gammas={cfg.run.gammas} "
              f"fast={cfg.run.fast} list={cfg.run.list}")
        print(f"[run] input={cfg.io.input_path} -> output={cfg.io.output_path}")

    # Imaging plane
    plane = Plane.from_cfg(
        cfg.plane.origin,
        cfg.plane.normal,
        cfg.plane.u_min,
        cfg.plane.u_max,
        cfg.plane.du,
        cfg.plane.v_min,
        cfg.plane.v_max,
        cfg.plane.dv,
    )

    # HDF5 output
    out_path = Path(cfg.io.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = write_init(str(out_path), cfg_path, cfg, plane)

    # Shared counters for this run
    counters: Dict[str, int] = {}

    # ---- Stage 1: adapter → raw events → hit-level filters → is_reconstructable ----
    if cfg.io.input_format == "phits_usrdef":
        from ngimager.filters.shapers import shape_events_for_cones, ShapeConfig
        from ngimager.filters.to_typed_events import shaped_to_typed_events
        from ngimager.filters.hit_filters import apply_hit_filters, is_reconstructable

        if diag_level >= 1:
            print("[pipeline] Using staged PHITS path: raw events → hits → shaped → typed")

        # Build adapter config, injecting detector-level material info.
        adapter_cfg: Dict[str, object] = dict(cfg.io.adapter)

        det_cfg = getattr(cfg, "detectors", None)
        if det_cfg is not None:
            mat_map = getattr(det_cfg, "material_map", None)
            if mat_map and "material_map" not in adapter_cfg:
                adapter_cfg["material_map"] = mat_map

            default_mat = getattr(det_cfg, "default_material", None)
            if default_mat and "default_material" not in adapter_cfg:
                adapter_cfg["default_material"] = default_mat

        adapter = make_adapter(adapter_cfg)

        raw_events_after_filters = []

        for ev in adapter.iter_raw_events(str(cfg.io.input_path)):
            hits = list(ev.get("hits", []))

            # Normalize event_type to 'n' / 'g' where possible
            et_raw = str(ev.get("event_type", "")).lower()
            if et_raw.startswith("n"):
                et = "n"
            elif et_raw.startswith("g"):
                et = "g"
            else:
                et = None

            counters["raw_events_total"] = counters.get("raw_events_total", 0) + 1

            # Hit-level filters
            filtered_hits = apply_hit_filters(hits, cfg.filters, counters, particle_type=et)

            # Early reconstructability decision (also updates *_unreconstructable counters)
            if not is_reconstructable(filtered_hits, cfg.filters, counters, event_type=et):
                continue

            if not filtered_hits:
                # Should be caught by is_reconstructable, but guard anyway
                continue

            ev2 = dict(ev)
            ev2["hits"] = filtered_hits
            if et is not None:
                ev2["event_type"] = et
            raw_events_after_filters.append(ev2)

        if diag_level >= 1:
            print(
                "[hits] raw_events_total={total} "
                "raw_events_after_filters={surv} "
                "raw_events_rejected_unreconstructable={rej}".format(
                    total=counters.get("raw_events_total", 0),
                    surv=len(raw_events_after_filters),
                    rej=counters.get("raw_events_rejected_unreconstructable", 0),
                )
            )

        # ---- Stage 2: Hits → shaped events → typed events ----
        shaped_events, shape_diag = shape_events_for_cones(
            raw_events_after_filters,
            ShapeConfig(),
            counters=counters,
        )

        if diag_level >= 1:
            print(
                "[shaper] total_events_in={total} "
                "shaped_n={sn} shaped_g={sg}".format(
                    total=shape_diag.total_events,
                    sn=shape_diag.shaped_neutron,
                    sg=shape_diag.shaped_gamma,
                )
            )

        if diag_level >= 2 and shaped_events:
            print(f"[shaper] Example shaped events (up to first 3):")
            for se in shaped_events[:3]:
                ts = [h.t_ns for h in se.hits]
                Ls = [h.L for h in se.hits]
                types = [h.type for h in se.hits]
                print(
                    f"    species={se.species} "
                    f"n_hits={len(se.hits)} "
                    f"types={types} "
                    f"t_ns={ts} "
                    f"L={Ls}"
                )

        events = shaped_to_typed_events(
            shaped_events,
            order_time=True,
        )

        # ---- Typed events diagnostics (Stage: hits → shaped → typed) ----
        # Species breakdown based on the typed-event objects themselves.
        # This does not assume any particular event-level filters yet.
        from ngimager.physics.events import NeutronEvent, GammaEvent
        n_n = sum(isinstance(ev, NeutronEvent) for ev in events)
        n_g = sum(isinstance(ev, GammaEvent) for ev in events)
        counters["events_typed_total"] = n_n + n_g
        counters["events_typed_n"] = n_n
        counters["events_typed_g"] = n_g
        # Placeholder for future event-level rejections (event filters)
        # so that a later filter stage can do:
        #   counters["events_rejected_filters"] = ...
        events_rejected = counters.get("events_rejected_filters", 0)

        if diag_level >= 1:
            print(
                "[events] typed_total={total} "
                "typed_n={n} typed_g={g} "
                "events_rejected_filters={rej}".format(
                    total=len(events),
                    n=n_n,
                    g=n_g,
                    rej=events_rejected,
                )
            )

    else:
        # For non-PHITS sources, keep the existing direct typed-event path.
        events = list(_iter_source_events(cfg))

    # Existing diagnostics on typed events
    if diag_level >= 1:
        print(f"[pipeline] Got {len(events)} events")
    if events:
        first = events[0]
        h1 = getattr(first, "h1", None)
        h2 = getattr(first, "h2", None)
        if diag_level >= 2:
            print(f"[pipeline] First event type: {type(first).__name__}")
            print(f"[pipeline] First event h1: {h1!r}")
            print(f"[pipeline] First event h2: {h2!r}")
            if h1 is not None:
                print(f"[pipeline] h1.r = {getattr(h1, 'r', None)}, t_ns={h1.t_ns}, L={h1.L}")
            if h2 is not None:
                print(f"[pipeline] h2.r = {getattr(h2, 'r', None)}, t_ns={h2.t_ns}, L={h2.L}")
            for ev in events[:3]:
                species = "n" if isinstance(ev, NeutronEvent) else "g" if isinstance(ev, GammaEvent) else "?"
                hlist = [getattr(ev, name) for name in ("h1", "h2", "h3") if hasattr(ev, name)]
                ts = [h.t_ns for h in hlist]
                Ls = [h.L for h in hlist]
                types = [h.type for h in hlist]
                print(
                    f"    {species}-event "
                    f"n_hits={len(hlist)} "
                    f"types={types} "
                    f"t_ns={ts} "
                    f"L={Ls}"
                )

    # Cones from events
    cone_ids, apex_xyz_cm, axis_xyz, theta_rad = _build_cones_from_events(cfg, events, plane)
    if diag_level >= 1:
        print(f"[pipeline] Built {len(cone_ids)} cones")
        if len(cone_ids) and diag_level >= 2:
            print("[pipeline] Example cone apex:", apex_xyz_cm[0])
            print("[pipeline] Example cone dir:", axis_xyz[0])
            print("[pipeline] Example cone theta[deg]:", np.degrees(theta_rad[0]))

    # SBP reconstruction
    from ngimager.imaging.sbp import reconstruct_sbp, Cone

    # Build Cone objects from the geometry arrays
    cones_for_sbp: list[Cone] = [
        Cone(apex=apex_xyz_cm[i], direction=axis_xyz[i], theta=float(theta_rad[i]))
        for i in range(len(cone_ids))
    ]

    recon = reconstruct_sbp(
        cones=cones_for_sbp,
        plane=plane,
        workers=cfg.run.workers,
        chunk_cones=cfg.run.chunk_cones,
        list_mode=cfg.run.list,
        # uncertainty_mode stays at default "off" for now
        progress=cfg.run.progress,
    )

    if diag_level >= 1:
        print("[pipeline] Recon summed image stats:",
              "min=", float(recon.summed.min()),
              "max=", float(recon.summed.max()),
              "sum=", float(recon.summed.sum()),
              "shape=", recon.summed.shape)

    # Summed image
    img = recon.summed.astype(np.float32)
    write_summed(f, "n", img)
    
    # Per-cone geometry
    write_cones(f, cone_ids, apex_xyz_cm, axis_xyz, theta_rad)
    
    # Per-event / per-hit physics (this links back via /lm/events dataset)
    write_events_hits(f, events)

    # List-mode extras
    if cfg.run.list:
        # LM pixel indices
        lm_indices = recon.lm_indices or []
        write_lm_indices(f, lm_indices)
        
    f.close()

    # Optional PNG export
    if getattr(cfg, "vis", None) and getattr(cfg.vis, "export_png_on_write", False):
        try:
            dset = getattr(cfg.vis, "summed_dataset", "/images/summed/n")
            out_png = save_summed_png(str(out_path), dataset=dset)
            if cfg.run.diagnostics_level >= 1:
                print(f"[pipeline] Wrote PNG {out_png} from {dset}")
        except Exception as e:
            if cfg.run.diagnostics_level >= 1:
                print(f"[pipeline] PNG export failed: {e!r}")

    return out_path


# ---------------------------------------------------------------------------
# Unified CLI entry point
# ---------------------------------------------------------------------------

app = typer.Typer(help="Unified NOVO imaging pipeline (ngimager.pipelines.core)")


@app.command()
def main(
    cfg_path: str = typer.Argument(
        ...,
        help="Path to TOML config file",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Override [run].fast = true (use aggressive fast settings)",
    ),
    list_mode: bool = typer.Option(
        False,
        "--list",
        help="Override [run].list = true (enable list-mode image output)",
    ),
    neutrons: Optional[bool] = typer.Option(
        None,
        "--neutrons / --no-neutrons",
        help="Enable or disable neutron processing; overrides [run].neutrons when set",
    ),
    gammas: Optional[bool] = typer.Option(
        None,
        "--gammas / --no-gammas",
        help="Enable or disable gamma processing; overrides [run].gammas when set",
    ),
):
    """
    Run the unified ng-imager pipeline for a single config.
    """
    out_path = run_pipeline(
        cfg_path,
        fast=fast if fast else None,
        list_mode=list_mode if list_mode else None,
        neutrons=neutrons,
        gammas=gammas,
    )
    typer.echo(str(out_path))


if __name__ == "__main__":
    app()
