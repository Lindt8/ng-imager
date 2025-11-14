from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Literal, Optional
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
from ngimager.physics.cones import build_cone_from_neutron
from ngimager.physics.events import NeutronEvent, GammaEvent, Event
from ngimager.physics.energy_strategies import make_energy_strategy
from ngimager.physics.priors import make_prior, Prior
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
    adapter = make_adapter(cfg.io.adapter)
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
    prior = make_prior(cfg.prior.model_dump(), plane)

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

        if not isinstance(ev, NeutronEvent):
            # TODO: gamma cones to be implemented; for now we ignore gammas.
            continue

        try:
            cone = build_cone_from_neutron(ev, energy_model, scatter_nucleus="H")
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

    # Events
    events = list(_iter_source_events(cfg))
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

    # List-mode extras
    if cfg.run.list:
        # LM pixel indices
        lm_indices = recon.lm_indices or []
        write_lm_indices(f, lm_indices)

        # Per-cone geometry
        write_cones(f, cone_ids, apex_xyz_cm, axis_xyz, theta_rad)

        # Per-event / per-hit physics (this links back via /lm/events dataset)
        write_events_hits(f, events)

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
