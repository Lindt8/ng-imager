from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Literal

import numpy as np

from ngimager.config.load import load_config
from ngimager.config.schemas import Config
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
from ngimager.sim.synth import synth_neutron_events_point_source
from ngimager.vis.hdf import save_summed_png


def _iter_source_events(cfg: Config) -> Iterable[Event]:
    """
    Unified event source:

    - cfg.run.source_type in {"dt", "cf252", "proton_center", "phits"} selects
      how En0 / prior is handled (for now, we mostly use it for synthetic).
    - cfg.io.input is passed to the adapter for real data.
    """
    if cfg.run.source_type == "proton_center":
        # # Synthetic point source, mainly for testing.
        # lut_registry = build_lut_registry(cfg.energy.lut_paths)
        # # crude choice: first proton LUT for M600
        # lut_name = next(iter(lut_registry.keys()))
        # lut = lut_registry[lut_name]
        lut_registry = build_lut_registry(cfg.energy.lut_paths)
        # Choose a material/species LUT:
        #   - Prefer built-in M600/proton if present.
        #   - Otherwise fall back to the first available material/species.
        if "M600" in lut_registry and "proton" in lut_registry["M600"]:
            material = "M600"
            species = "proton"
        else:
            # Fallback: arbitrary first combo
            material = next(iter(lut_registry.keys()))
            species = next(iter(lut_registry[material].keys()))
        lut = lut_registry[material][species]

        source = np.array(cfg.prior.point, dtype=float)
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

        events = synth_neutron_events_point_source(
            n_events=cfg.synth.n_events,
            source_xyz_cm=source,
            En0_MeV=cfg.energy.fixed_En_MeV,
            lut=lut,
            plane=plane,
            #material="M600",
            material=material,
            s12_cm=10.0,
        )
        yield from events
    else:
        # Real data path (ROOT/PHITS etc.)
        adapter = make_adapter(cfg.io.adapter)
        yield from adapter.iter_events(cfg.io.input)


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
            if j < 5 and cfg.run.diagnostics:
                print(f"[cones] Skipping event {j} during ordered/validate: {exc}")
            continue

        if not isinstance(ev, NeutronEvent):
            # TODO: gamma cones to be implemented; for now we ignore gammas.
            continue

        try:
            cone = build_cone_from_neutron(ev, energy_model, scatter_nucleus="H")
        except Exception as exc:
            if j < 5 and cfg.run.diagnostics:
                print(f"[cones] Failed to build cone from event {j}: {exc}")
            continue

        cones.append(cone)

        # Fast-mode: optional conservative cap on number of cones
        if (cfg.run.mode == "fast") and (cfg.run.max_cones is not None):
            if len(cones) >= cfg.run.max_cones:
                if cfg.run.diagnostics: 
                    print(f"[cones] Reached max_cones={cfg.run.max_cones}, stopping cone build.")
                break

    if not cones:
        if cfg.run.diagnostics:
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
    if cfg.run.diagnostics:
        print(f"[cones] Built {len(cones)} cones from {len(events)} events")
    return cone_ids, apex_xyz_cm, axis_xyz, theta_rad


def run_pipeline(
    cfg_path: str,
    mode_override: Literal["fast", "list"] | None = None,
) -> Path:
    """
    Unified imaging pipeline.

    Parameters
    ----------
    cfg_path : str
        Path to TOML configuration file.
    mode_override : "fast" | "list" | None
        If given, override cfg.run.mode. This is what fastmode.py / listmode.py use.

    Returns
    -------
    Path to written HDF5 file.
    """
    cfg_path = str(cfg_path)
    cfg = load_config(cfg_path)

    if mode_override is not None:
        cfg.run.mode = mode_override

    mode = cfg.run.mode
    list_mode = (mode == "list")

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
    out_path = Path(cfg.io.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = write_init(str(out_path), cfg_path, cfg, plane)

    # Events
    events = list(_iter_source_events(cfg))
    if cfg.run.diagnostics: 
        print(f"[pipeline] Got {len(events)} events")
    if events:
        first = events[0]
        h1 = getattr(first, "h1", None)
        h2 = getattr(first, "h2", None)
        if cfg.run.diagnostics:
            print(f"[pipeline] First event type: {type(first).__name__}")
            print(f"[pipeline] First event h1: {h1!r}")
            print(f"[pipeline] First event h2: {h2!r}")
            if h1 is not None:
                print(f"[pipeline] h1.r = {getattr(h1, 'r', None)}, t_ns={h1.t_ns}, L={h1.L}")
            if h2 is not None:
                print(f"[pipeline] h2.r = {getattr(h2, 'r', None)}, t_ns={h2.t_ns}, L={h2.L}")

    # Cones from events
    cone_ids, apex_xyz_cm, axis_xyz, theta_rad = _build_cones_from_events(cfg, events, plane)
    if cfg.run.diagnostics:
        print(f"[pipeline] Built {len(cone_ids)} cones")
        if len(cone_ids):
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
        list_mode=list_mode,
        # uncertainty_mode stays at default "off" for now
        progress=cfg.run.progress,
    )

    if cfg.run.diagnostics:
        print("[pipeline] Recon summed image stats:",
              "min=", float(recon.summed.min()),
              "max=", float(recon.summed.max()),
              "sum=", float(recon.summed.sum()),
              "shape=", recon.summed.shape)

    # Summed image
    img = recon.summed.astype(np.float32)
    write_summed(f, "n", img)

    # List-mode extras
    if list_mode:
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
            if getattr(cfg.run, "diagnostics", False):
                print(f"[pipeline] Wrote PNG {out_png} from {dset}")
        except Exception as e:
            if getattr(cfg.run, "diagnostics", False):
                print(f"[pipeline] PNG export failed: {e!r}")

    return out_path
