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
from ngimager.physics.cones import cone_from_neutron
from ngimager.physics.events import NeutronEvent, GammaEvent, Event
from ngimager.physics.energy_strategies import make_energy_strategy
from ngimager.physics.priors import make_prior
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
        # Synthetic point source, mainly for testing.
        lut_registry = build_lut_registry()
        # crude choice: first proton LUT for M600
        lut_name = next(iter(lut_registry.keys()))
        lut = lut_registry[lut_name]

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
            material="M600",
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn events into cone geometry arrays for SBP.

    Returns
    -------
    cone_ids, apex_xyz_cm, axis_xyz, theta_rad
    """
    lut_registry = build_lut_registry()
    energy_model = make_energy_strategy(cfg.energy, lut_registry=lut_registry)
    prior = make_prior(cfg.prior)

    cones = []
    for j, ev in enumerate(events):
        # enforce time ordering & sanity without crashing the whole run
        try:
            ev = ev.ordered(strict=False)  # returns same type
            ev.validate(strict=False)
        except Exception:
            # Skip pathological events
            continue

        if isinstance(ev, NeutronEvent):
            cone = cone_from_neutron(ev, energy_model, prior)
            cones.append(cone)
        else:
            # TODO: gamma cones to be implemented; for now we ignore gammas.
            continue

        # Fast-mode: optional conservative cap on number of cones
        if (cfg.run.mode == "fast") and (cfg.run.max_cones is not None):
            if len(cones) >= cfg.run.max_cones:
                break

    if not cones:
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    cone_ids = np.arange(len(cones), dtype=np.uint32)
    apex_xyz_cm = np.stack([c.apex for c in cones], axis=0).astype(np.float32)
    axis_xyz = np.stack([c.axis for c in cones], axis=0).astype(np.float32)
    theta_rad = np.array([c.theta for c in cones], dtype=np.float32)
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

    # Cones from events
    cone_ids, apex_xyz_cm, axis_xyz, theta_rad = _build_cones_from_events(cfg, events)

    # SBP reconstruction
    from ngimager.imaging.sbp import reconstruct_sbp

    recon = reconstruct_sbp(
        plane=plane,
        apex_xyz_cm=apex_xyz_cm,
        axis_xyz=axis_xyz,
        theta_rad=theta_rad,
        workers=cfg.run.workers,
        chunk_cones=cfg.run.chunk_cones,
        list_mode=list_mode,
        jit=cfg.run.jit,
        progress=cfg.run.progress,
    )

    # Summed image
    img = recon.image.astype(np.float32)
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
        save_summed_png(str(out_path))

    return out_path
