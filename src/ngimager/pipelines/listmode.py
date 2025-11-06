from __future__ import annotations
from pathlib import Path
import numpy as np
from ..config.load import load_config
from ..geometry.plane import Plane
from ..io.lm_store import write_init, write_summed, write_cones, write_lm_indices
from ..imaging.sbp import reconstruct_sbp, Cone
#from ..physics.hits import fake_hits
from ..io.lut import build_lut_registry
from ..sim.synth import synth_neutron_events_point_source
from ngimager.io.adapters import make_adapter
from ngimager.physics.events import NeutronEvent, GammaEvent
from ..physics.energy_strategies import make_energy_strategy
from ..io.lut import build_lut_registry
from ..physics.cones import build_cone_from_neutron
from ngimager.vis.hdf import save_summed_png
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=False)
except RuntimeError:
    pass


def _demo_cones() -> list[Cone]:
    O = np.array([0,0,0], dtype=float)
    D = np.array([0,0,1], dtype=float)
    theta = np.deg2rad(25.0)
    return [Cone(O, D, theta), Cone(O + [2,0,0], D, np.deg2rad(15.0))]

def iter_source_events(cfg, path):
    if cfg.io.source == "synth":
        # existing synthetic generator
        yield from synth_neutron_events_point_source(
            n_events=cfg.synth.n_events,
            source_xyz_cm=np.array(cfg.synth.source_xyz_cm, dtype=float),
            detector_xyz_cm=np.array(cfg.synth.detector_xyz_cm, dtype=float),
            lut=lut_registry[cfg.energy.lut_name],
            material=cfg.detectors.material
        )
    else:
        # real data path (ROOT/PHITS)
        adapter = make_adapter(cfg.io.adapter)  # e.g., {type="root", style="Joey", default_material="M600"}
        yield from adapter.iter_events(cfg.io.input)

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    pl = Plane.from_cfg(
        origin=cfg.plane.origin, normal=cfg.plane.normal,
        u_min=cfg.plane.u_min, u_max=cfg.plane.u_max, du=cfg.plane.du,
        v_min=cfg.plane.v_min, v_max=cfg.plane.v_max, dv=cfg.plane.dv,
        eu=cfg.plane.eu, ev=cfg.plane.ev
    )
    Path(cfg.io.output).parent.mkdir(parents=True, exist_ok=True)
    h5 = write_init(cfg.io.output, cfg_path, cfg, pl)

    # --- build demo events -------------------------------------------------------
    # # cones = _demo_cones()
    # hits = fake_hits(2)
    # nevt = NeutronEvent(hits[0], hits[1])
    # lut_reg = build_lut_registry(cfg.energy.lut_paths, cfg_path)
    # E_model = make_energy_strategy(cfg.energy, lut_reg)
    # cones = [build_cone_from_neutron(nevt, E_model)]

    # Build LUT registry (for Edep1 via ELUT) and energy model (currently unused here but kept for future)
    lut_reg = build_lut_registry(cfg.energy.lut_paths, cfg_path)

    # Choose material/species; here we use M600/proton built-in by default
    lut_M600_p = lut_reg["M600"]["proton"]

    # Synthesize a batch of neutron events from a point source
    source = np.array([0.0, 0.0, -500.0])  # matches default prior in configs
    En0 = 14.1  # MeV (change if you want)
    # events = synth_neutron_events_point_source(
    #    n_events=3000,  # adjust to taste
    #    source_xyz_cm=source,
    #    En0_MeV=En0,
    #    lut=lut_M600_p,
    #    plane=pl,
    #    material="M600",
    #    s12_cm=10.0,
    # )
    events = list(iter_source_events(cfg, cfg.io.input))  # or stream in chunks

    # Convert to cones (H scattering by default)
    cones = [build_cone_from_neutron(ev, energy_model=make_energy_strategy(cfg.energy, lut_reg), scatter_nucleus="H")
             for ev in events]

    res = reconstruct_sbp(
        cones,
        pl,
        list_mode=True,
        uncertainty_mode="off",
        workers=cfg.run.workers,
        chunk_cones=cfg.run.chunk_cones,
        progress=cfg.run.progress,
    )
    write_summed(h5, "n", res.summed)
    # Save cone parameters (demo)
    apex = np.stack([c.apex for c in cones]).astype("f4")
    direc = np.stack([c.dir  for c in cones]).astype("f4")
    theta = np.array([c.theta for c in cones], dtype="f4")
    write_cones(h5, "n", apex, direc, theta, None)
    write_lm_indices(h5, "n", res.lm_indices or [])
    h5.close()
    print(f"Wrote {cfg.io.output}")

    if getattr(cfg, "vis", None) and getattr(cfg.vis, "export_png_on_write", True):
        try:
            out_png = save_summed_png(cfg.io.output, dataset=getattr(cfg.vis, "summed_dataset", "/images/summed"))
            print(f"[viz] wrote {out_png}")
        except Exception as e:
            print(f"[viz] failed to export PNG: {e}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
