from __future__ import annotations
from pathlib import Path
import numpy as np
from ..config.load import load_config
from ..geometry.plane import Plane
from ..io.lm_store import write_init, write_summed
from ..imaging.sbp import reconstruct_sbp, Cone
from ..physics.hits import fake_hits
from ..physics.events import NeutronEvent
from ..physics.energy_strategies import make_energy_strategy
from ..io.lut import build_lut_registry
from ..physics.cones import build_cone_from_neutron
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=False)
except RuntimeError:
    pass


def _demo_cones() -> list[Cone]:
    # Temporary demo: one cone facing +z with a modest theta
    O = np.array([0,0,0], dtype=float)
    D = np.array([0,0,1], dtype=float)
    theta = np.deg2rad(20.0)
    return [Cone(O, D, theta)]

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
    # cones = _demo_cones()
    hits = fake_hits(2)
    nevt = NeutronEvent(hits[0], hits[1])
    lut_reg = build_lut_registry(cfg.energy.lut_paths, cfg_path)
    E_model = make_energy_strategy(cfg.energy, lut_reg)
    cones = [build_cone_from_neutron(nevt, E_model)]

    res = reconstruct_sbp(
        cones,
        pl,
        list_mode=False,
        uncertainty_mode="off",
        workers=cfg.run.workers,
        chunk_cones=cfg.run.chunk_cones,
        progress=cfg.run.progress,
    )
    write_summed(h5, "n", res.summed)
    h5.close()
    print(f"Wrote {cfg.io.output}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])

