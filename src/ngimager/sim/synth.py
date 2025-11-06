from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterable
from ..physics.hits import Hit
from ..physics.events import NeutronEvent
from ..io.lut import LUT
from ..physics.kinematics import C_CM_PER_NS, M_N_MEV
from ..geometry.plane import Plane

def speed_from_En_MeV(En: float) -> float:
    gamma = 1.0 + En / M_N_MEV
    beta = np.sqrt(1.0 - 1.0/(gamma*gamma))
    return beta * C_CM_PER_NS

def L_for_E(lut: LUT, E_target: float) -> float:
    E = lut.E; L = lut.L
    inc = E[-1] >= E[0]
    if not inc:
        E = E[::-1]; L = L[::-1]
    e = np.clip(E_target, E.min(), E.max())
    return float(np.interp(e, E, L))

def synth_neutron_events_point_source(
    n_events: int,
    source_xyz_cm: np.ndarray,
    En0_MeV: float,
    lut: LUT,
    plane: Plane,
    material: str = "M600",
    s12_cm: float = 10.0,
    rng: np.random.Generator | None = None,
) -> list[NeutronEvent]:
    """
    Generate 2-hit neutron events aimed so cones intersect the imaging plane:
      - u is chosen with u·n_plane > 0 (ray toward plane normal)
      - r1 placed just IN FRONT of the plane (so plane is behind apex wrt -u)
      - r2 = r1 + s12*u
    Axis used by the recon is Dhat = (r1 - r2) = -u, which points back toward the plane.
    """
    rng = rng or np.random.default_rng()
    events: list[NeutronEvent] = []

    n = plane.n
    P0 = plane.P0
    v0 = speed_from_En_MeV(En0_MeV)

    # Pick an Edep1 in-domain midrange for stability
    E_mid = float(0.5 * (lut.E.min() + lut.E.max()))
    E_sd  = max(1e-3, 0.1 * E_mid)

    for _ in range(n_events):
        # Choose u with positive projection onto plane normal (toward plane)
        while True:
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            if u @ n > 0.3:  # tilt toward plane normal
                break

        # Put apex r1 a little *in front of* the plane: (P0 - r1)·n > 0  => r1 is "ahead" of plane along +n
        # Let d be 10..40 cm in front of plane
        d_front = rng.uniform(10.0, 40.0)
        # Start at the plane origin and go *forward* along +n by d_front, then add small lateral jitter in-plane
        jitter_u = rng.uniform(-0.3, 0.3) * (plane.u_max - plane.u_min)
        jitter_v = rng.uniform(-0.3, 0.3) * (plane.v_max - plane.v_min)
        r1 = P0 + d_front * n + jitter_u * plane.eu + jitter_v * plane.ev

        # Second hit further along the ray *away* from the plane (same direction as u)
        r2 = r1 + s12_cm * u

        # Times: source -> r1 at En0; then r1 -> r2 at E' after loss
        t1 = np.linalg.norm(r1 - source_xyz_cm) / v0

        Edep1 = float(np.clip(rng.normal(E_mid, E_sd), lut.E.min(), lut.E.max()))
        Lval = L_for_E(lut, Edep1)
        Eprime = max(En0_MeV - Edep1, 0.5)
        v1 = speed_from_En_MeV(Eprime)
        t2 = t1 + (np.linalg.norm(r2 - r1) / v1)

        h1 = Hit(det_id=0, r=r1, t_ns=t1, L=Lval, material=material)
        h2 = Hit(det_id=1, r=r2, t_ns=t2, L=0.0, material=material)
        events.append(NeutronEvent(h1, h2))

    return events
