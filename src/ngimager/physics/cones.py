from __future__ import annotations
import numpy as np
from typing import Literal
from .events import NeutronEvent, GammaEvent
from ..imaging.sbp import Cone
from .energy_strategies import EnergyStrategy
from .kinematics import neutron_theta_from_hits

def build_cone_from_neutron(
    ev: NeutronEvent,
    energy_model: EnergyStrategy,
    scatter_nucleus: Literal["H","C"] = "H",
) -> Cone:
    """
    Build a neutron cone using:
      apex O = X1,
      axis Dhat = (X1 - X2) / ||X1 - X2||  (points from 2nd -> 1st, per primer),
      theta from primer equations using Edep1 (via ELUT) and E' (via ToF).
    """
    ev.validate()
    r1, r2 = ev.h1.r.astype(float), ev.h2.r.astype(float)
    t1, t2 = float(ev.h1.t_ns), float(ev.h2.t_ns)

    # Direction from 2nd -> 1st (matches primer)
    D = r1 - r2
    L = np.linalg.norm(D)
    if L <= 0:
        raise ValueError("Zero baseline between hits.")
    Dhat = D / L
    apex = r1.copy()

    # Edep1 from energy strategy (must map material/species; use proton band by default)
    # NOTE: Even if the overall "incident energy strategy" is FixedEn, we still need Edep1 for the angle.
    Edep1, _ = energy_model.first_scatter_energy(ev.h1, ev.h2, ev.h1.material, "proton")

    theta = neutron_theta_from_hits(r1, t1, r2, t2, Edep1_MeV=Edep1, scatter_nucleus=scatter_nucleus)
    return Cone(apex, Dhat, float(theta))

def build_cone_from_gamma(ev: GammaEvent, energy_model: EnergyStrategy) -> Cone:
    """
    Placeholder gamma version: axis = X1 - X2 (points toward presumed source),
    a constant opening until we wire Compton kinematics.
    """
    hits = ev.ordered()
    r1, r2 = hits[0].r, hits[1].r
    D = r1 - r2
    Dhat = D / np.linalg.norm(D)
    apex = r1.copy()
    theta = np.deg2rad(25.0)
    return Cone(apex, Dhat, theta)
