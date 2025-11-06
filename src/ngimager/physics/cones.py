from __future__ import annotations
import numpy as np
from typing import Literal
from .events import NeutronEvent, GammaEvent
from ..imaging.sbp import Cone
from .energy_strategies import EnergyStrategy

def build_cone_from_neutron(ev: NeutronEvent, energy_model: EnergyStrategy) -> Cone:
    """Compute cone apex, axis, and opening angle θ."""
    ev.validate()
    E_total, sigma_E = energy_model.first_scatter_energy(ev.h1, ev.h2, ev.h1.material, "proton")
    r1, r2 = ev.h1.r, ev.h2.r
    D = r2 - r1
    L = np.linalg.norm(D)
    Dhat = D / L
    apex = r1.copy()
    # crude placeholder for now
    theta = np.deg2rad(30.0)
    return Cone(apex, Dhat, theta)

def build_cone_from_gamma(ev: GammaEvent, energy_model: EnergyStrategy) -> Cone:
    """Placeholder: choose first->second segment as axis."""
    hits = ev.ordered()
    r1, r2 = hits[0].r, hits[1].r
    Dhat = (r2 - r1) / np.linalg.norm(r2 - r1)
    apex = r1.copy()
    theta = np.deg2rad(25.0)
    return Cone(apex, Dhat, theta)
