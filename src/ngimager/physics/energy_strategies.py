from __future__ import annotations
import numpy as np
from typing import Literal, Optional
from .hits import Hit
from ..io.lut import LUT

# --- Interfaces -------------------------------------------------------------

class EnergyStrategy:
    """Base protocol: compute first-scatter energy and optional σ."""
    name: str

    def first_scatter_energy(
        self,
        h1: Hit,
        h2: Hit | None,
        material: str,
        species: Literal["proton","carbon"] | None = "proton",
    ) -> tuple[float, float | None]:
        raise NotImplementedError

# --- Implementations --------------------------------------------------------

class EnergyFromLightLUT(EnergyStrategy):
    name = "ELUT"
    def __init__(self, lut_registry: dict[str, dict[str, LUT]]):
        self.luts = lut_registry

    def first_scatter_energy(self, h1, h2, material, species="proton"):
        lut = self.luts[material][species]
        E1, sigma = lut.eval(h1.L)
        return E1, sigma

class EnergyFromToF(EnergyStrategy):
    """Compute E' from ToF, then E_total = dE + E'."""
    name = "ToF"
    def __init__(self, timing_sigma_ns: float = 0.5):
        self.sigma_t = timing_sigma_ns

    def first_scatter_energy(self, h1, h2, material, species=None):
        # simplistic: distance / dt gives neutron velocity
        c = 29.9792  # cm/ns
        dt = (h2.t_ns - h1.t_ns)
        L = np.linalg.norm(h2.r - h1.r)
        v = L / dt
        En_after = 0.5 * 1.675e-27 * (v * 1e7)**2 / 1.602e-13  # MeV
        # placeholder: dE via light
        dE = h1.L * 1e-3
        return dE + En_after, None

class EnergyFromFixedIncident(EnergyStrategy):
    name = "FixedEn"
    def __init__(self, En_MeV: float = 14.1):
        self.En = En_MeV

    def first_scatter_energy(self, h1, h2, material, species=None):
        return self.En, None

class EnergyFromDeposited(EnergyStrategy):
    """
    Treat Hit.L as deposited energy (MeV) directly.

    This is intended for synthetic/sim sources like PHITS where Hit.L has
    already been filled from Edep_MeV in the adapter, so no E(L) inversion
    or ToF logic is needed.
    """
    name = "Edep"

    def first_scatter_energy(
        self,
        h1: Hit,
        h2: Hit | None,
        material: str,
        species: Literal["proton", "carbon"] | None = "proton",
    ) -> tuple[float, float | None]:
        # By construction, h1.L is the deposited energy in MeV.
        return float(h1.L), None



# --- Factory ----------------------------------------------------------------

def make_energy_strategy(cfg_energy, lut_registry=None) -> EnergyStrategy:
    if cfg_energy.strategy == "ELUT":
        return EnergyFromLightLUT(lut_registry)
    elif cfg_energy.strategy == "ToF":
        return EnergyFromToF()
    elif cfg_energy.strategy == "FixedEn":
        return EnergyFromFixedIncident(cfg_energy.fixed_En_MeV)
    if cfg_energy.strategy == "Edep":
        return EnergyFromDeposited()
    else:
        raise ValueError(f"Unknown strategy {cfg_energy.strategy}")
