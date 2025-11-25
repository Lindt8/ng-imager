from __future__ import annotations
import numpy as np
from typing import Literal, Optional
from .hits import Hit
from ..io.lut import LUT
from ngimager.physics.kinematics import tof_energy_relativistic

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
        """
        Parameters
        ----------
        h1, h2
            First and optional second hits in the event.
        material
            Material key (e.g. "OGS", "M600") for LUT-based strategies.
        species
            Recoil species key when relevant (e.g. "proton", "carbon").

        Returns
        -------
        Edep1_MeV : float
            First-scatter deposited energy [MeV] to feed into the kinematics.
        sigma_MeV : float or None
            Optional uncertainty estimate on Edep1, or None if not provided.
        """
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
    """
    Monoenergetic incident neutron energy (e.g. DT source).

    This strategy assumes a fixed incident neutron kinetic energy En.
    For a given 2-hit neutron event, we:

      1. Compute the post-scatter neutron energy E' from ToF between h1 and h2.
      2. Infer the first-scatter deposited energy as Edep1 = En - E'.
      3. Reject the event if E' >= En (non-physical upscatter).

    The returned value is Edep1, which downstream kinematics combine with
    E' to reconstruct En again. This keeps the math consistent with
    `neutron_theta_from_hits` while enforcing monoenergetic DT semantics.
    """
    name = "FixedEn"
    def __init__(self, En_MeV: float = 14.1):
        self.En = En_MeV

    def first_scatter_energy(self, h1, h2, material, species=None):
        # We require two hits to form a neutron double. If h2 is missing,
        # treat this as non-physical for the FixedEn strategy.
        if h2 is None:
            raise ValueError("FixedEn requires a second hit (h2) to compute ToF.")

        # Distance and time-of-flight between the two hits.
        r1 = np.asarray(h1.r, dtype=float)
        r2 = np.asarray(h2.r, dtype=float)
        s_cm = float(np.linalg.norm(r2 - r1))
        dt_ns = float(h2.t_ns - h1.t_ns)

        # Post-scatter neutron energy from ToF.
        try:
            Eprime_MeV = float(tof_energy_relativistic(s_cm, dt_ns))
        except Exception as exc:
            # Propagate as a ValueError so cone-building treats this as a
            # kinematic failure and counts it in the cone-failed counters.
            raise ValueError(f"FixedEn ToF energy computation failed: {exc}")

        # Monoenergetic incident energy En; any E' >= En implies upscatter.
        if Eprime_MeV >= self.En:
            # Explicit upscatter rejection: this will be counted via
            # events_cone_build_failed / events_cone_build_failed_n.
            raise ValueError(
                f"FixedEn non-physical upscatter: E'={Eprime_MeV:.3f} MeV >= En={self.En:.3f} MeV"
            )

        Edep1_MeV = self.En - Eprime_MeV
        return Edep1_MeV, None

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
