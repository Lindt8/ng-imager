# src/ngimager/physics/kinematics.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Constants in MeV/c^2 and cm/ns
C_CM_PER_NS = 29.9792458
M_N_MEV = 939.565          # neutron
M_P_MEV = 938.272          # proton
M_C12_MEV = 12*931.494 - 6*0.511  # carbon-12 (nuclear mass approx)
M_E_MEV = 0.511            # electron

@dataclass(frozen=True)
class Nucleus:
    name: str
    mass_MeV: float

NUCLEI = {
    "H": Nucleus("H", M_P_MEV),
    "C": Nucleus("C", M_C12_MEV),
}

def tof_energy_relativistic(s_cm: float, dt_ns: float) -> float:
    """
    Relativistic neutron KE E' [MeV] from flight distance s [cm] and time dt [ns].
    """
    if dt_ns <= 0:
        raise ValueError("Non-positive ToF; cannot compute E'.")
    v = s_cm / dt_ns  # cm/ns
    beta2 = (v / C_CM_PER_NS)**2
    if beta2 <= 0 or beta2 >= 1:
        raise ValueError("Non-physical beta^2 from s/dt.")
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    return (gamma - 1.0) * M_N_MEV

def theta_lab_from_Erecoil_En(E_recoil: float, E_n: float, A: float) -> float:
    """
    Compute neutron lab-frame scattering half-angle [rad] from E_recoil, E_n, and A = m_recoil/m_n.
    Follows primer equations for theta_CoM then lab mapping.
    """
    if E_recoil <= 0 or E_n <= 0 or E_recoil > E_n:
        raise ValueError("Non-physical energies for theta.")
    # theta_CoM
    cos_arg = 1.0 - (E_recoil / E_n) * ((1.0 + A)**2) / (2.0 * A)
    # guard numerical drift
    cos_arg = np.clip(cos_arg, -1.0, 1.0)
    theta_CoM = np.arccos(cos_arg)

    # theta_lab
    num = np.sin(theta_CoM)
    den = np.cos(theta_CoM) + (1.0 / A)
    return np.arctan2(num, den)

def mass_ratio_A(scatter_nucleus: str) -> float:
    nuc = NUCLEI.get(scatter_nucleus)
    if nuc is None:
        raise ValueError(f"Unsupported recoil nucleus '{scatter_nucleus}'. Use 'H' or 'C'.")
    return nuc.mass_MeV / M_N_MEV

def neutron_theta_from_hits(
    r1_cm: np.ndarray, t1_ns: float,
    r2_cm: np.ndarray, t2_ns: float,
    Edep1_MeV: float,
    scatter_nucleus: str = "H",
    return_En: bool = False,
) -> float:
    """
    Full calculation consistent with the NOVO primer:
      E' via ToF between hits 1->2 (relativistic),
      E_n = E' + Edep1,
      theta_lab from COM using A = m_recoil/m_n.
    
    If return_En is False (default), returns theta [rad].
    If return_En is True, returns (theta [rad], En [MeV]).
    """
    s = float(np.linalg.norm(r2_cm - r1_cm))
    dt = float(t2_ns - t1_ns)
    Eprime = tof_energy_relativistic(s, dt)      # MeV
    En = Eprime + Edep1_MeV                      # MeV
    A = mass_ratio_A(scatter_nucleus)
    theta = theta_lab_from_Erecoil_En(Edep1_MeV, En, A)
    if return_En:
        return float(theta), float(En)
    return float(theta)

def compton_incident_energy_from_second_scatter(
    dE1_MeV: float,
    dE2_MeV: float,
    theta2_rad: float,
) -> float:
    """
    Incident gamma energy Eg [MeV] from:

      - dE1: energy deposited at 1st scatter [MeV]
      - dE2: energy deposited at 2nd scatter [MeV]
      - theta2: angle between 1->2 and 2->3 baselines [rad]

    This mirrors the NOVO primer / legacy implementation:

        Eg = dE1 + 0.5 * ( dE2 + sqrt( dE2^2 + 4*dE2*me / (1 - cos(theta2)) ) )

    Raises
    ------
    ValueError
        If inputs are non-physical (negative energies, grazing angles, etc.).
    """
    if dE1_MeV <= 0.0 or dE2_MeV <= 0.0:
        raise ValueError(f"Non-positive gamma deposits: dE1={dE1_MeV}, dE2={dE2_MeV}")

    cos_t2 = float(np.cos(theta2_rad))
    denom = 1.0 - cos_t2
    if denom <= 0.0:
        raise ValueError(f"Non-physical second-scatter angle theta2={theta2_rad} (denominator <= 0).")

    radicand = dE2_MeV**2 + (4.0 * dE2_MeV * M_E_MEV / denom)
    if radicand <= 0.0:
        raise ValueError(f"Non-physical Compton radicand={radicand} for gamma cone.")

    Eg = dE1_MeV + 0.5 * (dE2_MeV + float(np.sqrt(radicand)))
    if not np.isfinite(Eg) or Eg <= 0.0:
        raise ValueError(f"Non-physical incident gamma energy Eg={Eg}")

    return Eg


def compton_theta_from_energies(Eg_MeV: float, Egp_MeV: float) -> float:
    """
    First Compton scatter angle theta1 [rad] from:

        Eg  : incident gamma energy [MeV]
        Egp : post-first-scatter gamma energy [MeV]

    Uses the standard Compton relation:

        cos(theta) = 1 + me * (1/Eg - 1/Egp)

    Raises
    ------
    ValueError
        If energies are non-physical or the argument to arccos is out of [-1, 1].
    """
    if Eg_MeV <= 0.0 or Egp_MeV <= 0.0 or Egp_MeV >= Eg_MeV:
        raise ValueError(f"Non-physical gamma energies Eg={Eg_MeV}, Eg'={Egp_MeV}")

    arg = 1.0 + M_E_MEV * ((1.0 / Eg_MeV) - (1.0 / Egp_MeV))

    # If we drift outside [-1, 1] more than a tiny epsilon, treat as non-physical
    if arg < -1.0 - 1e-6 or arg > 1.0 + 1e-6:
        raise ValueError(f"Compton argument out of domain: arg={arg}")

    arg = float(np.clip(arg, -1.0, 1.0))
    theta = float(np.arccos(arg))
    return theta

