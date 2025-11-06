# src/ngimager/physics/kinematics.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Constants in MeV/c^2 and cm/ns
C_CM_PER_NS = 29.9792458
M_N_MEV = 939.565          # neutron
M_P_MEV = 938.272          # proton
M_C12_MEV = 12*931.494 - 6*0.511  # carbon-12 (nuclear mass approx)


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
) -> float:
    """
    Full calculation consistent with the NOVO primer:
      E' via ToF between hits 1->2 (relativistic),
      E_n = E' + Edep1,
      theta_lab from COM using A = m_recoil/m_n.
    """
    s = float(np.linalg.norm(r2_cm - r1_cm))
    dt = float(t2_ns - t1_ns)
    Eprime = tof_energy_relativistic(s, dt)      # MeV
    En = Eprime + Edep1_MeV                      # MeV
    A = mass_ratio_A(scatter_nucleus)
    theta = theta_lab_from_Erecoil_En(Edep1_MeV, En, A)
    return theta
