from __future__ import annotations

import numpy as np
from typing import Literal

from ngimager.physics.events import NeutronEvent, GammaEvent
from ngimager.imaging.sbp import Cone
from ngimager.physics.energy_strategies import EnergyStrategy
from ngimager.physics.kinematics import neutron_theta_from_hits


def build_cone_from_neutron(
    ev: NeutronEvent,
    energy_model: EnergyStrategy,
    scatter_nucleus: Literal["H", "C"] = "H",
) -> Cone:
    """
    Build a neutron cone using the NOVO imaging primer convention:

      - apex O = X1 (first hit position),
      - axis D̂ = (X1 - X2) / ||X1 - X2|| (points from 2nd -> 1st hit),
      - opening angle theta from kinematics using:
          * Edep1 from the energy strategy (ELUT, Birks, etc.)
          * E' from time-of-flight (via neutron_theta_from_hits).

    Parameters
    ----------
    ev:
        NeutronEvent with h1, h2 populated (positions in cm, times in ns).
    energy_model:
        EnergyStrategy instance (e.g. ELutEnergy, FixedEn, etc.) providing
        Edep1 for the first scatter.
    scatter_nucleus:
        Target nucleus used in the kinematic model ("H" or "C").

    Returns
    -------
    Cone
        Cone(apex=O, axis=D̂, theta_rad).
    """
    # Basic sanity check
    ev.validate()
    h1, h2 = ev.h1, ev.h2

    r1 = h1.r.astype(float)
    r2 = h2.r.astype(float)
    t1 = float(h1.t_ns)
    t2 = float(h2.t_ns)

    # Direction from 2nd -> 1st (matches primer convention)
    D = r1 - r2
    L = np.linalg.norm(D)
    if L <= 0:
        raise ValueError("Zero baseline between hits in NeutronEvent.")
    Dhat = D / L
    apex = r1.copy()

    # E_dep at first scatter from the energy model.
    # Use proton band by default for scintillator response.
    Edep1_MeV, _ = energy_model.first_scatter_energy(
        h1,
        h2,
        h1.material,
        "proton",
    )

    # Kinematic opening angle
    theta = neutron_theta_from_hits(
        r1,
        t1,
        r2,
        t2,
        Edep1_MeV=Edep1_MeV,
        scatter_nucleus=scatter_nucleus,
    )

    return Cone(apex, Dhat, float(theta))


def build_cone_from_gamma(
    ev: GammaEvent,
    energy_model: EnergyStrategy,
) -> Cone:
    """
    Placeholder Compton cone builder.

    Currently:
      - apex at first hit position,
      - axis along X1 - X2 (pointing toward presumed source),
      - fixed opening angle until proper Compton kinematics are wired.

    This keeps the gamma pipeline plumbed without committing to final physics.
    """
    hits = ev.ordered()
    if len(hits) < 2:
        raise ValueError("GammaEvent must have at least two hits for a cone.")

    r1 = hits[0].r.astype(float)
    r2 = hits[1].r.astype(float)

    D = r1 - r2
    L = np.linalg.norm(D)
    if L <= 0:
        raise ValueError("Zero baseline between gamma hits.")

    Dhat = D / L
    apex = r1.copy()

    # Temporary: fixed angle (e.g. 25 degrees)
    theta = np.deg2rad(25.0)

    return Cone(apex, Dhat, theta)
