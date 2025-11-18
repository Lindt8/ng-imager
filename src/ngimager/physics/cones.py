from __future__ import annotations

import numpy as np
from typing import Literal

from ngimager.physics.events import NeutronEvent, GammaEvent
from ngimager.imaging.sbp import Cone
from ngimager.physics.energy_strategies import EnergyStrategy
from ngimager.physics.kinematics import (
    neutron_theta_from_hits,
    compton_incident_energy_from_second_scatter,
    compton_theta_from_energies,
)


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


def _gamma_cone_from_ordered_hits(
    h1,
    h2,
    h3,
) -> Cone | None:
    """
    Attempt to build a Compton cone from three *ordered* hits.

    This is per-hit-ordering and is deliberately "soft":
      - returns a Cone on success,
      - returns None if the configuration is non-physical.

    This makes it suitable for:
      - the current PHITS case (time-ordered hits),
      - future 3! permutation searches without try/except at the caller.

    Geometry convention (as in the NOVO primer and neutron cones):

      - apex O = X1 (first scatter position),
      - axis D̂ = (X1 - X2) / ||X1 - X2|| (points from 2nd -> 1st hit),
      - opening angle theta1 from Compton kinematics.
    """
    # Positions (cm)
    r1 = h1.r.astype(float)
    r2 = h2.r.astype(float)
    r3 = h3.r.astype(float)

    v12 = r2 - r1
    v23 = r3 - r2
    L12 = float(np.linalg.norm(v12))
    L23 = float(np.linalg.norm(v23))
    if L12 <= 0.0 or L23 <= 0.0:
        # Degenerate geometry
        return None

    # Second-scatter angle theta2 from geometry
    cos_theta2 = float(np.dot(v12, v23) / (L12 * L23))
    cos_theta2 = float(np.clip(cos_theta2, -1.0, 1.0))
    theta2 = float(np.arccos(cos_theta2))

    # Deposited energies at first and second scatters (MeV)
    # For gammas we treat Hit.L as the deposited energy (already calibrated).
    dE1 = float(getattr(h1, "L", 0.0) or 0.0)
    dE2 = float(getattr(h2, "L", 0.0) or 0.0)
    if dE1 <= 0.0 or dE2 <= 0.0:
        return None

    try:
        Eg = compton_incident_energy_from_second_scatter(dE1, dE2, theta2)
        Egp = Eg - dE1
        theta1 = compton_theta_from_energies(Eg, Egp)
    except ValueError:
        # Non-physical combination for this ordering
        return None

    # Reject extremely small angles (numerically unstable / not useful)
    theta1_min = np.deg2rad(1.0)
    if not np.isfinite(theta1) or theta1 < theta1_min:
        return None

    # Geometry of the cone:
    # apex at first scatter, axis from 2nd -> 1st (matches neutron convention)
    D = r1 - r2
    L = float(np.linalg.norm(D))
    if L <= 0.0:
        return None

    Dhat = D / L
    apex = r1.copy()

    return Cone(apex, Dhat, float(theta1))


def build_cone_from_gamma(
    ev: GammaEvent,
    energy_model: EnergyStrategy,
) -> Cone:
    """
    Build a Compton gamma cone from a three-hit GammaEvent.

    Current behavior (PHITS-oriented):

      - Use ev.ordered() so that h1, h2, h3 are in increasing time,
        which is physically the true order in PHITS data.
      - Attempt to build a cone from this ordered triplet using
        _gamma_cone_from_ordered_hits.
      - If no physically valid cone exists for this ordering, raise ValueError.

    Notes
    -----
    * For now, we do not use `energy_model` for gammas: Hit.L is already
      the deposited energy in MeV (Edep) from the adapter.

    * Future work:
        - generate all 3! permutations of (h1, h2, h3),
        - call _gamma_cone_from_ordered_hits on each,
        - incorporate priors to choose a "best" cone candidate.
      The external behavior (either returns a Cone or raises ValueError
      for the event) can remain unchanged.
    """
    # Ensure we have a time-ordered GammaEvent (PHITS case)
    ev_ord = ev.ordered(copy=True)

    cone = _gamma_cone_from_ordered_hits(ev_ord.h1, ev_ord.h2, ev_ord.h3)
    if cone is None:
        raise ValueError("GammaEvent cannot produce a physical Compton cone from ordered hits.")

    return cone


