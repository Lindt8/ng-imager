from __future__ import annotations

import numpy as np
from typing import Literal, Optional
import itertools
from itertools import permutations

from ngimager.physics.events import NeutronEvent, GammaEvent
from ngimager.imaging.sbp import Cone
from ngimager.geometry.plane import Plane
from ngimager.physics.priors import Prior, PointPrior, LinePrior
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

def _axis_towards_plane(apex: np.ndarray, direction: np.ndarray, plane: Plane) -> bool:
    """
    Return True if the cone axis from `apex` along `direction` intersects the plane
    in the positive t-direction (t_int > 0).

    Axis ray: X(t) = apex + t * direction
    Plane:   (X - P0) · n = 0

    t_int = (P0 - apex) · n / (direction · n)
    """
    n = plane.n
    denom = float(direction @ n)
    if abs(denom) < 1e-9:
        # Axis is (numerically) parallel to the plane: treat as not useful
        return False

    t_int = float((plane.P0 - apex) @ n / denom)
    return t_int > 0.0


def _plane_center(plane: Plane) -> np.ndarray:
    """
    Compute the geometric center of the finite imaging plane.

    This mirrors the intuitive "center of FOV" default when no explicit prior
    is provided.
    """
    u_c = 0.5 * (plane.u_min + plane.u_max)
    v_c = 0.5 * (plane.v_min + plane.v_max)
    return plane.plane_to_world(u_c, v_c)


def _prior_direction_vector(
    apex: np.ndarray,
    plane: Plane,
    prior: Optional[Prior],
) -> Optional[np.ndarray]:
    """
    Unit vector from `apex` toward the effective prior target.

    - If prior is None: use the imaging plane center.
    - If prior is PointPrior: use its point.
    - If prior is LinePrior: use the line midpoint (p0+p1)/2 for now.
    """
    if prior is None:
        target = _plane_center(plane)
    elif isinstance(prior, PointPrior):
        target = np.asarray(prior.point, dtype=float)
    elif isinstance(prior, LinePrior):
        # Use line midpoint as requested; a "closest point" option can be added later.
        target = 0.5 * (np.asarray(prior.p0, dtype=float) + np.asarray(prior.p1, dtype=float))
    else:
        # Fallback for any other Prior implementations: plane center
        target = _plane_center(plane)

    v = target - apex
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        return None
    return v / norm


def _score_cone_against_prior(
    cone: Cone,
    plane: Plane,
    prior: Optional[Prior],
) -> Optional[float]:
    """
    Compute Δ = |φ − θ| for a cone relative to the prior.

    - φ is the angle between the cone axis and the direction from apex toward
      the prior target.
    - θ is the cone's opening half-angle.

    Returns None if the prior direction is ill-defined (degenerate geometry).
    """
    d_prior = _prior_direction_vector(cone.apex, plane, prior)
    if d_prior is None:
        return None

    # Angle φ between axis (cone.dir) and line to prior
    cos_phi = float(np.clip(cone.dir @ d_prior, -1.0, 1.0))
    phi = float(np.arccos(cos_phi))
    theta = float(cone.theta)

    return abs(phi - theta)



def enumerate_gamma_cone_candidates(
    ev: GammaEvent,
) -> list[tuple[Cone, tuple[int, int, int]]]:
    """
    Enumerate all physically valid Compton cones for the 3! permutations
    of a three-hit GammaEvent.

    Parameters
    ----------
    ev:
        A GammaEvent with exactly three hits (h1, h2, h3). The event is
        assumed to be already validated for basic consistency.

    Returns
    -------
    candidates:
        List of (cone, perm) tuples where:

          - cone is a Cone instance produced by _gamma_cone_from_ordered_hits
          - perm is a tuple of indices (i0, i1, i2) into (h1, h2, h3),
            describing which hit played the role of first/second/third
            scatter in the kinematic construction.

        Only permutations that yield a physically valid Compton cone
        (non-negative energies, sensible angles, non-degenerate geometry)
        are returned. If no permutation is viable, the list is empty.

    Notes
    -----
    * This function is kinematics-only: it does NOT apply any priors
      or scoring; it simply reports all physically allowed cones.

    * Subsequent stages (e.g. in the pipeline) can:

        - apply event- or cone-level filters to the candidates, and
        - use spatial/energy priors to select a "best" cone for imaging.
    """
    # Access hits in a stable order; for now GammaEvent always has h1..h3.
    hits = [ev.h1, ev.h2, ev.h3]
    candidates: list[tuple[Cone, tuple[int, int, int]]] = []

    # Enumerate all permutations of (0, 1, 2). For each permutation, treat
    # hits[i0] as the first scatter, hits[i1] as the second, and hits[i2]
    # as the "third" (used only for geometry).
    for perm in itertools.permutations((0, 1, 2), 3):
        i0, i1, i2 = perm
        cone = _gamma_cone_from_ordered_hits(hits[i0], hits[i1], hits[i2])
        if cone is None:
            continue
        candidates.append((cone, perm))

    return candidates



def build_cone_from_gamma(
    ev: GammaEvent,
    energy_model: EnergyStrategy,
    plane: Optional[Plane] = None,
    prior: Optional[Prior] = None,
) -> Cone:
    """
    Build a Compton gamma cone from a three-hit GammaEvent.

    Behavior without plane/prior (backwards-compatible, PHITS-oriented):
      - Use ev.ordered() so that h1, h2, h3 are in increasing time,
        which is physically the true order in PHITS data.
      - Attempt to build a cone from this ordered triplet using
        _gamma_cone_from_ordered_hits.
      - If no physically valid cone exists for this ordering, raise ValueError.

    Enhanced behavior when `plane` is provided:
      - Generate all 3! permutations of (h1, h2, h3).
      - For each ordering:
          * call _gamma_cone_from_ordered_hits(h1, h2, h3),
          * discard if it returns None (non-physical),
          * discard if the cone axis does not point toward the plane
            (t_int <= 0 via _axis_towards_plane),
          * compute Δ = |φ − θ| using the configured prior or, if prior is None,
            the plane center as an implicit prior.
      - Select the candidate with minimal Δ.
      - If no candidate survives, fall back to the ordered (time) triplet
        as in the simple behavior; if that also fails, raise ValueError.

    Notes
    -----
    * For now, we do not use `energy_model` for gammas: Hit.L is already
      the deposited energy in MeV (Edep) from the adapter.

    * This function is designed so that callers who do not yet pass a Plane
      or Prior still get the old, simple behavior.
    """
    # Ensure we have a time-ordered GammaEvent (PHITS case)
    ev_ord = ev.ordered(copy=True)
    hits = [ev_ord.h1, ev_ord.h2, ev_ord.h3]

    # Backwards-compatible path: no plane provided → use only the ordered triplet
    if plane is None:
        cone = _gamma_cone_from_ordered_hits(*hits)
        if cone is None:
            raise ValueError(
                "GammaEvent cannot produce a physical Compton cone from ordered hits."
            )
        return cone

    # Full permutation + prior-aware scoring path
    best_cone: Cone | None = None
    best_score: float | None = None

    for h1, h2, h3 in permutations(hits, 3):
        c = _gamma_cone_from_ordered_hits(h1, h2, h3)
        if c is None:
            continue

        # Reject cones whose axis does not point toward the imaging plane
        if not _axis_towards_plane(c.apex, c.dir, plane):
            continue

        # Δ = |φ − θ| using explicit prior or implicit plane-center prior
        score = _score_cone_against_prior(c, plane, prior)
        if score is None:
            # Degenerate prior geometry; treat as unusable candidate
            continue

        if best_cone is None or score < best_score:
            best_cone = c
            best_score = score

    if best_cone is not None:
        return best_cone

    # If no candidate survived, fall back to the time-ordered triplet as a last resort
    fallback = _gamma_cone_from_ordered_hits(*hits)
    if fallback is None or not _axis_towards_plane(fallback.apex, fallback.dir, plane):
        raise ValueError(
            "GammaEvent cannot produce a physical Compton cone from any hit permutation."
        )

    return fallback



