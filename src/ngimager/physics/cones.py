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
    plane: Optional[Plane] = None,
    prior: Optional[Prior] = None,
    force_proton: bool = False,
    return_meta: bool = False,
) -> Cone | tuple[Cone, int]:
    """
    Build a neutron cone using the NOVO imaging primer convention:

      - apex O = X1 (first hit position),
      - axis D = unit vector along the scattered neutron direction (X2 - X1),
      - half-angle θ from elastic n–N kinematics in the lab frame.

    Behavior (high level)
    ---------------------
    * The event is assumed to be time-ordered (h1 before h2); callers
      should use ev.ordered() upstream, as the pipeline already does.

    * We always use the full kinematic chain from kinematics.py:

          E'   = E_n' from ToF between hits 1→2 (relativistic),
          En   = E' + E_dep,1,
          θ    = θ_lab(E_dep,1, En, A) with A = m_recoil / m_n,

      where E_dep,1 is obtained from `energy_model.first_scatter_energy(...)`
      and A is set by the assumed recoil nucleus ("H" or "C").

    * If `force_proton` is True, or if `plane` is None, we build a single
      proton-recoil hypothesis and return it (backwards-compatible path).

    * Otherwise, we build both proton and carbon hypotheses, reject any
      that are non-physical, enforce that the cone axis points toward
      the imaging plane, and then score the survivors against the prior
      using the same Δ = |φ − θ| metric used for gammas. The winner is
      the hypothesis with the smallest Δ.

      If both hypotheses fail scoring (e.g. degenerate prior geometry),
      we fall back to the proton-only construction.

    Metadata / return value
    -----------------------
    * By default (return_meta=False), the function returns only a Cone.

    * If return_meta is True, it returns (cone, recoil_code, En_MeV) where:

          recoil_code = 1  → proton recoil hypothesis ("H")
          recoil_code = 2  → carbon recoil hypothesis ("C")
          En_MeV      = incident neutron energy for the selected hypothesis

      This is suitable for compact storage in HDF5 and for cone-level
      max-energy cuts. Callers that do not care about the recoil species
      or En can continue to use the old, cone-only behavior.

    Notes
    -----
    * This function does not mutate the event or record which hypothesis
      "won"; that bookkeeping is left to callers via the returned
      recoil_code and En.
    """
    # Basic sanity: caller should already have ordered() + validate(), but
    # doing a light check here keeps this function robust for standalone use.
    ev.validate(strict=False)
    h1, h2 = ev.h1, ev.h2

    r1 = np.asarray(h1.r, dtype=float)
    r2 = np.asarray(h2.r, dtype=float)
    t1 = float(h1.t_ns)
    t2 = float(h2.t_ns)

    # Axis: scattered neutron direction (h1 → h2), normalized
    D = r1 - r2
    L = np.linalg.norm(D)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("Zero or non-physical baseline between hits in NeutronEvent.")

    Dhat = D / L
    apex = r1.copy()

    def _candidate_for_nucleus(
        recoil_nucleus: str,
    ) -> tuple[Optional[Cone], Optional[float], Optional[float]]:
        """
        Helper: construct a Cone for a given recoil nucleus label ("H" or "C")
        and return (cone, score, En) where:

          - cone  : Cone instance or None if non-physical
          - score : Δ = |φ − θ| if a prior is available, else None
          - En    : incident neutron energy [MeV] for this hypothesis, or None
        """
        # Decide which species key to use for the energy strategy.
        # For ELUT, we have distinct proton/carbon bands; for other
        # strategies (ToF, FixedEn, Edep) we keep proton-like behavior
        # for E_dep,1 to remain compatible with legacy usage.
        species_key = "proton"
        if getattr(energy_model, "name", None) == "ELUT":
            species_key = "proton" if recoil_nucleus == "H" else "carbon"

        # E_dep at first scatter from the energy model.
        try:
            Edep1_MeV, _ = energy_model.first_scatter_energy(
                h1,
                h2,
                h1.material,
                species=species_key,
            )
        except Exception:
            return None, None, None

        # Full COM → lab mapping using primer-consistent kinematics.
        try:
            theta, En = neutron_theta_from_hits(
                r1, t1,
                r2, t2,
                Edep1_MeV,
                scatter_nucleus=recoil_nucleus,
                return_En=True,
            )
        except Exception:
            return None, None, None

        # Reject clearly non-physical / degenerate angles
        if (not np.isfinite(theta)) or (theta <= 0.0) or (theta >= np.pi):
            return None, None, None

        c = Cone(apex=apex, direction=Dhat, theta=float(theta))

        # If we have a plane, enforce that the cone axis points toward it.
        if plane is not None and (not _axis_towards_plane(apex, Dhat, plane)):
            return None, None, None

        # If we don't have a plane, we can't do prior-based scoring.
        if plane is None:
            return c, None, float(En)

        score = _score_cone_against_prior(c, plane, prior)
        return c, score, float(En)

    # Proton-only path: either explicitly requested, or no plane available.
    if force_proton or plane is None:
        c_p, _, En_p = _candidate_for_nucleus("H")
        if c_p is None:
            raise ValueError("NeutronEvent cannot produce a physical proton-recoil cone.")

        if return_meta:
            # 1 = proton recoil
            return c_p, 1, float(En_p)
        return c_p

    # Full proton vs carbon hypothesis test with prior-aware scoring.
    # Store tuples of (score, recoil_nucleus, cone, En).
    candidates: list[tuple[float, str, Cone, float]] = []

    for nuc in ("H", "C"):
        c, score, En = _candidate_for_nucleus(nuc)
        if c is None:
            continue
        # If scoring failed (e.g. degenerate prior geometry), skip from
        # the prior-based comparison.
        if score is None:
            continue
        candidates.append((float(score), nuc, c, float(En)))

    if candidates:
        # Pick the cone with minimal Δ = |φ − θ|
        candidates.sort(key=lambda scn: scn[0])
        best_score, best_nuc, best_cone, best_En = candidates[0]

        if return_meta:
            recoil_code = 1 if best_nuc == "H" else 2  # 1=proton, 2=carbon
            return best_cone, recoil_code, float(best_En)
        return best_cone

    # If we got here, either both hypotheses were non-physical or prior
    # scoring failed for both. Fall back to proton-only construction
    # without using the prior.
    c_p, _, En_p = _candidate_for_nucleus("H")
    if c_p is None:
        raise ValueError("NeutronEvent cannot produce a physical cone (proton or carbon).")

    if return_meta:
        # In this fallback, we default to proton-like tagging.
        return c_p, 1, float(En_p)
    return c_p



def _gamma_cone_from_ordered_hits(
    h1,
    h2,
    h3,
    *,
    return_Eg: bool = False,
) -> Cone | None:
    """
    Attempt to build a Compton cone from three *ordered* hits.

    This is per-hit-ordering and is deliberately "soft":
      - returns a Cone on success,
      - returns None if the configuration is non-physical.

    If return_Eg is True, returns (cone, Eg_MeV) or (None, None).

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
        return (None, None) if return_Eg else None

    # Second-scatter angle theta2 from geometry
    cos_theta2 = float(np.dot(v12, v23) / (L12 * L23))
    cos_theta2 = float(np.clip(cos_theta2, -1.0, 1.0))
    theta2 = float(np.arccos(cos_theta2))

    # Deposited energies at first and second scatters (MeV)
    # For gammas we treat Hit.L as the deposited energy (already calibrated).
    dE1 = float(getattr(h1, "L", 0.0) or 0.0)
    dE2 = float(getattr(h2, "L", 0.0) or 0.0)
    if dE1 <= 0.0 or dE2 <= 0.0:
        return (None, None) if return_Eg else None

    try:
        Eg = compton_incident_energy_from_second_scatter(dE1, dE2, theta2)
        Egp = Eg - dE1
        theta1 = compton_theta_from_energies(Eg, Egp)
    except ValueError:
        # Non-physical combination for this ordering
        return (None, None) if return_Eg else None

    # Reject extremely small angles (numerically unstable / not useful)
    theta1_min = np.deg2rad(1.0)
    if not np.isfinite(theta1) or theta1 < theta1_min:
        return (None, None) if return_Eg else None

    # Geometry of the cone:
    # apex at first scatter, axis from 2nd -> 1st (matches neutron convention)
    D = r1 - r2
    L = float(np.linalg.norm(D))
    if L <= 0.0:
        return (None, None) if return_Eg else None

    Dhat = D / L
    apex = r1.copy()

    cone = Cone(apex, Dhat, float(theta1))
    if return_Eg:
        return cone, float(Eg)
    return cone

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
    return plane.center()


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
        cone = _gamma_cone_from_ordered_hits(hits[i0], hits[i1], hits[i2], return_Eg=False)
        if cone is None:
            continue
        candidates.append((cone, perm))

    return candidates



def build_cone_from_gamma(
    ev: GammaEvent,
    energy_model: EnergyStrategy,
    plane: Optional[Plane] = None,
    prior: Optional[Prior] = None,
    return_meta: bool = False,
    return_perm: bool = False,
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

    Return value
    ------------
    * If return_meta and return_perm are both False (default), returns only
      a Cone.
    * If return_meta is True and return_perm is False, returns (cone, Eg_MeV)
      where Eg_MeV is the incident gamma energy for the selected ordering.
    * If return_meta is False and return_perm is True, returns (cone, perm)
      where perm is a tuple (i0, i1, i2) with indices into the event's
      time-ordered hit list.
    * If both return_meta and return_perm are True, returns
      (cone, Eg_MeV, perm).

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

    def _pack_return(cone: Cone, Eg: float) -> Cone:
        """
        Helper to package return values according to return_meta/return_perm.
        """
        perm_default = (0, 0, 0)  # will be overridden when we really track a perm
        if not return_meta and not return_perm:
            return cone
        if return_meta and not return_perm:
            return cone, float(Eg)
        if not return_meta and return_perm:
            # For callers that only care about the permutation and not Eg,
            # the caller will override perm_default with the actual perm.
            return cone, perm_default
        # Both meta and permutation requested
        return cone, float(Eg), perm_default
    
    # Backwards-compatible path: no plane provided → use only the ordered triplet
    if plane is None:
        cone, Eg = _gamma_cone_from_ordered_hits(*hits, return_Eg=True)
        if cone is None:
            raise ValueError(
                "GammaEvent cannot produce a physical Compton cone from ordered hits."
            )

        # Simple case: the ordering is just (0, 1, 2) in time
        base_perm = (0, 1, 2)
        if not return_perm:
            if return_meta:
                return cone, float(Eg)
            return cone
        else:
            if return_meta:
                return cone, float(Eg), base_perm
            return cone, base_perm

    # Full permutation + prior-aware scoring path
    best_cone: Cone | None = None
    best_score: float | None = None
    best_Eg: float | None = None
    best_perm: tuple[int, int, int] | None = None

    for perm in permutations((0, 1, 2), 3):
        i0, i1, i2 = perm
        h1, h2, h3 = hits[i0], hits[i1], hits[i2]
        c, Eg = _gamma_cone_from_ordered_hits(h1, h2, h3, return_Eg=True)
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
            best_Eg = float(Eg)
            best_perm = perm

    if best_cone is not None and best_perm is not None:
        Eg = float(best_Eg) if best_Eg is not None else float("nan")
        if not return_perm:
            if return_meta:
                return best_cone, Eg
            return best_cone
        else:
            if return_meta:
                return best_cone, Eg, best_perm
            return best_cone, best_perm

    # If no candidate survived, fall back to the time-ordered triplet as a last resort
    fallback_cone, fallback_Eg = _gamma_cone_from_ordered_hits(*hits, return_Eg=True)
    if fallback_cone is None or not _axis_towards_plane(
        fallback_cone.apex, fallback_cone.dir, plane
    ):
        raise ValueError(
            "GammaEvent cannot produce a physical Compton cone from any hit permutation."
        )

    base_perm = (0, 1, 2)
    if not return_perm:
        if return_meta:
            return fallback_cone, float(fallback_Eg)
        return fallback_cone
    else:
        if return_meta:
            return fallback_cone, float(fallback_Eg), base_perm
        return fallback_cone, base_perm

