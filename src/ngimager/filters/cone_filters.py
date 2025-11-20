from __future__ import annotations

from typing import Dict, Optional
import math

from ngimager.config.schemas import ConesFiltersCfg
from ngimager.imaging.sbp import Cone
from ngimager.geometry.plane import Plane
from ngimager.physics.priors import Prior
from ngimager.physics.cones import _score_cone_against_prior


def _inc(counters: Dict[str, int], key: str, delta: int = 1) -> None:
    counters[key] = counters.get(key, 0) + delta


def _delta_theta_limit_rad(filters: ConesFiltersCfg, species: str) -> Optional[float]:
    """
    Resolve the effective Δθ max (in radians) for a given species.

    Priority:
        per-species override, else global, else None.
    """
    s = (species or "").lower()
    if s.startswith("n"):
        val_deg = filters.neutron.max_delta_theta_deg
    elif s.startswith("g"):
        val_deg = filters.gamma.max_delta_theta_deg
    else:
        val_deg = None

    if val_deg is None:
        val_deg = filters.max_delta_theta_deg

    if val_deg is None:
        return None

    return math.radians(float(val_deg))

def _incident_energy_limit_MeV(filters: ConesFiltersCfg, species: str) -> Optional[float]:
    """
    Resolve the effective max incident energy [MeV] for a given species.

    Priority:
        per-species override, else global, else None.
    """
    s = (species or "").lower()
    if s.startswith("n"):
        val = getattr(filters.neutron, "max_incident_energy_MeV", None)
    elif s.startswith("g"):
        val = getattr(filters.gamma, "max_incident_energy_MeV", None)
    else:
        val = None

    if val is None:
        val = filters.max_incident_energy_MeV

    return None if val is None else float(val)

def passes_delta_theta_cut(
    cone: Cone,
    species: str,
    plane: Plane,
    prior: Optional[Prior],
    filters: ConesFiltersCfg,
    counters: Dict[str, int],
    *,
    incident_energy_MeV: Optional[float] = None,
) -> bool:
    """
    Cone-level filters:

      1. Δθ = |φ − θ|, where:
         - φ is the angle between the cone axis and the direction from apex to
           the prior target (or plane center if prior is None),
         - θ is the cone opening half-angle.

         Uses _score_cone_against_prior (shared with neutron p/C selection
         and gamma permutation selection).

      2. max_incident_energy_MeV:
         - For neutrons, En is computed once in physics/kinematics and
           passed in from the cone builder.
         - For gammas, Eg is likewise computed in the gamma cone builder.

    Both cuts are optional; if a limit is not configured (or we don't have
    an incident_energy_MeV), that cut is skipped.

    Returns True if the cone is accepted, False if rejected.

    Counters (all optional, keyed by species when applicable)
    ---------------------------------------------------------
      - "cones_checked_delta_theta"
      - "cones_checked_delta_theta_n"
      - "cones_checked_delta_theta_g"
      - "cones_rejected_delta_theta"
      - "cones_rejected_delta_theta_n"
      - "cones_rejected_delta_theta_g"

      - "cones_checked_incident_energy"
      - "cones_checked_incident_energy_n"
      - "cones_checked_incident_energy_g"
      - "cones_rejected_incident_energy"
      - "cones_rejected_incident_energy_n"
      - "cones_rejected_incident_energy_g"
    """
    s = (species or "").lower()

    # ---- Δθ cut ---------------------------------------------------------
    limit_dtheta = _delta_theta_limit_rad(filters, species)

    # We always count that Δθ was "checked", even if no limit is set.
    _inc(counters, "cones_checked_delta_theta")
    if s.startswith("n"):
        _inc(counters, "cones_checked_delta_theta_n")
    elif s.startswith("g"):
        _inc(counters, "cones_checked_delta_theta_g")

    if limit_dtheta is not None:
        delta = _score_cone_against_prior(cone, plane, prior)
        if delta is not None and float(delta) > float(limit_dtheta):
            _inc(counters, "cones_rejected_delta_theta")
            if s.startswith("n"):
                _inc(counters, "cones_rejected_delta_theta_n")
            elif s.startswith("g"):
                _inc(counters, "cones_rejected_delta_theta_g")
            return False
        # If delta is None (degenerate prior geometry), we treat it as
        # "no Δθ cut applied" and continue.

    # ---- max incident energy cut ---------------------------------------
    limit_En = _incident_energy_limit_MeV(filters, species)

    # Count that we evaluated the incident-energy filter.
    _inc(counters, "cones_checked_incident_energy")
    if s.startswith("n"):
        _inc(counters, "cones_checked_incident_energy_n")
    elif s.startswith("g"):
        _inc(counters, "cones_checked_incident_energy_g")

    # No configured energy limit or no value to compare → accept w.r.t. this cut.
    if limit_En is None or incident_energy_MeV is None:
        return True

    if float(incident_energy_MeV) <= float(limit_En):
        return True

    # Rejected by incident-energy cut
    _inc(counters, "cones_rejected_incident_energy")
    if s.startswith("n"):
        _inc(counters, "cones_rejected_incident_energy_n")
    elif s.startswith("g"):
        _inc(counters, "cones_rejected_incident_energy_g")
    return False
