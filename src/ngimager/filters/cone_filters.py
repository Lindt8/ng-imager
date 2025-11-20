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


def passes_delta_theta_cut(
    cone: Cone,
    species: str,
    plane: Plane,
    prior: Optional[Prior],
    filters: ConesFiltersCfg,
    counters: Dict[str, int],
) -> bool:
    """
    Cone-level filter on Δθ = |φ − θ|, where:

      - φ is the angle between the cone axis and the direction from apex to
        the prior target (or plane center if prior is None),
      - θ is the cone opening half-angle.

    This uses the same scorer as is used for proton vs carbon selection and
    gamma permutation selection: _score_cone_against_prior.

    Returns True if the cone is accepted, False if rejected.

    Counters (all optional, keyed by species when applicable)
    ---------------------------------------------------------
      - "cones_checked_delta_theta"
      - "cones_checked_delta_theta_n"
      - "cones_checked_delta_theta_g"
      - "cones_rejected_delta_theta"
      - "cones_rejected_delta_theta_n"
      - "cones_rejected_delta_theta_g"
    """
    limit = _delta_theta_limit_rad(filters, species)
    # If no Δθ limit configured, do nothing but still count checks.
    _inc(counters, "cones_checked_delta_theta")
    s = (species or "").lower()
    if s.startswith("n"):
        _inc(counters, "cones_checked_delta_theta_n")
    elif s.startswith("g"):
        _inc(counters, "cones_checked_delta_theta_g")

    if limit is None:
        return True

    delta = _score_cone_against_prior(cone, plane, prior)
    if delta is None:
        # Degenerate prior geometry → don't apply this cut.
        return True

    if float(delta) <= float(limit):
        return True

    # Rejected
    _inc(counters, "cones_rejected_delta_theta")
    if s.startswith("n"):
        _inc(counters, "cones_rejected_delta_theta_n")
    elif s.startswith("g"):
        _inc(counters, "cones_rejected_delta_theta_g")
    return False
