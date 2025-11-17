from __future__ import annotations

from typing import Iterable, List, Dict, Optional

from ngimager.config.schemas import FiltersCfg
from ngimager.physics.hits import Hit


def _inc(counters: Dict[str, int], key: str, delta: int = 1) -> None:
    """
    Small helper to increment counters safely.
    """
    counters[key] = counters.get(key, 0) + delta


def apply_hit_filters(
    hits: Iterable[Hit],
    cfg: FiltersCfg,
    counters: Dict[str, int],
    *,
    particle_type: Optional[str] = None,
) -> List[Hit]:
    """
    Apply universal hit-level cuts (min/max light, allowed bars/materials).

    Parameters
    ----------
    hits :
        Input Hit objects for a single raw event.
    cfg :
        [filters] configuration.
    counters :
        Shared counters dict to be updated in-place.
    particle_type :
        Optional 'n' or 'g' (used to populate *_n / *_g counters).
    """
    hits_list = list(hits)

    # Totals before any cuts
    _inc(counters, "hits_total", len(hits_list))
    if particle_type == "n":
        _inc(counters, "hits_total_n", len(hits_list))
    elif particle_type == "g":
        _inc(counters, "hits_total_g", len(hits_list))

    kept: List[Hit] = []

    for h in hits_list:
        L = float(getattr(h, "L", 0.0))

        # Light/energy threshold cuts
        if (L < cfg.min_light) or (L > cfg.max_light):
            _inc(counters, "hits_rejected_threshold", 1)
            if particle_type == "n":
                _inc(counters, "hits_rejected_threshold_n", 1)
            elif particle_type == "g":
                _inc(counters, "hits_rejected_threshold_g", 1)
            continue

        # Bar whitelist (if provided)
        if cfg.bars_include and (h.det_id not in cfg.bars_include):
            # We don't yet track a dedicated counter for this; can be added later.
            continue

        # Material whitelist (if provided)
        if cfg.materials_include and (h.material not in cfg.materials_include):
            # Likewise, no dedicated counter yet.
            continue

        kept.append(h)

    _inc(counters, "hits_after_filters", len(kept))
    if particle_type == "n":
        _inc(counters, "hits_after_filters_n", len(kept))
    elif particle_type == "g":
        _inc(counters, "hits_after_filters_g", len(kept))
    
    return kept


def is_reconstructable(
    hits: Iterable[Hit],
    cfg: FiltersCfg,  # reserved for future, e.g. more complex criteria
    counters: Dict[str, int],
    *,
    event_type: Optional[str] = None,
) -> bool:
    """
    Early decision: does this raw event still have enough hits to ever form
    a reconstructable cone?

    For now:
        - neutron (event_type 'n'): require ≥ 2 hits
        - gamma   (event_type 'g'): require ≥ 3 hits
        - unknown: require ≥ 2 hits (conservative default)

    If not reconstructable, the appropriate raw_events_rejected_unreconstructable
    counters are incremented.
    """
    hits_list = list(hits)
    n_hits = len(hits_list)

    if event_type == "n":
        needed = 2
        suffix = "_n"
    elif event_type == "g":
        needed = 3
        suffix = "_g"
    else:
        needed = 2
        suffix = ""

    if n_hits < needed:
        _inc(counters, "raw_events_rejected_unreconstructable", 1)
        if suffix:
            _inc(counters, f"raw_events_rejected_unreconstructable{suffix}", 1)
        return False

    return True
