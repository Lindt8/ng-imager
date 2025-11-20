from __future__ import annotations

from typing import Iterable, List, Dict, Optional

from ngimager.config.schemas import HitsFiltersCfg
from ngimager.physics.hits import Hit


def _inc(counters: Dict[str, int], key: str, delta: int = 1) -> None:
    """
    Small helper to increment counters safely.
    """
    counters[key] = counters.get(key, 0) + delta


def _resolve_hits_cfg(
    cfg: HitsFiltersCfg,
    particle_type: Optional[str],
) -> Dict[str, object]:
    """
    Resolve effective hit-level filter parameters for a given species.

    Returns a dict with keys:
      min_light_MeVee, max_light_MeVee,
      bars_include, bars_exclude,
      materials_include, materials_exclude
    """
    base = cfg
    pt = (particle_type or "").lower()
    if pt == "n":
        ov = cfg.neutron
    elif pt == "g":
        ov = cfg.gamma
    else:
        ov = None

    def _pick(name: str):
        if ov is None:
            return getattr(base, name)
        val = getattr(ov, name, None)
        return getattr(base, name) if val is None else val

    return {
        "min_light_MeVee": float(_pick("min_light_MeVee")),
        "max_light_MeVee": float(_pick("max_light_MeVee")),
        "bars_include": list(_pick("bars_include")),
        "bars_exclude": list(_pick("bars_exclude")),
        "materials_include": list(_pick("materials_include")),
        "materials_exclude": list(_pick("materials_exclude")),
    }


def apply_hit_filters(
    hits: Iterable[Hit],
    cfg: HitsFiltersCfg,
    counters: Dict[str, int],
    *,
    particle_type: Optional[str] = None,
) -> List[Hit]:
    """
    Apply universal + species-specific hit-level cuts.

    Parameters
    ----------
    hits :
        Input Hit objects for a single raw event.
    cfg :
        [filters.hits] configuration (including .neutron/.gamma overrides).
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

    # Resolve effective thresholds/lists for this particle type
    eff = _resolve_hits_cfg(cfg, particle_type)
    min_L = eff["min_light_MeVee"]
    max_L = eff["max_light_MeVee"]
    bars_inc = eff["bars_include"]
    bars_exc = eff["bars_exclude"]
    mats_inc = eff["materials_include"]
    mats_exc = eff["materials_exclude"]

    kept: List[Hit] = []

    for h in hits_list:
        L = float(getattr(h, "L", 0.0))

        # Light/energy threshold cuts
        if (L < min_L) or (L > max_L):
            _inc(counters, "hits_rejected_threshold", 1)
            if particle_type == "n":
                _inc(counters, "hits_rejected_threshold_n", 1)
            elif particle_type == "g":
                _inc(counters, "hits_rejected_threshold_g", 1)
            continue

        # Bar exclude list (if provided)
        if bars_exc and (h.det_id in bars_exc):
            _inc(counters, "hits_rejected_bar_exclude", 1)
            if particle_type == "n":
                _inc(counters, "hits_rejected_bar_exclude_n", 1)
            elif particle_type == "g":
                _inc(counters, "hits_rejected_bar_exclude_g", 1)
            continue

        # Bar whitelist (if provided)
        if bars_inc and (h.det_id not in bars_inc):
            _inc(counters, "hits_rejected_bar_include", 1)
            if particle_type == "n":
                _inc(counters, "hits_rejected_bar_include_n", 1)
            elif particle_type == "g":
                _inc(counters, "hits_rejected_bar_include_g", 1)
            continue

        # Material exclude list (if provided)
        if mats_exc and (h.material in mats_exc):
            _inc(counters, "hits_rejected_material_exclude", 1)
            if particle_type == "n":
                _inc(counters, "hits_rejected_material_exclude_n", 1)
            elif particle_type == "g":
                _inc(counters, "hits_rejected_material_exclude_g", 1)
            continue

        # Material whitelist (if provided)
        if mats_inc and (h.material not in mats_inc):
            _inc(counters, "hits_rejected_material_include", 1)
            if particle_type == "n":
                _inc(counters, "hits_rejected_material_include_n", 1)
            elif particle_type == "g":
                _inc(counters, "hits_rejected_material_include_g", 1)
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
    cfg,  # reserved for future, e.g. more complex criteria
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
