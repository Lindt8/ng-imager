from __future__ import annotations

from typing import Iterable, List, Dict, Optional

from ngimager.config.schemas import HitsFiltersCfg
from ngimager.physics.hits import Hit


def _inc(counters: Dict[str, int], key: str, delta: int = 1) -> None:
    """
    Small helper to increment counters safely.
    """
    counters[key] = counters.get(key, 0) + delta


def _resolve_hits_cfg_for_species(
    cfg: HitsFiltersCfg,
    species: Optional[str],
) -> Dict[str, object]:
    """
    Resolve effective hit-level filter parameters for a given species.

    species: 'n', 'g', or None (global / unknown).

    Returns a dict with keys:
      min_light_MeVee, max_light_MeVee,
      psd_min, psd_max,
      bars_include, bars_exclude,
      materials_include, materials_exclude
    """
    base = cfg
    s = (species or "").lower()
    if s == "n":
        ov = cfg.neutron
    elif s == "g":
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
        "psd_min": _pick("psd_min"),
        "psd_max": _pick("psd_max"),
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
        Optional event-level 'n' or 'g'. Used as a fallback species tag
        when Hit.type is missing.
    """
    hits_list = list(hits)

    # Totals before any cuts (event-level accounting)
    _inc(counters, "hits_total", len(hits_list))
    if particle_type == "n":
        _inc(counters, "hits_total_n", len(hits_list))
    elif particle_type == "g":
        _inc(counters, "hits_total_g", len(hits_list))

    # Pre-resolve configs for species and unknown
    global_cfg = _resolve_hits_cfg_for_species(cfg, None)
    n_cfg = _resolve_hits_cfg_for_species(cfg, "n")
    g_cfg = _resolve_hits_cfg_for_species(cfg, "g")

    def cfg_for_hit(h: Hit) -> Dict[str, object]:
        # Prefer per-hit type (from adapters), fall back to event-level particle_type.
        h_type = getattr(h, "type", None)
        s: Optional[str]
        if isinstance(h_type, str) and h_type.lower() in ("n", "g"):
            s = h_type.lower()
        else:
            pt = (particle_type or "").lower()
            s = pt if pt in ("n", "g") else None

        if s == "n":
            return n_cfg
        if s == "g":
            return g_cfg
        return global_cfg

    def suffix_for_hit(h: Hit) -> str:
        h_type = getattr(h, "type", None)
        if isinstance(h_type, str):
            t = h_type.lower()
            if t == "n":
                return "_n"
            if t == "g":
                return "_g"
        # fall back to event-level type
        pt = (particle_type or "").lower()
        if pt == "n":
            return "_n"
        if pt == "g":
            return "_g"
        return ""

    kept: List[Hit] = []

    for h in hits_list:
        eff = cfg_for_hit(h)
        min_L = eff["min_light_MeVee"]
        max_L = eff["max_light_MeVee"]
        psd_min = eff["psd_min"]
        psd_max = eff["psd_max"]
        bars_inc = eff["bars_include"]
        bars_exc = eff["bars_exclude"]
        mats_inc = eff["materials_include"]
        mats_exc = eff["materials_exclude"]

        suffix = suffix_for_hit(h)

        L = float(getattr(h, "L", 0.0))

        # Light/energy threshold cuts
        if (L < min_L) or (L > max_L):
            _inc(counters, "hits_rejected_threshold", 1)
            if suffix:
                _inc(counters, f"hits_rejected_threshold{suffix}", 1)
            continue

        # PSD window cuts (if configured and if the hit carries a PSD value)
        # PSD is expected to be stored in h.extras["psd"] by the adapter.
        if (psd_min is not None) or (psd_max is not None):
            psd_val = None
            extras = getattr(h, "extras", None)
            if extras is not None:
                psd_val = extras.get("psd", None)

            if psd_val is not None:
                too_low = (psd_min is not None) and (psd_val < psd_min)
                too_high = (psd_max is not None) and (psd_val > psd_max)
                if too_low or too_high:
                    _inc(counters, "hits_rejected_psd", 1)
                    if suffix:
                        _inc(counters, f"hits_rejected_psd{suffix}", 1)
                    continue

        # Bar exclude list (if provided)
        if bars_exc and (h.det_id in bars_exc):
            _inc(counters, "hits_rejected_bar_exclude", 1)
            if suffix:
                _inc(counters, f"hits_rejected_bar_exclude{suffix}", 1)
            continue

        # Bar whitelist (if provided)
        if bars_inc and (h.det_id not in bars_inc):
            _inc(counters, "hits_rejected_bar_include", 1)
            if suffix:
                _inc(counters, f"hits_rejected_bar_include{suffix}", 1)
            continue

        # Material exclude list (if provided)
        if mats_exc and (h.material in mats_exc):
            _inc(counters, "hits_rejected_material_exclude", 1)
            if suffix:
                _inc(counters, f"hits_rejected_material_exclude{suffix}", 1)
            continue

        # Material whitelist (if provided)
        if mats_inc and (h.material not in mats_inc):
            _inc(counters, "hits_rejected_material_include", 1)
            if suffix:
                _inc(counters, f"hits_rejected_material_include{suffix}", 1)
            continue

        kept.append(h)

    # Totals after filters
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
