from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ngimager.config.schemas import EventsFiltersCfg
from ngimager.physics.events import NeutronEvent, GammaEvent, Event


def _inc(counters: Dict[str, int], key: str, delta: int = 1) -> None:
    counters[key] = counters.get(key, 0) + delta


def _resolve_event_params(
    cfg: EventsFiltersCfg,
    species: str,
) -> Tuple[List[float], float | None, float | None, float | None]:
    """
    Resolve effective event-level parameters for a given species:

      - ToF window [t_min, t_max] in ns
      - min_L1_MeVee (optional)
      - min_L2_MeVee (optional)
      - min_L_any_MeVee (optional)
    """
    s = species.lower()
    if s == "n":
        ov = cfg.neutron
    elif s == "g":
        ov = cfg.gamma
    else:
        ov = None

    # ToF window: per-species override falls back to global
    if ov is not None and ov.tof_window_ns is not None:
        tof = list(ov.tof_window_ns)
    else:
        tof = list(cfg.tof_window_ns)

    min_L1 = ov.min_L1_MeVee if ov is not None else None
    min_L2 = ov.min_L2_MeVee if ov is not None else None
    min_L_any = ov.min_L_any_MeVee if ov is not None else None

    return tof, min_L1, min_L2, min_L_any


def apply_event_filters(
    events: Sequence[Event],
    cfg: EventsFiltersCfg,
    counters: Dict[str, int],
) -> list[Event]:
    """
    Apply event-level filters, currently:

      - species-dependent ToF windows (Δt12 between first two hits)
      - neutron-specific min L thresholds for first and second scatters
      - gamma-specific min L threshold applied to all three scatters

    Events that fail are removed and counted in events_rejected_filters and
    more specific counters:

      - events_rejected_tof_window{,_n,_g}
      - events_rejected_L1_min_n
      - events_rejected_L2_min_n
      - events_rejected_L_any_min_g
    """
    kept: list[Event] = []

    for ev in events:
        if isinstance(ev, NeutronEvent):
            species = "n"
        elif isinstance(ev, GammaEvent):
            species = "g"
        else:
            # Unknown species → keep for now, no event-level cuts.
            kept.append(ev)
            continue

        _inc(counters, "events_total_for_filters", 1)
        if species == "n":
            _inc(counters, "events_total_for_filters_n", 1)
        elif species == "g":
            _inc(counters, "events_total_for_filters_g", 1)

        # We assume typed events always have at least two hits.
        h1 = ev.h1
        h2 = ev.h2
        dt = float(h2.t_ns - h1.t_ns)

        (tof_window, min_L1, min_L2, min_L_any) = _resolve_event_params(cfg, species)
        tmin, tmax = tof_window

        # --- ToF window cut ---
        if (dt < tmin) or (dt > tmax):
            _inc(counters, "events_rejected_filters", 1)
            _inc(counters, "events_rejected_tof_window", 1)
            if species == "n":
                _inc(counters, "events_rejected_tof_window_n", 1)
            elif species == "g":
                _inc(counters, "events_rejected_tof_window_g", 1)
            continue

        # --- Neutron-specific L thresholds (L1/L2) ---
        if species == "n":
            L1 = float(getattr(h1, "L", 0.0))
            L2 = float(getattr(h2, "L", 0.0))

            if (min_L1 is not None) and (L1 < min_L1):
                _inc(counters, "events_rejected_filters", 1)
                _inc(counters, "events_rejected_L1_min_n", 1)
                continue

            if (min_L2 is not None) and (L2 < min_L2):
                _inc(counters, "events_rejected_filters", 1)
                _inc(counters, "events_rejected_L2_min_n", 1)
                continue

        # --- Gamma-specific L threshold (any scatter) ---
        if species == "g" and (min_L_any is not None):
            Ls = [
                float(getattr(ev.h1, "L", 0.0)),
                float(getattr(ev.h2, "L", 0.0)),
                float(getattr(ev.h3, "L", 0.0)),
            ]
            if any(L < min_L_any for L in Ls):
                _inc(counters, "events_rejected_filters", 1)
                _inc(counters, "events_rejected_L_any_min_g", 1)
                continue

        kept.append(ev)

    _inc(counters, "events_after_filters", len(kept))
    return kept
