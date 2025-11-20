from __future__ import annotations

from typing import Dict, List, Sequence

from ngimager.config.schemas import EventsFiltersCfg
from ngimager.physics.events import NeutronEvent, GammaEvent, Event


def _inc(counters: Dict[str, int], key: str, delta: int = 1) -> None:
    counters[key] = counters.get(key, 0) + delta


def _resolve_tof_window_ns(
    cfg: EventsFiltersCfg,
    species: str,
) -> List[float]:
    """
    Resolve the effective ToF window [t_min, t_max] in ns for a given species.

    Priority:
        per-species override, else global.
    """
    s = species.lower()
    if s == "n":
        ov = cfg.neutron.tof_window_ns
    elif s == "g":
        ov = cfg.gamma.tof_window_ns
    else:
        ov = None

    if ov is not None:
        return list(ov)

    return list(cfg.tof_window_ns)


def apply_event_filters(
    events: Sequence[Event],
    cfg: EventsFiltersCfg,
    counters: Dict[str, int],
) -> list[Event]:
    """
    Apply simple event-level filters, currently ToF windows based on species.

    For now:
      - neutron: Δt12 = h2.t_ns - h1.t_ns must lie in the neutron ToF window
      - gamma:   Δt12 = h2.t_ns - h1.t_ns must lie in the gamma ToF window

    Events that fail are removed and counted in events_rejected_filters
    and events_rejected_tof_window{_n,_g}.
    """
    kept: list[Event] = []

    for ev in events:
        if isinstance(ev, NeutronEvent):
            species = "n"
        elif isinstance(ev, GammaEvent):
            species = "g"
        else:
            # Unknown species → keep for now, no ToF cut.
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

        tmin, tmax = _resolve_tof_window_ns(cfg, species)
        if (dt < tmin) or (dt > tmax):
            _inc(counters, "events_rejected_filters", 1)
            _inc(counters, "events_rejected_tof_window", 1)
            if species == "n":
                _inc(counters, "events_rejected_tof_window_n", 1)
            elif species == "g":
                _inc(counters, "events_rejected_tof_window_g", 1)
            continue

        kept.append(ev)

    _inc(counters, "events_after_filters", len(kept))
    return kept
