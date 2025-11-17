from __future__ import annotations

from typing import List, Sequence

from ngimager.physics.events import NeutronEvent, GammaEvent, Event
from ngimager.filters.shapers import ShapedEvent


def shaped_to_typed_events(
    shaped: Sequence[ShapedEvent],
    *,
    order_time: bool = True,
) -> List[Event]:
    """
    Convert shaped events (from shapers.shape_events_for_cones) into
    typed physics events (NeutronEvent / GammaEvent).

    Assumptions:
      - shaped[i].species is "n" or "g"
      - shaped[i].hits has exactly 2 hits for "n", 3 hits for "g"
      - hits are already canonical physics.hits.Hit objects
    """
    typed: List[Event] = []

    for ev in shaped:
        hh = list(ev.hits)
        meta = dict(ev.meta)

        if ev.species == "n":
            if len(hh) != 2:
                # Defensive: inconsistent shaped neutron multiplicity
                # We silently skip for now; could log into counters later.
                continue
            obj = NeutronEvent(h1=hh[0], h2=hh[1], meta=meta)
            typed.append(obj.ordered() if order_time else obj)

        elif ev.species == "g":
            if len(hh) != 3:
                # Defensive: inconsistent shaped gamma multiplicity
                continue
            obj = GammaEvent(h1=hh[0], h2=hh[1], h3=hh[2], meta=meta)
            typed.append(obj.ordered() if order_time else obj)

        else:
            # Unknown species are ignored for cone building; could increment
            # events_rejected_* counters in the caller.
            continue

    return typed
