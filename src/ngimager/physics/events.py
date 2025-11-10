# src/ngimager/physics/events.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Union

import numpy as np

from .hits import Hit

@dataclass(slots=True)
class NeutronEvent:
    """
    Two-scatter neutron event.

    Hits can be unsequenced when first ingested (e.g. from ROOT/PHITS),
    so use .ordered() to get a time-ordered event and .validate() to
    assert the ordering.
    """
    h1: Hit
    h2: Hit
    meta: Dict[str, Any] = field(default_factory=dict)

    def ordered(self, copy: bool = True) -> "NeutronEvent":
        """
        Return a NeutronEvent with hits ordered by t_ns (h1 earliest).

        If copy=False, reorders self in-place and returns self.
        """
        hits = [self.h1, self.h2]
        hits.sort(key=lambda h: h.t_ns)
        if copy:
            return NeutronEvent(h1=hits[0], h2=hits[1], meta=dict(self.meta))
        self.h1, self.h2 = hits
        return self

    def is_time_ordered(self, strict: bool = True) -> bool:
        if strict:
            return self.h1.t_ns < self.h2.t_ns
        return self.h1.t_ns <= self.h2.t_ns

    def validate(self, strict: bool = True) -> None:
        """
        Raise ValueError if the hits are not in time order.
        """
        if not self.is_time_ordered(strict=strict):
            raise ValueError(
                f"NeutronEvent time order violation: "
                f"h1.t_ns={self.h1.t_ns}, h2.t_ns={self.h2.t_ns}"
            )

@dataclass(slots=True)
class GammaEvent:
    """
    Three-interaction gamma event.

    As with NeutronEvent, hits can arrive unsequenced; use .ordered()
    to get a time-ordered copy and .validate() to assert ordering.
    """
    h1: Hit
    h2: Hit
    h3: Hit
    meta: Dict[str, Any] = field(default_factory=dict)

    def _sorted_hits(self) -> List[Hit]:
        return sorted((self.h1, self.h2, self.h3), key=lambda h: h.t_ns)

    def ordered(self, copy: bool = True) -> "GammaEvent":
        """
        Return a GammaEvent with hits sorted by t_ns (h1 earliest).

        If copy=False, reorders self in-place and returns self.
        """
        hits = self._sorted_hits()
        if copy:
            return GammaEvent(h1=hits[0], h2=hits[1], h3=hits[2], meta=dict(self.meta))
        self.h1, self.h2, self.h3 = hits
        return self

    def is_time_ordered(self, strict: bool = True) -> bool:
        t1, t2, t3 = self.h1.t_ns, self.h2.t_ns, self.h3.t_ns
        if strict:
            return (t1 < t2) and (t2 < t3)
        return (t1 <= t2) and (t2 <= t3)

    def validate(self, strict: bool = True) -> None:
        """
        Raise ValueError if hits are not in (weakly/strictly) increasing time.
        """
        if not self.is_time_ordered(strict=strict):
            raise ValueError(
                f"GammaEvent time order violation: "
                f"[{self.h1.t_ns}, {self.h2.t_ns}, {self.h3.t_ns}]"
            )

Event = Union[NeutronEvent, GammaEvent]
