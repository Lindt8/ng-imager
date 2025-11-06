from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Tuple

from .hits import Hit


def _sorted_hits_by_time(hits: Iterable[Hit]) -> List[Hit]:
    # Sort by (t_ns, det_id) to get deterministic ordering for ties
    return sorted(hits, key=lambda h: (h.t_ns, h.det_id))


@dataclass(slots=True)
class NeutronEvent:
    """
    Two-scatter neutron event.

    Notes:
      - Hits may arrive unsequenced; call .ordered() to get time-ascending copy.
      - .validate(strict=True) enforces strict time ordering (t1 < t2).
        Use strict=False to allow equality (t1 <= t2).
    """
    h1: Hit
    h2: Hit
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- Ordering helpers -------------------------------------------------
    def is_time_ordered(self, *, strict: bool = True) -> bool:
        return (self.h1.t_ns < self.h2.t_ns) if strict else (self.h1.t_ns <= self.h2.t_ns)

    def ordered(self, *, copy: bool = True, strict: bool = False) -> "NeutronEvent":
        """
        Return a time-ordered event (ascending t). If already ordered, either
        return self (copy=False) or a shallow copy (copy=True).
        """
        if self.is_time_ordered(strict=strict):
            return replace(self) if copy else self
        # swap
        return NeutronEvent(h1=self.h2, h2=self.h1, meta=dict(self.meta))

    # ---- Validation -------------------------------------------------------
    def validate(self, *, strict: bool = True) -> None:
        """
        Validate time ordering. Raises ValueError if invalid.
        """
        if not self.is_time_ordered(strict=strict):
            rel = "<" if strict else "<="
            raise ValueError(f"NeutronEvent time order violation: expected h1.t_ns {rel} h2.t_ns "
                             f"(got {self.h1.t_ns} and {self.h2.t_ns})")


@dataclass(slots=True)
class GammaEvent:
    """
    Three-interaction gamma event.

    Notes:
      - Hits can be unsequenced; call .ordered() to get a new, time-ordered event.
      - .validate(strict=True) enforces t1 < t2 < t3; with strict=False, allows ties.
    """
    h1: Hit
    h2: Hit
    h3: Hit
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- Ordering helpers -------------------------------------------------
    def ordered_hits(self) -> Tuple[Hit, Hit, Hit]:
        s = _sorted_hits_by_time((self.h1, self.h2, self.h3))
        return s[0], s[1], s[2]

    def is_time_ordered(self, *, strict: bool = True) -> bool:
        a, b, c = self.h1.t_ns, self.h2.t_ns, self.h3.t_ns
        if strict:
            return (a < b) and (b < c)
        return (a <= b) and (b <= c)

    def ordered(self, *, copy: bool = True, strict: bool = False) -> "GammaEvent":
        """
        Return a time-ordered event (ascending t). If already ordered, either
        return self (copy=False) or a shallow copy (copy=True).
        """
        if self.is_time_ordered(strict=strict):
            return replace(self) if copy else self
        h1, h2, h3 = self.ordered_hits()
        return GammaEvent(h1=h1, h2=h2, h3=h3, meta=dict(self.meta))

    # ---- Validation -------------------------------------------------------
    def validate(self, *, strict: bool = True) -> None:
        """
        Validate time ordering. Raises ValueError if invalid.
        """
        if not self.is_time_ordered(strict=strict):
            rel = "<" if strict else "<="
            a, b, c = self.h1.t_ns, self.h2.t_ns, self.h3.t_ns
            raise ValueError(f"GammaEvent time order violation: expected t1 {rel} t2 {rel} t3 "
                             f"(got {a}, {b}, {c})")
