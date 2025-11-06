from __future__ import annotations
from dataclasses import dataclass
from .hits import Hit

@dataclass
class NeutronEvent:
    """Two-scatter neutron event."""
    h1: Hit
    h2: Hit

    def validate(self):
        assert self.h1.t_ns < self.h2.t_ns, "hit1 must occur before hit2"

@dataclass
class GammaEvent:
    """Three-interaction gamma event."""
    h1: Hit
    h2: Hit
    h3: Hit

    def ordered(self) -> list[Hit]:
        return sorted([self.h1, self.h2, self.h3], key=lambda h: h.t_ns)
