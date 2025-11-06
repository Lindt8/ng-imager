from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Hit:
    det_id: int
    r: np.ndarray      # (3,) position [cm]
    t_ns: float        # timestamp [ns]
    L: float           # light output [arb]
    material: str      # "M600", "OGS", etc.

    # Optional uncertainties
    sigma_r: np.ndarray | None = None  # (3,)
    sigma_t_ns: float | None = None
    sigma_L: float | None = None

def fake_hits(n: int = 2) -> list[Hit]:
    """Generate placeholder hits for testing."""
    hits = []
    for i in range(n):
        r = np.array([i * 2.0, 0.0, 0.0])
        hits.append(Hit(det_id=i, r=r, t_ns=i * 5.0, L=500.0 + i*100.0, material="M600"))
    return hits
