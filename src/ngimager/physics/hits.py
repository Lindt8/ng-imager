from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

@dataclass(slots=True)
class Hit:
    """
    Canonical detector hit (physics layer).

    r: position [cm]
    t_ns: time [ns]
    L: light-like measure (e.g., Elong) (dimensionless or MeVee-scale per your LUT)
    material: detector material tag (e.g., "M600")
    extras: arbitrary per-hit fields preserved from input (psd, dE_MeV, raw columns...)
    """
    det_id: int
    r: np.ndarray  # shape (3,), dtype float
    t_ns: float
    L: float = 0.0
    material: str = "UNK"

    # Optional first-order uncertainties (hooks only; can be None/unused for now)
    sigma_r_cm: Optional[float] = None
    sigma_t_ns: Optional[float] = None
    sigma_L: Optional[float] = None

    # Preserve raw/source-specific fields for later filtering without polluting the core schema
    extras: Dict[str, Any] = field(default_factory=dict)

def fake_hits(n: int = 2) -> list[Hit]:
    """Generate placeholder hits for testing."""
    hits = []
    for i in range(n):
        r = np.array([i * 2.0, 0.0, 0.0])
        hits.append(Hit(det_id=i, r=r, t_ns=i * 5.0, L=500.0 + i*100.0, material="M600"))
    return hits
