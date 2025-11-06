# src/ngimager/physics/priors.py
from typing import Protocol
import numpy as np

class Prior(Protocol):
    name: str
    def weight_field(self, plane: Plane) -> np.ndarray:
        """Return (nv, nu) float32 weights in [0,1]."""

class PointPrior:
    def __init__(self, point_xyz: np.ndarray, strength: float = 1.0): ...
    def weight_field(self, plane: Plane) -> np.ndarray: ...

class LinePrior:
    def __init__(self, p0: np.ndarray, p1: np.ndarray, strength: float = 1.0): ...
    def weight_field(self, plane: Plane) -> np.ndarray: ...
