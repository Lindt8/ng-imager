# src/ngimager/physics/kinematics.py
from dataclasses import dataclass
import numpy as np
@dataclass
class Hit:
    det_id: int
    r: np.ndarray      # (3,)
    t_ns: float
    L: float           # light

@dataclass
class NeutronEvent:
    h1: Hit; h2: Hit

@dataclass
class Cone:
    apex: np.ndarray   # O (3,)
    dir:  np.ndarray   # Dhat (3,), unit
    theta: float       # radians
    sigma_theta: float | None = None
