from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n

@dataclass
class Plane:
    P0: np.ndarray  # (3,)
    n:  np.ndarray  # (3,), unit
    eu: np.ndarray  # (3,), unit
    ev: np.ndarray  # (3,), unit
    u_min: float; u_max: float; du: float
    v_min: float; v_max: float; dv: float

    @classmethod
    def from_cfg(cls, origin, normal, u_min, u_max, du, v_min, v_max, dv, eu=None, ev=None):
        P0 = np.asarray(origin, dtype=np.float64)
        n  = _unit(np.asarray(normal, dtype=np.float64))
        if eu is None or ev is None:
            # auto orthonormal basis
            t = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(t, n)) > 0.9:
                t = np.array([0.0, 1.0, 0.0])
            eu = _unit(np.cross(n, t))
            ev = _unit(np.cross(n, eu))
        else:
            eu = _unit(np.asarray(eu, dtype=np.float64))
            ev = _unit(np.asarray(ev, dtype=np.float64))

        def _is_intish(x: float) -> bool:
            return abs(x - round(x)) < 1e-6

        nu_f = (u_max - u_min) / du
        nv_f = (v_max - v_min) / dv
        if not _is_intish(nu_f) or not _is_intish(nv_f):
            raise ValueError(
                f"Grid does not land on integer bins: "
                f"(u_max-u_min)/du={nu_f:.6f}, (v_max-v_min)/dv={nv_f:.6f}. "
                f"Adjust du/dv or min/max."
            )
        
        return cls(P0, n, eu, ev, u_min, u_max, du, v_min, v_max, dv)

    @property
    def nu(self) -> int:
        return int(np.floor((self.u_max - self.u_min) / self.du + 1e-9)) + 1

    @property
    def nv(self) -> int:
        return int(np.floor((self.v_max - self.v_min) / self.dv + 1e-9)) + 1

    def world_to_plane(self, X: np.ndarray) -> tuple[float, float]:
        d = X - self.P0
        return float(d @ self.eu), float(d @ self.ev)

    def plane_to_world(self, u: float, v: float) -> np.ndarray:
        return self.P0 + u * self.eu + v * self.ev
