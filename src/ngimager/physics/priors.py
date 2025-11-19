# src/ngimager/physics/priors.py
from __future__ import annotations
from typing import Protocol, Optional, Literal
import numpy as np

from ngimager.geometry.plane import Plane

class Prior(Protocol):
    name: str
    def weight_field(self, plane: Plane) -> np.ndarray:
        """Return (nv, nu) float32 weights in [0,1]."""

class PointPrior:
    def __init__(self, point_xyz: np.ndarray, strength: float = 1.0):
        self.point = np.asarray(point_xyz, dtype=float)
        self.strength = float(strength)
        self.name = "point"

    def weight_field(self, plane: Plane) -> np.ndarray:
        # Simple 2D Gaussian centered at the projection of point onto plane.
        u0, v0 = plane.world_to_plane(self.point)
        u = np.linspace(plane.u_min, plane.u_max, plane.nu)
        v = np.linspace(plane.v_min, plane.v_max, plane.nv)
        U, V = np.meshgrid(u, v)
        sigma_u = 0.5 * (plane.u_max - plane.u_min)
        sigma_v = 0.5 * (plane.v_max - plane.v_min)
        W = np.exp(-0.5 * (((U - u0) / sigma_u) ** 2 + ((V - v0) / sigma_v) ** 2))
        return (self.strength * W).astype(np.float32)

class LinePrior:
    def __init__(self, p0: np.ndarray, p1: np.ndarray, strength: float = 1.0):
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.strength = float(strength)
        self.name = "line"

    def weight_field(self, plane: Plane) -> np.ndarray:
        # Very simple “fat line” prior: high weight in a band between projections of p0 and p1.
        u0, v0 = plane.world_to_plane(self.p0)
        u1, v1 = plane.world_to_plane(self.p1)
        u = np.linspace(plane.u_min, plane.u_max, plane.nu)
        v = np.linspace(plane.v_min, plane.v_max, plane.nv)
        U, V = np.meshgrid(u, v)
        # Distance from each (u,v) to line segment in the plane
        P = np.stack([U, V], axis=-1)
        A = np.array([u0, v0])
        B = np.array([u1, v1])
        AB = B - A
        t = np.clip(((P - A) @ AB) / (AB @ AB), 0.0, 1.0)
        proj = A + t[..., None] * AB
        dist = np.linalg.norm(P - proj, axis=-1)
        sigma = 0.2 * max(plane.u_max - plane.u_min, plane.v_max - plane.v_min)
        W = np.exp(-0.5 * (dist / sigma) ** 2)
        return (self.strength * W).astype(np.float32)

def make_prior(cfg_prior: dict, plane: Plane) -> Optional[Prior]:
    """
    Small factory used by pipelines.core; returns a Prior or None.

    Expected cfg_prior schema (from TOML, after Pydantic):

      [prior]
      type = "none" | "point" | "line"
      strength = 1.0

      # Point prior:
      # either:
      #   point = [x, y, z]
      # or (future) nested [prior.point] can be normalized upstream.

      # Line prior (preferred, nested):
      #   [prior.line]
      #   p0 = [x0, y0, z0]
      #   p1 = [x1, y1, z1]
      # or:
      #   [prior.line]
      #   r0        = [x0, y0, z0]
      #   direction = [dx, dy, dz]
      #
      # For backward compatibility we also accept flat:
      #   line_p0 = [x0, y0, z0]
      #   line_p1 = [x1, y1, z1]
    """
    #typ = (cfg_prior.get("type") or "none").lower()
    typ = cfg_prior.get("type", "none")
    strength = float(cfg_prior.get("strength", 1.0))

    if typ == "none":
        return None
    
    if typ == "point":
        #return PointPrior(np.asarray(cfg_prior["point"], dtype=float), strength=strength)
        if "point" in cfg_prior:
            p = np.asarray(cfg_prior["point"], dtype=float)
        else:
            # default: center of imaging plane
            p = plane.center  # or plane.origin() if we add such a helper
        return PointPrior(p, strength=strength)

    if typ == "line":
        line_cfg = cfg_prior.get("line")
        p0 = p1 = None

        if line_cfg is not None:
            # Preferred nested forms
            if "p0" in line_cfg and "p1" in line_cfg:
                p0 = np.asarray(line_cfg["p0"], dtype=float)
                p1 = np.asarray(line_cfg["p1"], dtype=float)
            elif "r0" in line_cfg and "direction" in line_cfg:
                r0 = np.asarray(line_cfg["r0"], dtype=float)
                direction = np.asarray(line_cfg["direction"], dtype=float)
                p0 = r0
                p1 = r0 + direction
            else:
                raise KeyError(
                    "prior.line must define either p0/p1 or r0/direction "
                    "(e.g. [prior.line] p0=[...] p1=[...] or r0=[...] direction=[...])"
                )
        else:
            # Backwards-compatible flat keys (if someone ever used them)
            if "line_p0" in cfg_prior and "line_p1" in cfg_prior:
                p0 = np.asarray(cfg_prior["line_p0"], dtype=float)
                p1 = np.asarray(cfg_prior["line_p1"], dtype=float)
            else:
                raise KeyError(
                    "Line prior configured with type='line' but no usable "
                    "line geometry found (expected [prior.line] or line_p0/line_p1)."
                )

        return LinePrior(p0, p1, strength=strength)
    
    raise ValueError(f"Unknown prior.type={cfg_prior['type']!r}")
