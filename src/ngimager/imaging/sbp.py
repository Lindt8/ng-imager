from __future__ import annotations
import numpy as np
from typing import Iterable, NamedTuple, Literal, List
from ..geometry.plane import Plane

class Cone:
    def __init__(self, apex: np.ndarray, direction: np.ndarray, theta: float, sigma_theta: float | None = None):
        self.apex = apex.astype(np.float64)
        d = direction.astype(np.float64)
        self.dir = d / np.linalg.norm(d)
        self.theta = float(theta)
        self.sigma_theta = sigma_theta

class ReconResult(NamedTuple):
    summed: np.ndarray
    lm_indices: List[np.ndarray] | None

def _cone_matrix(D: np.ndarray, theta: float) -> np.ndarray:
    # M = D D^T - cos^2(theta) I
    c = np.cos(theta)
    return np.outer(D, D) - (c * c) * np.eye(3)

def _conic_Q(M: np.ndarray, O: np.ndarray, plane: Plane) -> np.ndarray:
    # Substitute X = P0 + u eu + v ev into (X-O)^T M (X-O) = 0, collect to Q
    P0, eu, ev = plane.P0, plane.eu, plane.ev
    PO = (P0 - O)
    A = eu @ M @ eu
    B = 2.0 * (eu @ M @ ev)
    C = ev @ M @ ev
    D = 2.0 * (PO @ M @ eu)
    E = 2.0 * (PO @ M @ ev)
    F = PO @ M @ PO
    Q = np.array([[A, B/2.0, D/2.0],
                  [B/2.0, C, E/2.0],
                  [D/2.0, E/2.0, F]], dtype=np.float64)
    return Q

def _ellipse_from_Q(Q: np.ndarray):
    # Detect ellipse: discriminant > 0 for ellipse in conic Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    A, B, C = Q[0,0], 2*Q[0,1], Q[1,1]
    disc = B*B - 4*A*C
    if disc >= 0:
        return None  # parabola/hyperbola/degenerate -> skip for now
    # Find center by solving gradient = 0: [2A  B; B 2C][u v]^T = -[D E]
    D, E = 2*Q[0,2], 2*Q[1,2]
    M2 = np.array([[2*A, B],[B, 2*C]], dtype=np.float64)
    rhs = -np.array([D, E], dtype=np.float64)
    try:
        uv0 = np.linalg.solve(M2, rhs)
    except np.linalg.LinAlgError:
        return None
    # Translate to center, get quadratic form matrix for ellipse
    T = np.array([[1,0,uv0[0]],[0,1,uv0[1]],[0,0,1]], dtype=np.float64)
    Qc = T.T @ Q @ T
    # Now conic is [u v 1]T Qc [u v 1] = 0 with no linear terms
    Q2 = Qc[:2,:2]
    Fp = Qc[2,2]
    if Fp >= 0:
        return None
    # Eigen-decompose Q2 to get axes
    evals, evecs = np.linalg.eigh(Q2)
    if np.any(evals <= 0):
        return None
    a = np.sqrt(-Fp / evals[0])
    b = np.sqrt(-Fp / evals[1])
    R = evecs  # columns are principal directions
    return uv0, a, b, R

def _ellipse_poly(uv0, a, b, R, n=360):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    circ = np.stack([a*np.cos(t), b*np.sin(t)], axis=0)  # (2,n)
    pts = (R @ circ).T + uv0  # (n,2)
    return pts

def _pixels_from_poly(pts_uv: np.ndarray, plane: Plane) -> np.ndarray:
    # Rasterize polygon boundary as set of pixel centers it crosses (cheap approach)
    # Map u,v to pixel indices
    u_idx = np.clip(((pts_uv[:,0] - plane.u_min) / plane.du).astype(np.int64), 0, plane.nu-1)
    v_idx = np.clip(((pts_uv[:,1] - plane.v_min) / plane.dv).astype(np.int64), 0, plane.nv-1)
    flat = (v_idx * plane.nu + u_idx).astype(np.uint32)
    # Deduplicate to keep sparse list small
    return np.unique(flat)

def reconstruct_sbp(
    cones: Iterable[Cone],
    plane: Plane,
    list_mode: bool = False,
    uncertainty_mode: Literal["off","thicken","weighted"] = "off",
) -> ReconResult:
    img = np.zeros((plane.nv, plane.nu), dtype=np.uint32)
    lm: list[np.ndarray] | None = [] if list_mode else None
    for c in cones:
        M = _cone_matrix(c.dir, c.theta)
        Q = _conic_Q(M, c.apex, plane)
        el = _ellipse_from_Q(Q)
        if el is None:
            continue
        uv0, a, b, R = el
        pts = _ellipse_poly(uv0, a, b, R, n=360)
        idx = _pixels_from_poly(pts, plane)
        np.add.at(img.ravel(), idx, 1)
        if lm is not None:
            lm.append(idx)
    return ReconResult(img, lm)
