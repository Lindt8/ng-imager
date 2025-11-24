from __future__ import annotations
import os, sys
import numpy as np
from typing import Iterable, NamedTuple, Literal, List, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm  # optional, for progress bars
except Exception:
    tqdm = None  # noqa

from ..geometry.plane import Plane

SBPEngine = Literal["poly", "scan"]

# ----------------- public datatypes -----------------

class Cone:
    def __init__(self, apex: np.ndarray, direction: np.ndarray, theta: float, sigma_theta: float | None = None):
        self.apex = apex.astype(np.float64)
        d = direction.astype(np.float64)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("Cone.direction must be non-zero")
        self.dir = d / n
        self.theta = float(theta)
        self.sigma_theta = sigma_theta

class ReconResult(NamedTuple):
    summed: np.ndarray
    lm_indices: List[np.ndarray] | None

# ----------------- core math (shared with workers) -----------------

def _cone_matrix(D: np.ndarray, theta: float) -> np.ndarray:
    c = np.cos(theta)
    return np.outer(D, D) - (c * c) * np.eye(3)

def _conic_Q(M: np.ndarray, O: np.ndarray, plane: Plane) -> np.ndarray:
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
    """
    Extract ellipse parameters from a 3×3 conic matrix Q in (u, v, 1) coords.

    Returns (center_uv, a, b, R) or None if the conic is not an ellipse.
    """
    A, B2, C = Q[0,0], 2*Q[0,1], Q[1,1]
    disc = B2*B2 - 4*A*C
    if disc >= 0:
        return None  # not an ellipse (parabola/hyperbola/degenerate)
    D, E = 2*Q[0,2], 2*Q[1,2]
    M2 = np.array([[2*A, B2],[B2, 2*C]], dtype=np.float64)
    rhs = -np.array([D, E], dtype=np.float64)
    try:
        uv0 = np.linalg.solve(M2, rhs)
    except np.linalg.LinAlgError:
        return None
    # Translate to center
    T = np.array([[1,0,uv0[0]],[0,1,uv0[1]],[0,0,1]], dtype=np.float64)
    Qc = T.T @ Q @ T
    Q2 = Qc[:2,:2]
    Fp = Qc[2,2]
    if Fp >= 0:
        return None
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
    #u_idx = np.clip(((pts_uv[:,0] - plane.u_min) / plane.du).astype(np.int64), 0, plane.nu-1)
    #v_idx = np.clip(((pts_uv[:,1] - plane.v_min) / plane.dv).astype(np.int64), 0, plane.nv-1)
    #flat = (v_idx * plane.nu + u_idx).astype(np.uint32)
    #return np.unique(flat)
    u = pts_uv[:, 0];
    v = pts_uv[:, 1]
    in_u = (u >= plane.u_min) & (u <= plane.u_max)
    in_v = (v >= plane.v_min) & (v <= plane.v_max)
    keep = in_u & in_v
    if not np.any(keep):
        return np.empty(0, dtype=np.uint32)
    u_idx = ((u[keep] - plane.u_min) / plane.du).astype(np.int64)
    v_idx = ((v[keep] - plane.v_min) / plane.dv).astype(np.int64)
    flat = (v_idx * plane.nu + u_idx).astype(np.uint32)
    return np.unique(flat)

def _scan_conic_indices(Q: np.ndarray, plane: Plane) -> np.ndarray:
    """
    Matrix-math "scan" engine: walk vertical and horizontal lines at pixel
    centers, solve the conic equation analytically, and map all hits to pixels.

    Works for general conics (ellipses, hyperbolas, etc.) as long as they
    intersect the plane within the [u, v] window.
    """
    # Unpack symmetric Q
    q11 = Q[0, 0]
    q12 = Q[0, 1]
    q13 = Q[0, 2]
    q22 = Q[1, 1]
    q23 = Q[1, 2]
    q33 = Q[2, 2]

    # Pixel-center lines
    u_edges = np.linspace(plane.u_min, plane.u_max, plane.nu + 1)
    v_edges = np.linspace(plane.v_min, plane.v_max, plane.nv + 1)
    u_mids = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_mids = 0.5 * (v_edges[:-1] + v_edges[1:])

    us_samples: List[float] = []
    vs_samples: List[float] = []

    eps = 1e-14

    # Horizontal scan: fixed u, solve for v
    for u in u_mids:
        Av = q22
        Bv = 2.0 * q12 * u + 2.0 * q23
        Cv = q11 * u * u + 2.0 * q13 * u + q33

        roots: List[float] = []
        if abs(Av) < eps:
            if abs(Bv) < eps:
                continue
            v = -Cv / Bv
            roots = [v]
        else:
            disc = Bv * Bv - 4.0 * Av * Cv
            if disc < 0.0:
                continue
            sqrt_disc = np.sqrt(max(disc, 0.0))
            v1 = (-Bv - sqrt_disc) / (2.0 * Av)
            v2 = (-Bv + sqrt_disc) / (2.0 * Av)
            if disc <= eps:
                roots = [v1]
            else:
                roots = [v1, v2]

        for v in roots:
            if plane.v_min <= v <= plane.v_max:
                us_samples.append(u)
                vs_samples.append(v)

    # Vertical scan: fixed v, solve for u
    for v in v_mids:
        Au = q11
        Bu = 2.0 * q12 * v + 2.0 * q13
        Cu = q22 * v * v + 2.0 * q23 * v + q33

        roots: List[float] = []
        if abs(Au) < eps:
            if abs(Bu) < eps:
                continue
            u = -Cu / Bu
            roots = [u]
        else:
            disc = Bu * Bu - 4.0 * Au * Cu
            if disc < 0.0:
                continue
            sqrt_disc = np.sqrt(max(disc, 0.0))
            u1 = (-Bu - sqrt_disc) / (2.0 * Au)
            u2 = (-Bu + sqrt_disc) / (2.0 * Au)
            if disc <= eps:
                roots = [u1]
            else:
                roots = [u1, u2]

        for u in roots:
            if plane.u_min <= u <= plane.u_max:
                us_samples.append(u)
                vs_samples.append(v)

    if not us_samples:
        return np.empty(0, dtype=np.uint32)

    pts_uv = np.stack([np.asarray(us_samples), np.asarray(vs_samples)], axis=1)
    return _pixels_from_poly(pts_uv, plane)

def _ray_sample_indices(apex: np.ndarray, Dhat: np.ndarray, theta: float, plane: Plane, n_phi: int = 720) -> np.ndarray:
    # Build orthonormal basis around Dhat
    t = np.array([1.0, 0.0, 0.0])
    if abs(Dhat @ t) > 0.9:
        t = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(Dhat, t)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(Dhat, e1)

    # Ray directions on the cone surface: r_dir = cosθ Dhat + sinθ (cosφ e1 + sinφ e2)
    ct, st = np.cos(theta), np.sin(theta)
    phis = np.linspace(0.0, 2.0*np.pi, n_phi, endpoint=False)
    r_dirs = (ct * Dhat[None, :]) + st * (np.cos(phis)[:, None] * e1[None, :] + np.sin(phis)[:, None] * e2[None, :])

    # Intersect ray with plane: apex + s * r_dir hits plane when (P0 - apex)·n = s (r_dir·n)
    n = plane.n
    denom = r_dirs @ n
    good = np.abs(denom) > 1e-12
    if not np.any(good):
        return np.empty(0, dtype=np.uint32)

    s = ((plane.P0 - apex) @ n) / denom[good]

    # Only keep intersections IN FRONT of the apex
    in_front = s > 0
    if not np.any(in_front):
        return np.empty(0, dtype=np.uint32)

    X = apex[None, :] + s[in_front, None] * r_dirs[good][in_front]

    # Map to (u,v) in plane basis
    du = X - plane.P0
    u = du @ plane.eu
    v = du @ plane.ev

    # Reject out-of-bounds instead of clipping to edges
    in_u = (u >= plane.u_min) & (u <= plane.u_max)
    in_v = (v >= plane.v_min) & (v <= plane.v_max)
    keep = in_u & in_v
    if not np.any(keep):
        return np.empty(0, dtype=np.uint32)

    u_idx = ((u[keep] - plane.u_min) / plane.du).astype(np.int64)
    v_idx = ((v[keep] - plane.v_min) / plane.dv).astype(np.int64)
    flat = (v_idx * plane.nu + u_idx).astype(np.uint32)
    return np.unique(flat)

def cone_to_indices(
    c: Cone,
    plane: Plane,
    engine: SBPEngine = "poly",
    n_poly: int = 360,
) -> np.ndarray:
    """
    Unified entry point: cone → flat pixel indices.

    engine = "scan":
        Use matrix-math scanning across rows/columns (continuous arcs).
    engine = "poly":
        Use ellipse parameterization when possible, falling back to
        general ray sampling for non-elliptic conics.
    """
    M = _cone_matrix(c.dir, c.theta)
    Q = _conic_Q(M, c.apex, plane)

    if engine == "scan":
        return _scan_conic_indices(Q, plane)

    # Default: "poly" perimeter sampling
    el = _ellipse_from_Q(Q)
    if el is None:
        # Fallback: general ray sampling around the cone axis
        return _ray_sample_indices(c.apex, c.dir, c.theta, plane, n_phi=720)
    uv0, a, b, R = el
    pts = _ellipse_poly(uv0, a, b, R, n=n_poly)
    return _pixels_from_poly(pts, plane)


# ----------------- worker & reducer -----------------

def _process_chunk(
    cones: Sequence[Cone],
    plane: Plane,
    list_mode: bool,
    nu: int,
    n_poly: int,
    sbp_engine: SBPEngine,
) -> Tuple[np.ndarray, List[np.ndarray] | None]:
    """Worker: returns (flat_counts, maybe list-of-indices)."""
    flat_counts = np.zeros(nu * plane.nv, dtype=np.uint32)
    lm_list: List[np.ndarray] | None = [] if list_mode else None
    for c in cones:
        idx = cone_to_indices(c, plane, engine=sbp_engine, n_poly=n_poly)
        if idx.size:
            np.add.at(flat_counts, idx, 1)
            if lm_list is not None:
                lm_list.append(idx)
    return flat_counts, lm_list

def _auto_chunk_size(n_cones: int, nu: int, nv: int, workers: int) -> int:
    # heuristic: aim ~ few MB per chunk; ellipse has ~O(200) unique pixels typically
    target_pixels = 200 * 3000  # ~600k increments per chunk
    flat = nu * nv
    # guardrails
    base = max(1000, min(10000, int(target_pixels / max(1, 200))))
    # distribute across workers
    return max(1000, min(20000, int(max(base, n_cones // max(1, workers)))))


# ----------------- public API -----------------

def reconstruct_sbp(
    cones: Iterable[Cone],
    plane: Plane,
    list_mode: bool = False,
    uncertainty_mode: Literal["off", "thicken", "weighted"] = "off",
    workers: int | str = "auto",
    chunk_cones: int | str = "auto",
    progress: bool = True,
    n_poly: int = 360,
    sbp_engine: SBPEngine = "poly",
) -> ReconResult:
    """
    Parallel SBP (analytic conic). If workers==0, runs single-process.

    sbp_engine:
        "poly" – perimeter parametric ellipse (with ray fallback).
        "scan" – matrix-math scan across pixel-centered lines (continuous arcs).
    """
    # Normalize inputs
    cones_list = list(cones)
    N = len(cones_list)
    img = np.zeros((plane.nv, plane.nu), dtype=np.uint32)
    flat_len = plane.nv * plane.nu

    if N == 0:
        return ReconResult(img, [] if list_mode else None)

    if workers == "auto":
        workers = max(1, os.cpu_count() or 1)
    elif isinstance(workers, int):
        workers = max(0, workers)
    else:
        raise ValueError("workers must be int or 'auto'")

    # Single-process path (also good for debugging)
    if workers == 0 or N < 1500:
        hit_count = 0
        lm = [] if list_mode else None
        it = tqdm(cones_list, desc="SBP", unit="cone") if progress and tqdm else cones_list
        for c in it:
            idx = cone_to_indices(c, plane, engine=sbp_engine, n_poly=n_poly)
            if idx.size:
                hit_count += 1
                np.add.at(img.ravel(), idx, 1)
                if lm is not None:
                    lm.append(idx)
        print(f"SBP: {hit_count}/{N} cones intersected the plane")
        return ReconResult(img, lm)

    # Multi-process path
    if chunk_cones == "auto":
        chunk_cones = _auto_chunk_size(N, plane.nu, plane.nv, workers)
    else:
        chunk_cones = int(chunk_cones)

    # Chunk the work
    chunks: List[Sequence[Cone]] = [cones_list[i : i + chunk_cones] for i in range(0, N, chunk_cones)]

    # Progress bar over chunks
    pbar = tqdm(total=len(chunks), desc=f"SBP x{workers}", unit="chunk") if (progress and tqdm) else None

    flat_total = np.zeros(flat_len, dtype=np.uint32)
    lm_all: List[np.ndarray] | None = [] if list_mode else None

    # Use spawn-friendly ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process_chunk, ch, plane, list_mode, plane.nu, n_poly, sbp_engine) for ch in chunks]
        for fut in as_completed(futs):
            flat_counts, lm_list = fut.result()
            flat_total += flat_counts
            if lm_all is not None and lm_list:
                lm_all.extend(lm_list)
            if pbar:
                pbar.update(1)
    if pbar:
        pbar.close()

    img = flat_total.reshape(plane.nv, plane.nu)
    return ReconResult(img, lm_all)
