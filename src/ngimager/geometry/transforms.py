from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_vec3(values: Sequence[float]) -> np.ndarray:
    """
    Convert `values` to a 3-element float64 vector.

    Accepts any 1D iterable of length 3; raises ValueError otherwise.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected 3 values, got shape {arr.shape!r}")
    return arr.astype(np.float64)


def euler_xyz_deg_to_matrix(rotation_deg: Sequence[float]) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from XYZ Euler angles in degrees.

    Convention (matches what we discussed):

      - `rotation_deg = (rx, ry, rz)` in degrees.
      - Rotations applied in fixed order Rx → Ry → Rz.
      - Mathematical column-vector form:

            v_world = R_xyz @ v_local

    For row-vector data (shape (..., 3)), right-multiply by R.T instead.
    """
    rx_deg, ry_deg, rz_deg = _as_vec3(rotation_deg)
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # Elementary rotations (active, right-handed)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=np.float64,
    )
    Ry = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    )
    Rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Apply Rx then Ry then Rz:
    #   v_world = Rz @ (Ry @ (Rx @ v_local))
    return (Rz @ Ry) @ Rx


def apply_rigid_transform(
    points_cm: np.ndarray | Sequence[Sequence[float]],
    origin_cm: Sequence[float],
    rotation_deg: Sequence[float],
) -> np.ndarray:
    """
    Apply a rigid transform (rotation + translation) to 3D points.

    Parameters
    ----------
    points_cm
        Array-like of shape (..., 3). Interpreted as *row* vectors in cm.
    origin_cm
        Translation vector (x, y, z) in cm, giving the detector origin
        in the world frame.
    rotation_deg
        Euler angles (rx, ry, rz) in degrees, applied as Rx → Ry → Rz.

    Returns
    -------
    np.ndarray
        Transformed points with the same shape as `points_cm`.

    Notes
    -----
    Mathematically:

        p_world = R_xyz(rotation_deg) @ p_local + origin_cm

    With row-vector data, we implement this as:

        p_world_row = p_local_row @ R_xyz(rotation_deg).T + origin_cm
    """
    pts = np.asarray(points_cm, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError(
            f"`points_cm` must have shape (..., 3); got shape {pts.shape!r}"
        )

    R = euler_xyz_deg_to_matrix(rotation_deg)
    t = _as_vec3(origin_cm)

    pts_flat = pts.reshape(-1, 3)
    out_flat = pts_flat @ R.T + t  # row-vector convention
    return out_flat.reshape(pts.shape)


def is_identity_transform(
    origin_cm: Sequence[float],
    rotation_deg: Sequence[float],
    tol: float = 1e-9,
) -> bool:
    """
    Return True if the transform is (numerically) the identity.

    This lets the pipeline skip transform work when both the origin and
    all Euler angles are effectively zero.
    """
    o = _as_vec3(origin_cm)
    r = _as_vec3(rotation_deg)
    return np.allclose(o, 0.0, atol=tol) and np.allclose(r, 0.0, atol=tol)
