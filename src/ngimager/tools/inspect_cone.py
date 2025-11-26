#!/usr/bin/env python
"""
inspect_cone.py

Helper for tracing a single cone through the ngimager HDF5 file.

Given an ngimager HDF5 output and either:
  * a cone index,
  * an event index, or
  * an "imaged" cone index (from /lm/event_imaged_cone_id),

this tool reconstructs the full chain

    cone → (u, v) pixels → hits

and prints a human-readable summary. Optionally it can also plot the
per-cone footprint on the imaging plane.

Usage
-----

    python -m ngimager.tools.inspect_cone path/to/file.h5 --cone-index 42
    python -m ngimager.tools.inspect_cone path/to/file.h5 --event-index 10
    python -m ngimager.tools.inspect_cone path/to/file.h5 --imaged-cone-index 7

    # Show a quick imshow of the cone footprint:
    python -m ngimager.tools.inspect_cone file.h5 --cone-index 42 --plot

Notes
-----
- Intended for list-mode ngimager outputs (`run.list = true`), but will
  gracefully degrade when list-mode pixel data are missing.
- For large files, /lm/cone_pixel_indices can be big; this tool reads the
  whole dataset into memory, which is convenient but may be heavy for
  extremely large runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


@dataclass
class ConeTrace:
    """In-memory representation of one cone and its provenance."""
    cone_index: int
    event_index: int
    is_gamma: bool

    # Cone-level quantities
    cone_apex_cm: np.ndarray
    cone_axis_dir: np.ndarray
    cone_theta_deg: float
    cone_incident_energy_MeV: float
    cone_species_code: int
    cone_recoil_code: int
    gamma_hit_order: Optional[np.ndarray]

    # Hit-level quantities (as stored in /lm/hit_*)
    hits: List[Dict[str, Any]]

    # Pixel-level quantities
    pixels_available: bool
    flat_pixel_indices: np.ndarray
    u_idx: np.ndarray
    v_idx: np.ndarray
    image_uv: np.ndarray  # 2D array in (v, u) order


def _infer_image_shape(f: h5py.File) -> Tuple[int, int]:
    """
    Infer (nv, nu) from either /meta attributes or /images/summed.

    Returns
    -------
    nv, nu : int
        Number of pixels in v (rows) and u (columns).
    """
    # Prefer explicit grid info from /meta
    meta_grp = f.get("meta")
    if isinstance(meta_grp, h5py.Group):
        nv = meta_grp.attrs.get("grid.nv")
        nu = meta_grp.attrs.get("grid.nu")
        if nv is not None and nu is not None:
            return int(nv), int(nu)

    # Fallback: infer from any 2D dataset under /images/summed
    images_grp = f.get("/images/summed")
    if isinstance(images_grp, h5py.Group):
        for ds in images_grp.values():
            if isinstance(ds, h5py.Dataset) and ds.ndim == 2:
                nv_ds, nu_ds = ds.shape
                return int(nv_ds), int(nu_ds)

    raise RuntimeError(
        "Could not infer image shape: neither /meta grid.nv/grid.nu nor "
        "a 2D dataset under /images/summed is available."
    )


def _material_lookup(lm_grp: h5py.Group) -> Dict[int, str]:
    """
    Build a small mapping from material_id → label using
    /lm/hit_material_id_labels when present.
    """
    out: Dict[int, str] = {}
    labels_ds = lm_grp.get("hit_material_id_labels")
    if labels_ds is None:
        return out

    labels = np.asarray(labels_ds[...])
    for i, name in enumerate(labels):
        if isinstance(name, bytes):
            name_str = name.decode("utf-8", errors="replace")
        else:
            name_str = str(name)
        out[int(i)] = name_str
    return out


def _decode_event_type(code: int) -> str:
    if code == 0:
        return "neutron"
    if code == 1:
        return "gamma"
    return f"unknown({code})"


def _extract_cone_pixels(
    f: h5py.File,
    cone_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Return (flat_indices, u_idx, v_idx, pixels_available) for a single cone.

    flat_indices are the raw flattened indices as stored in
    /lm/cone_pixel_indices[:, 1].  u_idx and v_idx are the corresponding
    0-based u and v pixel indices on the imaging grid.

    pixels_available is True only if /lm/cone_pixel_indices exists; it is
    False when list-mode pixel data are absent.
    """
    lm_grp = f.get("/lm")
    if not isinstance(lm_grp, h5py.Group):
        # No list-mode group at all: no pixels
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            False,
        )

    ds = lm_grp.get("cone_pixel_indices")
    if ds is None:
        # List-mode imaging was not run or this is a pre-pixel file.
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            False,
        )

    arr = np.asarray(ds[...], dtype=np.uint32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(
            f"/lm/cone_pixel_indices has unexpected shape {arr.shape!r}; "
            "expected (K, 2)."
        )

    if arr.size == 0:
        return (
            np.zeros(0, dtype=np.uint32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            True,
        )

    nv, nu = _infer_image_shape(f)
    mask = arr[:, 0] == cone_index
    flat = arr[mask, 1]
    if flat.size == 0:
        # Cone exists but did not hit the plane
        return (
            flat.astype(np.uint32, copy=False),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            True,
        )

    v_idx = (flat // nu).astype(np.int64)
    u_idx = (flat % nu).astype(np.int64)
    return flat.astype(np.uint32, copy=False), u_idx, v_idx, True


def _reconstruct_image(
    nv: int,
    nu: int,
    u_idx: np.ndarray,
    v_idx: np.ndarray,
) -> np.ndarray:
    """
    Build a sparse 2D image (nv, nu) with integer counts per pixel.

    Multiple entries for the same (u, v) are summed.
    """
    img = np.zeros((nv, nu), dtype=np.int32)
    if u_idx.size == 0:
        return img
    np.add.at(img, (v_idx, u_idx), 1)
    return img


def trace_cone_from_index(f: h5py.File, cone_index: int) -> ConeTrace:
    """
    High-level helper: build a ConeTrace object for cone_index.
    """
    cones_grp = f.get("/cones")
    if not isinstance(cones_grp, h5py.Group):
        raise RuntimeError("This file has no /cones group; did the pipeline reach stage 3?")

    cone_ids = cones_grp.get("cone_id")
    if cone_ids is None:
        raise RuntimeError("Missing /cones/cone_id dataset.")

    n_cones = int(cone_ids.shape[0])
    if cone_index < 0 or cone_index >= n_cones:
        raise IndexError(f"cone_index {cone_index} is out of range [0, {n_cones}).")

    # Cone-level quantities (dataset names follow hdf5.md and lm_store.py)
    event_index_ds = cones_grp.get("event_index")
    if event_index_ds is None:
        raise RuntimeError("Missing /cones/event_index dataset for cone→event mapping.")
    event_index = int(event_index_ds[cone_index])

    apex_cm = np.asarray(cones_grp["apex_xyz_cm"][cone_index], dtype=np.float64)
    axis_dir = np.asarray(cones_grp["axis_xyz"][cone_index], dtype=np.float64)
    theta_rad = float(cones_grp["theta_rad"][cone_index])
    theta_deg = float(np.degrees(theta_rad))

    incident_E = float(cones_grp["incident_energy_MeV"][cone_index])
    species_code = int(cones_grp["species"][cone_index])
    recoil_code = int(cones_grp["recoil_code"][cone_index])

    gamma_hit_order_ds = cones_grp.get("gamma_hit_order")
    gamma_hit_order_row: Optional[np.ndarray]
    if gamma_hit_order_ds is not None:
        row = np.asarray(gamma_hit_order_ds[cone_index], dtype=np.int8)
        # Treat rows with any non-negative entry as meaningful sequencing.
        if row.ndim == 1 and row.size == 3 and np.any(row >= 0):
            gamma_hit_order_row = row
        else:
            gamma_hit_order_row = None
    else:
        gamma_hit_order_row = None

    # Event- and hit-level quantities
    lm_grp = f.get("/lm")
    hits: List[Dict[str, Any]] = []
    is_gamma = (species_code == 1)

    if isinstance(lm_grp, h5py.Group):
        event_type_ds = lm_grp.get("event_type")
        if event_type_ds is not None and event_index < event_type_ds.shape[0]:
            event_type_code = int(event_type_ds[event_index])
            is_gamma = (event_type_code == 1)

        # Hit-level datasets are optional; populate hits only if present.
        hit_pos_ds = lm_grp.get("hit_pos_cm")
        hit_t_ds = lm_grp.get("hit_t_ns")
        hit_L_ds = lm_grp.get("hit_L_mevee")
        hit_det_ds = lm_grp.get("hit_det_id")
        hit_mat_ds = lm_grp.get("hit_material_id")

        have_hits = all(
            ds is not None
            for ds in (hit_pos_ds, hit_t_ds, hit_L_ds, hit_det_ds, hit_mat_ds)
        )

        if have_hits:
            hit_pos = np.asarray(hit_pos_ds[event_index], dtype=np.float64)  # (3, 3)
            hit_t = np.asarray(hit_t_ds[event_index], dtype=np.float64)      # (3,)
            hit_L = np.asarray(hit_L_ds[event_index], dtype=np.float64)      # (3,)
            hit_det = np.asarray(hit_det_ds[event_index], dtype=np.int32)    # (3,)
            hit_mat = np.asarray(hit_mat_ds[event_index], dtype=np.int16)    # (3,)

            mat_lookup = _material_lookup(lm_grp)

            for j in range(hit_pos.shape[0]):
                r = hit_pos[j]
                # Sentinel convention: NaN position or det_id < 0 means "no hit"
                if np.all(np.isnan(r)) and hit_det[j] < 0:
                    continue
                mat_id = int(hit_mat[j])
                mat_name = mat_lookup.get(mat_id, "UNK" if mat_id < 0 else f"id={mat_id}")
                hits.append(
                    {
                        "hit_index": j,
                        "r_cm": r,
                        "t_ns": float(hit_t[j]),
                        "L_mevee": float(hit_L[j]),
                        "det_id": int(hit_det[j]),
                        "material_id": mat_id,
                        "material": mat_name,
                    }
                )

    # Pixel footprint (list-mode imaging); graceful when missing
    try:
        flat_pixels, u_idx, v_idx, pixels_available = _extract_cone_pixels(f, cone_index)
        nv, nu = _infer_image_shape(f)
        img = _reconstruct_image(nv, nu, u_idx, v_idx)
    except Exception:
        # Either grid or list-mode pixels are unavailable; fall back to an
        # empty image/pixel set but keep cone+hit data.
        flat_pixels = np.zeros(0, dtype=np.uint32)
        u_idx = np.zeros(0, dtype=np.int64)
        v_idx = np.zeros(0, dtype=np.int64)
        pixels_available = False
        img = np.zeros((0, 0), dtype=np.int32)

    return ConeTrace(
        cone_index=cone_index,
        event_index=event_index,
        is_gamma=is_gamma,
        cone_apex_cm=apex_cm,
        cone_axis_dir=axis_dir,
        cone_theta_deg=theta_deg,
        cone_incident_energy_MeV=incident_E,
        cone_species_code=species_code,
        cone_recoil_code=recoil_code,
        gamma_hit_order=gamma_hit_order_row,
        hits=hits,
        pixels_available=pixels_available,
        flat_pixel_indices=flat_pixels,
        u_idx=u_idx,
        v_idx=v_idx,
        image_uv=img,
    )


def _resolve_cone_index_from_args(f: h5py.File, args: argparse.Namespace) -> int:
    """
    Convert one of --cone-index / --event-index / --imaged-cone-index into a cone index.
    """
    specified = [
        args.cone_index is not None,
        args.event_index is not None,
        args.imaged_cone_index is not None,
    ]
    if sum(specified) != 1:
        raise SystemExit(
            "Exactly one of --cone-index, --event-index, or --imaged-cone-index "
            "must be provided."
        )

    if args.cone_index is not None:
        return int(args.cone_index)

    lm_grp = f.get("/lm")
    if not isinstance(lm_grp, h5py.Group):
        raise RuntimeError("This file has no /lm group; list-mode data not available.")

    if args.event_index is not None:
        # Map event index → cone index via /lm/event_cone_id
        ev_idx = int(args.event_index)
        ds = lm_grp.get("event_cone_id")
        if ds is None:
            raise RuntimeError(
                "Missing /lm/event_cone_id dataset; cannot map event index to cone."
            )
        if ev_idx < 0 or ev_idx >= ds.shape[0]:
            raise IndexError(f"event_index {ev_idx} is out of range [0, {ds.shape[0]}).")
        cone_index = int(ds[ev_idx])
        if cone_index < 0:
            raise RuntimeError(
                f"Event {ev_idx} did not produce a cone (event_cone_id = {cone_index})."
            )
        return cone_index

    # args.imaged_cone_index is not None
    imaged_idx = int(args.imaged_cone_index)
    # In ngimager, "imaged cone index" uses the same numbering as /cones/*
    # but appears in /lm/event_imaged_cone_id. We accept it directly as a cone
    # index, and optionally sanity-check that it appears in the table if present.
    ds_imaged = lm_grp.get("event_imaged_cone_id")
    if ds_imaged is not None:
        found = bool(np.any(np.asarray(ds_imaged[...], dtype=np.int32) == imaged_idx))
        if not found:
            raise RuntimeError(
                f"imaged cone index {imaged_idx} does not appear in /lm/event_imaged_cone_id."
            )
    return imaged_idx


def _summarize_trace(trace: ConeTrace, path: Path) -> None:
    """
    Print a human-readable summary of a ConeTrace.
    """
    print(f"\n=== Cone trace summary ===")
    print(f"File          : {path}")
    print(
        f"Cone index    : {trace.cone_index}\n"
        f"Event index   : {trace.event_index} "
        f"({'gamma' if trace.is_gamma else 'neutron'})"
    )

    print("\nCone geometry:")
    print(f"  apex_cm               = {trace.cone_apex_cm}")
    print(f"  axis_dir              = {trace.cone_axis_dir}")
    print(f"  theta_deg             = {trace.cone_theta_deg:.3f}")
    print(f"  incident_energy_MeV   = {trace.cone_incident_energy_MeV:.3f}")
    print(f"  species_code          = {trace.cone_species_code} (0=neutron, 1=gamma)")
    print(f"  recoil_code           = {trace.cone_recoil_code} (0=NA/gamma, 1=proton, 2=carbon)")

    if trace.gamma_hit_order is not None:
        order = trace.gamma_hit_order
        print("\nGamma hit sequencing (indices into /lm/hit_* for this event):")
        print(f"  h1 <- hit index {order[0]}")
        print(f"  h2 <- hit index {order[1]}")
        print(f"  h3 <- hit index {order[2]}")
    else:
        print("\nGamma hit sequencing: none stored for this cone (likely a neutron cone).")

    print("\nHits contributing to this cone (as stored in /lm/hit_*):")
    if not trace.hits:
        print("  (no hits available or /lm/hit_* datasets missing)")
    else:
        for h in trace.hits:
            print(
                f"  hit[{h['hit_index']}] det={h['det_id']}, "
                f"material={h['material']} "
                f"r_cm={np.asarray(h['r_cm'])}, "
                f"t_ns={h['t_ns']:.3f}, L_mevee={h['L_mevee']:.3f}"
            )

    n_pix = int(trace.flat_pixel_indices.size)
    n_pix_unique = int(np.unique(trace.flat_pixel_indices).size) if n_pix > 0 else 0
    print("\nPixel footprint on imaging plane:")
    if not trace.pixels_available:
        print("  list-mode pixel data not available (no /lm/cone_pixel_indices).")
    else:
        print(f"  entries in /lm/cone_pixel_indices for this cone : {n_pix}")
        print(f"  unique pixels touched                             : {n_pix_unique}")
        if n_pix > 0:
            print("  first few (u, v) indices:")
            for u, v in list(zip(trace.u_idx, trace.v_idx))[:10]:
                print(f"    (u={u}, v={v})")
        else:
            print("  this cone did not intersect the imaging plane (no pixels).")


def _plot_trace(trace: ConeTrace) -> None:
    """
    Show a simple imshow of the per-cone footprint.
    """
    import matplotlib.pyplot as plt

    img = trace.image_uv
    if img.size == 0 or not np.any(img):
        print("\n[plot] No pixels to plot for this cone.")
        return

    fig, ax = plt.subplots()
    im = ax.imshow(img, origin="lower")
    ax.set_title(f"Cone {trace.cone_index} footprint")
    ax.set_xlabel("u index")
    ax.set_ylabel("v index")
    fig.colorbar(im, ax=ax, label="counts per pixel")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a single cone → pixels → hits chain in an ngimager HDF5 file.",
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to ngimager HDF5 output.",
    )
    parser.add_argument(
        "--cone-index",
        type=int,
        default=None,
        help="Cone index into /cones/* datasets.",
    )
    parser.add_argument(
        "--event-index",
        type=int,
        default=None,
        help="Event index into /lm/hit_* datasets (mapped via /lm/event_cone_id).",
    )
    parser.add_argument(
        "--imaged-cone-index",
        type=int,
        default=None,
        help="Cone index known to have been imaged (checked against /lm/event_imaged_cone_id if present).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also show a quick imshow of the cone footprint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path: Path = args.file

    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    with h5py.File(path, "r") as f:
        cone_index = _resolve_cone_index_from_args(f, args)
        trace = trace_cone_from_index(f, cone_index)

    _summarize_trace(trace, path)

    if args.plot:
        _plot_trace(trace)


if __name__ == "__main__":
    main()
