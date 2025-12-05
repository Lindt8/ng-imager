"""
ngimager.tools.hdf5_to_root
===========================

This module provides a standalone HDF5 → ROOT converter for ngimager output
files.  Although it is distributed as part of the ngimager package, the module
is intentionally self-contained and can be used independently — for example by
downloading *just this file* from GitHub without installing ngimager.

Dependencies
------------
Only two Python packages are required:

    - h5py     (for reading the ngimager HDF5 file)
    - uproot   (for writing the output ROOT file)

No part of ngimager's internal codebase is imported here; the converter makes
no assumptions beyond the documented HDF5 file structure.

Standalone Usage (Command Line)
-------------------------------
If you have Python along with `h5py` and `uproot` installed, you can run:

    python hdf5_to_root.py my_run.h5
    python hdf5_to_root.py my_run.h5 -o output.root
    python hdf5_to_root.py my_run.h5 --overwrite

The script will write `my_run.root` (or the specified output path) containing
ROOT TTrees for list-mode hits, cones, list-mode imaging pixel mappings,
summed SBP images, and file/run metadata.

Standalone Usage (Python API)
-----------------------------
You may also import and call the converter from a standalone script:

    from pathlib import Path
    from hdf5_to_root import convert_hdf5_to_root

    convert_hdf5_to_root(Path("my_run.h5"), Path("my_run.root"), overwrite=True)

Running Under ngimager (Installed Package)
------------------------------------------
When installed as part of ngimager, the `ng-hdf2root` console entry point is
registered automatically and can be invoked as:

    ng-hdf2root my_run.h5

The behavior is identical to running `main()` from this module.

Purpose
-------
The converter is designed for ROOT-centric analysis workflows.  It flattens the
HDF5 list-mode representation into ROOT-friendly TTrees that preserve all
linkages between events, hits, cones, and (when present) list-mode imaging
pixels.  This enables fast, flexible histogramming and correlation analysis in
ROOT with minimal dependence on the rest of ngimager.

"""


from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import argparse

import h5py
import numpy as np
import uproot


def _ensure_group(f: h5py.File, name: str) -> h5py.Group:
    try:
        return f[name]
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Required group '{name}' is missing in HDF5 file.") from exc


def _read_dataset_optional(
    grp: h5py.Group,
    name: str,
    shape_first: int,
    dtype,
    default_value=0,
) -> np.ndarray:
    """
    Read a 1D dataset of expected length `shape_first` if present; otherwise
    return a filled array of `default_value`.
    """
    if name in grp:
        arr = np.asarray(grp[name][:], dtype=dtype)
        if arr.shape[0] != shape_first:
            raise ValueError(
                f"Dataset '{grp.name}/{name}' has unexpected length "
                f"{arr.shape[0]} (expected {shape_first})."
            )
        return arr
    return np.full(shape_first, default_value, dtype=dtype)


def _build_lm_tree(f: h5py.File) -> Dict[str, np.ndarray]:
    """
    Build flattened per-hit arrays for the 'lm' TTree.

    Each row corresponds to a single physical hit belonging to a neutron
    or gamma event. Event-level quantities are repeated for each hit.
    """
    lm = _ensure_group(f, "lm")

    # Required event-level dataset
    event_type = np.asarray(lm["event_type"][:], dtype=np.uint8)
    n_events = int(event_type.shape[0])

    # Optional provenance datasets
    event_meta_run_id = _read_dataset_optional(
        lm, "event_meta_run_id", n_events, np.int32, 0
    )
    event_meta_file_ix = _read_dataset_optional(
        lm, "event_meta_file_ix", n_events, np.int32, 0
    )

    # Optional event→cone linkage datasets (from write_event_cone_survival)
    event_cone_id = _read_dataset_optional(
        lm, "event_cone_id", n_events, np.int32, -1
    )
    event_imaged_cone_id = _read_dataset_optional(
        lm, "event_imaged_cone_id", n_events, np.int32, -1
    )

    # Optional survival table (mapping style)
    # /lm/event_survival : [N_events, 3] columns:
    #   0: event_index
    #   1: first_cone_index
    #   2: first_imaged_cone_index
    if "event_survival" in lm:
        surv = np.asarray(lm["event_survival"][:], dtype=np.int32)
        if surv.shape[0] != n_events or surv.shape[1] != 3:
            raise ValueError(
                f"'/lm/event_survival' has shape {surv.shape}, expected (N_events, 3)."
            )
        ev_surv_event_index = surv[:, 0]
        ev_surv_first_cone = surv[:, 1]
        ev_surv_first_imaged_cone = surv[:, 2]
    else:
        ev_surv_event_index = np.full(n_events, -1, dtype=np.int32)
        ev_surv_first_cone = np.full(n_events, -1, dtype=np.int32)
        ev_surv_first_imaged_cone = np.full(n_events, -1, dtype=np.int32)

    # Hit slots: [N_events, 3, 3] for positions, etc.
    hit_pos = np.asarray(lm["hit_pos_cm"][:], dtype=np.float32)      # (N, 3, 3)
    hit_t = np.asarray(lm["hit_t_ns"][:], dtype=np.float32)          # (N, 3)
    hit_L = np.asarray(lm["hit_L_mevee"][:], dtype=np.float32)       # (N, 3)
    hit_det = np.asarray(lm["hit_det_id"][:], dtype=np.int32)        # (N, 3)
    hit_mat = np.asarray(lm["hit_material_id"][:], dtype=np.int16)   # (N, 3)

    if hit_pos.shape[0] != n_events:
        raise ValueError(
            f"'/lm/hit_pos_cm' has length {hit_pos.shape[0]}, "
            f"expected {n_events} along axis 0."
        )

    # Sentinel convention from docs:
    # - For neutron events, only slots 0–1 are real; slot 2 has NaNs / -1.
    # - For gamma events, slots 0–2 are real.
    # More generally, any slot with det_id < 0 or non-finite position is ignored.
    valid = (hit_det >= 0) & np.isfinite(hit_pos).all(axis=2)

    # Flatten [N_events, 3] → [N_events * 3]
    n_slots = hit_det.shape[1]
    event_index = np.repeat(np.arange(n_events, dtype=np.int32), n_slots)
    hit_index = np.tile(np.arange(n_slots, dtype=np.int8), n_events)

    valid_flat = valid.reshape(-1)
    event_index_flat = event_index[valid_flat]
    hit_index_flat = hit_index[valid_flat]

    pos_flat = hit_pos.reshape(-1, 3)[valid_flat]
    t_flat = hit_t.reshape(-1)[valid_flat]
    L_flat = hit_L.reshape(-1)[valid_flat]
    det_flat = hit_det.reshape(-1)[valid_flat]
    mat_flat = hit_mat.reshape(-1)[valid_flat]

    # Repeat event-level quantities per hit.
    event_type_flat = event_type[event_index_flat]
    event_meta_run_id_flat = event_meta_run_id[event_index_flat]
    event_meta_file_ix_flat = event_meta_file_ix[event_index_flat]
    event_cone_id_flat = event_cone_id[event_index_flat]
    event_imaged_cone_id_flat = event_imaged_cone_id[event_index_flat]
    ev_surv_event_index_flat = ev_surv_event_index[event_index_flat]
    ev_surv_first_cone_flat = ev_surv_first_cone[event_index_flat]
    ev_surv_first_imaged_cone_flat = ev_surv_first_imaged_cone[event_index_flat]

    arrays: Dict[str, Any] = {
        # Event context
        "event_index": event_index_flat.astype(np.int32),
        "event_type": event_type_flat.astype(np.uint8),
        "event_meta_run_id": event_meta_run_id_flat.astype(np.int32),
        "event_meta_file_ix": event_meta_file_ix_flat.astype(np.int32),
        "event_cone_id": event_cone_id_flat.astype(np.int32),
        "event_imaged_cone_id": event_imaged_cone_id_flat.astype(np.int32),
        "event_survival_event_index": ev_surv_event_index_flat.astype(np.int32),
        "event_survival_first_cone_index": ev_surv_first_cone_flat.astype(np.int32),
        "event_survival_first_imaged_cone_index": ev_surv_first_imaged_cone_flat.astype(
            np.int32
        ),
        # Hit information
        "hit_index": hit_index_flat.astype(np.int8),
        "hit_pos_x_cm": pos_flat[:, 0].astype(np.float32),
        "hit_pos_y_cm": pos_flat[:, 1].astype(np.float32),
        "hit_pos_z_cm": pos_flat[:, 2].astype(np.float32),
        "hit_t_ns": t_flat.astype(np.float32),
        "hit_L_mevee": L_flat.astype(np.float32),
        "hit_det_id": det_flat.astype(np.int32),
        "hit_material_id": mat_flat.astype(np.int16),
    }
    return arrays


def _build_cones_tree(f: h5py.File) -> Dict[str, np.ndarray] | None:
    """
    Build per-cone arrays for the 'cones' TTree.

    Returns None if the /cones group is absent.
    """
    if "cones" not in f:
        return None

    cones = f["cones"]
    cone_id = np.asarray(cones["cone_id"][:], dtype=np.int32)
    n_cones = int(cone_id.shape[0])

    apex = np.asarray(cones["apex_xyz_cm"][:], dtype=np.float32)
    axis = np.asarray(cones["axis_xyz"][:], dtype=np.float32)
    theta = np.asarray(cones["theta_rad"][:], dtype=np.float32)
    inc_E = np.asarray(cones["incident_energy_MeV"][:], dtype=np.float32)
    event_index = np.asarray(cones["event_index"][:], dtype=np.int32)

    species = np.asarray(cones["species"][:], dtype=np.uint8)
    recoil_code = np.asarray(cones["recoil_code"][:], dtype=np.uint8)

    if "gamma_hit_order" in cones:
        gamma_hit_order = np.asarray(cones["gamma_hit_order"][:], dtype=np.int8)
        if gamma_hit_order.ndim != 2 or gamma_hit_order.shape[1] != 3:
            raise ValueError(
                f"'/cones/gamma_hit_order' has unexpected shape "
                f"{gamma_hit_order.shape}."
            )
        if gamma_hit_order.shape[0] != n_cones:
            raise ValueError(
                "gamma_hit_order length must match number of cones: "
                f"{gamma_hit_order.shape[0]} vs {n_cones}"
            )
        gh0 = gamma_hit_order[:, 0]
        gh1 = gamma_hit_order[:, 1]
        gh2 = gamma_hit_order[:, 2]
    else:
        gh0 = np.full(n_cones, -1, dtype=np.int8)
        gh1 = np.full(n_cones, -1, dtype=np.int8)
        gh2 = np.full(n_cones, -1, dtype=np.int8)

    arrays: Dict[str, Any] = {
        "cone_id": cone_id.astype(np.int32),
        "event_index": event_index.astype(np.int32),
        "species": species.astype(np.uint8),
        "recoil_code": recoil_code.astype(np.uint8),
        # Kinematics: this is the key incident energy spectrum per cone.
        "incident_energy_MeV": inc_E.astype(np.float32),
        # Geometry
        "apex_x_cm": apex[:, 0].astype(np.float32),
        "apex_y_cm": apex[:, 1].astype(np.float32),
        "apex_z_cm": apex[:, 2].astype(np.float32),
        "axis_x": axis[:, 0].astype(np.float32),
        "axis_y": axis[:, 1].astype(np.float32),
        "axis_z": axis[:, 2].astype(np.float32),
        "theta_rad": theta.astype(np.float32),
        # Gamma hit ordering
        "gamma_hit_order_0": gh0.astype(np.int8),
        "gamma_hit_order_1": gh1.astype(np.int8),
        "gamma_hit_order_2": gh2.astype(np.int8),
    }
    return arrays


def _build_cone_pixels_tree(f: h5py.File) -> Dict[str, np.ndarray] | None:
    """
    Build per-pixel arrays for the cone→pixel mapping ('cone_pixels' TTree).

    Returns None if /lm/cone_pixel_indices is absent.
    """
    if "lm" not in f:
        return None
    lm = f["lm"]
    if "cone_pixel_indices" not in lm:
        return None

    cpi = np.asarray(lm["cone_pixel_indices"][:], dtype=np.uint32)
    if cpi.ndim != 2 or cpi.shape[1] != 2:
        raise ValueError(
            f"'/lm/cone_pixel_indices' has unexpected shape {cpi.shape}, "
            "expected (K, 2)."
        )

    cone_id = cpi[:, 0].astype(np.int32)
    flat = cpi[:, 1].astype(np.uint32)

    meta = _ensure_group(f, "meta")
    try:
        nu = int(meta.attrs["grid.nu"])
        nv = int(meta.attrs["grid.nv"])
        u_min = float(meta.attrs["grid.u_min"])
        v_min = float(meta.attrs["grid.v_min"])
        du = float(meta.attrs["grid.du"])
        dv = float(meta.attrs["grid.dv"])
    except KeyError as exc:
        raise RuntimeError(
            "Required imaging plane attributes missing under /meta."
        ) from exc

    u_index = (flat % nu).astype(np.int32)
    v_index = (flat // nu).astype(np.int32)

    u_cm = u_min + (u_index.astype(np.float64) + 0.5) * du
    v_cm = v_min + (v_index.astype(np.float64) + 0.5) * dv

    arrays: Dict[str, Any] = {
        "cone_id": cone_id,
        "flat_pixel_index": flat,
        "u_index": u_index,
        "v_index": v_index,
        "u_cm": u_cm.astype(np.float32),
        "v_cm": v_cm.astype(np.float32),
        "nu": np.full_like(u_index, nu, dtype=np.int32),
        "nv": np.full_like(v_index, nv, dtype=np.int32),
    }
    return arrays


def _build_images_summed_tree(f: h5py.File) -> Dict[str, np.ndarray] | None:
    """
    Build a compact representation of summed SBP images.

    The 'images_summed' TTree has one entry per species with branches:
      - species : string (e.g. "n", "g", "all")
      - nu, nv  : grid dimensions
      - counts  : flattened image counts (length = nu * nv)
    """
    if "images" not in f:
        return None
    images = f["images"]
    if "summed" not in images:
        return None
    summed = images["summed"]

    species_list = []
    counts_list = []

    for name in ("n", "g", "all"):
        if name in summed:
            img = np.asarray(summed[name][:], dtype=np.float32)
            counts_list.append(img.reshape(-1))
            species_list.append(name)

    if not species_list:
        return None

    # Assume all images share the same shape (nv, nu)
    nv, nu = summed[species_list[0]].shape
    n_species = len(species_list)

    species_arr = np.array(species_list, dtype=object)
    nu_arr = np.full(n_species, nu, dtype=np.int32)
    nv_arr = np.full(n_species, nv, dtype=np.int32)
    counts_arr = np.stack(counts_list, axis=0).astype(np.float32)

    arrays: Dict[str, Any] = {
        "species": species_arr,
        "nu": nu_arr,
        "nv": nv_arr,
        "counts": counts_arr,
    }
    return arrays


def _build_file_meta_tree(f: h5py.File) -> Dict[str, np.ndarray]:
    """
    Build a one-entry 'file_meta' TTree mirroring a few HDF5 root attributes.
    """
    fmt = str(f.attrs.get("format_version", ""))
    created = str(f.attrs.get("created_utc", ""))
    software = str(f.attrs.get("software", ""))
    run_command = str(f.attrs.get("run_command", ""))

    arrays: Dict[str, Any] = {
        "format_version": np.array([fmt], dtype=object),
        "created_utc": np.array([created], dtype=object),
        "software": np.array([software], dtype=object),
        "run_command": np.array([run_command], dtype=object),
    }
    return arrays


def _build_run_meta_tree(f: h5py.File) -> Dict[str, np.ndarray] | None:
    """
    Build a key/value TTree from /meta/run_meta, if present.

    Supports both layouts:
      - attributes on /meta/run_meta (older design)
      - string datasets under /meta/run_meta (current implementation)
    """
    if "meta" not in f:
        return None
    meta = f["meta"]
    if "run_meta" not in meta:
        return None
    run_meta = meta["run_meta"]

    keys: list[str] = []
    vals: list[str] = []

    # Prefer attributes if present (matches older docs).
    attr_keys = list(run_meta.attrs.keys())
    if attr_keys:
        keys.extend(attr_keys)
        vals.extend([str(run_meta.attrs[k]) for k in attr_keys])
    else:
        # Current implementation stores one dataset per key.
        for ds_name in run_meta.keys():
            ds = run_meta[ds_name]
            try:
                value = ds[()].decode("utf-8") if isinstance(ds[()], (bytes, bytearray)) else str(ds[()])
            except Exception:
                value = str(ds[()])
            keys.append(ds_name)
            vals.append(value)

    if not keys:
        return None

    arrays: Dict[str, Any] = {
        "key": np.array(keys, dtype=object),
        "value": np.array(vals, dtype=object),
    }
    return arrays


def convert_hdf5_to_root(
    hdf_path: Path,
    root_path: Path,
    overwrite: bool = False,
) -> None:
    """
    Convert a single ngimager HDF5 output file into a ROOT file using uproot.

    Parameters
    ----------
    hdf_path:
        Input HDF5 path produced by ngimager.
    root_path:
        Output ROOT path to create.
    overwrite:
        If False (default), refuse to overwrite an existing file.
    """
    if not hdf_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf_path}")

    if root_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing ROOT file: {root_path}"
        )

    with h5py.File(hdf_path, "r") as f, uproot.recreate(root_path) as root_file:
        # Per-hit/per-event list-mode tree
        lm_arrays = _build_lm_tree(f)
        root_file["lm"] = lm_arrays

        # Cones (including incident_energy_MeV)
        cones_arrays = _build_cones_tree(f)
        if cones_arrays is not None:
            root_file["cones"] = cones_arrays

        # Cone→pixel mappings (list-mode imaging)
        cone_pix_arrays = _build_cone_pixels_tree(f)
        if cone_pix_arrays is not None:
            root_file["cone_pixels"] = cone_pix_arrays

        # Summed images (optional)
        images_arrays = _build_images_summed_tree(f)
        if images_arrays is not None:
            root_file["images_summed"] = images_arrays

        # File-level metadata
        meta_arrays = _build_file_meta_tree(f)
        root_file["file_meta"] = meta_arrays

        # Free-form run metadata (optional)
        run_meta_arrays = _build_run_meta_tree(f)
        if run_meta_arrays is not None:
            root_file["run_meta"] = run_meta_arrays


def _default_output_path(hdf_path: Path) -> Path:
    if hdf_path.suffix.lower() == ".h5":
        return hdf_path.with_suffix(".root")
    if hdf_path.suffix.lower() == ".hdf5":
        return hdf_path.with_suffix(".root")
    return hdf_path.with_suffix(hdf_path.suffix + ".root")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an ngimager HDF5 output file into a ROOT file suitable for "
            "histogramming and further analysis."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input HDF5 file produced by ngimager.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output ROOT file path (defaults to INPUT with .root extension).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing ROOT file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    hdf_path: Path = args.input
    root_path: Path = args.output or _default_output_path(hdf_path)

    try:
        convert_hdf5_to_root(hdf_path, root_path, overwrite=args.overwrite)
    except Exception as exc:
        raise SystemExit(f"[ng-hdf2root] ERROR: {exc}") from exc

    print(f"[ng-hdf2root] Wrote ROOT file: {root_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
