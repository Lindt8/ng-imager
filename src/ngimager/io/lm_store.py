from __future__ import annotations
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import h5py
import numpy as np
import re
from datetime import datetime, timezone
from pathlib import Path
from ngimager.config.schemas import Config
from ngimager.config.load import snapshot_config_toml
from ngimager.geometry.plane import Plane
from ngimager.physics.events import NeutronEvent, GammaEvent

# Optional helpers for discovering installed package name/version
try:  # Python 3.8+
    from importlib import metadata as importlib_metadata
except Exception:  # pragma: no cover - older Python
    importlib_metadata = None

try:  # Python 3.11+
    import tomllib as _tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - allow older Python or missing module
    try:
        import tomli as _tomllib  # type: ignore[import-not-found]
    except Exception:
        _tomllib = None

FORMAT_VERSION = "1.0"


def _detect_software_tag() -> str:
    """
    Best-effort detection of the software tag to store in the HDF5 root.

    Preference order:
      1. Installed distribution metadata for the 'ngimager' package.
      2. [project] name/version from a nearby pyproject.toml.
      3. Fallback string if neither is available.

    This is intentionally conservative: any failure just yields a generic tag
    rather than raising.
    """
    default = "ng-imager (version unknown)"
    pkg_name = "ngimager"

    # 1) Try installed package metadata
    if importlib_metadata is not None:
        try:
            ver = importlib_metadata.version(pkg_name)
            try:
                meta = importlib_metadata.metadata(pkg_name)
                name = meta.get("Name") or pkg_name
            except Exception:
                name = pkg_name
            return f"{name} {ver}"
        except Exception:
            pass

    # 2) Try reading pyproject.toml near this file
    if _tomllib is not None:
        try:
            here = Path(__file__).resolve()
            for parent in (here,) + tuple(here.parents):
                candidate = parent / "pyproject.toml"
                if candidate.is_file():
                    with candidate.open("rb") as fh:
                        data = _tomllib.load(fh)
                    project = data.get("project") or {}
                    name = project.get("name", pkg_name)
                    ver = project.get("version")
                    if ver:
                        return f"{name} {ver}"
                    return str(name)
        except Exception:
            pass

    return default

SOFTWARE_ATTR = _detect_software_tag()

def write_init(path: str, cfg_path: str, cfg: Config, plane: Plane) -> h5py.File:
    f = h5py.File(path, "w")
    # Root attrs
    f.attrs["format_version"] = FORMAT_VERSION
    f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
    f.attrs["software"] = SOFTWARE_ATTR
    f.attrs["config_text"] = snapshot_config_toml(cfg_path)
    f.attrs["readme"] = (
        "HDF5 output produced by ng-imager. "
        "See the ng-imager documentation on GitHub for full layout details."
    )
    f.attrs["docs_url"] = "https://github.com/Lindt8/ng-imager"  # adjust if needed

    # /meta
    meta = f.create_group("meta")
    meta.attrs["plane.P0"] = plane.P0
    meta.attrs["plane.n"] = plane.n
    meta.attrs["plane.eu"] = plane.eu
    meta.attrs["plane.ev"] = plane.ev
    meta.attrs["grid.u_min"] = plane.u_min
    meta.attrs["grid.u_max"] = plane.u_max
    meta.attrs["grid.du"] = plane.du
    meta.attrs["grid.v_min"] = plane.v_min
    meta.attrs["grid.v_max"] = plane.v_max
    meta.attrs["grid.dv"] = plane.dv
    meta.attrs["grid.nu"] = plane.nu
    meta.attrs["grid.nv"] = plane.nv

    # Run configuration flags
    # (we store these as simple attrs for quick inspection)
    meta.attrs["run_fast"] = bool(getattr(cfg.run, "fast", False))
    meta.attrs["run_list"] = bool(getattr(cfg.run, "list", False))
    meta.attrs["run_neutron"] = bool(getattr(cfg.run, "neutrons", True))
    meta.attrs["run_gamma"] = bool(getattr(cfg.run, "gammas", True))
    meta.attrs["run_stop_stage"] = getattr(cfg.run, "stop_stage", "")

    # Optional human-readable run label for visualization.
    run_plot_label = getattr(cfg.run, "plot_label", None)
    if run_plot_label:
        meta.attrs["run_plot_label"] = str(run_plot_label)

    # Optional free-form run-level metadata table from [run.meta].
    # Stored under /meta/run_meta as one string dataset per key.
    run_meta = getattr(cfg.run, "meta", {}) or {}
    if run_meta:
        run_meta_grp = meta.require_group("run_meta")
        str_dt = h5py.string_dtype(encoding="utf-8")
        for key, value in run_meta.items():
            ds_name = str(key).replace("/", "_")
            if ds_name in run_meta_grp:
                del run_meta_grp[ds_name]
            run_meta_grp.create_dataset(
                ds_name,
                data=np.array(str(value), dtype=str_dt),
            )
    
    # Human-readable overview of this HDF5 layout
    readme_text = """
    ng-imager HDF5 layout (format_version = {fmt}):

    Top-level:
      /meta
          - imaging plane geometry and basic run flags as attributes
          - README  (this text)
      /images/summed
          - 'n', 'g', or 'all' 2D images (float32)
      /cones
          - cone_id, apex_xyz_cm, axis_xyz, theta_rad
          - species (0=neutron, 1=gamma), species_labels
          - recoil_code (0=NA, 1=proton, 2=carbon), recoil_code_labels
          - incident_energy_MeV (En for n, Eg for g)
          - event_index (row index into /lm/event_* arrays)
          - gamma_hit_order (for gammas: indices into /lm/hit_* in [first, second, third] order)
      /lm
          - event_type (0=neutron, 1=gamma), event_type_labels
          - hit_pos_cm, hit_t_ns, hit_L_mevee, hit_det_id, hit_material_id
          - material_id_labels (mapping indices → material strings)
          - cone_pixel_indices (rows of [cone_id, flat_pixel_index])
          - event_cone_id, event_imaged_cone_id (per-event survival table)
      /counters
          - scalar integer diagnostics for each pipeline stage

    The original TOML config used for this run is stored as a single string
    in the root attribute 'config_text'. For full documentation, see the
    ng-imager repository and docs at: https://github.com/Lindt8/ng-imager
    """.format(fmt=FORMAT_VERSION).strip()

    if "README" in meta:
        del meta["README"]
    meta.create_dataset(
        "README",
        data=np.array(readme_text, dtype=h5py.string_dtype()),
    )

    # ------------------------------------------------------------------
    # Optional extra text metadata files from config
    # ------------------------------------------------------------------
    extra_files = getattr(getattr(cfg, "io", None), "extra_text_files", {}) or {}
    if extra_files:
        extra_grp = meta.require_group("extra_text")

        # For traceability, store a simple label→path mapping as an attribute.
        # We keep it human-readable, but the real payload is in the datasets.
        try:
            mapping_strings = [
                f"{label}::{path}" for label, path in extra_files.items()
            ]
            extra_grp.attrs["source_paths"] = np.array(
                mapping_strings,
                dtype=h5py.string_dtype(),
            )
        except Exception:
            # If anything goes wrong with the attribute, just skip it;
            # the individual datasets are more important.
            pass

        # Resolve relative paths against the TOML config directory.
        cfg_dir = Path(cfg_path).resolve().parent

        for label, path_str in extra_files.items():
            # Be conservative about dataset names: forbid slashes.
            ds_name = str(label).replace("/", "_")

            try:
                p = Path(path_str)
                if not p.is_absolute():
                    p = (cfg_dir / p).resolve()
                with p.open("r", encoding="utf-8") as fh:
                    txt = fh.read()
                data = np.array(txt, dtype=h5py.string_dtype())
            except Exception as exc:
                # On failure, still create a small diagnostic dataset instead
                # of aborting the whole run.
                msg = f"[ngimager] Failed to read file {path_str!r}: {exc}"
                data = np.array(msg, dtype=h5py.string_dtype())

            if ds_name in extra_grp:
                del extra_grp[ds_name]
            extra_grp.create_dataset(ds_name, data=data)

    return f


def _counter_stage(key: str) -> int:
    """
    Heuristic mapping from counter key to pipeline stage:

      1: raw events → hits
      2: hits → shaped/typed events (event-level filters)
      3: events → cones (cone-level filters)
      4: cones → images

    Keys that do not match any known pattern are assigned stage 0.
    """
    if key.startswith("hits_") or key.startswith("raw_events_"):
        return 1

    if key.startswith("shaped_") or key.startswith("events_typed_"):
        return 2
    if key.startswith("events_total_for_filters") or key.startswith("events_after_filters"):
        return 2
    if key.startswith("events_rejected_"):
        return 2

    if key.startswith("events_cone_") or key.startswith("cones_"):
        return 3

    if key.startswith("images_") or key.startswith("image_"):
        return 4

    return 0


def write_counters(f: h5py.File, counters: Dict[str, int]) -> None:
    """
    Store scalar counters under /meta/counters as attributes.

    Each key in `counters` becomes an attribute on the /meta/counters group,
    prefixed with a stage number:

        S1_... → Stage 1 (raw events → hits)
        S2_... → Stage 2 (hits → shaped/typed → event filters)
        S3_... → Stage 3 (events → cones → cone filters)
        S4_... → Stage 4 (cones → images)

    This forces a "chronological" ordering when viewed in tools like HDFView
    (which sort attributes alphabetically).
    """
    meta = f.require_group("meta")
    if "counters" in meta:
        del meta["counters"]
    grp = meta.create_group("counters")

    # Sort by (stage, original key) so that attributes appear grouped by stage,
    # then alphabetically within each stage.
    for key in sorted(counters.keys(), key=lambda k: (_counter_stage(k), k)):
        stage = _counter_stage(key)
        if stage > 0:
            out_key = f"S{stage}_{key}"
        else:
            out_key = key
        value = counters[key]
        try:
            grp.attrs[out_key] = int(value)
        except Exception:
            grp.attrs[out_key] = str(value)





def _ensure_summed_group(f: h5py.File):
    return f.require_group("images").require_group("summed")


def write_summed(
    f: h5py.File,
    species: str,
    img: np.ndarray,
) -> None:
    """
    Write summed image for a given species.

    Parameters
    ----------
    f : open h5py.File
    species : "n" | "g" | "all" (string key)
    img : 2D numpy array (nv, nu), float or int
    """
    grp = _ensure_summed_group(f)
    dset_name = species
    if dset_name in grp:
        del grp[dset_name]
    grp.create_dataset(dset_name, data=img.astype(np.float32), compression="gzip")

def write_projections(
    f: h5py.File,
    species: str,
    img: np.ndarray,
    roi: tuple[float, float, float, float] | None = None,
) -> None:
    """
    Write 1D u/v projections for a given summed image.

    Layout (all under /images/summed/projections/{species}):

        u      : [nu] float32, sum over v (rows)
        v      : [nv] float32, sum over u (cols)
        u_roi  : [nu] float32, ROI-limited u projection (optional)
        v_roi  : [nv] float32, ROI-limited v projection (optional)

    Parameters
    ----------
    f : open h5py.File
        HDF5 file handle (already open in read/write mode).
    species : str
        "n", "g", or "all".
    img : ndarray
        2D image array with shape (nv, nu).
    roi : tuple or None
        Optional ROI in plane coordinates (cm):

            (u_min_cm, u_max_cm, v_min_cm, v_max_cm)

        If provided, ROI-limited projections are computed by summing only the
        pixels whose centers fall inside this rectangle, and bins outside the
        ROI range are set to zero.
    """
    summed_grp = _ensure_summed_group(f)
    proj_root = summed_grp.require_group("projections")
    grp = proj_root.require_group(species)

    img = np.asarray(img, dtype=np.float32)
    nv, nu = img.shape

    # --- Global projections -------------------------------------------------
    proj_u = img.sum(axis=0, dtype=np.float64)  # sum over v (rows)
    proj_v = img.sum(axis=1, dtype=np.float64)  # sum over u (cols)

    for name, data in (("u", proj_u), ("v", proj_v)):
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=data.astype(np.float32), compression="gzip")

    # Clear any stale ROI datasets if we are not writing ROI this time.
    if roi is None:
        for name in ("u_roi", "v_roi"):
            if name in grp:
                del grp[name]
        for attr in ("roi_u_min_cm", "roi_u_max_cm", "roi_v_min_cm", "roi_v_max_cm"):
            if attr in grp.attrs:
                del grp.attrs[attr]
        return

    # --- ROI-limited projections -------------------------------------------
    meta = f["meta"].attrs
    u_min_cm = float(meta["grid.u_min"])
    du_cm = float(meta["grid.du"])
    v_min_cm = float(meta["grid.v_min"])
    dv_cm = float(meta["grid.dv"])

    roi_u_min_cm, roi_u_max_cm, roi_v_min_cm, roi_v_max_cm = roi

    # Pixel-center positions in cm
    u_centers = u_min_cm + (np.arange(nu) + 0.5) * du_cm
    v_centers = v_min_cm + (np.arange(nv) + 0.5) * dv_cm

    u_mask = (u_centers >= roi_u_min_cm) & (u_centers <= roi_u_max_cm)
    v_mask = (v_centers >= roi_v_min_cm) & (v_centers <= roi_v_max_cm)

    # Full-length projections, zeros outside ROI
    proj_u_roi = np.zeros_like(proj_u)
    proj_v_roi = np.zeros_like(proj_v)

    if np.any(u_mask) and np.any(v_mask):
        block = img[np.ix_(v_mask, u_mask)]
        proj_u_roi[u_mask] = block.sum(axis=0, dtype=np.float64)
        proj_v_roi[v_mask] = block.sum(axis=1, dtype=np.float64)

    for name, data in (("u_roi", proj_u_roi), ("v_roi", proj_v_roi)):
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=data.astype(np.float32), compression="gzip")

    grp.attrs["roi_u_min_cm"] = float(roi_u_min_cm)
    grp.attrs["roi_u_max_cm"] = float(roi_u_max_cm)
    grp.attrs["roi_v_min_cm"] = float(roi_v_min_cm)
    grp.attrs["roi_v_max_cm"] = float(roi_v_max_cm)



def write_cones(
    f: h5py.File,
    cone_ids: np.ndarray,
    apex_xyz_cm: np.ndarray,
    axis_xyz: np.ndarray,
    theta_rad: np.ndarray,
    species: np.ndarray,
    recoil_code: np.ndarray,
    incident_energy_MeV: np.ndarray,
    event_index: np.ndarray,
    gamma_hit_order: np.ndarray | None = None,
) -> None:
    """
    Store per-cone geometric and classification parameters under /cones.

    Layout:
      /cones/cone_id             : [N]   uint32 
      /cones/apex_xyz_cm         : [N,3] float32
      /cones/axis_xyz            : [N,3] float32
      /cones/theta_rad           : [N]   float32
      /cones/species             : [N]   uint8  (0=neutron, 1=gamma)
      /cones/recoil_code         : [N]   uint8  (0=NA, 1=proton, 2=carbon)
      /cones/incident_energy_MeV : [N]   float32 (En for n, Eg for g)
      /cones/event_index         : [N]   int32 (row index into /lm/event_* arrays)
      /cones/gamma_hit_order     : [N,3] int8  (optional; see below)

      /cones/species_labels      : ["0=neutron", "1=gamma"]
      /cones/recoil_code_labels  : ["0=NA", "1=proton", "2=carbon"]

    Notes
    -----
    * For gamma cones (species == 1), gamma_hit_order[i] = (i0, i1, i2) gives
      the indices into /lm/hit_*[event_index[i], :, :] that correspond to
      (first scatter, second scatter, third point) used to build that cone.
    * For neutron cones (species == 0), gamma_hit_order[i] is (-1, -1, -1)
      and should be ignored.
    """
    grp = f.require_group("cones")
    for name in (
            "cone_id",
            "apex_xyz_cm",
            "axis_xyz",
            "theta_rad",
            "species",
            "recoil_code",
            "incident_energy_MeV",
            "event_index",
            "gamma_hit_order",
            "species_labels",
            "recoil_code_labels",
    ):
        if name in grp:
            del grp[name]

    cone_ids = cone_ids.astype(np.uint32)
    apex_xyz_cm = apex_xyz_cm.astype(np.float32)
    axis_xyz = axis_xyz.astype(np.float32)
    theta_rad = theta_rad.astype(np.float32)

    grp.create_dataset(
        "cone_id",
        data=cone_ids,
        compression="gzip",
    )
    grp.create_dataset(
        "apex_xyz_cm",
        data=apex_xyz_cm,
        compression="gzip",
    )
    grp.create_dataset(
        "axis_xyz",
        data=axis_xyz,
        compression="gzip",
    )
    grp.create_dataset(
        "theta_rad",
        data=theta_rad,
        compression="gzip",
    )

    # Species: 0 = neutron, 1 = gamma.
    if species is None:
        species_arr = np.zeros_like(cone_ids, dtype=np.uint8)
    else:
        species_arr = np.asarray(species, dtype=np.uint8)
    d_species = grp.create_dataset(
        "species",
        data=species_arr,
        compression="gzip",
    )
    d_species.attrs["legend"] = np.array(
        ["0=neutron", "1=gamma"],
        dtype=h5py.string_dtype(),
    )

    # Recoil code: 0 = unknown / N/A, 1 = proton, 2 = carbon.
    if recoil_code is None:
        recoil_arr = np.zeros_like(cone_ids, dtype=np.uint8)
    else:
        recoil_arr = np.asarray(recoil_code, dtype=np.uint8)
    d_recoil = grp.create_dataset(
        "recoil_code",
        data=recoil_arr,
        compression="gzip",
    )
    d_recoil.attrs["legend"] = np.array(
        ["0=unknown_or_gamma", "1=proton", "2=carbon"],
        dtype=h5py.string_dtype(),
    )

    # Visible legends as datasets (similar to /lm/materials/labels)
    species_labels = np.array(
        ["0=neutron", "1=gamma"],
        dtype=h5py.string_dtype(),
    )
    recoil_labels = np.array(
        ["0=NA/gamma/unknown", "1=proton", "2=carbon"],
        dtype=h5py.string_dtype(),
    )

    grp.create_dataset("species_labels", data=species_labels)
    grp.create_dataset("recoil_code_labels", data=recoil_labels)

    grp.create_dataset(
        "incident_energy_MeV",
        data=incident_energy_MeV.astype(np.float32),
        compression="gzip",
    )

    grp.create_dataset(
        "event_index",
        data=event_index.astype(np.int32),
        compression="gzip",
    )

    if gamma_hit_order is not None:
        gamma_hit_order = np.asarray(gamma_hit_order, dtype=np.int8)
        if gamma_hit_order.ndim != 2 or gamma_hit_order.shape[1] != 3:
            raise ValueError(
                "gamma_hit_order must have shape (N_cones, 3). "
                f"Got {gamma_hit_order.shape!r}."
            )
        if gamma_hit_order.shape[0] != cone_ids.shape[0]:
            raise ValueError(
                "gamma_hit_order length must match number of cones: "
                f"{gamma_hit_order.shape[0]} vs {cone_ids.shape[0]}"
            )
        grp.create_dataset(
            "gamma_hit_order",
            data=gamma_hit_order,
            compression="gzip",
        )





def write_lm_indices(
    f: h5py.File,
    lm_cone_pixel_lists: list[tuple[int, np.ndarray]],
) -> None:
    """
    Store list-mode indices mapping cones -> (u,v) pixels.

    We store:
      /lm/cone_pixel_indices : ragged array of (cone_id, flat_index) pairs

    where:
      - cone_id is the index into /cones/cone_id
      - flat_index is the flattened pixel index (row-major) on the imaging plane.
    """
    grp = f.require_group("lm")

    # Flatten all LM lists with cone_id
    all_rows: list[np.ndarray] = []

    for cone_id, arr in lm_cone_pixel_lists:
        if arr is None:
            continue
        flat = np.asarray(arr, dtype=np.uint32).ravel()
        if flat.size == 0:
            continue
        cone_ids = np.full_like(flat, int(cone_id), dtype=np.uint32)
        stacked = np.vstack([cone_ids, flat]).T  # (M,2)
        all_rows.append(stacked)

    if all_rows:
        all_rows_arr = np.concatenate(all_rows, axis=0)
    else:
        all_rows_arr = np.zeros((0, 2), dtype=np.uint32)

    # Single, clearly-named dataset for cone→pixel mapping
    if "cone_pixel_indices" in grp:
        del grp["cone_pixel_indices"]
    grp.create_dataset("cone_pixel_indices", data=all_rows_arr, compression="gzip")

    # Alias for convenience under /images/list_mode; this is a standard HDF5
    # soft link, so it does not duplicate data on disk.
    images_grp = f.require_group("images")
    list_mode_grp = images_grp.require_group("list_mode")
    if "cone_pixel_indices" in list_mode_grp:
        del list_mode_grp["cone_pixel_indices"]
    list_mode_grp["cone_pixel_indices"] = h5py.SoftLink("/lm/cone_pixel_indices")
    
    # The old /lm/indices and /lm/events datasets are intentionally no longer
    # written to avoid confusion about their semantics.
    if "indices" in grp:
        del grp["indices"]
    if "events" in grp:
        del grp["events"]


def _flatten_hits_for_ragged(phits_events: Sequence[Dict[str, Any]]
                             ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert variable-length PHITS-style events (with 'hits' list) into ragged columns.

    Returns:
      event_ptr: (N_events+1,) int64 — CSR-style pointers into the flat hit arrays.
      cols: dict of 1D arrays (len M = total hits):
            'x_cm','y_cm','z_cm','t_ns','Edep_MeV','reg' (dtypes float32/float64 and int32)
      Also returns event-level arrays in cols['events/…'] for convenience:
            'event_type' (uint8: 0=unknown,1=n,2=g,3=mixed), 'iomp','batch','history','no','name' (int64)
    """
    n_events = len(phits_events)
    ptr = np.zeros(n_events + 1, dtype=np.int64)
    # First pass: count hits per event
    k = 0
    for i, ev in enumerate(phits_events):
        nh = len(ev.get("hits", []))
        k += nh
        ptr[i + 1] = k

    M = int(k)
    x = np.empty(M, dtype=np.float32)
    y = np.empty(M, dtype=np.float32)
    z = np.empty(M, dtype=np.float32)
    t = np.empty(M, dtype=np.float32)
    e = np.empty(M, dtype=np.float32)
    reg = np.empty(M, dtype=np.int32)

    # Event-level metadata (fill with zeros/defaults if missing)
    ev_type_map = {"n": 1, "g": 2, "mixed": 3}
    etype = np.zeros(n_events, dtype=np.uint8)
    iomp  = np.zeros(n_events, dtype=np.int64)
    batch = np.zeros(n_events, dtype=np.int64)
    hist  = np.zeros(n_events, dtype=np.int64)
    eno   = np.zeros(n_events, dtype=np.int64)
    name  = np.zeros(n_events, dtype=np.int64)

    # Second pass: fill flat hits and per-event meta
    w = 0
    for i, ev in enumerate(phits_events):
        etype[i] = ev_type_map.get(ev.get("event_type", ""), 0)
        iomp[i]  = int(ev.get("iomp", 0))
        batch[i] = int(ev.get("batch", 0))
        hist[i]  = int(ev.get("history", 0))
        eno[i]   = int(ev.get("no", 0))
        name[i]  = int(ev.get("name", 0))
        hits = ev.get("hits", [])
        for h in hits:
            if hasattr(h, "r"):  # Hit object
                # r is cm; L is light-like; Edep_MeV may be in extras
                x[w], y[w], z[w] = float(h.r[0]), float(h.r[1]), float(h.r[2])
                t[w] = float(getattr(h, "t_ns"))
                # Prefer Edep_MeV if available in extras; else fall back to L
                e[w] = float(getattr(h, "extras", {}).get("Edep_MeV", getattr(h, "L", 0.0)))
                reg[w] = int(getattr(h, "det_id", 0))
            else:  # dict-like
                x[w]   = float(h.get("x_cm", 0.0))
                y[w]   = float(h.get("y_cm", 0.0))
                z[w]   = float(h.get("z_cm", 0.0))
                t[w]   = float(h.get("t_ns", 0.0))
                e[w]   = float(h.get("Edep_MeV", 0.0))
                reg[w] = int(h.get("reg", h.get("det_id", 0)))
            w += 1

    cols = {
        "x_cm": x, "y_cm": y, "z_cm": z, "t_ns": t, "Edep_MeV": e, "reg": reg,
        "events/event_type": etype,
        "events/iomp": iomp, "events/batch": batch, "events/history": hist,
        "events/no": eno, "events/name": name,
    }
    return ptr, cols

def write_lm_ragged(h5: h5py.File, phits_events: Sequence[Dict[str, Any]], *, group: str = "/lm") -> None:
    """
    Write variable-length list-mode (ragged) datasets for events with arbitrary hit multiplicity.
    This is ADDITIVE and does not modify existing fixed-shape datasets you already write elsewhere.
    """
    if group.endswith("/"):
        group = group[:-1]
    g_hits = h5.require_group(f"{group}/hits")
    g_ev   = h5.require_group(f"{group}/events")

    event_ptr, cols = _flatten_hits_for_ragged(phits_events)

    # Event pointer (CSR)
    if "event_ptr" in g_hits:
        del g_hits["event_ptr"]
    g_hits.create_dataset("event_ptr", data=event_ptr, dtype="i8")

    # Flat hit columns
    for key in ("x_cm", "y_cm", "z_cm", "t_ns", "Edep_MeV", "reg"):
        if key in g_hits:
            del g_hits[key]
        g_hits.create_dataset(key, data=cols[key])

    # Event-level arrays
    for key in ("event_type", "iomp", "batch", "history", "no", "name"):
        arr = cols[f"events/{key}"]
        if key in g_ev:
            del g_ev[key]
        g_ev.create_dataset(key, data=arr)


# store per-event / per-hit physics data for list-mode
def write_events_hits(
    f: h5py.File,
    events: list[NeutronEvent | GammaEvent],
) -> None:
    """
    Store per-event and per-hit data for list-mode analysis.

    Layout (all under /lm):

      /lm/materials/labels    : [M]  array of material strings
      /lm/event_type          : [N]  uint8, 0=n, 1=g
      /lm/event_meta_run_id   : [N]  int32 (optional meta)
      /lm/event_meta_file_ix  : [N]  int32 (optional meta)
      /lm/hit_pos_cm          : [N,3,3] float32 (event, hit_index, xyz)
      /lm/hit_t_ns            : [N,3]   float32
      /lm/hit_L_mevee         : [N,3]   float32
      /lm/hit_det_id          : [N,3]   int32
      /lm/hit_material_id     : [N,3]   int16

    Convention:
      - Neutron events use hits [0,1] and leave slot 2 as NaN/-1.
      - Gamma events use hits [0,1,2].
    """
    if not events:
        return

    N = len(events)

    # Helper: always return a list[Hit] in time order for any supported event
    def _ordered_hits(ev: NeutronEvent | GammaEvent):
        ev_ord = ev.ordered()
        if isinstance(ev_ord, NeutronEvent):
            return [ev_ord.h1, ev_ord.h2]
        elif isinstance(ev_ord, GammaEvent):
            return [ev_ord.h1, ev_ord.h2, ev_ord.h3]
        else:
            raise TypeError(f"Unsupported event type in write_events_hits: {type(ev_ord)!r}")

    # Gather materials to build a small vocabulary
    material_labels: set[str] = set()
    for ev in events:
        for h in _ordered_hits(ev):
            # Hit.material is a required field in our current design; we still
            # defensively allow None just in case.
            mat = getattr(h, "material", None)
            if mat is not None:
                material_labels.add(mat)

    material_list = sorted(material_labels)
    material_to_id = {m: i for i, m in enumerate(material_list)}

    def mat_id(mat: str | None) -> int:
        if mat is None:
            return -1
        return material_to_id.get(mat, -1)

    # Allocate arrays
    hit_pos = np.full((N, 3, 3), np.nan, dtype=np.float32)
    hit_t = np.full((N, 3), np.nan, dtype=np.float32)
    hit_L = np.full((N, 3), np.nan, dtype=np.float32)
    hit_det = np.full((N, 3), -1, dtype=np.int32)
    hit_mat = np.full((N, 3), -1, dtype=np.int16)
    ev_type = np.zeros(N, dtype=np.uint8)  # 0=n,1=g

    # very light meta placeholders
    ev_run = np.full(N, -1, dtype=np.int32)
    ev_file_ix = np.full(N, -1, dtype=np.int32)

    for i, ev in enumerate(events):
        hits = _ordered_hits(ev)
        is_gamma = isinstance(ev, GammaEvent)
        ev_type[i] = 1 if is_gamma else 0

        # very generic meta → two common keys, everything else stays in ev.meta
        if getattr(ev, "meta", None):
            if "run" in ev.meta:
                try:
                    ev_run[i] = int(ev.meta["run"])
                except Exception:
                    pass
            if "file_index" in ev.meta:
                try:
                    ev_file_ix[i] = int(ev.meta["file_index"])
                except Exception:
                    pass

        for j, h in enumerate(hits[:3]):
            r = np.asarray(h.r, dtype=float).reshape(3)
            hit_pos[i, j, :] = r
            hit_t[i, j] = float(h.t_ns)
            hit_L[i, j] = float(h.L)
            hit_det[i, j] = int(h.det_id) if h.det_id is not None else -1
            hit_mat[i, j] = mat_id(getattr(h, "material", None))

    lm_grp = f.require_group("lm")

    # Store material vocabulary as a flat label list:
    # index into this with hit_material_id
    mat_labels = np.array(material_list, dtype=h5py.string_dtype())
    if "hit_material_id_labels" in lm_grp:
        del lm_grp["hit_material_id_labels"]
    lm_grp.create_dataset("hit_material_id_labels", data=mat_labels)

    def _replace_or_create(name: str, data: np.ndarray):
        if name in lm_grp:
            del lm_grp[name]
        lm_grp.create_dataset(name, data=data, compression="gzip")

    _replace_or_create("event_type", ev_type)
    # Add a legend for event_type: 0 = neutron, 1 = gamma.
    d_event_type = lm_grp["event_type"]
    d_event_type.attrs["legend"] = np.array(
        ["0=neutron", "1=gamma"],
        dtype=h5py.string_dtype(),
    )
    _replace_or_create("event_meta_run_id", ev_run)
    _replace_or_create("event_meta_file_ix", ev_file_ix)
    _replace_or_create("hit_pos_cm", hit_pos)
    _replace_or_create("hit_t_ns", hit_t)
    _replace_or_create("hit_L_mevee", hit_L)
    _replace_or_create("hit_det_id", hit_det)
    _replace_or_create("hit_material_id", hit_mat)
    
    # Legend for event_type (0=neutron, 1=gamma) as a visible dataset
    event_type_labels = np.array(
        ["0=neutron", "1=gamma"],
        dtype=h5py.string_dtype(),
    )
    if "event_type_labels" in lm_grp:
        del lm_grp["event_type_labels"]
    lm_grp.create_dataset("event_type_labels", data=event_type_labels)



def read_summed(path: str, species: str = "n") -> np.ndarray:
    path = str(path)
    with h5py.File(path, "r") as f:
        grp = f["images"]["summed"]
        if species not in grp:
            raise KeyError(f"{species} not found in /images/summed of {path}")
        arr = np.array(grp[species], dtype=np.float32)
    return arr


def write_event_cone_survival(
    f: h5py.File,
    event_cone_id: np.ndarray,
    event_imaged_cone_id: np.ndarray,
) -> None:
    """
    Store per-event survival information linking events → cones.

    Layout (all under /lm):

      /lm/event_cone_id         : [N_events] int32
          For each event row i (as in /lm/event_type, /lm/hit_*):
              - cone_id of the cone built from this event, or -1 if no cone.

      /lm/event_imaged_cone_id  : [N_events] int32
          For each event row i:
              - cone_id of the cone that both exists AND hits the imaging plane
                (has non-empty pixel set), or -1 if none.

    Notes
    -----
    * event index i is simply the row index into /lm/event_type, /lm/hit_*.
    * event_imaged_cone_id is only meaningfully populated when [run].list = true;
      for non-list runs it will typically be all -1.
    """
    lm_grp = f.require_group("lm")

    if "event_cone_id" in lm_grp:
        del lm_grp["event_cone_id"]
    lm_grp.create_dataset(
        "event_cone_id",
        data=event_cone_id.astype(np.int32),
        compression="gzip",
    )

    if "event_imaged_cone_id" in lm_grp:
        del lm_grp["event_imaged_cone_id"]
    lm_grp.create_dataset(
        "event_imaged_cone_id",
        data=event_imaged_cone_id.astype(np.int32),
        compression="gzip",
    )


def write_root_novo_meta(f: h5py.File, meta: Dict[str, Any]) -> None:
    """
    Persist NOVO DDAQ ROOT run-level metadata into HDF5 under /meta/root_novo_ddaq.

    The input 'meta' is expected to be a flat mapping from branch names
    in the ROOT 'meta' TTree to Python scalars/strings, as returned by
    RootNovoDdaqAdapter.read_meta_tree().

    Layout
    ------
      /meta/root_novo_ddaq           : group
        attrs:
          InputFileName, OutputFileName, CDFFileName, PSDCutsFileName
          SampleRate, NumDet, NumThreads, WriteHistograms,
          MergeMode, CardOffsetChannel, UsePositionVeto

        /meta/root_novo_ddaq/detectors
          det_id            : [NumDet] int32
          pos               : [NumDet, 3] float32   (PosX, PosY, PosZ)   [mm]
          dim               : [NumDet, 3] float32   (DimX, DimY, DimZ)   [mm]
          rot_deg           : [NumDet, 3] float32   (RotX, RotY, RotZ)   [deg]
          local_time_offset : [NumDet] float32      [ns]
          global_time_offset: [NumDet] float32      [ns]
          pos_cal_file      : [NumDet] string
          energy_cal_file   : [NumDet] string
          is_start_det      : [NumDet] int8
          is_laser_det      : [NumDet] int8
    """
    meta_root = f.require_group("meta").require_group("root_novo_ddaq")

    # ---- Run-level scalars as attributes ----
    attr_keys = [
        "InputFileName",
        "OutputFileName",
        "CDFFileName",
        "PSDCutsFileName",
        "SampleRate",
        "NumDet",
        "NumThreads",
        "WriteHistograms",
        "MergeMode",
        "CardOffsetChannel",
        "UsePositionVeto",
    ]
    for key in attr_keys:
        if key not in meta:
            continue
        val = meta[key]
        if isinstance(val, np.generic):
            val = val.item()
        meta_root.attrs[key] = val
    
    # ---- Attempt to extract run number from metadata / filenames ----
    #
    # Priority:
    #   1. Explicit numeric run key if present (e.g. "RunNumber" or "RunNo").
    #   2. Infer from InputFileName / OutputFileName by looking for the last
    #      block of digits before ".root" (e.g. "..._000041.root" → 41).
    run_number_int: int | None = None
    run_number_str: str | None = None

    # 1) If the meta dict already has an explicit run number, prefer that.
    for key in ("RunNumber", "RunNo", "runNumber", "run_no"):
        if key in meta:
            v = meta[key]
            if isinstance(v, np.generic):
                v = v.item()
            try:
                run_number_int = int(v)
                run_number_str = f"{run_number_int:d}"
            except Exception:
                # Fall back to string representation if not cleanly integer.
                run_number_str = str(v)
            break

    # 2) Otherwise, try to infer from the filename pattern.
    if run_number_int is None:
        fname = meta.get("InputFileName") or meta.get("OutputFileName")
        if isinstance(fname, np.generic):
            fname = fname.item()
        if fname:
            try:
                # Work with just the basename, e.g. "coinc_detector_..._000041.root"
                base = str(Path(fname).name)
                # Look for the last run of digits before ".root"
                m = re.search(r"(\d+)\.root$", base)
                if m:
                    run_number_str = m.group(1)
                    try:
                        run_number_int = int(run_number_str)
                    except Exception:
                        # Keep the string even if int conversion fails
                        pass
            except Exception:
                pass

    # If we found something, store it as attributes.
    if run_number_int is not None:
        meta_root.attrs["RunNumber"] = int(run_number_int)
    if run_number_str is not None:
        meta_root.attrs["RunNumber_str"] = str(run_number_str)

    
    
    # ---- Detector table ----
    num_det = int(meta.get("NumDet", 0) or 0)
    if num_det <= 0:
        return

    det_grp = meta_root.require_group("detectors")

    # Helper to (re)create a dataset
    def _create_or_replace(name: str, data: np.ndarray, **kwargs) -> h5py.Dataset:
        if name in det_grp:
            del det_grp[name]
        return det_grp.create_dataset(name, data=data, compression="gzip", **kwargs)

    det_ids = np.arange(num_det, dtype=np.int32)
    _create_or_replace("det_id", det_ids)

    # Numeric tables
    pos = np.zeros((num_det, 3), dtype=np.float32)
    dim = np.zeros((num_det, 3), dtype=np.float32)
    rot = np.zeros((num_det, 3), dtype=np.float32)
    local_off = np.zeros(num_det, dtype=np.float32)
    global_off = np.zeros(num_det, dtype=np.float32)
    is_start = np.zeros(num_det, dtype=np.int8)
    is_laser = np.zeros(num_det, dtype=np.int8)

    pos_cal: List[str] = []
    e_cal: List[str] = []

    for i in range(num_det):
        def _get(name: str, default=0.0):
            key = f"Det{i}_{name}"
            v = meta.get(key, default)
            if isinstance(v, np.generic):
                v = v.item()
            return v

        pos[i, 0] = float(_get("PosX", 0.0))
        pos[i, 1] = float(_get("PosY", 0.0))
        pos[i, 2] = float(_get("PosZ", 0.0))

        dim[i, 0] = float(_get("DimX", 0.0))
        dim[i, 1] = float(_get("DimY", 0.0))
        dim[i, 2] = float(_get("DimZ", 0.0))

        rot[i, 0] = float(_get("RotX", 0.0))
        rot[i, 1] = float(_get("RotY", 0.0))
        rot[i, 2] = float(_get("RotZ", 0.0))

        local_off[i] = float(_get("LocalTimeOffset", 0.0))
        global_off[i] = float(_get("GlobalTimeOffset", 0.0))

        # Filenames
        pc = meta.get(f"Det{i}_PosCalFileName", "")
        ec = meta.get(f"Det{i}_EnergyCalFileName", "")
        pos_cal.append(str(pc))
        e_cal.append(str(ec))

        # Flags (stored as ints)
        is_start[i] = int(_get("IsStartDet", 0))
        is_laser[i] = int(_get("IsLaserDet", 0))

    ds_pos = _create_or_replace("pos", pos)
    ds_pos.attrs["units"] = "mm"

    ds_dim = _create_or_replace("dim", dim)
    ds_dim.attrs["units"] = "mm"

    ds_rot = _create_or_replace("rot_deg", rot)
    ds_rot.attrs["units"] = "deg"

    ds_local = _create_or_replace("local_time_offset", local_off)
    ds_local.attrs["units"] = "ns"

    ds_global = _create_or_replace("global_time_offset", global_off)
    ds_global.attrs["units"] = "ns"

    _create_or_replace(
        "is_start_det",
        is_start,
    )
    _create_or_replace(
        "is_laser_det",
        is_laser,
    )

    # String datasets
    str_dt = h5py.string_dtype(encoding="utf-8")
    _create_or_replace("pos_cal_file", np.asarray(pos_cal, dtype=str_dt))
    _create_or_replace("energy_cal_file", np.asarray(e_cal, dtype=str_dt))


