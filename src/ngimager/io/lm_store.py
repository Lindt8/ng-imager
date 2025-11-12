from __future__ import annotations
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import h5py
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from ngimager.config.schemas import Config
from ngimager.config.load import snapshot_config_toml
from ngimager.geometry.plane import Plane
from ngimager.physics.events import NeutronEvent, GammaEvent

FORMAT_VERSION = "1.0"


def write_init(path: str, cfg_path: str, cfg: Config, plane: Plane) -> h5py.File:
    f = h5py.File(path, "w")
    # Root attrs
    f.attrs["format_version"] = FORMAT_VERSION
    f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
    f.attrs["software"] = "ng-imager 0.1.0"
    f.attrs["config_text"] = snapshot_config_toml(cfg_path)

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
    return f


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


def write_cones(
    f: h5py.File,
    cone_ids: np.ndarray,
    apex_xyz_cm: np.ndarray,
    axis_xyz: np.ndarray,
    theta_rad: np.ndarray,
) -> None:
    """
    Store per-cone geometric parameters under /cones.

    Parameters
    ----------
    cone_ids : (N,) int
    apex_xyz_cm : (N,3) float
    axis_xyz : (N,3) float (unit vectors)
    theta_rad : (N,) float (half-angle)
    """
    grp = f.require_group("cones")
    if "cone_id" in grp:
        del grp["cone_id"]
    if "apex_xyz_cm" in grp:
        del grp["apex_xyz_cm"]
    if "axis_xyz" in grp:
        del grp["axis_xyz"]
    if "theta_rad" in grp:
        del grp["theta_rad"]

    grp.create_dataset("cone_id", data=cone_ids.astype(np.uint32), compression="gzip")
    grp.create_dataset("apex_xyz_cm", data=apex_xyz_cm.astype(np.float32), compression="gzip")
    grp.create_dataset("axis_xyz", data=axis_xyz.astype(np.float32), compression="gzip")
    grp.create_dataset("theta_rad", data=theta_rad.astype(np.float32), compression="gzip")


def write_lm_indices(
    f: h5py.File,
    lm_indices: list[np.ndarray],
) -> None:
    """
    Store list-mode indices mapping cones -> (u,v) pixels.

    We store:
      /lm/indices : ragged array of (cone_id, flat_index) pairs
      /lm/events  : (event_id, cone_id) mapping (event_id is row index in event arrays)
    """
    grp = f.require_group("lm")
    # Flatten all LM lists with cone_id
    all_rows = []
    event_rows = []

    cone_id = 0
    for ev_id, arr in enumerate(lm_indices):
        if arr.size == 0:
            continue
        flat = arr.astype(np.uint32).ravel()
        cone_ids = np.full_like(flat, cone_id, dtype=np.uint32)
        stacked = np.vstack([cone_ids, flat]).T  # (M,2)
        all_rows.append(stacked)
        event_rows.append([ev_id, cone_id])
        cone_id += 1

    if all_rows:
        all_rows_arr = np.concatenate(all_rows, axis=0)
    else:
        all_rows_arr = np.zeros((0, 2), dtype=np.uint32)

    if "indices" in grp:
        del grp["indices"]
    grp.create_dataset("indices", data=all_rows_arr, compression="gzip")

    # /lm/events: event_id <-> cone_id mapping
    if event_rows:
        events_arr = np.asarray(event_rows, dtype=np.uint32)
    else:
        events_arr = np.zeros((0, 2), dtype=np.uint32)

    if "events" in grp:
        del grp["events"]
    grp.create_dataset("events", data=events_arr, compression="gzip")


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

    Layout:

    /lm/event_type         (N_events,) uint8      0=neutron, 1=gamma
    /lm/event_meta_run_id  (N_events,) int32      optional, -1 if missing
    /lm/event_meta_file_ix (N_events,) int32      optional, -1 if missing

    /lm/hit_pos_cm         (N_events, 3, 3) float32   [event, hit_index, xyz]
    /lm/hit_t_ns           (N_events, 3)    float32
    /lm/hit_L_mevee        (N_events, 3)    float32
    /lm/hit_det_id         (N_events, 3)    int32
    /lm/hit_material_id    (N_events, 3)    int16    (encoded from string labels)

    Convention:
      - Neutron events use hits [0,1], hit 2 is NaN / -1.
      - Gamma events use hits [0,1,2].
    """
    if not events:
        return

    N = len(events)

    # Gather materials to build a small vocabulary
    material_labels: set[str] = set()
    for ev in events:
        for h in ev.ordered():
            if h.material is not None:
                material_labels.add(h.material)
    material_list = sorted(material_labels)
    material_to_id = {m: i for i, m in enumerate(material_list)}
    # small helper for encoding
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
        ordered_hits = ev.ordered()
        is_gamma = isinstance(ev, GammaEvent)
        ev_type[i] = 1 if is_gamma else 0

        # very generic meta → two common keys, everything else stays in ev.meta
        if ev.meta:
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

        for j, h in enumerate(ordered_hits[:3]):
            r = np.asarray(h.r, dtype=float).reshape(3)
            hit_pos[i, j, :] = r
            hit_t[i, j] = float(h.t_ns)
            hit_L[i, j] = float(h.L)
            hit_det[i, j] = int(h.det_id) if h.det_id is not None else -1
            hit_mat[i, j] = mat_id(getattr(h, "material", None))

    lm_grp = f.require_group("lm")

    # Store material vocabulary under /lm/materials
    mats_grp = lm_grp.require_group("materials")
    # Clear existing
    for name in list(mats_grp.keys()):
        del mats_grp[name]
    mats_grp.create_dataset("labels", data=np.array(material_list, dtype=h5py.string_dtype()))

    def _replace_or_create(name: str, data: np.ndarray):
        if name in lm_grp:
            del lm_grp[name]
        lm_grp.create_dataset(name, data=data, compression="gzip")

    _replace_or_create("event_type", ev_type)
    _replace_or_create("event_meta_run_id", ev_run)
    _replace_or_create("event_meta_file_ix", ev_file_ix)
    _replace_or_create("hit_pos_cm", hit_pos)
    _replace_or_create("hit_t_ns", hit_t)
    _replace_or_create("hit_L_mevee", hit_L)
    _replace_or_create("hit_det_id", hit_det)
    _replace_or_create("hit_material_id", hit_mat)


def read_summed(path: str, species: str = "n") -> np.ndarray:
    path = str(path)
    with h5py.File(path, "r") as f:
        grp = f["images"]["summed"]
        if species not in grp:
            raise KeyError(f"{species} not found in /images/summed of {path}")
        arr = np.array(grp[species], dtype=np.float32)
    return arr
