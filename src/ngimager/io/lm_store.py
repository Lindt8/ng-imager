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


def write_cones(
    f: h5py.File,
    cone_ids: np.ndarray,
    apex_xyz_cm: np.ndarray,
    axis_xyz: np.ndarray,
    theta_rad: np.ndarray,
    species: np.ndarray,
    recoil_code: np.ndarray,
    incident_energy_MeV: np.ndarray,
) -> None:
    """
    Store per-cone geometric parameters under /cones.

    Parameters
    ----------
    f : open h5py.File
    cone_ids : (N,) int
    apex_xyz_cm : (N,3) float
    axis_xyz : (N,3) float (unit vectors)
    theta_rad : (N,) float (half-angle)
    species : (N,) uint8, optional
        Particle species per cone:
            0 = neutron, 1 = gamma.
        If None, all zeros (neutrons) are stored.
    recoil_code : (N,) uint8, optional
        Recoil tagging per cone:
            0 = unknown / not applicable
            1 = proton recoil (for neutron cones)
            2 = carbon recoil (for neutron cones)
        If None, all zeros are stored.
    incident_energy_MeV : (N,)  float32 (En for n, Eg for g)
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

    grp.create_dataset(
        "incident_energy_MeV",
        data=incident_energy_MeV.astype(np.float32),
        compression="gzip",
    )




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

    # Store material vocabulary under /lm/materials
    mats_grp = lm_grp.require_group("materials")
    # Clear existing
    for name in list(mats_grp.keys()):
        del mats_grp[name]
    mats_grp.create_dataset(
        "labels",
        data=np.array(material_list, dtype=h5py.string_dtype()),
    )

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


def read_summed(path: str, species: str = "n") -> np.ndarray:
    path = str(path)
    with h5py.File(path, "r") as f:
        grp = f["images"]["summed"]
        if species not in grp:
            raise KeyError(f"{species} not found in /images/summed of {path}")
        arr = np.array(grp[species], dtype=np.float32)
    return arr



