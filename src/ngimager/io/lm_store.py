from __future__ import annotations
import h5py
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from ..config.schemas import Config
from ..config.load import snapshot_config_toml
from ..geometry.plane import Plane

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
    meta.attrs["plane.n"]  = plane.n
    meta.attrs["plane.eu"] = plane.eu
    meta.attrs["plane.ev"] = plane.ev
    meta.attrs["grid.u_min"] = plane.u_min
    meta.attrs["grid.u_max"] = plane.u_max
    meta.attrs["grid.du"]    = plane.du
    meta.attrs["grid.v_min"] = plane.v_min
    meta.attrs["grid.v_max"] = plane.v_max
    meta.attrs["grid.dv"]    = plane.dv
    meta.attrs["grid.nu"]    = plane.nu
    meta.attrs["grid.nv"]    = plane.nv
    return f

def _ensure_summed_group(f: h5py.File):
    return f.require_group("images").require_group("summed")

def write_summed(f: h5py.File, kind: str, image: np.ndarray):
    g = _ensure_summed_group(f)
    ds = g.create_dataset(kind, data=image, dtype=image.dtype, compression="gzip", compression_opts=5, chunks=True)
    return ds

def write_cones(f: h5py.File, kind: str, apex: np.ndarray, direc: np.ndarray, theta: np.ndarray, sigma_theta: np.ndarray | None = None):
    g = f.require_group("cones").require_group(kind)
    g.create_dataset("apex", data=apex, dtype="f4", compression="gzip", compression_opts=5, chunks=True)
    g.create_dataset("dir",  data=direc, dtype="f4", compression="gzip", compression_opts=5, chunks=True)
    g.create_dataset("theta", data=theta, dtype="f4", compression="gzip", compression_opts=5, chunks=True)
    if sigma_theta is not None:
        g.create_dataset("sigma_theta", data=sigma_theta, dtype="f4", compression="gzip", compression_opts=5, chunks=True)

def write_lm_indices(f: h5py.File, kind: str, indices: list[np.ndarray]):
    g = f.require_group("lm").require_group(kind)
    # varlen uint32
    vlen = h5py.vlen_dtype(np.dtype("uint32"))
    ds = g.create_dataset("indices", (len(indices),), dtype=vlen, compression="gzip", compression_opts=5, chunks=True)
    for i, arr in enumerate(indices):
        ds[i] = np.asarray(arr, dtype=np.uint32)

# --- Readers used by post-analysis or tests
def read_summed(path: str | Path, kind: str = "n") -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["images"]["summed"][kind][...]

def iter_lm_indices(path: str | Path, kind: str = "n"):
    with h5py.File(path, "r") as f:
        ds = f["lm"][kind]["indices"]
        for i in range(ds.shape[0]):
            yield ds[i]
