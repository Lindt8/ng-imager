from __future__ import annotations

from pathlib import Path
import json

from .schemas import Config

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # py<=310


def _resolve_io_paths(data: dict, cfg_dir: Path) -> dict:
    """
    Resolve I/O-related paths in the raw TOML data relative to the config
    file directory.

    Rules:
      - [io].input_path, [io].output_path, [io].restart_path:
          * if absolute → keep as-is
          * if relative → cfg_dir / path, then .resolve()
      - [io.extra_text_files].* values:
          * same rule as above

    This function mutates the 'data' dict in-place and also returns it
    for convenience.
    """
    io = data.get("io")
    if not isinstance(io, dict):
        return data

    def _resolve_one(val: str) -> str:
        p = Path(val)
        if p.is_absolute():
            return str(p)
        return str((cfg_dir / p).resolve())

    # Primary IO paths
    for key in ("input_path", "output_path", "restart_path"):
        val = io.get(key)
        if isinstance(val, str) and val.strip() != "":
            io[key] = _resolve_one(val)

    # Extra text files (keys are dataset names; values are file paths)
    extra = io.get("extra_text_files")
    if isinstance(extra, dict):
        for k, v in list(extra.items()):
            if isinstance(v, str) and v.strip() != "":
                extra[k] = _resolve_one(v)

    return data


def load_config(path: str | Path) -> Config:
    """
    Load a TOML config file into a Config object.

    All paths inside the [io] table are interpreted relative to the
    *location of the TOML file* (cfg_dir), except when they are absolute.

    In particular:
      - [io].input_path
      - [io].output_path
      - [io].restart_path
      - [io.extra_text_files].*

    CLI overrides (e.g. --input-path/--output-path in ng-run) are applied
    later and remain relative to the current working directory.
    """
    p = Path(path)
    cfg_dir = p.resolve().parent
    data = tomllib.loads(p.read_text())
    data = _resolve_io_paths(data, cfg_dir)
    return Config(**data)


def snapshot_config_toml(path: str | Path) -> str:
    """Return the raw TOML text for embedding in HDF5 metadata."""
    return Path(path).read_text()


def json_dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
