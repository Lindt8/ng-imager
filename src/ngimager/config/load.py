from __future__ import annotations
from .schemas import Config
from pathlib import Path
import json

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # py<=310

def load_config(path: str | Path) -> Config:
    p = Path(path)
    data = tomllib.loads(p.read_text())
    return Config(**data)

def snapshot_config_toml(path: str | Path) -> str:
    """Return the raw TOML text for embedding in HDF5 metadata."""
    return Path(path).read_text()

def json_dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

