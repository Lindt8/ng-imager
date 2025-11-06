"""
ngimager.io.adapters

Modular readers that turn external NOVO data sources (PHITS dumps or
experiment/MC ROOT trees) into normalized physics-layer events
(ngimager.physics.hits.Hit; ngimager.physics.events.{NeutronEvent,GammaEvent})
for the cone builder.

Design goals
------------
- Keep I/O concerns isolated from physics/kinematics.
- Normalize units on ingest:
  * distances -> cm
  * times     -> ns
- Be tolerant to schema variants by using small, explicit field maps.
- Stream (iterate) large files without loading everything into RAM.
- Remain side-effect free: yield Python objects; HDF5 is handled downstream.

Entry points
------------
- class ROOTAdapter: reads NOVO ROOT trees ("Joey" or "Lena" styles).
- class PHITSAdapter: reads tabular PHITS lists (CSV/Parquet/HDF5).
- function make_adapter(cfg): factory from the [io.adapter] TOML section.

Config (example)
----------------
[io]
input = "data/run42.root"

[io.adapter]
type = "root"                 # "root" | "phits"
style = "Joey"                # ROOT styles: "Joey" | "Lena"
unit_pos_is_mm = true
time_units = "ns"             # "ns" | "ps"
require_gamma_triples = false # keep filtering in pipeline by default
default_material = "M600"     # tag assigned to all hits unless mapped
# fieldmap = { x1="x1_custom", Elong1="ElongFirst", ... }

"""
from __future__ import annotations

from typing import Dict, Iterator, Literal, Optional, Any

# Optional imports (guarded)
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import uproot  # type: ignore
except Exception:  # pragma: no cover
    uproot = None  # type: ignore

import numpy as np

# ---------------------------------------------------------------------------
# Use canonical physics-layer types
# ---------------------------------------------------------------------------
from ngimager.physics.hits import Hit  # NOTE: Hit should have: det_id:int, r:np.ndarray(3), t_ns:float, L:float, material:str, extras:dict[str,Any]=...
from ngimager.physics.events import NeutronEvent, GammaEvent  # NOTE: should accept meta:dict[str,Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CM_PER_MM = 0.1

def _mm_to_cm(v: float) -> float:
    return v * _CM_PER_MM

def _as_opt_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Base adapter API
# ---------------------------------------------------------------------------

class BaseAdapter:
    """
    Abstract adapter interface.

    Yields physics-layer events normalized to cm/ns (and L if present).
    """

    def iter_events(self, path: str):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ROOT adapter
# ---------------------------------------------------------------------------

class ROOTAdapter(BaseAdapter):
    """
    Read NOVO ROOT files produced by the sort code or by MC emulating it.

    Two common schema flavors are supported via `style`:
      - "Joey" (default): tree key 'image_tree;1'
      - "Lena":           tree key 'ntuple_NCE_raw;1'

    Field names can be overridden via `fieldmap` if your tree differs.

    Parameters
    ----------
    style : Literal['Joey','Lena']
    require_gamma_triples : bool
        If True, only yield GammaEvent when all three hits exist; else be permissive.
    unit_pos_is_mm : bool
        If True (default), convert positions from mm -> cm.
    keep_mevee : bool
        If True, use Elong* branches as 'L' (light-like) for Hit; else set L=0.0.
    time_branch_units : Literal['ns','ps']
    fieldmap : Dict[str,str]
        Override mapping from canonical keys (x1,y1,...) to your branch names.
    default_material : str
        Material tag to attach to hits (e.g., "M600").
    """

    _DEFAULT_KEYS_JOEY = {
        # positions [mm]
        "x1": "x1", "y1": "y1", "z1": "z1",
        "x2": "x2", "y2": "y2", "z2": "z2",
        "x3": "x3", "y3": "y3", "z3": "z3",
        # times
        "t1": "t1", "t2": "t2", "t3": "t3",
        # optional light-like / energy-ish
        "Elong1": "Elong1", "Elong2": "Elong2", "Elong3": "Elong3",
        "dE1": "dE1", "dE2": "dE2", "dE3": "dE3",
        # detector ID & PSD
        "det1": "det1", "det2": "det2", "det3": "det3",
        "psd1": "psd1", "psd2": "psd2", "psd3": "psd3",
        # particle id (0=unk,1=n,2=g,3=laser) — optional
        "par1": "particle1", "par2": "particle2", "par3": "particle3",
    }

    _DEFAULT_KEYS_LENA = {
        "x1": "hit_x1", "y1": "hit_y1", "z1": "hit_z1",
        "x2": "hit_x2", "y2": "hit_y2", "z2": "hit_z2",
        "x3": "hit_x3", "y3": "hit_y3", "z3": "hit_z3",
        "t1": "hit_time1", "t2": "hit_time2", "t3": "hit_time3",
        "Elong1": "elong1", "Elong2": "elong2", "Elong3": "elong3",
        "dE1": "hit_edep1", "dE2": "hit_edep2", "dE3": "hit_edep3",
        "det1": "det1", "det2": "det2", "det3": "det3",
        "psd1": "psd1", "psd2": "psd2", "psd3": "psd3",
        "par1": "particle1", "par2": "particle2", "par3": "particle3",
    }

    def __init__(
        self,
        style: Literal["Joey", "Lena"] = "Joey",
        require_gamma_triples: bool = False,
        unit_pos_is_mm: bool = True,
        keep_mevee: bool = True,
        time_branch_units: Literal["ns", "ps"] = "ns",
        fieldmap: Optional[Dict[str, str]] = None,
        default_material: str = "M600",
    ) -> None:
        if uproot is None:  # pragma: no cover
            raise RuntimeError("uproot is required for ROOTAdapter but is not installed.")
        self.style = style
        self.unit_pos_is_mm = unit_pos_is_mm
        self.keep_mevee = keep_mevee
        self.require_gamma_triples = require_gamma_triples
        self.time_scale = 0.001 if time_branch_units == "ps" else 1.0
        base = self._DEFAULT_KEYS_JOEY if style == "Joey" else self._DEFAULT_KEYS_LENA
        self.keys: Dict[str, str] = {**base, **(fieldmap or {})}
        self.tree_key = "image_tree;1" if style == "Joey" else "ntuple_NCE_raw;1"
        self.default_material = default_material

    def iter_events(self, path: str):
        with uproot.open(path) as f:
            try:
                tree = f[self.tree_key]
            except Exception:
                first_key = next(iter(f.keys()))
                tree = f[first_key]

            # Iterate in chunks for streaming
            for arrays in tree.iterate(filter_name=list(self.keys.values()), step_size="100 MB"):
                A = {k: arrays.get(v) for k, v in self.keys.items() if v in arrays}
                n = len(next(iter(A.values()))) if A else 0

                for i in range(n):
                    # Build physics Hit with 'extras' preserved for filtering
                    def h(which: Literal["1", "2", "3"]) -> Hit:
                        x = A.get(f"x{which}")[i]
                        y = A.get(f"y{which}")[i]
                        z = A.get(f"z{which}")[i]
                        t = A.get(f"t{which}")[i] * self.time_scale
                        if self.unit_pos_is_mm:
                            x, y, z = _mm_to_cm(x), _mm_to_cm(y), _mm_to_cm(z)

                        rvec = np.array([float(x), float(y), float(z)], dtype=float)
                        det = int(A.get(f"det{which}")[i]) if A.get(f"det{which}") is not None else 0

                        elong = A.get(f"Elong{which}")
                        L = float(elong[i]) if (self.keep_mevee and elong is not None) else 0.0

                        # Preserve auxiliary fields for later filtering
                        extras: Dict[str, Any] = {}
                        for key in (f"dE{which}", f"psd{which}", f"Elong{which}", f"par{which}"):
                            if A.get(key) is not None:
                                val = A[key][i]
                                if val is not None:
                                    try:
                                        extras[key] = float(val)
                                    except Exception:
                                        try:
                                            extras[key] = int(val)
                                        except Exception:
                                            extras[key] = val

                        return Hit(
                            det_id=det,
                            r=rvec,
                            t_ns=float(t),
                            L=L,
                            material=self.default_material,
                            extras=extras,
                        )

                    entry_meta = {
                        "source": "ROOT",
                        "file": path,
                        "tree": getattr(tree, "name", str(self.tree_key)),
                        "entry_index": int(i),
                    }

                    has12 = all(A.get(k) is not None for k in ("x1", "y1", "z1", "t1", "x2", "y2", "z2", "t2"))
                    has3  = all(A.get(k) is not None for k in ("x3", "y3", "z3", "t3"))

                    if self.require_gamma_triples and has3:
                        yield GammaEvent(h1=h("1"), h2=h("2"), h3=h("3"), meta=entry_meta)
                        continue

                    # Permissive default: yield gamma if triple available, else neutron if double available
                    if has3:
                        yield GammaEvent(h1=h("1"), h2=h("2"), h3=h("3"), meta=entry_meta)
                        continue
                    if has12:
                        yield NeutronEvent(h1=h("1"), h2=h("2"), meta=entry_meta)


# ---------------------------------------------------------------------------
# PHITS adapter
# ---------------------------------------------------------------------------

class PHITSAdapter(BaseAdapter):
    """
    Read tabular event lists exported from PHITS post-processing.

    Supported inputs: CSV (.csv), Parquet (.parquet/.pq), HDF (.h5/.hdf5).

    The adapter expects row-wise events. Each row is either a neutron double
    or a gamma triple. You may supply a `fieldmap` describing your column names.

    Canonical field names (columns):
      - x1,y1,z1,t1 ; x2,y2,z2,t2 ; [x3,y3,z3,t3]
      - det1,det2,[det3] ; L1,L2,[L3] (or elong1,elong2,[elong3])
      - type (optional) values: 'n'|'g' ; if absent we infer by presence of 3rd hit

    Units are assumed mm (pos) and ns (time) unless overridden.
    """

    def __init__(
        self,
        fieldmap: Optional[Dict[str, str]] = None,
        unit_pos_is_mm: bool = True,
        time_units: Literal["ns", "ps"] = "ns",
        default_material: str = "M600",
    ) -> None:
        if pd is None:  # pragma: no cover
            raise RuntimeError("pandas is required for PHITSAdapter but is not installed.")
        self.unit_pos_is_mm = unit_pos_is_mm
        self.time_scale = 0.001 if time_units == "ps" else 1.0
        self.map = fieldmap or {}
        self.default_material = default_material

    def _read_table(self, path: str):
        p = path.lower()
        if p.endswith(".csv"):
            return pd.read_csv(path)
        if p.endswith(".parquet") or p.endswith(".pq"):
            return pd.read_parquet(path)
        if p.endswith(".h5") or p.endswith(".hdf5"):
            return pd.read_hdf(path)
        raise ValueError(f"Unsupported PHITS table format: {path}")

    def _hit_from_row(self, r, which: str) -> Optional[Hit]:
        keys = [f"x{which}", f"y{which}", f"z{which}", f"t{which}"]
        if not all(k in r and pd.notna(r[k]) for k in keys):
            return None

        x, y, z, t = r[f"x{which}"], r[f"y{which}"], r[f"z{which}"], r[f"t{which}"]
        if self.unit_pos_is_mm:
            x, y, z = _mm_to_cm(float(x)), _mm_to_cm(float(y)), _mm_to_cm(float(z))

        rvec = np.array([float(x), float(y), float(z)], dtype=float)
        det = int(r.get(f"det{which}") or 0)

        # Support either L* or elong* column names
        L_val = r.get(f"L{which}")
        if L_val is None:
            L_val = r.get(f"elong{which}")
        L = float(L_val) if (L_val is not None and pd.notna(L_val)) else 0.0

        # Preserve any per-hit extras that exist in the row
        extras: Dict[str, Any] = {}
        for k in (f"dE{which}", f"psd{which}", f"elong{which}", f"L{which}"):
            if k in r and pd.notna(r[k]):
                try:
                    extras[k] = float(r[k])
                except Exception:
                    extras[k] = r[k]

        return Hit(
            det_id=det,
            r=rvec,
            t_ns=float(t) * self.time_scale,
            L=L,
            material=self.default_material,
            extras=extras,
        )

    def iter_events(self, path: str):
        df = self._read_table(path)
        # Apply field rename mapping once, if provided
        if self.map:
            keep = {dst: src for dst, src in self.map.items() if src in df.columns}
            if keep:
                df = df.rename(columns={v: k for k, v in keep.items()})

        for idx, r in df.iterrows():
            h1 = self._hit_from_row(r, "1")
            h2 = self._hit_from_row(r, "2")
            h3 = self._hit_from_row(r, "3")
            if h1 is None or h2 is None:
                continue

            typ = r.get("type")
            row_meta = {"source": "PHITS", "file": path, "row_index": int(idx)}

            if (typ == "g") and (h3 is not None):
                yield GammaEvent(h1=h1, h2=h2, h3=h3, meta=row_meta)
            elif (typ is None) and (h3 is not None):
                # permissive default if a third hit exists
                yield GammaEvent(h1=h1, h2=h2, h3=h3, meta=row_meta)
            else:
                yield NeutronEvent(h1=h1, h2=h2, meta=row_meta)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_adapter(cfg: Dict) -> BaseAdapter:
    """
    Create an adapter from a config dict (from TOML/CLI).

    Expected keys under [io.adapter]:
      type: "root" | "phits"
      style: "Joey" | "Lena"            (ROOT-only)
      unit_pos_is_mm: bool
      time_units: "ns" | "ps"
      require_gamma_triples: bool       (ROOT-only)
      default_material: str
      fieldmap: { canonical -> actual }
    """
    typ = (cfg.get("type") or "root").lower()

    if typ == "root":
        return ROOTAdapter(
            style=cfg.get("style", "Joey"),
            require_gamma_triples=bool(cfg.get("require_gamma_triples", False)),
            unit_pos_is_mm=bool(cfg.get("unit_pos_is_mm", True)),
            keep_mevee=bool(cfg.get("keep_mevee", True)),
            time_branch_units=cfg.get("time_units", "ns"),
            fieldmap=cfg.get("fieldmap"),
            default_material=cfg.get("default_material", "M600"),
        )

    if typ == "phits":
        return PHITSAdapter(
            fieldmap=cfg.get("fieldmap"),
            unit_pos_is_mm=bool(cfg.get("unit_pos_is_mm", True)),
            time_units=cfg.get("time_units", "ns"),
            default_material=cfg.get("default_material", "M600"),
        )

    raise ValueError(f"Unknown adapter type: {typ}")


# ---------------------------------------------------------------------------
# Simple smoke tests (manual) — run: python -m ngimager.io.adapters root file.root
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m ngimager.io.adapters <root|phits> <path>")
        sys.exit(1)
    kind, path = sys.argv[1], sys.argv[2]
    ad = make_adapter({"type": kind})
    for j, ev in zip(range(5), ad.iter_events(path)):
        print(f"[{j:03d}] {type(ev).__name__} {getattr(ev, 'meta', {})}")
