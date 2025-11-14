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

"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Iterable, List, Literal, Optional, Any, Mapping, Protocol
import io
import pandas as pd
import re

# Optional imports (guarded)
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
from ngimager.io.canonicalize import canonicalize_events_inplace
from ngimager.config.materials import MaterialResolver
from ngimager.filters.shapers import shape_events_for_cones, ShapeConfig
from ngimager.filters.to_typed_events import shaped_to_typed_events

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


# --- helper: sniff if a text file looks like 'short' usrdef ---
def _phits_usrdef_is_short(head: list[str]) -> bool:
    # conservative heuristic: first non-comment, non-empty line column count
    for ln in head:
        s = ln.strip()
        if not s or s.startswith(("#", "*", "!")):
            continue
        ncols = len(s.split())
        # When split, the length should be 21 (18 values + 3 punctuation marks) for neutron events
        # 28 for gamma events
        return ncols < 16
    return True


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

def parse_phits_usrdef_short(path: str | Path) -> List[Dict[str, Any]]:
    """
    Parse PHITS 'usrdef.out' short format into variable-multiplicity events.
    The [T-Userdefined] source code for this tally and documentation can be found at:
    https://github.com/Lindt8/T-Userdefined/tree/main/multi-coincidence_ng

    Input row format (tokens; delimiters ';' and ',' are cosmetic):
        event_type  #iomp  #batch  #history  #no  #name  ;  reg  Edep(MeV)  x(cm)  y(cm)  z(cm)  t(ns)  ,  reg  Edep  x  y  z  t  ,  ...

    Where:
      - event_type: 'ne' (neutron) or 'ge' (gamma)
      - #iomp, #batch, #history, #no, #name: integers (PHITS bookkeeping)
      - For each hit: reg (int), Edep_MeV (float), x_cm (float), y_cm (float), z_cm (float), t_ns (float)
      - 2 hits min for 'ne', 3 hits min for 'ge', but higher multiplicities may appear.

    Returns a list of dicts, each with:
      {
        "event_type": "n" | "g",
        "iomp": int, "batch": int, "history": int, "no": int, "name": int,
        "hits": [
           {"reg": int, "Edep_MeV": float, "x_cm": float, "y_cm": float, "z_cm": float, "t_ns": float},
           ...
        ],
        "source": "PHITS",
        "format": "usrdef.short",
      }

    NOTE: This function performs *no* physics decisions (pair/triple selection, species mixing, etc.).
          It preserves all hits in the order they appear. Shaping happens downstream.
    """
    p = Path(path)
    events: List[Dict[str, Any]] = []

    # Fast replacements: remove cosmetic delimiters; keep whitespace tokenization stable.
    delim_re = re.compile(r"[;,]")

    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(("!", "#")):
                continue

            # Normalize delimiters to spaces and split.
            line = delim_re.sub(" ", line)
            parts = line.split()
            if not parts:
                continue

            # Header: event_type + five ints
            # Defensive checks: ensure we have at least 6 tokens before hits begin.
            if len(parts) < 6:
                continue

            ev_type_tok = parts[0].lower()
            if ev_type_tok not in ("ne", "ge"):
                # If PHITS writes other tags in the future, skip for now (could log)
                continue

            try:
                iomp   = int(parts[1])
                batch  = int(parts[2])
                hist   = int(parts[3])
                no     = int(parts[4])
                name   = int(parts[5])
            except ValueError:
                # Malformed header row; skip
                continue

            # Remaining tokens are in groups of 6 per hit
            toks = parts[6:]
            if len(toks) < 6:
                # No hits present; skip this row
                continue

            if len(toks) % 6 != 0:
                # Truncated or malformed line; drop trailing incomplete group
                toks = toks[: (len(toks)//6) * 6]

            hits: List[Dict[str, Any]] = []
            for i in range(0, len(toks), 6):
                try:
                    reg  = int(toks[i + 0])
                    edep = float(toks[i + 1])   # MeV
                    x    = float(toks[i + 2])   # cm
                    y    = float(toks[i + 3])   # cm
                    z    = float(toks[i + 4])   # cm
                    t    = float(toks[i + 5])   # ns
                except ValueError:
                    # Skip this hit if any conversion fails
                    continue
                hits.append({
                    "reg": reg,
                    "Edep_MeV": edep,
                    "x_cm": x, "y_cm": y, "z_cm": z,
                    "t_ns": t,
                })

            if not hits:
                continue

            events.append({
                "event_type": "n" if ev_type_tok == "ne" else "g",
                "iomp": iomp, "batch": batch, "history": hist, "no": no, "name": name,
                "hits": hits,
                "source": "PHITS",
                "format": "usrdef.short",
            })

    return events

def from_phits_usrdef(path: str | Path, *, format_hint: Literal["short","auto"]="auto", 
                      resolver: MaterialResolver | None = None) -> List[Dict[str, Any]]:
    """
    Public convenience entry point for PHITS usrdef ingestion.
    Currently supports the 'short' format. 'auto' is reserved for future sniffing.
    """
    # In the future: sniff tokens/columns to choose short vs full.
    events = parse_phits_usrdef_short(path)
    canonicalize_events_inplace(events)

    # Resolve material from detector/region id via config (optional)
    if resolver is None: 
        resolver = MaterialResolver.from_env_or_defaults()

    # Convert dict-hits → Hit objects (keep source fields in extras)
    for ev in events:
        hits_H: List[Hit] = []
        for h in ev["hits"]:
            det = int(h["det_id"]) if "det_id" in h else int(h.get("reg", 0))
            r = np.array([h["x_cm"], h["y_cm"], h["z_cm"]], dtype=float)
            extras = dict(h.get("__extras__", {}))
            # Keep Edep explicitly in extras if present
            if "Edep_MeV" in h:
                extras.setdefault("Edep_MeV", h["Edep_MeV"])
            material = resolver.material_for(det)
            hits_H.append(Hit(det_id=det, r=r, t_ns=float(h["t_ns"]), L=float(h.get("L", extras.get("Edep_MeV", 0.0))),
                              material=material, extras=extras))
        ev["hits"] = hits_H
    return events



class PHITSAdapter(BaseAdapter):
    """
    Read tabular event lists exported from PHITS post-processing.

    Supported inputs: CSV (.csv), Parquet (.parquet/.pq), HDF (.h5/.hdf5).

    The adapter expects row-wise events. Each row is either a neutron double
    or a gamma triple. 

    Canonical field names (columns):
      - x1,y1,z1,t1 ; x2,y2,z2,t2 ; [x3,y3,z3,t3]
      - det1,det2,[det3] ; L1,L2,[L3] (or elong1,elong2,[elong3])
      - type (optional) values: 'n'|'g' ; if absent we infer by presence of 3rd hit

    Units are assumed mm (pos) and ns (time) unless overridden.
    """

    def __init__(
        self,
        unit_pos_is_mm: bool = True,
        time_units: Literal["ns", "ps"] = "ns",
        default_material: str = "M600",
        material_map: Optional[Dict[int, str]] = None,
    ) -> None:
        self.unit_pos_is_mm = unit_pos_is_mm
        self.time_scale = 0.001 if time_units == "ps" else 1.0
        self.default_material = default_material
        #mat_map = kwargs.get("material_map", None)
        #default_mat = kwargs.get("default_material", "UNK")
        self._material_resolver = MaterialResolver.from_mapping(material_map, default=default_material)

    def _read_table(self, path: str):
        '''
        Generic table loader, used by future adapters; not currently used in the primary PHITS usrdef path
        '''
        p = Path(path)
        suffix = p.suffix.lower()
        
        # reading from custom [T-Userdefined] output: https://github.com/Lindt8/T-Userdefined/tree/main/multi-coincidence_ng
        if suffix == ".out":
            # route to the short-format reader; if it fails, raise with guidance
            #return from_phits_usrdef(p) #,
            raise ValueError("PHITS usrdef .out is ragged; iter_events() handles it directly.")

        if suffix in {".csv"}:
            df = pd.read_csv(p)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        elif suffix in {".h5", ".hdf5"}:
            df = pd.read_hdf(p)
        else:
            raise ValueError(f"Unrecognized PHITSAdapter input: {p.name} (expected .csv/.parquet/.h5 or usrdef .out)")

    
    def iter_events(self, path: str) -> Iterable[NeutronEvent | GammaEvent]:
        """
        Unified iterator:
          - If 'path' ends with .out (PHITS usrdef, ragged): parse→Hit→shape→typed and yield typed events.
          - Otherwise (CSV/Parquet/HDF): fall back to the existing table-based row iterator.
        """
        p = Path(path)
        if p.suffix.lower() == ".out":
            # 1) parse usrdef → Hit objects (your current helper)
            #raw = from_phits_usrdef(p)     # returns list of dicts where 'hits' are Hit objects (as you implemented)
            #raw = from_phits_usrdef(p, resolver=self._material_resolver)
            events = from_phits_usrdef(p, resolver=self._material_resolver)
            # 2) shape variable multiplicity into pairs/triples (policy from config later; defaults okay now)
            shaped, _diag = shape_events_for_cones(events, ShapeConfig())
            # 3) convert shaped → typed NeutronEvent/GammaEvent
            typed = shaped_to_typed_events(shaped, default_material=self.default_material, order_time=True)
            # 4) yield typed events to the pipeline
            for ev in typed:
                yield ev
            return

        # Fallback: table-based path (unchanged behavior)
        df = self._read_table(path)
        for _, r in df.iterrows():
            # Your existing table-row → typed conversion logic stays as-is here.
            # Example (pseudocode placeholder; keep your real code):
            # ev = self._row_to_event(r)  # existing function
            # yield ev
            raise NotImplementedError("Table row→typed event conversion is unchanged; keep your existing code here.")
    

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
    """
    typ = (cfg.get("type") or "root").lower()

    if typ == "root":
        return ROOTAdapter(
            style=cfg.get("style", "Joey"),
            require_gamma_triples=bool(cfg.get("require_gamma_triples", False)),
            unit_pos_is_mm=bool(cfg.get("unit_pos_is_mm", True)),
            keep_mevee=bool(cfg.get("keep_mevee", True)),
            time_branch_units=cfg.get("time_units", "ns"),
            default_material=cfg.get("default_material", "M600"),
        )

    if typ == "phits":
        return PHITSAdapter(
            unit_pos_is_mm=bool(cfg.get("unit_pos_is_mm", True)),
            time_units=cfg.get("time_units", "ns"),
            default_material=cfg.get("default_material", "UNK"),
            material_map=cfg.get("material_map"),
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
