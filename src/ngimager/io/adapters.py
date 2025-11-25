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
- class ROOTAdapter: reads NOVO ROOT trees ("novo_ddaq" or "hvl_geant4" styles).
- class PHITSAdapter: reads tabular PHITS lists (CSV/Parquet/HDF5).
- function make_adapter(cfg): factory from the [io.adapter] TOML section.

Config (example)
----------------
[io]
input = "data/run42.root"

[io.adapter]
type = "root"                 # "root" | "phits"
style = "novo_ddaq"                # ROOT styles: "novo_ddaq" | "hvl_geant4"
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

    def iter_raw_events(self, path: str):
        """
        Yield 'raw' events as collections of canonical Hit objects.

        Semantics:
          - Each yielded item represents a single raw coincidence window.
          - For PHITS usrdef, this is a dict with at least:
                {
                    "event_type": "n" | "g" | ...,
                    "hits": [Hit, Hit, ...],
                    ... (bookkeeping fields)
                }
          - Other adapters may choose a different raw representation, but
            must include a 'hits' field with a sequence of Hit objects.
        """
        raise NotImplementedError

    def iter_events(self, path: str):
        """
        Yield fully-typed physics events (NeutronEvent / GammaEvent, etc.)
        ready for cone building.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ROOT adapter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ROOT NOVO DDAQ adapter
# ---------------------------------------------------------------------------

@dataclass
class RootNovoDdaqAdapter(BaseAdapter):
    """
    Adapter for NOVO DDAQ ROOT files (\"image_tree\" + optional \"meta\" tree).

    This adapter:
      - reads the main coincidence tree (image_tree) and yields raw events
        with canonicalized hits, and
      - can optionally read the run-level metadata tree (meta) via
        `read_meta_tree` for passthrough into HDF5.

    Parameters
    ----------
    tree_key : str
        Name of the ROOT TTree containing the imaging events
        (default: \"image_tree\").
    unit_pos_is_mm : bool
        If True, hit positions are stored in mm and converted to cm.
    time_units : {\"ns\", \"ps\"}
        Units of the time branches (converted to ns).
    default_material : str
        Material tag to use when no mapping is provided.
    material_map : dict[int,str] or None
        Mapping from det_id to material name.
    require_gamma_triples : bool
        If True, drop gamma events that do not have exactly 3 hits.
    meta_tree_key : str or None
        Name of the metadata TTree (default \"meta\"). If None, metadata
        extraction is disabled.
    """

    tree_key: str = "image_tree"
    unit_pos_is_mm: bool = True
    time_units: Literal["ns", "ps"] = "ns"
    default_material: str = "UNK"
    material_map: Optional[Dict[int, str]] = None
    meta_tree_key: Optional[str] = "meta"

    def __post_init__(self) -> None:
        # MaterialResolver uses detectors.material_map + detectors.default_material
        self._material_resolver = MaterialResolver.from_mapping(
            self.material_map, default=self.default_material
        )
        if self.time_units == "ns":
            self._time_scale = 1.0
        elif self.time_units == "ps":
            self._time_scale = 1e-3
        else:
            raise ValueError(f"Unsupported time_units {self.time_units!r}")

    # --- internals -----------------------------------------------------

    def _find_tree(self, path: str):
        """
        Open the ROOT file and locate the image tree.

        We first try exact keys:
          - 'image_tree'
          - 'image_tree;1'

        and then fall back to any key containing 'image_tree'.
        """
        if uproot is None:  # pragma: no cover - optional dep
            raise RuntimeError(
                "uproot is required to read ROOT files but is not installed. "
                "Install ngimager with the 'root' extras or add uproot to your environment."
            )

        f = uproot.open(path)  # type: ignore[arg-type]

        # Try exact and ROOT-suffixed keys first.
        for cand in (self.tree_key, f"{self.tree_key};1"):
            if cand in f:
                return f[cand]

        # Fallback: any key that contains tree_key.
        for k in f.keys():
            name = str(k)
            if self.tree_key in name:
                return f[name]

        raise KeyError(f"Could not find tree {self.tree_key!r} in ROOT file {path!r}")

    def _find_meta_tree(self, f):
        """
        Best-effort lookup for the NOVO 'meta' TTree.

        Returns the TTree object or None if not found.
        """
        if self.meta_tree_key:
            preferred = [self.meta_tree_key, f"{self.meta_tree_key};1"]
        else:
            preferred = []

        # Fallback common names
        for base in ("meta", "Meta", "META"):
            preferred.extend([base, f"{base};1"])

        seen = set()
        for key in preferred:
            if key in seen:
                continue
            seen.add(key)
            try:
                obj = f[key]
            except Exception:
                continue
            try:
                from uproot.behaviors.TTree import TTree  # type: ignore
            except Exception:  # pragma: no cover
                return obj
            if isinstance(obj, TTree):
                return obj

        return None
    
    # --- BaseAdapter API -----------------------------------------------

    def iter_raw_events(self, path: str):
        """
        Yield raw coincidence windows as dicts:

            {
              "hits": [Hit, ...],
              "multi": int,          # as stored in the ROOT tree, if present
              "entry": int,          # global entry index
              "source": "ROOT_NOVO_DDAQ",
            }

        This method is intentionally conservative and does **not** make
        any physics decisions about which hits belong to neutron vs
        gamma events; it simply exposes the coincidence window.
        """
        path = str(path)
        tree = self._find_tree(path)

        # Determine which hit indices (1,2,3,...) are present in this tree.
        prefixes = ("x", "y", "z", "t", "dE", "psd", "det", "particle", "clipped")
        hit_indices: set[int] = set()
        for name in tree.keys():
            name_str = str(name)
            for p in prefixes:
                if name_str.startswith(p):
                    suffix = name_str[len(p):]
                    if suffix.isdigit():
                        hit_indices.add(int(suffix))

        if not hit_indices:
            # No per-hit branches found; nothing to yield.
            return

        indices = sorted(hit_indices)

        # Restrict arrays to just the branches we actually need.
        existing = set(str(n) for n in tree.keys())
        branch_names: list[str] = []
        if "multi" in existing:
            branch_names.append("multi")
        for idx in indices:
            for p in prefixes:
                key = f"{p}{idx}"
                if key in existing:
                    branch_names.append(key)

        pos_scale = _CM_PER_MM if self.unit_pos_is_mm else 1.0
        time_scale = self._time_scale

        entry_offset = 0
        # Stream in chunks to avoid loading the full file into RAM.
        for arrays in tree.iterate(filter_name=branch_names, step_size="100 MB", library="np"):
            # NOTE: uproot returns dict-like arrays with NumPy arrays as values.
            multi_arr = arrays.get("multi")
            # Determine the row count from any branch.
            some_arr = next(iter(arrays.values()))
            n_rows = len(some_arr)

            for i in range(n_rows):
                hits: list[Hit] = []

                for idx in indices:
                    # Minimal required fields; if positions or time are missing, skip this hit.
                    try:
                        x = arrays[f"x{idx}"][i]
                        y = arrays[f"y{idx}"][i]
                        z = arrays[f"z{idx}"][i]
                        t = arrays[f"t{idx}"][i]
                        det = arrays[f"det{idx}"][i]
                    except KeyError:
                        continue

                    # Some entries may be "empty" placeholders; guard against obvious sentinels.
                    try:
                        det_id = int(det)
                    except Exception:
                        continue
                    if det_id < 0:
                        continue

                    # Positions in cm
                    x_cm = float(x) * pos_scale
                    y_cm = float(y) * pos_scale
                    z_cm = float(z) * pos_scale
                    rvec = np.array([x_cm, y_cm, z_cm], dtype=float)

                    # Times in ns
                    t_ns = float(t) * time_scale

                    # Use dE as light-like quantity (MeVee) for now.
                    dE_arr = arrays.get(f"dE{idx}")
                    L_mevee = float(dE_arr[i]) if dE_arr is not None else 0.0

                    # Optional PSD / particle / clipped info goes into extras.
                    extras: Dict[str, Any] = {}
                    psd_arr = arrays.get(f"psd{idx}")
                    if psd_arr is not None:
                        extras["psd"] = float(psd_arr[i])

                    part_code = None
                    part_arr = arrays.get(f"particle{idx}")
                    if part_arr is not None:
                        try:
                            part_code = int(part_arr[i])
                        except Exception:
                            part_code = None
                    # Only keep neutron (1) and gamma (2) hits for imaging.
                    if part_code not in (1, 2):
                        # e.g. laser or other special hits: drop them
                        continue
                    extras["particle_code"] = part_code
                    h_type = "n" if part_code == 1 else "g"

                    clipped_arr = arrays.get(f"clipped{idx}")
                    if clipped_arr is not None:
                        is_clipped = bool(clipped_arr[i])
                        if is_clipped:
                            # Drop this hit entirely; legacy imaging does not use clipped hits
                            continue
                        extras["clipped"] = is_clipped

                    # Map particle code (if present) to a simple hit type.
                    h_type: Optional[str] = None
                    if part_code == 1:
                        h_type = "n"
                    elif part_code == 2:
                        h_type = "g"
                    elif part_code == 3:
                        h_type = "laser"

                    # Resolve material from detector ID if mapping is available.
                    material = self._material_resolver.material_for(det_id)

                    hits.append(
                        Hit(
                            det_id=det_id,
                            r=rvec,
                            t_ns=t_ns,
                            L=L_mevee,
                            material=material,
                            type=h_type,
                            extras=extras,
                        )
                    )

                if not hits:
                    # Nothing usable in this coincidence window entry.
                    continue

                multi_val = int(multi_arr[i]) if multi_arr is not None else len(hits)
                yield {
                    "hits": hits,
                    "multi": multi_val,
                    "entry": entry_offset + i,
                    "source": "ROOT_NOVO_DDAQ",
                }

            entry_offset += n_rows

    def iter_events(self, path: str):
        """
        Placeholder: higher-level event shaping for NOVO DDAQ ROOT data.

        For now, the ng-imager pipeline should consume `iter_raw_events`
        and run the standard shaping / filtering stack on top. This method
        is defined only to satisfy the BaseAdapter interface.
        """
        raise NotImplementedError(
            "RootNovoDdaqAdapter.iter_events is not implemented yet; "
            "use iter_raw_events() via a staged pipeline."
        )

    def read_meta_tree(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Read the NOVO 'meta' TTree (if present) and return a flat dict
        mapping branch names → Python scalars/strings.

        This is intended for run-level metadata passthrough into HDF5.
        Returns None if no compatible meta tree is found.
        """
        if uproot is None:  # pragma: no cover
            return None

        path = str(path)
        f = uproot.open(path)
        try:
            tree = self._find_meta_tree(f)
            if tree is None:
                return None

            arrays = tree.arrays(library="np")
        finally:
            try:
                f.close()
            except Exception:
                pass

        if not arrays:
            return {}

        meta: Dict[str, Any] = {}
        for key, vals in arrays.items():
            # Expect exactly one entry; take the first.
            try:
                v = vals[0]
            except Exception:
                v = vals

            if isinstance(v, np.generic):
                v = v.item()

            if isinstance(v, (bytes, bytearray)):
                try:
                    v = v.decode("utf-8", "ignore")
                except Exception:
                    v = repr(v)

            meta[str(key)] = v

        return meta


# Backwards-compat alias; to be removed once config / factory use the new name.
ROOTAdapter = RootNovoDdaqAdapter


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
        ev_type = ev.get("event_type", "UNK") # event-level particle type from PHITS ("n" | "g")
        # Normalize to our canonical one-letter codes
        if ev_type.startswith("n"):
            hit_type = "n"
        elif ev_type.startswith("g"):
            hit_type = "g"
        else:
            hit_type = "UNK"
        for h in ev["hits"]:
            det = int(h["det_id"]) if "det_id" in h else int(h.get("reg", 0))
            r = np.array([h["x_cm"], h["y_cm"], h["z_cm"]], dtype=float)
            extras = dict(h.get("__extras__", {}))
            # Keep Edep explicitly in extras if present
            if "Edep_MeV" in h:
                extras.setdefault("Edep_MeV", h["Edep_MeV"])
            material = resolver.material_for(det)
            hits_H.append(Hit(det_id=det, r=r, t_ns=float(h["t_ns"]), L=float(h.get("L", extras.get("Edep_MeV", 0.0))),
                              type=hit_type, material=material, extras=extras))
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
    
    def iter_raw_events(self, path: str):
        """
        Yield PHITS 'raw' events as dicts whose 'hits' entry is a list of
        canonical Hit objects.

        For usrdef .out files this wraps `from_phits_usrdef`, which:
          - parses the ragged usrdef text,
          - canonicalizes hit fields to x_cm / y_cm / z_cm / t_ns / Edep_MeV / L,
          - and converts each hit dict into a physics.hits.Hit, resolving the
            material via this adapter's MaterialResolver.

        For table-like PHITS exports (CSV/Parquet/HDF5) we currently don't have
        a native raw-event representation, so we conservatively reconstruct a
        minimal raw event around each typed event.
        """
        p = Path(path)
        suffix = p.suffix.lower()

        if suffix == ".out":
            # `from_phits_usrdef` already returns a List[Dict] where
            #   ev["hits"] : List[Hit]
            # and event-level bookkeeping fields from the usrdef line.
            events = from_phits_usrdef(p, resolver=self._material_resolver)
            for ev in events:
                yield ev
            return

        # Fallback: wrap typed events as single raw events (non-.out inputs).
        from ngimager.physics.events import NeutronEvent, GammaEvent  # local import to avoid cycles

        for ev in self.iter_events(path):
            if isinstance(ev, NeutronEvent):
                hits = [ev.h1, ev.h2]
                ev_type = "n"
            elif isinstance(ev, GammaEvent):
                hits = [ev.h1, ev.h2, ev.h3]
                ev_type = "g"
            else:
                # Unknown/unsupported event type; skip
                continue

            yield {
                "event_type": ev_type,
                "hits": hits,
                "meta": getattr(ev, "meta", {}),
            }

    
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
            raw_events = self.iter_raw_events(path)
            #events = from_phits_usrdef(p, resolver=self._material_resolver)
            # 2) shape variable multiplicity into pairs/triples (policy from config later; defaults okay now)
            shaped, _diag = shape_events_for_cones(raw_events, ShapeConfig())
            #shaped, _diag = shape_events_for_cones(events, ShapeConfig())
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
      style: "novo_ddaq" | "hvl_geant4"            (ROOT-only)
      unit_pos_is_mm: bool
      time_units: "ns" | "ps"
      require_gamma_triples: bool       (ROOT-only)
      default_material: str
    """
    typ = (cfg.get("type") or "root").lower()

    if typ == "root":
        return ROOTAdapter(
            unit_pos_is_mm=bool(cfg.get("unit_pos_is_mm", True)),
            time_units=cfg.get("time_units", "ns"),
            default_material=cfg.get("default_material", "UNK"),
            material_map=cfg.get("material_map"),
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
