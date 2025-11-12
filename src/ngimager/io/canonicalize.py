# src/ngimager/io/canonicalize.py
from __future__ import annotations
from typing import Dict, Any, Iterable, Mapping, List

_CANON_KEYS = {
    # canonical_key: tuple of fallback source keys
    "x_cm": ("x_cm", "x", "Xcm", "xPos_cm"),
    "y_cm": ("y_cm", "y", "Ycm", "yPos_cm"),
    "z_cm": ("z_cm", "z", "Zcm", "zPos_cm"),
    "t_ns": ("t_ns", "t", "time_ns", "Tns"),
    # L is “light-like” / MeVee-ish. If missing, fill from Edep_MeV when appropriate.
    "L":    ("L", "Elong", "L_MeVee"),
    # Energy deposition (MeV). May be absent for some sources.
    "Edep_MeV": ("Edep_MeV", "Edep", "edep"),
    # Detector/region identity
    "det_id": ("det_id", "reg", "det", "region"),
}

def _first(h: Mapping[str, Any], names: Iterable[str], default=None):
    for k in names:
        if k in h:
            return h[k]
    return default

def canonicalize_events_inplace(events: List[Dict[str, Any]]) -> None:
    """
    Ensure each hit dict has the canonical keys used by filters/shapers:
        x_cm, y_cm, z_cm, t_ns, L, Edep_MeV, det_id
    Missing values are filled conservatively (L from Edep_MeV if absent).
    Mutates in place; safe to call on PHITS/ROOT outputs.
    """
    for ev in events:
        hits = ev.get("hits", [])
        canon_hits = []
        for h in hits:
            ch = {}
            ch["x_cm"] = float(_first(h, _CANON_KEYS["x_cm"], 0.0))
            ch["y_cm"] = float(_first(h, _CANON_KEYS["y_cm"], 0.0))
            ch["z_cm"] = float(_first(h, _CANON_KEYS["z_cm"], 0.0))
            ch["t_ns"] = float(_first(h, _CANON_KEYS["t_ns"], 0.0))
            # energy + light-like
            edep = _first(h, _CANON_KEYS["Edep_MeV"], None)
            ch["Edep_MeV"] = float(edep) if edep is not None else 0.0
            L = _first(h, _CANON_KEYS["L"], None)
            ch["L"] = float(L) if L is not None else float(ch["Edep_MeV"])  # fallback
            # det/region
            det = _first(h, _CANON_KEYS["det_id"], 0)
            ch["det_id"] = int(det)
            # preserve all source fields as extras
            ch["__extras__"] = dict(h)
            canon_hits.append(ch)
        ev["hits"] = canon_hits
