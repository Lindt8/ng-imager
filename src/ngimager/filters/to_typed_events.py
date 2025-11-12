from __future__ import annotations
from typing import Dict, Any, List, Tuple, Union

import numpy as np

from ngimager.physics.hits import Hit
from ngimager.physics.events import NeutronEvent, GammaEvent

def _hit_from_canon(h: Any, *, material_default: str="UNK") -> Hit:
    """
    Accept either an already-typed Hit or a canonical hit dict and return a Hit.
    Tolerates several common field-name variants to ease adapter reuse.
    """
    # 1) Already a Hit → return as-is
    if isinstance(h, Hit):
        return h

    # 2) Dict-like canonical hit
    if not isinstance(h, dict):
        raise TypeError(f"Unsupported hit type for conversion: {type(h)}")

    def first_key(d: Dict[str, Any], *candidates, default=None):
        for k in candidates:
            if k in d: return d[k]
        return default

    # Coordinates (assume centimeters if not otherwise stated)
    x = float(first_key(h, "x_cm", "x", default=0.0))
    y = float(first_key(h, "y_cm", "y", default=0.0))
    z = float(first_key(h, "z_cm", "z", default=0.0))
    r = np.array([x, y, z], dtype=float)

    det_id = int(first_key(h, "det_id", "det", default=-1))
    t_ns = float(first_key(h, "t_ns", "time_ns", "t", default=0.0))
    # Light-like measure (MeVee); keep flexible to support various adapters
    L = float(first_key(h, "L", "Elong", "Elong_MeVee", "L_MeVee", "dE_MeV", default=0.0))
    material = first_key(h, "material", default=material_default) or material_default

    # Preserve any extra raw/source fields
    used = {"det_id","det","x_cm","x","y_cm","y","z_cm","z",
            "t_ns","time_ns","t","L","Elong","Elong_MeVee","L_MeVee","dE_MeV","material"}
    extras = {k: v for k, v in h.items() if k not in used}
    return Hit(det_id=det_id, r=r, t_ns=t_ns, L=L, material=material, extras=extras)

def shaped_to_typed_events(
    shaped: List[Dict[str, Any]],
    *,
    default_material: str = "UNK",
    order_time: bool = True,
) -> List[Union[NeutronEvent, GammaEvent]]:
    """
    Convert shaped dict events (from shapers.shape_events_for_cones) into typed events.
    If order_time=True, return .ordered() copies to enforce t_ns increasing order.
    """
    typed: List[Union[NeutronEvent, GammaEvent]] = []
    for ev in shaped:
        et = ev.get("event_type")
        # each entry may already be a Hit or a dict; _hit_from_canon handles both
        hh = [_hit_from_canon(h, material_default=default_material) for h in ev.get("hits", [])]
        meta = dict(ev.get("meta", {}))

        if et == "n":
            assert len(hh) == 2, "Neutron shaped event must have exactly 2 hits"
            obj = NeutronEvent(h1=hh[0], h2=hh[1], meta=meta)
            typed.append(obj.ordered() if order_time else obj)
        elif et == "g":
            assert len(hh) == 3, "Gamma shaped event must have exactly 3 hits"
            obj = GammaEvent(h1=hh[0], h2=hh[1], h3=hh[2], meta=meta)
            typed.append(obj.ordered() if order_time else obj)
        else:
            # Unknown types are ignored for cone building; could log/collect later.
            continue

    return typed
