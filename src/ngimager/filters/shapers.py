# src/ngimager/filters/shapers.py
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Literal, Tuple, Any

Policy = Literal["time_asc", "energy_desc", "all_combinations"]

@dataclass
class ShapeConfig:
    neutron_policy: Policy = "time_asc"
    gamma_policy: Policy = "time_asc"
    max_combinations: int = 5000  # safety for 'all_combinations'

@dataclass
class ShapeDiagnostics:
    total_events: int = 0
    neutron_in: int = 0
    gamma_in: int = 0
    shaped_neutron: int = 0
    shaped_gamma: int = 0
    dropped_neutron: int = 0
    dropped_gamma: int = 0
    policy_hits_limit: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)

    def inc(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


# --- Generic accessors so shapers work with either dict-hits or Hit objects ---
def _hit_t_ns(h: Any) -> float:
    # Hit object
    if hasattr(h, "t_ns"):
        return float(getattr(h, "t_ns"))
    # dict-like
    return float(h.get("t_ns", 0.0))

def _hit_energy_desc_key(h: Any) -> float:
    """
    Sorting key for 'energy_desc' policy:
      Prefer Hit.L (light-like), else dict['L'], else dict['Edep_MeV'], else 0.
    """
    # Hit object: L is defined
    if hasattr(h, "L"):
        return float(getattr(h, "L") or 0.0)
    # dict-like
    if "L" in h:
        return float(h["L"] or 0.0)
    if "Edep_MeV" in h:
        return float(h["Edep_MeV"] or 0.0)
    return 0.0


def _pair_indices(hits: List[Any], policy: Policy, max_combos: int) -> List[Tuple[int,int]]:
    n = len(hits)
    if n < 2:
        return []
    if policy == "time_asc":
        idx = sorted(range(n), key=lambda i: _hit_t_ns(hits[i]))
        return [(idx[0], idx[1])]
    if policy == "energy_desc":
        idx = sorted(range(n), key=lambda i: _hit_energy_desc_key(hits[i]), reverse=True)
        return [(idx[0], idx[1])]
    # all combinations
    out: List[Tuple[int,int]] = []
    for a,b in combinations(range(n), 2):
        out.append((a,b))
        if len(out) >= max_combos:
            break
    return out

def _triple_indices(hits: List[Any], policy: Policy, max_combos: int) -> List[Tuple[int,int,int]]:
    n = len(hits)
    if n < 3:
        return []
    if policy == "time_asc":
        idx = sorted(range(n), key=lambda i: _hit_t_ns(hits[i]))
        return [(idx[0], idx[1], idx[2])]
    if policy == "energy_desc":
        idx = sorted(range(n), key=lambda i: _hit_energy_desc_key(hits[i]), reverse=True)
        return [(idx[0], idx[1], idx[2])]
    # all combinations
    out: List[Tuple[int,int,int]] = []
    for a,b,c in combinations(range(n), 3):
        out.append((a,b,c))
        if len(out) >= max_combos:
            break
    return out

def shape_events_for_cones(
    phits_events: Iterable[dict],
    cfg: ShapeConfig | None = None,
) -> Tuple[List[dict], ShapeDiagnostics]:
    """
    Convert variable-length PHITS events into fixed 2- or 3-hit shaped events for cone building.
    Each output item:
      - neutron: {"event_type":"n","hits":[h1,h2],"meta":{orig ev meta + indices}}
      - gamma  : {"event_type":"g","hits":[h1,h2,h3],"meta":{...}}
    No physics beyond selection policy is applied here.
    """
    if cfg is None:
        cfg = ShapeConfig()
    diag = ShapeDiagnostics()
    shaped: List[dict] = []

    for ev in phits_events:
        diag.total_events += 1
        et = ev.get("event_type", "")
        hits = list(ev.get("hits", []))  # may be dicts or Hit objects; handled by accessors
        if et == "n":
            diag.neutron_in += 1
            pairs = _pair_indices(hits, cfg.neutron_policy, cfg.max_combinations)
            if not pairs:
                diag.dropped_neutron += 1
                diag.inc("neutron_insufficient_hits")
                continue
            for (i,j) in pairs:
                shaped.append({
                    "event_type": "n",
                    "hits": [hits[i], hits[j]],
                    "meta": {
                        "source_meta": {k: ev.get(k) for k in ("iomp","batch","history","no","name")},
                        "selected_indices": (i,j),
                        "policy": cfg.neutron_policy,
                    },
                })
            diag.shaped_neutron += len(pairs)
        elif et == "g":
            diag.gamma_in += 1
            triples = _triple_indices(hits, cfg.gamma_policy, cfg.max_combinations)
            if not triples:
                diag.dropped_gamma += 1
                diag.inc("gamma_insufficient_hits")
                continue
            for (i,j,k) in triples:
                shaped.append({
                    "event_type": "g",
                    "hits": [hits[i], hits[j], hits[k]],
                    "meta": {
                        "source_meta": {k2: ev.get(k2) for k2 in ("iomp","batch","history","no","name")},
                        "selected_indices": (i,j,k),
                        "policy": cfg.gamma_policy,
                    },
                })
            diag.shaped_gamma += len(triples)
        else:
            diag.inc("unknown_event_type")
            # we purposely do not drop; could route elsewhere if needed

    return shaped, diag
