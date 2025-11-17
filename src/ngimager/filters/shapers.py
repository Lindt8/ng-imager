# src/ngimager/filters/shapers.py
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Literal, Tuple, Any

from ngimager.physics.hits import Hit

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


# --- Generic accessors so shapers work with canonical Hit objects ---
def _hit_t_ns(h: Hit) -> float:
    """
    Time key for sorting hits in ascending order.
    Assumes canonical physics.hits.Hit with attribute t_ns.
    """
    return float(h.t_ns)


def _hit_energy_desc_key(h: Hit) -> float:
    """
    Sorting key for 'energy_desc' policy.

    For now we treat Hit.L as the “light-like” energy surrogate.
    If L is not set or zero, this simply returns 0.0.
    """
    # L is always present on Hit; treat None/0 as 0.0
    return float(getattr(h, "L", 0.0) or 0.0)

def _hit_species(h: Hit) -> str | None:
    """
    Map a Hit.type string onto 'n' or 'g'.

    Expected conventions:
      - Hit.type starting with 'n' => neutron
      - Hit.type starting with 'g' => gamma
      - 'UNK' or anything else => None (ignored for shaping)
    """
    s = str(getattr(h, "type", "") or "").strip().lower()
    if not s:
        return None

    if s.startswith("n"):
        return "n"
    if s.startswith("g"):
        return "g"
    return None

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

def _select_hits(
    hits: List[Hit],
    k: int,
    policy: Policy,
    max_combinations: int,
    diag: ShapeDiagnostics,
    species: str,
) -> List[List[Hit]]:
    """
    Select hits for one species in a single raw event according to policy.

    Returns a list of hit-lists (each will become one shaped event).
    """
    if len(hits) < k:
        # Not enough hits to form even one event
        if species == "n":
            diag.dropped_neutron += 1
            diag.inc("neutron_insufficient_hits")
        elif species == "g":
            diag.dropped_gamma += 1
            diag.inc("gamma_insufficient_hits")
        return []

    if policy == "time_asc":
        ordered = sorted(hits, key=_hit_t_ns)
        return [ordered[:k]]

    if policy == "energy_desc":
        ordered = sorted(hits, key=_hit_energy_desc_key, reverse=True)
        return [ordered[:k]]

    # all_combinations
    combos_iter = combinations(hits, k)
    selected: List[List[Any]] = []
    for i, combo in enumerate(combos_iter):
        if i >= max_combinations:
            diag.policy_hits_limit += 1
            diag.inc(f"{species}_combinations_capped")
            break
        selected.append(list(combo))
    return selected

def shape_events_for_cones(
    raw_events: Iterable[Dict[str, Any]],
    cfg: ShapeConfig,
) -> Tuple[List[Dict[str, Any]], ShapeDiagnostics]:
    """
    Shape raw coincidence windows into candidate 2-hit neutron and 3-hit gamma events.

    Inputs:
      raw_events: iterable of dicts, each with at least a 'hits' key.
                  'hits' must be a sequence of canonical physics.hits.Hit objects.
                  (Adapters and/or canonicalization are responsible for constructing Hits.)
      cfg: ShapeConfig controlling policies and caps.

    Outputs:
      shaped: list of shaped event dicts with:
                - 'event_type': 'n' or 'g'
                - 'hits': list of hits (same objects as input)
                - plus any original metadata keys preserved
      diag:   ShapeDiagnostics with counters and reasons.
    """
    diag = ShapeDiagnostics()
    shaped: List[Dict[str, Any]] = []

    for ev in raw_events:
        diag.total_events += 1
        hits = list(ev.get("hits", []))
        if not hits:
            diag.inc("no_hits")
            continue

        # Partition by species using Hit.type when available
        n_hits: List[Any] = []
        g_hits: List[Any] = []
        for h in hits:
            sp = _hit_species(h)
            if sp == "n":
                n_hits.append(h)
            elif sp == "g":
                g_hits.append(h)
            else:
                # Unknown species; currently ignore for shaping
                diag.inc("unknown_species_hit")

        if n_hits:
            diag.neutron_in += 1
            selected_n = _select_hits(
                n_hits, k=2,
                policy=cfg.neutron_policy,
                max_combinations=cfg.max_combinations,
                diag=diag,
                species="n",
            )
            for hs in selected_n:
                ev_out = {k: v for k, v in ev.items() if k not in ("hits", "event_type")}
                ev_out["event_type"] = "n"
                ev_out["hits"] = hs
                shaped.append(ev_out)
                diag.shaped_neutron += 1

        if g_hits:
            diag.gamma_in += 1
            selected_g = _select_hits(
                g_hits, k=3,
                policy=cfg.gamma_policy,
                max_combinations=cfg.max_combinations,
                diag=diag,
                species="g",
            )
            for hs in selected_g:
                ev_out = {k: v for k, v in ev.items() if k not in ("hits", "event_type")}
                ev_out["event_type"] = "g"
                ev_out["hits"] = hs
                shaped.append(ev_out)
                diag.shaped_gamma += 1

        if not n_hits and not g_hits:
            diag.inc("no_usable_species")

    return shaped, diag
