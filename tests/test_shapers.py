from pathlib import Path
import numpy as np
import pytest

from ngimager.io.adapters import parse_phits_usrdef_short
from ngimager.filters.shapers import shape_events_for_cones, ShapeConfig

EXAMPLE = Path("examples/imaging_datasets/PHITS_simple_ng_source/usrdef.out")
pytestmark = pytest.mark.skipif(not EXAMPLE.exists(), reason="example usrdef missing")

def test_shape_time_asc_neutron_gamma_default():
    raw = parse_phits_usrdef_short(EXAMPLE)
    shaped, diag = shape_events_for_cones(raw, ShapeConfig(neutron_policy="time_asc", gamma_policy="time_asc"))
    assert diag.total_events == len(raw)
    # We expect some shaped outputs (dataset has both ne & ge)
    assert len(shaped) > 0
    # All shaped items must have correct multiplicity
    for ev in shaped:
        if ev["event_type"] == "n":
            assert len(ev["hits"]) == 2
        elif ev["event_type"] == "g":
            assert len(ev["hits"]) == 3

def test_shape_all_combinations_cap():
    raw = parse_phits_usrdef_short(EXAMPLE)
    # This policy can explode combos; ensure cap holds
    cfg = ShapeConfig(neutron_policy="all_combinations", gamma_policy="all_combinations", max_combinations=10)
    shaped, _ = shape_events_for_cones(raw, cfg)
    # For any event with >=2 or >=3 hits, we won't exceed 10 combos per event/species
    # Do a loose upper bound check
    assert len(shaped) <= len(raw) * 10
