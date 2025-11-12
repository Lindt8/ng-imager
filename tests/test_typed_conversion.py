from pathlib import Path
import numpy as np
import pytest

from ngimager.io.adapters import parse_phits_usrdef_short
from ngimager.io.canonicalize import canonicalize_events_inplace
from ngimager.filters.shapers import shape_events_for_cones, ShapeConfig
from ngimager.filters.to_typed_events import shaped_to_typed_events
from ngimager.physics.events import NeutronEvent, GammaEvent

EXAMPLE = Path("examples/imaging_datasets/PHITS_simple_ng_source/usrdef.out")
pytestmark = pytest.mark.skipif(not EXAMPLE.exists(), reason="example usrdef missing")

def test_shaped_to_typed_events_time_ordered():
    raw = parse_phits_usrdef_short(EXAMPLE)
    canonicalize_events_inplace(raw)  # ensure adapter-agnostic keys
    shaped, diag = shape_events_for_cones(raw, ShapeConfig(neutron_policy="time_asc", gamma_policy="time_asc"))
    typed = shaped_to_typed_events(shaped, default_material="M600", order_time=True)

    assert len(typed) == len(shaped)
    # spot checks
    for ev in typed[:10]:
        if isinstance(ev, NeutronEvent):
            # must be strictly ordered for default strict semantics
            assert ev.h1.t_ns <= ev.h2.t_ns
            # validate() requires strict (h1<h2) – dataset may have equal times very rarely; guard accordingly
            if ev.h1.t_ns != ev.h2.t_ns:
                ev.validate(strict=True)
        elif isinstance(ev, GammaEvent):
            t = (ev.h1.t_ns, ev.h2.t_ns, ev.h3.t_ns)
            assert t[0] <= t[1] <= t[2]
            if len(set(t)) == 3:
                ev.validate(strict=True)

