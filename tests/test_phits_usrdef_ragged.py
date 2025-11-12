import os
from pathlib import Path
import math

import numpy as np
import h5py
import pytest

from ngimager.io.adapters import parse_phits_usrdef_short
from ngimager.io.lm_store import write_lm_ragged


# --- Helpers -----------------------------------------------------------------

EXAMPLE_DIR = Path("examples/imaging_datasets/PHITS_simple_ng_source")
USRDEF_PATH = EXAMPLE_DIR / "usrdef.out"

pytestmark = pytest.mark.skipif(
    not USRDEF_PATH.exists(),
    reason=f"Example usrdef file not found at {USRDEF_PATH}",
)


def _is_finite_number(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


# --- Tests -------------------------------------------------------------------

def test_parse_phits_usrdef_short_basic():
    events = parse_phits_usrdef_short(USRDEF_PATH)
    # Basic presence
    assert isinstance(events, list) and len(events) > 0

    # Spot-check structure of first few events
    max_check = min(5, len(events))
    for ev in events[:max_check]:
        # Required event keys
        for k in ("event_type", "iomp", "batch", "history", "no", "name", "hits"):
            assert k in ev

        # event_type mapped
        assert ev["event_type"] in ("n", "g")

        # hits present and variable-length allowed
        hits = ev["hits"]
        assert isinstance(hits, list) and len(hits) >= (2 if ev["event_type"] == "n" else 3)

        # spot-check first hit numeric fields are finite
        h0 = hits[0]
        for k in ("Edep_MeV", "x_cm", "y_cm", "z_cm", "t_ns"):
            assert k in h0 and _is_finite_number(h0[k])

        # region is int-like
        assert "reg" in h0 and isinstance(h0["reg"], int)


def test_write_lm_ragged_roundtrip(tmp_path: Path):
    events = parse_phits_usrdef_short(USRDEF_PATH)
    assert len(events) > 0

    out = tmp_path / "lm_ragged.h5"
    with h5py.File(out, "w") as f:
        write_lm_ragged(f, events)

    # Re-open and verify datasets
    with h5py.File(out, "r") as f:
        assert "/lm/hits/event_ptr" in f
        g_hits = f["/lm/hits"]
        g_ev = f["/lm/events"]

        event_ptr = g_hits["event_ptr"][...]
        x = g_hits["x_cm"][...]
        y = g_hits["y_cm"][...]
        z = g_hits["z_cm"][...]
        t = g_hits["t_ns"][...]
        e = g_hits["Edep_MeV"][...]
        reg = g_hits["reg"][...]

        # Basic shape checks
        N = len(events)
        assert event_ptr.shape == (N + 1,)
        M = int(event_ptr[-1])  # total hits
        for arr in (x, y, z, t, e, reg):
            assert arr.shape == (M,)

        # event_ptr monotonic and starts at 0
        assert event_ptr[0] == 0
        assert np.all(event_ptr[1:] >= event_ptr[:-1])

        # Event-level arrays
        etype = g_ev["event_type"][...]
        iomp  = g_ev["iomp"][...]
        batch = g_ev["batch"][...]
        hist  = g_ev["history"][...]
        eno   = g_ev["no"][...]
        name  = g_ev["name"][...]

        assert etype.shape == (N,)
        assert iomp.shape == (N,)
        assert batch.shape == (N,)
        assert hist.shape == (N,)
        assert eno.shape == (N,)
        assert name.shape == (N,)

        # Reconstruct per-event hit counts from CSR and compare against source
        counts_from_ptr = np.diff(event_ptr)
        counts_from_src = np.array([len(ev["hits"]) for ev in events], dtype=np.int64)
        np.testing.assert_array_equal(counts_from_ptr, counts_from_src)

        # Spot check a few values for alignment:
        # Take first event with >= 2 hits, compare first hit values
        idx_event = int(np.argmax(counts_from_src >= 2))
        start, end = event_ptr[idx_event], event_ptr[idx_event + 1]
        assert end > start
        # Choose first hit position in flat arrays
        w = int(start)
        ev0 = events[idx_event]
        h0 = ev0["hits"][0]
        assert np.isclose(x[w], h0["x_cm"])
        assert np.isclose(y[w], h0["y_cm"])
        assert np.isclose(z[w], h0["z_cm"])
        assert np.isclose(t[w], h0["t_ns"])
        assert np.isclose(e[w], h0["Edep_MeV"])
        assert int(reg[w]) == int(h0["reg"])

        # event_type encoding: 0=unknown, 1=n, 2=g, 3=mixed
        # Ensure mapping matches the original event_type for the sampled event
        expect_code = 1 if ev0["event_type"] == "n" else 2
        assert int(etype[idx_event]) == expect_code
