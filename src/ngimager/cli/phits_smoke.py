# src/ngimager/cli/phits_smoke.py
'''
A small CLI that runs: 
parse → Hit → shape → typed → (try) cones+SBP → write HDF5 ragged + PNG if imaging is reachable. 
It won’t touch core.run_pipeline. If cones/SBP aren’t available or signatures differ, 
it still writes LM ragged and prints diagnostics.
'''
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Union

import h5py

from ngimager.io.adapters import from_phits_usrdef
from ngimager.filters.shapers import shape_events_for_cones, ShapeConfig
from ngimager.filters.to_typed_events import shaped_to_typed_events
from ngimager.io.lm_store import write_lm_ragged
from ngimager.physics.events import NeutronEvent, GammaEvent

def _try_build_cones_and_sbp(typed: List[Union[NeutronEvent, GammaEvent]]):
    """
    Best-effort: try to import cone builders and SBP and return (summed_img or None, diagnostics str).
    This avoids breaking if APIs differ; you can firm up once we lock signatures.
    """
    try:
        from ngimager.physics.cones import build_cone_from_neutron, build_cone_from_gamma
        from ngimager.imaging.sbp import ReconResult  # type: ignore
        from ngimager.imaging.sbp import rasterize_cones  # hypothetical; adjust to your API
    except Exception as e:
        return None, f"[smoke] Imaging skipped: imports failed ({e})"

    cones = []
    for ev in typed:
        try:
            if isinstance(ev, NeutronEvent):
                cones.append(build_cone_from_neutron(ev))
            elif isinstance(ev, GammaEvent):
                cones.append(build_cone_from_gamma(ev))
        except Exception:
            # skip malformed events silently in smoke mode
            continue

    if not cones:
        return None, "[smoke] No cones built from typed events"

    try:
        # Adjust to your actual API: plane/grid selection, etc.
        recon: ReconResult = rasterize_cones(cones)  # noqa
        return recon.summed, "[smoke] SBP reconstruction succeeded"
    except Exception as e:
        return None, f"[smoke] SBP failed ({e})"

def main():
    ap = argparse.ArgumentParser(description="PHITS usrdef smoke pipeline")
    ap.add_argument("usrdef", type=Path, help="Path to PHITS usrdef.out (short format)")
    ap.add_argument("-o", "--out", type=Path, default=Path("phits_smoke.h5"), help="Output HDF5")
    ap.add_argument("--policy-n", default="time_asc", choices=["time_asc","energy_desc","all_combinations"])
    ap.add_argument("--policy-g", default="time_asc", choices=["time_asc","energy_desc","all_combinations"])
    ap.add_argument("--max-combos", type=int, default=5000)
    args = ap.parse_args()

    # Ingest → Hit
    events = from_phits_usrdef(args.usrdef)
    print(f"[smoke] Parsed {len(events)} histories with variable multiplicity")

    # Shape → typed
    shaped, diag = shape_events_for_cones(
        events,
        ShapeConfig(neutron_policy=args.policy_n, gamma_policy=args.policy_g, max_combinations=args.max_combos),
    )
    print(f"[smoke] Shaped events: N={len(shaped)} (n_in={diag.neutron_in}, g_in={diag.gamma_in}, "
          f"n_shaped={diag.shaped_neutron}, g_shaped={diag.shaped_gamma}, dropped={diag.dropped_neutron+diag.dropped_gamma})")

    typed = shaped_to_typed_events(shaped, default_material="UNK", order_time=True)
    print(f"[smoke] Typed events: {len(typed)}")

    # Write LM ragged no matter what
    with h5py.File(args.out, "w") as f:
        write_lm_ragged(f, events)
    print(f"[smoke] Wrote ragged LM to {args.out}")

    # Attempt imaging
    img, msg = _try_build_cones_and_sbp(typed)
    print(msg)
    if img is not None:
        try:
            # Store a simple image dataset for quick inspection
            with h5py.File(args.out, "a") as f:
                if "/images/summed/n" in f:
                    del f["/images/summed/n"]
                f.create_dataset("/images/summed/n", data=img)
            # Optional PNG via your existing helper
            try:
                from ngimager.vis.hdf import save_summed_png
                png = Path(args.out).with_suffix(".png")
                save_summed_png(str(args.out), dataset="/images/summed/n", out_path=str(png))
                print(f"[smoke] PNG saved to {png}")
            except Exception as e:
                print(f"[smoke] PNG export skipped: {e}")
        except Exception as e:
            print(f"[smoke] Could not save image to HDF5: {e}")

if __name__ == "__main__":
    main()
