from pathlib import Path
import h5py

from ngimager.io.adapters import parse_phits_usrdef_short
from ngimager.io.lm_store import write_lm_ragged

USRDEF = Path("../../../examples/imaging_datasets/PHITS_simple_ng_source/usrdef.out")
#OUT = Path("phits_ragged_out.h5")
OUT = USRDEF.parent.joinpath("usrdef_ragged_out.h5")

def main():
    evs = parse_phits_usrdef_short(USRDEF)
    with h5py.File(OUT, "w") as f:
        write_lm_ragged(f, evs)
    print(f"Wrote {OUT} with {len(evs)} events")

if __name__ == "__main__":
    main()
