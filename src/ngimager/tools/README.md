# Extra Tools

The `generate_lut` directory contains code and data files for generating look-up tables for describing energy deposited as a function of light response detected.  It has its own README document.

The `phits_usrdef_2_legacy.py` script converts the PHITS output from [the multi-coincidence neutron and gamma-ray detection [T-Userdefined] tally](https://github.com/Lindt8/T-Userdefined/tree/main/multi-coincidence_ng) into the format expected by the [legacy imaging code](../../../legacy).

The `bundle_repo.py` script compiles all the text-based documents (within certain constraints) in this repository into a single text file.