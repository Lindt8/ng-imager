# ng-imager

**ng-imager** is a modular Python toolkit for reconstructing **neutron** and **gamma** images from multi-element NOVO detector data.

It replaces a monolithic legacy script with a clean, restartable pipeline:

- hits → events → cones → images

and provides:

- Explicit separation of **physics**, **geometry**, **imaging**, **configuration**, and **I/O**
- Flexible energy strategies (ELUT, ToF, fixed-energy, direct Edep)
- Unified neutron + gamma imaging, including proton vs carbon recoil handling
- Structured filters at the hit, event, and cone level, with detailed counters
- HDF5 output that fully records the configuration, priors, counters, and list-mode data

---

## Names and Packages

- **Repository**: `ng-imager`  
  GitHub: https://github.com/Lindt8/ng-imager

- **Python package**: `ngimager`  
  Imported as:

      import ngimager

- **Planned PyPI package**: `ngimager`  
  (Eventually installable via `pip install ngimager`.)

Throughout the docs:

- “**ng-imager**” refers to the project as a whole.
- “`ngimager`” refers to the Python package and APIs.

---

## What the Pipeline Does

At a high level, the ng-imager pipeline:

1. **Reads raw data** via an adapter
   - PHITS usrdef text tallies
   - (Planned / developing) ROOT data, ng-imager HDF5 restarts

2. **Builds hits and typed events**
   - Hit-level filters remove obviously unphysical or irrelevant hits.
   - Shaping turns raw events into neutron 2-hit and gamma 3-hit candidates.
   - Event-level filters (ToF windows, light thresholds) refine the sample.

3. **Constructs cones**
   - Neutron cones with explicit proton vs carbon recoil hypotheses
   - Gamma cones via Compton kinematics for 3-hit events
   - Prior-based selection (and filtering) using Δθ = |φ − θ|

4. **Reconstructs images**
   - Simple back-projection (SBP) onto a user-defined imaging plane
   - Optional list-mode output mapping cones ↔ pixels ↔ events ↔ hits

5. **Writes a self-contained HDF5 file**
   - Full config snapshot
   - Plane and grid geometry
   - Per-hit and per-event physics
   - Per-cone geometry, classification, and kinematics
   - Summed neutron, gamma, and combined images
   - Counters and traces linking each stage

---

## Getting Started

If you’re new to ng-imager, start here:

1. **Quickstart**

   A step-by-step walkthrough using the example PHITS configuration:

       docs/quickstart.md

2. **Configuration (TOML)**

   Detailed documentation for all fields in the TOML config:

       docs/config.md

3. **HDF5 Output Layout**

   How to interpret the ng-imager HDF5 file, including list-mode datasets:

       docs/hdf5.md

4. **Architecture / Design Primer**

   High-level design document that explains how the modules fit together:

       docs/architecture.md

5. **API Reference**

   Auto-generated reference for the `ngimager` Python package:

       docs/api/index.md

---

## Current Status

- Neutron imaging: functional and validated against the legacy NOVO pipeline.
- Gamma imaging: functional and validated on legacy-formatted model data.
- Proton vs carbon neutron recoil handling: implemented and recorded in HDF5.
- Event, hit, and cone filters: implemented with counters and HDF5 export.
- Fast and list modes: wired through configuration and HDF5 output.

The codebase is still evolving, but the overall architecture and I/O formats are intended to be stable enough for experimental and analysis usage.

---

## Contributing

If you’d like to:

- Add new adapters (e.g. other simulation formats),
- Extend the imaging methods,
- Or improve documentation and examples,

please open an issue or pull request on GitHub:

    https://github.com/Lindt8/ng-imager

The goal is to keep ng-imager modular, well-documented, and physically transparent, so that both new and existing NOVO analyses can be expressed in a clean, modern Python codebase.
