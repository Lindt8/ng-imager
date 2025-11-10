# API Reference

This page documents the public Python API of **ng-imager**.

This section is automatically generated from the Python docstrings using [mkdocstrings](https://mkdocstrings.github.io/). It covers the main simulation, imaging, and reconstruction modules that form the ng-imager pipeline.
  
The modules are grouped roughly by how you use them in an imaging workflow:

- **Pipelines**: high-level entry points that wire everything together.
- **Physics & geometry**: event / cone construction and projection math.
- **I/O & configuration**: reading event data, HDF5 storage, configuration.
- **CLI & visualization**: the command-line app and simple plotting utilities.
- **Simulation & tools**: synthetic data generation and LUT utilities.

---

## Pipelines

High-level orchestration for running NOVO imaging in different modes.

### `ngimager.pipelines.fastmode`

Fast, aggressively-cut imaging pipeline for quick “beam-time” feedback.

::: ngimager.pipelines.fastmode

### `ngimager.pipelines.listmode`

Full list-mode pipeline that preserves per-cone information for post-analysis.

::: ngimager.pipelines.listmode

---

## Physics: hits, events, kinematics, and cones

Building blocks that turn detector-level hits into reconstructed event cones.

### `ngimager.physics.hits`

Hit-level representations (positions, times, deposited light/energy) and helpers.

::: ngimager.physics.hits

### `ngimager.physics.events`

Composite neutron and gamma events built from multiple hits.

::: ngimager.physics.events

### `ngimager.physics.kinematics`

Kinematic relationships for neutrons and gamma rays (e.g. scatter angles, ToF).

::: ngimager.physics.kinematics

### `ngimager.physics.cones`

Construction of event cones (vertex, axis, half-angle) from reconstructed events.

::: ngimager.physics.cones

### `ngimager.physics.energy_strategies`

Strategies for assigning energies to scatters (ELUT, ToF, fixed-energy, etc.).

::: ngimager.physics.energy_strategies

### `ngimager.physics.priors`

Source and geometry priors used to weight cones and regularize imaging.

::: ngimager.physics.priors

---

## Geometry & Imaging

Geometry primitives and the simple back-projection (SBP) imager.

### `ngimager.geometry.plane`

Imaging plane representation and coordinate transforms (u–v basis, etc.).

::: ngimager.geometry.plane

### `ngimager.imaging.sbp`

Simple back-projection implementation that projects cones onto an image plane.

::: ngimager.imaging.sbp

---

## I/O & Configuration

Adapters for raw data, list-mode storage, LUT loading, and config handling.

### `ngimager.io.adapters`

Adapters that convert external event formats (e.g. ROOT, PHITS-like) into ng-imager hits/events.

::: ngimager.io.adapters

### `ngimager.io.lm_store`

List-mode HDF5 storage layout and helpers for reading/writing cone datasets.

::: ngimager.io.lm_store

### `ngimager.io.lut`

Loading and interpolating light-response lookup tables (LUTs) for scintillators.

::: ngimager.io.lut

### `ngimager.config.schemas`

Pydantic schemas that define the TOML configuration structure.

::: ngimager.config.schemas

### `ngimager.config.load`

User-facing helpers for loading and validating configuration from TOML files.

::: ngimager.config.load

---

## CLI & Visualization

Command-line entry point and basic visualization utilities.

### `ngimager.cli.viz`

The `novo-viz` CLI application: entry point for running imaging from the shell.

::: ngimager.cli.viz

### `ngimager.vis.hdf`

Convenience functions for visualizing images stored in HDF5 output files.

::: ngimager.vis.hdf

---

## Simulation & Tools

Synthetic data generation and developer utilities.

### `ngimager.sim.synth`

Simple synthetic / toy generators useful for testing the imaging chain.

::: ngimager.sim.synth

### `ngimager.tools.bundle_repo`

Utility for snapshotting the repository (e.g. for embedding into an HDF5 file).

::: ngimager.tools.bundle_repo

### `ngimager.tools.generate_lut.NOVO_light_response_functions`

Light-response fitting and LUT generation for NOVO’s scintillators (M600, OGS).

::: ngimager.tools.generate_lut.NOVO_light_response_functions



---

## Light-response LUT tools

The `ngimager.tools.generate_lut` module contains functions for building, fitting, and using light-response lookup tables (LUTs) for NOVO scintillators.

::: ngimager.tools.generate_lut.NOVO_light_response_functions
    options:
      show_root_heading: true
      show_root_full_path: false
      members_order: source
      heading_level: 2

---

## Legacy Components

These modules are kept for reference and validation purposes but are not part of the main public API.

::: ngimager.legacy.expNOVO_imager_legacy
    options:
      show_root_full_path: false
      heading_level: 2

---

_TODO: Extend this page as the package matures to include the data I/O and analysis submodules, once their docstrings are finalized._


