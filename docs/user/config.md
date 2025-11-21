# Configuration: TOML Reference

This page describes the TOML configuration format used by **ngimager** (the internal Python package for the **ng-imager** project).

The examples below are based on:

    examples/configs/phits_usrdef_simple.toml

and reflect the **current, working** configuration schema used by the pipeline.

---

## 1. Top-Level Layout

At the highest level, a config file typically contains:

- [run] – global run controls (species, fast/list mode, diagnostics)
- [pipeline] – where to stop the pipeline (hits / events / cones / image)
- [io] – input / output paths and adapter options
- [detectors] – detector → material mapping
- [plane] – imaging plane geometry and grid
- [filters] – hit / event / cone filters (+ optional fast overrides)
- [energy] – energy strategy (ELUT, ToF, fixed, Edep)
- [prior] – spatial prior for source location
- [uncertainty] – smearing / uncertainty options
- [vis] – visualization / PNG export options

A minimal skeleton looks like:

    [run]
    ...

    [pipeline]
    ...

    [io]
    ...

    [detectors]
    ...

    [plane]
    ...

    [filters]
    ...

    [energy]
    ...

    [prior]
    ...

    [uncertainty]
    ...

    [vis]
    ...

---

## 2. [run] – Global Run Controls

    [run]
    # High-level data source context
    # One of: "cf252" | "dt" | "proton_center" | "phits"
    source_type = "phits"

    # Which species to process
    neutrons = true
    gammas   = true

    # Behavioral toggles
    fast = false        # enable fast-mode cuts / plane overrides
    list = true         # enable list-mode image output

    # Performance / execution
    workers     = 0     # 0 = single-process; "auto" or >0 in newer configs
    chunk_cones = "auto"
    jit         = false
    progress    = true

    # Diagnostics
    diagnostics_level = 2   # 0=off, 1=minimal, 2=verbose

    # Limits
    max_cones = 50000       # hard cap on number of cones to build / image

Notes:

- `fast = true` activates fast-mode overrides defined under `[filters.fast]`.
- `list = true` asks the recon to keep per-cone pixel indices and to emit list-mode datasets under `/lm/...` in the HDF5 output.
- `workers = 0` forces a single-process SBP path (nice for debugging); `"auto"` or an integer > 0 uses multi-process SBP.

---

## 3. [pipeline] – How Far to Run

    [pipeline]
    # Where to stop the pipeline:
    #   "hits"   – stop after hit construction / hit filters
    #   "events" – stop after shaped/typed events
    #   "cones"  – stop after cone construction
    #   "image"  – full pipeline (cones → image)
    until = "image"

At the moment, most runs use `until = "image"`. Future work will make it easy to restart from intermediate stages using ng-imager HDF5 output.

---

## 4. [io] – Input / Output and Adapters

    [io]
    input_format = "phits_usrdef"  # "phits_usrdef" | "root_novo_ddaq" | "hdf5_ngimager"
    input_path   = "examples/imaging_datasets/PHITS_simple_ng_source/usrdef.out"
    output_path  = "examples/imaging_datasets/PHITS_simple_ng_source/usrdef_out.h5"

    # If true, overwrite an existing output file; if false, error out if the
    # file already exists.
    hdf5_overwrite = true

### 4.1. [io.adapter] – PHITS usrdef Adapter

For the PHITS user-defined tally case:

    [io.adapter]
    type  = "phits"   # selects the PHITS-style adapter
    style = "usrdef"  # indicates the custom usrdef text layout

    # Units / conventions
    unit_pos_is_mm       = false   # false → positions already in cm
    time_units           = "ns"    # "ns" recommended
    require_gamma_triples = true   # enforce 3-hit gamma events for Compton cones

Other adapters may accept different sub-keys under `[io.adapter]`; for now, the documented / tested case is the `"phits"` + `"usrdef"` combination.

---

## 5. [detectors] – Material Mapping

Detector configuration is mainly used to assign materials to regions / bars, which then flow into hit construction and LUT lookup.

    [detectors]
    default_material = "OGS"

    [detectors.material_map]
    200 = "OGS"
    210 = "M600"
    220 = "OGS"
    230 = "M600"
    240 = "OGS"

Keys:

- `default_material`: fallback material name for any detector/bar ID not explicitly listed.
- `material_map`: maps integer detector IDs (regions) to material names (strings). These names must match the keys used under `[energy].lut_paths`.

---

## 6. [plane] – Imaging Plane Geometry

The imaging plane is specified in world coordinates:

    [plane]
    origin = [0.0, 15.0, 0.0]   # point on the plane (cm)
    normal = [0.0, 1.0, 0.0]    # plane normal (unit-ish vector)

    # In-plane axes (u and v). Must be linearly independent of each other and of the normal.
    u_axis = [1.0, 0.0, 0.0]
    v_axis = [0.0, 0.0, 1.0]

    # u / v extents and sampling (cm)
    u_min = -20.0
    u_max =  20.0
    du    =   0.1

    v_min = -20.0
    v_max =  20.0
    dv    =   0.1

Notes:

- Internally, the plane object normalizes `normal`, `u_axis`, and `v_axis` and computes `nu`, `nv` from `(u_max - u_min)/du` and `(v_max - v_min)/dv`.
- All coordinates are in **cm**.

---

## 7. [filters] – Hit, Event, and Cone Filters

Filters are split into three conceptual levels, with neutron/gamma-specific overrides:

- `[filters.hits]` and `[filters.hits.neutron]` / `[filters.hits.gamma]`
- `[filters.events]` and `[filters.events.neutron]` / `[filters.events.gamma]`
- `[filters.cones]` and `[filters.cones.neutron]` / `[filters.cones.gamma]`
- `[filters.fast]` – optional fast-mode overrides

The guiding principle is:

- The universal section (`[filters.X]`) defines global defaults.
- Species-specific sections (`[filters.X.neutron]`, `[filters.X.gamma]`) override only the keys they specify.

### 7.1. Hit-Level Filters

    [filters.hits]
    # Universal hit cuts applied before event shaping
    min_light_MeVee = 0.0
    max_light_MeVee = 200.0

    # Optional bar / material inclusion / exclusion
    bars_include       = []    # e.g. [200, 210, 220]
    bars_exclude       = []    # e.g. [230]
    materials_include  = []    # e.g. ["M600", "OGS"]
    materials_exclude  = []    # e.g. ["DEAD_BAR_MATERIAL"]

    [filters.hits.neutron]
    # optional overrides / additions for neutron hits
    # (empty in the example config)

    [filters.hits.gamma]
    # optional overrides / additions for gamma hits
    # (empty in the example config)

The hit filters are applied uniformly to all hits in a raw event before any event-level shaping or typing occurs.

### 7.2. Event-Level Filters

    [filters.events]
    # Universal event cuts (applied after shaping into typed NeutronEvent/GammaEvent)
    tof_window_ns = [0.0, 30.0]

    [filters.events.neutron]
    # Override only what you need; unspecified keys fall back to [filters.events]
    tof_window_ns = [0.0, 30.0]
    min_L1_MeVee  = 0.1   # minimum first-scatter light for neutron events
    min_L2_MeVee  = 0.1   # minimum second-scatter light for neutron events

    [filters.events.gamma]
    tof_window_ns = [0.0, 10.0]
    min_L_any_MeVee = 0.05  # minimum light on any of the gamma hits

Typical uses:

- Narrow `tof_window_ns` for neutron/gamma separately.
- Require a minimum first/second scatter light for neutron 2-hit events.
- Require at least one sufficiently bright gamma hit for 3-hit gamma events.

Event filters operate on **typed events** (NeutronEvent / GammaEvent) and feed into cone building.

### 7.3. Cone-Level Filters

    [filters.cones]
    # Universal cone cuts
    max_delta_theta_deg     = 90.0   # Δθ = |φ - θ|, prior-consistency limit
    max_incident_energy_MeV = 250.0  # reject events with unphysically high incident energy

    [filters.cones.neutron]
    # e.g., neutron-specific overrides (empty in the example)
    # max_delta_theta_deg     = 5.0
    # max_incident_energy_MeV = 30.0

    [filters.cones.gamma]
    # e.g., gamma-specific overrides (empty in the example)
    # max_delta_theta_deg     = 8.0
    # max_incident_energy_MeV = 10.0

The cone filters operate on:

- The **selected** neutron cone (after the proton vs carbon recoil decision)
- The **selected** gamma cone (after permutation + prior selection)

`max_delta_theta_deg` uses the same prior-based scoring as cone selection (Δθ = |φ − θ|).  
`max_incident_energy_MeV` uses the incident energy stored on each cone (`incident_energy_MeV`), which is computed by the neutron/gamma kinematics code and passed through from the cone builders.

### 7.4. [fast] – Fast-Mode Overrides

Fast mode (`run.fast = true`) activates a small set of more aggressive defaults. The exact details depend on the code, but conceptually:

    [fast]
    # More aggressive light thresholds
    min_L1_MeVee   = 0.8   # MeVee, first neutron scatter
    min_L2_MeVee   = 0.4   # MeVee, second neutron scatter
    min_L_any_MeVee = 0.2  # MeVee, any gamma deposit considered
    
    # Stop after this many cones have been built and imaged
    max_cones = 20000
    
    # Coarsen the imaging plane by this integer factor:
    # du' = du * plane_downsample, dv' = dv * plane_downsample
    plane_downsample = 2

Any fast-mode cut that is defined will override the corresponding normal cut when `run.fast = true`. The exact field names and behavior are kept intentionally minimal so that the fast-mode logic does not diverge too far from the main configuration.

---

## 8. [energy] – Energy Strategy

    [energy]
    # One of: "ELUT" | "ToF" | "FixedEn" | "Edep"
    strategy = "Edep"

    # If strategy = "FixedEn", this specifies the fixed incident neutron energy:
    fixed_En_MeV = 14.1

    # Inversion LUTs for light → energy (used when strategy = "ELUT")
    [energy.lut_paths.OGS]
    proton = "data/lut/OGS/lut_OGS_proton_Birks.npz"
    carbon = "data/lut/OGS/lut_OGS_carbon_Birks.npz"

    [energy.lut_paths.M600]
    proton = "data/lut/M600/lut_M600_proton_Birks.npz"
    carbon = "data/lut/M600/lut_M600_carbon_Birks.npz"

    # When true, neutron cone construction forces proton recoils for kinematics
    force_proton_recoils = false

Strategies:

- `"ELUT"` – Use per-material, per-species lookup tables to invert light (MeVee) → deposited energy Edep (MeV).
- `"ToF"` – Use time-of-flight between hits to estimate neutron energy (placeholder, mainly experimental).
- `"FixedEn"` – Assume a fixed incident neutron energy (e.g. 14.1 MeV DT source); used where appropriate.
- `"Edep"` – Use deposited energy directly from the adapter (`Hit.L` in MeV).

For neutron cone building, the kinematics are always routed through `neutron_theta_from_hits`, with the energy strategy providing the first-scatter deposited energy.

---

## 9. [prior] – Spatial Priors

    [prior]
    # One of: "point" | "line"
    type = "point"

    # For point prior:
    point = [0.0, 0.0, -500.0]

    # For line prior (example):
    # [prior.line]
    # p0 = [x0, y0, z0]
    # p1 = [x1, y1, z1]

    # Prior strength (dimensionless weighting)
    strength = 1.0

The prior is used for:

- Selecting between proton vs carbon neutron recoil hypotheses.
- Selecting the best permutation for gamma events.
- Cone-level Δθ filtering (`max_delta_theta_deg`).

When no explicit prior is given, the imaging plane center is used as an implicit prior target.

---

## 10. [uncertainty] – Smearing and Uncertainties

    [uncertainty]
    enabled = false
    smearing = "thicken"          # "thicken" | "weighted"
    sigma_doi_cm        = 0.35
    sigma_transverse_cm = 0.346
    sigma_time_ns       = 0.5
    use_lut_bands       = false   # if true, use LUT-provided bands where available

This block is currently mostly reserved for future work. In the current code, the SBP reconstruction runs in a nominal, non-smeared mode unless uncertainty support is explicitly wired in.

---

## 11. [vis] – Visualization

    [vis]
    export_png_on_write    = true
    species                = ["n", "g", "all"]
    center_on_plane_center = true
    flip_vertical          = true
    axis_units             = "cm"          # "cm" or "mm"
    cmap                   = "cividis"
    filename_pattern       = "{species}_{stem}.{ext}"
    extra_formats          = []

    # Advanced / legacy option:
    # summed_dataset       = "/images/summed/n"

Fields:

- `export_png_on_write` – when true, the pipeline automatically renders image files from the reconstructed HDF5 output after stage 4 completes.

- `species` – list of which summed images to render from `/images/summed`.
  Allowed values:

  - `"n"` – neutron-only image (`/images/summed/n`)
  - `"g"` – gamma-only image (`/images/summed/g`)
  - `"all"` – combined neutron + gamma image (`/images/summed/all`, written only when both species are present)

- `center_on_plane_center` – if true, the u–v axes are shifted so that `(0, 0)` is at the imaging plane center. If false, the axes use the raw `grid.u_min` / `grid.v_min` coordinates stored in the HDF5 metadata.

- `flip_vertical` – flips the plotted image vertically relative to the natural v-axis orientation. This is mainly useful for visual comparison with legacy images.

- `axis_units` – units used for axis labels: `"cm"` (native) or `"mm"`. The underlying HDF5 metadata always stores geometry in centimeters; `"mm"` simply rescales labels by a factor of 10.

- `cmap` – Matplotlib colormap name to use for the image (e.g. `"cividis"`).

- `filename_pattern` – pattern for automatically generated file names. It is a Python `str.format` template and may reference:

  - `{stem}` – the stem of the HDF5 file name (e.g. `"phits_usrdef_simple"`)
  - `{species}` – `"n"`, `"g"`, or `"all"`
  - `{ext}` – the file extension (`"png"`, `"pdf"`, …)

- `extra_formats` – list of additional output formats to write alongside PNG (for example `["pdf"]` to also write vector PDF figures with rasterized image data).

The older `summed_dataset` field is kept for backward compatibility with earlier versions and with the `ng-viz h5-to-png` command. For standard summed images you usually do not need to set it.


---

## 12. Putting It Together

The example config file:

    examples/configs/phits_usrdef_simple.toml

is the simplest working end-to-end configuration. It is a good template to copy and edit for new runs. The sections documented above reflect the current, tested state of the code; if a value appears in that TOML, it should be described here, and vice-versa.

If you run into a mismatch between the docs and what the code accepts, treat the running code + `phits_usrdef_simple.toml` as authoritative and adjust this file accordingly.
