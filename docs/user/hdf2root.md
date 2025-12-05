# HDF5 → ROOT conversion (`ng-hdf2root`)

This page documents the small helper tool `ng-hdf2root`, which converts an
`ngimager` HDF5 output file into a ROOT file for further analysis and
visualization.

The conversion is done entirely in Python using `h5py` and `uproot`. You do
*not* need a full C++ ROOT installation in order to *run* the converter, but
you will of course need ROOT (or a Python ROOT ecosystem) to analyze the
resulting `.root` file.

---

## 1. Overview

The converter is provided as a small CLI:

    ng-hdf2root my_run.h5

This reads the ngimager HDF5 file `my_run.h5` and writes a ROOT file
`my_run.root` in the same directory.

The converter is designed to:

- preserve the **hit → event → cone → pixel** relationships from the HDF5
  layout,
- expose flattened, ROOT-friendly TTrees for:
    - per-hit list-mode analysis,
    - per-cone analysis,
    - cone→pixel mappings (list-mode imaging),
- carry along a minimal amount of run metadata for bookkeeping, and
- avoid depending on a full ROOT installation at conversion time.

Summed SBP images are also exported in a simple TTree form, so users can create
“quick check” plots in ROOT that are directly comparable to the PNGs produced
by `ngimager`.

---

## 2. Basic usage

### 2.1 Command line

Convert a single HDF5 file:

    ng-hdf2root path/to/my_run.h5

By default, this will create:

- `path/to/my_run.root`

You can override the output path:

    ng-hdf2root path/to/my_run.h5 -o path/to/custom_name.root

To allow overwriting an existing ROOT file, pass `--overwrite`:

    ng-hdf2root path/to/my_run.h5 --overwrite

The converter exits with a non-zero status code and a short error message if:

- the input HDF5 file does not exist,
- the output ROOT file already exists and `--overwrite` was not given, or
- the HDF5 file is missing required groups/datasets.

### 2.2 As a Python module

You can also call the converter from Python:

```python
from pathlib import Path
from ngimager.tools.hdf5_to_root import convert_hdf5_to_root

hdf_path = Path("my_run.h5")
root_path = Path("my_run.root")
convert_hdf5_to_root(hdf_path, root_path, overwrite=True)
``` 

This is useful if you want to script batches of conversions.

---

## 3. ROOT file structure

The converter produces several TTrees designed for common analysis patterns.

### 3.1 Per-hit list-mode tree: `lm`

The `lm` tree is the main workhorse for hit-level analysis. It has one row per
*hit* (not per event). Event-level quantities are repeated for each hit.

Branches:

- Event context:
    - `event_index`         (int32)   – row index into the original `/lm` event
      arrays.
    - `event_type`          (uint8)   – `0 = neutron`, `1 = gamma`.
    - `event_meta_run_id`   (int32)   – optional provenance from the adapter.
    - `event_meta_file_ix`  (int32)   – optional file index from the adapter.
    - `event_surv_stage0`   (int32)   – survival flag/counter for Stage 0.
    - `event_surv_stage1`   (int32)   – survival flag/counter for Stage 1.
    - `event_surv_stage2`   (int32)   – survival flag/counter for Stage 2.
    - `event_cone_id`       (int32)   – “best” cone ID for this event
      (`-1` if none).
    - `event_imaged_cone_id` (int32)  – “best imaged” cone ID
      (`-1` if none).

- Hit information:
    - `hit_index`           (int8)    – slot index within the event
      (`0`, `1`, or `2`).
    - `hit_pos_x_cm`        (float32) – x coordinate in cm.
    - `hit_pos_y_cm`        (float32) – y coordinate in cm.
    - `hit_pos_z_cm`        (float32) – z coordinate in cm.
    - `hit_t_ns`            (float32) – time in ns.
    - `hit_L_mevee`         (float32) – light / energy proxy in MeVee.
    - `hit_det_id`          (int32)   – detector ID.
    - `hit_material_id`     (int16)   – index into the material table.

Only *valid* hit slots are exported. Slots with:

- `hit_det_id < 0`, or
- non-finite coordinates in `/lm/hit_pos_cm`

are treated as “empty” and omitted.

This makes it natural to build 1D/2D histograms like “incident neutron energy
for all imaged events” or “time difference between two neutron scatters” in
ROOT using standard cuts on the event-level columns.

In many experimental analyses, you may also want spectra of deposited
light/energy at individual scatters. The `hit_L_mevee` branch is the
per-hit calibrated light (in MeVee for real data, or `Edep` in MeV for
PHITS-style sources) and is often used as a proxy for deposited energy.


#### Example: basic histograms in ROOT (C++)

```cpp
TFile *f = TFile::Open("my_run.root");
TTree *lm = (TTree*)f->Get("lm");

// Histogram of hit light for neutron events only
lm->Draw("hit_L_mevee>>hL(200,0,10)", "event_type == 0");

// Compare light at first vs second scatters in neutron events
lm->Draw(
    "hit_L_mevee>>hL1(200,0,10)",
    "event_type == 0 && hit_index == 0"
);
lm->Draw(
    "hit_L_mevee>>hL2(200,0,10)",
    "event_type == 0 && hit_index == 1",
    "same"
);
```

#### Example: time difference between first and second neutron scatters

```cpp
// For neutron events, hits typically occupy slots 0 and 1.
// Define Δt = t(slot 1) - t(slot 0) by combining lm rows.

TTree *lm = (TTree*)f->Get("lm");
lm->Draw(
    "hit_t_ns - hit_t_ns",
    "event_type == 0 && hit_index == 1",
    ""
);
```

In practice you might instead build a friend tree or use RDataFrame to do a
join-style operation over `event_index` and `hit_index`. The important part is
that all the necessary information is available in a straightforward, flat
layout.

---

### 3.2 Per-cone tree: `cones`

The `cones` tree exposes one row per reconstructed cone.

Branches:

- Linkage:
    - `cone_id`             (int32)   – row index into the original `/cones`
      group.
    - `event_index`         (int32)   – index into the `lm` event arrays.

- Classification:
    - `species`             (uint8)   – `0 = neutron`, `1 = gamma`.
    - `recoil_code`         (uint8)   – `0 = NA/gamma/unknown`,
      `1 = proton`, `2 = carbon`.

- Kinematics and geometry:
    - `incident_energy_MeV` (float32) – **kinematically inferred incident
      neutron or gamma energy** for this cone. For neutrons this comes from
      ToF + deposited energy at the first scatter; for gammas from Compton
      kinematics (as documented in the HDF5 format and architecture docs).
    - `apex_x_cm`, `apex_y_cm`, `apex_z_cm` (float32) – cone apex position.
    - `axis_x`, `axis_y`, `axis_z` (float32) – unit direction vector.
    - `theta_rad` (float32) – cone half-angle in radians.

- Gamma hit ordering:
    - `gamma_hit_order_0/1/2` (int8) – hit slot indices used in the Compton
      sequence for gamma cones; `(-1, -1, -1)` for neutron cones.

Typical use in ROOT:

```cpp
TTree *cones = (TTree*)f->Get("cones");

// Spectrum of incident neutron energies for imaged cones only
cones->Draw(
    "incident_energy_MeV>>hEn(200,0,20)",
    "species == 0"
);
```

You can correlate cones with hits via `event_index`, or via `cone_id` and the
cone-pixel tree described below.

---

### 3.3 Cone→pixel mappings: `cone_pixels`

When list-mode imaging is enabled (`run.list = true`), the HDF5 file contains a
mapping from cones to imaging pixels via:

- `/lm/cone_pixel_indices`  – rows of `(cone_id, flat_pixel_index)`

The converter exposes this as a `cone_pixels` TTree with one row per
intersection:

- `cone_id`          (int32)   – index into the `cones` tree.
- `flat_pixel_index` (uint32)  – flattened `(u, v)` index:
    - `flat = v * nu + u`
- `u_index`          (int32)   – pixel index along the `u` axis.
- `v_index`          (int32)   – pixel index along the `v` axis.
- `u_cm`, `v_cm`     (float32) – pixel center coordinates in cm.
- `nu`, `nv`         (int32)   – grid dimensions; repeated for convenience.

This lets you, for example, generate list-mode images directly in ROOT by
plotting `u_cm` vs `v_cm` and applying arbitrary cone or event selections.

---

### 3.4 Summed images: `images_summed`

Summed SBP images from `/images/summed` are exported in a compact TTree:

- Tree name: `images_summed`
- One row per species (`"n"`, `"g"`, `"all"`).

Branches:

- `species`  (string)   – `"n"`, `"g"`, or `"all"`.
- `nu`, `nv` (int32)    – image dimensions.
- `counts`   (float32[]) – flattened image data (row-major order,
  length `nu * nv`).

Reconstruction example in ROOT (C++):

```cpp
TTree *img = (TTree*)f->Get("images_summed");
img->Draw("counts", "species == \"n\"");
```

To turn this back into a 2D histogram you can either:

- copy `counts` into a TH2 manually, or
- use Python/ROOT together (e.g. `uproot` + `matplotlib`) for quick
  visual cross-checks against the PNGs produced by `ngimager`.

---

### 3.5 Run metadata: `file_meta` and `run_meta`

Two small helper trees carry run metadata:

- `file_meta` (one entry)
    - `format_version` – HDF5 format version (string).
    - `created_utc`    – ISO-8601 timestamp when the file was written.
    - `software`       – ngimager software tag.
    - `run_command`    – command line used to launch the run, if available.

- `run_meta` (optional, one row per key)
    - `key`   – attribute name from `/meta/run_meta`.
    - `value` – stringified attribute value.

These are intended for bookkeeping and plotting annotations (e.g. adding the
beam description or configuration label as a text box in your ROOT canvases).

---

## 4. Typical workflows in ROOT

Here are a few common analysis patterns you might follow.

### 4.1 1D spectra from list-mode hits

```cpp
TFile *f = TFile::Open("my_run.root");
TTree *lm = (TTree*)f->Get("lm");

// Incident neutron energy proxy from hit_L_mevee for first neutron scatters
lm->Draw(
    "hit_L_mevee>>hL(200,0,10)",
    "event_type == 0 && hit_index == 0"
);
```

### 4.2 2D correlations from cones

```cpp
TTree *cones = (TTree*)f->Get("cones");

// Incident energy vs cone angle for neutron cones
cones->Draw(
    "theta_rad:incident_energy_MeV>>h(200,0,20, 180,0,3.2)",
    "species == 0",
    "colz"
);
```

#### Example: incident energy spectrum from cone kinematics

```cpp
TTree *cones = (TTree*)f->Get("cones");

// 1D spectrum of kinematic incident energy (neutrons only)
cones->Draw(
    "incident_energy_MeV>>hEn(200,0,20)",
    "species == 0"
);
```

You can adjust the species selection (`species == 0` for neutrons,
`species == 1` for gammas) or apply additional cuts on `recoil_code`,
`theta_rad`, or anything else carried by the `cones` tree.


### 4.3 List-mode imaging

```cpp
TTree *cp = (TTree*)f->Get("cone_pixels");

// Simple list-mode image in (u, v) with counts per pixel
cp->Draw(
    "v_index:u_index>>hUV(200,0,200, 200,0,200)",
    "",
    "colz"
);
```

You can combine this with cuts on the `cones` or `lm` trees (via `cone_id` or
`event_index`) using friend trees or RDataFrame to build more sophisticated
selection logic.

---

## 5. Notes and limitations

- The converter currently focuses on TTrees rather than native ROOT histogram
  classes. This keeps the implementation simple and robust and lets you build
  whatever histograms you like on the ROOT side.
- Event-level quantities are repeated per hit in the `lm` tree. This increases
  file size slightly but greatly simplifies typical ROOT usage.
- The tool does not attempt to be aware of all possible future HDF5 extras;
  it consumes only the documented layout (see the HDF5 format doc) and ignores
  unknown groups/datasets.
- If a future format version introduces new fields in `/lm` or `/cones`, the
  converter can be extended to pass those through as additional branches
  without breaking existing analyses.

For further details on the HDF5 side, see the main [HDF5 Output Format](hdf5.md)
documentation.
