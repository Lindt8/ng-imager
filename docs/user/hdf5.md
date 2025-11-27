# HDF5 Output Format

The ngimager pipeline writes a single HDF5 file per run. The file is
designed to be:

- **self-describing** — contains a config snapshot and a README,
- **traceable** — you can go from pixels → cones → events → hits (and back),
- **diagnostic-friendly** — counters from each stage are stored.

This page documents the current layout.

---

## Top-level structure

At the top level, the file contains:

- attributes:
    - `format_version` (str): ng-imager HDF5 schema version.
    - `created_utc` (str): ISO-8601 timestamp of file creation.
    - `software` (str): software tag, e.g. `"ng-imager 0.3.1"`, derived from the installed package metadata.
    - `run_command` (str, optional): shell-style command string used to launch the run, e.g. `python -m ngimager.pipelines.core examples/configs/foo.toml --fast`.
    - `run_command_argv_json` (str, optional): JSON-encoded `argv` list corresponding to `run_command`, e.g. `["python","-m","ngimager.pipelines.core","examples/configs/foo.toml","--fast"]`.
    - `config_text` (str): verbatim TOML config snapshot as used.
- groups:
    - `/meta`
    - `/images`
    - `/cones`
    - `/lm` (list-mode and event/hit info; always present, richer in list mode)

---

## /meta

`/meta` holds geometry, run flags, human-readable text, and counters.

### Geometry and run info

Group attributes:

- `plane.P0`  : `[3]` float64, plane origin in cm
- `plane.n`   : `[3]` float64, plane normal
- `plane.eu`  : `[3]` float64, plane u-basis
- `plane.ev`  : `[3]` float64, plane v-basis

- `grid.u_min`, `grid.u_max`, `grid.du`
- `grid.v_min`, `grid.v_max`, `grid.dv`
- `grid.nu`, `grid.nv`

Run flags:

- `run_fast`     : bool
- `run_list`     : bool
- `run_neutron`  : bool
- `run_gamma`    : bool

### Config snapshot and README

Datasets:

- `/meta/config_toml` — the full TOML configuration used for this run,
  stored as a single variable-length UTF-8 string.
- `/meta/readme` — a short README (array of strings) summarizing the layout and
  pointing to the online documentation. (You are here, welcome.)

#### `/meta/config_effective`

A structured snapshot of the effective configuration used for the run,
including any CLI overrides that were applied on top of the TOML file.

It is represented as three parallel 1D datasets:

- `/meta/config_effective/keys`   (str[N]) – dotted field paths such as
  `"run.fast"`, `"run.list"`, `"io.input.path"`, `"run.meta.beam"`.
- `/meta/config_effective/types`  (str[N]) – Python type names for each entry,
  e.g. `"bool"`, `"int"`, `"float"`, `"str"`, `"list"`, etc.
- `/meta/config_effective/values` (str[N]) – JSON-encoded representation of the value when possible (e.g. `true`, `"foo"`, `["a","b"]`), otherwise a stringified `repr(...)` of the value.

This table mirrors `cfg.model_dump()` at the time of writing the HDF5 file and therefore reflects the actual settings used, not just the raw TOML contents.



### Extra text metadata 

If `[io.extra_text_files]` is set in the TOML config, ng-imager will
embed those files under:

- `/meta/extra_text/{key}`

Each dataset is a single variable-length UTF-8 string containing the full contents of the corresponding file.

For example, with:

```toml
[io.extra_text_files]
phits_input = "phits/input/deck.inp"
daq_config  = "daq/config/daq_settings.txt"
```

you would get:

- `/meta/extra_text/phits_input`
- `/meta/extra_text/daq_config`

storing the PHITS input deck and DAQ config text, respectively.


### Run metadata from config

For convenience, ng-imager can carry high-level run descriptors from
the TOML into the HDF5 file. When a `[run.meta]` table is present in
the config, all of its key–value pairs are mirrored into:

- `/meta/run_meta` : group

Each key in `[run.meta]` becomes a string attribute on this group. For
example, with:

    [run]
    plot_label = "175 MeV p, target B, det config 3"

    [run.meta]
    beam      = "175 MeV proton"
    target    = "Geometry B"
    det_setup = "Arrangement 3"
    facility  = "PTB"

the HDF5 file will contain:

- group `/meta/run_meta` with attributes:
    - `beam      = "175 MeV proton"`
    - `target    = "Geometry B"`
    - `det_setup = "Arrangement 3"`
    - `facility  = "PTB"`

If `[run].plot_label` is set, its value is also stored as the
`plot_label` attribute on `/meta/run_meta`. Visualization tools
(including the built-in pipeline PNG export and the `ng-viz` CLI) may
use this string as a figure title or annotation.

The `/meta/run_meta` group is deliberately free-form: consumers should
treat it as optional and robust to extra keys. It is intended to
capture human-facing run descriptions that travel with the file rather
than physics-critical configuration (which remains in
`/meta/config_toml`).



### Counters

Counters are stored as simple datasets under `/meta/counters`:

- `/meta/counters/s1_raw_events_total`
- `/meta/counters/s1_hits_after_filters`
- `/meta/counters/s2_events_after_filters`
- `/meta/counters/s3_cones_kept_n`
- `/meta/counters/s3_cones_kept_g`
- `/meta/counters/s3_cones_rejected_delta_theta`
- … etc.

The exact key set may grow over time, but the naming convention is:

- `s1_*` — Stage 1 (raw events → hits)
- `s2_*` — Stage 2 (hits → shaped/typed events → event filters)
- `s3_*` — Stage 3 (events → cones + cone filters)
- `s4_*` — Stage 4 (cones → images)

All counters are `int64` and are intended for quick QA and experiment
diagnostics.

---

## /images/summed

Summed images live under `/images/summed`:

- `/images/summed/n`   : `[nv, nu]` float32 — neutron-only SBP image
- `/images/summed/g`   : `[nv, nu]` float32 — gamma-only SBP image (if `run.gammas = true`)
- `/images/summed/all` : `[nv, nu]` float32 — `n + g`, only written when both
  species are present and produce non-zero images.

These datasets use gzip compression and match the grid described under `/meta`.


### `/images/summed/projections`

When `[vis.projections].enabled = true` in the TOML config, the pipeline writes 1D u/v projections of each summed image under `/images/summed/projections`.

Layout:

```text
/images/summed/projections/n/u       # [nu] float32, sum over v (rows)
/images/summed/projections/n/v       # [nv] float32, sum over u (cols)
/images/summed/projections/n/u_roi   # [nu] float32, ROI-limited (optional)
/images/summed/projections/n/v_roi   # [nv] float32, ROI-limited (optional)

# similarly for "g" and "all"
```

Here:

- `u[i]` is the sum of all pixels in column `i` of `/images/summed/n`
(i.e. integrated along the v-axis),
- `v[j]` is the sum of all pixels in row `j` (integrated along u),
- `nu` and `nv` match the imaging grid (`grid.nu`, `grid.nv`) stored in `/meta`,
- the mapping from index → coordinate uses the same grid as the 2D images:

```
u_center[i] = grid.u_min + (i + 0.5) * grid.du
v_center[j] = grid.v_min + (j + 0.5) * grid.dv
```

If a rectangular ROI is configured via `[vis.projections]`, the ROI-limited projections (`u_roi`, `v_roi`) are computed by summing only pixels whose centers fall inside the ROI. The ROI bounds (in cm) are stored as attributes on each species group, e.g.:

``` 
/images/summed/projections/n/attrs:
    roi_u_min_cm
    roi_u_max_cm
    roi_v_min_cm
    roi_v_max_cm
```

These projections provide a convenient sanity check for the u/v orientation and are intended to simplify downstream 1D analyses (e.g. peak finding, edge detection) without needing to re-derive them from the 2D images.


### `/images/summed/projections/*/metrics` — Projection Statistics and Fitting Results

When `[vis.projections.metrics]` is enabled, ng-imager computes and
stores per-axis (u, v) metrics for each species (`n`, `g`, `all`).

Metrics are written under:

    /images/summed/projections/{species}/metrics/u
    /images/summed/projections/{species}/metrics/v

If a rectangular ROI is configured, additional groups exist:

    /images/summed/projections/{species}/metrics/u_roi
    /images/summed/projections/{species}/metrics/v_roi

Each of these groups contains scalar (0D) datasets describing the
corresponding 1D projection curve.

#### Summary statistics (`compute_summary = true`)

Scalar datasets:

- `total_counts` (float64) – sum of all counts in the projection
- `mean_cm`      (float64) – weighted mean coordinate (cm)
- `median_cm`    (float64) – median coordinate (cm) from the CDF
- `std_cm`       (float64) – standard deviation (cm) about the mean
- `summary_ok`   (bool)    – validity flag (true if enough counts, etc.)

`summary_ok` is set to `false` and the coordinate-valued metrics are
set to NaN if the total counts fall below `min_counts`.

#### Peak metrics (`compute_peak = true`)

Peak metrics are currently based on the location of the **maximum bin**
of the projection (no Gaussian fit yet):

- `peak_pos_cm`  (float64) – coordinate of the maximum bin (cm)
- `peak_value`   (float64) – value of the maximum bin (counts)
- `peak_ok`      (bool)    – validity flag

#### Fractional edges (`compute_edges = true`)

Edges are derived from the normalized cumulative integral of the
projection, using configuration:

- `edge_low_frac`   (float64, attribute on metrics/u and metrics/v)
- `edge_high_frac`  (float64, attribute on metrics/u and metrics/v)
- `min_counts`      (float64, attribute)

Scalar datasets in each metrics group:

- `edge_low_cm`    (float64) – coordinate where CDF crosses `edge_low_frac`
- `edge_high_cm`   (float64) – coordinate where CDF crosses `edge_high_frac`
- `edge_width_cm`  (float64) – `edge_high_cm - edge_low_cm`
- `edges_ok`       (bool)    – validity flag

#### ROI metrics

If an ROI is defined via `[vis.projections]`, the ROI-limited metrics
are stored under:

    /images/summed/projections/{species}/metrics/u_roi
    /images/summed/projections/{species}/metrics/v_roi

These groups contain the same dataset names and semantics as the
non-ROI metrics. ROI projections use the *full* u/v index range; bins
outside the ROI are zero.

#### 2D centroid (`compute_centroid = true`)

Stored at:

    /images/summed/projections/{species}/metrics/centroid

Datasets:

- `u_centroid`   (float64)
- `v_centroid`   (float64)
- `total_counts` (float64)

These describe the centroid of the full 2D reconstructed image.

---

### Additional notes for downstream tools

- Metrics are always **datasets**, not attributes (except the edge
  configuration, which lives as attributes on `metrics/u` and
  `metrics/v`).
- Tools should detect metric availability by checking dataset
  existence.
- Coordinate values are stored in **centimeters**, regardless of
  plot-unit setting.
- ROI bounds appear as attributes under each species projection group:

    roi_u_min_cm
    roi_u_max_cm
    roi_v_min_cm
    roi_v_max_cm


---

### Additional notes for downstream tools

- Metrics are always **datasets**, not attributes (except fractional edge config).
- Tools should detect metric availability by checking dataset existence.
- Coordinate values are stored in **centimeters**, regardless of plot-unit setting.
- ROI bounds appear as attributes under the species projection group:

```
roi_u_min_cm
roi_u_max_cm
roi_v_min_cm
roi_v_max_cm
```




---

## /cones

`/cones` stores the geometric and physical properties of each cone.

Datasets:

- `/cones/cone_id`             : `[N_cones]` uint32
- `/cones/apex_xyz_cm`         : `[N_cones, 3]` float32
- `/cones/axis_xyz`            : `[N_cones, 3]` float32 (unit vectors)
- `/cones/theta_rad`           : `[N_cones]` float32
- `/cones/incident_energy_MeV` : `[N_cones]` float32
- `/cones/event_index`         : `[N_cones]` int32  
    - Row index into `/lm/event_type` and `/lm/hit_*` for the event that produced this cone.
- `/cones/gamma_hit_order`     : `[N_cones, 3]` int8  
    - For gamma cones (those with `species == 1`), each row is a triple `(i0, i1, i2)` giving the indices into `/lm/hit_*[event_index, :, :]` that correspond to the (first scatter, second scatter, third point) used to build the Compton cone. For neutron cones (`species == 0`), the row is `(-1, -1, -1)` and should be ignored.

Classification:

- `/cones/species`             : `[N_cones]` uint8
- `/cones/species_labels`      : string array legend,
    - e.g. `["0=neutron", "1=gamma"]`
- `/cones/recoil_code`         : `[N_cones]` uint8
- `/cones/recoil_code_labels`  : string array legend,
    - e.g. `["0=NA/gamma/unknown", "1=proton", "2=carbon"]`

Interpretation:

- `species` distinguishes neutron and gamma cones.
- `recoil_code` distinguishes proton vs carbon recoils for neutron cones.
- `incident_energy_MeV` is the kinematically inferred incident particle energy for that cone:
    - for neutrons, from ToF + deposited energy at the first scatter;
    - for gammas, from Compton kinematics.
- `event_index` + `gamma_hit_order` allow you to recover, for each gamma cone, exactly which of the three stored hits in `/lm/hit_*` were interpreted as first/second/third in the selected Compton ordering.


---

## /lm: events and hits

The `/lm` group contains per-event and per-hit data used for list-mode analysis.
It is written regardless of `run.list`; list-mode *imaging* adds extra datasets.

### Event-level datasets

- `/lm/event_type`         : `[N_events]` uint8 (`0=neutron`, `1=gamma`)
- `/lm/event_type_labels`  : legend array, e.g. `["0=neutron", "1=gamma"]`
- `/lm/event_meta_run_id`  : `[N_events]` int32 (optional provenance)
- `/lm/event_meta_file_ix` : `[N_events]` int32 (optional provenance)

An “event” here means a fully typed `NeutronEvent` or `GammaEvent` that has
survived hit-level and event-level filters (Stage 1–2).

### Hit-level datasets

Hits are stored in fixed slots per event:

- `/lm/hit_pos_cm`      : `[N_events, 3, 3]` float32
- `/lm/hit_t_ns`        : `[N_events, 3]` float32
- `/lm/hit_L_mevee`     : `[N_events, 3]` float32
- `/lm/hit_det_id`      : `[N_events, 3]` int32
- `/lm/hit_material_id` : `[N_events, 3]` int16

Conventions:

- Neutron events use slots 0 and 1; slot 2 is filled with NaNs / -1.
- Gamma events use all three slots.
- `hit_L_mevee` is `Hit.L` (calibrated light in MeVee for real data or `Edep`
  in MeV for PHITS-style sources).

Material labels:

- `/lm/material_id_labels` : string array mapping `hit_material_id` values
  back to human-readable material names (`"OGS"`, `"M600"`, etc.).

---

## /lm: list-mode imaging (optional)

When `run.list = true`, the SBP reconstruction also tracks which pixels each
cone contributes to. This information is written to `/lm` in a form that
makes it easy to map:

> pixels → cones → events → hits

### Per-cone pixel indices

- `/lm/cone_pixel_indices` : `[K, 2]` uint32
    - Canonical location: each row is `(cone_id, flat_pixel_index)`.
- `/images/list_mode/cone_pixel_indices` : alias (HDF5 soft link) to `/lm/cone_pixel_indices`


Each row is:

```text
(cone_id, flat_pixel_index)
```

where:

- `cone_id` is an index into `/cones/cone_id` and the other cone arrays, and
- `flat_pixel_index` is a flattened `(u, v)` index:

```
flat = v * nu + u
```

You can recover `(u, v)` from `flat` using `divmod(flat, nu)`.

Only cones that actually intersect the imaging plane appear in this dataset.

### Event survival table

To make the pipeline’s decisions transparent, a small “survival table” is stored:

- `/lm/event_survival` : `[N_events, 3]` int32

Columns:

1. `event_index` — row index into the `/lm` event/hit datasets.
2. `first_cone_index` — index into `/cones/*` for the first cone built from
   this event, or `-1` if the event never produced a cone.
3. `first_imaged_cone_index` — cone index for the first cone that both was
   built and intersected the plane (i.e. appeared in `/lm/cone_pixel_indices`),
   or `-1` if none of that event’s cones hit the plane.

Together, the mapping looks like:

```text
hit (lm/hit_*)   ← event_index ← event_survival[:, 0]
                               ↘ first_cone_index      → cones/*
                                ↘ first_imaged_cone_index → cone_pixel_indices
```

If you need more detailed correlations (e.g. many cones per event), you can
walk:

1. `/cones/event_index` to find all cones belonging to an event;
2. `/lm/cone_pixel_indices` to find all pixels touched by each cone.

---

## Adapter-specific extras

Adapters are allowed to attach additional, source-specific information under `/meta` and elsewhere, as long as the core layout described above remains stable.

At the moment there are two main adapter families with extra payloads:

---

### PHITS adapter (`phits_usrdef`)

The PHITS adapter may populate ragged list-mode datasets under `/lm/hits` and `/lm/events` as a more PHITS-like representation of the input, mainly for debugging and cross-checks:

- `/lm/hits`  
    - `event_ptr` : CSR-style pointer array (length = N_events + 1)  
    - `x_cm, y_cm, z_cm, t_ns, Edep_MeV, reg` : flat per-hit arrays

- `/lm/events`  
    - `event_type` : 0 = unknown, 1 = n, 2 = g, 3 = mixed  
    - `iomp, batch, history, no, name` : PHITS event bookkeeping

These datasets are optional and may be omitted when the PHITS adapter is not used.

---

### NOVO DDAQ ROOT adapter (`root_novo_ddaq`)

When the NOVO DDAQ ROOT adapter is used, additional run-level metadata from the ROOT `meta` TTree are persisted under:

- `/meta/root_novo_ddaq` : group

Run-level scalar fields are stored as group attributes, mirroring the ROOT metadata where available:

- `InputFileName`
- `OutputFileName`
- `CDFFileName`
- `PSDCutsFileName`
- `SampleRate`
- `NumDet`
- `NumThreads`
- `WriteHistograms`
- `MergeMode`
- `CardOffsetChannel`
- `UsePositionVeto`

In addition, a normalized integer run identifier is exposed as:

- `run_number` (attribute on `/meta/root_novo_ddaq`)

This is intended to capture the “run number” used in NOVO data taking (e.g. `..._000041.root` → `run_number = 41`). When the adapter cannot infer a run number, it may omit this attribute or use a sentinel value.

Per-detector geometric and timing metadata from the ROOT file are collected into a small table under:

- `/meta/root_novo_ddaq/detectors` : group

with the following datasets (length = `NumDet`):

- `det_id`            : `[NumDet] int32`  
- `pos`               : `[NumDet, 3] float32` – detector position (`PosX`, `PosY`, `PosZ`) in **mm**  
- `dim`               : `[NumDet, 3] float32` – detector dimensions (`DimX`, `DimY`, `DimZ`) in **mm**  
- `rot_deg`           : `[NumDet, 3] float32` – detector rotations (`RotX`, `RotY`, `RotZ`) in **degrees**  
- `local_time_offset` : `[NumDet] float32` – local timing offset in **ns**  
- `global_time_offset`: `[NumDet] float32` – global timing offset in **ns**  
- `pos_cal_file`      : `[NumDet] string` – per-detector position calibration filenames  
- `energy_cal_file`   : `[NumDet] string` – per-detector energy calibration filenames  
- `is_start_det`      : `[NumDet] int8` – 1 if marked as a start detector, else 0  
- `is_laser_det`      : `[NumDet] int8` – 1 if marked as a laser detector, else 0  

Each numeric dataset carries a `units` attribute (`"mm"`, `"deg"`, `"ns"`) to make downstream interpretation explicit.

The `/meta/root_novo_ddaq` layout is intended to be forward compatible: new attributes or datasets may be added in future versions, but existing ones should not be removed or change meaning.


---

## Versioning and compatibility

The root attribute `format_version` records the HDF5 schema version. Minor
additions (e.g. new counters or label arrays) are made in a way that keeps
existing code working; new consumers should always check for dataset
existence before assuming it is present.

If you are scripting against the file format, consider:

- gating on `format_version`,
- using the label arrays (`*_labels`) rather than hard-coding integer codes, and
- relying on `/meta/config_toml` to reproduce or document the run settings.
