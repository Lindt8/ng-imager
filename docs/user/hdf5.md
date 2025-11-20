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
  - `format_version`
  - `created_utc`
  - `software`
  - `config_text` (legacy snapshot, kept for compatibility)
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

---

## /cones

`/cones` stores the geometric and physical properties of each cone.

Datasets:

- `/cones/cone_id`             : `[N_cones]` uint32
- `/cones/apex_xyz_cm`         : `[N_cones, 3]` float32
- `/cones/axis_xyz`            : `[N_cones, 3]` float32 (unit vectors)
- `/cones/theta_rad`           : `[N_cones]` float32
- `/cones/incident_energy_MeV` : `[N_cones]` float32

Classification:

- `/cones/species`             : `[N_cones]` uint8
- `/cones/species_labels`      : string array legend,
                                 typically `["0=neutron", "1=gamma"]`
- `/cones/recoil_code`         : `[N_cones]` uint8
- `/cones/recoil_code_labels`  : string array legend,
                                 typically `["0=unknown/NA", "1=proton", "2=carbon"]`
- `/cones/event_index`         : `[N_cones]` uint32, index into the event arrays
                                 under `/lm` (see below).

Notes:

- `incident_energy_MeV` is the kinematically inferred incident particle energy
  for that cone:
  - for neutrons, from ToF + deposited energy at the first scatter;
  - for gammas, from Compton kinematics.
- `species` and `recoil_code` allow you to separate neutron/gamma cones and
  distinguish proton vs carbon recoils in post-processing.

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

Each row is:

```text
(cone_id, flat_pixel_index)
```

where:

- `cone_id` is an index into `/cones/cone_id` and the other cone arrays, and
- `flat_pixel_index` is a flattened `(u, v)` index:

```python
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

Some adapters (notably PHITS) may also populate ragged list-mode datasets
for arbitrary-multiplicity events:

- `/lm/hits/event_ptr`    : `[N_events+1]` int64 CSR pointer
- `/lm/hits/x_cm`, `y_cm`, `z_cm`, `t_ns`, `Edep_MeV`, `reg` : flat arrays
- `/lm/events/event_type`, `iomp`, `batch`, `history`, `no`, `name`

These are additive and do not conflict with the fixed-slot `/lm/hit_*` and
`/lm/event_*` datasets described above.

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
