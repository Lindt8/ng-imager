
# ng-imager Architecture & Pipeline Overview  
*(Design document and development guideline)*

> This document describes how `ngimager` **should** be structured and behave.  
> It serves both as:
> - a **design reference** for refactors and new features, and  
> - a description of the **current implementation state**, especially where explicitly noted (e.g. §15.5, §16).
>
> As the code evolves, either the implementation should be brought back into alignment with this document, or this document should be updated to describe the new, intentional design.

---

## 1. Purpose and Scope

`ngimager` is a modular, maintainable re-implementation of the legacy `expNOVO_imager_legacy.py` script for neutron and gamma imaging with NOVO detectors.

Core goals:

- Reproduce (and eventually improve on) the physics and imaging behavior of the legacy script, especially **2D SBP images from neutron and gamma cones** given equivalent inputs.
- Replace the legacy monolithic script with a **modular, testable, config-driven** package.
- Keep NOVO-specific details (detector layouts, materials, PHITS/ROOT formats, specific acquisition systems) **isolated** in adapters and configuration, so the **physics and imaging core can be reused**.
- Support a single unified pipeline whose behavior can be modified via two orthogonal toggles:
    - **Fast mode**: more aggressive filtering and limits for quick feedback during experiments.
    - **List mode**: additional per-cone imaging output for deep post-processing.

The legacy script remains the behavioral reference: given equivalent inputs and reasonable settings, `ngimager` should produce images that are physically consistent with the legacy SBP images (even if the file formats and small numerical details differ).

---

## 2. High-Level Dataflow

The pipeline transforms **raw coincident event data** into **images**, with explicit intermediate representations.

At a high level:

1. **Load config** (`.toml`).
2. **Select adapter** (PHITS / ROOT / HDF5) based on config.
3. **Adapter emits raw events**: each raw event is a *set of correlated/coincident hits*.
4. **Apply hit-level filters**, in a single place, using canonical `Hit` objects (or equivalent dicts that the adapter has normalized into the canonical `Hit` shape).
5. **Discard raw events** that no longer have enough hits to be kinematically reconstructable (e.g., fewer than 2 valid neutron hits or 3 valid gamma hits).
6. **Shape hits into imaging-viable Events** (neutron 2-hit, gamma 3-hit, etc.).
7. **Apply event-level filters**.
8. Use the configured **energy strategy & kinematics** (plus priors) to build a **cone** for each surviving event, possibly after testing several internal hypotheses (neutron proton vs carbon recoil, gamma permutations).
9. **Apply cone-level filters** (Δθ, max incident energy, etc.), yielding at most one final cone per event (or zero if none are viable).
10. **Image cones** (SBP initially, other methods later).
11. **Write results to HDF5**, including enough metadata to trace `hit → event → cone → image pixel`.

Depending on `run.neutrons` and `run.gammas`, only the chosen particle types are shaped into events, propagated into cones, and imaged; the other type is ignored at all stages.

### Conceptual pseudocode (current implementation)

This reflects the four-stage design and the current `ngimager.pipelines.core.run_pipeline`:

```python
def run_pipeline(cfg_path: str) -> Path:
    # Load TOML → Config (pydantic)
    cfg = load_config(cfg_path)

    # CLI flags override [run], [io] fields (fast, list, neutrons, gammas, I/O paths, ...)
    apply_cli_overrides(cfg)

    diag_level = cfg.run.diagnostics_level
    plane = Plane.from_cfg(
        cfg.plane.origin,
        cfg.plane.normal,
        cfg.plane.u_min, cfg.plane.u_max, cfg.plane.du,
        cfg.plane.v_min, cfg.plane.v_max, cfg.plane.dv,
    )

    # Create HDF5 and write /meta (geometry, run flags, config snapshot)
    out_path = Path(cfg.io.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = write_init(str(out_path), cfg_path, cfg, plane)

    counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Stage 1: Raw events → hits (hit-level filters + reconstructability)
    # ------------------------------------------------------------------
    if cfg.io.input_format == "phits_usrdef":
        adapter = make_adapter(_inject_detector_materials(cfg.io.adapter, cfg.detectors))

        raw_events_after_filters: list[dict] = []
        for ev in adapter.iter_raw_events(str(cfg.io.input_path)):
            hits = list(ev.get("hits", []))

            # Normalize event_type = 'n' / 'g' / None (PHITS-only hint)
            et = normalize_event_type(ev.get("event_type", ""))

            counters["raw_events_total"] = counters.get("raw_events_total", 0) + 1

            # Hit-level filters (min/max light, bar/material whitelists, etc.)
            hits = apply_hit_filters(
                hits,
                cfg.filters.hits,
                counters,
                particle_type=et,
            )

            # Early reconstructability decision (2 hits for n, 3 for g)
            if not is_reconstructable(hits, cfg.filters.events, counters, event_type=et):
                continue
            if not hits:
                continue

            ev2 = dict(ev)
            ev2["hits"] = hits
            if et is not None:
                ev2["event_type"] = et
            raw_events_after_filters.append(ev2)

        # Stage 2a: hits → shaped events → typed events (NeutronEvent / GammaEvent)
        shaped_events, shape_diag = shape_events_for_cones(
            raw_events_after_filters,
            ShapeConfig(),
            counters=counters,
        )
        events = shaped_to_typed_events(shaped_events, order_time=True)

        # Stage 2b: event-level filters (ToF windows, L1/L2/any thresholds)
        events = apply_event_filters(
            events,
            cfg.filters.events,
            counters,
        )

    else:
        # Non-PHITS adapters are expected to yield fully-typed Event objects
        adapter = make_adapter(cfg.io.adapter)
        events = list(adapter.iter_events(str(cfg.io.input_path)))
        events = apply_event_filters(
            events,
            cfg.filters.events,
            counters,
        )

    # Counters now reflect Stage 1–2 survival (events_typed_*, events_after_filters, ...)

    # ------------------------------------------------------------------
    # Stage 3: Events → cones (species-aware, with cone-level filters)
    # ------------------------------------------------------------------
    lut_registry = build_lut_registry(cfg.energy.lut_paths)
    energy_model = make_energy_strategy(cfg.energy, lut_registry=lut_registry)
    prior = make_prior(cfg.prior.model_dump(), plane)

    (
        cone_ids,
        apex_xyz_cm,
        axis_xyz,
        theta_rad,
        cone_species,
        recoil_code,
        incident_energy_MeV,
    ) = _build_cones_from_events(cfg, events, plane, counters)

    # Stage 3 includes:
    #   - neutron vs gamma cone builders
    #   - proton vs carbon hypothesis test for neutrons
    #   - Δθ = |φ − θ| cuts via [filters.cones]
    #   - max_incident_energy_MeV cuts via [filters.cones]
    # Counters track attempted/successful cone builds and per-species rejections.

    # ------------------------------------------------------------------
    # Stage 4: cones → images (SBP + optional list-mode)
    # ------------------------------------------------------------------
    cones_for_sbp = [
        Cone(apex=apex_xyz_cm[i], direction=axis_xyz[i], theta=float(theta_rad[i]))
        for i in range(len(cone_ids))
    ]

    recon = reconstruct_sbp(
        cones=cones_for_sbp,
        plane=plane,
        workers=cfg.run.workers,
        chunk_cones=cfg.run.chunk_cones,
        list_mode=cfg.run.list,
        progress=cfg.run.progress,
    )

    # Species-separated summed images + combined "all"
    write_summed(f, "n", recon.summed_n)
    write_summed(f, "g", recon.summed_g)
    if recon.summed_all is not None:
        write_summed(f, "all", recon.summed_all)

    # Per-cone geometry + classification
    write_cones(
        f,
        cone_ids,
        apex_xyz_cm,
        axis_xyz,
        theta_rad,
        species=cone_species,
        recoil_code=recoil_code,
        incident_energy_MeV=incident_energy_MeV,
    )

    # Per-event / per-hit physics
    write_events_hits(f, events)

    # Optional list-mode extras: cone→pixel indices and survival mapping
    if cfg.run.list:
        write_lm_indices(f, recon.lm_indices, events, cone_ids)

    # Counters (for diagnostics and offline QA)
    write_counters(f, counters)

    f.close()
    return out_path
```

**Key points:**

- Adapters emit **raw events** that already group coincident hits (the acquisition system’s notion of “event”), but thresholds in acquisition are lower than post-processing thresholds.
- Hit-level cuts are applied early. Raw events that no longer contain enough candidate hits to ever form a reconstructable cone are discarded before shaping.
- Shaping and typing are separate from the adapter, so physics and filtering logic is shared across PHITS, ROOT, and future formats.
- Cone-building uses the **energy strategy** and priors directly inside the neutron/gamma builders. Neutron H/C hypotheses and gamma permutations are evaluated internally, but each event ultimately contributes **at most one** selected cone.
- Each stage maintains **counters** for accepted/rejected objects, and HDF5 output reflects only the surviving objects after the full chain, while still allowing early-exit runs and detailed diagnostics.

---

## 3. Configuration: The `.toml` File

The `.toml` config is the single source of truth for all non-data settings. It should be:

- **Explicit**: parameters are clearly described and named.
- **Non-redundant**: no value should have to be repeated in multiple places.
- **Overridable by CLI** for operational convenience (e.g. `--fast`, `--list`, particle toggles, and I/O paths).

Example configs live under `examples/configs/`. They are intentionally **comment-heavy** and serve as both runnable examples and documentation.

### 3.1. Top-Level Sections

These map to Pydantic models in `ngimager.config.schemas`.

#### `[run]`

General pipeline behavior, particle-type toggles, and diagnostics:

```toml
[run]
fast              = false      # enable "fast" presets (see [fast])
list              = false      # enable list-mode (per-cone footprints)
neutrons          = true       # process neutron events
gammas            = true       # process gamma events
max_events        = 0          # 0 = no limit, >0 = cap typed events
max_cones         = 0          # 0 = no limit, >0 = cap selected cones
diagnostics_level = 1          # 0 = silent, 1 = normal, 2 = verbose
workers           = "auto"     # SBP worker count: "auto" or integer
chunk_cones       = "auto"     # SBP chunk size: "auto" or integer
progress          = true       # show SBP progress bars if available
sbp_engine        = "scan"   # "scan" (continuous, matrix math) or "poly" (fast perimeter)
jit               = true            # use Numba acceleration for the scan engine if available

```

- `fast` and `list` are **modifiers** of the standard pipeline, not separate pipelines.
- `max_events` and `max_cones` are primarily useful for tests and “fast feedback” runs.

**CLI overrides (design + current behavior):**

Exposed by the Typer CLI in `ngimager.pipelines.core` (invoked via `python -m ngimager.pipelines.core` and, in future, via a `ngimager` console script):

- `--fast / --no-fast`  
  Override `run.fast`.

- `--list / --no-list`  
  Override `run.list`.

- `--neutrons / --no-neutrons`  
  Override `run.neutrons`.

- `--gammas / --no-gammas`  
  Override `run.gammas`.

- `--input-path PATH`  
  Override `[io].input_path`.

- `--output-path PATH`  
  Override `[io].output_path`.

A future CLI flag like `--until=STAGE` may override `[pipeline].until` once restart-from-HDF5 is implemented.

#### `[io]`

Input/output format and paths:

```toml
[io]
input_path   = "examples/data/phits_usrdef_simple.out"
input_format = "phits_usrdef"    # or "root_novo_ddaq", "hdf5_ngimager" (planned)
output_path  = "output/phits_usrdef_simple.h5"

[io.adapter]
type  = "phits_usrdef"           # adapter type
style = "novo"                   # adapter-specific parsing style (if applicable)
# additional adapter-specific knobs live here
```

- `input_format` selects the adapter family (`"phits_usrdef"`, `"root_novo_ddaq"`, `"hdf5_ngimager"` in the future).
- `[io.adapter].type` is passed to `make_adapter` and typically matches `input_format` for PHITS and ROOT paths.
- CLI options `--input-path` and `--output-path` override `input_path` and `output_path` at runtime, without having to edit the TOML.

`"hdf5_ngimager"` is reserved for a **future** restart/resume mode (see §12.3).

#### `[pipeline]`

Controls how far the pipeline runs (future-proofing for restarts):

```toml
[pipeline]
until = "image"  # "hits" | "events" | "cones" | "image"
```

- Currently, the main pipeline **always** runs to images; `until` is tracked as an architectural target for early-exit and restart control.

#### `[detectors]`

Detector-level configuration, especially mapping from IDs/regions to materials:

```toml
[detectors]
default_material = "M600"   # fallback if no explicit map entry is found

[detectors.material_map]
200 = "M600"
210 = "M600"
220 = "M600"

# [detectors.geometry]
# # reserved for future geometry metadata such as bar centers, axes, etc.
# [[detectors.geometry.bars]]
# det_id   = 200
# center   = [x_cm, y_cm, z_cm]
# axis     = [ux, uy, uz]
# length   = 100.0
```

- For PHITS workflows, detailed bar geometry often lives in the PHITS model; here we primarily configure **materials**.
- For real detectors and ROOT adapters, `[detectors.geometry]` will be the place to describe bar centers, orientations, and sizes in a re-usable way.

> Implementation note: adapters consult `detectors.material_map` and `detectors.default_material` to assign `Hit.material`. No separate default material should be injected via `[io.adapter]`.

#### `[energy]`

Neutron energy strategy and LUT configuration:

```toml
[energy]
# Neutron energy strategy:
# - "ELUT"    : use E(L) LUTs for the first scatter (per material)
# - "ToF"     : use time-of-flight to estimate neutron energy
# - "FixedEn" : use a fixed incident energy (e.g. DT source)
# - "Edep"    : treat Hit.L as deposited energy in MeV (e.g. PHITS Edep_MeV)
strategy = "Edep"

# For neutrons, assume recoils are always protons (true)
# or consider both proton and carbon recoils (false).
force_proton_recoils = false

# Used only when strategy = "FixedEn"
fixed_En_MeV = 14.1

# LUT paths (used by "ELUT" and optionally by "Edep")
# Keys: lut_paths.<MATERIAL>.<SPECIES>
#   - species is usually "proton" or "carbon"
[energy.lut_paths]
M600.proton = "lut/M600_proton_E_vs_L.npz"
OGS.proton  = "lut/OGS_proton_E_vs_L.npz"
# M600.carbon = "lut/M600_carbon_E_vs_L.npz"
# OGS.carbon  = "lut/OGS_carbon_E_vs_L.npz"
```

- `strategy = "Edep"` is the natural choice for PHITS workflows, where `Hit.L` is directly the deposited energy (`Edep_MeV`).
- `strategy = "ELUT"` uses calibrated `E(L)` LUTs to invert light output to deposited energy.
- `strategy = "ToF"` and `"FixedEn"` are intended for timing-based neutron experiments and share the same neutron kinematics machinery.

**Bundled LUTs (design + intent):**

For common NOVO scintillators like **M600** and **OGS**, the canonical proton and carbon LUTs will ship as part of the `ngimager` package (e.g. under `ngimager/data/lut/...`) and be used as **defaults**. The intent is:

- When running from a PyPI install (`pip install ngimager`), users should be able to set:

  ```toml
  [energy]
  strategy = "ELUT"
  # no explicit lut_paths for M600 / OGS needed in the simplest case
  ```

- `build_lut_registry` will then:
  - consult `energy.lut_paths` if an explicit path is provided, **else**
  - fall back to internal, bundled LUTs for known materials (M600, OGS, etc.).

Custom materials or non-standard LUTs can override the defaults by explicitly specifying `energy.lut_paths.<MATERIAL>.<SPECIES>`.

For neutron events:

- When `force_proton_recoils = false` and a prior is available, the neutron cone builder evaluates both H and C hypotheses and selects the one with the smaller Δ = |θ_calc − θ_est|.
- When `force_proton_recoils = true`, the code always assumes proton recoils (legacy behavior), bypassing H/C discrimination.

#### `[plane]`

Imaging plane specification, matching the `Plane.from_cfg(...)` constructor used in the pipeline:

```toml
[plane]
origin = [0.0, 0.0, 0.0]   # [x,y,z] of plane origin (cm)
normal = [1.0, 0.0, 0.0]   # plane normal (unit-ish)

# u coordinate along an orthonormal basis vector in the plane
u_min = -50.0
u_max =  50.0
du    =   0.5

# v coordinate along the second orthonormal basis vector in the plane
v_min = -50.0
v_max =  50.0
dv    =   0.5
```

- `Plane` derives an orthonormal `(û, v̂)` basis from `normal`.
- The image shape is `(nv, nu)`, where `nu = round((u_max - u_min)/du)` and likewise for `v`.

#### `[filters]`

Filtering is organized around the three physical objects in the pipeline:

1. **Hits** — `[filters.hits]`
2. **Events** — `[filters.events]`
3. **Cones** — `[filters.cones]`

Each supports:

- **universal** parameters (apply to both species), and
- **optional per-species overrides** under `.neutron` and `.gamma`.

Example:

```toml
[filters.hits]
min_light_MeVee    = 0.0
max_light_MeVee    = 200.0
bars_include       = []            # e.g. [200, 210, 220]
bars_exclude       = []
materials_include  = []            # e.g. ["M600", "OGS"]
materials_exclude  = []

[filters.hits.neutron]
# optional overrides for neutron hits

[filters.hits.gamma]
# optional overrides for gamma hits

[filters.events]
tof_window_ns    = [0.0, 30.0]     # global ToF window (if used)
min_L1_MeVee     = 0.0
min_L2_MeVee     = 0.0
min_L_any_MeVee  = 0.0

[filters.events.neutron]
# e.g. more aggressive L1/L2 thresholds for neutron scatters

[filters.events.gamma]
# e.g. tighter ToF window for gamma triples

[filters.cones]
max_delta_theta_deg     = 5.0      # global Δθ limit (deg), or null for no cut
max_incident_energy_MeV = 0.0      # 0 or null = no limit, >0 = physical cap

[filters.cones.neutron]
# max_delta_theta_deg     = 3.0
# max_incident_energy_MeV = 250.0

[filters.cones.gamma]
# max_delta_theta_deg     = 8.0
# max_incident_energy_MeV = 100.0
```

At runtime, the code resolves an *effective* value for each quantity by:

1. checking the per-species override (if present),
2. otherwise falling back to the universal value, and
3. otherwise disabling that specific cut.

Counters are maintained for each stage and, when relevant, per species (e.g. `cones_rejected_delta_theta_n`, `events_rejected_L_any_min_g`). These show up both in the CLI diagnostics and as datasets under `/meta/counters` in the HDF5 file.

#### `[fast]` (fast-mode presets)

Fast mode is enabled via:

- `run.fast = true` in the TOML, or
- the `--fast` CLI flag (which sets `run.fast = true` for that run).

The `[fast]` block defines **only** those settings that differ from the defaults when `run.fast` is on. Typical example:

```toml
[fast]
# More aggressive light thresholds
min_L1_MeVee    = 0.8   # MeVee, first neutron scatter
min_L2_MeVee    = 0.4   # MeVee, second neutron scatter
min_L_any_MeVee = 0.2   # MeVee, any gamma or neutron deposit

# Example cone cap for faster SBP:
# max_cones = 20000     # stop after this many cones are imaged

# Coarsen the imaging plane by an integer factor:
# du' = du * plane_downsample, dv' = dv * plane_downsample
plane_downsample = 1    # 1 = no change, 2 = 2× coarser in each dimension
```

Implementation:

- When `run.fast` is true, the pipeline applies these presets *on top of* the base `[filters.events]`, `[filters.hits]`, `[run]`, and `[plane]` values (e.g. by overriding `min_L*`, `max_cones`, and by effectively scaling `du`/`dv` by `plane_downsample` when building the fast-mode plane).
- Fast mode is purely a **performance / diagnostics** knob; it does not change the underlying physics.

#### `[prior]`

Source priors:

```toml
[prior]
type = "point"  # or "line"

[prior.point]
r0 = [x_cm, y_cm, z_cm]

[prior.line]
# either origin + direction ...
r0        = [x0_cm, y0_cm, z0_cm]
direction = [dx, dy, dz]
# ... or two points
# p0        = [x0_cm, y0_cm, z0_cm]
# p1        = [x1_cm, y1_cm, z1_cm]
```

- This nested-table style (`[prior.point]` / `[prior.line]`) is the **preferred** form going forward.
- Older configs may still use inline keys like `prior.point.r0` or even `prior.point = [...]`; the implementation may continue to accept those for backward compatibility, but new configs should follow the nested-table pattern.

`make_prior(cfg.prior, plane)` creates a `Prior` instance (or a degenerate "plane center" prior if none is configured).

#### `[uncertainty]`

Resolution models (for future extensions):

```toml
[uncertainty.energy]
# placeholder: parameters for σ_E(E)

[uncertainty.time]
# parameters for σ_t

[uncertainty.doi]
# parameters for DOI resolution
```

These are currently either unused or used only for diagnostics; later they can feed into uncertainty-aware imaging.

#### `[vis]` and `[hdf5]`

Visualization / PNG export and HDF5 metadata:

```toml
[vis]
export_png_on_write = true
summed_dataset      = "/images/summed/all"  # which dataset to render
png_dir             = "images/"
colormap            = "viridis"             # used by visualization, not SBP

[hdf5]
include_config      = true                  # (design) store TOML as /meta/config_toml
include_repo_bundle = true                  # (design) store code snapshot if available
```

- `vis.hdf.save_summed_png` uses `[vis]` to decide what PNGs to write.
- The `[hdf5]` block is a **design hook**: the current implementation always writes the config TOML text and, when available, a repo bundle; in the future, these toggles will gate that behavior.

---

## 4. Input Adapters: Raw Events and Hits

Adapters translate source-specific raw data into a **canonical representation**.

Material assignment is governed solely by `[detectors.material_map]` and `[detectors.default_material]`. The adapter layer must not accept a distinct default material from `[io.adapter]`; detector-level settings are authoritative.

### 4.1. Raw Events

Adapters emit **raw events**:

- Each raw event corresponds to:
    - a PHITS history line (`usrdef.out`), or
    - an experimental trigger/coincidence window in ROOT, etc.
- Each raw event contains **all hits in the coincidence window**:
    - multiple detectors, possibly multiple particles (n + γ), and noise.
- Each raw event is represented as a small dict-like object, with at least:
    - `hits`: a list of canonical `Hit` objects (preferred) or normalized hit dicts,
    - optional metadata such as `event_type` (for PHITS-only hints), run IDs, etc.

Conceptual interface:

```python
class BaseAdapter(ABC):
    def iter_raw_events(self, path: Path) -> Iterable[dict]:
        """
        Yield raw events. Each event dict contains a 'hits' list
        (canonical Hit objects or equivalent) and any adapter-specific metadata.
        """
        ...
```

- `PHITSAdapter` implements `iter_raw_events` and is now the canonical path for PHITS data.
- ROOT and other adapters may still expose `iter_events(...)` that directly yield typed `Event` objects; the long-term goal is for all real sources to support `iter_raw_events(...)`, with shaping/typing handled in the common pipeline.

### 4.2. Hit Construction and Hit-Level Filters

From each raw event:

1. **Adapters normalize to canonical `Hit` objects**

    - For PHITS and ROOT, raw per-hit records are converted inside the adapter into `physics.hits.Hit`.
    - Adapters must ensure:

      ```python
      Hit.type     in {"n", "g", "UNK"}  # particle species
      Hit.material = scintillator name   # from [detectors.material_map]
      ```

    - Legacy event-level fields like `event_type` may still be used as **hints** (especially in PHITS simulation data), but the long-term goal is for shaper and filters to rely primarily on `Hit.type`.

2. **Apply hit-level cuts**

    ```python
    hits = list(ev["hits"])  # canonical Hit objects
    hits = apply_hit_filters(hits, cfg.filters.hits, counters, particle_type=et)
    ```

    Where `counters` includes entries such as:

    - `hits_total`
    - `hits_total_n`, `hits_total_g`
    - `hits_rejected_threshold`, etc.

3. **Check reconstructability**

    Determine whether the event is still *potentially reconstructable*:

    - Count surviving neutron hits, gamma hits, etc.
    - If there are fewer than 2 candidate neutron hits and fewer than 3 candidate gamma hits, the raw event can be discarded early.

    ```python
    if not is_reconstructable(hits, cfg.filters.events, counters, event_type=et):
        counters["raw_events_rejected_unreconstructable"] += 1
        continue
    ```

This approach obeys the real acquisition model:

- Raw data is already in “coincidence windows” (crude events).
- Acquisition thresholds are looser; post-processing thresholds refine which hits (and events) we keep.

### 4.3 PHITS Adapter

The `PHITSAdapter` implements `iter_raw_events` and is the canonical source for PHITS `usrdef` output.

- `Hit.L` = `Edep_MeV` directly.
- `event_type` hints (“neutron” / “gamma”) are normalized.
- PHITS has no PSD; PSD-related hit filters are automatically ignored.

### 4.4 ROOT NOVO DDAQ Adapter 

The ROOT adapter (`input_format = "root_novo_ddaq"`) is now operational and includes:

- Complete parsing of NOVO DDAQ coincidences.
- Extraction of:
    - hit positions (detector-frame)
    - hit times
    - light quantities (elong / MeVee)
    - PSD values → exposed as `Hit.extras["psd"]`
- Hit-level PSD filtering through `[filters.hits]` and `[filters.hits.neutron/gamma]`.
- Canonical `Hit` objects based on detector coordinates.
- Event-level timing preserved as emitted by DDAQ.
- Metadata passthrough:
    - The `meta` ROOT TTree is parsed and stored in HDF5 as `/meta/root/*` with a structured layout.
    - Detector dimension, position, rotation, calibration files, global time offsets, etc. are preserved.
- Fully integrated with the new `[detectors.geometry]` transform system:
    - Per-detector corrections
    - Global detector-to-world transform

### 4.5 HDF5 Restart Adapter (Planned)

A future adapter will support `input_format = "hdf5_ngimager"` for restarting the pipeline at arbitrary stages without recomputing hits or cones.


### 4.6 Detector Geometry Transforms (New)

Experimental and simulation datasets may provide hit coordinates in reference frames other than the world/imaging frame. `ngimager` now supports a general-purpose, adapter-agnostic transform stage before filters:

1. **Per-detector transforms**  
   From `[[detectors.geometry.detectors]]`, each detector element may specify:
   - `id`
   - `origin_cm`
   - `rotation_deg`

   These apply a rigid transform mapping bar-local coordinates into the detector frame.  
   Useful for correcting known misplacements after data acquisition.

2. **Global detector-frame → world-frame transform**  
   From `[detectors.geometry.frame]`:
   - `origin_cm`
   - `rotation_deg`

   This maps detector-frame hit coordinates into world/imaging coordinates.

Transforms follow:

    r_det_corrected = R_xyz(rot_det) @ r_local + origin_det
    r_world = R_xyz(rot_glob) @ r_det_corrected + origin_glob




---

## 5. From Hits to Events: Shaping and Typing

After hit-level filtering, we convert surviving raw events into **imaging-viable Events**.

### 5.1. Shaping

The **shaper** decides how to partition hits within each raw event into candidate events suitable for imaging.

Responsibilities:

- Separate neutron from gamma hits.
- Handle higher multiplicities:
    - Multiple neutron candidates in one window → potentially multiple 2-hit neutron events.
    - Gamma hits ≥ 3 → one or more 3-hit gamma events.
- Handle multi-particle mixed windows by splitting them into neutron and gamma events.

Current interface (PHITS path):

```python
from ngimager.filters.shapers import shape_events_for_cones, ShapeConfig

shaped_events, shape_diag = shape_events_for_cones(
    raw_events_after_filters,
    ShapeConfig(),
    counters=counters,
)
```

- `raw_events_after_filters` is a list of event dicts (each with `hits` and optional metadata).
- `ShapeConfig` holds shaper-level knobs (e.g. gamma multiplicity policy).
- `shape_diag` exposes basic diagnostics (counts of shaped neutron/gamma events).

`ShapedEvent` is a lightweight structure holding:

- a list of `Hit`s,
- a particle type (`"n"` or `"g"`),
- basic metadata (original raw event index, etc.).

Counters at this stage might include:

- `shaped_events_total`
- `shaped_events_n`
- `shaped_events_g`
- `raw_events_rejected_shaping` (if nothing usable can be formed)

### 5.2. Typed Events

Next, we convert `ShapedEvent` objects into physics-aware event classes:

- `NeutronEvent`: for 2-hit neutron events.
- `GammaEvent`: for 3-hit Compton events.

Current interface:

```python
from ngimager.physics.events import NeutronEvent, GammaEvent
from ngimager.filters.to_typed_events import shaped_to_typed_events

events = shaped_to_typed_events(
    shaped_events,
    order_time=True,  # enforces time-ascending hits inside each event
)
```

**Invariant:** `NeutronEvent` and `GammaEvent` **always** hold references to their constituent `Hit` objects. This guarantee is reflected both in memory and in the HDF5 representation, where cones can be traced back to events and events to hits via indices.

Counters incremented here include:

- `events_typed_total`
- `events_typed_n`
- `events_typed_g`

---

## 6. Event-Level Filters

Once we have typed events, we apply **event-level cuts** based on timing and light thresholds.

Examples (current implementation):

- Neutron events:
    - Event-level time-of-flight window: `tof_window_ns = [t_min, t_max]`.
    - Minimum `L1`, `L2` (first and second neutron scatters) in MeVee.
- Gamma events:
    - The same `tof_window_ns` (if configured).
    - Minimum `L_any` across all hits (shared `min_L_any_MeVee` cut).

Centralized interface:

```python
from ngimager.filters.event_filters import apply_event_filters

events = apply_event_filters(
    events,
    cfg.filters.events,
    counters,
)
```

Counters include, for example:

- `events_total_for_filters`
- `events_after_filters`
- `events_rejected_tof_window`
- `events_rejected_L1_min_n`
- `events_rejected_L2_min_n`
- `events_rejected_L_any_min_g`

This keeps event selection logic **configurable and testable** and avoids hidden “magic cuts” in cone-building code.

---

## 7. Energy Strategies for Neutrons (and PHITS Edep)

The **energy strategy** determines how the first-scatter deposited energy is obtained for neutron events, and how that energy is interpreted for neutron kinematics.

At present, energy strategies are defined primarily for **neutrons**. Gamma cones derive their kinematics directly from deposited energies plus geometry (see `physics.kinematics`).

### 7.1. Strategy Interface

Concrete strategies live in `ngimager.physics.energy_strategies` and all implement a common interface, e.g.:

```python
class EnergyStrategy:
    """Compute first-scatter energy (and optional σ) for neutron events."""
    name: str

    def first_scatter_energy(
        self,
        h1: Hit,
        h2: Hit | None,
        material: str,
        species: str = "proton",
    ) -> tuple[float, float | None]:
        """
        Return (Edep1_MeV, sigma_E_MeV_or_None) for the first scatter.
        """
        ...
```

Factory:

```python
from ngimager.physics.energy_strategies import make_energy_strategy
from ngimager.io.lut import build_lut_registry

lut_registry = build_lut_registry(cfg.energy.lut_paths)
energy_model = make_energy_strategy(cfg.energy, lut_registry=lut_registry)
```

Usage (inside neutron cone building):

```python
Edep1_MeV, _ = energy_model.first_scatter_energy(
    h1,
    h2,
    h1.material,
    species="proton" or "carbon",
)
```

The cone builders then combine this deposited energy with the neutron kinematics (`neutron_theta_from_hits`, etc.) to obtain the cone opening angle and the **incident neutron energy** used for filtering and HDF5 output.

### 7.2. Light-Based Strategy (E(L) LUT)

For `strategy = "ELUT"`:

- The energy strategy uses calibrated `E(L)` lookup tables per material (OGS, M600, …).
- LUTs live in `.npz` files and are loaded via `io.lut.LUT`.
- Given a measured light output `L` for the first scatter, the LUT returns deposited energy for a chosen recoil species (proton or carbon).

This directly yields the recoil energy required for neutron kinematics, and the cone builder can then compute the incident energy `En` via:

- `Eprime` (post-scatter neutron KE) from ToF between hits 1 and 2, and
- `En = Eprime + Edep1`.

Bundled LUTs for M600 and OGS follow the design described in §3.1 `[energy]`: if no explicit `lut_paths` entry is provided for these materials, the strategy will look up internal package resources.

### 7.3. Edep Strategy (PHITS)

For `strategy = "Edep"`:

- The adapter sets `Hit.L` to be **deposited energy in MeV** (e.g. PHITS `Edep_MeV`).
- The energy strategy simply interprets `L` as deposited energy, possibly with material-aware handling.
- This keeps PHITS workflows simple and avoids double-conversion.

This is the default in `phits_usrdef_simple.toml` and is the basis for current neutron and gamma PHITS tests.

### 7.4. ToF and Fixed-Incident Strategies

For timing-based experiments, two strategies exist:

- `strategy = "ToF"`:
    - Uses ToF between a start detector and the first neutron hit to estimate the **before-scatter** energy.
    - Uses ToF between the first and second neutron hits to estimate the **after-scatter** energy.
    - Recoil energy is `E_recoil = E_before − E_after`.

- `strategy = "FixedEn"`:
    - Uses a fixed initial incident energy `fixed_En_MeV` (e.g. DT source).
    - Still uses ToF between the first and second neutron hits for `E_after`.
    - Recoil energy is `E_recoil = E_before_fixed − E_after`.

These strategies share the same kinematic machinery in `physics.kinematics`. They are wired but still need dedicated validation on real or legacy datasets (see §16.5).

### 7.5. Proton vs Carbon Recoil Hypotheses

Elastic scattering in organic scintillators can occur on hydrogen or carbon nuclei. The neutron cone builder (`build_cone_from_neutron`) uses the energy strategy and kinematics to evaluate:

1. A proton-recoil hypothesis (`scatter_nucleus = "H"`).
2. A carbon-recoil hypothesis (`scatter_nucleus = "C"`).

For each branch:

- Obtain `Edep1_MeV` via `energy_model.first_scatter_energy(...)` with `species="proton"` or `"carbon"` (for ELUT/Edep).
- Compute `θ` via `neutron_theta_from_hits(...)` using the appropriate mass ratio `A`.
- Build a candidate `Cone` and ensure it points toward the imaging plane.
- If a prior is available, compute Δ = |φ − θ| where:
    - φ is the angle between the cone axis and the direction toward the prior/plane center,
    - θ is the kinematically derived half-angle.

Selection logic:

- If `force_proton_recoils = true` or no plane is available, only the proton branch is built.
- Otherwise, both branches are evaluated; the branch with the smaller Δ is selected (if both are physical).
- The selected branch’s **recoil species** is encoded as `recoil_code` in `/cones/recoil_code` (0=unknown, 1=proton, 2=carbon).

The **incident neutron energy** used for cone-level filtering and HDF5 output (`/cones/incident_energy_MeV`) is derived from this same kinematic chain (`Eprime + Edep1` for the chosen branch).

### 7.6. Gamma Energies

For gamma events:

- Deposited energies at the first and second scatters (dE₁, dE₂) are taken from `Hit.L` (which is `Edep_MeV` for PHITS, or a calibrated energy for real data).
- Incident gamma energy `Eg` and scatter angles are computed via:

    - `compton_incident_energy_from_second_scatter(dE1, dE2, θ₂)`  
    - `compton_theta_from_energies(Eg, Egp)`

These functions live in `physics.kinematics` and mirror the NOVO primer. The resulting `Eg` is also stored in `/cones/incident_energy_MeV` for gamma cones.

---

## 8. Priors and Sequencing Logic

### 8.1. Source Priors

Priors reflect what we know about where radiation is coming from:

- `PointPrior`: for point sources (e.g., Cf-252, DT tubes).
- `LinePrior`: for extended line-like sources (e.g., proton beams, irradiation lines).

Created via:

```python
prior = make_prior(cfg.prior, plane)
```

They are used in:

- **Neutron cone interpretation** (proton vs carbon branch selection).
- **Gamma permutation selection** (which hit ordering best matches the prior).
- Cone-level Δθ filtering (`passes_delta_theta_cut`).

If no prior is configured, the imaging plane center is used as a degenerate prior.

### 8.2. Sequencing and Event Interpretation

For gamma events:

- A 3-hit gamma event has 6 possible permutations (orderings).
- Many permutations are physically impossible (e.g. negative or out-of-domain Compton angles).
- The gamma cone builder (`build_cone_from_gamma`):
    - Tests all permutations.
    - Uses kinematics and priors to score each viable permutation.
    - Selects the permutation with minimal Δ = |φ − θ| as the final cone for that event.

For neutron events:

- The neutron cone builder (`build_cone_from_neutron`) evaluates proton vs carbon hypotheses as described in §7.5.
- Each hypothesis yields a different half-angle and, potentially, a different incident energy.

**Design note:**  

The long-term goal is to expose a unified API:

- `enumerate_candidate_cones(event, prior, energy_model)` → `List[CandidateCone]`
- `select_cone(candidates, filters, prior)` → `Optional[Cone]`

At present, this candidate enumeration + selection logic is **embedded inside** `build_cone_from_neutron` and `build_cone_from_gamma`. The roadmap (§16.4, §16.6) tracks future work to factor this out into dedicated functions and to record richer per-event metadata (e.g. gamma permutation choice).

---

## 9. Cone Representation

Cones are constructed from events, energy strategy, kinematics, and priors.

The imaging backend uses a minimal `Cone` dataclass defined in `ngimager.imaging.sbp`:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Cone:
    apex: np.ndarray       # (3,) apex position [cm]
    direction: np.ndarray  # (3,) unit axis direction
    theta: float           # half-angle [rad]
```

These parameters **fully define the analytic cone**:

- Points `x` on the cone satisfy:
  ```text
  (x - apex) · direction = |x - apex| cos(theta)
  ```

All other metadata (event index, particle species, recoil code, incident energy, etc.) is stored **alongside** the cone arrays in HDF5:

- `/cones/event_index`
- `/cones/species`
- `/cones/recoil_code`
- `/cones/incident_energy_MeV`

This keeps the `Cone` dataclass **imaging-method agnostic**, while still embedding all necessary geometric information.

Counters in the cones stage include:

- `events_cone_build_attempted`, `events_cone_build_attempted_n`, `events_cone_build_attempted_g`
- `events_cone_build_success`, `events_cone_build_failed`, with `_n` / `_g` variants
- `cones_kept`, `cones_kept_n`, `cones_kept_g`
- `cones_n_recoil_proton`, `cones_n_recoil_carbon`, `cones_n_recoil_unknown`
- `cones_checked_delta_theta*` and `cones_rejected_delta_theta*`

---

## 10. Imaging Back-End (SBP and Future Methods)

The first imaging back-end is **simple back projection (SBP)**, but the architecture is intended to support others later.

### 10.1. Imaging Plane

`ngimager.geometry.plane.Plane` encodes:

- origin `origin` (3,)
- normal vector `n̂`
- orthonormal basis vectors `(û, v̂)` in the plane
- mappings:
    - from 3D coordinates to plane `(u, v)`, and
    - from `(u, v)` back to 3D.

It is constructed from `[plane]` config entries via `Plane.from_cfg(...)`.

### 10.2. SBP Rasterization

`ngimager.imaging.sbp.reconstruct_sbp`:

- Takes a collection of **selected** `Cone` objects and a `Plane`.
- Computes cone–plane intersections analytically.
- Rasterizes each intersection into the plane’s pixel grid.

SBP now supports two rasterization engines (selectable via `[run].sbp_engine`):

- `"scan"` (default) — A full matrix-math engine that solves the conic equation on every horizontal and vertical pixel-centered scan line. This method guarantees continuous ellipse/hyperbola arcs and is resolution-adaptive.
- `"poly"` — The legacy-like parametric perimeter sampler.  
  Fast, but may produce dotted arcs at high resolutions.

When `[run].jit = true` and Numba is installed, the scan engine uses a JIT-compiled inner loop, providing large speedups while keeping the exact same mathematical behavior.

The imaging stage chooses the engine with:

    sbp_engine = cfg.run.sbp_engine # "scan" or "poly"
    use_jit = cfg.run.jit # True/False


Current return type (`ReconResult`):

- `summed_n`: summed image from neutron cones.
- `summed_g`: summed image from gamma cones.
- `summed_all`: `summed_n + summed_g` if both exist, else `None`.
- `lm_indices`: optional list-mode data, mapping cone indices to pixel indices (see §12.1).

SBP is imaging-method-agnostic in the sense that it only depends on the generic `Cone` representation and `Plane`.

### 10.3. Future Imaging Methods

The long-term design is to have an `imaging` dispatcher:

```python
def image_cones(cones, plane, cfg: Config, counters: dict):
    method = cfg.imaging.method  # e.g. "sbp" now, "mlem" later
    if method == "sbp":
        return reconstruct_sbp(cones, plane, cfg.run.list, ...)
    elif method == "mlem":
        ...
```

Currently, the pipeline calls `reconstruct_sbp` directly with parameters sourced from `[run]`. A dedicated `[imaging]` section and multiple back-ends (e.g. MLEM) are on the roadmap.

---

## 11. Fast and List options: Modifiers of the Default Pipeline

There is a **single unified pipeline**. “Fast” and “list” are orthogonal **modifiers** of its behavior, not separate pipelines or alternate control flows.

### 11.1. Default Behavior

Without any modifiers:

- Reasonable thresholds and limits as specified in `[filters]` and `[run]`.
- Full SBP summed images (`/images/summed/n`, `/images/summed/g`, `/images/summed/all`).
- Hits, events, and selected cones stored in HDF5.
- No per-cone pixel footprints.

### 11.2. Fast Mode (`run.fast`)

When `fast = true` (TOML) or `--fast` is passed on the CLI:

- The pipeline applies stricter light thresholds from `[fast]` (if present).
- `run.max_cones` and/or `run.max_events` can be set to moderate values.
- Optionally, the imaging plane can be coarsened by `fast.plane_downsample` (see §3.1).

Fast mode is designed for **quick feedback** during experiment setup and exploratory runs. It does **not** change physics; it simply yields fewer events/cones and a potentially coarser image.

### 11.3. List Mode (`run.list`)

When `list = true` (TOML) or `--list` is passed on the CLI:

- SBP still produces the same summed images.
- In addition, it computes, for each cone that intersects the plane, the set of pixel indices where that cone contributed counts.
- These per-cone footprints are stored in `/lm/cone_pixel_indices`:

    - Each row holds `(cone_index, flat_pixel_index)`.
    - `flat_pixel_index = v * nu + u` (row-major flattening).

- A small “survival table” `/lm/event_survival` links:

    - `event_index` (row in `/lm/...`),
    - `first_cone_index` (or `-1` if no cone was built),
    - `first_imaged_cone_index` (or `-1` if the cone missed the plane).

Fast and list can be combined (e.g. fast thresholds but list-mode storage). The combination is fully defined, though less common in practice.

---

## 12. HDF5 Data Model and Partial Pipelines

The HDF5 format is the primary output. It is designed to support:

- **Full pipeline outputs** (hits, events, cones, images).
- **Optional list-mode footprints**.
- **Traceability** from hits to pixels and back.
- **Future** support for partial pipelines and restart/resume.

### 12.1. HDF5 layout and traceability

The HDF5 format is designed so that you can trace:

> **hit → event → cone → pixels**

and back again.

Main groups:

- `/meta` — geometry, run flags, config snapshot, counters, README.
- `/images/summed` — species-separated and combined SBP images.
- `/cones` — per-cone geometry + classification.
- `/lm` — list-mode per-event/per-hit data and mappings.

Key pieces:

- `/meta/config_toml` — full TOML config text used for this run.
- `/meta/readme` — short README describing the layout and pointing to online docs.
- `/meta/counters/*` — one small dataset per counter (e.g. `s1_raw_events_total`), named with stage prefixes so HDFView sorts them in roughly pipeline order.

Cones:

- `/cones/cone_id`              : `[N_cones]` uint32
- `/cones/apex_xyz_cm`          : `[N_cones, 3]` float32
- `/cones/axis_xyz`             : `[N_cones, 3]` float32
- `/cones/theta_rad`            : `[N_cones]` float32
- `/cones/incident_energy_MeV`  : `[N_cones]` float32  
  (incident neutron or gamma energy used in the kinematic calculation)
- `/cones/species`              : `[N_cones]` uint8 (`0=neutron`, `1=gamma`)
- `/cones/recoil_code`          : `[N_cones]` uint8  
  (`0=unknown/NA`, `1=proton`, `2=carbon`)
- `/cones/event_index`          : `[N_cones]` uint32  
  (row index into `/lm/...` arrays)
- `/cones/species_labels`       : `[2]` string array legend for `species`
- `/cones/recoil_code_labels`   : `[3]` string array legend for `recoil_code`

List mode and events:

- `/lm/event_type`              : `[N_events]` uint8 (`0=neutron`, `1=gamma`)
- `/lm/event_type_labels`       : `[2]` string array legend for `event_type`
- `/lm/event_meta_run_id`       : `[N_events]` int32 (0 for PHITS runs)
- `/lm/event_meta_file_ix`      : `[N_events]` int32 (0 for single-file runs)

Per-hit quantities (currently supporting up to 3 hits per event for both neutrons and gammas):

- `/lm/hit_pos_cm`              : `[N_events, 3, 3]` float32 (`[event, hit_idx, xyz]`)
- `/lm/hit_t_ns`                : `[N_events, 3]` float32
- `/lm/hit_L_mevee`             : `[N_events, 3]` float32
- `/lm/hit_det_id`              : `[N_events, 3]` int32
- `/lm/hit_material_id`         : `[N_events, 3]` int16

Material vocabulary:

- `/lm/material_id_labels`      : `[N_materials]` string array mapping material IDs to names  
  (e.g., `["M600", "OGS"]`)

List-mode cone footprints (only when `run.list = true`):

- `/lm/cone_pixel_indices`      : `[K, 2]` uint32, each row `(cone_index, flat_pixel_index)`.

Survival table:

- `/lm/event_survival`          : `[N_events, 3]` int32, columns:
  - `event_index`              — row in `/lm/...`
  - `first_cone_index`         — cone index built from this event or `-1`
  - `first_imaged_cone_index`  — cone index that actually intersected the plane or `-1`

Together, `/lm/event_survival`, `/cones/event_index`, and `/lm/cone_pixel_indices` provide a complete map from any pixel in the list-mode data back to the exact cone, event, and hit data that generated it.

### 12.2. Writing at Multiple Stages (Design)

The long-term plan is for the pipeline to support **early exits** and **partial writes** governed by `[pipeline].until`:

- For `until = "hits"`:
    - Write hit-level datasets only.
- For `until = "events"`:
    - Write hits + events.
- For `until = "cones"`:
    - Write hits + events + cones.
- For `until = "image"` (current behavior):
    - Write hits + events + cones + images.

At the moment, the main production path is `until = "image"`. Intermediate-stage writing will be implemented once HDF5 restart logic is in place (see §12.3).

### 12.3. Resuming from HDF5 (Planned)

When `input_format = "hdf5_ngimager"` is fully implemented, the intended behavior is:

- If `/cones/*` exists but `/images/summed/*` does not:
    - Start at cones → images.
- If `/lm/*` events exist but `/cones/*` does not:
    - Start at events → cones → images.
- If only hit-level datasets exist:
    - Start at hits → events → cones → images.

This will enable:

- Re-running imaging with different plane/imaging settings without recomputing events/cones.
- Upgrading non-list-mode HDF5 files to list-mode by rerunning only SBP.

**Current status:** HDF5 restart is still on the roadmap and not yet implemented. The only supported `input_format` values in production pipelines are `"phits_usrdef"` (and, in future, `"root_novo_ddaq"`).

### 12.4. Converting Between List-Mode and Non-List-Mode Outputs

Conceptually, for a completed ngimager HDF5 file:

- **List-mode → non-list-mode**  
  Delete list-mode-specific datasets (`/lm/cone_pixel_indices` and `/lm/event_survival` if desired). Hits, events, cones, and summed images remain intact.

- **Non-list-mode → list-mode** (planned)  
  Once HDF5 restart is available, re-run the pipeline with:
    - `input_format = "hdf5_ngimager"`
    - `[pipeline].until = "image"`
    - `run.list = true`

  The imaging stage would be re-executed to populate `/lm/cone_pixel_indices` without rebuilding hits/events/cones.

Until restart is implemented, list-mode has to be enabled at the time of the original run.

---

## 13. Diagnostics, Logging, and Counters

Diagnostics are controlled by `run.diagnostics_level`:

- `0`: no diagnostic messages (except fatal errors).
- `1`: minimal messages indicating:
    - Stage entry/exit (hits/events/cones/images).
    - Object counts (e.g., number of events, cones).
    - A final per-stage counter summary.
- `2`: verbose messages with:
    - Detailed adapter parsing notes.
    - Sample events/cones.
    - More granular information for debugging.

Example usage pattern:

```python
if cfg.run.diagnostics_level >= 1:
    print(f"[s1] raw_events_total={counters.get('raw_events_total', 0)}")

if cfg.run.diagnostics_level >= 2 and j < 5:
    print(f"\t[shaper] example shaped event: {se}")
```

Level 1 is intended for humans during normal runs, level 2 for deeper debugging.

### 13.1. Counters

A shared `counters` dict is passed through the pipeline and records:

- Raw events:
    - `raw_events_total`
    - `raw_events_rejected_unreconstructable`
- Hits:
    - `hits_total`
    - `hits_total_n`, `hits_total_g`
    - `hits_rejected_threshold`, plus per-species variants
- Shaped + typed events:
    - `shaped_events_total`, `shaped_events_n`, `shaped_events_g`
    - `events_typed_total`, `events_typed_n`, `events_typed_g`
- Event-level filters:
    - `events_total_for_filters`
    - `events_after_filters`
    - `events_rejected_tof_window`
    - `events_rejected_L1_min_n`, `events_rejected_L2_min_n`
    - `events_rejected_L_any_min_g`
- Cones:
    - `events_cone_build_attempted`, with `_n` / `_g` variants
    - `events_cone_build_success`, `events_cone_build_failed`, with `_n` / `_g`
    - `cones_kept`, `cones_kept_n`, `cones_kept_g`
    - `cones_n_recoil_proton`, `cones_n_recoil_carbon`, `cones_n_recoil_unknown`
    - `cones_checked_delta_theta*` and `cones_rejected_delta_theta*`
- Imaging:
    - `cones_imaged_total`
    - `cones_imaged_n`, `cones_imaged_g`
    - (optional) SBP timing and pixel-touch statistics in future.

**Pattern:** wherever it is conceptually meaningful to separate by particle type, we aim for:

- a `_total` counter (neutrons + gammas),
- a `_n` counter (neutrons only),
- a `_g` counter (gammas only),

even if not all names are fully standardized yet.

Counters are:

- printed in a stage-wise summary at the end of the run (for `diagnostics_level >= 1`), and
- stored under `/meta/counters` in the HDF5 file for every run.

---

## 14. Legacy Timing and Future-Proofing

Legacy timing-based workflows are supported via the energy strategies (see §7):

- **ToF-based paths**:
    - Maintained via `strategy = "ToF"` and corresponding ToF parameters.
    - Useful for setups using a start detector and ToF for incident energy.
- **Fixed incident energy**:
    - `strategy = "FixedEn"` for known beam energies.
- **Light-based E(L) LUT**:
    - `strategy = "ELUT"` using calibrated E(L) curves.
- **PHITS Edep**:
    - `strategy = "Edep"` for MC workflows where `Hit.L` is directly `Edep_MeV`.

All these behaviors are encapsulated in `energy_strategies` and configured via `[energy]`, not scattered across the pipeline.

Uncertainty models (energy, DOI, timing) are designed to be pluggable; they will be integrated once supporting data (resolution curves, etc.) is available.

---

## 15. Coding Style and Conventions

Guidelines:

- Use **dataclasses** for structured data (`Hit`, `NeutronEvent`, `GammaEvent`, `Cone`).
- Use **type hints** extensively.
- Keep modules focused:
    - `physics.*`: physical models, kinematics, priors, event definitions.
    - `geometry.*`: planes and coordinate transforms.
    - `imaging.*`: turning cones into images.
    - `io.*`: adapters, HDF5 I/O, LUT loading.
    - `filters.*`: hit/event/cone shaping and selection.
    - `pipelines.*`: orchestration and CLIs.
    - `sim.*`: not used in the production pipeline; synthetic data belongs in examples/tests.
- NOVO-specific quirks live in:
    - `io.adapters`, `tools/*`, and configuration.
    - not in the physics/imaging core.

The aim is for someone familiar with the legacy script to be able to read this code and see where each piece migrated, and for new contributors to navigate via module names + this document.

### 15.5 Current Implementation Status (Updated)

As of the current ngimager snapshot:

- The unified pipeline (`pipelines.core.run_pipeline`) is stable and serves as the only supported orchestration path.

- **ROOT adapter fully implemented**:
  - Canonical Hit construction
  - PSD extraction and hit-level PSD filtering
  - Coordinate positions in detector frame
  - Full metadata passthrough (`/meta/root/*`)
  - Integration with detector/world transform system

- **Detector geometry transforms implemented**:
  - Per-detector corrections (`[[detectors.geometry.detectors]]`)
  - Global detector-frame → world-frame transform (`[detectors.geometry.frame]`)
  - Transform application occurs before hit-level filters

- **Energy strategies**:
  - E(L) LUT, Edep (PHITS), FixedEn → fully implemented
  - FixedEn validated on real-data DT 14.8 MeV dataset (matching legacy behavior)
  - ToF partially implemented, pending validation
  - Up-scattering rejection added to FixedEn and ToF branches

- **Hit- and event-level filters**:
  - Complete, including PSD, light thresholds, bar/material include/exclude lists
  - Reconstructability logic applied early

- **Neutron cone building**:
  - Full proton vs carbon recoil hypothesis testing
  - Prior-based Δ selection implemented
  - Directionality (cone pointing toward plane) enforced
  - Incident energy calculation consistent with primer and legacy code
  - H/C recoil code exported to HDF5

- **Gamma cone building**:
  - All 6 permutations of 3-hit events tested
  - Kinematic rejection + prior-based Δ selection
  - Directionality check implemented
  - Incident gamma energy calculation implemented and stored

- **Imaging**:
  - SBP “scan” engine validated against legacy
  - SBP “poly” engine implemented
  - Numba acceleration available
  - List-mode (“lm_indices” and “event_survival”) implemented

- **Counters**:
  - Extensive per-stage counters implemented
  - Counter summaries printed automatically at `diagnostics_level ≥ 1`
  - Counters written to `/meta/counters/*`

- **ROOT metadata passthrough**:
  - Fully implemented with structural normalization

Most remaining work centers on:
- HDF5 restart mode
- ToF validation
- Unified candidate-cone API
- Per-bar geometry and physical bar models


---

## 16. Roadmap (Refactor and Implementation Checklist)

This checklist tracks migration from the current state to the architecture described here.

### 16.1. Cleanup and Deletion of Redundant Paths

- [x] Remove any parallel PHITS→Hit/Event paths; ensure `PHITSAdapter` is the **only** canonical route for PHITS.
- [x] Deprecate and remove `sim.*` from the active pipeline; if synthetic capabilities are retained, move them to examples/tests.
- [x] Ensure only one `Hit` class exists and is used everywhere.

### 16.2. Adapters and Raw Events

- [x] `PHITSAdapter.iter_raw_events` returns canonical Hit objects grouped into raw coincidence windows.
- [x] Implement/clean up `ROOTAdapter.iter_raw_events` with `input_format = "root_novo_ddaq"` for the current acquisition system.
- [x] Implement the early `is_reconstructable` logic after hit-level filtering to discard unviable raw events, with appropriate counters.

### 16.3. Shaper and Typed Events

- [x] Make `shape_events_for_cones` the single entry point from raw-event hits to shaped events.
- [x] Make `shaped_to_typed_events` the only path to `NeutronEvent`/`GammaEvent`.
- [x] Ensure typed events always carry canonical `Hit` objects, and implement HDF5 round-trip storage.
- [ ] Audit hit- and event-level counters to consistently follow the `_total` / `_n` / `_g` naming pattern.

### 16.4. Filters, Priors, and Sequencing

- [x] Further centralize event and cone selection logic into `filters` modules, driven fully by `[filters]` config.
- [x] Ensure priors are only defined in `physics.priors` and configured via `[prior]`.
- [ ] Expose a unified candidate-cone API (`enumerate_candidate_cones`, `select_cone`) instead of embedding this logic in the cone builders.
- [ ] Implement storage of gamma sequencing choice and neutron recoil interpretation at the **event** level in HDF5 (cone-level recoil metadata already exists).

### 16.5. Energy Strategy Integration

- [x] Wire `make_energy_strategy(cfg.energy, lut_registry)` into the pipeline, with all neutron recoil energies obtained via this interface.
- [x] Integrate E(L) LUTs for OGS and M600 via `io.lut.LUT`, and ship canonical LUTs with the package for out-of-the-box usage.
- [x] `FixedEn` validated with real DT dataset, and `ELUT` also validated against it, reproduces results well.
- [ ] ToF validation pending

### 16.6. Cone Construction and Imaging

- [x] Ensure `physics.cones` provides the canonical functions for building cones from events (neutrons and gammas).
- [x] Confirm `imaging.sbp.reconstruct_sbp` works directly from the minimal `Cone` dataclass and `Plane`.
- [x] Implement optional per-cone sparse footprints used only when `run.list` is true, and validate basic end-to-end traceability through `/lm/cone_pixel_indices` and `/lm/event_survival`.
- [x] Ensure cone and imaging counters follow the `_n` / `_g` / `_total` naming pattern where meaningful, and record per-stage runtimes.
- [x] Implement gamma cone-building for 3-hit Compton events in `physics.cones`, including permutation testing and prior-based Δ = |φ − θ| scoring; treat this path as validated for PHITS-based model data, with ROOT validation pending.
- [x] Extend neutron cone construction to support explicit proton vs carbon candidate branches, including directionality checks analogous to the gamma `t_int > 0` test.
- [x] Implement combined n/g/all SBP images at the imaging stage (`/images/summed/n`, `/images/summed/g`, `/images/summed/all`).
- [x] Implemented dual SBP engines ("scan" and "poly") with configuration via `[run].sbp_engine` and optional Numba acceleration via `[run].jit`.
- [x] "scan" engine reproduces the legacy `image_via_matrix_math` behavior and guarantees continuous conic arcs on the imaging plane.

Neutron proton vs carbon recoil hypothesis handling:

- [x] Adopt primer-consistent neutron cone geometry (apex at h1, axis from h2→h1).
- [x] Specify species-aware recoil interpretation tied to energy strategies (ELUT, Edep).
- [x] Define candidate-cone generation for H and C branches with Δ-based prior scoring and directionality checks.
- [ ] Factor out H/C candidate enumeration and selection into the unified candidate-cone framework (rather than keeping it only inside `build_cone_from_neutron`).
- [x] Add cone-level recoil metadata in HDF5 (`/cones/recoil_code`) and track basic per-recoil counters.

Cone-level angle-difference filtering (Δθ):

- [x] Add `max_delta_theta_deg` to `[filters.cones]` with species-specific overrides.
- [x] Apply Δ = |φ − θ| thresholds during cone selection via `passes_delta_theta_cut`.
- [x] Add associated counters and expose the filtering results in `/meta/counters`.

### 16.6.1 Gamma Cone Status

The gamma cone construction path is now implemented and exercised through the unified pipeline:

- Gamma events are restricted to 3-hit Compton triples.
- All 6 permutations of the hit ordering are tested; kinematically invalid permutations are rejected.
- Each viable candidate cone undergoes an axis-toward-plane check to ensure only physically meaningful half-cones are retained.
- Candidate cones are scored using the prior-aware metric:

\[
\Delta = \left| \phi_{\text{prior}} - \theta_{\text{Compton}} \right|,
\]

where `φ_prior` is the angle between the cone axis and the direction toward the prior (or plane center), and `θ_Compton` is the Compton scatter angle implied by the permutation.

The candidate with minimal Δ is selected as the final cone for that event.

**Gamma imaging status:**  

The Compton (gamma) path is **fully wired and validated** against the legacy NOVO code on available PHITS-derived model datasets. Remaining work mainly concerns ROOT adapter integration and additional automated tests.

### 16.7. Unified Pipeline and CLI

- [x] Make `pipelines.core.run_pipeline` the central pipeline function.
- [x] Provide a Typer-based CLI entry (`python -m ngimager.pipelines.core`) that respects `run.fast`, `run.list`, `run.neutrons`, and `run.gammas` via flags (`--fast`, `--list`, `--neutrons/--no-neutrons`, `--gammas/--no-gammas`) and adds overrides for `[io].input_path` and `[io].output_path` (`--input-path`, `--output-path`).
- [ ] Add a CLI flag to override `[pipeline].until` (e.g. `--until=cones`) once restart-from-HDF5 is implemented.
- [ ] Implement full `pipeline.until` gating and support resuming from ngimager HDF5 files (`input_format = "hdf5_ngimager"`).
- [ ] Add a console script entry point (e.g. `ngimager`) in `pyproject.toml` as a thin wrapper around the existing Typer CLI, so users can run `ngimager run config.toml` from any directory without invoking `python -m ...` manually.

### 16.8. HDF5 and Visualization

- [x] **Back-tracing**: events, cones, and list-mode pixels are linked via `/lm/event_survival`, `/cones/event_index`, and `/lm/cone_pixel_indices`.
- [x] **Species-separated images**: `/images/summed/n` and `/images/summed/g` are written when the corresponding species are enabled; `/images/summed/all` is written when both are non-empty.
- [x] **Counters in file**: the full counters dict is stored under `/meta/counters/*`.
- [ ] **Restart from cones/image**: add support for `input_format = "hdf5_ngimager"` as an imaging-only restart mode.
- [ ] **CLI wrappers / nicer entry points**: expose a user-friendly `ngimager` console script (e.g. `ngimager run config.toml`).
- [ ] Wrap PNG export in a clean CLI function that calls `vis.hdf.save_summed_png` driven by `[vis]`, and optionally writes PNGs for all available summed images (n, g, all).
- [ ] Support imaging-only reruns from existing cones (ngimager HDF5 input) to generate list-mode per-cone images from previously non-list-mode outputs.
- [ ] For the ROOT adapter, implement propagation of metadata, including run number, into the HDF5 output (`/lm/event_meta_run_id`).

### 16.9. Documentation and Example Configs

- [ ] Add one or more **fully commented** example `.toml` files in `examples/configs/` that demonstrate all major features (PHITS, ROOT, ToF-based experiments).
- [ ] Provide a worked example for:
    - `examples/imaging_datasets/PHITS_simple_ng_source/usrdef.out`
- [ ] Keep this document synchronized with the code and use it as the basis of a “Developer Tour” page in `docs/dev/architecture.md` (or similar).

---

This document is intended to be the **ground truth design** for `ngimager`. As code evolves, it should either be updated to reflect the new reality or drive refactors back into alignment, keeping the system clean, maintainable, and physically faithful to the NOVO imaging goals.
```
