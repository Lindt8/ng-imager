# ng-imager Architecture & Pipeline Overview  
*(Design document and development guideline)*

> This document describes how `ngimager` **should** be structured and behave, not necessarily how the current codebase looks today. It is the reference for refactors and future development, and it is meant to supersede any ad-hoc design that emerged during early implementation experiments.

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
4. **Construct canonical Hits** and apply **universal hit-level filters**.
5. **Discard raw events** that no longer have enough hits to be kinematically reconstructable (e.g., fewer than 2 valid neutron hits or 3 valid gamma hits).
6. **Shape hits into imaging-viable Events** (neutron 2-hit, gamma 3-hit, etc.).
7. **Apply event-level filters**.
8. **Apply energy strategy & priors**, and enumerate **candidate cones** for each event (multiple cones per event are allowed).
9. **Apply cone-level filtering and selection**, picking at most one cone per event to be imaged (or zero if none are viable).
10. **Image cones** (SBP initially, other methods later).
11. **Write results to HDF5**, optionally at multiple stages.

Depending on `run.use_neutrons` and `run.use_gammas`, only the chosen particle types are shaped into events, propagated into cones, and imaged; the other type is ignored at all stages.


Conceptual pseudocode:

```python
def run_pipeline(cfg: Config, input_path: Path) -> Path:
    adapter = make_adapter(cfg.io, cfg.detector, cfg.materials)
    raw_events = adapter.iter_raw_events(input_path)

    # Stage 1: Raw events → Hits (with hit-level filtering)
    hits_by_raw_event = []
    for raw_event in raw_events:
        hits = [
            dict_hits_to_Hit(raw_hit, cfg)
            for raw_hit in raw_event
        ]
        hits = apply_hit_filters(hits, cfg.filters.hits, counters)
        if is_reconstructable(hits, cfg.filters):
            hits_by_raw_event.append(hits)
        else:
            counters["raw_events_rejected_unreconstructable"] += 1

    write_hits_stage(hits_by_raw_event, cfg, counters)

    if cfg.run.stop_stage == "hits":
        finalize_stats_and_metadata(cfg, counters)
        return cfg.io.output_path

    # Stage 2: Hits → Shaped events → Typed events
    shaped_events = shape_events_for_cones(hits_by_raw_event, cfg.filters, counters)
    typed_events = shaped_to_typed_events(shaped_events, cfg.filters, counters)
    filtered_events = list(filter_events(typed_events, cfg.filters, counters))

    write_events_stage(filtered_events, cfg, counters)

    if cfg.run.stop_stage == "events":
        finalize_stats_and_metadata(cfg, counters)
        return cfg.io.output_path

    # Stage 3: Events → Candidate cones → Selected cones
    energy_strategy = make_energy_strategy(cfg.energy, cfg.materials)
    prior = make_prior(cfg.prior)

    annotated_events = [
        energy_strategy.annotate_event(e, counters)
        for e in filtered_events
    ]

    candidate_cones = enumerate_candidate_cones(annotated_events, prior, counters)
    selected_cones = select_cones(candidate_cones, prior, cfg.filters.cones, counters)

    write_cones_stage(selected_cones, cfg, counters)

    if cfg.run.stop_stage == "cones":
        finalize_stats_and_metadata(cfg, counters)
        return cfg.io.output_path

    # Stage 4: Cones → Images
    plane = make_plane(cfg.plane)
    images = image_cones(selected_cones, plane, cfg, counters)

    write_images_stage(images, cfg, counters)

    finalize_stats_and_metadata(cfg, counters)
    return cfg.io.output_path
```

**Key points:**

- Adapters emit **raw events** that already group coincident hits (the acquisition system’s notion of “event”), but thresholds in acquisition are lower than post-processing thresholds.
- Universal hit-level cuts are applied early. Raw events that no longer contain enough candidate hits to ever form a reconstructable cone are discarded before shaping.
- Shaping and typing are separate from the adapter, so physics and filtering logic is shared across PHITS, ROOT, and future formats.
- Cone-building is a two-stage concept:
  - Enumerate all plausible cones per event (e.g., neutron proton vs carbon, gamma permutations).
  - Use priors + cone-level filters to select at most one final cone per event.
- Each stage maintains **counters** for accepted/rejected objects, and HDF5 output can reflect only the surviving objects after the full chain, while still allowing early-exit runs.

---

## 3. Configuration: The `.toml` File

The `.toml` config is the single source of truth for all non-data settings. It should be:

- **Explicit**: parameters are clearly described and named.
- **Non-redundant**: no value should have to be repeated in multiple places.
- **Overridable by CLI** for operational convenience (e.g. `--fast`, `--list`).

In actual example configs under `examples/configs/`, we expect **liberal use of comments** documenting what each section and field is for.

### 3.1. Proposed Top-Level Sections

These map naturally to `ngimager.config.schemas`.

#### `[run]`

General pipeline behavior, particle-type toggles, and diagnostics:

- `fast = false`  
  - Use more aggressive thresholds and limits for speed. Modifies the default behavior; does not replace it.
- `list = false`  
  - Enable list-mode image outputs (per-cone footprints). Also a modifier of the default behavior.
- `use_neutrons = true`  
  - If `false`, neutron hits/events/cones/images are ignored and not produced. Allows gamma-only imaging.
- `use_gammas = true`  
  - If `false`, gamma hits/events/cones/images are ignored and not produced. Allows neutron-only imaging.
- `stop_stage = "images"`  
  - One of `"hits" | "events" | "cones" | "images"`. Controls how far the pipeline runs.
- `max_events = 0`  
  - 0 means no limit; otherwise, stop after this many (typed) events (after particle-type toggles are applied).
- `max_cones = 0`  
  - 0 means no limit; otherwise, stop after this many **selected** cones (after particle-type toggles are applied).
- `diagnostic_level = 1`  
  - `0`: silent (no diagnostic prints except fatal errors)  
  - `1`: minimal, important pipeline messages  
  - `2`: verbose, detailed debugging info (these messages are indented with an extra tab for readability)

> **CLI overrides:**  
> - `--fast` and `--no-fast` override `run.fast`.  
> - `--list` and `--no-list` override `run.list`.  
> - `--stop-stage` can override `run.stop_stage`.  
> - Future CLI flags like `--neutrons-only` / `--gammas-only` may override `use_neutrons` / `use_gammas`.  
> The CLI always loads the config first, then applies overrides before calling `run_pipeline`.


#### `[io]`

Input/output format and paths:

- `input_path = "..."`  
- `input_format = "phits_usrdef" | "root_novo_ddaq" | "hdf5_ngimager"`  
- `output_path = "..."`  
- `hdf5_overwrite = true/false`

If `input_format = "hdf5_ngimager"`, the pipeline can **resume** from partially processed ngimager output.

The string `"root_novo_ddaq"` refers to the current NOVO acquisition system; future acquisition formats can be added as additional `input_format` values with new adapters.

#### `[detector]`

Detector geometry and mapping from IDs/regions to materials etc.:

```toml
layout = "mNOVO_vX"

[[detector.bars]]
det_id   = 200
material = "OGS"
position = [x_cm, y_cm, z_cm]
axis     = [ux, uy, uz]  # bar axis direction
# optional: length, width, height, region_id, etc.
```

All information about NOVO-specific numbering, region codes, and orientations lives here.

#### `[materials]`

Scintillator materials and associated LUTs:

```toml
[[materials.scintillators]]
name    = "OGS"
lut_npz = "path/to/OGS_proton_E_vs_L.npz"

[[materials.scintillators]]
name    = "M600"
lut_npz = "path/to/M600_proton_E_vs_L.npz"
```

Optional: additional physics descriptors (density, Z, etc.) if needed later.

#### `[energy]`

Energy strategies for **neutron events**:

```toml
strategy = "light_lut"  # or "tof" or "fixed_incident"

[energy.light_lut]
default_material = "OGS"   # used when material-specific LUT selection is ambiguous

[energy.tof]
start_detector  = "tagger"
flight_path_cm  = 100.0

[energy.fixed_incident]
En_MeV = 14.1
```

- `light_lut` uses calibrated E(L) LUTs per material (the modern default).  
- `tof` uses time-of-flight for incident neutron energy (legacy compatible).  
- `fixed_incident` uses a constant incident energy (e.g. DT source).  

These strategies are currently designed for **neutron kinematics**; gamma energy handling is much less critical for the present imaging method and may be added later if needed.

#### `[plane]`

Imaging plane specification:

```toml
origin   = [x0, y0, z0]
normal   = [nx, ny, nz]
u_axis   = [ux, uy, uz]  # optional if deducible
v_axis   = [vx, vy, vz]  # optional
u_extent = [umin, umax]
v_extent = [vmin, vmax]
nu       = 256
nv       = 256
```

This maps directly to `geometry.Plane`.

#### `[filters]`

Thresholds and cuts:

```toml
[filters.hits]
min_light_n = 0.5  # MeVee or equivalent
min_light_g = 0.1

[filters.events]
min_dt_ns_n = 0.0
max_dt_ns_n = 50.0
# gamma-specific cuts, etc.

[filters.cones]
min_opening_angle_deg = 5.0
max_opening_angle_deg = 60.0
# other cone-level quality metrics
```

Hit, event, and cone cuts are all driven from here; there should be **no hard-coded cuts** scattered across modules.

#### `[prior]`

Source priors:

```toml
type = "point"  # or "line"

[prior.point]
r0 = [x_cm, y_cm, z_cm]

[prior.line]
r0        = [x0_cm, y0_cm, z0_cm]
direction = [dx, dy, dz]
```

Extensible to more complex priors later (e.g. volumetric distributions, tabulated distributions).

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

Initially unused or used only for diagnostics; later can feed into uncertainty-aware imaging.

#### `[vis]`

Visualization / PNG export:

```toml
export_png_on_write = true
png_dir             = "images/"
colormap            = "viridis"   # used by visualization, not core SBP
```

---

## 4. Input Adapters: Raw Events and Hits

Adapters translate source-specific raw data into a **canonical representation**.

### 4.1. Raw Events

Adapters emit **raw events**:

- Each raw event corresponds to:
  - A PHITS history line (`usrdef.out`), or
  - An experimental trigger/coincidence window in ROOT, etc.
- Each raw event contains **all hits in the coincidence window**:
  - Multiple detectors, possibly multiple particles (n + γ), and noise.

Interface sketch:

```python
class BaseAdapter(ABC):
    def iter_raw_events(self, path: Path) -> Iterable[Iterable[dict]]:
        """
        Yield raw events, each as an iterable of raw-hit dicts.
        """
        ...
```

`PHITSAdapter` and `ROOTAdapter` implement this, hiding format quirks.

### 4.2. Hit Construction and Hit-Level Filters

From each raw event:

1. Convert raw hits to canonical `Hit` objects via a single function:

   ```python
   def dict_hits_to_Hit(raw: dict, cfg: Config) -> Hit:
       """
       Map raw fields (det_id, region, energy deposit, time, position, etc.)
       to a canonical Hit, using cfg.detector and cfg.materials.
       """
       ...
   ```

2. Apply **universal hit-level cuts** (e.g. min light/energy):

   ```python
   hits = [dict_hits_to_Hit(r, cfg) for r in raw_event]
   hits = apply_hit_filters(hits, cfg.filters.hits, counters)
   ```

   Here `counters` includes things like:

   - `hits_total`
   - `hits_rejected_threshold`
   - etc.

3. Determine whether the event is still *potentially reconstructable*:

   - Count surviving neutron hits, gamma hits, etc.
   - If there are fewer than 2 candidate neutron hits and fewer than 3 candidate gamma hits, the raw event can be discarded early.

```python
if not is_reconstructable(hits, cfg.filters):
    counters["raw_events_rejected_unreconstructable"] += 1
    continue  # drop this raw event
```

This approach obeys the real acquisition model:

- Raw data is already in “coincidence windows” (crude events).
- Acquisition thresholds are looser; post-processing thresholds refine which hits (and events) we keep.

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
- Potentially handle multi-particle mixed windows, splitting them into separate neutron and gamma events.

Interface:

```python
def shape_events_for_cones(
    hits_by_raw_event: Iterable[Iterable[Hit]],
    cfg: FiltersCfg,
    counters: dict,
) -> Iterable[ShapedEvent]:
    """
    For each raw event's survived hits, yield zero or more ShapedEvent
    objects describing candidate neutron/gamma events.
    """
```

`ShapedEvent` is a lightweight structure holding:

- A list of `Hit`s.
- A particle type (e.g. `"n"` or `"g"`).
- Basic metadata (original raw event index, multiplicity, etc.).

Counters at this stage might include:

- `shaped_events_total`
- `shaped_events_n`
- `shaped_events_g`
- `raw_events_rejected_shaping` (if nothing usable can be formed)

### 5.2. Typed Events

Next, we convert `ShapedEvent` objects into physics-aware event classes:

- `NeutronEvent`: always carries its constituent `Hit`s.
- `GammaEvent`: always carries its constituent `Hit`s.

Interface:

```python
from ngimager.physics.events import NeutronEvent, GammaEvent

def shaped_to_typed_events(
    shaped: Iterable[ShapedEvent],
    cfg: FiltersCfg,
    counters: dict,
) -> Iterable[Union[NeutronEvent, GammaEvent]]:
    ...
```

**Invariant:** `NeutronEvent` and `GammaEvent` *always* hold references to their constituent `Hit` objects. This guarantee is reflected both in memory and in the HDF5 representation, where cones can be traced back to events and events to hits via indices.

Counters incremented here might track:

- `typed_events_n`
- `typed_events_g`
- `typed_events_rejected` (e.g., invalid combinations)




---

## 6. Event-Level Filters

Once we have typed events, we apply **event-level cuts** based on timing, geometry, and other derived quantities.

Examples:

- Neutron events:
  - Time between scatters within [min_dt_ns_n, max_dt_ns_n].
  - Minimum bar separation and geometry cuts.
- Gamma events:
  - Timing consistency within the timing resolution.
  - Basic geometric sanity checks (e.g., bar separation, positions).

Centralized interface:

```python
def filter_events(
    events: Iterable[Union[NeutronEvent, GammaEvent]],
    cfg: FiltersCfg,
    counters: dict,
) -> Iterable[Union[NeutronEvent, GammaEvent]]:
    ...
```

Counters:

- `events_total_before_filters`
- `events_rejected_time_window`
- `events_rejected_geometry`
- `events_passed`

This keeps the event selection logic **configurable and testable** and avoids hidden “magic cuts” in random places.

---

## 7. Energy Strategies for Neutrons and E(L) LUT Integration

The **energy strategy** determines how neutron energy information is obtained for kinematic reconstruction.

At present, energy strategies are defined for **neutrons**:

- For neutron cones, the key quantity is the **recoil energy** (proton or carbon).
- There are multiple ways to obtain the “before” and “after” neutron energies.

Supported neutron strategies:

- `EnergyFromLightLUT` (modern default).
- `EnergyFromToF` (legacy timing-based with start detector).
- `EnergyFromFixedIncident` (fixed initial energy, e.g. DT).

Factory:

```python
strategy = make_energy_strategy(cfg.energy, cfg.materials)
```

Usage:

```python
annotated_events = [
    strategy.annotate_event(e, counters)
    for e in events
]
```

`annotate_event` attaches to `NeutronEvent`:

- Estimates of incident neutron energy before the first scatter.
- Estimates of neutron energy after the first scatter (where needed).
- Recoil energy deposits per scatter, for both proton and carbon interpretations if applicable.

### 7.1. Light-Based Strategy (E(L) LUT)

For **EnergyFromLightLUT**:

- We use calibrated `E(L)` look-up tables for each scintillator material (OGS, M600, etc.).
- The LUT returns deposited energy from measured light output for proton and/or carbon recoils.
- This directly yields the recoil energy required for cone kinematics.

Internally, this uses `io.lut.LUT` objects, which themselves may be produced using scripts that utilize `NOVO_light_response_functions`.

### 7.2. ToF-Based Strategy

For **EnergyFromToF**, neutron energies are derived from timing:

- “Before” and “after” energies are constructed from:
  - Timing between a “start” detector and the first neutron hit (for initial energy).
  - Timing between the first and second neutron hits (for post-scatter energy).
- For example:
  1. A start detector sees a prompt gamma associated with a neutron’s production (e.g., Cf-252 spontaneous fission or a beam pulse on a photoneutron source).
  2. The known flight path between the source and start detector, combined with the start detector time, determines the neutron production time.
  3. The known distance between the production point and the first interaction location, and the time between production and first interaction, give the **before-scatter neutron energy**.
  4. The time between the first and second detector hits, along with the known geometry between those hits, gives the **after-scatter neutron energy**.
  5. The **recoil energy** used in cone kinematics is then:
     ```text
     E_recoil = E_before - E_after
     ```
- This supports legacy experiments where timing was the primary means of determining neutron energies.

### 7.3. Fixed-Incident Strategy

For **EnergyFromFixedIncident**:

- A fixed initial incident neutron energy is assumed (e.g., DT reactions).
- The **after-scatter energy** is still obtained via ToF between the first and second neutron hits.
- Recoil energy is again:
  ```text
  E_recoil = E_before_fixed - E_after
  ```
  where `E_before_fixed` is a constant specified in the config.

### 7.4. Current Scope and Gamma Events

Currently, energy strategies are primarily designed for **neutron events**; gamma events for the SBP-style imaging do not require as detailed energy treatment for the existing reconstruction logic. If future gamma imaging methods require more detailed gamma energy estimation, extensions can be added to the energy strategy system.

All these energy strategies are encapsulated in `energy_strategies` and configured via `[energy]`, not scattered across the pipeline.

---

## 8. Priors and Sequencing Logic

### 8.1. Source Priors

Priors reflect what we know about where radiation is coming from:

- `PointPrior`: for point sources (e.g., Cf-252, DT tubes).
- `LinePrior`: for extended line-like sources (e.g., proton beams in a phantom).

Created via:

```python
prior = make_prior(cfg.prior)
```

They are used both in:

- **Event interpretation** (e.g., deciding which neutron recoil interpretation or which gamma permutation is more plausible).
- **Cone-level selection** (e.g., choosing which cone from several candidates best matches the prior).
- Potentially **imaging** (e.g., weighting cones based on intersection with the prior).

### 8.2. Sequencing and Event Interpretation

For gamma events:

- A 3-hit gamma event has 6 possible permutations (orderings).
- Many of these may be physically implausible.
- Sequencing logic:
  - For each permutation, construct a candidate cone geometry.
  - Evaluate its kinematic plausibility and consistency with the source prior.
  - Assign a quality metric (e.g., likelihood score, χ²-like measure).
  - These candidate cones are passed on to the cone-selection stage.

For neutron events:

- There is ambiguity between proton vs carbon recoil interpretations.
- For each neutron event, we can construct:
  - A proton-based candidate cone.
  - A carbon-based candidate cone.
- Each candidate has different opening angles and orientations given the energy and kinematics.
- Again, priors and kinematic consistency can be used to score these candidates.

**Design approach:**

- `enumerate_candidate_cones`:
  - For each neutron event:
    - Build one or more candidate cones (e.g., proton-based and carbon-based).
  - For each gamma event:
    - Build candidate cones corresponding to each viable permutation.
  - Each candidate cone carries metadata describing:
    - Its originating event.
    - Proton vs carbon assumption (for neutrons).
    - Hit ordering/permutation (for gammas).
    - Diagnostics and scores.

- `select_cones`:
  - Takes all candidate cones for a given event and:
    - Applies cone-level filters.
    - Uses priors and quality metrics to select at most one cone per event.
    - Can choose “no cone” if none are acceptable.
  - Records which candidate was selected and why (via metadata and counters).

**Event objects must store:**

- For `GammaEvent`:
  - The chosen ordering (e.g. a permutation of indices `[0,1,2] → [2,0,1]`).
- For `NeutronEvent`:
  - The chosen recoil interpretation (e.g. `"proton"` or `"carbon"`).

This information is persisted to HDF5 so downstream analyses can see what was inferred independently of raw timestamps.

---

## 9. Cone Representation

Cones are constructed from energy-annotated, sequenced events and priors.

`enumerate_candidate_cones` and `select_cones` work with a `Cone` dataclass:

```python
@dataclass
class Cone:
    r0: np.ndarray      # apex position (3,)
    k_hat: np.ndarray   # unit axis direction (3,)
    theta: float        # opening angle [rad]
    event_index: int    # index into events dataset
    particle_type: Literal["n", "g"]
    candidate_type: str # e.g. 'proton', 'carbon', 'gamma_perm_012', etc.
    score: float        # quality metric used during selection
    meta: dict          # extra details: energies, ordering, etc.
```

These parameters **fully define the analytic cone**:

- Points `x` on the cone satisfy:
  ```text
  (x - r0) · k_hat = |x - r0| cos(theta)
  ```

**Design choice on analytic cone equations:**

- The `Cone` class itself stores the **canonical geometric parameters** (`r0`, `k_hat`, `theta`), which are sufficient to recover the analytic equation.
- Imaging backends (e.g. SBP, MLEM) may construct alternative internal representations (e.g. matrices for quadric surfaces), but these are **derived** from the `Cone` parameters and not stored as part of the core `Cone` model.
- This keeps `Cone` simple and imaging-method agnostic, while still embedding all necessary analytic information.

Counters here include:

- `candidate_cones_total`
- `candidate_cones_rejected_filters`
- `selected_cones_total` (with separate counts for neutrons/gammas and for proton/carbon selections).


---

## 10. Imaging Back-End (SBP and Future Methods)

The first imaging back-end is **simple back projection (SBP)**, but the architecture is intended to support others later.

### 10.1. Imaging Plane

`ngimager.geometry.plane.Plane` encodes:

- Origin `P0`
- Normal vector `n̂`
- Basis vectors `êu`, `êv`
- Mappings:
  - From 3D coordinates to plane `(u, v)`.
  - From `(u, v)` back to 3D.

This object is constructed from `[plane]` config entries.

### 10.2. SBP Rasterization

`ngimager.imaging.sbp.reconstruct_sbp` roughly:

- Takes a collection of **selected** `Cone` objects and a `Plane`.
- For each cone:
  - Computes cone–plane intersection analytically.
  - Rasterizes the intersection into the plane’s pixel grid.
- Produces:
  - A **summed image** (2D array).
  - Optionally, per-cone sparse footprints (for list-mode).

SBP is imaging-method-agnostic in the sense that it only depends on the generic `Cone` representation and `Plane`.

### 10.3. Pluggable Imaging Methods

The imaging step is a dispatcher:

```python
def image_cones(cones, plane, cfg: Config, counters: dict):
    method = cfg.imaging.method  # e.g. "sbp" now, "mlem" later
    if method == "sbp":
        return reconstruct_sbp(cones, plane, cfg.imaging, counters)
    elif method == "mlem":
        ...
```

Future methods (MLEM, SOE, etc.) will reuse the same `Cone` and `Plane` abstractions.

Counters for imaging might include:

- `cones_imaged_total`
- `sbp_pixels_touched_total`
- `sbp_time_seconds`

---

## 11. Fast vs List Mode: Modifiers of the Default Pipeline

There is a **single unified pipeline**. “Fast” and “list” are orthogonal **modifiers** of its behavior, not separate pipelines or modes that change control flow.

### 11.1. Default Behavior

Without any modifiers:

- Reasonable thresholds and limits.
- Full SBP summed images.
- Hits, events, and selected cones stored in HDF5.
- No per-cone images.

### 11.2. Fast Mode (`run.fast`)

- Uses alternative or stricter settings:
  - Higher hit-level thresholds.
  - Stronger event and cone cuts.
  - Possibly lower `max_events` / `max_cones`.
- Aimed at **quick feedback** (e.g., during experiments).
- Still writes:
  - Hits, events, cones, and summed images.
  - It may just be a subset of what default settings would produce.

Fast mode is configured via `[run] fast = true` but can also be enabled via CLI `--fast` (CLI overrides the config). Minimal vs verbose diagnostics still apply.

### 11.3. List Mode (`run.list`)

- When `list = true`, in addition to the default outputs, the pipeline:
  - Computes **per-cone sparse images**:
    - For each cone, store the pixel indices and weights where its SBP footprint deposited counts.
  - Writes these list-mode images to `/images/listmode/*` in HDF5.

List mode is configured via `[run] list = true` or CLI `--list`.

Fast and list can be combined (e.g., fast thresholds but list-mode storage). The combination might be less common operationally but is conceptually well-defined.

---

## 12. HDF5 Data Model and Partial Pipelines

The HDF5 format is the primary output, and it should support:

- **Full pipeline outputs** (hits, events, cones, images).
- **Partial pipeline outputs** (e.g., hits + events only).
- **Resuming** from partially processed data.
- **Consistent views** of what survived all active filters at each stage.

### 12.1. HDF5 Layout (Suggested)

- `/meta`
  - `config_toml`
  - `git_commit`
  - `ngimager_version`
  - `run_timestamp`
  - `run_fast` (bool), `run_list` (bool)
  - `run_stop_stage`
  - `counters` (group or JSON blob with pipeline counts)
- `/hits/n`, `/hits/g`
  - Ragged or structured datasets:
    - `det_id`, `t_ns`, `L`, `x_cm`, `y_cm`, `z_cm`, `material`, etc.
- `/events/n`, `/events/g`
  - Event datasets:
    - References (indices) into `/hits/*`.
    - Per-event energies, timing deltas, sequencing choices, etc.
- `/cones/n`, `/cones/g`
  - Cone datasets:
    - `r0` (3 floats), `k_hat` (3 floats), `theta` (1 float).
    - Index into events.
    - Particle type and candidate type.
- `/images/summed/n`, `/images/summed/g`
  - Main SBP images (2D arrays).
- `/images/listmode/n`, `/images/listmode/g` (optional)
  - Per-cone sparse footprints:
    - For each cone index K:
      - `pixel_indices` (1D int array)
      - `weights` (1D float array)

**Guarantees:**

- It must always be possible to trace:

  ```text
  image pixel   → cone(s)   → event   → hits
  ```

  via indices and datasets, in all modes (default, fast, list).

- When the pipeline runs all the way to cones or images, the HDF5 file should **not** contain earlier-stage objects that correspond to cones or events that were ultimately rejected. Practical options include:
  - Writing to temporary datasets and then compacting to “final” datasets.
  - Writing all candidates but then rewriting/compacting the corresponding HDF5 groups after selection.

Partial runs (e.g., `stop_stage = "events"`) naturally represent the state **before** later filters and selection have been applied.

### 12.2. Writing at Multiple Stages

The pipeline can write intermediate results **incrementally**, but the final HDF5 layout should be consistent with the selected objects after all enabled filters.

Recommended behavior:

- For `stop_stage` less than `"images"`:
  - Write out the current stage’s results (e.g., hits, events) as they stand at that stage, without later pruning.
- For full pipeline runs (`stop_stage = "images"`):
  - Maintain internal buffers during processing.
  - When the pipeline has finished selection:
    - Write **only** the surviving hits/events/cones to the final HDF5 groups.
    - Store all counters describing how many objects were rejected at each stage.

### 12.3. Resuming from HDF5

When `input_format = "hdf5_ngimager"`:

- If `/cones/*` exists but `/images/*` does not:
  - Start at cones → images.
- If `/events/*` exists but `/cones/*` does not:
  - Start at events → cones → images.
- If only `/hits/*` exist:
  - Start at events → cones → images.

This enables:

- Re-running imaging with different plane/imaging settings without recomputing events/cones.
- Sharing hits+events+cones with collaborators who may implement their own imaging.


### 12.4. Converting Between List-Mode and Non-List-Mode Outputs

For a completed ngimager HDF5 file, it should be straightforward to move between:

- A **non-list-mode** representation (summed images only), and
- A **list-mode** representation (summed images + per-cone sparse footprints).

Two common workflows:

1. **List-mode → non-list-mode**

   - This is trivial: delete the `/images/listmode/*` groups from the HDF5 file.
   - Hits, events, cones, and summed images remain intact.

2. **Non-list-mode → list-mode**

   - Start from an HDF5 file that already contains selected cones and summed images, but no list-mode images.
   - Re-run the pipeline with:
     - `input_format = "hdf5_ngimager"`
     - `run.stop_stage = "images"`
     - `run.list = true`
   - The pipeline detects existing hits/events/cones, skips rebuilding them, and re-runs only the imaging stage, this time computing and writing per-cone sparse footprints into `/images/listmode/*`.

This makes it cheap to “upgrade” a previously run dataset from non-list-mode to list-mode without redoing the entire event and cone construction chain.


---

## 13. Diagnostics, Logging, and Counters

Diagnostics are gated by `run.diagnostic_level`:

- `0`: no diagnostic messages (except fatal errors).
- `1`: minimal messages indicating:
  - Stage entry/exit (hits/events/cones/images).
  - Counts (e.g., number of events, cones).
  - **Per-stage runtimes** (e.g., “hits stage took 0.42 s”, “imaging stage took 3.1 s”).
  - A final counter summary at the end of the run.
- `2`: verbose messages, including:
  - Detailed adapter parsing notes.
  - Filter statistics per stage.
  - Fine-grained timing information useful for profiling (sub-stage timers).
  - These verbose lines should be indented with a leading tab (`\t`) to visually distinguish them from level-1 outputs.

Example usage:

```python
t0 = time.perf_counter()
# ... run hits stage ...
t1 = time.perf_counter()

if cfg.run.diagnostic_level >= 1:
    print(f"[pipeline] hits stage completed in {t1 - t0:.3f} s")

if cfg.run.diagnostic_level >= 2:
    print(f"\t[pipeline] hits stage parsed {counters['hits_total']} hits")
```

This keeps level 1 useful for humans (you always see stage runtimes and the counter summary) and level 2 for more granular profiling noise.

### 13.1. Counters

A shared `counters` dict is passed through the pipeline and used to record:

- Raw events (type-agnostic):
  - `raw_events_total`
  - `raw_events_rejected_unreconstructable`
  - `raw_events_rejected_shaping`
- Hits (per-particle where meaningful):
  - `hits_total`
  - `hits_total_n`, `hits_total_g`
  - `hits_rejected_threshold`
  - `hits_rejected_threshold_n`, `hits_rejected_threshold_g`
- Events:
  - `shaped_events_total`
  - `shaped_events_n`, `shaped_events_g`
  - `typed_events_total`
  - `typed_events_n`, `typed_events_g`
  - `events_rejected_time_window_total`
  - `events_rejected_time_window_n`, `events_rejected_time_window_g`
  - `events_rejected_geometry_total`
  - `events_rejected_geometry_n`, `events_rejected_geometry_g`
  - `events_passed_total`
  - `events_passed_n`, `events_passed_g`
- Cones:
  - `candidate_cones_total`
  - `candidate_cones_n`, `candidate_cones_g`
  - `candidate_cones_rejected_filters_total`
  - `candidate_cones_rejected_filters_n`, `candidate_cones_rejected_filters_g`
  - `selected_cones_total`
  - `selected_cones_n`, `selected_cones_g`
  - `selected_cones_proton`, `selected_cones_carbon`  # neutron-only refinements
- Imaging:
  - `cones_imaged_total`
  - `cones_imaged_n`, `cones_imaged_g`
  - `sbp_pixels_touched_total`
  - `sbp_pixels_touched_n`, `sbp_pixels_touched_g`
  - `sbp_time_seconds`

**Pattern:** wherever it is conceptually meaningful to separate by particle type, we maintain **three** counters:

- A `_total` counter (neutrons + gammas),
- A `_n` counter (neutrons only),
- A `_g` counter (gammas only).

A summary of these counters should be:

- Printed at the end of the run for `diagnostic_level >= 1` (minimal and verbose).
- Stored under `/meta/counters` in the HDF5 output (for **all** diagnostic levels).

This allows:

- Quick insight into where events are being rejected, separated by particle type.
- Traceability of “N final events imaged out of M raw events” with breakdown by stage and by neutron/gamma.




---

## 14. Legacy Timing and Future-Proofing

Legacy timing-based workflows are supported via the energy strategies (Section 7):

- **ToF-based paths**:
  - Maintained via `EnergyFromToF`, configured in `[energy.tof]`.
  - Useful for setups using a start detector and time-of-flight for incident energy.
- **Fixed incident energy**:
  - `EnergyFromFixedIncident` for known beam energies.
- **Light-based E(L) LUT**:
  - `EnergyFromLightLUT` using calibrated E(L) curves; this is the default for modern NOVO experiments with calibrated light response for OGS/M600.

These paths share a common conceptual picture:

- Neutron energies **before** and **after** the first scatter are estimated.
- Recoil energy is computed as `E_before - E_after` (except in LUT-based approaches where recoil energy is directly inferred from light).
- The recoil energy feeds directly into neutron cone kinematics.

All such behaviors are isolated to the energy strategy and config, not hard-coded if/else blocks scattered across the pipeline.

Uncertainty models (energy, DOI, timing) are kept modular to be turned on when supporting data (e.g. measured resolution curves) is integrated.

---

## 15. Coding Style and Conventions

Guidelines:

- Use **dataclasses** for structured data (Hit, NeutronEvent, GammaEvent, Cone).
- Use **type hints** extensively:
  - `Hit`, `NeutronEvent`, `GammaEvent`, `Cone`, `Plane`, `np.ndarray`, etc.
- Keep modules focused:
  - `physics.*`: physical models, kinematics, priors, event definitions.
  - `geometry.*`: planes and coordinate transforms.
  - `imaging.*`: turning cones into images.
  - `io.*`: adapters, HDF5 I/O, LUT loading.
  - `filters.*`: hit/event/cone shaping and selection.
  - `pipelines.*`: orchestration and CLIs.
  - `sim.*`: **deprecated for active pipelines**; if synthetic data is kept at all, it should live primarily in examples/tests rather than the core package.
- NOVO-specific quirks live in:
  - `io.adapters`, `tools/*`, and configuration.
  - Not in the physics/imaging core.

The aim is for someone familiar with the legacy script to be able to read this code and see where each piece migrated, and for new contributors to navigate the code via module names + this document.

---

## 16. Roadmap (Refactor and Implementation Checklist)

This checklist tracks migration from the current state to the architecture described here.

### 16.1. Cleanup and Deletion of Redundant Paths

- [ ] Remove any parallel PHITS→Hit/Event paths; ensure `PHITSAdapter` + `dict_hits_to_Hit` is the **only** canonical route for PHITS.
- [ ] Deprecate and remove `sim.*` from the active pipeline; if synthetic capabilities are retained, move them to examples/tests.
- [ ] Ensure only one `Hit` class exists and is used everywhere.

### 16.2. Adapters and Raw Events

- [ ] Ensure `PHITSAdapter.iter_raw_events` returns raw events as collections of hit dicts.
- [ ] Implement/clean up `ROOTAdapter.iter_raw_events` with `input_format = "root_novo_ddaq"` for the current acquisition system.
- [ ] Implement the early `is_reconstructable` logic after hit-level filtering to discard unviable raw events, with appropriate counters.

### 16.3. Shaper and Typed Events

- [ ] Make `shape_events_for_cones` the single entry point from raw-event hits to shaped events.
- [ ] Make `shaped_to_typed_events` the only path to `NeutronEvent`/`GammaEvent`.
- [ ] Guarantee that all event classes always carry their `Hit` lists and that indices in HDF5 allow round-tripping.
- [ ] Ensure counters at the hit and event levels follow the `_n` / `_g` / `_total` naming pattern where meaningful.


### 16.4. Filters, Priors, and Sequencing

- [ ] Centralize event and cone selection logic into `filters` modules, driven by `[filters]` config.
- [ ] Ensure priors are only defined in `physics.priors` and configured via `[prior]`.
- [ ] Implement `enumerate_candidate_cones` and `select_cones` so that:
  - Multiple candidate cones per event (proton vs carbon, gamma permutations) are supported.
  - At most one cone per event is ultimately selected.
- [ ] Implement storage of gamma sequencing choice and neutron recoil interpretation in the event objects and HDF5.

### 16.5. Energy Strategy Integration

- [ ] Wire `make_energy_strategy(cfg.energy, cfg.materials)` into the pipeline, with all neutron energy calculations happening via this interface.
- [ ] Integrate E(L) LUTs for OGS and M600 via `io.lut.LUT`.
- [ ] Validate `EnergyFromToF` and `EnergyFromFixedIncident` paths with simple tests.
- [ ] Document the conceptual picture in code comments, referencing this document.

### 16.6. Cone Construction and Imaging

- [ ] Ensure `physics.cones` provides the canonical functions for building candidate cones from events.
- [ ] Confirm `imaging.sbp.reconstruct_sbp` works directly from the `Cone` dataclass and `Plane`.
- [ ] Implement optional per-cone sparse footprints used only when `run.list` is true.
- [ ] Ensure cone and imaging counters follow the `_n` / `_g` / `_total` naming pattern where meaningful, and that per-stage runtimes are recorded and reported.


### 16.7. Unified Pipeline and CLI

- [ ] Make `pipelines.core.run_pipeline` the central pipeline function.
- [ ] Deprecate/remove `pipelines.fastmode` and `pipelines.listmode` in favor of a single CLI that respects `run.fast`, `run.list`, `run.use_neutrons`, and `run.use_gammas` from config and CLI flags.
- [ ] Implement CLI flags:
  - `--fast` / `--no-fast`
  - `--list` / `--no-list`
  - `--stop-stage`
  - optional convenience flags like `--neutrons-only` / `--gammas-only` mapped to `use_neutrons` / `use_gammas`
- [ ] Implement `run.stop_stage` gating at the main stages and support resuming from ngimager HDF5 files (`input_format = "hdf5_ngimager"`).


### 16.8. HDF5 and Visualization

- [ ] Finalize and document the HDF5 layout described above.
- [ ] Ensure hits/events/cones/images store the necessary indices for full back-tracing.
- [ ] Ensure final HDF5 outputs (for full runs) contain only objects that survive all active filters, with counters and metadata describing what was rejected at each stage.
- [ ] Wrap PNG export in a clean CLI function that calls `vis.hdf.save_summed_png` driven by `[vis]`.
- [ ] Support imaging-only reruns from existing cones (ngimager HDF5 input) to generate list-mode per-cone images from previously non-list-mode outputs.


### 16.9. Documentation and Example Configs

- [ ] Add an example `.toml` in `examples/configs/` with extensive comments explaining each section and option.
- [ ] Provide a worked example for:
  - `examples/imaging_datasets/PHITS_simple_ng_source/usrdef.out`
- [ ] Use this document as the basis of a “Developer Tour” page in `docs/dev/architecture.md` (or similar) to keep the code and architecture aligned.

---

This document is intended to be the **ground truth design** for `ngimager`. As code evolves, it should either be updated to reflect the new reality or drive refactors back into alignment, keeping the system clean, maintainable, and physically faithful to the NOVO imaging goals.
