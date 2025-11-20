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
4. **Apply universal hit-level filters**, using the adapter-emitted Hit objects.
5. **Discard raw events** that no longer have enough hits to be kinematically reconstructable (e.g., fewer than 2 valid neutron hits or 3 valid gamma hits).
6. **Shape hits into imaging-viable Events** (neutron 2-hit, gamma 3-hit, etc.).
7. **Apply event-level filters**.
8. **Apply energy strategy & priors**, and enumerate **candidate cones** for each event (multiple cones per event are allowed).
9. **Apply cone-level filtering and selection**, picking at most one cone per event to be imaged (or zero if none are viable).
10. **Image cones** (SBP initially, other methods later).
11. **Write results to HDF5**, optionally at multiple stages.

Depending on `run.use_neutrons` and `run.use_gammas`, only the chosen particle types are shaped into events, propagated into cones, and imaged; the other type is ignored at all stages.


### Conceptual pseudocode (current implementation)

This reflects the four-stage design and the current `ngimager.pipelines.core.run_pipeline`:

```python
def run_pipeline(cfg_path: str) -> Path:
    # Load TOML → Config (pydantic)
    cfg = load_config(cfg_path)

    # CLI flags override [run] fields (fast, list, neutrons, gammas, ...)
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

            # Normalize event_type = 'n' / 'g' / None
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
- `neutrons = true`  
    - If `false`, neutron hits/events/cones/images are ignored and not produced. Allows gamma-only imaging.
- `gammas = true`  
    - If `false`, gamma hits/events/cones/images are ignored and not produced. Allows neutron-only imaging.
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
> - `--stop-stage` can override `pipeline.until`.  
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

#### `[pipeline]`

This is where pipeline-related settings, such as a premature stopping point (e.g., stop after building cones but before imaging) is specified. 

- `until = "image"`  
    - One of `"hits" | "events" | "cones" | "image"`. Controls how far the pipeline runs.


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



```toml
# Neutron energy strategy:
# - "ELUT": use E(L) LUTs for the first scatter (per material)
# - "ToF":  use time-of-flight to estimate neutron energy change
# - "FixedEn": use a fixed incident energy (e.g. DT source)
# - "Edep": treat Hit.L as deposited energy directly (e.g. PHITS Edep_MeV)
strategy = "Edep"                 # ELUT | ToF | FixedEn | Edep

# For neutrons, assume recoils are always protons (true) 
#     or consider also carbon recoils (false)
force_proton_recoils = false      # true | false

[energy.light_lut]
default_material = "OGS"   # used when material-specific LUT selection is ambiguous

[energy.tof]
start_detector  = "tagger"
flight_path_cm  = 100.0

[energy.fixed_incident]
En_MeV = 14.1
```
Energy strategies for **neutron events**:

- `ELUT` uses calibrated E(L) LUTs per material (the modern default).  
- `ToF` uses time-of-flight for incident neutron energy (legacy compatible).  
- `FixedEn` uses a constant incident energy (e.g. DT source).  
- `Edep` assumes the provided `L` is actually energy deposition in MeV instead of light output in MeVee (as is the case for PHITS-scored energy deposion)

These strategies are currently designed for **neutron kinematics**; gamma energy handling is much less critical for the present imaging method and may be added later if needed.

For neutron events there is an ambiguity between proton and carbon recoils in organic scintillators. When using strategies that can support species-specific recoil estimates (e.g. ELUT, PHITS Edep), the pipeline can construct both proton and carbon neutron candidate cones and choose between them using a prior-based angle-difference metric:

    Δ = |θ_calc − θ_est|

This behavior is controlled by the configuration key `force_proton_recoils`:

- `force_proton_recoils = false` (default):
    - For species-aware strategies (ELUT, Edep), the pipeline builds both proton and carbon neutron cones when a prior is available and selects between them using Δ.
    - For legacy strategies (ToF, FixedEn), the deposited energy is effectively proton-like, but the H/C branch for scattering angle computation may still be evaluated if needed for comparison or reproduction of legacy behavior.

- `force_proton_recoils = true`:
    - Always assume proton recoils for neutrons.
    - Bypasses H/C hypothesis testing and prior-based discrimination.
    - Useful for strict legacy reproduction and debugging.

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

Filtering is organized around the three physical objects in the pipeline plus the final imaging step:

1. **Hits** — `[filters.hits]`
2. **Events** — `[filters.events]`
3. **Cones** — `[filters.cones]`
4. **Images** — (implicit; mainly controlled by SBP settings and `run.fast`)

Each of the three filter blocks supports:

- **universal** parameters (apply to both species), and
- **optional per-species overrides** under `.neutron` and `.gamma`.

For example:

```toml
[filters.hits]
min_light_MeVee = 50.0
max_light_MeVee = 1.0e6
bars_include = []
bars_exclude = []
materials_include = []
materials_exclude = []

[filters.hits.neutron]
# overrides or tightens universal values for neutron hits only

[filters.hits.gamma]
# overrides / tightens for gamma hits only

[filters.events]
tof_window_ns  = [0.0, 30.0]
min_L1_MeVee   = 0.0
min_L2_MeVee   = 0.0
min_L_any_MeVee = 0.0

[filters.events.neutron]
# e.g. more aggressive L1/L2 thresholds for neutron scatters

[filters.events.gamma]
# e.g. tighter ToF window for gamma triplets

[filters.cones]
max_delta_theta_deg     = 5.0
max_incident_energy_MeV = 20.0

[filters.cones.neutron]
max_delta_theta_deg     = 3.0
max_incident_energy_MeV = 20.0   # e.g. DT source

[filters.cones.gamma]
max_delta_theta_deg     = 8.0
max_incident_energy_MeV = 5.0
```

At runtime the code resolves an *effective* value for each quantity by:

1. checking the per-species override (if present),
2. otherwise falling back to the universal value, and
3. otherwise disabling that specific cut.

Counters are maintained for each stage and, when relevant, per species (e.g. `hits_rejected_threshold_n`, `events_rejected_tof_window`, `cones_rejected_delta_theta_g`). These show up both in the CLI diagnostics and as datasets under `/meta/counters` in the HDF5 file.

#### Fast mode

Fast mode is enabled via:

- `run.fast = true` in the TOML, or
- the `--fast` CLI flag (which overrides the TOML).

Fast mode does *not* change the physics; instead it switches to a set of more aggressive but reasonably safe defaults for:

- hit and event-level light thresholds (higher `min_L1/2/any_MeVee`),
- a cap on the total number of cones (`run.max_cones`), and
- a coarser imaging plane (larger `du`, `dv`, and/or a smaller FOV).

The exact heuristics live in `ngimager.config.schemas` and `ngimager.pipelines.core` and are intentionally minimal: the expectation is that most experiments will tune the base filters in the TOML and use `run.fast` as a “get an image quickly” toggle during setup.

#### `[prior]`

Source priors:

```toml
type = "point"  # or "line"

[prior.point]
r0 = [x_cm, y_cm, z_cm]

[prior.line]
r0        = [x0_cm, y0_cm, z0_cm]
direction = [dx, dy, dz]
# or
p0        = [x0_cm, y0_cm, z0_cm]
p1        = [x1_cm, y1_cm, z1_cm]
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

Material assignment is governed solely by `[detectors.material_map]` and `[detectors.default_material]`. The adapter layer must not accept a distinct default material from `[io.adapter]`; detector-level settings are authoritative.

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

*(Note: the current implementation uses `adapter.iter_events(...)` directly and returns typed `Event` objects. The `iter_raw_events` API above is the intended long-term interface and will replace direct event emission as the shaper and filters are completed.)*


`PHITSAdapter` and `ROOTAdapter` implement this, hiding format quirks.

### 4.2. Hit Construction and Hit-Level Filters

From each raw event:

1. Convert raw hits to canonical `Hit` objects through the adapter:

    - Raw hit dicts from the adapter are converted inside the adapter itself (e.g., `PHITSAdapter` or `ROOTAdapter`). Adapters must emit canonical `Hit` objects directly or provide an internal method ensuring all Hits follow the unified `physics.hits.Hit` structure.  
    - All adapters must set:
        - ```
          Hit.type      ∈ {"n","g","UNK"}    # particle species
          Hit.material  = scintillator name  # from [detectors.material_map]
          ```
    - The shaper, event typing, and cone-building stages rely exclusively on Hit.type. Legacy event-level “event_type” fields from raw adapters must not be used.

The shaper, event typing, and cone-building stages rely exclusively on Hit.type.
Legacy event-level “event_type” fields from raw adapters must not be used.

2. Apply **universal hit-level cuts** (e.g. min light/energy):

    ```python
    # adapter already returns canonical Hit objects
    hits = list(raw_event)
    # apply universal hit-level cuts
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

(Implementation note: Shaping and typed-event construction are implemented and used in the PHITS path. PHITSAdapter now emits canonical Hit objects, which pass through apply_hit_filters → is_reconstructable → shape_events_for_cones → shaped_to_typed_events to produce NeutronEvent and GammaEvent instances exactly as described here.)


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

### 7.4. Proton vs carbon recoil hypotheses

Elastic scattering of neutrons in organic scintillators can occur on hydrogen or carbon nuclei. To account for this:

1. The first-scatter deposited energy E_dep,1 is interpreted under both recoil hypotheses.
2. A scattering angle θ_calc is computed for each branch using CoM→lab mapping.
3. θ_calc is compared against the prior-derived θ_est, and the branch with the smallest Δ = |θ_calc − θ_est| is selected when H/C inference is enabled.

ELUT and PHITS Edep strategies provide species-resolved recoil estimates via per-species LUTs. ToF and FixedEn behave proton-like by legacy convention but may still evaluate both A=1 and A≈12 kinematic branches if required.

The configuration key:

    [energy]
    force_proton_recoils = true|false

controls whether proton-only reconstruction is enforced (legacy behavior) or whether H/C hypotheses are generated and later selected using priors and Δ.


### 7.5. Current Scope and Gamma Events

Currently, energy strategies are primarily designed for **neutron events**; gamma events for the SBP-style imaging do not require as detailed energy treatment for the existing reconstruction logic. If future gamma imaging methods require more detailed gamma energy estimation, extensions can be added to the energy strategy system.

Currently, PHITS-based workflows typically use the `Edep` strategy so that `Hit.L` can be treated as deposited energy in MeV (e.g., PHITS `Edep_MeV`) for both neutron and gamma events. This keeps the PHITS path simple while allowing LUT-based and timing-based strategies for experiments using calibrated light outputs.


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
- Each candidate cone is scored against the prior using the Δ = |θ_calc − θ_est| metric. The cone selection stage chooses at most one branch per event (or none if both are non-physical or filtered out).
- The selected recoil hypothesis is recorded in event- and cone-level metadata and ultimately written to HDF5 for downstream analysis.

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

## 11. Fast and List options: Modifiers of the Default Pipeline

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

### 12.1. HDF5 layout and traceability

The HDF5 format is designed so that you can trace everything:

> **hit → event → cone → pixels**

and back again.

The main groups are:

- `/meta` — geometry, run flags, config snapshot, counters, README
- `/images/summed` — species-separated and combined SBP images
- `/cones` — per-cone geometry + classification
- `/lm` — list-mode per-event/per-hit data and mappings

Key pieces:

- `/meta/config_toml` — full TOML config text used for this run.
- `/meta/readme` — a short README describing the layout and pointing to the online docs.
- `/meta/counters/*` — one small dataset per counter (e.g. `s1_hits_after_filters`),
  named with stage prefixes so HDFView sorts them roughly in pipeline order.

Cones:

- `/cones/cone_id`              : `[N_cones]` uint32
- `/cones/apex_xyz_cm`          : `[N_cones, 3]` float32
- `/cones/axis_xyz`             : `[N_cones, 3]` float32
- `/cones/theta_rad`            : `[N_cones]` float32
- `/cones/incident_energy_MeV`  : `[N_cones]` float32 (incident neutron or gamma energy
                                  used in the kinematic calculation)
- `/cones/species`              : `[N_cones]` uint8 (`0=neutron`, `1=gamma`)
- `/cones/recoil_code`          : `[N_cones]` uint8 (`0=unknown/NA`, `1=proton`, `2=carbon`)
- `/cones/event_index`          : `[N_cones]` uint32, row index into `/lm/...` arrays
- `/cones/species_labels`       : string array legend for `species`
- `/cones/recoil_code_labels`   : string array legend for `recoil_code`

List mode and events:

- `/lm/event_type`              : `[N_events]` uint8 (`0=neutron`, `1=gamma`)
- `/lm/event_type_labels`       : legend array for `event_type`
- `/lm/event_meta_run_id`       : `[N_events]` int32
- `/lm/event_meta_file_ix`      : `[N_events]` int32
- `/lm/hit_pos_cm`              : `[N_events, 3, 3]` float32
- `/lm/hit_t_ns`                : `[N_events, 3]` float32
- `/lm/hit_L_mevee`             : `[N_events, 3]` float32
- `/lm/hit_det_id`              : `[N_events, 3]` int32
- `/lm/hit_material_id`         : `[N_events, 3]` int16
- `/lm/material_id_labels`      : string array mapping material IDs back to names

In list-mode runs (`run.list = true`), SBP also produces per-cone pixel hits:

- `/lm/cone_pixel_indices`      : `[K, 2]` uint32, each row `(cone_id, flat_pixel_index)`
  mapping cone index → image pixel(s). The image is flattened with
  `flat = v * nu + u`.

To make debugging and post-processing easier, there is a small “survival table”:

- `/lm/event_survival`          : `[N_events, 3]` int32, columns:
  - `event_index`              — row in `/lm/...`
  - `first_cone_index`         — cone index built from this event or `-1`
  - `first_imaged_cone_index`  — cone index that actually intersected the plane or `-1`

Together, `/lm/event_survival`, `/cones/event_index`, and `/lm/cone_pixel_indices` provide
a complete map from any pixel in the list-mode image back to the exact cone, event,
and hit data that generated it.



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

When `input_format = "hdf5_ngimager"`, the pipeline interprets the contents of the HDF5 file and resumes *exactly* from the stage corresponding to `[pipeline].until`:

- If `/cones/*` exists but `/images/*` does not:
    - Start at cones → images.
- If `/events/*` exists but `/cones/*` does not:
    - Start at events → cones → images.
- If only `/hits/*` exist:
    - Start at hits → events → cones → images.

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
         - `pipeline.until = "image"`
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

## 15.5 Current Implementation Status (for developers)

As of the current ngimager snapshot:

- The unified pipeline (`pipelines.core.run_pipeline`) is active and is the only supported end-to-end entry point.

- The PHITS path follows the intended staged flow:

  - `PHITSAdapter.iter_raw_events(...)` emits canonical `Hit` objects grouped into raw coincidence windows.
  - Hit-level cuts are applied via `apply_hit_filters`, followed by an `is_reconstructable` check that can discard raw events early after hit filtering.
  - Surviving hits are passed through `shape_events_for_cones` (shaper) and then `shaped_to_typed_events` to yield `NeutronEvent` and `GammaEvent` instances.
  - Typed events always carry their constituent `Hit` objects.

- Shaping and typed-event conversion are implemented for both neutrons and gammas:
  - Neutron events are currently restricted to simple 2-hit topologies.
  - Gamma events are currently restricted to 3-hit events; the default sequencing policy for PHITS data is time-ordered (`gamma_policy = "time_asc"`).

- Hit-level filters and basic event-level filters are in place and plumbed through the PHITS path, with counters that distinguish neutrons vs gammas where meaningful. A full audit of counters against the `_total` / `_n` / `_g` naming convention is still pending.

- Energy strategies are implemented in `physics.energy_strategies` and are used in the pipeline:
  - `make_energy_strategy(cfg.energy, cfg.materials)` is called from the cones stage.
  - The **Edep** strategy is used for PHITS workflows where `Hit.L` represents deposited energy (e.g. `Edep_MeV`), matching the current PHITS toy examples.
  - LUT-based and ToF-based neutron strategies exist and are wired, but still need dedicated validation on real or legacy datasets.

- Cone construction is handled in `physics.cones`:

  - Neutron cones:
    - Built from `NeutronEvent` plus the energy strategy, using the standard kinematics.
    - Directionality tests based on cone–plane intersection (`t_int > 0`) are implemented for gamma cones and are planned for neutrons to ensure cones point toward the imaging plane.

  - Gamma cones:
    - Implemented for 3-hit Compton events following the NOVO primer:
      - All 6 permutations of the 3 hits are tested.
      - Kinematically impossible permutations are rejected (invalid Compton angle, etc.).
      - For each viable permutation, a candidate cone is built.
    - Candidate cones are scored using a prior-aware angle metric:
      - Compute the estimated scatter angle θ_est from the prior (vector from apex to prior point).
      - Compute the angle between the cone axis and the vector from apex to prior point.
      - Use Δ = |θ_calc − θ_est| as the quality metric; the permutation with the smallest Δ is selected.
      - When no explicit prior is configured, the imaging plane center is used as a reasonable default prior point.
    - This gamma cone path is implemented and produces images.

- Priors (`physics.priors`) support both point and line priors and are constructed via `make_prior(cfg.prior)`. The same prior object is used for neutron and gamma cone selection, with a fallback to the imaging plane center when no prior is provided.

- HDF5 output follows the unified layout described in §12:
  - `/lm` stores event-level information, including `event_type` (0 = neutron, 1 = gamma).
  - `/cones` stores cone parameters plus a `species` field (0 = neutron, 1 = gamma).
  - Indices allow tracing cones back to events and hits.
  - At present, summed images are written to `/images/summed/n`. Species-split gamma images and combined `/images/summed/g` and `/images/summed/all` are planned but not yet implemented.

- SBP imaging (`imaging.sbp.reconstruct_sbp`) is functional and operates directly on the `Cone` dataclass and `Plane`. List-mode (per-cone sparse footprints) infrastructure exists but still needs a full pass of testing and documentation.

Overall, the PHITS→hits→events→cones→SBP path is operational for both neutrons and gammas, with gamma imaging conceptually implemented but awaiting robust validation and improved diagnostics.


---

## 16. Roadmap (Refactor and Implementation Checklist)

This checklist tracks migration from the current state to the architecture described here.

### 16.1. Cleanup and Deletion of Redundant Paths

- [x] Remove any parallel PHITS→Hit/Event paths; ensure `PHITSAdapter` is the **only** canonical route for PHITS.
- [x] Deprecate and remove `sim.*` from the active pipeline; if synthetic capabilities are retained, move them to examples/tests.
- [x] Ensure only one `Hit` class exists and is used everywhere.

### 16.2. Adapters and Raw Events

- [x] `PHITSAdapter.iter_raw_events` now returns canonical Hit objects grouped into raw coincidence windows as designed.
- [ ] Implement/clean up `ROOTAdapter.iter_raw_events` with `input_format = "root_novo_ddaq"` for the current acquisition system.
- [x] Implement the early `is_reconstructable` logic after hit-level filtering to discard unviable raw events, with appropriate counters.

### 16.3. Shaper and Typed Events

- [x] Make `shape_events_for_cones` the single entry point from raw-event hits to shaped events.
- [x] Make `shaped_to_typed_events` the only path to `NeutronEvent`/`GammaEvent`.
- [x] Typed events now always carry canonical `Hit` objects, and HDF5 round-trip storage is implemented.
- [ ] Ensure counters at the hit and event levels follow the `_n` / `_g` / `_total` naming pattern where meaningful, and audit existing counters for consistency.


### 16.4. Filters, Priors, and Sequencing


- [ ] Centralize event and cone selection logic into `filters` modules, driven by `[filters]` config (currently, some selection and scoring logic still lives in `physics.cones` and `pipelines.core`).
- [x] Ensure priors are only defined in `physics.priors` and configured via `[prior]`.
- [ ] Implement `enumerate_candidate_cones` and `select_cones` so that:
    - Multiple candidate cones per event (proton vs carbon, gamma permutations) are supported in a unified way.
    - At most one cone per event is ultimately selected by a dedicated selector function (rather than ad-hoc selection inside the cone-building code).
- [ ] Implement storage of gamma sequencing choice and neutron recoil interpretation in the event objects and HDF5.


### 16.5. Energy Strategy Integration

- [x] Wire `make_energy_strategy(cfg.energy, cfg.materials)` into the pipeline, with all neutron energy calculations happening via this interface.
- [x] Integrate E(L) LUTs for OGS and M600 via `io.lut.LUT`.
- [ ] Validate `EnergyFromToF` and `EnergyFromFixedIncident` paths with simple tests.
- [ ] Document the conceptual picture in code comments, referencing this document.

### 16.6. Cone Construction and Imaging

- [x] Ensure `physics.cones` provides the canonical functions for building candidate cones from events (both neutrons and gammas).
- [x] Confirm `imaging.sbp.reconstruct_sbp` works directly from the `Cone` dataclass and `Plane`.
- [ ] Implement optional per-cone sparse footprints used only when `run.list` is true, and validate them end-to-end.
- [ ] Ensure cone and imaging counters follow the `_n` / `_g` / `_total` naming pattern where meaningful, and that per-stage runtimes are recorded and reported.
- [x] Implement initial gamma cone-building for 3-hit Compton events in `physics.cones`, including permutation testing and prior-based Δ = |θ_calc − θ_est| scoring; treat this path as “provisionally implemented” pending further validation and ROOT adapter integration.
- [x] Extend neutron cone construction to support explicit proton vs carbon candidate branches, including directionality checks analogous to the gamma `t_int > 0` test, and integrate this into the unified candidate/selection framework.
- [ ] Implement combined n/g/all SBP images at the imaging stage (see §16.8 for `/images/summed/[n|g|all]`).
- [ ] Neutron proton vs carbon recoil hypothesis handling:
    - [x] Adopt primer-consistent neutron cone geometry (apex at h1, axis from h2→h1).
    - [x] Specify species-aware recoil interpretation tied to energy strategies (ELUT, Edep).
    - [x] Define candidate-cone generation for H and C branches with Δ-based prior scoring.
    - [ ] Integrate full H/C candidate enumeration and selection into the unified cone-candidate framework.
    - [ ] Add complete HDF5 recoil metadata and ensure counters reflect per-species filtering.
- [ ] Cone-level angle-difference filtering (Δθ):
    - [ ] Add `max_angle_diff_deg` to `[filters.cones]` with species-specific overrides.
    - [ ] Apply Δ = |θ_calc − θ_est| thresholds during cone selection.
    - [ ] Add associated counters and expose the filtering results in `/meta/counters`.


### 16.6.1 Gamma Cone Status (Provisional)

The gamma cone construction path described in §16.6 is now implemented in `physics.cones` and exercised through the unified pipeline:

- Gamma events are restricted to 3-hit Compton triples.
- All 3! permutations of the hit ordering are tested.
- Kinematically invalid permutations (e.g. impossible Compton angle) are rejected.
- Each viable candidate cone undergoes an axis-toward-plane check (`t_int > 0`) to ensure only physically meaningful half-cones are retained.
- Candidate cones are scored using a prior-aware metric:

    \[
        \Delta = \left| \phi_{\text{prior}} - \theta_{\text{Compton}} \right|,
    \]

    where:
    - \(\phi_{\text{prior}}\) is the angle between the cone axis and the vector from the relevant scatter point toward the prior location (point or line prior; defaults to the imaging-plane center),
    - \(\theta_{\text{Compton}}\) is the scatter angle implied by the candidate ordering.

- The candidate with minimal Δ is selected as the single "finalist" cone for that event.

This constitutes a full first implementation of gamma Compton-cone building and selection consistent with the NOVO imaging primer and the legacy code’s behavior. 

#### Gamma imaging status

The Compton (gamma) path is now **fully wired and validated** against the legacy NOVO code, at least for the model datasets we have. The current implementation:

- builds Compton cones via `build_cone_from_gamma`, using:
  - three-hit `GammaEvent` objects,
  - permutations of hit ordering when a plane is available,
  - kinematics from `physics.kinematics` for Eg and θ₁, and
  - the same Δ = |φ − θ| prior scoring used for neutrons;
- reproduces legacy images on legacy-formatted PHITS data that has been converted into the new `phits_usrdef` adapter format; and
- shares the same filter stack (hit-level, event-level, cone-level) and HDF5 output conventions as the neutron path.

Earlier uncertainty about gamma reconstruction was traced to a buggy custom PHITS tally, not to the imaging chain itself. With clean input, the new gamma implementation matches the legacy behavior.



### 16.7. Unified Pipeline and CLI

- [x] Make `pipelines.core.run_pipeline` the central pipeline function.
- [x] Deprecate/remove `pipelines.fastmode` and `pipelines.listmode` in favor of a single CLI that respects `run.fast`, `run.list`, `run.use_neutrons`, and `run.use_gammas` from config and CLI flags.
- [ ] Implement CLI flags:
    - `--fast` / `--no-fast`
    - `--list` / `--no-list`
    - `--stop-stage`
    - optional convenience flags like `--neutrons-only` / `--gammas-only` mapped to `use_neutrons` / `use_gammas`
- [ ] Implement `pipeline.until` gating at the main stages and support resuming from ngimager HDF5 files (`input_format = "hdf5_ngimager"`).


### 16.8. HDF5 and Visualization

- [x] **Back-tracing**: events, cones, and list-mode pixels are now linked via explicit indices (`/lm/event_survival`, `/cones/event_index`, `/lm/cone_pixel_indices`).
- [x] **Species-separated images**: `/images/summed/n` and `/images/summed/g` are always written when the corresponding species are enabled; `/images/summed/all` is written when both are non-empty.
- [x] **Counters in file**: the full counters dict is stored under `/meta/counters/*`.
- [ ] **Restart from cones/image**: the current pipeline always starts from events; an “imaging-only” restart mode using an existing ngimager HDF5 file is still a future
  enhancement.
- [ ] **CLI wrappers / nicer entry points**: higher-level console scripts (e.g. `ngimager run config.toml`) are still on the roadmap.
- [ ] Wrap PNG export in a clean CLI function that calls `vis.hdf.save_summed_png` driven by `[vis]`, outputs a PNG per summed image (so, at most, 3: n + g + all)
- [ ] Support imaging-only reruns from existing cones (ngimager HDF5 input) to generate list-mode per-cone images from previously non-list-mode outputs.
- [ ] For ROOT adapter, implement propagation of metadata, including run number (to `/lm/event_meta_run_id`), into the HDF5 output.



### 16.9. Documentation and Example Configs

- [ ] Add an example `.toml` in `examples/configs/` with extensive comments explaining each section and option.
- [ ] Provide a worked example for:
    - `examples/imaging_datasets/PHITS_simple_ng_source/usrdef.out`
- [ ] Use this document as the basis of a “Developer Tour” page in `docs/dev/architecture.md` (or similar) to keep the code and architecture aligned.

---

This document is intended to be the **ground truth design** for `ngimager`. As code evolves, it should either be updated to reflect the new reality or drive refactors back into alignment, keeping the system clean, maintainable, and physically faithful to the NOVO imaging goals.
