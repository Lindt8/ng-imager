# Quickstart

This page gives a minimal workflow for running `ngimager` on PHITS
user-defined tally output and inspecting the results.

For more detail, see:

- [Configuration](config.md)
- [HDF5 Output Format](hdf5.md)
- [Architecture](../dev/architecture.md)

---

## 0. Names and Where Things Live

There are three closely related names you will see:

- **GitHub repository**: `ng-imager`
- **Python package (import name)**: `ngimager`
- **Future PyPI package name (planned)**: `ngimager`

In practice:

- Install it from PyPI with:

    ```bash
    pip install ngimager
    ```

- You import the code in Python as:

    ```python
    import ngimager
    ```

- Run it with a config TOML file (and optional CLI options):

    ```bash
    ng-run --help 
    ng-run my_config.toml 
    ng-run my_config.toml -i data/input.root -o results/output.h5
    ```
  
For now, installation is done from a local clone (editable install).

---

## 1. Install ngimager (development)

### From PyPI via `pip` (recommended)


```bash
pip install ngimager
```

This will provide:

- the `ngimager` Python package, and
- the CLI tools: `ng-run`, `ng-viz`, and `ng-inspect`.

You can check that the package and CLI are available:

```bash
python -c "import ngimager; print(ngimager.__file__)"
ng-run --help
ng-viz --help
```

### From source

1. Clone the repository:

    ```bash
    git clone https://github.com/Lindt8/ng-imager.git
    cd ng-imager
    ```

2. (Recommended) Create and activate a virtual environment, for example with `venv`:

    ```bash
    python -m venv .venv
    source .venv/bin/activate     # Linux / macOS
    .venv\Scripts\activate        # Windows PowerShell / CMD
    ```

3. Install the package in editable mode along with its dependencies:

    ```bash
    pip install -e ".[dev]"
    ```

   or, if you just want the core runtime:

    ```bash
    pip install -e .
    ```

After this, `ngimager` should be importable in Python:

```bash
python -c "import ngimager; print(ngimager.__file__)"
```

and the CLI entry points should be available:

```bash
ng-run --help
ng-viz --help
```

If those commands are not found, double-check that your virtual
environment is active and that the `pip install -e .` step completed
successfully.

---

## 2. Prepare a config TOML

Copy the example config and adjust:

```bash
cp examples/configs/phits_usrdef_simple.toml my_config.toml
```

Edit `my_config.toml` and set at least:

- `[io].input_path` — path to your PHITS user-defined tally output;
- `[io].output_path` — where to write the HDF5 file;
- `[detectors].material_map` — map your region IDs to material labels;
- `[plane]` — origin, normal, and grid for the imaging plane.

A very simple starting point (see the [Configuration](config.md) page for details):

```toml
[run]
neutrons = true
gammas   = true
fast     = false
list     = false
diagnostics_level = 1

[io]
input_path   = "phits_output/usrdef.dat"
input_format = "phits_usrdef"
output_path  = "results/example.h5"

[io.adapter]
type  = "phits"
style = "usrdef"

[detectors]
default_material = "OGS"

[detectors.material_map]
200 = "OGS"
210 = "M600"

[plane]
origin = [0.0, 0.0, 100.0]
normal = [0.0, 0.0, -1.0]
u_min  = -50.0
u_max  =  50.0
du     =   1.0
v_min  = -50.0
v_max  =  50.0
dv     =   1.0

[filters.hits]
min_light_MeVee = 50.0
max_light_MeVee = 1.0e6

[filters.events]
tof_window_ns   = [0.0, 30.0]
min_L1_MeVee    = 0.0
min_L2_MeVee    = 0.0
min_L_any_MeVee = 0.0

[filters.cones]
max_delta_theta_deg     = 5.0
max_incident_energy_MeV = 20.0

[energy]
strategy = "Edep"
force_proton_recoils = false

[prior]
type     = "point"
point    = [0.0, 0.0, 0.0]
strength = 1.0

[vis]
export_png_on_write    = true

[vis.projections]
enabled      = true

[vis.projections.metrics]
enabled = true 
```

### 2.1 Using ELUT with packaged M600/OGS LUTs

If your detectors are standard NOVO M600 / OGS scintillators and you
want to use the light→energy inversion LUTs, you can simply set:

```toml
[energy]
strategy = "ELUT"
force_proton_recoils = false
```

and **omit** `[energy.lut_paths]` entirely. The packaged LUTs for M600
and OGS (proton and carbon) will be used automatically according to
your `[detectors].material_map`.

You only need to define `[energy.lut_paths.*]` if you want to override
these defaults or add LUTs for new materials.

### 2.2 Path resolution: config vs CLI

Paths defined inside the TOML config under `[io]` are interpreted
**relative to the config file’s location**. This includes:

- `[io].input_path`
- `[io].output_path`
- `[io].restart_path` (future)
- `[io.extra_text_files]` values

For example, if your config lives at:

    configs/my_config.toml

and you write:

```toml
[io]
input_path  = "../data/usrdef.out"
output_path = "../results/example.h5"
```

then both paths are resolved relative to `configs/`, no matter where
you run `ng-run` from.

By contrast, any CLI overrides such as:

    ng-run configs/my_config.toml \
      --input-path ./override_input.out \
      --output-path ./override_output.h5

treat `./override_input.out` and `./override_output.h5` as paths
relative to your current shell working directory.



---

## 3. Run the pipeline

### 3.1 Recommended: via CLI (`ng-run`)

From the project root (or any directory, as long as `ngimager` is
installed in your environment):

```bash
ng-run my_config.toml
```

You should see log messages for each stage:

- Stage 1: raw events → hits (hit-level filters)
- Stage 2: hits → shaped/typed events → event filters
- Stage 3: events → cones (with cone filters)
- Stage 4: cones → images (SBP + optional projections and PNG export)

and a short summary at the end with counters per stage.

The command prints the path to the generated HDF5 file.

#### Useful `ng-run` CLI flags

- `--fast`  
  Enable fast-mode presets (higher thresholds, cone cap, coarser image).

- `--list`  
  Enable list-mode imaging output (`/lm/cone_pixel_indices`, etc.).

- `--neutrons / --no-neutrons`  
  Enable/disable neutron processing.

- `--gammas / --no-gammas`  
  Enable/disable gamma processing.

- `--input-path PATH`  
  Override `[io].input_path` from the TOML file.

- `--output-path PATH`  
  Override `[io].output_path` from the TOML file.

- `--plot-label "TEXT"`  
  Override `[run].plot_label` for this run (used in visualization and
  stored in the HDF5 `/meta` tree).

Example:

```bash
ng-run my_config.toml \
  --fast \
  --list \
  --no-gammas \
  --input-path /data/phits/usrdef.out \
  --output-path results/quicklook.h5 \
  --plot-label "DT beam, quicklook"
```

### 3.2 Alternative: `python -m` (module runner)

If you prefer not to use the console script, you can run the same
pipeline via the module entry point:

```bash
python -m ngimager.pipelines.core my_config.toml
```

The same flags are available:

```bash
python -m ngimager.pipelines.core my_config.toml --fast --list --no-gammas
```

### 3.3 Optional: convert HDF5 output to ROOT

If you or your collaborators prefer working in ROOT, you can convert the
ngimager HDF5 output into a ROOT file using the `ng-hdf2root` helper:

    ng-hdf2root my_run.h5

This will produce `my_run.root` in the same directory. You can then open this
file in ROOT (C++ or PyROOT) and build histograms from:

- per-hit list-mode data (`lm` tree),
- reconstructed cones (`cones` tree), and
- list-mode imaging information (`cone_pixels` and `images_summed` trees).

For details of the ROOT layout and more examples, see the dedicated
[`HDF5 → ROOT conversion` page](hdf2root.md).


---

## 4. Inspect the HDF5 output

You can inspect the file with:

- [**HDFView**](https://github.com/HDFGroup/hdfview/releases/tag/v3.3.2) — a GUI browser, nice for sanity checks on groups and datasets.
- **Python + h5py** — for scripting and analysis.

Example Python session:

```python
import h5py
import numpy as np

with h5py.File("results/example.h5", "r") as f:
    img_n = np.array(f["images"]["summed"]["n"])
    print("neutron image shape:", img_n.shape)

    cones = f["cones"]
    theta = np.array(cones["theta_rad"])
    species = np.array(cones["species"])
    print("n cones:", (species == 0).sum(), "g cones:", (species == 1).sum())

    # Run-level metadata from [run.meta]
    if "run_meta" in f["meta"]:
        print("run_meta keys:", list(f["meta"]["run_meta"].keys()))
```

See the [HDF5 Output Format](hdf5.md) page for a full description of the
groups and how to map pixels ↔ cones ↔ events ↔ hits.

---

## 5. Render images from an existing HDF5 file

If you already have an HDF5 output file (from `ng-run` or a previous
run), you can use the visualization CLI to re-render images with
different visualization settings.

The main entry point is:

```bash
ng-viz summed results/example.h5
```

By default this:

- Reads `/images/summed/{n,g,all}` from the HDF5 file,
- Uses the imaging plane metadata under `/meta`,
- Writes PNG files alongside the input HDF5 file, with names like:

    ```
    n_example.png
    g_example.png
    all_example.png
    ```

Useful options:

- `--species` / `-s` – choose which summed images to render, any subset of
  `["n", "g", "all"]`.
- `--axis-units` – `"cm"` (default) or `"mm"` for the u/v axes.
- `--cmap` – Matplotlib colormap (`"cividis"`, `"viridis"`, etc.).
- `--filename-pattern` – override the output filename pattern.
- `--format` / `-f` – write additional formats (e.g. `--format png --format pdf`).
- `--plot-label` – override the run label annotation in the figure, instead
  of using any value stored in `/meta`.

Example:

```bash
ng-viz summed results/example.h5 \
  --species n g \
  --axis-units mm \
  --cmap viridis \
  --plot-label "175 MeV p, target B, det config 3"
```

This is intended to mirror the visualization automatically produced by
the core pipeline, while letting you re-style or re-label the plots
without re-running reconstruction.

---

## 6. Next steps

Once you have basic images working, you may want to:

- Tune `[filters.*]` to match your experiment (thresholds, ToF windows, cone cuts).
- Enable `run.fast = true` for quick-look images during setup.
- Enable `run.list = true` to get full list-mode mappings for offline analysis.
- Switch `[energy].strategy` to `"ELUT"` to use the packaged M600/OGS LUTs,
  or to `"FixedEn"` / `"ToF"` for specific physics studies.
- Compare reconstructed energy spectra (e.g. `/cones/incident_energy_MeV`) with
  Monte Carlo truth or analytic expectations.

The goal is that you can:

- Treat the TOML config as the single source of truth for a run.
- Keep the HDF5 output as a fully self-contained record of the inputs, filters,
  counters, and resulting images.

For a deeper dive into the design and physics, see
[`docs/dev/architecture.md`](../dev/architecture.md) and the NOVO imaging primer PDF in the repository.


---

## 7. Inspect a single cone (`ng-inspect`)

For debugging and educational purposes, `ngimager` ships with a small
CLI tool that walks the full chain

> cone → (u, v) pixels → hits

for a single cone in an HDF5 output file.

Once `ngimager` is installed (editable or normal install), you should
have the `ng-inspect` entry point available:

    
    ng-inspect --help
    

### 7.1 Basic usage

The minimal call is:

    
    ng-inspect path/to/output.h5 --cone-index 42
    

This:

- Reads `path/to/output.h5`,
- Looks up cone index `42` under `/cones/*`,
- Follows `/cones/event_index` back to the source event in `/lm`,
- Prints:
    - cone geometry and incident energy,
    - species and neutron recoil code,
    - (for gamma cones) the stored gamma hit ordering from `/cones/gamma_hit_order`,
    - the 2–3 hits used (from `/lm/hit_*`),
    - and a summary of that cone’s pixel footprint on the imaging plane (if list-mode imaging was enabled).

Example:

    
    ng-inspect results/example_listmode.h5 --cone-index 42
    

### 7.2 Selecting a cone by event or “imaged cone” index

Instead of specifying a cone index directly, you can also start from an
event index or an “imaged cone” index:

    
    # Map an event index → cone index via /lm/event_cone_id
    ng-inspect results/example_listmode.h5 --event-index 10

    
    # Use an imaged-cone index (checked against /lm/event_imaged_cone_id
    # when present)
    ng-inspect results/example_listmode.h5 --imaged-cone-index 7
    

Exactly one of `--cone-index`, `--event-index`, or `--imaged-cone-index`
must be provided.

### 7.3 Plotting the per-cone footprint

With `run.list = true`, the HDF5 file contains the sparse cone→pixel
mapping in `/lm/cone_pixel_indices`. `ng-inspect` can turn this into a
simple `(v, u)` image and show it with Matplotlib:

    
    ng-inspect results/example_listmode.h5 --cone-index 42 --plot
    

This pops up a small `imshow` window where:

- the axes are pixel indices `(u, v)` on the imaging plane, and
- the colorbar shows counts per pixel (multiple entries for the same pixel are summed).

If list-mode imaging was not enabled for a run (i.e. `run.list = false`,
or the file was produced by an earlier pipeline stage that did not yet
write pixel mappings), `ng-inspect` will still print the cone and hit
information, but will report:

> list-mode pixel data not available (no /lm/cone_pixel_indices).

and `--plot` will produce a short message instead of a figure.

---

This tool doubles as live documentation: for anyone wanting to explore
how to walk `/cones/event_index`, `/cones/gamma_hit_order`,
`/lm/hit_*` and `/lm/cone_pixel_indices` programmatically, `ng-inspect`
shows the full mapping in a concrete, reproducible way.






