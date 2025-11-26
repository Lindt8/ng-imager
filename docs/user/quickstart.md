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

- You clone the project from:

    ```bash
    https://github.com/Lindt8/ng-imager
    ```

- You import the code in Python as:

    ```python
    import ngimager
    ```

- Once the project is published to PyPI, you will install it with:

    ```bash
    pip install ngimager
    ```
  
For now, installation is done from a local clone (editable install).

---

## 1. Install ngimager (development)

### From PyPI via `pip` (not yet distributed)



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

---

## 4. Inspect the HDF5 output

You can inspect the file with:

- **HDFView** — a GUI browser, nice for sanity checks on groups and datasets.
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
