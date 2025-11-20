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

      https://github.com/Lindt8/ng-imager

- You import the code in Python as:

      import ngimager

- Once the project is published to PyPI, you will install it with:

      pip install ngimager

For now, installation is done from a local clone (editable install).

---

## 1. Install ngimager (development)

### From PyPI via `pip` (not yet distributed)



### From source

1. Clone the repository:

       git clone https://github.com/Lindt8/ng-imager.git
       cd ng-imager

2. (Recommended) Create and activate a virtual environment, for example with `venv`:

       python -m venv .venv
       source .venv/bin/activate     # Linux / macOS
       .venv\Scripts\activate        # Windows PowerShell / CMD

3. Install the package in editable mode along with its dependencies:

       pip install -e ".[dev]"

   or, if you just want the core runtime:

       pip install -e .

After this, `ngimager` should be importable in Python:

    python -c "import ngimager; print(ngimager.__file__)"

and lets you run the pipeline via `python -m`.

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
kind = "phits_usrdef"

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
type   = "point"
point  = [0.0, 0.0, 0.0]
strength = 1.0
```

---

## 3. Run the pipeline

From the project root:

```bash
python -m ngimager.pipelines.core my_config.toml
```

You should see log messages for each stage:

- Stage 1: raw events → hits (hit-level filters)
- Stage 2: hits → shaped/typed events → event filters
- Stage 3: events → cones (with cone filters)
- Stage 4: cones → images (SBP)

and a short summary at the end with counters per stage.

The command returns the path to the generated HDF5 file.

### Useful CLI flags

- `--fast` — enable fast-mode presets (higher thresholds, cone cap, coarser image).
- `--list` — enable list-mode imaging output (`/lm/cone_pixel_indices`, etc.).
- `--neutrons / --no-neutrons` — enable/disable neutron processing.
- `--gammas / --no-gammas` — enable/disable gamma processing.

Example:

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
```

See the [HDF5 Output Format](hdf5.md) page for a full description of the
groups and how to map pixels ↔ cones ↔ events ↔ hits.

---


## 5. Next steps

Once you have basic images working, you may want to:

- Tune `[filters.*]` to match your experiment (thresholds, ToF windows, cone cuts).
- Enable `run.fast = true` for quick-look images during setup.
- Enable `run.list = true` to get full list-mode mappings for offline analysis.
- Compare reconstructed energy spectra (e.g. `/cones/incident_energy_MeV`) with
  Monte Carlo truth or analytic expectations.

The goal is that you can:

- Treat the TOML config as the single source of truth for a run.
- Keep the HDF5 output as a fully self-contained record of the inputs, filters, counters, and resulting images.

For a deeper dive into the design and physics, see
[`docs/dev/architecture.md`](../dev/architecture.md) and the NOVO imaging primer
PDF in the repository.
