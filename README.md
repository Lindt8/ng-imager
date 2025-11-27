# ng-imager
[![CI](https://github.com/Lindt8/ng-imager/actions/workflows/ci.yml/badge.svg)](https://github.com/Lindt8/ng-imager/actions/workflows/ci.yml)
[![Docs](https://github.com/Lindt8/ng-imager/actions/workflows/docs.yml/badge.svg)](https://Lindt8.github.io/ng-imager/)
[![GitHub Pages](https://img.shields.io/badge/docs-online-brightgreen?style=flat&logo=readthedocs)](https://Lindt8.github.io/ng-imager/)
[![PyPI](https://img.shields.io/pypi/v/ngimager.svg)](https://pypi.org/project/ngimager/)

**ng-imager** is a modular Python toolkit for imaging **neutrons** and **gamma rays** in the NOVO detector system.

It turns detector-level hit data into images via a clean, restartable pipeline:

> hits → events → cones → images

and is designed to replace a monolithic legacy script with:

- Explicit separation of physics, geometry, imaging, configuration, and I/O  
- Flexible energy strategies (ELUT, ToF, fixed–energy, direct Edep)  
- Unified neutron and gamma imaging, including proton vs carbon recoil handling  
- Structured hit / event / cone filters with detailed counters  
- Self-describing HDF5 output suitable for long-term analysis and archiving  

---

## Repository vs Python Package Name

- **Repository name**: `ng-imager`  
- **Python package (import name)**: `ngimager`  
- **PyPI package**: `ngimager`

After installing from source, you import the package as:

```python
import ngimager
```

and use the CLI entry points:

- `ng-run` – main imaging pipeline
- `ng-viz` – visualization / re-rendering of summed images
- `ng-inspect` – inspect a single cone and its pixel footprint (optional tool)


---

## Install 

### From PyPI (recommended)

```bash
pip install ngimager
```

Check that the package and CLI are available:

```bash
python -c "import ngimager; print(ngimager.__file__)"
ng-run --help
ng-viz --help

```

### From Source (development install)

```bash
git clone https://github.com/Lindt8/ng-imager.git
cd ng-imager

python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

This installs `ngimager` in editable mode along with development dependencies (tests, docs, etc.). For a minimal runtime install, you can also use:

```bash
pip install -e .
```

---

## First Run: Example PHITS Config

The repository includes a simple PHITS usrdef example:

- Config: `examples/configs/phits_usrdef_simple.toml`  
- Input: `examples/imaging_datasets/PHITS_simple_ng_source/usrdef.out`

Run the unified pipeline with:

```bash
ng-run examples/configs/phits_usrdef_simple.toml
```

or equivalently:

```bash
python -m ngimager.pipelines.core examples/configs/phits_usrdef_simple.toml
```

This will:

1. Read PHITS usrdef events via the adapter.  
2. Construct hits and apply hit-level filters.  
3. Shape and type events (neutron 2-hit, gamma 3-hit).  
4. Apply event- and cone-level filters (ToF windows, light thresholds, Δθ, incident energy).  
5. Build neutron and gamma cones (with proton vs carbon recoil handling for neutrons).  
6. Reconstruct a simple back-projection (SBP) image on the configured plane.  
7. Write an HDF5 file containing:

   - `/images/summed/n`, `/images/summed/g`, and `/images/summed/all` (where available)  
   - `/cones/...` with per-cone geometry, species, recoil code, and incident energy  
   - `/lm/...` with per-event and per-hit physics and optional list-mode cone pixel indices  
   - `/meta/counters/...` summarizing all filters and pipeline stages  
   - A snapshot of the TOML config used for the run  

You can then use ng-viz to re-render images from that HDF5 file:

```bash
ng-viz summed results/example.h5
```


---

## Documentation

Full documentation is hosted at:

- https://Lindt8.github.io/ng-imager/

Key sections:

- **Quickstart** – run the example config and inspect the results.  
- **Configuration** – complete TOML reference (`docs/config.md`).  
- **HDF5 Layout** – how to interpret the output file (`docs/hdf5.md`).  
- **Architecture** – design overview and long-term roadmap (`docs/architecture.md`).  
- **API Reference** – auto-generated documentation from Python docstrings (`docs/api/index.md`).  

---

## Status

- Neutron imaging: functional and validated against the legacy NOVO imaging code.  
- Gamma imaging: functional and validated on legacy-formatted model data.  
- Proton vs carbon recoil hypotheses: implemented, selected via priors, and stored in HDF5.  
- Hit / event / cone filters: implemented with counters and HDF5 export.  
- Fast and list modes: wired through configuration and I/O.

The codebase is under active development, but the core architecture and HDF5 layout are intended to be stable enough for experimental usage and future extension.

---

## Contributing

Contributions are very welcome. Useful areas include:

- New I/O adapters (additional simulation formats, experimental data).  
- Alternative imaging algorithms / uncertainty models.  
- Improved tests, examples, and documentation.

If you’d like to help:

1. File an issue describing your idea or bug.  
2. Open a pull request from a feature branch.  
3. Use the existing example config and tests to validate your changes.

Repository:

```text
https://github.com/Lindt8/ng-imager
```

The aim is to keep ng-imager **physically transparent**, **modular**, and **pleasant to work with** for both simulation and experimental campaigns.