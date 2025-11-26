"""
Top-level package for ng-imager (ngimager).

The public user-facing entry points are:

- ngimager.pipelines.core.run_pipeline  → used by the `ng-run` CLI
- ngimager.cli.viz.app                  → used by the `ng-viz` CLI

Internally, the package is organized into:

- physics   – hits, events, kinematics, cones, energy strategies
- geometry  – imaging plane and coordinate transforms
- imaging   – SBP and future imaging algorithms
- io        – adapters, list-mode HDF5 storage, LUT loading
- config    – Pydantic config schemas and loading helpers
- pipelines – high-level orchestration / CLI pipeline entry points
- vis       – visualization utilities
- tools     – helper scripts and LUT generators
"""

__all__ = [
    "pipelines",
    "cli",
    "config",
    "geometry",
    "imaging",
    "io",
    "physics",
    "vis",
    "tools",
]
