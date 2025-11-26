"""
High-level imaging pipelines for ng-imager.

The main entry point is `run_pipeline`, which is what the `ng-run`
command-line tool calls under the hood.
"""

from .core import run_pipeline

__all__ = ["run_pipeline"]
