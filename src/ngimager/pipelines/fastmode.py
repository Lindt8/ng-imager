from __future__ import annotations

import sys
from pathlib import Path
import typer

from ngimager.pipelines.core import run_pipeline

app = typer.Typer(help="Fast (summary) NOVO imaging pipeline")


@app.command()
def main(cfg_path: str = typer.Argument(..., help="Path to TOML config file")):
    """
    Run the fast-mode imaging pipeline.

    This is effectively the unified pipeline with mode='fast':
      - aggressive cuts / max_cones controlled in [run] section
      - only summed image is written (no list-mode payloads)
    """
    out = run_pipeline(cfg_path, mode_override="fast")
    typer.echo(f"Wrote HDF5: {out}")


if __name__ == "__main__":
    # Allow `python -m ngimager.pipelines.fastmode config.toml`
    if len(sys.argv) > 1 and sys.argv[1].endswith(".toml"):
        main(sys.argv[1])
    else:
        typer.run(main)
