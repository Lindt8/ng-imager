from __future__ import annotations

import sys
import typer

from ngimager.pipelines.core import run_pipeline

app = typer.Typer(help="Full list-mode NOVO imaging pipeline")


@app.command()
def main(cfg_path: str = typer.Argument(..., help="Path to TOML config file")):
    """
    Run the list-mode imaging pipeline.

    This is the unified pipeline with mode='list':
      - all cones are imaged (subject to filters)
      - per-cone geometry and list-mode pixel indices are stored
      - per-event / per-hit physics is stored under /lm/*
    """
    out = run_pipeline(cfg_path, mode_override="list")
    typer.echo(f"Wrote HDF5: {out}")


if __name__ == "__main__":
    # Allow `python -m ngimager.pipelines.listmode config.toml`
    if len(sys.argv) > 1 and sys.argv[1].endswith(".toml"):
        main(sys.argv[1])
    else:
        typer.run(main)
