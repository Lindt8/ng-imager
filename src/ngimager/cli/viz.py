from __future__ import annotations

import typer
from pathlib import Path
from typing import Optional

from ngimager.vis.hdf import save_summed_png

app = typer.Typer(help="NOVO imaging visualization tools")

@app.command("h5-to-png")
def h5_to_png(
    h5_path: str = typer.Argument(..., help="Path to HDF5 file containing /images/summed"),
    dataset: str = typer.Option("/images/summed", "--dataset", "-d", help="Dataset path"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output PNG path (defaults to file.png)"),
):
    """Render a 2D dataset from HDF5 (default /images/summed) to a PNG."""
    out_png = save_summed_png(h5_path, out_png=out, dataset=dataset)
    typer.echo(f"Wrote {out_png}")

if __name__ == "__main__":
    app()
