from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from ngimager.vis.hdf import render_summed_images, save_summed_png

app = typer.Typer(help="ngimager visualization tools")


@app.command("summed")
def summed(
    h5_path: str = typer.Argument(
        ...,
        help="Path to ng-imager HDF5 file (must contain /images/summed/*).",
    ),
    species: List[str] = typer.Option(
        ["n"],
        "--species",
        "-s",
        help="Which images to render from /images/summed: any of 'n', 'g', 'all'.",
    ),
    center_on_plane_center: bool = typer.Option(
        True,
        "--center-on-plane-center/--no-center-on-plane-center",
        help="Center axes on the imaging plane center.",
    ),
    flip_vertical: bool = typer.Option(
        False,
        "--flip-vertical/--no-flip-vertical",
        help="Flip the plotted image vertically (mainly for legacy comparison).",
    ),
    filename_pattern: str = typer.Option(
        "{species}_{stem}.{ext}",
        "--filename-pattern",
        help="Python format string for output filenames; may use {stem}, {species}, {ext}.",
    ),
    fmt: List[str] = typer.Option(
        ["png"],
        "--format",
        "-f",
        help="Output format(s) to write (e.g. 'png', 'pdf').",
    ),
):
    """
    Render one or more /images/summed/{n,g,all} datasets to image files.
    """
    out_paths = render_summed_images(
        h5_path,
        species=species,
        filename_pattern=filename_pattern,
        center_on_plane_center=center_on_plane_center,
        flip_vertical=flip_vertical,
        formats=fmt,
    )
    if not out_paths:
        typer.echo("No images were written (check that /images/summed/* exist).")
    else:
        for p in out_paths:
            typer.echo(str(p))


@app.command("h5-to-png")
def h5_to_png(
    h5_path: str = typer.Argument(
        ...,
        help="Path to HDF5 file containing a 2D dataset.",
    ),
    dataset: str = typer.Option(
        "/images/summed/n",
        "--dataset",
        "-d",
        help="Path of the 2D dataset to render (default /images/summed/n).",
    ),
    out: Optional[str] = typer.Option(
        None,
        "--out",
        "-o",
        help="Output PNG path (defaults to <h5stem>.png).",
    ),
):
    """
    Render a single 2D dataset from HDF5 to a PNG (legacy-style helper).
    """
    out_png = save_summed_png(h5_path, out_png=out, dataset=dataset)
    typer.echo(str(out_png))


if __name__ == "__main__":
    app()
