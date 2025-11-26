from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from ngimager.vis.hdf import render_summed_images

app = typer.Typer(help="ngimager visualization tools")


@app.command("summed")
def summed(
    h5_path: str = typer.Argument(
        ...,
        help="Path to ng-imager HDF5 file (must contain /images/summed/*).",
    ),
    species: List[str] = typer.Option(
        ["n", "g", "all"],
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
        True,
        "--flip-vertical/--no-flip-vertical",
        help="Flip the plotted image vertically (mainly for legacy comparison).",
    ),
    axis_units: str = typer.Option(
        "cm",
        "--axis-units",
        help="Axis units for plotting: 'cm' or 'mm'.",
    ),
    cmap: str = typer.Option(
        "cividis",
        "--cmap",
        help="Matplotlib colormap to use (e.g. 'cividis', 'viridis').",
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
    plot_label: Optional[str] = typer.Option(
        None,
        "--plot-label",
        help=(
            "Override run plot label annotation (defaults to any value stored "
            "under /meta.run_plot_label in the HDF5 file)."
        ),
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
        axis_units=axis_units,
        cmap=cmap,
        formats=fmt,
        plot_label=plot_label,
    )
    if not out_paths:
        typer.echo("No images were written (check that /images/summed/* exist).")
    else:
        for p in out_paths:
            typer.echo(str(p))


if __name__ == "__main__":
    app()
