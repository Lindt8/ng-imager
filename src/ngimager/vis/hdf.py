from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SPECIES_LABEL = {
    "n": "Neutron",
    "g": "Gamma",
    "all": "Neutron + Gamma",
}


def _safe_get_attr(attrs: h5py.AttributeManager, key: str, default: float | None = None) -> float | None:
    if key not in attrs:
        return default
    try:
        return float(attrs[key])
    except Exception:
        return default


def _basis_world_name(vec: np.ndarray, tol: float = 1e-3) -> Optional[str]:
    """
    Try to infer if a basis vector is aligned with ±x, ±y, or ±z.
    Returns "x", "y", "z", or None.
    """
    v = np.asarray(vec, dtype=float)
    if not np.all(np.isfinite(v)):
        return None
    norm = np.linalg.norm(v)
    if norm == 0:
        return None
    v = v / norm

    basis = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }

    best_name: Optional[str] = None
    best_dot = 0.0
    for name, axis in basis.items():
        d = abs(float(np.dot(v, axis)))
        if d > best_dot:
            best_dot = d
            best_name = name
    if best_dot >= 1.0 - tol:
        return best_name
    return None


def _image_axes_from_meta(
    meta: Optional[h5py.Group],
    *,
    center_on_plane_center: bool = True,
) -> tuple[Optional[tuple[float, float, float, float]], str, str]:
    """
    Derive imshow extent and axis labels from /meta attributes.

    Returns (extent, xlabel, ylabel), where extent is (u_min, u_max, v_min, v_max)
    in centimeters, or None if we cannot infer physical axes.
    """
    if meta is None:
        return None, "u (pixels)", "v (pixels)"

    attrs = meta.attrs

    u_min = _safe_get_attr(attrs, "grid.u_min")
    u_max = _safe_get_attr(attrs, "grid.u_max")
    v_min = _safe_get_attr(attrs, "grid.v_min")
    v_max = _safe_get_attr(attrs, "grid.v_max")

    extent: Optional[tuple[float, float, float, float]]
    if None in (u_min, u_max, v_min, v_max):
        extent = None
    else:
        if center_on_plane_center:
            u_center = 0.5 * (u_min + u_max)  # type: ignore[operator]
            v_center = 0.5 * (v_min + v_max)  # type: ignore[operator]
        else:
            u_center = 0.0
            v_center = 0.0
        extent = (
            float(u_min - u_center),  # type: ignore[operator,arg-type]
            float(u_max - u_center),  # type: ignore[operator,arg-type]
            float(v_min - v_center),  # type: ignore[operator,arg-type]
            float(v_max - v_center),  # type: ignore[operator,arg-type]
        )

    # Try to infer world-axis alignment for u, v
    eu = attrs.get("plane.eu")
    ev = attrs.get("plane.ev")
    u_world = _basis_world_name(np.asarray(eu)) if eu is not None else None
    v_world = _basis_world_name(np.asarray(ev)) if ev is not None else None

    if u_world and v_world:
        xlabel = f"u / {u_world} [cm]"
        ylabel = f"v / {v_world} [cm]"
    elif u_world:
        xlabel = f"u / {u_world} [cm]"
        ylabel = "v [cm]"
    elif v_world:
        xlabel = "u [cm]"
        ylabel = f"v / {v_world} [cm]"
    else:
        xlabel = "u [cm]"
        ylabel = "v [cm]"

    return extent, xlabel, ylabel


def _cone_counts(meta: Optional[h5py.Group]) -> dict[str, Optional[int]]:
    """
    Fetch simple cone counts from /meta/counters if present.

    Returns dict with keys "n" and "g", each mapping to an int or None.
    """
    counts: dict[str, Optional[int]] = {"n": None, "g": None}
    if meta is None or "counters" not in meta:
        return counts

    counters = meta["counters"]
    for key, species in (("s3_cones_kept_n", "n"), ("s3_cones_kept_g", "g")):
        if key in counters:
            try:
                value = int(counters[key][()])
            except Exception:
                value = None
            counts[species] = value

    return counts


def _render_single_image(
    data: np.ndarray,
    out_path: Path,
    *,
    extent: Optional[tuple[float, float, float, float]],
    xlabel: str,
    ylabel: str,
    title: str,
    subtitle: Optional[str] = None,
    count_text: Optional[str] = None,
    flip_vertical: bool = False,
    rasterized: bool = False,
) -> None:
    """
    Internal helper: render a 2D array to an image file.
    """
    if flip_vertical:
        data = np.flipud(data)

    fig, ax = plt.subplots(figsize=(6, 5))

    if extent is not None:
        im = ax.imshow(data, origin="lower", extent=extent, rasterized=rasterized)
    else:
        im = ax.imshow(data, origin="lower", rasterized=rasterized)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Counts (arb.)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    lines = []
    if subtitle:
        lines.append(subtitle)
    if count_text:
        lines.append(count_text)

    if lines:
        ax.text(
            0.01,
            0.99,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75),
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_summed_png(
    h5_path: str | Path,
    out_png: Optional[str] = None,
    dataset: str = "/images/summed/n",
) -> str:
    """
    Render a single 2D dataset from HDF5 to a PNG.

    Historically this was used for `/images/summed/*`, but it will accept any
    2D dataset path. Axes are labeled using the imaging-plane metadata when
    available; otherwise pixel indices are used.

    This function is mainly for CLI helpers; new code should prefer
    `render_summed_images` when working with /images/summed/{n,g,all}.
    """
    h5_path = Path(h5_path)

    if out_png is None:
        out_path = h5_path.with_suffix(".png")
    else:
        out_path = Path(out_png)

    with h5py.File(h5_path, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset {dataset!r} not found in {h5_path}")
        data = np.asarray(f[dataset])
        if data.ndim != 2:
            raise ValueError(f"Dataset {dataset!r} is not 2D (shape={data.shape!r})")

        meta = f["meta"] if "meta" in f else None
        extent, xlabel, ylabel = _image_axes_from_meta(meta, center_on_plane_center=True)

        _render_single_image(
            data,
            out_path,
            extent=extent,
            xlabel=xlabel,
            ylabel=ylabel,
            title=dataset,
            subtitle=str(h5_path.name),
            count_text=None,
            flip_vertical=False,
            rasterized=False,
        )

    return str(out_path)


def render_summed_images(
    h5_path: str | Path,
    species: Sequence[str] = ("n",),
    *,
    filename_pattern: str = "{species}_{stem}.{ext}",
    center_on_plane_center: bool = True,
    flip_vertical: bool = False,
    formats: Sequence[str] = ("png",),
) -> list[Path]:
    """
    Render one or more `/images/summed/{n,g,all}` datasets from an ng-imager
    HDF5 file.

    Parameters
    ----------
    h5_path:
        Path to the ng-imager HDF5 output file.
    species:
        Iterable of species strings: any combination of "n", "g", "all".
    filename_pattern:
        Format string for output filenames. It may reference:
          - {stem}:    stem of the HDF5 file name
          - {species}: "n", "g", or "all"
          - {ext}:     file extension (e.g. "png", "pdf")
    center_on_plane_center:
        If True, shift the plotting coordinates so that (u, v) = (0, 0) is at
        the imaging plane center.
    flip_vertical:
        If True, flip the plotted image vertically relative to the natural
        v-axis orientation. This is mainly useful for comparison with legacy
        images.
    formats:
        Iterable of output formats (e.g. ["png", "pdf"]).

    Returns
    -------
    list[Path]
        A list of written image paths.
    """
    h5_path = Path(h5_path)
    species_list = [s for s in species if s in ("n", "g", "all")]
    if not species_list:
        return []

    # Deduplicate formats while preserving order
    fmt_list: list[str] = []
    for fmt in formats:
        ext = str(fmt).lower()
        if not ext:
            continue
        if ext not in fmt_list:
            fmt_list.append(ext)

    out_paths: list[Path] = []

    with h5py.File(h5_path, "r") as f:
        meta = f["meta"] if "meta" in f else None
        extent, xlabel, ylabel = _image_axes_from_meta(
            meta,
            center_on_plane_center=center_on_plane_center,
        )
        counts = _cone_counts(meta)

        for s in species_list:
            dset_path = f"/images/summed/{s}"
            if dset_path not in f:
                continue

            data = np.asarray(f[dset_path])
            if data.ndim != 2:
                continue

            label = _SPECIES_LABEL.get(s, s)
            title = f"{label} cones – {h5_path.name}"
            subtitle = dset_path

            count_text: Optional[str] = None
            n = counts.get("n")
            g = counts.get("g")
            if s == "n" and n is not None:
                count_text = f"{n} neutron event cones"
            elif s == "g" and g is not None:
                count_text = f"{g} gamma event cones"
            elif s == "all" and (n is not None or g is not None):
                parts = []
                if n is not None:
                    parts.append(f"{n} n")
                if g is not None:
                    parts.append(f"{g} g")
                if parts:
                    count_text = f"{' + '.join(parts)} event cones"

            for ext in fmt_list:
                filename = filename_pattern.format(
                    stem=h5_path.stem,
                    species=s,
                    ext=ext,
                )
                out_path = h5_path.with_name(filename)

                _render_single_image(
                    data,
                    out_path,
                    extent=extent,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=title,
                    subtitle=subtitle,
                    count_text=count_text,
                    flip_vertical=flip_vertical,
                    rasterized=(ext == "pdf"),
                )
                out_paths.append(out_path)

    return out_paths
