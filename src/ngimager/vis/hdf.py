from __future__ import annotations

from pathlib import Path
from typing import Sequence, Literal, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


def _count_cones_for_species(f: h5py.File, species: str) -> Optional[int]:
    """
    Count how many cones of the requested species exist in /cones/species.

    species="n"   → number of cones with code 0
    species="g"   → number of cones with code 1
    species="all" → total number of cones
    """
    try:
        ds = f["cones"]["species"]
        arr = np.asarray(ds[:], dtype=np.uint8)
    except Exception:
        return None

    if species == "n":
        return int(np.count_nonzero(arr == 0))
    if species == "g":
        return int(np.count_nonzero(arr == 1))
    if species == "all":
        return int(arr.size)
    return None


def _basis_world_name(vec: np.ndarray, tol: float = 1e-3) -> Optional[str]:
    """
    Try to infer if a basis vector is aligned with ±x, ±y, or ±z.
    Returns "x", "y", "z", or None.
    """
    v = np.asarray(vec, dtype=float)
    if not np.all(np.isfinite(v)):
        return None
    norm = np.linalg.norm(v)
    if norm == 0.0:
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


def _axes_from_meta(
    attrs: h5py.AttributeManager,
    center_on_plane_center: bool,
    axis_units: Literal["cm", "mm"],
) -> tuple[
    tuple[float, float, float, float],     # extent (u_min, u_max, v_min, v_max) in plot units
    tuple[str, str],                       # axis_labels (xlabel, ylabel)
    tuple[float, float],                   # pixel_sizes (du_plot, dv_plot)
    tuple[float, float, float, float],     # uv_range_cm (u_min_cm, u_max_cm, v_min_cm, v_max_cm)
]:
    """
    Build the imshow extent, axis labels (with world-axis hints when possible),
    pixel sizes, and native (cm) ranges from /meta.
    """
    u_min_cm = float(attrs["grid.u_min"])
    u_max_cm = float(attrs["grid.u_max"])
    v_min_cm = float(attrs["grid.v_min"])
    v_max_cm = float(attrs["grid.v_max"])
    du_cm = float(attrs["grid.du"])
    dv_cm = float(attrs["grid.dv"])

    u0 = u_min_cm
    u1 = u_max_cm
    v0 = v_min_cm
    v1 = v_max_cm

    if center_on_plane_center:
        u_mid = 0.5 * (u0 + u1)
        v_mid = 0.5 * (v0 + v1)
        u0 -= u_mid
        u1 -= u_mid
        v0 -= v_mid
        v1 -= v_mid

    # Plot units (cm or mm)
    scale = 10.0 if axis_units == "mm" else 1.0
    unit_str = axis_units

    u0_plot = u0 * scale
    u1_plot = u1 * scale
    v0_plot = v0 * scale
    v1_plot = v1 * scale

    du_plot = du_cm * scale
    dv_plot = dv_cm * scale

    extent = (u0_plot, u1_plot, v0_plot, v1_plot)

    # Try to infer world-axis alignment for u, v
    u_world = None
    v_world = None
    if "plane.eu" in attrs:
        u_world = _basis_world_name(np.asarray(attrs["plane.eu"]))
    if "plane.ev" in attrs:
        v_world = _basis_world_name(np.asarray(attrs["plane.ev"]))

    if u_world and v_world:
        xlabel = f"u / {u_world} [{unit_str}]"
        ylabel = f"v / {v_world} [{unit_str}]"
    elif u_world:
        xlabel = f"u / {u_world} [{unit_str}]"
        ylabel = f"v [{unit_str}]"
    elif v_world:
        xlabel = f"u [{unit_str}]"
        ylabel = f"v / {v_world} [{unit_str}]"
    else:
        xlabel = f"u [{unit_str}]"
        ylabel = f"v [{unit_str}]"

    axis_labels = (xlabel, ylabel)
    uv_range_cm = (u_min_cm, u_max_cm, v_min_cm, v_max_cm)

    return extent, axis_labels, (du_plot, dv_plot), uv_range_cm



def render_summed_images(
    h5_path: str | Path,
    species: Sequence[str] = ("n", "g", "all"),
    filename_pattern: str = "{species}_{stem}.{ext}",
    center_on_plane_center: bool = True,
    flip_vertical: bool = True,
    axis_units: Literal["cm", "mm"] = "cm",
    cmap: str = "cividis",
    formats: Sequence[str] = ("png",),
    projections: bool = False,
    roi_u_min_cm: float | None = None,
    roi_u_max_cm: float | None = None,
    roi_v_min_cm: float | None = None,
    roi_v_max_cm: float | None = None,
) -> list[Path]:
    """
    Render `/images/summed/*` datasets from an ng-imager HDF5 file to image files.

    When `projections=True`, each figure shows:
      - the main 2D image (u vs v),
      - a 1D projection along u **above** the image,
      - a 1D projection along v **to the left** of the image,
      - an optional ROI rectangle (if roi_*_cm are provided),
      - an annotation of the number of cones contributing to that species.
    """

    h5_path = Path(h5_path)
    stem = h5_path.stem

    # Normalize species and formats
    species_list: list[str] = []
    for s in species:
        s = str(s).lower()
        if s in ("n", "g", "all") and s not in species_list:
            species_list.append(s)

    fmt_list: list[str] = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        if fmt and fmt not in fmt_list:
            fmt_list.append(fmt)
    if not fmt_list:
        fmt_list = ["png"]

    # ROI rectangle in cm
    roi_cm: tuple[float, float, float, float] | None = None
    if (
        roi_u_min_cm is not None
        and roi_u_max_cm is not None
        and roi_v_min_cm is not None
        and roi_v_max_cm is not None
    ):
        roi_cm = (
            float(roi_u_min_cm),
            float(roi_u_max_cm),
            float(roi_v_min_cm),
            float(roi_v_max_cm),
        )

    out_paths: list[Path] = []

    with h5py.File(h5_path, "r") as f:
        if "images" not in f or "summed" not in f["images"]:
            return []

        summed_grp = f["images"]["summed"]
        meta_attrs = f["meta"].attrs

        has_n = "n" in summed_grp
        has_g = "g" in summed_grp

        for sp in species_list:
            # Skip "all" if we don't have both species present.
            if sp == "all" and not (has_n and has_g):
                continue
            if sp not in summed_grp:
                continue

            img = np.array(summed_grp[sp], dtype=np.float32)
            nv, nu = img.shape

            # Axes / extent / pixel sizes from metadata
            extent, axis_labels, (du_plot, dv_plot), (
                u_min_cm,
                u_max_cm,
                v_min_cm,
                v_max_cm,
            ) = _axes_from_meta(meta_attrs, center_on_plane_center, axis_units)

            # Pixel centers in cm
            du_cm = float(meta_attrs["grid.du"])
            dv_cm = float(meta_attrs["grid.dv"])
            u_centers_cm = u_min_cm + (np.arange(nu) + 0.5) * du_cm
            v_centers_cm = v_min_cm + (np.arange(nv) + 0.5) * dv_cm

            # Global projections
            proj_u = img.sum(axis=0)  # shape (nu,)
            proj_v = img.sum(axis=1)  # shape (nv,)

            proj_u_roi = None
            proj_v_roi = None

            if roi_cm is not None:
                ru0, ru1, rv0, rv1 = roi_cm
                u_mask = (u_centers_cm >= ru0) & (u_centers_cm <= ru1)
                v_mask = (v_centers_cm >= rv0) & (v_centers_cm <= rv1)

                proj_u_roi = np.zeros_like(proj_u)
                proj_v_roi = np.zeros_like(proj_v)

                if np.any(u_mask) and np.any(v_mask):
                    block = img[np.ix_(v_mask, u_mask)]
                    proj_u_roi[u_mask] = block.sum(axis=0)
                    proj_v_roi[v_mask] = block.sum(axis=1)

            # Convert centers to plot units (cm or mm) and apply centering
            u_mid_cm = 0.5 * (u_min_cm + u_max_cm)
            v_mid_cm = 0.5 * (v_min_cm + v_max_cm)
            unit_scale = 10.0 if axis_units == "mm" else 1.0

            if center_on_plane_center:
                u_centers_plot = (u_centers_cm - u_mid_cm) * unit_scale
                v_centers_plot = (v_centers_cm - v_mid_cm) * unit_scale
            else:
                u_centers_plot = u_centers_cm * unit_scale
                v_centers_plot = v_centers_cm * unit_scale

            # ----------------- Figure layout -----------------
            if projections and (nu > 1 or nv > 1):
                # Layout:
                #   row 0: [ empty | top u-proj | empty ]
                #   row 1: [ left v-proj | image | colorbar ]
                fig = plt.figure(figsize=(8.0, 8.0))
                gs = GridSpec(
                    2,
                    3,
                    width_ratios=[1.3, 4.0, 0.4],
                    height_ratios=[1.3, 4.0],
                    wspace=0.08,
                    hspace=0.08,
                    figure=fig,
                )

                ax_top = fig.add_subplot(gs[0, 1])
                ax_img = fig.add_subplot(gs[1, 1], sharex=ax_top)
                ax_left = fig.add_subplot(gs[1, 0], sharey=ax_img)
                ax_cb = fig.add_subplot(gs[1, 2])

                # Hide redundant tick labels
                plt.setp(ax_top.get_xticklabels(), visible=False)
                # We'll show v-ticks and the v-label on the LEFT projection,
                # and hide y tick labels on the main image to avoid overlap.
                plt.setp(ax_img.get_yticklabels(), visible=False)

            else:
                # Simple image + colorbar layout (no projections)
                fig = plt.figure(figsize=(6.0, 5.0))
                gs = GridSpec(1, 2, width_ratios=[12.0, 0.6], figure=fig)
                ax_img = fig.add_subplot(gs[0, 0])
                ax_cb = fig.add_subplot(gs[0, 1])
                ax_top = None
                ax_left = None

            # ----------------- Main image -----------------
            im = ax_img.imshow(
                img,
                origin="lower",
                extent=extent,
                cmap=cmap,
                aspect="auto",
            )
            if flip_vertical:
                ax_img.invert_yaxis()

            # Lock limits to image bounds (no white gaps)
            ax_img.set_xlim(extent[0], extent[1])
            ax_img.set_ylim(extent[2], extent[3])

            ax_img.set_xlabel(axis_labels[0])
            # In projections mode, the left panel owns the v-axis label.
            if projections and ax_left is not None:
                ax_img.set_ylabel("")
            else:
                ax_img.set_ylabel(axis_labels[1])

            # Colorbar
            cbar = fig.colorbar(im, cax=ax_cb)
            px_unit = axis_units
            cbar.set_label(
                f"counts per {du_plot:g} × {dv_plot:g} {px_unit}² pixel"
            )

            # ROI rectangle
            if projections and roi_cm is not None:
                ru0, ru1, rv0, rv1 = roi_cm
                if center_on_plane_center:
                    ru0_plot = (ru0 - u_mid_cm) * unit_scale
                    ru1_plot = (ru1 - u_mid_cm) * unit_scale
                    rv0_plot = (rv0 - v_mid_cm) * unit_scale
                    rv1_plot = (rv1 - v_mid_cm) * unit_scale
                else:
                    ru0_plot = ru0 * unit_scale
                    ru1_plot = ru1 * unit_scale
                    rv0_plot = rv0 * unit_scale
                    rv1_plot = rv1 * unit_scale

                rect = Rectangle(
                    (ru0_plot, rv0_plot),
                    ru1_plot - ru0_plot,
                    rv1_plot - rv0_plot,
                    fill=False,
                    linestyle="--",
                    linewidth=1.3,
                    edgecolor="white",
                    alpha=0.9,
                )
                ax_img.add_patch(rect)

            # Cone-count annotation (above image, inside its axes coordinates)
            n_cones = _count_cones_for_species(f, sp)
            if n_cones is not None:
                if sp == "all":
                    txt = f"{n_cones} n+g event cones"
                elif sp == "n":
                    txt = f"{n_cones} neutron event cones"
                elif sp == "g":
                    txt = f"{n_cones} gamma event cones"
                else:
                    txt = f"{n_cones} event cones"
                ax_img.text(
                    0.5,
                    1.01,
                    txt,
                    transform=ax_img.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            # ----------------- Projections -----------------
            if projections and ax_top is not None and ax_left is not None:
                # -------------------------
                # U-projection (top panel)
                # -------------------------
                # Primary axis: "all"
                line_all_u, = ax_top.plot(
                    u_centers_plot,
                    proj_u,
                    color="C0",
                    label="all",
                )
                ax_top.set_ylabel("Σ counts (over v)")
                ax_top.grid(alpha=0.2)

                if proj_u_roi is not None:
                    # Secondary y-axis: scaled to ROI only
                    ax_top2 = ax_top.twinx()
                    line_roi_u, = ax_top2.plot(
                        u_centers_plot,
                        proj_u_roi,
                        linestyle="--",
                        color="C1",
                        label="ROI",
                    )

                    # Tight y-limits based on nonzero ROI values
                    nz = proj_u_roi[proj_u_roi > 0]
                    if nz.size > 0:
                        ymin, ymax = float(nz.min()), float(nz.max())
                        pad = 0.05 * (ymax - ymin) if ymax > ymin else max(ymax * 0.1, 1.0)
                        ax_top2.set_ylim(ymin - pad, ymax + pad)
                    else:
                        ax_top2.set_ylim(0.0, 1.0)

                    # Inside ticks on the right, colored to match ROI curve
                    ax_top2.yaxis.set_label_position("right")
                    ax_top2.yaxis.tick_right()
                    ax_top2.tick_params(
                        axis="y",
                        direction="out",
                        pad=3,  # small positive: just inside the frame
                        colors="C1",
                        labelcolor="C1",
                    )
                    ax_top2.set_ylabel("")

                    # Legend showing both curves, built from primary & secondary handles
                    ax_top.legend(
                        [line_all_u, line_roi_u],
                        ["all", "ROI"],
                        loc="upper right",
                        fontsize=8,
                    )
                else:
                    ax_top.legend(loc="upper right", fontsize=8)

                # -------------------------
                # V-projection (left panel)
                # -------------------------
                # Primary axis: "all"
                line_all_v, = ax_left.plot(
                    proj_v,
                    v_centers_plot,
                    color="C0",
                    label="all",
                )

                ax_left.invert_xaxis()
                ax_left.set_xlabel("Σ counts (over u)")
                ax_left.set_ylabel(axis_labels[1])  # v-label lives here
                ax_left.grid(alpha=0.2)

                if proj_v_roi is not None:
                    # Secondary x-axis (top): scaled to ROI only
                    ax_left2 = ax_left.twiny()
                    line_roi_v, = ax_left2.plot(
                        proj_v_roi,
                        v_centers_plot,
                        linestyle="--",
                        color="C1",
                        label="ROI",
                    )

                    # Tight x-limits from nonzero ROI
                    nz = proj_v_roi[proj_v_roi > 0]
                    if nz.size > 0:
                        xmin, xmax = float(nz.min()), float(nz.max())
                        pad = 0.05 * (xmax - xmin) if xmax > xmin else max(xmax * 0.1, 1.0)
                        ax_left2.set_xlim(xmin - pad, xmax + pad)
                    else:
                        ax_left2.set_xlim(0.0, 1.0)

                    # Keep "zero on the right" for secondary axis as well
                    ax_left2.invert_xaxis()

                    # Inside ticks at the top edge, in ROI color
                    ax_left2.xaxis.set_label_position("top")
                    ax_left2.xaxis.tick_top()
                    ax_left2.tick_params(
                        axis="x",
                        direction="out",
                        pad=3,  # small positive: just inside
                        colors="C1",
                        labelcolor="C1",
                    )
                    ax_left2.set_xlabel("")
                    ax_left2.set_ylabel("")

            # Suptitle for the whole figure
            species_label = {
                "n": "n",
                "g": "g",
                "all": "n+g",
            }.get(sp, sp)
            fig.suptitle(f"{stem} : {species_label} cones", y=0.98)

            # Leave room for suptitle
            fig.subplots_adjust(top=0.93)

            # Save in all requested formats
            for fmt in fmt_list:
                out_name = filename_pattern.format(
                    stem=stem,
                    species=sp,
                    ext=fmt,
                )
                out_path = h5_path.with_name(out_name)
                dpi = 150 if fmt in ("png", "jpg", "jpeg") else None
                fig.savefig(out_path, dpi=dpi)
                out_paths.append(out_path)

            plt.close(fig)

    return out_paths


