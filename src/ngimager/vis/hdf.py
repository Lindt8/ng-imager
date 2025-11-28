from __future__ import annotations

from pathlib import Path
from typing import Sequence, Literal, Optional, Mapping, Any

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Centralized styles for projections and metric overlays
_PROJ_STYLE_ALL = {"color": "C0", "linestyle": "-"}
_PROJ_STYLE_ROI = {"color": "C1", "linestyle": "--"}
_METRIC_STYLE_PEAK = {"color": "C2", "linestyle": "-."}
_METRIC_STYLE_EDGE = {"color": "C3", "linestyle": ":"}

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


def _load_projection_metrics(
    summed_grp: h5py.Group,
    species: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Load projection metrics for a given species into a nested dict structure.

    The returned structure is:

        {
            "u": {
                "all": {metric_name: value, ...},   # if available
                "roi": {metric_name: value, ...},   # if available
            },
            "v": {
                "all": {...},
                "roi": {...},
            },
        }

    This assumes the current ng-imager layout:

        /images/summed/projections/<species>/metrics/u
        /images/summed/projections/<species>/metrics/u_roi
        /images/summed/projections/<species>/metrics/v
        /images/summed/projections/<species>/metrics/v_roi

    Missing groups simply do not appear in the nested dict.
    """
    proj_root = summed_grp.get("projections")
    if proj_root is None or not isinstance(proj_root, h5py.Group):
        return {}

    sp_root = proj_root.get(species)
    if sp_root is None or not isinstance(sp_root, h5py.Group):
        return {}

    g = sp_root.get("metrics")
    if not isinstance(g, h5py.Group):
        return {}

    metrics_root: h5py.Group = g

    def _read_group_datasets(grp: h5py.Group) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, obj in grp.items():
            if isinstance(obj, h5py.Dataset):
                value = obj[()]
                # Convert 0-d numpy arrays to Python scalars where possible
                try:
                    value = value.item()
                except AttributeError:
                    pass
                out[name] = value
        return out

    out: dict[str, dict[str, dict[str, Any]]] = {"u": {}, "v": {}}

    for axis_name in ("u", "v"):
        axis_metrics: dict[str, dict[str, Any]] = {}

        all_grp = metrics_root.get(axis_name)
        roi_grp = metrics_root.get(f"{axis_name}_roi")

        if isinstance(all_grp, h5py.Group):
            axis_metrics["all"] = _read_group_datasets(all_grp)
        if isinstance(roi_grp, h5py.Group):
            axis_metrics["roi"] = _read_group_datasets(roi_grp)

        if axis_metrics:
            out[axis_name] = axis_metrics

    return out


def _resolve_projection_metrics(
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metrics_source: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Filter the raw metrics dict according to the requested metrics_source.

    Parameters
    ----------
    metrics :
        Output of _load_projection_metrics(...), keyed as metrics[axis][source].
        axis   : "u" or "v"
        source : "all" or "roi"
    metrics_source :
        One of "auto", "all", "roi", "both" (case-insensitive).

    Returns
    -------
    dict
        Same structure as the input, but with axes/sources possibly filtered.

        For example, with metrics_source="all":

            {
                "u": {"all": {...}},
                "v": {"all": {...}},
            }
    """
    msrc = (metrics_source or "auto").lower()
    result: dict[str, dict[str, dict[str, Any]]] = {"u": {}, "v": {}}

    def _resolve_axis(axis_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        if not axis_metrics:
            return {}

        has_all = "all" in axis_metrics and axis_metrics["all"]
        has_roi = "roi" in axis_metrics and axis_metrics["roi"]

        if msrc == "all":
            return {"all": axis_metrics["all"]} if has_all else {}
        if msrc == "roi":
            return {"roi": axis_metrics["roi"]} if has_roi else {}
        if msrc == "both":
            out: dict[str, dict[str, Any]] = {}
            if has_all:
                out["all"] = axis_metrics["all"]
            if has_roi:
                out["roi"] = axis_metrics["roi"]
            return out

        # "auto" (default) and unknown:
        # Prefer ROI when available, otherwise all.
        if has_roi:
            return {"roi": axis_metrics["roi"]}
        if has_all:
            return {"all": axis_metrics["all"]}
        return {}

    for axis_name in ("u", "v"):
        axis_metrics = metrics.get(axis_name, {})
        resolved = _resolve_axis(axis_metrics)
        if resolved:
            result[axis_name] = resolved

    return result





def _resolve_projection_plot_prefs(
    has_roi: bool,
    metrics_source: str,
    curve_mode: str,
) -> tuple[bool, bool, bool]:
    """
    Decide which curves to draw and which metrics to prefer.

    Parameters
    ----------
    has_roi :
        True when ROI-limited projections/metrics are available.
    metrics_source :
        One of "auto", "all", "roi", "both" (case-insensitive).
    curve_mode :
        One of "all+roi", "all_only", "roi_only" (case-insensitive).

    Returns
    -------
    draw_all, draw_roi, prefer_roi_metrics
        draw_all :
            Whether to draw the all-pixels projection curve.
        draw_roi :
            Whether to draw the ROI-limited projection curve.
        prefer_roi_metrics :
            Whether metrics helpers should prefer ROI metrics when both
            ROI and all-pixels metrics exist.
    """
    msrc = (metrics_source or "auto").lower()
    cmode = (curve_mode or "all+roi").lower()

    # -----------------------------
    # Curve visibility (all / ROI)
    # -----------------------------
    if not has_roi:
        # No ROI projections: always fall back to "all" only.
        draw_all = True
        draw_roi = False
    else:
        if cmode == "all_only":
            draw_all, draw_roi = True, False
        elif cmode == "roi_only":
            draw_all, draw_roi = False, True
        else:  # "all+roi" or anything unexpected
            draw_all, draw_roi = True, True

    # -----------------------------
    # Which metrics to prefer?
    # -----------------------------
    if not has_roi:
        # No ROI metrics: always use "all" if present.
        prefer_roi_metrics = False
    else:
        if msrc == "all":
            prefer_roi_metrics = False
        elif msrc == "roi":
            prefer_roi_metrics = True
        else:
            # "auto" / "both" / unknown:
            # tie metrics to whichever curve is emphasized.
            if draw_roi and not draw_all:
                prefer_roi_metrics = True
            elif draw_all and not draw_roi:
                prefer_roi_metrics = False
            else:
                # Both curves visible → default to ROI metrics
                prefer_roi_metrics = True

    return draw_all, draw_roi, prefer_roi_metrics



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
    plot_label: str | None = None,
    metrics_source: str = "auto",
    curve_mode: str = "all+roi",
    annotate_summary: str = "compact",
    show_metrics_panel: bool = False,
    show_peak_markers: bool = True,
    show_edge_markers: bool = True,
    show_centroid_2d: bool = False,
) -> list[Path]:
    """
    Render `/images/summed/*` datasets from an ng-imager HDF5 file to image files.

    When `projections=True`, each figure shows:
      - the main 2D image (u vs v),
      - a 1D projection along u **above** the image,
      - a 1D projection along v **to the left** of the image,
      - an optional ROI rectangle (if roi_*_cm are provided),
      - an annotation of the number of cones contributing to that species.
      - and (when available) a run-level plot label drawn from [run].plot_label
        or from the `plot_label` argument.
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

        # Default plot label: prefer explicit argument, then /meta attribute.
        stored_plot_label: str | None = None
        if "run_plot_label" in meta_attrs:
            try:
                stored_plot_label = str(meta_attrs["run_plot_label"])
            except Exception:
                stored_plot_label = None

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

            # Flags describing ROI / curves / metrics
            has_roi_projections = (proj_u_roi is not None) and (proj_v_roi is not None)
            draw_all_curve = True
            draw_roi_curve = False
            prefer_roi_metrics = False

            if projections:
                draw_all_curve, draw_roi_curve, prefer_roi_metrics = _resolve_projection_plot_prefs(
                    has_roi_projections,
                    metrics_source,
                    curve_mode,
                )

            # ------------------------------------------------------------------
            # Optional: read precomputed projection metrics (positions in cm)
            # ------------------------------------------------------------------
            peak_u_cm = peak_v_cm = None
            edge_low_u_cm = edge_high_u_cm = None
            edge_low_v_cm = edge_high_v_cm = None
            
            def _format_axis_summary(
                axis_label: str,
                metrics: Mapping[str, Any],
                mode: str,
            ) -> Optional[str]:
                """
                Build a small annotation string for one axis from its metrics.

                axis_label: "u" or "v"
                mode      : "off" | "compact" | "full"
                """
                mode = (mode or "off").lower()
                if mode == "off":
                    return None

                def _get(name: str) -> Optional[float]:
                    val = metrics.get(name)
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        return None
                    if not np.isfinite(val):
                        return None
                    return val

                peak = _get("peak_pos_cm")
                width = _get("edge_width_cm")
                mean = _get("mean_cm")
                std = _get("std_cm")
                edge_low = _get("edge_low_cm")
                edge_high = _get("edge_high_cm")

                # Convert from cm to the chosen axis_units (cm or mm)
                scale = 10.0 if axis_units == "mm" else 1.0
                unit_label = axis_units

                def _conv(x: Optional[float]) -> Optional[float]:
                    if x is None:
                        return None
                    return x * scale

                peak = _conv(peak)
                width = _conv(width)
                mean = _conv(mean)
                std = _conv(std)
                edge_low = _conv(edge_low)
                edge_high = _conv(edge_high)

                parts: list[str] = []

                if mode == "compact":
                    # Prefer peak + width when available
                    if peak is not None:
                        parts.append(f"peak={peak:.2f} {unit_label}")
                    if width is not None:
                        parts.append(f"width={width:.2f} {unit_label}")
                    if not parts and mean is not None and std is not None:
                        parts.append(f"mean={mean:.2f}±{std:.2f} {unit_label}")
                else:  # "full"
                    if peak is not None:
                        parts.append(f"peak={peak:.2f} {unit_label}")
                    if mean is not None:
                        parts.append(f"mean={mean:.2f} {unit_label}")
                    if std is not None:
                        parts.append(f"std={std:.2f} {unit_label}")
                    if edge_low is not None and edge_high is not None:
                        parts.append(
                            f"edges=[{edge_low:.2f}, {edge_high:.2f}] {unit_label}"
                        )
                    elif width is not None:
                        parts.append(f"width={width:.2f} {unit_label}")

                if not parts:
                    return None

                lines = [f"{axis_label}:"]
                for p in parts:
                    lines.append(f"  {p}")
                return "\n".join(lines)

            def _render_metrics_panel(
                    ax_panel,
                    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
            ) -> None:
                """
                Render a table of metrics into ax_panel.

                metrics[axis][source] -> dict of scalar values
                axis   in {"u", "v"}
                source in {"all", "roi"}
                """
                # Collect rows in a stable order
                rows: list[tuple[str, str, Mapping[str, Any]]] = []
                for axis_name in ("u", "v"):
                    axis_metrics = metrics.get(axis_name, {})
                    for src_name in ("all", "roi"):
                        m = axis_metrics.get(src_name)
                        if m:
                            rows.append((axis_name, src_name, m))

                ax_panel.set_axis_off()

                if not rows:
                    ax_panel.text(
                        0.0,
                        1.0,
                        "No projection metrics available",
                        transform=ax_panel.transAxes,
                        ha="left",
                        va="top",
                        fontsize=7,
                    )
                    return

                # Helper to fetch and convert from cm to chosen axis_units
                scale = 10.0 if axis_units == "mm" else 1.0
                unit_label = axis_units

                def _get(m: Mapping[str, Any], name: str) -> Optional[float]:
                    val = m.get(name)
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        return None
                    if not np.isfinite(val):
                        return None
                    return val * scale

                # Build table rows:
                # [axis/source, peak, mean, median, std, low, high, width]
                table_rows: list[list[str]] = []
                for axis_name, src_name, m in rows:
                    peak = _get(m, "peak_pos_cm")
                    mean = _get(m, "mean_cm")
                    median = _get(m, "median_cm")
                    std = _get(m, "std_cm")
                    edge_low = _get(m, "edge_low_cm")
                    edge_high = _get(m, "edge_high_cm")
                    width = _get(m, "edge_width_cm")

                    def _fmt(x: Optional[float]) -> str:
                        return f"{x:.2f}" if x is not None else ""

                    row = [
                        f"{axis_name}/{src_name}",
                        _fmt(peak),
                        _fmt(mean),
                        _fmt(median),
                        _fmt(std),
                        _fmt(edge_low),
                        _fmt(edge_high),
                        _fmt(width),
                    ]
                    table_rows.append(row)

                col_labels = [
                    "axis/src",
                    f"peak [{unit_label}]",
                    f"mean [{unit_label}]",
                    f"median [{unit_label}]",
                    f"std [{unit_label}]",
                    f"low [{unit_label}]",
                    f"high [{unit_label}]",
                    f"width [{unit_label}]",
                ]

                table = ax_panel.table(
                    cellText=table_rows,
                    colLabels=col_labels,
                    loc="upper center",
                    cellLoc="center",
                    bbox=[0.0, -0.15, 1.0, 0.9],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1.0, 1.2)

            metrics_sel: dict[str, dict[str, dict[str, Any]]] = {}

            if projections:
                # Load all available metrics for this species
                metrics_raw = _load_projection_metrics(summed_grp, sp)
                metrics_sel = _resolve_projection_metrics(metrics_raw, metrics_source)

                def _pick_axis_metrics(
                    axis_name: str,
                ) -> tuple[dict[str, Any] | None, Optional[str]]:
                    """
                    Return (metrics, source) for the given axis.

                    source ∈ {"all", "roi"} or None if no metrics available.
                    """
                    axis_metrics = metrics_sel.get(axis_name, {})
                    if not axis_metrics:
                        return None, None

                    has_all = "all" in axis_metrics and axis_metrics["all"]
                    has_roi = "roi" in axis_metrics and axis_metrics["roi"]

                    msrc = (metrics_source or "auto").lower()
                    if msrc == "all" and has_all:
                        return axis_metrics["all"], "all"
                    if msrc == "roi" and has_roi:
                        return axis_metrics["roi"], "roi"
                    if msrc == "both":
                        # For overlays, prefer ROI when both exist
                        if has_roi:
                            return axis_metrics["roi"], "roi"
                        if has_all:
                            return axis_metrics["all"], "all"
                        return None, None

                    # "auto" or unknown: prefer ROI, else all
                    if has_roi:
                        return axis_metrics["roi"], "roi"
                    if has_all:
                        return axis_metrics["all"], "all"
                    return None, None

                u_metrics, u_metrics_source = _pick_axis_metrics("u")
                v_metrics, v_metrics_source = _pick_axis_metrics("v")


                if u_metrics is not None:
                    peak_u_cm = u_metrics.get("peak_pos_cm")
                    edge_low_u_cm = u_metrics.get("edge_low_cm")
                    edge_high_u_cm = u_metrics.get("edge_high_cm")

                if v_metrics is not None:
                    peak_v_cm = v_metrics.get("peak_pos_cm")
                    edge_low_v_cm = v_metrics.get("edge_low_cm")
                    edge_high_v_cm = v_metrics.get("edge_high_cm")
                
                # -------------------------
                # Centroid metrics (cm)
                # Prefer ROI centroid if available, else ALL
                # -------------------------
                u_centroid_cm = None
                v_centroid_cm = None

                # ROI has priority
                if u_metrics_source == "roi":
                    u_centroid_cm = u_metrics.get("mean_cm")
                elif u_metrics_source == "all":
                    u_centroid_cm = u_metrics.get("mean_cm")
                else:
                    # "auto" case handled via u_metrics_source
                    if u_metrics is not None:
                        u_centroid_cm = u_metrics.get("mean_cm")

                if v_metrics_source == "roi":
                    v_centroid_cm = v_metrics.get("mean_cm")
                elif v_metrics_source == "all":
                    v_centroid_cm = v_metrics.get("mean_cm")
                else:
                    if v_metrics is not None:
                        v_centroid_cm = v_metrics.get("mean_cm")

                
                # Edge fractions (0–1) from metrics/u and metrics/v attributes
                edge_fracs: dict[str, tuple[Optional[float], Optional[float]]] = {}

                try:
                    proj_root = summed_grp.get("projections")
                    if isinstance(proj_root, h5py.Group):
                        sp_root = proj_root.get(sp)
                    else:
                        sp_root = None
                    if isinstance(sp_root, h5py.Group):
                        metrics_root = sp_root.get("metrics")
                    else:
                        metrics_root = None

                    if isinstance(metrics_root, h5py.Group):
                        for axis_name in ("u", "v"):
                            axis_grp = metrics_root.get(axis_name)
                            if not isinstance(axis_grp, h5py.Group):
                                continue
                            low = axis_grp.attrs.get("edge_low_frac", None)
                            high = axis_grp.attrs.get("edge_high_frac", None)
                            try:
                                low_f = float(low) if low is not None else None
                            except (TypeError, ValueError):
                                low_f = None
                            try:
                                high_f = float(high) if high is not None else None
                            except (TypeError, ValueError):
                                high_f = None
                            if low_f is not None or high_f is not None:
                                edge_fracs[axis_name] = (low_f, high_f)
                except Exception:
                    edge_fracs = {}



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
            
            # Convert metric positions (if any) into plot units
            peak_u_plot = edge_low_u_plot = edge_high_u_plot = None
            peak_v_plot = edge_low_v_plot = edge_high_v_plot = None
            u_centroid_plot = None
            v_centroid_plot = None

            def _cm_to_plot_u(x_cm: Optional[float]) -> Optional[float]:
                if x_cm is None:
                    return None
                if center_on_plane_center:
                    return (x_cm - u_mid_cm) * unit_scale
                return x_cm * unit_scale

            def _cm_to_plot_v(y_cm: Optional[float]) -> Optional[float]:
                if y_cm is None:
                    return None
                if center_on_plane_center:
                    return (y_cm - v_mid_cm) * unit_scale
                return y_cm * unit_scale

            if peak_u_cm is not None:
                peak_u_plot = _cm_to_plot_u(peak_u_cm)
            if edge_low_u_cm is not None:
                edge_low_u_plot = _cm_to_plot_u(edge_low_u_cm)
            if edge_high_u_cm is not None:
                edge_high_u_plot = _cm_to_plot_u(edge_high_u_cm)

            if peak_v_cm is not None:
                peak_v_plot = _cm_to_plot_v(peak_v_cm)
            if edge_low_v_cm is not None:
                edge_low_v_plot = _cm_to_plot_v(edge_low_v_cm)
            if edge_high_v_cm is not None:
                edge_high_v_plot = _cm_to_plot_v(edge_high_v_cm)
                
            # Convert centroids
            if u_centroid_cm is not None:
                u_centroid_plot = _cm_to_plot_u(u_centroid_cm)
            if v_centroid_cm is not None:
                v_centroid_plot = _cm_to_plot_v(v_centroid_cm)


            # ----------------- Figure layout -----------------
            if projections and (nu > 1 or nv > 1):
                # Decide whether a metrics panel makes sense for this species
                has_any_metrics = bool(
                    metrics_sel
                    and any(metrics_sel.get(ax) for ax in ("u", "v"))
                )
                want_metrics_panel = show_metrics_panel and has_any_metrics

                if want_metrics_panel:
                    # Layout with metrics panel as a wide bottom row:
                    #   row 0: [ legend | top u-proj | empty ]
                    #   row 1: [ left v-proj | image | colorbar ]
                    #   row 2: [       metrics table (spans all 3 columns)      ]
                    fig = plt.figure(figsize=(8.0, 8.5))
                    gs = GridSpec(
                        3,
                        3,
                        width_ratios=[1.3, 4.0, 0.4],
                        height_ratios=[1.3, 4.0, 0.9],  # shrink bottom row
                        wspace=0.08,
                        hspace=0.12,  # slightly reduced vertical spacing
                        figure=fig,
                    )

                    ax_legend = fig.add_subplot(gs[0, 0])
                    ax_top = fig.add_subplot(gs[0, 1])
                    ax_img = fig.add_subplot(gs[1, 1], sharex=ax_top)
                    ax_left = fig.add_subplot(gs[1, 0], sharey=ax_img)
                    ax_cb = fig.add_subplot(gs[1, 2])
                    ax_metrics = fig.add_subplot(gs[2, :])
                    ax_metrics.set_axis_off()
                else:
                    # Layout (no metrics panel):
                    #   row 0: [ legend | top u-proj | empty ]
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

                    ax_legend = fig.add_subplot(gs[0, 0])
                    ax_top = fig.add_subplot(gs[0, 1])
                    ax_img = fig.add_subplot(gs[1, 1], sharex=ax_top)
                    ax_left = fig.add_subplot(gs[1, 0], sharey=ax_img)
                    ax_cb = fig.add_subplot(gs[1, 2])
                    ax_metrics = None


                # Global legend in the legend panel
                ax_legend.axis("off")
                legend_handles = [
                    Line2D([0], [0], label="all", **_PROJ_STYLE_ALL),
                    Line2D([0], [0], label="ROI", **_PROJ_STYLE_ROI),
                    Line2D([0], [0], label="peak", **_METRIC_STYLE_PEAK),
                    Line2D([0], [0], label="edges", **_METRIC_STYLE_EDGE),
                ]
                ax_legend.legend(
                    handles=legend_handles,
                    loc="center left",
                    bbox_to_anchor=(-0.1, 0.5),
                    borderaxespad=0.0,
                    frameon=False,
                    fontsize=8,
                )

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
            
            # 2D centroid crosshair overlay
            if projections and show_centroid_2d:
                if (u_centroid_plot is not None) and (v_centroid_plot is not None):

                    # Short crosshair arms (5% of image span)
                    u_span = extent[1] - extent[0]
                    v_span = extent[3] - extent[2]
                    hu = 0.05 * u_span
                    hv = 0.05 * v_span

                    # Horizontal line
                    ax_img.hlines(
                        v_centroid_plot,
                        u_centroid_plot - hu,
                        u_centroid_plot + hu,
                        colors="white",
                        linewidth=1.2,
                        alpha=0.9,
                    )
                    # Vertical line
                    ax_img.vlines(
                        u_centroid_plot,
                        v_centroid_plot - hv,
                        v_centroid_plot + hv,
                        colors="white",
                        linewidth=1.2,
                        alpha=0.9,
                    )

                    # Annotation (top-left inside image)
                    ax_img.text(
                        0.02, 0.98,
                        f"centroid = ({u_centroid_plot:.2f}, {v_centroid_plot:.2f}) {axis_units}",
                        transform=ax_img.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="white",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="black",
                            edgecolor="none",
                            alpha=0.4,
                        ),
                    )

            
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
                ax_top.set_ylabel("Σ counts (over v)")
                ax_top.grid(alpha=0.2)

                line_all_u = None
                line_roi_u = None
                ax_top2 = None

                # Primary axis: draw "all" if enabled
                if draw_all_curve:
                    (line_all_u,) = ax_top.plot(
                        u_centers_plot,
                        proj_u,
                        label="all",
                        **_PROJ_STYLE_ALL,
                    )

                # ROI curve (if available and enabled)
                if draw_roi_curve and proj_u_roi is not None:
                    if draw_all_curve:
                        # Secondary y-axis: scaled to ROI only
                        ax_top2 = ax_top.twinx()
                        (line_roi_u,) = ax_top2.plot(
                            u_centers_plot,
                            proj_u_roi,
                            label="ROI",
                            **_PROJ_STYLE_ROI,
                        )

                        # Tight y-limits based on nonzero ROI values
                        nz = proj_u_roi[proj_u_roi > 0]
                        if nz.size > 0:
                            ymin, ymax = float(nz.min()), float(nz.max())
                            pad = 0.05 * (ymax - ymin) if ymax > ymin else max(
                                ymax * 0.1,
                                1.0,
                            )
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
                            colors=_PROJ_STYLE_ROI["color"],
                            labelcolor=_PROJ_STYLE_ROI["color"],
                        )
                        ax_top2.set_ylabel("")
                    else:
                        # ROI-only mode: single axis
                        (line_roi_u,) = ax_top.plot(
                            u_centers_plot,
                            proj_u_roi,
                            label="ROI",
                            **_PROJ_STYLE_ROI,
                        )

                        # Tight y-limits from ROI curve on the primary axis
                        nz = proj_u_roi[proj_u_roi > 0]
                        if nz.size > 0:
                            ymin, ymax = float(nz.min()), float(nz.max())
                            pad = 0.05 * (ymax - ymin) if ymax > ymin else max(
                                ymax * 0.1,
                                1.0,
                            )
                            ax_top.set_ylim(ymin - pad, ymax + pad)
                        else:
                            ax_top.set_ylim(0.0, 1.0)

                # Overlay u-axis metrics if available
                if peak_u_plot is not None and show_peak_markers:
                    ax_top.axvline(
                        peak_u_plot,
                        linewidth=1.0,
                        **_METRIC_STYLE_PEAK,
                    )
                if edge_low_u_plot is not None and show_edge_markers:
                    ax_top.axvline(
                        edge_low_u_plot,
                        linewidth=1.0,
                        **_METRIC_STYLE_EDGE,
                    )
                if edge_high_u_plot is not None and show_edge_markers:
                    ax_top.axvline(
                        edge_high_u_plot,
                        linewidth=1.0,
                        **_METRIC_STYLE_EDGE,
                    )
                    
                # Tiny annotations showing edge_low_frac / edge_high_frac (in %)
                u_low_frac, u_high_frac = edge_fracs.get("u", (None, None))
                if edge_low_u_plot is not None and u_low_frac is not None and show_edge_markers:
                    ax_top.text(
                        edge_low_u_plot,
                        1.01,
                        f"{u_low_frac * 100:.0f}%",
                        transform=ax_top.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=5,
                        color=_METRIC_STYLE_EDGE["color"],
                    )
                if edge_high_u_plot is not None and u_high_frac is not None and show_edge_markers:
                    ax_top.text(
                        edge_high_u_plot,
                        1.01,
                        f"{u_high_frac * 100:.0f}%",
                        transform=ax_top.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=5,
                        color=_METRIC_STYLE_EDGE["color"],
                    )


                # Optional numeric summary annotation for u
                mode_summary = (annotate_summary or "off").lower()
                if mode_summary != "off" and u_metrics is not None:
                    label_u = "u"
                    if u_metrics_source in ("all", "roi"):
                        label_u = f"u/{u_metrics_source}"
                    txt_u = _format_axis_summary(label_u, u_metrics, mode_summary)
                    if txt_u:
                        ax_top.text(
                            0.02,
                            0.98,
                            txt_u,
                            transform=ax_top.transAxes,
                            ha="left",
                            va="top",
                            fontsize=7,
                            color="black",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                facecolor="white",
                                edgecolor="none",
                                alpha=0.7,
                            ),
                        )

                # -------------------------
                # V-projection (left panel)
                # -------------------------
                ax_left.set_xlabel("Σ counts (over u)")
                ax_left.set_ylabel(axis_labels[1])  # v-label lives here
                ax_left.grid(alpha=0.2)

                line_all_v = None
                line_roi_v = None
                ax_left2 = None

                # Primary axis: draw "all" if enabled
                if draw_all_curve:
                    (line_all_v,) = ax_left.plot(
                        proj_v,
                        v_centers_plot,
                        label="all",
                        **_PROJ_STYLE_ALL,
                    )

                # ROI curve (if available and enabled)
                if draw_roi_curve and proj_v_roi is not None:
                    if draw_all_curve:
                        # Secondary x-axis (top): scaled to ROI only
                        ax_left2 = ax_left.twiny()
                        (line_roi_v,) = ax_left2.plot(
                            proj_v_roi,
                            v_centers_plot,
                            label="ROI",
                            **_PROJ_STYLE_ROI,
                        )

                        # Tight x-limits from nonzero ROI
                        nz = proj_v_roi[proj_v_roi > 0]
                        if nz.size > 0:
                            xmin, xmax = float(nz.min()), float(nz.max())
                            pad = 0.05 * (xmax - xmin) if xmax > xmin else max(
                                xmax * 0.1,
                                1.0,
                            )
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
                            colors=_PROJ_STYLE_ROI["color"],
                            labelcolor=_PROJ_STYLE_ROI["color"],
                        )
                        ax_left2.set_xlabel("")
                        ax_left2.set_ylabel("")
                    else:
                        # ROI-only mode: single axis
                        (line_roi_v,) = ax_left.plot(
                            proj_v_roi,
                            v_centers_plot,
                            label="ROI",
                            **_PROJ_STYLE_ROI,
                        )

                        # Tight x-limits from ROI curve on the primary axis
                        nz = proj_v_roi[proj_v_roi > 0]
                        if nz.size > 0:
                            xmin, xmax = float(nz.min()), float(nz.max())
                            pad = 0.05 * (xmax - xmin) if xmax > xmin else max(
                                xmax * 0.1,
                                1.0,
                            )
                            ax_left.set_xlim(xmin - pad, xmax + pad)
                        else:
                            ax_left.set_xlim(0.0, 1.0)

                # Overlay v-axis metrics if available
                if peak_v_plot is not None and show_peak_markers:
                    ax_left.axhline(
                        peak_v_plot,
                        linewidth=1.0,
                        **_METRIC_STYLE_PEAK,
                    )
                if edge_low_v_plot is not None and show_edge_markers:
                    ax_left.axhline(
                        edge_low_v_plot,
                        linewidth=1.0,
                        **_METRIC_STYLE_EDGE,
                    )
                if edge_high_v_plot is not None and show_edge_markers:
                    ax_left.axhline(
                        edge_high_v_plot,
                        linewidth=1.0,
                        **_METRIC_STYLE_EDGE,
                    )
                    
                # Tiny annotations showing edge_low_frac / edge_high_frac (in %)
                v_low_frac, v_high_frac = edge_fracs.get("v", (None, None))
                if edge_low_v_plot is not None and v_low_frac is not None and show_edge_markers:
                    ax_left.text(
                        1.002,  # just to the right of the axis frame
                        edge_low_v_plot,
                        f"{v_low_frac * 100:.0f}%",
                        transform=ax_left.get_yaxis_transform(),
                        ha="left",
                        va="center",
                        fontsize=5,
                        color=_METRIC_STYLE_EDGE["color"],
                    )
                if edge_high_v_plot is not None and v_high_frac is not None and show_edge_markers:
                    ax_left.text(
                        1.002,
                        edge_high_v_plot,
                        f"{v_high_frac * 100:.0f}%",
                        transform=ax_left.get_yaxis_transform(),
                        ha="left",
                        va="center",
                        fontsize=5,
                        color=_METRIC_STYLE_EDGE["color"],
                    )

                # Optional numeric summary annotation for v
                mode_summary = (annotate_summary or "off").lower()
                if mode_summary != "off" and v_metrics is not None:
                    label_v = "v"
                    if v_metrics_source in ("all", "roi"):
                        label_v = f"v/{v_metrics_source}"
                    txt_v = _format_axis_summary(label_v, v_metrics, mode_summary)
                    if txt_v:
                        ax_left.text(
                            0.02,
                            0.98,
                            txt_v,
                            transform=ax_left.transAxes,
                            ha="left",
                            va="top",
                            fontsize=7,
                            color="black",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                facecolor="white",
                                edgecolor="none",
                                alpha=0.7,
                            ),
                        )

                
                # Keep "zero on the right" for the primary v-projection axis as well
                ax_left.invert_xaxis()

                # Render the metrics panel, if present
                if projections and (nu > 1 or nv > 1) and ax_metrics is not None:
                    _render_metrics_panel(ax_metrics, metrics_sel)

            
            # Suptitle for the whole figure
            species_label = {
                "n": "n",
                "g": "g",
                "all": "n+g",
            }.get(sp, sp)

            effective_label = plot_label or stored_plot_label
            if effective_label:
                title = f"{effective_label}\n{stem} : {species_label} cones"
            else:
                title = f"{stem} : {species_label} cones"
            
            fig.suptitle(title, y=0.98)

            # Leave room for suptitle. When the metrics panel is present, we
            # pull the whole subplot region down a bit (smaller `top`) and
            # also extend it further toward the bottom (smaller `bottom`) so
            # that:
            #   - the u-annotation and % labels have more headroom under the title
            #   - the metrics table drops below the u-axis label instead of
            #     overlapping it, filling the remaining white space.
            if not show_metrics_panel:
                fig.subplots_adjust(top=0.93)
            else:
                fig.subplots_adjust(top=0.92, bottom=0.05)

            # ----------------------------------------------------------------------
            # Add ngimager footer annotation (version + hyperlink)
            # ----------------------------------------------------------------------

            # HDF5 attributes already loaded earlier in the function
            software = f.attrs.get("software", "")
            docs_url = f.attrs.get("docs_url", "")

            # Text formatting (subtle, italic, gray)
            footer_font = {
                'color': '#666666',
                'weight': 'normal',
                'size': 8,
                'style': 'italic'
            }

            # Build footer string (only include URL/version if present)
            footer_parts = ["Figure generated by ng-imager"]
            if docs_url:
                footer_parts.append(f"· {docs_url}")
            if software:
                footer_parts.append(f"· {software}")

            footer_text = " ".join(footer_parts)

            # Place text at bottom-left of figure
            # url=docs_url makes this a working hyperlink in PDF output
            fig.text(
                0.005, 0.005,
                footer_text,
                fontdict=footer_font,
                ha='left',
                va='bottom',
                url=docs_url if docs_url else None
            )

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


