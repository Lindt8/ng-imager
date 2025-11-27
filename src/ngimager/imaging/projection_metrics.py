from __future__ import annotations

from typing import Dict, Optional
import numpy as np

from ngimager.config.schemas import ProjectionMetricsCfg, ProjectionAxisMetricsCfg


def _summary_metrics(
    coords_cm: np.ndarray,
    counts: np.ndarray,
    axis_cfg: ProjectionAxisMetricsCfg,
) -> Dict[str, float | bool]:
    total = float(np.sum(counts))
    out: Dict[str, float | bool] = {"total_counts": total}

    if (not axis_cfg.compute_summary) or total <= 0.0 or total < axis_cfg.min_counts:
        out["summary_ok"] = False
        out["mean_cm"] = float("nan")
        out["std_cm"] = float("nan")
        out["median_cm"] = float("nan")
        return out

    w = counts.astype(float)
    mean = float(np.sum(coords_cm * w) / total)
    var = float(np.sum(w * (coords_cm - mean) ** 2) / total)
    std = float(np.sqrt(max(var, 0.0)))

    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    median = float(np.interp(0.5, cdf, coords_cm))

    out["summary_ok"] = True
    out["mean_cm"] = mean
    out["std_cm"] = std
    out["median_cm"] = median
    return out


def _peak_metrics(
    coords_cm: np.ndarray,
    counts: np.ndarray,
    axis_cfg: ProjectionAxisMetricsCfg,
) -> Dict[str, float | bool]:
    out: Dict[str, float | bool] = {}

    total = float(np.sum(counts))
    if (not axis_cfg.compute_peak) or total <= 0.0 or total < axis_cfg.min_counts:
        out["peak_ok"] = False
        out["peak_pos_cm"] = float("nan")
        out["peak_value"] = float("nan")
        return out

    idx = int(np.argmax(counts))
    out["peak_ok"] = True
    out["peak_pos_cm"] = float(coords_cm[idx])
    out["peak_value"] = float(counts[idx])
    return out


def _edge_metrics(
    coords_cm: np.ndarray,
    counts: np.ndarray,
    axis_cfg: ProjectionAxisMetricsCfg,
) -> Dict[str, float | bool]:
    out: Dict[str, float | bool] = {}

    total = float(np.sum(counts))
    if (not axis_cfg.compute_edges) or total <= 0.0 or total < axis_cfg.min_counts:
        out["edges_ok"] = False
        out["edge_low_cm"] = float("nan")
        out["edge_high_cm"] = float("nan")
        out["edge_width_cm"] = float("nan")
        return out

    w = counts.astype(float)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]

    low = float(np.interp(axis_cfg.edge_low_frac, cdf, coords_cm))
    high = float(np.interp(axis_cfg.edge_high_frac, cdf, coords_cm))
    width = high - low

    out["edges_ok"] = True
    out["edge_low_cm"] = low
    out["edge_high_cm"] = high
    out["edge_width_cm"] = width
    return out


def _metrics_for_curve(
    coords_cm: np.ndarray,
    counts: Optional[np.ndarray],
    axis_cfg: ProjectionAxisMetricsCfg,
) -> Dict[str, float | bool]:
    """
    Compute metrics for one 1D curve (either "all" or "roi").
    Returns a dict of scalar metrics; some may be NaN if disabled or invalid.
    """
    if counts is None:
        # No curve defined (e.g. no ROI) → empty dict
        return {}

    counts = np.asarray(counts, dtype=float)
    out: Dict[str, float | bool] = {}

    # Always record total, even if below min_counts
    out["total_counts"] = float(np.sum(counts))

    # Summary, peak, edges (each may mark *_ok=False if below thresholds)
    out.update(_summary_metrics(coords_cm, counts, axis_cfg))
    out.update(_peak_metrics(coords_cm, counts, axis_cfg))
    out.update(_edge_metrics(coords_cm, counts, axis_cfg))

    return out


def compute_projection_metrics(
    u_centers_cm: np.ndarray,
    v_centers_cm: np.ndarray,
    proj_u: np.ndarray,
    proj_v: np.ndarray,
    proj_u_roi: Optional[np.ndarray],
    proj_v_roi: Optional[np.ndarray],
    cfg: Optional[ProjectionMetricsCfg],
) -> Dict[str, Dict[str, Dict[str, float | bool]]]:
    """
    Compute metrics for u/v projections (all + ROI) given configuration.

    Returns a nested dict:

        {
          "u": {
            "all": {...metrics...},
            "roi": {...metrics...}  # omitted if no ROI
          },
          "v": {
            "all": {...},
            "roi": {...}
          }
        }
    """
    if cfg is None or not cfg.enabled:
        return {}

    u_centers_cm = np.asarray(u_centers_cm, dtype=float)
    v_centers_cm = np.asarray(v_centers_cm, dtype=float)
    proj_u = np.asarray(proj_u, dtype=float)
    proj_v = np.asarray(proj_v, dtype=float)

    result: Dict[str, Dict[str, Dict[str, float | bool]]] = {
        "u": {},
        "v": {},
    }

    # u-axis
    u_all = _metrics_for_curve(u_centers_cm, proj_u, cfg.u)
    if u_all:
        result["u"]["all"] = u_all

    if proj_u_roi is not None:
        u_roi = _metrics_for_curve(u_centers_cm, proj_u_roi, cfg.u)
        if u_roi:
            result["u"]["roi"] = u_roi

    # v-axis
    v_all = _metrics_for_curve(v_centers_cm, proj_v, cfg.v)
    if v_all:
        result["v"]["all"] = v_all

    if proj_v_roi is not None:
        v_roi = _metrics_for_curve(v_centers_cm, proj_v_roi, cfg.v)
        if v_roi:
            result["v"]["roi"] = v_roi

    return result
