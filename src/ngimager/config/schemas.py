from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Literal, Optional, Dict, List, Union, Any

class RunCfg(BaseModel):
    """
        Global run controls that apply to the entire pipeline.

        source_type:
            "cf252" | "dt" | "proton_center" | "phits"
        fast:
            Enable fast-mode overrides (see FastCfg and [fast] section).
        list:
            Enable list-mode imaging output (/lm/cone_pixel_indices, etc.).
    """

    # Which species to process
    neutrons: bool = True
    gammas: bool = True

    # Behavioral toggles
    fast: bool = False  # use aggressive "fast" settings
    list: bool = False  # enable list-mode imaging output

    # Experiment/source context
    source_type: Literal["cf252", "dt", "proton_center", "phits"] = "proton_center"

    # Performance / execution
    workers: Union[int, Literal["auto"]] = 0
    chunk_cones: Union[int, Literal["auto"]] = "auto"
    jit: bool = False
    progress: bool = True

    # Diagnostics
    diagnostics_level: int = 1  # 0=off, 1=minimal, 2=verbose

    # Limits
    max_cones: Optional[int] = None

    # SBP imaging engine: "scan" (matrix scan) or "poly" (perimeter sampler)
    sbp_engine: str = "scan"

    # Human-readable label for this run, used by visualization and stored
    # in HDF5 under /meta as the "run_plot_label" attribute.
    plot_label: Optional[str] = None

    # Free-form run-level metadata table. Exposed in TOML as:
    #
    #   [run.meta]
    #   beam   = "175 MeV protons"
    #   target = "geometry B"
    #   setup  = "detector arrangement 3"
    #
    # and stored into HDF5 under /meta/run_meta for downstream tools.
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("diagnostics_level")
    def _diag_range(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError("diagnostics_level must be 0, 1, or 2")
        return v

class IOCfg(BaseModel):
    """
    I/O paths and high-level source description.

    TOML:

    [io]
    input_path   = "..."
    input_format = "phits_usrdef"  # "phits_usrdef" | "root_novo_ddaq" | "hdf5_ngimager"
    output_path  = "..."
    """

    input_path: str
    input_format: Literal["phits_usrdef", "root_novo_ddaq", "hdf5_ngimager"] = "phits_usrdef"
    output_path: str

    # Adapter-specific sub-config, e.g. [io.adapter]
    adapter: Dict[str, Any] = Field(default_factory=dict)
    restart_path: Optional[str] = None  # Not yet implemented
    hdf5_overwrite: bool = False  # Not yet implemented

    # Optional extra text metadata files to embed into the HDF5.
    #
    # Populated from a TOML table:
    #
    #   [io.extra_text_files]
    #   phits_input = "path/to/phits.inp"
    #   daq_config  = "path/to/daq_config.txt"
    #
    # Keys become dataset names under /meta/extra_text; values are file
    # paths (relative to the TOML config file unless absolute).
    extra_text_files: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry configuration models
# ---------------------------------------------------------------------------

class DetectorFrameGeometry(BaseModel):
    """
    A simple rigid transform that maps detector-local coordinates to
    world coordinates:

        p_world = R_xyz(rotation_deg) @ p_local + origin_cm

    where rotation_deg = [rx, ry, rz] are Euler angles in degrees,
    applied in the fixed order Rx → Ry → Rz.
    """

    origin_cm: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_deg: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class PerDetectorGeometry(BaseModel):
    """
    OPTIONAL (stub for future expansion).

    Describes an individual detector tile or module's placement within
    the detector-frame coordinate system. For now, ng-imager does not
    apply per-detector transforms, but the schema entry is accepted and
    stored for future expansion.
    """
    id: int
    origin_cm: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_deg: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class DetectorsGeometryCfg(BaseModel):
    """
    Global detector-array geometry.

    - 'frame' describes the overall detector coordinate frame relative to
      world coordinates.
    - 'detectors' is an optional list for fine-grained per-detector
      transforms (currently unused, but reserved for future support).
    """

    frame: DetectorFrameGeometry = Field(default_factory=DetectorFrameGeometry)
    detectors: List[PerDetectorGeometry] = Field(default_factory=list)



class DetectorsCfg(BaseModel):
    """
    Mapping from detector IDs/regions to materials and optional geometry.

    TOML:

    [detectors]
    default_material = "OGS"

    [detectors.material_map]
    200 = "OGS"
    210 = "M600"
    ...

    # Optional global detector-frame → world-frame transform:
    [detectors.geometry.frame]
    origin_cm    = [0.0, 0.0, 0.0]
    rotation_deg = [0.0, 0.0, 0.0]

    # OPTIONAL (stub for future expansion): per-detector transforms
    [[detectors.geometry.detectors]]
    id          = 0
    origin_cm    = [0.0, 0.0, 0.0]
    rotation_deg = [0.0, 0.0, 0.0]
    """

    material_map: Dict[int, str] = Field(default_factory=dict)
    default_material: str = "UNK"

    # Detector-frame → world-frame geometry description.
    geometry: DetectorsGeometryCfg = Field(default_factory=DetectorsGeometryCfg)





class PipelineCfg(BaseModel):
    """
    Controls how far through the pipeline we run.

    until = "hits" | "events" | "cones" | "image"
    """

    until: Literal["hits", "events", "cones", "image"] = "image"


class PlaneCfg(BaseModel):
    origin: List[float]
    normal: List[float]
    u_axis: Optional[List[float]] = None # eu
    v_axis: Optional[List[float]] = None # ev
    u_min: float; u_max: float; du: float
    v_min: float; v_max: float; dv: float

# --- Filters configuration ---------------------------------------------------

class HitSpeciesOverrides(BaseModel):
    """
    Species-specific overrides for hit-level filters.

    All fields are optional; when None, the universal [filters.hits] value is used.
    """
    min_light_MeVee: Optional[float] = None
    max_light_MeVee: Optional[float] = None
    psd_min: Optional[float] = None
    psd_max: Optional[float] = None
    bars_include: Optional[List[int]] = None
    bars_exclude: Optional[List[int]] = None
    materials_include: Optional[List[str]] = None
    materials_exclude: Optional[List[str]] = None


class HitsFiltersCfg(BaseModel):
    """
    Hit-level filters with universal defaults plus neutron/gamma overrides.

    TOML:

      [filters.hits]
      min_light_MeVee = 50.0
      max_light_MeVee = 1.0e6
      psd_min         = 0.0
      psd_max         = 1.0
      bars_include      = []
      bars_exclude      = []
      materials_include = []
      materials_exclude = []

      [filters.hits.neutron]
      min_light_MeVee = 100.0   # optional override; others fall back to [filters.hits]
      psd_min         = 0.2     # optional override; if omitted, uses [filters.hits]
      psd_max         = 0.6

      [filters.hits.gamma]
      # optional overrides...
    """
    # Universal defaults
    min_light_MeVee: float = 0.0
    max_light_MeVee: float = 1.0e12
    psd_min: Optional[float] = None
    psd_max: Optional[float] = None
    bars_include: List[int] = Field(default_factory=list)
    bars_exclude: List[int] = Field(default_factory=list)
    materials_include: List[str] = Field(default_factory=list)
    materials_exclude: List[str] = Field(default_factory=list)

    # Species-specific overrides
    neutron: HitSpeciesOverrides = Field(default_factory=HitSpeciesOverrides)
    gamma: HitSpeciesOverrides = Field(default_factory=HitSpeciesOverrides)


class EventSpeciesOverrides(BaseModel):
    """
    Species-specific overrides for event-level filters.

    All fields are optional; when None, the universal [filters.events] value
    is used for ToF, and L-thresholds default to "no extra cut".
    """
    tof_window_ns: Optional[List[float]] = None
    min_L1_MeVee: Optional[float] = None
    min_L2_MeVee: Optional[float] = None
    min_L_any_MeVee: Optional[float] = None



class EventsFiltersCfg(BaseModel):
    """
    Event-level filters with universal defaults plus neutron/gamma overrides.

    TOML:

      [filters.events]
      tof_window_ns = [0.0, 30.0]

      [filters.events.neutron]
      tof_window_ns = [0.0, 30.0]
      min_L1_MeVee = 0.0
      min_L2_MeVee = 0.0

      [filters.events.gamma]
      tof_window_ns = [0.0, 30.0]
      min_L_any_MeVee = 0.0
    """
    tof_window_ns: List[float] = Field(default_factory=lambda: [0.0, 1.0e9])
    neutron: EventSpeciesOverrides = Field(default_factory=EventSpeciesOverrides)
    gamma: EventSpeciesOverrides = Field(default_factory=EventSpeciesOverrides)


class ConeSpeciesOverrides(BaseModel):
    """
    Species-specific overrides for cone-level filters.
    """
    max_delta_theta_deg: Optional[float] = None
    max_incident_energy_MeV: Optional[float] = None


class ConesFiltersCfg(BaseModel):
    """
    Cone-level filters with universal defaults plus neutron/gamma overrides.

    TOML:

      [filters.cones]
      max_delta_theta_deg = 5.0

      [filters.cones.neutron]
      max_delta_theta_deg = 3.0

      [filters.cones.gamma]
      max_delta_theta_deg = 8.0
    """
    max_delta_theta_deg: Optional[float] = None
    max_incident_energy_MeV: Optional[float] = None
    neutron: ConeSpeciesOverrides = Field(default_factory=ConeSpeciesOverrides)
    gamma: ConeSpeciesOverrides = Field(default_factory=ConeSpeciesOverrides)


class FiltersCfg(BaseModel):
    """
    Top-level filter configuration, split into hits / events / cones.
    """
    hits: HitsFiltersCfg = Field(default_factory=HitsFiltersCfg)
    events: EventsFiltersCfg = Field(default_factory=EventsFiltersCfg)
    cones: ConesFiltersCfg = Field(default_factory=ConesFiltersCfg)


class FastCfg(BaseModel):
    """
    Fast-mode override knobs.

    Applied only when [run].fast = true; otherwise ignored.
    """
    # Event-level light thresholds (units explicitly in name)
    min_L1_MeVee: Optional[float] = None
    min_L2_MeVee: Optional[float] = None
    min_L_any_MeVee: Optional[float] = None

    # Hard cap on number of cones actually built/imaged
    max_cones: Optional[int] = None

    # Imaging plane downsampling factor:
    #   1 or None → no change
    #   2         → double du,dv (coarser grid, ~4x fewer pixels)
    plane_downsample: Optional[int] = None

    # Optional override of the SBP engine for fast-mode runs.
    # None → use run.sbp_engine (which defaults to "scan"),
    # otherwise one of "poly" or "scan".
    sbp_engine: Optional[str] = None


class EnergyCfg(BaseModel):
    """
    Energy strategy configuration.

    strategy:
        "ELUT"       – invert light via E(L) LUTs (per-material, per-species)
        "ToF"        – simple ToF-based estimate (placeholder)
        "FixedEn"    – fixed incident neutron energy (e.g. 14.1 MeV source)
        "Edep"       – direct deposited energy (PHITS-style adapters)
    """
    strategy: Literal["ELUT","ToF","FixedEn", "Edep"] = "ELUT"
    #fixed_En_MeV: float = 14.1
    fixed_En_MeV: float | None = None
    lut_paths: Dict[str, Dict[str, str]] = {}   # material -> species -> path
    force_proton_recoils: bool = False

class PriorCfg(BaseModel):
    type: Literal["point","line"] = "point"
    point: Optional[List[float]] = None
    line: Optional[Dict[str, List[float]]] = None
    strength: float = 1.0

class UncertaintyCfg(BaseModel):
    enabled: bool = False
    smearing: Literal["thicken","weighted"] = "thicken"
    sigma_doi_cm: float = 0.35
    sigma_transverse_cm: float = 0.346
    sigma_time_ns: float = 0.5
    use_lut_bands: bool = False


class ProjectionAxisMetricsCfg(BaseModel):
    """
    Per-axis projection metrics controls.

    These are applied separately for the u and v axes.
    """

    compute_summary: bool = True
    # mean/median/std, total counts

    compute_peak: bool = True
    # simple argmax-based peak position / height

    compute_edges: bool = False
    # CDF-based edge positions (for line / range studies)

    edge_low_frac: float = 0.2
    edge_high_frac: float = 0.8
    # fractions in (0, 1), e.g. 0.2 and 0.8 → 20–80% edge width

    min_counts: float = 100.0
    # below this total counts, metrics are marked invalid (NaN + *_ok=False)

    @field_validator("edge_low_frac", "edge_high_frac")
    @classmethod
    def _check_edge_fracs(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("edge_*_frac must be in the open interval (0, 1)")
        return v

    @field_validator("edge_high_frac")
    @classmethod
    def _check_edge_order(
            cls,
            v: float,
            info: ValidationInfo,
    ) -> float:
        # In Pydantic v2, the second argument is ValidationInfo; use info.data.
        data = info.data or {}
        low = data.get("edge_low_frac")
        if low is not None and v <= low:
            raise ValueError("edge_high_frac must be > edge_low_frac")
        return v

class ProjectionMetricsCfg(BaseModel):
    """
    Controls whether projection metrics are computed and written to HDF5.
    """

    enabled: bool = False
    # master toggle – when false, no metrics are computed/written

    u: ProjectionAxisMetricsCfg = Field(default_factory=ProjectionAxisMetricsCfg)
    v: ProjectionAxisMetricsCfg = Field(default_factory=ProjectionAxisMetricsCfg)


class ProjectionPlotCfg(BaseModel):
    """
    Controls how projection metrics are visualized in vis/hdf.py.

    (Plotting code will read these; they do not affect HDF5 contents.)

    TOML example:

        [vis.projections.plot]
        # Basic visual toggles
        show_peak_markers = true   # vertical / horizontal lines at peak_pos_cm
        show_edge_markers = true   # lines at edge_low_cm / edge_high_cm
        show_centroid_2d  = false  # crosshair at 2D centroid (if computed)

        # Which metrics to use when both "all" and ROI curves exist
        metrics_source    = "auto"     # "auto" | "all" | "roi" | "both"
        curve_mode        = "all+roi"  # "all+roi" | "all_only" | "roi_only"

        # Numeric summaries on the figure
        # - "off"     : no text annotations
        # - "compact" : minimal one-line summary per axis
        # - "full"    : include more fields (e.g. edges) when available
        annotate_summary  = "compact"

        # Optional extra panel with a table of metrics (future)
        show_metrics_panel = false
    """

    show_peak_markers: bool = True
    show_edge_markers: bool = True
    show_centroid_2d: bool = False

    # Which metrics to use when both "all" and ROI curves exist:
    #   "auto" : prefer ROI if available, otherwise fall back to "all"
    #   "all"  : use only "all" metrics
    #   "roi"  : use only ROI metrics (if missing, no metrics are shown)
    #   "both" : expose both "all" and ROI metrics (styling handled in vis/hdf.py)
    metrics_source: Literal["auto", "all", "roi", "both"] = "auto"

    # Which projection curves to draw:
    #   "all+roi" : show both all-pixels and ROI curves (current behavior)
    #   "all_only": show only the all-pixels projection
    #   "roi_only": show only the ROI-limited projection (if present)
    curve_mode: Literal["all+roi", "all_only", "roi_only"] = "all+roi"

    # Numeric annotations on the projections:
    #   "off"     : no numeric text
    #   "compact" : minimal, high-level summary
    #   "full"    : extended summary (e.g. including edges)
    annotate_summary: Literal["off", "compact", "full"] = "compact"

    # When true, plotting code may add a dedicated metrics panel
    # (e.g. a small table of key values) next to the projections.
    show_metrics_panel: bool = False



class VisProjectionsCfg(BaseModel):
    """
    Configuration for 1D projections and their analysis/visualization.

    TOML:

        [vis.projections]
        enabled      = true
        roi_u_min_cm = -5.0
        roi_u_max_cm =  5.0
        roi_v_min_cm = -5.0
        roi_v_max_cm =  5.0

        [vis.projections.metrics]
        enabled = true

        [vis.projections.metrics.u]
        compute_summary = true
        compute_peak    = true
        compute_edges   = false
        edge_low_frac   = 0.2
        edge_high_frac  = 0.8
        min_counts      = 100.0

        [vis.projections.metrics.v]
        compute_summary = true
        compute_peak    = true
        compute_edges   = true

        [vis.projections.plot]
        show_peak_markers = true
        show_edge_markers = true
        show_centroid_2d  = false

        metrics_source    = "auto"     # "auto" | "all" | "roi" | "both"
        curve_mode        = "all+roi"  # "all+roi" | "all_only" | "roi_only"

        annotate_summary  = "compact"  # "off" | "compact" | "full"
        show_metrics_panel = false
    """

    enabled: bool = False

    # Optional ROI window in plane coordinates (cm).
    # If any of these are None, the ROI is treated as "disabled".
    roi_u_min_cm: Optional[float] = None
    roi_u_max_cm: Optional[float] = None
    roi_v_min_cm: Optional[float] = None
    roi_v_max_cm: Optional[float] = None

    metrics: ProjectionMetricsCfg = Field(default_factory=ProjectionMetricsCfg)
    plot: ProjectionPlotCfg = Field(default_factory=ProjectionPlotCfg)

    @field_validator("roi_u_max_cm")
    @classmethod
    def _check_roi_u(
            cls,
            vmax: Optional[float],
            info: ValidationInfo,
    ) -> Optional[float]:
        # In Pydantic v2, `info` is ValidationInfo; use `info.data` to access other fields.
        data = info.data or {}
        vmin = data.get("roi_u_min_cm")
        if vmax is not None and vmin is not None and vmax < vmin:
            raise ValueError("roi_u_max_cm must be >= roi_u_min_cm")
        return vmax

    @field_validator("roi_v_max_cm")
    @classmethod
    def _check_roi_v(
            cls,
            vmax: Optional[float],
            info: ValidationInfo,
    ) -> Optional[float]:
        data = info.data or {}
        vmin = data.get("roi_v_min_cm")
        if vmax is not None and vmin is not None and vmax < vmin:
            raise ValueError("roi_v_max_cm must be >= roi_v_min_cm")
        return vmax

    def roi_bounds_cm(self) -> Optional[tuple[float, float, float, float]]:
        """
        Return (u_min, u_max, v_min, v_max) in cm if a full ROI is defined,
        otherwise None.
        """
        if (
            self.roi_u_min_cm is None
            or self.roi_u_max_cm is None
            or self.roi_v_min_cm is None
            or self.roi_v_max_cm is None
        ):
            return None
        return (
            float(self.roi_u_min_cm),
            float(self.roi_u_max_cm),
            float(self.roi_v_min_cm),
            float(self.roi_v_max_cm),
        )


class VisCfg(BaseModel):
    """
    Visualization configuration.

    These options control automatic image export from the pipeline and provide
    defaults for the standalone `ng-viz` CLI.
    """

    # When true, the pipeline writes image files after reconstruction completes.
    export_png_on_write: bool = True

    # Legacy / advanced option: a single dataset path.
    # Kept for backward compatibility with older configs and helper scripts.
    summed_dataset: str = "/images/summed/n"

    # 1D projection outputs and analysis
    projections: VisProjectionsCfg = Field(default_factory=VisProjectionsCfg)

    # Which summed images to render automatically from `/images/summed`.
    #
    # Allowed values:
    #   "n"   – neutron-only image  (`/images/summed/n`)
    #   "g"   – gamma-only image    (`/images/summed/g`)
    #   "all" – combined n+g image  (`/images/summed/all`, only when both exist)
    species: list[Literal["n", "g", "all"]] = Field(
        default_factory=lambda: ["n", "g", "all"],
        description="List of image species to render automatically.",
    )

    # If true, shift the plotting coordinates so that (u, v) = (0, 0) is at
    # the imaging plane center. If false, use raw grid.u_min/grid.v_min.
    center_on_plane_center: bool = True

    # If true, flip the plotted image vertically relative to the natural v-axis
    # orientation. This is mainly useful for matching legacy images visually.
    flip_vertical: bool = False

    # Units for plotting axes: "cm" (native) or "mm".
    # Internally, grids are stored in cm; mm just rescales labels.
    axis_units: Literal["cm", "mm"] = "cm"

    # Matplotlib colormap name to use for images (e.g. "cividis", "viridis").
    cmap: str = "cividis"

    # File naming pattern for automatic exports. Available placeholders:
    #
    #   {stem}    – stem of the HDF5 filename (e.g. "phits_usrdef_simple")
    #   {species} – "n", "g", or "all"
    #   {ext}     – file extension ("png", "pdf", ...)
    filename_pattern: str = "{species}_{stem}.{ext}"

    # Additional output formats beyond PNG. The pipeline always writes PNG
    # when export_png_on_write is true; any extra formats listed here will be
    # written alongside (e.g. ["pdf"]).
    extra_formats: list[str] = Field(default_factory=list)

    # Optional 1D u/v projections and ROI configuration.
    projections: VisProjectionsCfg = Field(default_factory=VisProjectionsCfg)




class Config(BaseModel):
    """
    Top-level TOML configuration.
    """

    run: RunCfg
    io: IOCfg
    detectors: DetectorsCfg = Field(default_factory=DetectorsCfg)
    plane: PlaneCfg
    filters: FiltersCfg
    energy: EnergyCfg
    prior: PriorCfg
    uncertainty: UncertaintyCfg
    vis: VisCfg = Field(default_factory=VisCfg)
    pipeline: PipelineCfg = Field(default_factory=PipelineCfg)
    fast: FastCfg = Field(default_factory=FastCfg)

