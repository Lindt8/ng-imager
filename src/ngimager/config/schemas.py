from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
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

class DetectorsCfg(BaseModel):
    """
    Mapping from detector IDs/regions to materials and (later) geometry.

    TOML:

    [detectors]
    default_material = "OGS"

    [detectors.material_map]
    200 = "OGS"
    210 = "M600"
    ...
    """

    material_map: Dict[int, str] = Field(default_factory=dict)
    default_material: str = "UNK"

    # Placeholder for future geometry (bar positions/orientations, etc.)
    geometry: Dict[str, Any] = Field(default_factory=dict)


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


class VisProjectionsCfg(BaseModel):
    """
    Configuration for 1D u/v projections of the summed images.

    All coordinates are in the native imaging-plane units (cm), matching
    the [plane] section and /meta.grid.* attributes in the HDF5 output.
    """

    # Enable computation and storage of 1D projections in the HDF5 file,
    # and use them when rendering images (u/v side panels).
    enabled: bool = False

    # Optional rectangular region-of-interest (ROI) in (u, v) coordinates.
    # If all four are provided, ROI-limited projections are computed by
    # summing only pixels whose centers fall inside this window.
    roi_u_min_cm: float | None = None
    roi_u_max_cm: float | None = None
    roi_v_min_cm: float | None = None
    roi_v_max_cm: float | None = None



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

