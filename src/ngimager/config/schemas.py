from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Dict, List, Union, Any

class RunCfg(BaseModel):
    """
        Global run controls (see docs/dev/architecture.md §3.1).
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
    workers: Union[int, Literal["auto"]] = "auto"
    chunk_cones: Union[int, Literal["auto"]] = "auto"
    jit: bool = False
    progress: bool = True

    # Diagnostics
    diagnostics_level: int = 1  # 0=off, 1=minimal, 2=verbose

    # Limits
    max_cones: Optional[int] = None

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
    input_format = "phits"         # "phits" | "root_novo_ddaq"
    output_path  = "..."
    """

    input_path: str
    input_format: Literal["phits_usrdef", "root_novo_ddaq", "hdf5_ngimager"] = "phits_usrdef"
    output_path: str

    # Adapter-specific sub-config, e.g. [io.adapter]
    adapter: Dict[str, Any] = Field(default_factory=dict)

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
    eu: Optional[List[float]] = None
    ev: Optional[List[float]] = None
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
      bars_include      = []
      bars_exclude      = []
      materials_include = []
      materials_exclude = []

      [filters.hits.neutron]
      min_light_MeVee = 100.0   # optional override; others fall back to [filters.hits]

      [filters.hits.gamma]
      # optional overrides...
    """
    # Universal defaults
    min_light_MeVee: float = 0.0
    max_light_MeVee: float = 1.0e12
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

class VisCfg(BaseModel):
    export_png_on_write: bool = True
    # Default to neutron summed image, matching lm_store layout
    summed_dataset: str = "/images/summed/n"


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

