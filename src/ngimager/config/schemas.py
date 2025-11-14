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

class FiltersCfg(BaseModel):
    min_light: float = 0.0
    max_light: float = 1e12
    tof_window_ns: List[float] = [0.0, 1e9]
    bars_include: List[int] = []
    materials_include: List[str] = []

class EnergyCfg(BaseModel):
    strategy: Literal["ELUT","ToF","FixedEn"] = "ELUT"
    fixed_En_MeV: float = 14.1
    lut_paths: Dict[str, Dict[str, str]] = {}   # material -> species -> path

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

