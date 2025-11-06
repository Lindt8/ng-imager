from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Dict, List, Union

class RunCfg(BaseModel):
    mode: Literal["fast","list"] = "fast"
    source_type: Literal["cf252","dt","proton_center","phits"] = "proton_center"
    workers: Union[int, Literal["auto"]] = "auto"
    chunk_cones: Union[int, Literal["auto"]] = "auto"
    jit: bool = False
    progress: bool = True

class IOCfg(BaseModel):
    input: str
    output: str
    detector_map: str

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

class Config(BaseModel):
    run: RunCfg
    io: IOCfg
    plane: PlaneCfg
    filters: FiltersCfg
    energy: EnergyCfg
    prior: PriorCfg
    uncertainty: UncertaintyCfg
