# src/ngimager/config/materials.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class MaterialResolver:
    det_to_material: Dict[int, str]
    default_material: str = "UNK"

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[int, str]] = None, default: str = "UNK"):
        return cls(det_to_material=dict(mapping or {}), default_material=default)

    @classmethod
    def from_env_or_defaults(cls):
        """
        Placeholder: later load from RunCfg/IOCfg (TOML) if available.
        For now returns empty mapping → 'UNK'.
        """
        return cls.from_mapping({})

    def material_for(self, det_id: int) -> str:
        return self.det_to_material.get(det_id, self.default_material)
