from __future__ import annotations
from dataclasses import dataclass
import importlib.resources as res
from pathlib import Path
import numpy as np
from typing import Dict

def builtin_lut_path(material: str, species: str) -> Path:
    """Return path to a built-in LUT .npz for given material/species."""
    try:
        return res.files(f"ngimager.data.lut.{material}") / f"lut_{material}_{species}_Birks.npz"
    except ModuleNotFoundError:
        raise FileNotFoundError(f"No built-in LUT found for {material}/{species}")

@dataclass
class LUT:
    L: np.ndarray
    E: np.ndarray
    meta: dict
    E_lo: np.ndarray | None = None
    E_hi: np.ndarray | None = None

    def eval(self, Lval: float) -> tuple[float, float | None]:
        e = float(np.interp(Lval, self.L, self.E))
        if self.E_lo is not None and self.E_hi is not None:
            elo = float(np.interp(Lval, self.L, self.E_lo))
            ehi = float(np.interp(Lval, self.L, self.E_hi))
            sigma = 0.5 * (ehi - elo) / 1.0  # crude ~1σ from 68% band
            return e, sigma
        return e, None

def load_npz_lut(path: str | Path) -> LUT:
    p = Path(path)
    with np.load(p, allow_pickle=True) as z:
        keys = set(z.files)

        # meta (optional)
        meta = dict(z["meta"].item()) if "meta" in keys else {}

        # Try common naming conventions for arrays
        # 1) original:           L / E
        # 2) verbose:                L_vals / E_vals
        # 3) generic:                light / energy
        # 4) inverse naming:         L_inv / E_inv
        # 5) combined table:         table[:,0]=L, table[:,1]=E
        candidates = [
            ("L", "E"),
            ("L_vals", "E_vals"),
            ("light", "energy"),
            ("L_inv", "E_inv"),
        ]

        L = E = None
        for Lk, Ek in candidates:
            if Lk in keys and Ek in keys:
                L = z[Lk].astype(np.float64)
                E = z[Ek].astype(np.float64)
                break

        if L is None or E is None:
            if "table" in keys:
                tbl = z["table"]
                if tbl.ndim == 2 and tbl.shape[1] >= 2:
                    L = tbl[:, 0].astype(np.float64)
                    E = tbl[:, 1].astype(np.float64)

        if L is None or E is None:
            raise KeyError(
                f"Could not find LUT arrays in {p.name}. "
                f"Expected one of {candidates} or 'table'. Found keys: {sorted(keys)}"
            )

        # Optional uncertainty bands (various namings)
        E_lo = E_hi = None
        band_candidates = [
            ("E_lo", "E_hi"),
            ("Emin", "Emax"),
            ("E_lo_vals", "E_hi_vals"),
        ]
        for lo, hi in band_candidates:
            if lo in keys and hi in keys:
                E_lo = z[lo].astype(np.float64)
                E_hi = z[hi].astype(np.float64)
                break

    return LUT(L=L, E=E, meta=meta, E_lo=E_lo, E_hi=E_hi)

def build_lut_registry(lut_paths: dict, cfg_file: str | Path | None = None) -> dict[str, dict[str, LUT]]:
    """
    Build LUT registry from config.  Looks for files relative to the config file first,
    then falls back to built-in ngimager.data.lut defaults.
    """
    reg: dict[str, dict[str, LUT]] = {}
    base = Path(cfg_file).parent if cfg_file else Path.cwd()

    for material, species_map in lut_paths.items():
        reg[material] = {}
        for species, raw_path in species_map.items():
            # Resolve relative paths against the config's directory
            p = Path(raw_path)
            if not p.is_absolute():
                p = (base / p).resolve()
            if p.exists():
                reg[material][species] = load_npz_lut(p)
                continue

            # Fall back to built-in data
            builtin = builtin_lut_path(material, species)
            if builtin.exists():
                reg[material][species] = load_npz_lut(builtin)
                continue

            raise FileNotFoundError(f"LUT for {material}/{species} not found: {raw_path}")

    return reg
