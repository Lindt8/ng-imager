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

def build_lut_registry(
    lut_paths: Dict[str, Dict[str, str]] | None,
    base_dir: str | Path | None = None,
) -> Dict[str, Dict[str, LUT]]:
    """
    Build a registry mapping material -> species -> LUT.

    Parameters
    ----------
    lut_paths
        Configuration-style mapping, e.g.:

            {
                "M600": {"proton": "data/lut/M600/lut_M600_proton_Birks.npz"},
                "OGS":  {"carbon": "custom/OGS_carbon.npz"},
            }

        Paths may be relative; they are resolved against `base_dir` when given.
        If a configured path does not exist on disk but a built-in LUT is
        available for that material/species (M600/OGS proton/carbon), the
        built-in is used as a fallback.

        When a material/species is *omitted* entirely from `lut_paths`, this
        function will still inject built-in defaults for common NOVO
        scintillators (M600, OGS).

    base_dir
        Base directory for resolving relative paths (typically the directory
        containing the TOML config). If None, uses the current working
        directory.

    Returns
    -------
    dict
        Nested dictionary: {material: {species: LUT, ...}, ...}
    """
    if lut_paths is None:
        lut_paths = {}

    base = Path(base_dir) if base_dir is not None else Path(".")

    registry: Dict[str, Dict[str, LUT]] = {}

    # ------------------------------------------------------------------
    # 1) Explicit configuration entries
    # ------------------------------------------------------------------
    for material, species_map in lut_paths.items():
        if not species_map:
            continue

        mat_key = str(material)
        mat_reg = registry.setdefault(mat_key, {})

        for species, raw_path in species_map.items():
            sp_key = str(species)

            # Resolve path if provided
            path: Path
            if raw_path:
                p = Path(raw_path)
                if not p.is_absolute():
                    p = base / p
                if p.exists():
                    path = p
                else:
                    # Config path is missing; fall back to built-in if available
                    try:
                        path = builtin_lut_path(mat_key, sp_key)
                    except FileNotFoundError as exc:
                        raise FileNotFoundError(
                            f"LUT path '{raw_path}' for {mat_key}/{sp_key} "
                            f"does not exist and no built-in LUT is available."
                        ) from exc
            else:
                # Empty string / falsy path => force built-in for known materials
                try:
                    path = builtin_lut_path(mat_key, sp_key)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"No LUT path specified for {mat_key}/{sp_key} "
                        f"and no built-in LUT is available."
                    ) from exc

            mat_reg[sp_key] = load_npz_lut(path)

    # ------------------------------------------------------------------
    # 2) Built-in defaults for common NOVO scintillators
    #
    #    This is what lets a fresh 'pip install ng-imager' user write:
    #
    #        [energy]
    #        strategy = "ELUT"
    #
    #    and rely on packaged M600/OGS proton+carbon LUTs without any
    #    [energy.lut_paths.*] block.
    # ------------------------------------------------------------------
    builtin_defaults = {
        "M600": ("proton", "carbon"),
        "OGS": ("proton", "carbon"),
    }

    for material, species_list in builtin_defaults.items():
        mat_reg = registry.setdefault(material, {})
        for species in species_list:
            if species in mat_reg:
                # User already configured (or partially overrode) this species.
                continue
            try:
                path = builtin_lut_path(material, species)
            except FileNotFoundError:
                # If we somehow don't ship this LUT, just skip quietly.
                continue
            mat_reg[species] = load_npz_lut(path)

    return registry
