Created by Hunter N. Ratliff, 2025-10-17

This code generates light response functions/lookup-tables (LUTs), forward and
inverse, for NOVO's M600 and OGS scintillators using my SRIM calculations as a
basis for a Birks function fit whose parameters are optimized for proton light
response data collected in NOVO's March 2024 PTB experiments.


# Light-Response Fitting and LUT Generation


This script builds physics-based light-response models for plastic scintillators
and exports fast lookup tables (LUTs) to convert measured light output (MeVee)
into recoil energy (MeV). It supports both proton and carbon recoils and produces
figures for fit quality and inverse-response uncertainty bands.

## What this script does 

1) Loads stopping power data (SRIM) for protons and carbon and converts to linear
   stopping power using the scintillator density.
2) Loads experimental calibration data from PTB that map proton recoil energy
   to measured light output.
3) Fits a Birks-type light-yield model (Birks or Birks–Chou) to the calibration data.
4) Optionally constrains S to 1 using gamma Compton-edge calibration (MeVee scale).
5) Builds dense forward and inverse LUTs:
   - Forward: E -> L(E)
   - Inverse: L -> E(L), uniform grid in L for fast np.interp
6) Computes 68 percent confidence bands on E(L) by sampling the fitted
   parameters and propagating to inverse LUTs.
7) Exports portable artifacts (NPZ, CSV, JSON metadata) and generates plots.

## Inputs

- SRIM stopping power files for H and C ions for each scintillator:
  - Must include energy (MeV) and mass stopping power (MeV cm^2 / g).
  - Energy range ideally covers 1 keV to at least 100–250 MeV.
- Scintillator density rho (g/cm^3) for each material.
- Experimental proton light-response calibration:
  - Arrays of Ep_MeV (proton recoil energies) and L_MeVee (measured light output).
  - Optional grouping labels for different neutron energies (En_indices, En_strs)
    to visualize subsets.
- Gamma Compton calibration (performed upstream):
  - Data acquisition already outputs MeVee. This allows S to be fixed to 1 or softly
    constrained near 1.

## Outputs

For each scintillator and species (proton, carbon):

- NPZ file: basepath.npz
  - Arrays: L_inv (MeVee), E_inv (MeV)
  - Optional arrays: E_inv_lo, E_inv_hi (16th and 84th percentile inverse bands)
  - Metadata object with model, parameters, density, fit stats, grid sizes, timestamp
- CSV file: basepath.csv
  - Two columns: L_inv_MeVee, E_inv_MeV (plaintext for sharing and longevity)
- JSON metadata: basepath.meta.json
  - Human-readable metadata mirror of the NPZ meta
- Plots (if enabled):
  - Birks fit and residuals (stacked) per scintillator
  - Inverse response E(L) with 68 percent bands for proton and carbon
  - Zoomed carbon inverse plot

## Methods and process

1) Units and data prep
   - Convert mass stopping power to linear: dE/dx [MeV/cm] = rho * (dE/dx)_mass
     [MeV cm^2 / g].
   - Interpolate dE/dx(E) with a monotone, nonnegative interpolant
     (shape-preserving cubic, or safe wrapper).

2) Light-response model
   - Birks: dL/dx = S * (dE/dx) / (1 + kB * dE/dx)
   - Birks–Chou (optional): dL/dx = S * (dE/dx) / (1 + kB * dE/dx + C * (dE/dx)^2)
   - Total light for a recoil of energy E is the integral of dL/dE over energy.
     Numerically integrate over a dense E grid.

3) Parameter fitting
   - Nonlinear least squares (scipy.optimize.least_squares) on residuals
     L_model(Ei) - L_data,i.
   - Residual variance scaling: covariance = sigma^2 * (J^T J)^-1 with
     sigma^2 = SSE / (N - p).
   - Report best-fit parameters, 1 sigma uncertainties, R^2, adjusted R^2, RMSE.

4) Handling S (electron-equivalent scale)
   - If data are already in MeVee via Compton-edge calibration, fix S = 1 (hard)
     or apply a soft Gaussian prior on S near 1 (e.g., sigma 0.01–0.02).
   - This removes S–kB degeneracy and stabilizes extrapolation.

5) Building LUTs
   - Forward: compute L(E) on a dense E grid (e.g., up to 250 MeV).
   - Inverse: create a uniformly spaced L grid and tabulate E(L) with np.interp.
   - Save proton and carbon inverse LUTs separately. Use float32 for compact
     storage and fast lookup.

6) Uncertainty bands (optional)
   - Draw samples of [S, kB, C] from the multivariate normal defined by the fitted
     covariance.
   - For each sample, compute inverse E(L) onto the fixed L grid.
   - Take the 16th and 84th percentiles across samples at each L to form a 68
     percent confidence band.
   - Store E_inv_lo and E_inv_hi alongside the central inverse LUT.

7) Plotting (optional)
   - Fit-quality figure: top panel shows data and model L(E); bottom panel shows
     percent residuals.
   - Inverse figure: E(L) central curve with 68 percent band for proton and carbon.
   - Carbon often appears highly quenched; use a zoomed L range (e.g., L < 8 MeVee)
     or annotate unreachable regions using elastic kinematic caps.

## How to use the LUTs downstream

- Load NPZ: L_inv, E_inv. Convert MeVee to recoil energy with
  Ep = np.interp(L_meas, L_inv, E_inv).
  Fast example drop-in code:
  ```
  pack = np.load("lut_M600_proton.npz", allow_pickle=True)
  L_inv = pack["L_inv"]; E_inv = pack["E_inv"]
  def Ep_from_L(L): return np.interp(L, L_inv, E_inv)
  ```
- If uncertainty bands were exported: compute Ep_lo and Ep_hi via the same
  interpolation on E_inv_lo and E_inv_hi.
- Use proton and carbon LUTs in parallel and let imaging logic choose between
  hypotheses or carry both with weights.
- Optional: clip carbon solutions using an elastic kinematic ceiling given a
  neutron energy bound.

## Configuration knobs

- Model selection: use_Chou_C_term boolean to include the C term.
- S handling: lock S exactly to 1 via lock_S_to_1 = True or via bounds,
  or set a soft prior on S with prior_sigma.
- Grids: E_max, nE for forward integration; nL for inverse grid density.
- Band sampling: number of parameter draws, sample filtering for monotonicity
  and stability.

## Assumptions and caveats

- Electron-equivalent calibration is already applied upstream; therefore S should
  be 1 or tightly constrained near 1.
- The carbon LUT is more uncertain in practice without carbon-tagged calibration;
  use it as a conservative branch and apply kinematic caps where appropriate.
- Extrapolation beyond the calibration Ep range is supported but rely on the band
  to communicate model uncertainty.
- Monotonicity is required for E(L) inversion; pathological parameter draws are
  rejected.

## Troubleshooting

- Inverse interpolation error (requires at least two unique L points): occurs if
  a sampled parameter set produces nearly flat L(E). The sampler filters such
  draws; increase sample count or tighten priors if too many draws are rejected.
- Large parameter uncertainties under Birks–Chou: usually indicates kB and C are
  highly correlated and C is weakly identifiable; prefer simple Birks unless
  low-energy data demand C.
- Odd high-L divergence between Birks and Chou: typically due to unconstrained S;
  fix S via Compton calibration.

## Dependencies

- numpy, scipy, matplotlib
- Hunters_tools (https://github.com/Lindt8/Hunters-tools/blob/master/Hunters_tools.py)
  module import (used for plotting)
- No runtime dependency on scipy in downstream imaging if you use saved L_inv and
  E_inv with np.interp.

## Files written (per scintillator and species)

- basepath.npz: L_inv, E_inv, and optional E_inv_lo, E_inv_hi, plus metadata.
- basepath.csv: plaintext columns L_inv_MeVee, E_inv_MeV.
- basepath.meta.json: metadata (scintillator, species, model, parameters, density,
  fit stats, grid sizes, timestamp).
- Figures: fit and inverse-band plots if saving is enabled.

This design yields a transparent, physics-backed model with fast and portable
inverse LUTs for experimental imaging.