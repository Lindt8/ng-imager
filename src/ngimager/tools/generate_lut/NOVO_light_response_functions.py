'''
Created by Hunter N. Ratliff, 2025-10-17
This code generates light response functions/lookup-tables (LUTs), forward and
inverse, for NOVO's M600 and OGS scintillators using my SRIM calculations as a
basis for a Birks function fit whose parameters are optimized for proton light
response data collected in NOVO's March 2024 PTB experiments.

# ==============================================================================
# Light-Response Fitting and LUT Generation — Explainer
# ==============================================================================

This script builds physics-based light-response models for plastic scintillators
and exports fast lookup tables (LUTs) to convert measured light output (MeVee)
into recoil energy (MeV). It supports both proton and carbon recoils and produces
figures for fit quality and inverse-response uncertainty bands.

## What the script does (high level)

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
'''
from pathlib import Path
from datetime import datetime
import pickle, lzma, json
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from Hunters_tools import *

# =======================================================================
# SETTINGS AND FILES
# =======================================================================

base_path = Path.cwd()
images_path = base_path / 'images'
save_plots = True
show_plots = True
image_extensions = ['.pdf', '.png']
figi = 1  # initialize figure index var

scint_mats = ['M600', 'OGS']
density = {'M600':1.207, 'OGS':1.2591}  # g/cm^3
colors = {'M600':'tab:orange', 'OGS':'tab:blue'}  # used in plots

# Briks' fit first guesses (from Joey's values)
# S in MeVee/MeV
# kB in mg/MeV/cm^2
# kB_linear in cm/MeV
# C in (mg/MeV/cm^2)^2
# C_linear in (cm/MeV)^2
use_Chou_C_term = False  # if True, using Birks-Chou model with C != 0; if False, plain Birks with C = 0.
lock_S_to_1 = True  # I think this should be True, S=1, if we are using Compton edges to define the MeVee scale.
Birks_params = {
    'M600':{'S':1.00, 'kB':14.4, 'kB_linear':14.4*0.001/density['M600'], 'C':0, 'C_linear':0},
    'OGS':{'S':0.83, 'kB':5.5, 'kB_linear':5.5*0.001/density['OGS'], 'C':0, 'C_linear':0}
}


'''
As a note, these files are those directly produced by SRIM. The "SRIM_*.dat" files Joey used
have column 1 units of MeV and column 2 units of keV / (mg/cm^2) (or, equivalently, MeV / (g/cm^2)).
'''
SRIM_ions = ['H', 'C']
SRIM_files = {}
for scint in scint_mats:
    SRIM_files[scint] = {}
    for ion in SRIM_ions:
        SRIM_files[scint][ion] =  Path(base_path, 'SRIM_data', f'SRIM_{scint}_{ion}-ions.txt')


exp_light_response_data_M600 = Path(base_path, 'LightOutput_M600.dat')
exp_light_response_data_OGS = Path(base_path, 'LightOutput_OGS.dat')
exp_light_response_files = {'M600':exp_light_response_data_M600, 'OGS':exp_light_response_data_OGS}



# =======================================================================
# FUNCTIONS: FILE PARSING AND UTILITIES
# =======================================================================

def read_SRIM_output(path_to_SRIM_output):
    '''
    Parses a SRIM output file, returning a dictionary object with particle energies in MeV and
    mass stopping powers in MeV / (g/cm^2).
    '''
    E_MeV = []
    dEdx_ele = []
    dEdx_nuc = []
    dEdx_tot = []
    E_to_MeV_mults = {'meV':1e-9, 'eV':1e-6, 'keV':1e-3, 'MeV':1, 'GeV':1e+3, 'TeV':1e+6}
    stopping_units = ''
    with open(path_to_SRIM_output, 'r') as f:
        lines = f.readlines()
        in_data_table_section = False
        for line in lines:
            if '  --------------  ---------- ---------- ----------  ----------  ----------' in line:
                in_data_table_section = True
                continue
            if '-----------------------------------------------------------' in line:
                in_data_table_section = False
                continue
            if "Stopping Units =" in line:
                stopping_units = line.split('=')[-1].strip() # should be 'MeV / (mg/cm2)', but good to check anyways
                if stopping_units != 'MeV / (mg/cm2)':
                    print(f'WARNING: Stopping units are {stopping_units}, not the expected "MeV / (mg/cm2)".')
            if not in_data_table_section: continue
            parts = line.split()
            E_unit = parts[1].strip()
            E_MeV.append(float(parts[0])*E_to_MeV_mults[E_unit])
            dEdx_ele.append(float(parts[2]))
            dEdx_nuc.append(float(parts[3]))
            dEdx_tot.append(dEdx_ele[-1]+dEdx_nuc[-1])
    # scale up mass topping powers to MeV / (g/cm^2)
    stopping_units_mult = 1000
    SRIM_output = {
        'E_MeV':np.array(E_MeV),
        'dEdx_units':'MeV / (g/cm^2)',
        'dEdx_units_TeX':r'MeV/(g/cm$^2$)',
        'dEdx_electronic':np.array(dEdx_ele)*stopping_units_mult,
        'dEdx_nuclear':np.array(dEdx_nuc)*stopping_units_mult,
        'dEdx_total':np.array(dEdx_tot)*stopping_units_mult,
    }
    return SRIM_output

def read_light_response_file(path_to_light_output_data):
    '''
    This function is for reading Joey's two-column light response data points files "LightOutput_*.dat".
    Column 1 is the recoil proton energy Ep / neutron energy lost dEn in MeV, and
    Column 2 is the light response in MeVee
    It returns a dictionary object where the Ep and L pairs are proerly ordered, ascending by Ep
    Blank lines delimit values taken from different source neutron energies
    '''
    PTB_En_vals = ['14.8 MeV', '6.5 MeV', '2.5 MeV']
    En_index = 0
    PTB_En_indices = []
    Ep_MeV = []
    L_MeVee = []
    with open(path_to_light_output_data, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip())==0:
                En_index += 1
                continue
            parts = line.split()
            Ep_MeV.append(float(parts[0]))
            L_MeVee.append(float(parts[1]))
            PTB_En_indices.append(En_index)
    # Now reorder by Ep
    #Ep_MeV, L_MeVee = zip(*sorted(zip(Ep_MeV, L_MeVee)))
    return {'Ep_MeV':np.array(Ep_MeV), 'L_MeVee':np.array(L_MeVee), 'En_strs':PTB_En_vals, 'En_indices':np.array(PTB_En_indices)}

def ensure_monotone_increasing(x, y):
    # Enforce strictly increasing x and de-duplicate
    sorter = np.argsort(x)
    x_sorted = np.asarray(x)[sorter]
    y_sorted = np.asarray(y)[sorter]
    mask = np.concatenate([[True], np.diff(x_sorted) > 0])
    return x_sorted[mask], y_sorted[mask]

def make_dedx_interpolant(E, dEdx):
    # PCHIP is shape-preserving & monotone; avoids oscillations
    return PchipInterpolator(E, dEdx, extrapolate=True)

def pm_fmt(val, err, unit=None, sig_figs=2):
    """
    Format 'val ± err' preserving significant digits,
    handling very small numbers (e.g., 0.00301 ± 0.00012).
    """
    # Safety checks
    if err is None or not np.isfinite(err) or err <= 0:
        s = f"{val:.{sig_figs}g}"
        return f"{s} {unit}" if unit else s

    # Determine the order of magnitude of the uncertainty
    exp_err = int(np.floor(np.log10(err)))
    # Round to 'sig_figs' significant digits in the uncertainty
    rounded_err = round(err, -exp_err + (sig_figs - 1))
    # Match value rounding to same decimal place
    decimals = max(0, -(exp_err - (sig_figs - 1)))
    fmt = f"{{0:.{decimals}f}} ± {{1:.{decimals}f}}"
    s = fmt.format(val, rounded_err)

    # Handle very small values nicely (avoid scientific when not needed)
    if abs(val) < 1e-3 or abs(rounded_err) < 1e-3:
        s = f"{val:.{sig_figs}e} ± {err:.{sig_figs}e}"

    return f"{s} {unit}" if unit else s

# =======================================================================
# FUNCTIONS: BIRKS LAW MODEL (Birks–Chou when using C term)
# =======================================================================

def light_integral_grid(E_grid, dedx_func, S, kB, C=0.0):
    """
    Returns L(E) tabulated on E_grid using cumulative trapezoid integration of dL/dE.
    dL/dE = S / (1 + kB*dEdx + C*dEdx^2)
    """
    dEdx_vals = dedx_func(E_grid)
    denom = 1.0 + kB * dEdx_vals + C * dEdx_vals**2
    integrand = S / denom
    L_grid = cumulative_trapezoid(integrand, E_grid, initial=0.0)
    return L_grid

def make_forward_inverse_LUT(dedx_func, S, kB, C=0.0, E_max=250.0, nE=125001):
    """
    Build dense forward LUT (E->L) and inverse (L->E) interpolants.
    nE large => smooth & accurate integrals + inversion.
    """
    E_grid = np.linspace(0.0, E_max, nE)
    L_grid = light_integral_grid(E_grid, dedx_func, S, kB, C)
    # Ensure strict monotonicity for inversion (should be true physically)
    L_grid_mon, E_grid_mon = ensure_monotone_increasing(L_grid, E_grid)
    # Forward: E->L (fast linear or PCHIP)
    L_of_E = PchipInterpolator(E_grid, L_grid, extrapolate=False)
    # Inverse: L->E via PCHIP on swapped axes
    E_of_L = PchipInterpolator(L_grid_mon, E_grid_mon, extrapolate=False)
    return (E_grid, L_grid, L_of_E, E_of_L)


# =======================================================================
# FUNCTIONS: FITTING TO CALIBRATION DATA
# =======================================================================

def fit_birks_params(E_data, L_data, dedx_func, init=(1.0, 0.01, 0.0), use_C=False, bounds=((0, 0, 0), (np.inf, np.inf, np.inf)), prior_S=None, prior_sigma=None):
    """
    Fit (S, kB [, C]) by minimizing residuals on L(E).
    E_data in MeV (proton energy), L_data in MeVee.
    init: (S, kB[, C]) - use Joey's numbers as initial values
    bounds: ((Smin, kBmin, Cmin), (Smax, kBmax, Cmax))
    """
    E_data = np.asarray(E_data, dtype=float)
    L_data = np.asarray(L_data, dtype=float)
    E_grid = np.linspace(0.0, max(1.05*np.max(E_data), 20.0), 20001)  # ensure dense coverage

    def residuals(p):
        S, kB = p[0], p[1]
        C = p[2] if use_C else 0.0
        L_grid = light_integral_grid(E_grid, dedx_func, S, kB, C)
        L_model = np.interp(E_data, E_grid, L_grid)
        r = (L_model - L_data)
        if (prior_S is not None) and (prior_sigma is not None) and (prior_sigma > 0):
            r = np.concatenate([r, np.array([(S - prior_S)/prior_sigma])])
        return r

    p0 = np.array(init[:(3 if use_C else 2)], dtype=float)
    lower = np.array(bounds[0][:len(p0)], dtype=float)
    upper = np.array(bounds[1][:len(p0)], dtype=float)

    opt = least_squares(residuals, p0, bounds=(lower, upper), jac='2-point')
    # Covariance estimate from the Jacobian
    theta = opt.x
    r = opt.fun
    J = opt.jac            # (N x p) numerical Jacobian
    N = len(r)
    p = len(theta)
    dof = max(1, N - p)

    # SSE and residual variance
    SSE = float(np.dot(r, r))
    sigma2_hat = SSE / dof

    # Covariance via (J^T J)^(-1) scaled by sigma^2 (with SVD fallback)
    # Note: least_squares returns J at the solution for residuals, so J^T J is the GN Hessian approx.
    JTJ = J.T @ J
    try:
        cov_unscaled = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        # robust SVD-based pseudo-inverse if nearly singular
        U, s, VT = np.linalg.svd(J, full_matrices=False)
        cov_unscaled = VT.T @ np.diag(1.0 / (s**2)) @ VT
    cov = sigma2_hat * cov_unscaled

    # Standard errors
    se = np.sqrt(np.diag(cov))
    # 95% CIs using normal approx (or use t-dist if you prefer)
    ci95 = np.column_stack([theta - 1.96*se, theta + 1.96*se])

    # Simple goodness-of-fit metrics (unweighted)
    L_bar = float(np.mean(L_data))
    SST = float(np.sum((L_data - L_bar)**2))
    R2 = 1.0 - (SSE / SST) if SST > 0 else np.nan
    R2_adj = 1.0 - (1.0 - R2) * (N - 1) / dof
    RMSE = np.sqrt(sigma2_hat)
    '''
    # (Assumes residual variance ~ 1; rescale by sigma^2 if known)
    J = opt.jac
    _, s, VT = np.linalg.svd(J, full_matrices=False)
    threshold = np.finfo(float).eps * max(J.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:len(s)]
    cov = VT.T @ np.diag(1/s**2) @ VT
    '''
    # Pack results
    if use_C:
        S_hat, kB_hat, C_hat = opt.x
        seS, sekB, seC = se
        cov_full = cov
    else:
        S_hat, kB_hat = opt.x
        C_hat = 0.0
        #cov = np.pad(cov, ((0,1),(0,1)), mode='constant')
        seS, sekB = se
        seC = 0.0
        # pad covariance to 3x3 for uniform handling elsewhere
        cov_full = np.zeros((3,3), float); cov_full[:2,:2] = cov
        ciC = (0.0, 0.0)
        ci95 = np.vstack([ci95, [0.0, 0.0]])
    return dict(x=np.array([S_hat, kB_hat, C_hat]),
                success=opt.success,
                cost=opt.cost,
                message=opt.message,
                nfev=opt.nfev,
                se=np.array([seS, sekB, seC]),
                cov=cov_full,
                ci95=ci95,
                stats=dict(
                    N=N, p=p, dof=dof, SSE=SSE, RMSE=RMSE, R2=R2, R2_adj=R2_adj, sigma2=sigma2_hat
                ),
                )




# =======================================================================
# FUNCTIONS: UNCERTAINTY PROPAGATION (MONTE CARLO APPROACH)
# =======================================================================

def sample_params(mean, cov, n=64, nonneg=True):
    rng = np.random.default_rng(1337)
    draws = rng.multivariate_normal(mean, cov, size=n)
    if nonneg:
        draws = np.clip(draws, 0, None)
    return draws

def is_valid_params(S, kB, C, dedx_func, E_max=250.0, nE=20001, min_span=1e-6):
    E_grid = np.linspace(0.0, E_max, nE)
    L_grid = light_integral_grid(E_grid, dedx_func, S, kB, C)
    # monotonic & enough dynamic range
    if not np.all(np.isfinite(L_grid)):
        return False
    if np.ptp(L_grid) < min_span:  # essentially flat
        return False
    # strictly increasing (numerically)
    if not np.all(np.diff(L_grid) > 0):
        # allow tiny numerical plateaus, then enforce strictness
        d = np.diff(L_grid)
        if not np.all(d >= -1e-12) or np.count_nonzero(d <= 1e-12) > 0.01*len(d):
            return False
    return True

def make_inverse_sampler(dedx_func, params_samples, E_max=250.0, nE=125001):
    """
    Build many inverse interpolants E(L) for uncertainty bands.
    Returns a list of callables E_of_L_samplers.
    """
    samplers = []
    for (S, kB, C) in params_samples:
        if not is_valid_params(S, kB, C, dedx_func, E_max=E_max, nE=min(40001, nE)):
            continue  # skip pathological draws
        _, _, _, E_of_L = make_forward_inverse_LUT(dedx_func, S, kB, C, E_max, nE)
        samplers.append(E_of_L)
    return samplers

def inverse_with_bands(L_vals, E_of_L_central, E_of_L_samplers):
    """
    For each L, return median and central 68% interval of E across samplers.
    """
    L_vals = np.atleast_1d(L_vals)
    E_med = E_of_L_central(L_vals)
    if len(E_of_L_samplers) == 0:
        return E_med, None, None
    Es = np.vstack([s(L_vals) for s in E_of_L_samplers])
    lo, hi = np.nanpercentile(Es, [16, 84], axis=0)
    return E_med, lo, hi

def sample_params_posdef(mean, cov, n=100, floors=(1e-4, 1e-4, 0.0)):
    # Assume mean=[S,kB,C], cov small. Sample S,kB in log-space; C optional.
    import numpy as np
    rng = np.random.default_rng(1234)
    S_mu = np.log(max(mean[0], floors[0]))
    kB_mu = np.log(max(mean[1], floors[1]))
    S_sigma = 0.25  # tune to your covariance scale
    kB_sigma = 0.25
    S_draw = np.exp(rng.normal(S_mu, S_sigma, size=n))
    kB_draw = np.exp(rng.normal(kB_mu, kB_sigma, size=n))
    if len(mean) > 2 and mean[2] > 0:
        C_mu = np.log(max(mean[2], 1e-8))
        C_draw = np.exp(rng.normal(C_mu, 0.5, size=n))
    else:
        C_draw = np.zeros(n)
    return np.column_stack([S_draw, kB_draw, C_draw])

def compute_inverse_band(dedx_fun, params_samples, L_inv_ref, *, E_max=250.0, nE=125001):
    """
    Build 68% (16/84) percentile inverse E(L) on a *fixed* L grid (L_inv_ref)
    by sampling Birks params. Returns (E_inv_lo, E_inv_hi, E_inv_med).
    Any pathological samples (non-monotone L(E)) are skipped.
    """
    E_inv_samples = []
    E_grid = np.linspace(0.0, E_max, int(nE))
    for (S_s, kB_s, C_s) in params_samples:
        # Forward L(E) for this draw
        L_grid = light_integral_grid(E_grid, dedx_fun, S_s, kB_s, C_s)
        # Must be monotone to invert
        dL = np.diff(L_grid)
        if not np.all(np.isfinite(L_grid)):
            continue
        if np.ptp(L_grid) < 1e-9:
            continue
        # allow tiny plateaus; reject if too many
        if np.count_nonzero(dL <= 1e-12) > 0.01 * len(dL):
            continue
        # Inverse onto fixed L grid
        E_inv_s = np.interp(L_inv_ref, L_grid, E_grid)
        E_inv_samples.append(E_inv_s)

    if len(E_inv_samples) == 0:
        return None, None, None

    E_inv_samples = np.asarray(E_inv_samples, dtype=np.float64)
    E_inv_lo  = np.percentile(E_inv_samples, 16, axis=0).astype(np.float32)
    E_inv_hi  = np.percentile(E_inv_samples, 84, axis=0).astype(np.float32)
    E_inv_med = np.percentile(E_inv_samples, 50, axis=0).astype(np.float32)
    return E_inv_lo, E_inv_hi, E_inv_med


# =======================================================================
# FUNCTIONS: LIGHT YIELD INTEGRATION
# =======================================================================

def dL_from_step(dE, E_mid, dedx_fun, S, kB, C=0.0):
    # dE > 0 (energy lost), E_mid is the particle kinetic energy at step midpoint (MeV)
    lam = dedx_fun(E_mid)  # MeV/cm
    return S * dE / (1.0 + kB*lam + C*lam*lam)

def accumulate_light_from_steps(steps, dedx_fun, S, kB, C=0.0):
    """
    steps: iterable of dicts with keys {'dE', 'E_mid'} for a given recoil track
    returns total light for that track
    """
    return sum(dL_from_step(st['dE'], st['E_mid'], dedx_fun, S, kB, C) for st in steps)


class BirksLUT:
    def __init__(self, species, dedx_func, S, kB, C=0.0, E_max=250.0, nE=125001):
        self.species = species
        self.S = S; self.kB = kB; self.C = C
        self.E_max = E_max
        self.gridE, self.gridL, self.L_of_E, self.E_of_L = make_forward_inverse_LUT(dedx_func, S, kB, C, E_max, nE)
        self._dedx_func = dedx_func

    def E_from_L(self, L):
        return self.E_of_L(L)

    def L_from_E(self, E):
        return self.L_of_E(E)

    def as_arrays(self):
        return self.gridE, self.gridL


# =======================================================================
# FUNCTIONS: LUT (LOOK-UP TABLE) SAVING
# =======================================================================
def build_inverse_L_grid(E_fine, L_fine, nL=60001):
    """
    Make a uniformly spaced grid in L (monotone), then tabulate E(L) with np.interp.
    """
    L_min, L_max = float(L_fine[0]), float(L_fine[-1])
    L_inv = np.linspace(L_min, L_max, int(nL), dtype=np.float32)
    E_inv = np.interp(L_inv, L_fine, E_fine).astype(np.float32)
    return L_inv, E_inv

def save_lut_npz_csv(basepath, L_inv, E_inv, meta):
    """
    Saves:
      - basepath + ".npz"  (binary, fast)
      - basepath + ".csv"  (plaintext, two columns: L_inv,E_inv)
      - basepath + ".meta.json" (small JSON metadata, human-readable)

    Parameters
    ----------
    basepath : str | Path
        Path *without* extension (e.g., Path("results/lut_M600_proton")).
        The function appends .npz, .csv, and .meta.json automatically.
    L_inv, E_inv : array-like
        Inverse lookup arrays: L_inv (MeVee) and E_inv (MeV).
    meta : dict
        Metadata dictionary describing the LUT contents.
    """
    basepath = Path(basepath)
    basepath.parent.mkdir(parents=True, exist_ok=True)
    npz_path  = basepath.with_suffix(".npz")
    csv_path  = basepath.with_suffix(".csv")
    json_path = basepath.with_suffix(".meta.json")
    # NPZ
    np.savez(npz_path, L_inv=L_inv, E_inv=E_inv, meta=np.array([meta], dtype=object))
    # CSV (plaintext)
    arr = np.column_stack([L_inv, E_inv])
    header = "L_inv_MeVee,E_inv_MeV"
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="", fmt="%.8g")
    # JSON meta (plaintext)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_lut_npz(npz_path):
    pack = np.load(npz_path, allow_pickle=True)
    L_inv = pack["L_inv"].astype(np.float32)
    E_inv = pack["E_inv"].astype(np.float32)
    meta  = pack["meta"][0].item() if "meta" in pack else {}
    Lmin, Lmax = float(L_inv[0]), float(L_inv[-1])

    def Ep_from_L(L_meas, *, clip=True, fill_value=np.nan):
        L = np.asarray(L_meas, dtype=float)
        if clip:
            Lc = np.clip(L, Lmin, Lmax)
            return np.interp(Lc, L_inv, E_inv)
        out = np.interp(np.clip(L, Lmin, Lmax), L_inv, E_inv)
        out[L < Lmin] = fill_value
        out[L > Lmax] = fill_value
        return out

    return L_inv, E_inv, meta, Ep_from_L

# =======================================================================
# ANALYSIS AND LUT (LOOK-UP TABLE) GENERATION
# =======================================================================

LUT_and_fitting_results = {'M600':{}, 'OGS':{}}

for si, scint in enumerate(scint_mats):
    # 1) Load SRIM (protons)
    SRIM_H_dict = read_SRIM_output(SRIM_files[scint]['H'])
    E_p = SRIM_H_dict['E_MeV']
    dEdx_p_mass = SRIM_H_dict['dEdx_total']
    # Convert mass stopping power to linear stopping power
    dEdx_p = dEdx_p_mass * density[scint] # MeV/cm
    dedx_p_func = make_dedx_interpolant(E_p, dEdx_p)

    # 2) Calibration points (proton recoils)
    exp_response_dict = read_light_response_file(exp_light_response_files[scint])
    E_data = exp_response_dict['Ep_MeV']
    L_data = exp_response_dict['L_MeVee']

    # 3) Fit (start with Joey's values)
    S0, kB0 = Birks_params[scint]['S'], Birks_params[scint]['kB_linear']
    use_C = use_Chou_C_term  # try False first; switch True if residuals demand it (if mostly positive over 3ish MeV)
    bounds = ((0.1, 0.0001, 0.0), (5.0, 0.1, 0.01))  # widen/narrow as you see fit

    if lock_S_to_1: # Keeps S near 1 but lets tiny corrections through if QDC-to-MeVee mapping isn't perfectly linear.
        prior_S = 1.0
        prior_sigma = 0.02  # Set to 0 to hard lock S to 0.
    else:
        prior_S = None
        prior_sigma = None
    fit = fit_birks_params(E_data, L_data, dedx_p_func, init=(S0, kB0, 0.0), use_C=use_C, bounds=bounds, prior_S=prior_S, prior_sigma=prior_sigma)
    S_hat, kB_hat, C_hat = fit["x"]
    stats = fit["stats"]
    print(f"[{scint}] Best fit: S={S_hat:.6g}, kB={kB_hat:.6g}, C={C_hat:.3g} | "
          f"R^2={stats['R2']:.5f}, RMSE={stats['RMSE']:.4g}")
    if use_C:
        C_se = fit["se"][2]
        ci_lo, ci_hi = C_hat - 1.96*C_se, C_hat + 1.96*C_se
        C_significant = not (ci_lo <= 0 <= ci_hi)
        if not C_significant:
            print('WARNING: the 95% confidence interval for C includes 0, meaning Birks alone (rather than Birks-Chou) is likely more suitable.')

    # 4) Build fast forward (E->L) and inverse (L->E) LUTs up to 250 MeV
    lut_p = BirksLUT("proton", dedx_p_func, S_hat, kB_hat, C_hat, E_max=250.0, nE=125001)
    E_fine_p, L_fine_p = lut_p.as_arrays()

    # 4b) Build inverse L-grid for speed (uniform in L)
    L_inv_p, E_inv_p = build_inverse_L_grid(E_fine_p, L_fine_p, nL=60001)

    # Example: invert measured light to Ep quickly
    #L_meas = np.array([0.5, 1.0, 2.5, 4.0])  # MeVee
    #Ep_est = lut_p.E_from_L(L_meas)          # MeV
    #print("Ep:", Ep_est)

    # 5) Uncertainty bands for the inversion
    #    Build 68% band (16th/84th percentiles) by sampling params
    params_samples = sample_params_posdef(np.array([S_hat, kB_hat, C_hat]), fit["cov"], n=100)
    # --- Proton band on proton L-grid ---
    E_inv_lo_p, E_inv_hi_p, E_inv_med_p = compute_inverse_band(dedx_p_func, params_samples, L_inv_p, E_max=250.0, nE=125001)


    # 6) (Optional) Carbon branch in parallel
    SRIM_C_dict = read_SRIM_output(SRIM_files[scint]['C'])
    E_C = SRIM_C_dict['E_MeV']
    dEdx_C_mass = SRIM_C_dict['dEdx_total']
    dEdx_C = dEdx_C_mass * density[scint] # MeV/cm
    dedx_C_func = make_dedx_interpolant(E_C, dEdx_C)
    # Reuse same S,kB[,C]; if we later gather carbon-tagged data, can refit separately.
    lut_C = BirksLUT("carbon", dedx_C_func, S_hat, kB_hat, C_hat, E_max=250.0, nE=125001)
    E_fine_C, L_fine_C = lut_C.as_arrays()
    L_inv_C, E_inv_C = build_inverse_L_grid(E_fine_C, L_fine_C, nL=60001)

    # --- Carbon band on carbon L-grid ---
    E_inv_lo_C, E_inv_hi_C, E_inv_med_C = compute_inverse_band(dedx_C_func, params_samples, L_inv_C, E_max=250.0, nE=125001)

    # For each measured L, you can compute both Ep(L) and EC(L):
    #EC_est = lut_C.E_from_L(L_meas)
    #print("EC:", EC_est)

    # 7) Save portable LUTs (NPZ + CSV) with metadata
    timestamp = datetime.utcnow().isoformat() + "Z"
    meta_p = dict(
        scintillator=scint,
        species="proton",
        model="Birks-Chou" if use_C and C_hat != 0 else "Birks",
        params=dict(S=float(S_hat), kB_cm_per_MeV=float(kB_hat), C_cm2_per_MeV2=float(C_hat)),
        density_g_per_cm3=float(density[scint]),
        energy_max_MeV=250.0,
        n_points=len(L_inv_p),
        fit_stats=stats,
        created_utc=timestamp
    )
    meta_C = dict(meta_p, species="carbon")

    model_name_str = 'Birks-Chou' if use_C else 'Birks'
    base_p = base_path / '..' / '..' / 'data' / 'lut' / f"{scint}" / f"lut_{scint}_proton_{model_name_str}"
    base_C = base_path / '..' / '..' / 'data' / 'lut' / f"{scint}" / f"lut_{scint}_carbon_{model_name_str}"
    save_lut_npz_csv(base_p, L_inv_p, E_inv_p, meta_p)
    save_lut_npz_csv(base_C, L_inv_C, E_inv_C, meta_C)

    # 8) Example usage (fast) with np.interp
    L_meas = np.array([0.5, 1.0, 2.5, 4.0])  # MeVee
    Ep_est = np.interp(L_meas, L_inv_p, E_inv_p)
    EC_est = np.interp(L_meas, L_inv_C, E_inv_C)
    print(f"[{scint}] Ep_est={Ep_est}, EC_est={EC_est}")


    # 9) Store minimal results in-memory (arrays + params + stats)
    LUT_and_fitting_results[scint] = {
        'proton': {
            'E_fine':E_fine_p, 'L_fine':L_fine_p,  # forward grid E (MeV) and L(E) (MeVee)
            'L_inv': L_inv_p, 'E_inv': E_inv_p,  # inverse grid L (MeVee) and E(L) (MeV)
            'params': {'S': S_hat, 'kB_linear': kB_hat, 'C_linear': C_hat},
            'E_inv_lo':E_inv_lo_p, 'E_inv_hi':E_inv_hi_p, 'E_inv_med':E_inv_med_p,
            'fit_stats': stats
        },
        'carbon': {
            'E_fine':E_fine_C, 'L_fine':L_fine_C,  # forward grid
            'L_inv': L_inv_C, 'E_inv': E_inv_C,  # inverse grid
            'params': {'S': S_hat, 'kB_linear': kB_hat, 'C_linear': C_hat},
            'E_inv_lo':E_inv_lo_C, 'E_inv_hi':E_inv_hi_C, 'E_inv_med':E_inv_med_C,
        },
        'calib': {
            'E_data_MeV': E_data,
            'L_data_MeVee': L_data,
            'En_strs': exp_response_dict.get('En_strs'),
            'En_indices': exp_response_dict.get('En_indices'),
        },
        'fit': fit
    }

'''
    # Store results in a dictionary object
    LUT_and_fitting_results[scint] = {
        'LUT_p':lut_p,
        'E_data_MeV':E_data,
        'L_data_MeVee':L_data,
        'En_strs':exp_response_dict['En_strs'],
        'En_indices':exp_response_dict['En_indices'],
        'dedx_p_func':dedx_p_func,
        'fit':fit,
        'Birks_params':{
            'initial_guess':Birks_params[scint],
            'found':{'S':S_hat, 'kB':kB_hat*1000*density[scint], 'kB_linear':kB_hat, 'C':None, 'C_linear':C_hat},
        },
        'uncertainty':{
            'samplers':samplers
        }
    }
'''



# =======================================================================
# PLOTTING
# =======================================================================
if save_plots or show_plots:
    # Birks fit plots
    # initialize figure
    if use_Chou_C_term:
        title_str = "Birks-Chou light response curve fit"
    else:
        title_str = "Birks light response curve fit"
    x_label_str = 'Proton recoil energy $E_p$ [MeV]'
    y_label_str_main = 'Light output $L$ [MeVee]'
    y_label_str_residuals = 'Residual [%]'
    fig, ax = plt.subplots(
            2, 1, figsize=(4, 5), sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
            num=figi
        )
    xdata_lists = []
    ydata_lists = []
    xdata_residuals_lists = []
    ydata_residuals_lists = []
    data_labels = []
    color = []
    residuals_color = []
    alpha = []
    linestyle = []
    marker = []
    residuals_marker = []
    avail_markers = ['s', 'v', '+']

    inv_xdata_lists = []
    inv_ydata_lists = []
    inv_yerr_lists = []
    inv_data_labels = []
    inv_colors = []
    inv_ls = []
    inv_alpha = []

    for si, scint in enumerate(scint_mats):
        E_plot = np.linspace(0, 250, 2000)
        #L_model = LUT_and_fitting_results[scint]['LUT_p'].L_from_E(E_plot)
        #L_model_data = LUT_and_fitting_results[scint]['LUT_p'].L_from_E(LUT_and_fitting_results[scint]['E_data_MeV'])
        #residuals_pct = 100.0 * (L_model_data - LUT_and_fitting_results[scint]['L_data_MeVee']) / LUT_and_fitting_results[scint]['L_data_MeVee']
        E_fine = LUT_and_fitting_results[scint]['proton']['E_fine']
        L_fine = LUT_and_fitting_results[scint]['proton']['L_fine']

        # Forward model via fast linear interpolation
        L_model = np.interp(E_plot, E_fine, L_fine)

        E_data = LUT_and_fitting_results[scint]['calib']['E_data_MeV']
        L_data = LUT_and_fitting_results[scint]['calib']['L_data_MeVee']

        L_model_data = np.interp(E_data, E_fine, L_fine)
        residuals_pct = 100.0 * (L_model_data - L_data) / L_data

        # Birks fit
        xdata_lists.append(E_plot)
        ydata_lists.append(L_model)
        data_labels.append(scint + ' model fit')
        color.append(colors[scint])
        alpha.append(1)
        linestyle.append('-')
        marker.append('')

        # Measured data
        PTB_En_indices = LUT_and_fitting_results[scint]['calib']['En_indices']
        iE0 = PTB_En_indices==0
        iE1 = PTB_En_indices==1
        iE2 = PTB_En_indices==2
        for i, iE in enumerate([iE0, iE1, iE2]):
            xdata_residuals_lists.append(E_data[iE])
            ydata_residuals_lists.append(residuals_pct[iE])

            xdata_lists.append(E_data[iE])
            ydata_lists.append(L_data[iE])
            #data_labels.append(scint + rf' PTB data, $E_n=${LUT_and_fitting_results[scint]["En_strs"][i]}')
            if i==0:
                data_labels.append(scint + r' PTB data')
            else:
                data_labels.append('')
            color.append(colors[scint])
            alpha.append(1)
            linestyle.append('')
            marker.append(avail_markers[i])
            residuals_color.append(colors[scint])
            residuals_marker.append(avail_markers[i])

            if si==1: # add extra nan points for labels
                xdata_lists.append([np.nan, np.nan])
                ydata_lists.append([np.nan, np.nan])
                data_labels.append(rf' PTB: $E_n=${LUT_and_fitting_results[scint]["calib"]["En_strs"][i]}')
                color.append('k')
                alpha.append(1)
                linestyle.append('')
                marker.append(avail_markers[i])

        # Fit error band
        # PROTON
        L_inv = LUT_and_fitting_results[scint]['proton']['L_inv']
        E_inv = LUT_and_fitting_results[scint]['proton']['E_inv']
        E_inv_lo = LUT_and_fitting_results[scint]['proton'].get('E_inv_lo', None)
        E_inv_hi = LUT_and_fitting_results[scint]['proton'].get('E_inv_hi', None)
        L_grid = np.linspace(float(L_inv[0]), float(L_inv[-1]), 1000)
        E_med = np.interp(L_grid, L_inv, E_inv)
        inv_xdata_lists.append(L_grid)
        inv_ydata_lists.append(E_med)
        inv_err_pairs = []
        #for ierr in range(len(E_med)):
        #    inv_err_pairs.append([E_med[ierr]-E_lo[ierr], E_hi[ierr]-E_med[ierr]])
        #inv_yerr_lists.append(np.array(inv_err_pairs).T)
        #inv_data_labels.append(scint + r', fit with $1\sigma$ ($\approx$68%) confidence interval')
        if (E_inv_lo is not None) and (E_inv_hi is not None):
            E_lo = np.interp(L_grid, L_inv, E_inv_lo)
            E_hi = np.interp(L_grid, L_inv, E_inv_hi)
            inv_err_pairs = np.vstack([E_med - E_lo, E_hi - E_med])  # shape (2, N)
            inv_yerr_lists.append(inv_err_pairs)
            inv_data_labels.append(scint + r', protons, $1\sigma$ band')
            #inv_data_labels.append(scint + r', fit with $1\sigma$ ($\approx$68%) confidence interval')
        else:
            inv_yerr_lists.append(None)
            inv_data_labels.append(scint + r', protons')
        inv_colors.append(colors[scint])
        inv_ls.append('-')
        inv_alpha.append(1)

        # CARBON
        L_inv_C = LUT_and_fitting_results[scint]['carbon']['L_inv']
        E_inv_C = LUT_and_fitting_results[scint]['carbon']['E_inv']
        E_inv_lo_C = LUT_and_fitting_results[scint]['carbon'].get('E_inv_lo', None)
        E_inv_hi_C = LUT_and_fitting_results[scint]['carbon'].get('E_inv_hi', None)
        L_grid_C = np.linspace(float(L_inv_C[0]), float(L_inv_C[-1]), 1000)
        E_med_C  = np.interp(L_grid_C, L_inv_C, E_inv_C)
        inv_xdata_lists.append(L_grid_C)
        inv_ydata_lists.append(E_med_C)
        if (E_inv_lo_C is not None) and (E_inv_hi_C is not None):
            E_lo_C = np.interp(L_grid_C, L_inv_C, E_inv_lo_C)
            E_hi_C = np.interp(L_grid_C, L_inv_C, E_inv_hi_C)
            inv_err_pairs_C = np.vstack([E_med_C - E_lo_C, E_hi_C - E_med_C])
            inv_yerr_lists.append(inv_err_pairs_C)
            inv_data_labels.append(scint + r', carbon, $1\sigma$ band')
        else:
            inv_yerr_lists.append(None)
            inv_data_labels.append(scint + r', carbon')
        inv_colors.append(colors[scint])
        inv_ls.append(':')
        inv_alpha.append(1)

    plot_E_x_bounds = [0, 12]
    plot_L_y_bounds = [0, 10]

    fig_w_in, fig_h_in = 7, 8
    # Main model fit subplot
    fig, ax = fancy_plot(
                xdata_lists,
                ydata_lists,
                figi=figi,
                x_label_str='',
                y_label_str=y_label_str_main,
                data_labels=data_labels,
                title_str=title_str,
                x_scale='linear',
                y_scale='linear',
                x_limits=plot_E_x_bounds,
                y_limits=plot_L_y_bounds,
                color=color,
                alpha=alpha,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                legend_position='upper right',
                fig=fig,
                ax=ax,
                spnrows=2,
                spindex=1,
                legend_ncol=1,
                fig_width_inch=fig_w_in,
                fig_height_inch=fig_h_in
            )

    # Fit text
    # Equation
    birks_dx_tex = r"$\frac{dL}{dx} \;=\; S \,\frac{\,dE/dx\,}{1 + k_B\, (dE/dx)}$"
    birks_E_tex = r"$L(E) \;=\; \int_{0}^{E} \frac{S}{1 + k_B\,\Lambda(E')} \, dE', \quad \Lambda(E')=\Big(\frac{dE}{dx}\Big)(E')$"
    birkschou_dx_tex = r"$\frac{dL}{dx} \;=\; S \,\frac{\,dE/dx\,}{1 + k_B\, (dE/dx) + C\, (dE/dx)^2}$"
    birkschou_E_tex = r"$L(E) \;=\; \int_{0}^{E} \frac{S}{1 + k_B\,\Lambda(E') + C\,\Lambda(E')^{2}} \, dE', \quad \Lambda(E')=\Big(\frac{dE}{dx}\Big)(E')$"
    if use_Chou_C_term:
        text_on_plot = birkschou_dx_tex
    else:
        text_on_plot = birks_dx_tex
    images_pos_text_color = 'k'
    fs = 14
    slx, sly = 0.02, 0.98
    t = ax.text(slx, sly, text_on_plot, color=images_pos_text_color, horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, fontsize=fs)
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='#969696'))
    # Fit parameter values
    fs = 10
    for si, scint in enumerate(scint_mats):
        #S_hat  = LUT_and_fitting_results[scint]['Birks_params']['found']['S']
        #kB_hat = LUT_and_fitting_results[scint]['Birks_params']['found']['kB_linear']
        #C_hat  = LUT_and_fitting_results[scint]['Birks_params']['found']['C_linear']
        #S_err  = LUT_and_fitting_results[scint]['fit']['se'][0]
        #kB_err = LUT_and_fitting_results[scint]['fit']['se'][1]
        #C_err  = LUT_and_fitting_results[scint]['fit']['se'][2]
        # Best-fits
        p = LUT_and_fitting_results[scint]['proton']['params']
        S_hat, kB_hat, C_hat = p['S'], p['kB_linear'], p['C_linear']

        # Uncertainties (if you kept the fit dict)
        S_err, kB_err, C_err = (LUT_and_fitting_results[scint]['fit']['se'])

        # Fit metrics (R^2, RMSE, etc.)
        stats = LUT_and_fitting_results[scint]['proton']['fit_stats'] \
                if 'fit_stats' in LUT_and_fitting_results[scint]['proton'] \
                else LUT_and_fitting_results[scint]['fit']['stats']
        RMSE, R2, R2a = stats['RMSE'], stats['R2'], stats['R2_adj']
        #S_err, kB_err, C_err = np.sqrt(np.diag(LUT_and_fitting_results[scint]['fit']['cov']))  # beware: C_err=0 if you fixed C=0
        S_unit  = "MeVee/MeV"
        kB_unit = "cm/MeV"
        C_unit  = "(cm/MeV)$^2$"
        print(f"[{scint}] ", 'S = ',S_hat, '+/-',  S_err,  S_unit)
        print(f"[{scint}] ", 'kB = ', kB_hat, '+/-', kB_err, kB_unit)
        if use_Chou_C_term:
            param_box = (
                r"$\mathbf{{{:}}}$".format(scint) + r" $\mathbf{Fit\ Parameters}$" "\n"
                + r"$S =$ "  + pm_fmt(S_hat,  S_err,  S_unit)  + "\n"
                + r"$k_B =$ "+ pm_fmt(kB_hat, kB_err, kB_unit) + "\n"
                + r"$C =$ "  + pm_fmt(C_hat,  C_err,  C_unit) + "\n"
                + r"RMSE = {:.5f} MeVee".format(RMSE) + "\n"
                + r"$R^2$ = {:.4g} ".format(R2) + ", "
                + r"$R^2_a$ = {:.4g} ".format(R2a)
                )
        else:
            param_box = (
                r"$\mathbf{{{:}}}$".format(scint) + r" $\mathbf{Fit\ Parameters}$" "\n"
                + r"$S =$ "  + pm_fmt(S_hat,  S_err,  S_unit)  + "\n"
                + r"$k_B =$ "+ pm_fmt(kB_hat, kB_err, kB_unit) + "\n"
                + r"RMSE = {:.5f} MeVee".format(RMSE) + "\n"
                + r"$R^2$ = {:.4g} ".format(R2) + ", "
                + r"$R^2_a$ = {:.4g} ".format(R2a)
                )
        fs = 10
        slx = 0.02
        sly = 0.87 - (si*0.29)
        t = ax.text(slx, sly, param_box, color=images_pos_text_color, horizontalalignment='left',
                     verticalalignment='top', transform=ax.transAxes, fontsize=fs)
        t.set_bbox(dict(facecolor=colors[scint], alpha=0.2, edgecolor='#969696'))

    # Residuals subplot
    fig, ax = fancy_plot(
                xdata_residuals_lists,
                ydata_residuals_lists,
                figi=figi,
                x_label_str=x_label_str,
                y_label_str=y_label_str_residuals,
                title_str='',
                x_scale='linear',
                y_scale='linear',
                x_limits=plot_E_x_bounds,
                y_limits=[None, None],
                #y_limits=[-20, 20],
                color=residuals_color,
                #alpha=alpha,
                #linestyle=linestyle,
                marker=residuals_marker,
                markersize=4,
                #legend_position='right',
                fig=fig,
                ax=ax,
                spnrows=2,
                spindex=2,
                fig_width_inch=fig_w_in,
                fig_height_inch=fig_h_in
            )

    if save_plots:
        for ext in image_extensions:
            plot_filename = slugify(title_str) + ext  # or use fig.canvas.get_window_title()
            plot_save_path = Path.joinpath(images_path, plot_filename)
            fig.savefig(plot_save_path, facecolor=(0, 0, 0, 0))
        if not show_plots: plt.close(figi)
    figi += 1



    # Make plot showing uncertainty band of L to Ep function
    '''
    Interpretation:
    The shaded region represents the propagated 1σ (≈68%) confidence interval for the inferred proton recoil energy
    E(L) obtained by Monte Carlo sampling of the fitted Birks parameters from their covariance matrix.
    '''
    if use_Chou_C_term:
        title_str = r"Inverse light response function (Birks-Chou fit)"
    else:
        title_str = r"Inverse light response function (Birks fit)"
    title_str += '\n' + r'with $1\sigma$ ($\approx$68%) confidence interval'
    x_label_str = 'Light output $L$ [MeVee]'
    y_label_str = 'Recoil energy $E_{p|C}$ [MeV]'
    fig_w_in, fig_h_in = 6, 4
    plot_E_bounds = [0, 200]
    plot_L_bounds = [0, 150]
    fig, ax = fancy_plot(
                inv_xdata_lists,
                inv_ydata_lists,
                yerr_lists=inv_yerr_lists,
                errorstyle='band',
                figi=figi,
                x_label_str=x_label_str,
                y_label_str=y_label_str,
                data_labels=inv_data_labels,
                title_str=title_str,
                x_scale='linear',
                y_scale='linear',
                x_limits=plot_L_bounds,
                y_limits=plot_E_bounds,
                color=inv_colors,
                alpha=inv_alpha,
                linestyle=inv_ls,
                marker='',
                legend_position='lower right',
                fig_width_inch=fig_w_in,
                fig_height_inch=fig_h_in
            )
    if save_plots:
        for ext in image_extensions:
            plot_filename = slugify(title_str) + ext  # or use fig.canvas.get_window_title()
            plot_save_path = Path.joinpath(images_path, plot_filename)
            fig.savefig(plot_save_path, facecolor=(0, 0, 0, 0))
        #if not show_plots: plt.close(figi)

    # Save a separate version zoomed in to carbon curve
    if use_Chou_C_term and not lock_S_to_1:
        ax.set_xlim(0, 1)       # x-axis in MeVee
        ax.set_ylim(0, 80)      # y-axis in MeV
        ax.legend(loc='center right', frameon=True)
    else:
        ax.set_xlim(0, 3)       # x-axis in MeVee
        ax.set_ylim(0, 80)      # y-axis in MeV
        ax.legend(loc='center right', frameon=True)
    if save_plots:
        for ext in image_extensions:
            plot_filename = slugify(title_str) + '_C-zoomed' + ext  # or use fig.canvas.get_window_title()
            plot_save_path = Path.joinpath(images_path, plot_filename)
            fig.savefig(plot_save_path, facecolor=(0, 0, 0, 0))
        if not show_plots: plt.close(figi)

    if show_plots: # restore bounds
        ax.set_xlim(plot_L_bounds)       # x-axis in MeVee
        ax.set_ylim(plot_E_bounds)      # y-axis in MeV
        ax.legend(loc='lower right', frameon=True)


    figi += 1





if show_plots: plt.show()
