#!/usr/bin/env python3
"""
Figure-14-style z-chi_eff PySR analysis with GWTC-4 + PySR overlays
for BOTH Spline and Linear models, INCLUDING science diagnostics.

This script combines:
  1. The overlay plotting style of the user's working q-chi_eff scripts
  2. The science diagnostics from the earlier z-chi_eff analysis

Outputs:
  - overlay figures
  - gradient figures
  - per-model NPZ files
  - per-model fit summaries
  - per-model diagnostics
  - posterior diagnostic summaries
  - text report
  - cross-model summary

Expected files:
    BBHCorr_zchieffSplineCorrelationModel.h5
    BBHCorr_zchieffLinearCorrelationModel.h5
"""

from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from pysr import PySRRegressor

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

DATADIR = "."
OUTDIR = os.path.join(DATADIR, "outdir_zchieff_figure14_overlay_with_diagnostics")
os.makedirs(OUTDIR, exist_ok=True)

MODEL_FILES = {
    "Spline": os.path.join(DATADIR, "BBHCorr_zchieffSplineCorrelationModel.h5"),
    "Linear": os.path.join(DATADIR, "BBHCorr_zchieffLinearCorrelationModel.h5"),
}

RANDOM_SEED = 42
TEST_SIZE = 0.25
SIGMA_FLOOR = 1e-6

ZMIN = 0.0
ZMAX = 1.9

# windows for diagnostics
LOW_Z_WINDOW = (0.0, 0.4)
HIGH_Z_WINDOW = (1.0, 1.9)
Z0_GRAD = 1.0

# PySR settings
N_POSTERIOR_DRAWS_TO_FIT = 150
N_FIT_POINTS = 150

# Gradient smoothing for display
USE_SMOOTHING = True
SAVGOL_WINDOW = 41
SAVGOL_POLYORDER = 3

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 12,
    "figure.dpi": 150,
})

# Overlay plotting style: same logic as working q-chi_eff code
COLORS = {
    "Spline": "#1f77b4",
    "Linear": "#d62728",
}
GWTC4_LINESTYLE = "-"
PYSR_LINESTYLE = "--"
ALPHA_GWTC4 = 0.16
ALPHA_PYSR = 0.10

PYSR_KWARGS = dict(
    niterations=180,
    maxsize=20,
    populations=30,
    population_size=80,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["log", "exp", "sqrt", "square", "abs"],
    extra_sympy_mappings={"square": lambda x: x**2},
    model_selection="best",
    verbosity=0,
    deterministic=True,
    parallelism="serial",
    parsimony=0.001,
    turbo=True,
    precision=64,
    temp_equation_file=False,
    output_jax_format=False,
    output_torch_format=False,
)


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class FitSummary:
    model_name: str
    target_name: str
    equation: str
    complexity: float
    train_mse: float
    test_mse: float
    constant_test_mse: float
    linear_test_mse: float
    improvement_vs_constant: float
    improvement_vs_linear: float


@dataclass
class CurveDiagnostics:
    model_name: str
    target_name: str
    low_z_mean_slope: float
    high_z_mean_slope: float
    global_mean_abs_curvature: float
    zero_crossing_z: Optional[float]
    derivative_at_z1: float
    endpoint_lowz: float
    endpoint_highz: float


@dataclass
class PosteriorSummary:
    model_name: str
    target_name: str
    pr_increasing_low_z: float
    pr_decreasing_low_z: float
    pr_increasing_high_z: float
    pr_decreasing_high_z: float
    pr_zero_crossing: float
    median_zero_crossing_z: Optional[float]
    derivative_z1_median: float
    derivative_z1_p05: float
    derivative_z1_p95: float
    pr_narrows_toward_high_z: Optional[float]
    pr_broadens_toward_high_z: Optional[float]


# ============================================================
# Utilities
# ============================================================

def percentile_summary(draws: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.percentile(draws, 50, axis=0)
    lo = np.percentile(draws, 5, axis=0)
    hi = np.percentile(draws, 95, axis=0)
    return med, lo, hi


def sanitize_sigma(sig: np.ndarray, floor: float = SIGMA_FLOOR) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    sig = np.where(np.isfinite(sig), sig, np.nan)
    sig = np.where(sig > floor, sig, floor)
    return sig


def subsample_draws(draws: np.ndarray, n_keep: int, seed: int = RANDOM_SEED) -> np.ndarray:
    n_draws = draws.shape[0]
    if n_draws <= n_keep:
        return draws
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_draws, size=n_keep, replace=False)
    return draws[idx]


def trim_z_range(z_grid: np.ndarray, *arrays: np.ndarray, zmin=ZMIN, zmax=ZMAX):
    mask = (z_grid >= zmin) & (z_grid <= zmax)
    out = [z_grid[mask]]
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            out.append(arr[mask])
        elif arr.ndim == 2:
            out.append(arr[:, mask])
        else:
            raise ValueError("Unexpected ndim in trim_z_range.")
    return tuple(out)


def build_feature_matrix(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.column_stack([
        z,
        z**2,
        np.log(z + 1e-6),
        1 + z,
        np.log(1 + z),
        z * (1 + z),
        np.sqrt(z + 1e-6),
        np.abs(z - 1.0),
        (z - 1.0)**2,
    ])


def gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(y, x)


def second_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(np.gradient(y, x), x)


def maybe_smooth(y: np.ndarray, use_smoothing: bool = USE_SMOOTHING) -> np.ndarray:
    if not use_smoothing:
        return y
    n = len(y)
    window = min(SAVGOL_WINDOW, n if n % 2 == 1 else n - 1)
    if window < 5:
        return y
    if window % 2 == 0:
        window -= 1
    polyorder = min(SAVGOL_POLYORDER, window - 2)
    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")


def window_mask(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (x >= lo) & (x <= hi)


def safe_zero_crossing(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    signs = np.sign(y)
    idx = np.where(signs[:-1] * signs[1:] < 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    try:
        f = interp1d(x, y, kind="linear")
        return float(brentq(lambda zz: float(f(zz)), x[i], x[i + 1]))
    except Exception:
        return float(np.interp(0.0, [y[i], y[i + 1]], [x[i], x[i + 1]]))


# ============================================================
# Validation tests
# ============================================================

def run_basic_tests(model_name: str, z_grid: np.ndarray, mu_rates: np.ndarray, sigma_rates: np.ndarray):
    if z_grid.ndim != 1:
        raise ValueError(f"{model_name}: z_grid is not 1D.")
    if mu_rates.ndim != 2:
        raise ValueError(f"{model_name}: mu_rates is not 2D.")
    if sigma_rates.ndim != 2:
        raise ValueError(f"{model_name}: sigma_rates is not 2D.")
    if mu_rates.shape[1] != len(z_grid):
        raise ValueError(f"{model_name}: mu_rates shape incompatible with z_grid.")
    if sigma_rates.shape[1] != len(z_grid):
        raise ValueError(f"{model_name}: sigma_rates shape incompatible with z_grid.")
    if not np.all(np.isfinite(z_grid)):
        raise ValueError(f"{model_name}: z_grid contains non-finite values.")
    if not np.all(np.diff(z_grid) >= 0):
        raise ValueError(f"{model_name}: z_grid is not monotonic non-decreasing.")
    if not np.all(np.isfinite(mu_rates)):
        raise ValueError(f"{model_name}: mu_rates contains non-finite values.")
    if not np.all(np.isfinite(sigma_rates)):
        raise ValueError(f"{model_name}: sigma_rates contains non-finite values.")
    if np.any(sigma_rates <= 0):
        raise ValueError(f"{model_name}: sigma_rates contains non-positive values after sanitization.")


def test_prediction_shapes(model_name: str, z_grid: np.ndarray, y_pred: np.ndarray, quantity_name: str):
    if y_pred.ndim != 1:
        raise ValueError(f"{model_name} {quantity_name}: prediction is not 1D.")
    if len(y_pred) != len(z_grid):
        raise ValueError(f"{model_name} {quantity_name}: prediction length mismatch.")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError(f"{model_name} {quantity_name}: prediction contains non-finite values.")


def test_draw_prediction_array(model_name: str, arr: np.ndarray, quantity_name: str):
    if arr.ndim != 2:
        raise ValueError(f"{model_name} {quantity_name}: posterior prediction array is not 2D.")
    if arr.shape[0] == 0:
        raise ValueError(f"{model_name} {quantity_name}: no successful posterior draw fits.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{model_name} {quantity_name}: posterior predictions contain non-finite values.")


# ============================================================
# HDF5 loading
# ============================================================

def load_zchieff_file(h5_file: str) -> Dict[str, np.ndarray]:
    with h5py.File(h5_file, "r") as f:
        mu_pos = np.asarray(f["posterior/rates_on_grids/mu_chieff/positions"][0], dtype=float)
        mu_rates = np.asarray(f["posterior/rates_on_grids/mu_chieff/rates"][:], dtype=float)

        sigma_pos = np.asarray(f["posterior/rates_on_grids/sigma_chieff/positions"][0], dtype=float)
        sigma_rates = np.asarray(f["posterior/rates_on_grids/sigma_chieff/rates"][:], dtype=float)

    if len(mu_pos) != len(sigma_pos) or not np.allclose(mu_pos, sigma_pos):
        raise ValueError("mu_chieff and sigma_chieff positions do not match.")

    return {
        "z_grid": mu_pos,
        "mu_rates": mu_rates,
        "sigma_rates": sigma_rates,
    }


# ============================================================
# Baseline models and PySR fitting
# ============================================================

def fit_constant_baseline(y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    return np.full_like(x_test, np.mean(y_train), dtype=float)


def fit_linear_baseline(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    lr = LinearRegression()
    lr.fit(x_train.reshape(-1, 1), y_train)
    return lr.predict(x_test.reshape(-1, 1))


def fit_pysr_target(
    z_grid: np.ndarray,
    y_grid: np.ndarray,
    target_name: str,
    random_state: int,
    out_prefix: str,
) -> Tuple[PySRRegressor, FitSummary, np.ndarray]:
    idx = np.round(np.linspace(0, len(z_grid) - 1, N_FIT_POINTS)).astype(int)
    z_fit = z_grid[idx]
    y_fit = y_grid[idx]

    X = build_feature_matrix(z_fit)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_fit, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    model = PySRRegressor(
        random_state=random_state,
        **PYSR_KWARGS,
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    const_pred = fit_constant_baseline(y_train, X_test[:, 0])
    lin_pred = fit_linear_baseline(X_train[:, 0], y_train, X_test[:, 0])

    train_mse = mean_squared_error(y_train, pred_train)
    test_mse = mean_squared_error(y_test, pred_test)
    const_mse = mean_squared_error(y_test, const_pred)
    lin_mse = mean_squared_error(y_test, lin_pred)

    eqs = model.equations_.copy()
    best_row = eqs.sort_values("loss").iloc[0]
    equation = str(best_row["equation"])
    complexity = float(best_row["complexity"])

    summary = FitSummary(
        model_name="",
        target_name=target_name,
        equation=equation,
        complexity=complexity,
        train_mse=float(train_mse),
        test_mse=float(test_mse),
        constant_test_mse=float(const_mse),
        linear_test_mse=float(lin_mse),
        improvement_vs_constant=float((const_mse - test_mse) / const_mse if const_mse > 0 else np.nan),
        improvement_vs_linear=float((lin_mse - test_mse) / lin_mse if lin_mse > 0 else np.nan),
    )

    eqs_to_save = eqs.copy()
    for col in eqs_to_save.columns:
        if eqs_to_save[col].dtype == object:
            eqs_to_save[col] = eqs_to_save[col].astype(str)
    eqs_to_save.to_csv(f"{out_prefix}_equations.csv", index=False)

    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(asdict(summary), f, indent=2)

    X_full = build_feature_matrix(z_grid)
    y_pred_full = np.asarray(model.predict(X_full), dtype=float)

    return model, summary, y_pred_full


def fit_pysr_curve(
    z_grid: np.ndarray,
    y_grid: np.ndarray,
    random_state: int,
) -> Tuple[PySRRegressor, np.ndarray, pd.DataFrame]:
    idx = np.round(np.linspace(0, len(z_grid) - 1, N_FIT_POINTS)).astype(int)
    z_fit = z_grid[idx]
    y_fit = y_grid[idx]

    X_fit = build_feature_matrix(z_fit)
    model = PySRRegressor(
        random_state=random_state,
        **PYSR_KWARGS,
    )
    model.fit(X_fit, y_fit)

    X_full = build_feature_matrix(z_grid)
    y_pred = np.asarray(model.predict(X_full), dtype=float)

    eqs = model.equations_
    return model, y_pred, eqs


def fit_pysr_draws(
    z_grid: np.ndarray,
    draws: np.ndarray,
    random_state_base: int,
    quantity_name: str,
    model_name: str,
    outdir: str,
) -> np.ndarray:
    pred_list = []
    draw_dir = os.path.join(outdir, f"{model_name}_{quantity_name}_posterior_draws")
    os.makedirs(draw_dir, exist_ok=True)

    for i, y_draw in enumerate(draws):
        try:
            model, y_pred, eqs = fit_pysr_curve(
                z_grid=z_grid,
                y_grid=y_draw,
                random_state=random_state_base + i,
            )
            pred_list.append(y_pred)

            if eqs is not None:
                eqs_to_save = eqs.copy()
                for col in eqs_to_save.columns:
                    if eqs_to_save[col].dtype == object:
                        eqs_to_save[col] = eqs_to_save[col].astype(str)
                eqs_to_save.to_csv(os.path.join(draw_dir, f"draw_{i:04d}_equations.csv"), index=False)

            print(f"  {model_name} {quantity_name}: fitted draw {i+1}/{len(draws)}")
        except Exception as e:
            print(f"  [WARN] {model_name} {quantity_name} draw {i} failed: {e}")

    if len(pred_list) == 0:
        raise RuntimeError(f"No successful PySR draw fits for {model_name} {quantity_name}")

    return np.stack(pred_list, axis=0)


# ============================================================
# Diagnostics
# ============================================================

def curve_diagnostics(model_name: str, target_name: str, z: np.ndarray, y: np.ndarray) -> CurveDiagnostics:
    dy = gradient(y, z)
    d2y = second_derivative(y, z)

    low_mask = window_mask(z, *LOW_Z_WINDOW)
    high_mask = window_mask(z, *HIGH_Z_WINDOW)

    return CurveDiagnostics(
        model_name=model_name,
        target_name=target_name,
        low_z_mean_slope=float(np.mean(dy[low_mask])) if np.any(low_mask) else np.nan,
        high_z_mean_slope=float(np.mean(dy[high_mask])) if np.any(high_mask) else np.nan,
        global_mean_abs_curvature=float(np.mean(np.abs(d2y))),
        zero_crossing_z=safe_zero_crossing(z, y),
        derivative_at_z1=float(np.interp(Z0_GRAD, z, dy)),
        endpoint_lowz=float(y[0]),
        endpoint_highz=float(y[-1]),
    )


def summarize_posterior(model_name: str, target_name: str, z: np.ndarray, pred_draws: np.ndarray) -> PosteriorSummary:
    inc_low = []
    dec_low = []
    inc_high = []
    dec_high = []
    zero_cross = []
    deriv_z1 = []

    for y in pred_draws:
        dy = gradient(y, z)

        low_mask = window_mask(z, *LOW_Z_WINDOW)
        high_mask = window_mask(z, *HIGH_Z_WINDOW)

        inc_low.append(np.mean(dy[low_mask] > 0) if np.any(low_mask) else np.nan)
        dec_low.append(np.mean(dy[low_mask] < 0) if np.any(low_mask) else np.nan)
        inc_high.append(np.mean(dy[high_mask] > 0) if np.any(high_mask) else np.nan)
        dec_high.append(np.mean(dy[high_mask] < 0) if np.any(high_mask) else np.nan)

        zc = safe_zero_crossing(z, y)
        zero_cross.append(zc)
        deriv_z1.append(float(np.interp(Z0_GRAD, z, dy)))

    valid_zc = np.array([zz for zz in zero_cross if zz is not None], dtype=float)

    pr_narrow = None
    pr_broad = None
    if target_name.lower().startswith("logsigma"):
        pr_narrow = float(np.mean(np.array(deriv_z1) < 0))
        pr_broad = float(np.mean(np.array(deriv_z1) > 0))

    return PosteriorSummary(
        model_name=model_name,
        target_name=target_name,
        pr_increasing_low_z=float(np.nanmean(inc_low)),
        pr_decreasing_low_z=float(np.nanmean(dec_low)),
        pr_increasing_high_z=float(np.nanmean(inc_high)),
        pr_decreasing_high_z=float(np.nanmean(dec_high)),
        pr_zero_crossing=float(np.mean([zz is not None for zz in zero_cross])),
        median_zero_crossing_z=float(np.median(valid_zc)) if len(valid_zc) else None,
        derivative_z1_median=float(np.median(deriv_z1)),
        derivative_z1_p05=float(np.quantile(deriv_z1, 0.05)),
        derivative_z1_p95=float(np.quantile(deriv_z1, 0.95)),
        pr_narrows_toward_high_z=pr_narrow,
        pr_broadens_toward_high_z=pr_broad,
    )


# ============================================================
# Plotting
# ============================================================

def plot_top_middle_panels(results: Dict[str, Dict[str, np.ndarray]], outpath: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # mu(z)
    ax = axes[0]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        z = r["z_grid"]
        c = COLORS[model_name]

        ax.fill_between(z, r["mu_gwtc4_lo"], r["mu_gwtc4_hi"], color=c, alpha=ALPHA_GWTC4)
        ax.fill_between(z, r["mu_pysr_band_lo"], r["mu_pysr_band_hi"], color=c, alpha=ALPHA_PYSR)

        ax.plot(z, r["mu_gwtc4_med"], color=c, lw=2.2, ls=GWTC4_LINESTYLE,
                label=f"{model_name} GWTC-4 median")
        ax.plot(z, r["mu_pysr_med"], color=c, lw=2.0, ls=PYSR_LINESTYLE,
                label=f"{model_name} PySR median")

    ax.axhline(0, color="k", lw=1, ls=":", alpha=0.7)
    ax.set_xlim(ZMIN, ZMAX)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\mu_{\chi_\mathrm{eff}}(z)$")
    ax.set_title(r"Mean effective spin vs redshift")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    # sigma(z)
    ax = axes[1]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        z = r["z_grid"]
        c = COLORS[model_name]

        ax.fill_between(z, r["sigma_gwtc4_lo"], r["sigma_gwtc4_hi"], color=c, alpha=ALPHA_GWTC4)
        ax.fill_between(z, r["sigma_pysr_band_lo"], r["sigma_pysr_band_hi"], color=c, alpha=ALPHA_PYSR)

        ax.plot(z, r["sigma_gwtc4_med"], color=c, lw=2.2, ls=GWTC4_LINESTYLE,
                label=f"{model_name} GWTC-4 median")
        ax.plot(z, r["sigma_pysr_med"], color=c, lw=2.0, ls=PYSR_LINESTYLE,
                label=f"{model_name} PySR median")

    ax.set_xlim(ZMIN, ZMAX)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\sigma_{\chi_\mathrm{eff}}(z)$")
    ax.set_title(r"Width of effective-spin distribution vs redshift")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_gradient_panels(results: Dict[str, Dict[str, np.ndarray]], outpath: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # dmu/dz
    ax = axes[0]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        z = r["z_grid"]
        c = COLORS[model_name]

        ax.fill_between(z, r["dmu_gwtc4_lo_s"], r["dmu_gwtc4_hi_s"], color=c, alpha=ALPHA_GWTC4)
        ax.fill_between(z, r["dmu_pysr_band_lo_s"], r["dmu_pysr_band_hi_s"], color=c, alpha=ALPHA_PYSR)

        ax.plot(z, r["dmu_gwtc4_med_s"], color=c, lw=2.2, ls=GWTC4_LINESTYLE,
                label=f"{model_name} GWTC-4 median")
        ax.plot(z, r["dmu_pysr_med_s"], color=c, lw=2.0, ls=PYSR_LINESTYLE,
                label=f"{model_name} PySR median")

    ax.axhline(0, color="k", lw=1, ls=":", alpha=0.7)
    ax.set_xlim(ZMIN, ZMAX)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$d\mu_{\chi_\mathrm{eff}}/dz$")
    ax.set_title(r"Gradient of mean effective spin")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    # dlnsigma/dz
    ax = axes[1]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        z = r["z_grid"]
        c = COLORS[model_name]

        ax.fill_between(z, r["dlogsigma_gwtc4_lo_s"], r["dlogsigma_gwtc4_hi_s"], color=c, alpha=ALPHA_GWTC4)
        ax.fill_between(z, r["dlogsigma_pysr_band_lo_s"], r["dlogsigma_pysr_band_hi_s"], color=c, alpha=ALPHA_PYSR)

        ax.plot(z, r["dlogsigma_gwtc4_med_s"], color=c, lw=2.2, ls=GWTC4_LINESTYLE,
                label=f"{model_name} GWTC-4 median")
        ax.plot(z, r["dlogsigma_pysr_med_s"], color=c, lw=2.0, ls=PYSR_LINESTYLE,
                label=f"{model_name} PySR median")

    ax.axhline(0, color="k", lw=1, ls=":", alpha=0.7)
    ax.set_xlim(ZMIN, ZMAX)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$d\ln\sigma_{\chi_\mathrm{eff}}/dz$")
    ax.set_title(r"Gradient of log width")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# ============================================================
# Per-model analysis
# ============================================================

def analyze_model(model_name: str, h5_file: str) -> Dict[str, np.ndarray]:
    print("\n" + "=" * 80)
    print(f"Running model: {model_name}")
    print("=" * 80)

    data = load_zchieff_file(h5_file)

    z_grid, mu_rates, sigma_rates = trim_z_range(
        data["z_grid"],
        data["mu_rates"],
        sanitize_sigma(data["sigma_rates"]),
    )

    run_basic_tests(model_name, z_grid, mu_rates, sigma_rates)

    # GWTC-4 summaries
    mu_gwtc4_med, mu_gwtc4_lo, mu_gwtc4_hi = percentile_summary(mu_rates)
    sigma_gwtc4_med, sigma_gwtc4_lo, sigma_gwtc4_hi = percentile_summary(sigma_rates)

    logsigma_rates = np.log(sigma_rates)
    logsigma_gwtc4_med, logsigma_gwtc4_lo, logsigma_gwtc4_hi = percentile_summary(logsigma_rates)

    model_outdir = os.path.join(OUTDIR, model_name)
    os.makedirs(model_outdir, exist_ok=True)

    # Median symbolic fits with null-model diagnostics
    mu_model_fit, mu_fit_summary, mu_pysr_med = fit_pysr_target(
        z_grid=z_grid,
        y_grid=mu_gwtc4_med,
        target_name="mu_chieff_median",
        random_state=42 if model_name == "Spline" else 142,
        out_prefix=os.path.join(model_outdir, "mu_chieff_median"),
    )
    mu_fit_summary.model_name = model_name

    logsigma_model_fit, logsigma_fit_summary, logsigma_pysr_med = fit_pysr_target(
        z_grid=z_grid,
        y_grid=logsigma_gwtc4_med,
        target_name="logsigma_chieff_median",
        random_state=77 if model_name == "Spline" else 177,
        out_prefix=os.path.join(model_outdir, "logsigma_chieff_median"),
    )
    logsigma_fit_summary.model_name = model_name

    sigma_pysr_med = np.exp(logsigma_pysr_med)

    test_prediction_shapes(model_name, z_grid, mu_pysr_med, "mu median PySR")
    test_prediction_shapes(model_name, z_grid, logsigma_pysr_med, "logsigma median PySR")
    test_prediction_shapes(model_name, z_grid, sigma_pysr_med, "sigma median PySR")

    # Also keep raw equations tables in same style as overlay code
    mu_model_simple, _, mu_eqs = fit_pysr_curve(
        z_grid=z_grid,
        y_grid=mu_gwtc4_med,
        random_state=RANDOM_SEED + (0 if model_name == "Spline" else 100),
    )
    logsigma_model_simple, _, logsigma_eqs = fit_pysr_curve(
        z_grid=z_grid,
        y_grid=logsigma_gwtc4_med,
        random_state=RANDOM_SEED + (1 if model_name == "Spline" else 101),
    )

    mu_eqs_to_save = mu_eqs.copy()
    logsigma_eqs_to_save = logsigma_eqs.copy()
    for df in [mu_eqs_to_save, logsigma_eqs_to_save]:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)
    mu_eqs_to_save.to_csv(os.path.join(model_outdir, "mu_median_equations.csv"), index=False)
    logsigma_eqs_to_save.to_csv(os.path.join(model_outdir, "logsigma_median_equations.csv"), index=False)

    # Median-curve diagnostics
    mu_diag = curve_diagnostics(model_name, "mu_chieff", z_grid, mu_pysr_med)
    logsigma_diag = curve_diagnostics(model_name, "logsigma_chieff", z_grid, logsigma_pysr_med)

    # Posterior-draw symbolic fits
    mu_draws_sub = subsample_draws(mu_rates, N_POSTERIOR_DRAWS_TO_FIT, seed=RANDOM_SEED + 11)
    logsigma_draws_sub = subsample_draws(logsigma_rates, N_POSTERIOR_DRAWS_TO_FIT, seed=RANDOM_SEED + 22)

    mu_pysr_draw_preds = fit_pysr_draws(
        z_grid=z_grid,
        draws=mu_draws_sub,
        random_state_base=1000 if model_name == "Spline" else 2000,
        quantity_name="mu_chieff",
        model_name=model_name,
        outdir=OUTDIR,
    )
    logsigma_pysr_draw_preds = fit_pysr_draws(
        z_grid=z_grid,
        draws=logsigma_draws_sub,
        random_state_base=3000 if model_name == "Spline" else 4000,
        quantity_name="logsigma_chieff",
        model_name=model_name,
        outdir=OUTDIR,
    )
    sigma_pysr_draw_preds = np.exp(logsigma_pysr_draw_preds)

    test_draw_prediction_array(model_name, mu_pysr_draw_preds, "mu posterior PySR")
    test_draw_prediction_array(model_name, logsigma_pysr_draw_preds, "logsigma posterior PySR")
    test_draw_prediction_array(model_name, sigma_pysr_draw_preds, "sigma posterior PySR")

    # Posterior summaries from symbolic fits
    mu_post = summarize_posterior(model_name, "mu_chieff", z_grid, mu_pysr_draw_preds)
    logsigma_post = summarize_posterior(model_name, "logsigma_chieff", z_grid, logsigma_pysr_draw_preds)

    # PySR credible bands
    mu_pysr_band_med, mu_pysr_band_lo, mu_pysr_band_hi = percentile_summary(mu_pysr_draw_preds)
    sigma_pysr_band_med, sigma_pysr_band_lo, sigma_pysr_band_hi = percentile_summary(sigma_pysr_draw_preds)
    logsigma_pysr_band_med, logsigma_pysr_band_lo, logsigma_pysr_band_hi = percentile_summary(logsigma_pysr_draw_preds)

    # Gradient bands
    # GWTC-4: differentiate each posterior draw first, then summarize
    dmu_gwtc4_draws = np.array([gradient(y, z_grid) for y in mu_rates])
    dlogsigma_gwtc4_draws = np.array([gradient(y, z_grid) for y in logsigma_rates])

    dmu_gwtc4_med, dmu_gwtc4_lo, dmu_gwtc4_hi = percentile_summary(dmu_gwtc4_draws)
    dlogsigma_gwtc4_med, dlogsigma_gwtc4_lo, dlogsigma_gwtc4_hi = percentile_summary(dlogsigma_gwtc4_draws)

    # PySR: differentiate each symbolic posterior prediction first, then summarize
    dmu_pysr_draws = np.array([gradient(y, z_grid) for y in mu_pysr_draw_preds])
    dlogsigma_pysr_draws = np.array([gradient(y, z_grid) for y in logsigma_pysr_draw_preds])

    dmu_pysr_med = gradient(mu_pysr_med, z_grid)
    dlogsigma_pysr_med = gradient(logsigma_pysr_med, z_grid)

    dmu_pysr_band_med, dmu_pysr_band_lo, dmu_pysr_band_hi = percentile_summary(dmu_pysr_draws)
    dlogsigma_pysr_band_med, dlogsigma_pysr_band_lo, dlogsigma_pysr_band_hi = percentile_summary(dlogsigma_pysr_draws)

    # Smoothed versions for plotting only
    dmu_gwtc4_med_s = maybe_smooth(dmu_gwtc4_med)
    dmu_gwtc4_lo_s = maybe_smooth(dmu_gwtc4_lo)
    dmu_gwtc4_hi_s = maybe_smooth(dmu_gwtc4_hi)

    dlogsigma_gwtc4_med_s = maybe_smooth(dlogsigma_gwtc4_med)
    dlogsigma_gwtc4_lo_s = maybe_smooth(dlogsigma_gwtc4_lo)
    dlogsigma_gwtc4_hi_s = maybe_smooth(dlogsigma_gwtc4_hi)

    dmu_pysr_med_s = maybe_smooth(dmu_pysr_med)
    dmu_pysr_band_lo_s = maybe_smooth(dmu_pysr_band_lo)
    dmu_pysr_band_hi_s = maybe_smooth(dmu_pysr_band_hi)

    dlogsigma_pysr_med_s = maybe_smooth(dlogsigma_pysr_med)
    dlogsigma_pysr_band_lo_s = maybe_smooth(dlogsigma_pysr_band_lo)
    dlogsigma_pysr_band_hi_s = maybe_smooth(dlogsigma_pysr_band_hi)

    # Save summary tables
    pd.DataFrame([asdict(mu_fit_summary), asdict(logsigma_fit_summary)]).to_csv(
        os.path.join(model_outdir, "median_fit_summaries.csv"),
        index=False,
    )
    pd.DataFrame([asdict(mu_diag), asdict(logsigma_diag)]).to_csv(
        os.path.join(model_outdir, "median_curve_diagnostics.csv"),
        index=False,
    )
    pd.DataFrame([asdict(mu_post), asdict(logsigma_post)]).to_csv(
        os.path.join(model_outdir, "posterior_diagnostic_summaries.csv"),
        index=False,
    )

    # Save arrays
    np.savez(
        os.path.join(model_outdir, "results.npz"),
        z_grid=z_grid,

        mu_gwtc4_med=mu_gwtc4_med,
        mu_gwtc4_lo=mu_gwtc4_lo,
        mu_gwtc4_hi=mu_gwtc4_hi,

        sigma_gwtc4_med=sigma_gwtc4_med,
        sigma_gwtc4_lo=sigma_gwtc4_lo,
        sigma_gwtc4_hi=sigma_gwtc4_hi,

        logsigma_gwtc4_med=logsigma_gwtc4_med,
        logsigma_gwtc4_lo=logsigma_gwtc4_lo,
        logsigma_gwtc4_hi=logsigma_gwtc4_hi,

        mu_pysr_med=mu_pysr_med,
        mu_pysr_band_med=mu_pysr_band_med,
        mu_pysr_band_lo=mu_pysr_band_lo,
        mu_pysr_band_hi=mu_pysr_band_hi,

        sigma_pysr_med=sigma_pysr_med,
        sigma_pysr_band_med=sigma_pysr_band_med,
        sigma_pysr_band_lo=sigma_pysr_band_lo,
        sigma_pysr_band_hi=sigma_pysr_band_hi,

        logsigma_pysr_med=logsigma_pysr_med,
        logsigma_pysr_band_med=logsigma_pysr_band_med,
        logsigma_pysr_band_lo=logsigma_pysr_band_lo,
        logsigma_pysr_band_hi=logsigma_pysr_band_hi,

        dmu_gwtc4_med=dmu_gwtc4_med,
        dmu_gwtc4_lo=dmu_gwtc4_lo,
        dmu_gwtc4_hi=dmu_gwtc4_hi,

        dlogsigma_gwtc4_med=dlogsigma_gwtc4_med,
        dlogsigma_gwtc4_lo=dlogsigma_gwtc4_lo,
        dlogsigma_gwtc4_hi=dlogsigma_gwtc4_hi,

        dmu_pysr_med=dmu_pysr_med,
        dmu_pysr_band_med=dmu_pysr_band_med,
        dmu_pysr_band_lo=dmu_pysr_band_lo,
        dmu_pysr_band_hi=dmu_pysr_band_hi,

        dlogsigma_pysr_med=dlogsigma_pysr_med,
        dlogsigma_pysr_band_med=dlogsigma_pysr_band_med,
        dlogsigma_pysr_band_lo=dlogsigma_pysr_band_lo,
        dlogsigma_pysr_band_hi=dlogsigma_pysr_band_hi,
    )

    best_mu_eq = str(mu_eqs.sort_values("loss").iloc[0]["equation"])
    best_logsigma_eq = str(logsigma_eqs.sort_values("loss").iloc[0]["equation"])

    with open(os.path.join(model_outdir, "best_equations.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"mu_chieff(z) = {best_mu_eq}\n")
        f.write(f"log sigma_chieff(z) = {best_logsigma_eq}\n")

    # Human-readable report
    report = []
    report.append(f"Model: {model_name}")
    report.append("")
    report.append("Best symbolic equations")
    report.append(f"  mu_chieff(z) = {best_mu_eq}")
    report.append(f"  log sigma_chieff(z) = {best_logsigma_eq}")
    report.append("")
    report.append("Median-curve diagnostics")
    report.append(
        f"  mu_chieff: low-z slope={mu_diag.low_z_mean_slope:.6g}, "
        f"high-z slope={mu_diag.high_z_mean_slope:.6g}, "
        f"zero crossing={mu_diag.zero_crossing_z}, "
        f"dmu/dz at z=1.0 = {mu_diag.derivative_at_z1:.6g}"
    )
    report.append(
        f"  log sigma_chieff: low-z slope={logsigma_diag.low_z_mean_slope:.6g}, "
        f"high-z slope={logsigma_diag.high_z_mean_slope:.6g}, "
        f"dlnsigma/dz at z=1.0 = {logsigma_diag.derivative_at_z1:.6g}"
    )
    report.append("")
    report.append("Posterior symbolic summaries")
    report.append(f"  Pr[dmu/dz > 0 on low-z window]  = {mu_post.pr_increasing_low_z:.4f}")
    report.append(f"  Pr[dmu/dz < 0 on low-z window]  = {mu_post.pr_decreasing_low_z:.4f}")
    report.append(f"  Pr[dmu/dz > 0 on high-z window] = {mu_post.pr_increasing_high_z:.4f}")
    report.append(f"  Pr[dmu/dz < 0 on high-z window] = {mu_post.pr_decreasing_high_z:.4f}")
    report.append(f"  Pr[mu(z) crosses zero]          = {mu_post.pr_zero_crossing:.4f}")
    report.append(f"  median zero-crossing z          = {mu_post.median_zero_crossing_z}")
    report.append(
        f"  dmu/dz at z=1.0: median={mu_post.derivative_z1_median:.6g}, "
        f"90% interval=[{mu_post.derivative_z1_p05:.6g}, {mu_post.derivative_z1_p95:.6g}]"
    )
    report.append(
        f"  dlnsigma/dz at z=1.0: median={logsigma_post.derivative_z1_median:.6g}, "
        f"90% interval=[{logsigma_post.derivative_z1_p05:.6g}, {logsigma_post.derivative_z1_p95:.6g}]"
    )
    report.append(f"  Pr[narrows toward high z]       = {logsigma_post.pr_narrows_toward_high_z:.4f}")
    report.append(f"  Pr[broadens toward high z]      = {logsigma_post.pr_broadens_toward_high_z:.4f}")

    with open(os.path.join(model_outdir, "report.txt"), "w") as f:
        f.write("\n".join(report))

    print("\n".join(report))

    return {
        "z_grid": z_grid,

        "mu_gwtc4_med": mu_gwtc4_med,
        "mu_gwtc4_lo": mu_gwtc4_lo,
        "mu_gwtc4_hi": mu_gwtc4_hi,

        "sigma_gwtc4_med": sigma_gwtc4_med,
        "sigma_gwtc4_lo": sigma_gwtc4_lo,
        "sigma_gwtc4_hi": sigma_gwtc4_hi,

        "logsigma_gwtc4_med": logsigma_gwtc4_med,
        "logsigma_gwtc4_lo": logsigma_gwtc4_lo,
        "logsigma_gwtc4_hi": logsigma_gwtc4_hi,

        "mu_pysr_med": mu_pysr_med,
        "mu_pysr_band_med": mu_pysr_band_med,
        "mu_pysr_band_lo": mu_pysr_band_lo,
        "mu_pysr_band_hi": mu_pysr_band_hi,

        "sigma_pysr_med": sigma_pysr_med,
        "sigma_pysr_band_med": sigma_pysr_band_med,
        "sigma_pysr_band_lo": sigma_pysr_band_lo,
        "sigma_pysr_band_hi": sigma_pysr_band_hi,

        "logsigma_pysr_med": logsigma_pysr_med,
        "logsigma_pysr_band_med": logsigma_pysr_band_med,
        "logsigma_pysr_band_lo": logsigma_pysr_band_lo,
        "logsigma_pysr_band_hi": logsigma_pysr_band_hi,

        "dmu_gwtc4_med_s": dmu_gwtc4_med_s,
        "dmu_gwtc4_lo_s": dmu_gwtc4_lo_s,
        "dmu_gwtc4_hi_s": dmu_gwtc4_hi_s,

        "dlogsigma_gwtc4_med_s": dlogsigma_gwtc4_med_s,
        "dlogsigma_gwtc4_lo_s": dlogsigma_gwtc4_lo_s,
        "dlogsigma_gwtc4_hi_s": dlogsigma_gwtc4_hi_s,

        "dmu_pysr_med_s": dmu_pysr_med_s,
        "dmu_pysr_band_lo_s": dmu_pysr_band_lo_s,
        "dmu_pysr_band_hi_s": dmu_pysr_band_hi_s,

        "dlogsigma_pysr_med_s": dlogsigma_pysr_med_s,
        "dlogsigma_pysr_band_lo_s": dlogsigma_pysr_band_lo_s,
        "dlogsigma_pysr_band_hi_s": dlogsigma_pysr_band_hi_s,
    }


# ============================================================
# Cross-model summary
# ============================================================

def build_cross_model_summary():
    rows = []

    for model_name in MODEL_FILES:
        model_outdir = os.path.join(OUTDIR, model_name)
        fit_csv = os.path.join(model_outdir, "median_fit_summaries.csv")
        diag_csv = os.path.join(model_outdir, "median_curve_diagnostics.csv")
        post_csv = os.path.join(model_outdir, "posterior_diagnostic_summaries.csv")

        if not (os.path.exists(fit_csv) and os.path.exists(diag_csv) and os.path.exists(post_csv)):
            continue

        fit_df = pd.read_csv(fit_csv)
        diag_df = pd.read_csv(diag_csv)
        post_df = pd.read_csv(post_csv)

        for target in ["mu_chieff", "logsigma_chieff"]:
            fit_target = "mu_chieff_median" if target == "mu_chieff" else "logsigma_chieff_median"

            fit_row = fit_df[fit_df["target_name"] == fit_target].iloc[0]
            diag_row = diag_df[diag_df["target_name"] == target].iloc[0]
            post_row = post_df[post_df["target_name"] == target].iloc[0]

            rows.append({
                "model_name": model_name,
                "target_name": target,
                "equation": fit_row["equation"],
                "test_mse": fit_row["test_mse"],
                "improvement_vs_constant": fit_row["improvement_vs_constant"],
                "improvement_vs_linear": fit_row["improvement_vs_linear"],
                "low_z_mean_slope": diag_row["low_z_mean_slope"],
                "high_z_mean_slope": diag_row["high_z_mean_slope"],
                "derivative_at_z1": diag_row["derivative_at_z1"],
                "pr_increasing_low_z": post_row["pr_increasing_low_z"],
                "pr_decreasing_low_z": post_row["pr_decreasing_low_z"],
                "pr_increasing_high_z": post_row["pr_increasing_high_z"],
                "pr_decreasing_high_z": post_row["pr_decreasing_high_z"],
                "pr_zero_crossing": post_row["pr_zero_crossing"],
                "median_zero_crossing_z": post_row["median_zero_crossing_z"],
                "pr_narrows_toward_high_z": post_row["pr_narrows_toward_high_z"],
                "pr_broadens_toward_high_z": post_row["pr_broadens_toward_high_z"],
            })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "cross_model_summary.csv"), index=False)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Figure 14 style z-chi_eff plots with GWTC-4 + PySR overlays")
    print("with science diagnostics")
    print("=" * 80)

    results = {}
    for model_name, h5_file in MODEL_FILES.items():
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"Missing file: {h5_file}")
        results[model_name] = analyze_model(model_name, h5_file)

    plot_top_middle_panels(
        results,
        outpath=os.path.join(OUTDIR, "figure14_style_top_middle_pysr_overlay.png"),
    )
    print("Saved: figure14_style_top_middle_pysr_overlay.png")

    plot_gradient_panels(
        results,
        outpath=os.path.join(OUTDIR, "figure14_style_gradients_pysr_overlay.png"),
    )
    print("Saved: figure14_style_gradients_pysr_overlay.png")

    build_cross_model_summary()
    print("Saved: cross_model_summary.csv")

    print(f"\nDone. Outputs written to:\n  {OUTDIR}")


if __name__ == "__main__":
    main()