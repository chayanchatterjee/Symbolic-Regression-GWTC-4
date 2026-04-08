#!/usr/bin/env python3
"""
Figure-11-style q-chi_eff analysis with published curves + PySR curves
for BOTH Spline and Linear models.

What this script does
---------------------
1. Loads:
      BBHCorr_qchieffSplineCorrelationModel.h5
      BBHCorr_qchieffLinearCorrelationModel.h5

2. Uses the correct keys:
      posterior/rates_on_grids/mu_chieff/positions
      posterior/rates_on_grids/mu_chieff/rates
      posterior/rates_on_grids/sigma_chieff/positions
      posterior/rates_on_grids/sigma_chieff/rates

3. Computes published median and 90% CI from posterior grids.

4. Fits PySR to:
      - median mu_chieff(q)
      - median log sigma_chieff(q)
   and also to a subset of posterior draws for both quantities.

5. Builds PySR 90% CI from the posterior-draw symbolic fits.

6. Plots Figure-11-style panels:
      - top-style: mu_chieff(q)
      - middle-style: sigma_chieff(q)
   overlaying published and PySR results for BOTH Spline and Linear.

7. Makes gradient plots for:
      - dmu/dq
      - dlnsigma/dq
   with optional smoothing.

Notes
-----
- PySR is fit to log sigma for stability, then exponentiated for sigma plots.
- The "published" CI is from the original posterior grids.
- The "PySR" CI is from fitting PySR separately to posterior draws and
  taking the 5th/50th/95th percentiles of the symbolic predictions.
"""

from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pysr import PySRRegressor

from scipy.signal import savgol_filter


warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = "."
OUTDIR = os.path.join(DATADIR, "outdir_qchieff_figure11_pysr_overlay")
os.makedirs(OUTDIR, exist_ok=True)

MODEL_FILES = {
    "Spline": os.path.join(DATADIR, "BBHCorr_qchieffSplineCorrelationModel.h5"),
    "Linear": os.path.join(DATADIR, "BBHCorr_qchieffLinearCorrelationModel.h5"),
}

RANDOM_SEED = 42
SIGMA_FLOOR = 1e-6
QMIN = 0.05
QMAX = 1.0

# Number of posterior draws to fit with PySR
# Increase if you want smoother PySR credible bands, but this gets expensive.
N_POSTERIOR_DRAWS_TO_FIT = 150

# PySR training settings
N_FIT_POINTS = 150

# Gradient smoothing
USE_SMOOTHING = True
SAVGOL_WINDOW = 41   # must be odd and <= len(q_grid)
SAVGOL_POLYORDER = 3

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 12,
    "figure.dpi": 150,
})

# Published vs PySR colors/styles
COLORS = {
    "Spline": "#1f77b4",
    "Linear": "#d62728",
}
PYSR_LINESTYLE = "--"
PUBLISHED_LINESTYLE = "-"

ALPHA_PUBLISHED = 0.16
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


def trim_q_range(q_grid: np.ndarray, *arrays: np.ndarray, qmin=QMIN, qmax=QMAX):
    mask = (q_grid >= qmin) & (q_grid <= qmax)
    out = [q_grid[mask]]
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            out.append(arr[mask])
        elif arr.ndim == 2:
            out.append(arr[:, mask])
        else:
            raise ValueError("Unexpected ndim in trim_q_range.")
    return tuple(out)


def build_feature_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.column_stack([
        q,
        q**2,
        np.log(q + 1e-6),
        1 - q,
        (1 - q)**2,
        q * (1 - q),
        np.sqrt(q),
        np.log(1 + q),
        np.abs(q - 0.5),
    ])


def gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.gradient(y, x)


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


# ============================================================
# Data loading
# ============================================================

def load_qchieff_file(h5_file: str) -> Dict[str, np.ndarray]:
    """
    Correct keys based on the user's working code.
    """
    with h5py.File(h5_file, "r") as f:
        mu_pos = np.asarray(f["posterior/rates_on_grids/mu_chieff/positions"][0], dtype=float)
        mu_rates = np.asarray(f["posterior/rates_on_grids/mu_chieff/rates"][:], dtype=float)

        sigma_pos = np.asarray(f["posterior/rates_on_grids/sigma_chieff/positions"][0], dtype=float)
        sigma_rates = np.asarray(f["posterior/rates_on_grids/sigma_chieff/rates"][:], dtype=float)

        chi_pos = np.asarray(f["posterior/rates_on_grids/chi_eff/positions"][0], dtype=float)
        chi_rates = np.asarray(f["posterior/rates_on_grids/chi_eff/rates"][:], dtype=float)

    return {
        "q_grid": mu_pos,
        "mu_rates": mu_rates,
        "sigma_rates": sigma_rates,
        "chi_pos": chi_pos,
        "chi_rates": chi_rates,
    }


# ============================================================
# PySR fitting
# ============================================================

def fit_pysr_curve(
    q_grid: np.ndarray,
    y_grid: np.ndarray,
    random_state: int,
) -> Tuple[PySRRegressor, np.ndarray, pd.DataFrame]:
    idx = np.round(np.linspace(0, len(q_grid) - 1, N_FIT_POINTS)).astype(int)
    q_fit = q_grid[idx]
    y_fit = y_grid[idx]

    X_fit = build_feature_matrix(q_fit)
    model = PySRRegressor(
        random_state=random_state,
        **PYSR_KWARGS,
    )
    model.fit(X_fit, y_fit)

    X_full = build_feature_matrix(q_grid)
    y_pred = np.asarray(model.predict(X_full), dtype=float)

    eqs = model.equations_
    return model, y_pred, eqs


def fit_pysr_draws(
    q_grid: np.ndarray,
    draws: np.ndarray,
    random_state_base: int,
    quantity_name: str,
    model_name: str,
    outdir: str,
) -> np.ndarray:
    """
    Fit PySR separately to each posterior draw.
    Returns array of symbolic predictions on the full q-grid:
        shape = (n_successful_draws, n_q)
    """
    pred_list = []
    draw_dir = os.path.join(outdir, f"{model_name}_{quantity_name}_posterior_draws")
    os.makedirs(draw_dir, exist_ok=True)

    for i, y_draw in enumerate(draws):
        try:
            model, y_pred, eqs = fit_pysr_curve(
                q_grid=q_grid,
                y_grid=y_draw,
                random_state=random_state_base + i,
            )
            pred_list.append(y_pred)

            if eqs is not None:
                eqs.to_csv(os.path.join(draw_dir, f"draw_{i:04d}_equations.csv"), index=False)

            print(f"  {model_name} {quantity_name}: fitted draw {i+1}/{len(draws)}")
        except Exception as e:
            print(f"  [WARN] {model_name} {quantity_name} draw {i} failed: {e}")

    if len(pred_list) == 0:
        raise RuntimeError(f"No successful PySR draw fits for {model_name} {quantity_name}")

    return np.stack(pred_list, axis=0)


# ============================================================
# Per-model analysis
# ============================================================

def analyze_model(model_name: str, h5_file: str) -> Dict[str, np.ndarray]:
    print("\n" + "=" * 80)
    print(f"Running model: {model_name}")
    print("=" * 80)

    data = load_qchieff_file(h5_file)

    q_grid, mu_rates, sigma_rates = trim_q_range(
        data["q_grid"],
        data["mu_rates"],
        sanitize_sigma(data["sigma_rates"]),
    )

    # Published summaries
    mu_pub_med, mu_pub_lo, mu_pub_hi = percentile_summary(mu_rates)
    sigma_pub_med, sigma_pub_lo, sigma_pub_hi = percentile_summary(sigma_rates)

    logsigma_rates = np.log(sigma_rates)
    logsigma_pub_med, logsigma_pub_lo, logsigma_pub_hi = percentile_summary(logsigma_rates)

    # PySR median fits
    mu_model, mu_pysr_med, mu_eqs = fit_pysr_curve(
        q_grid=q_grid,
        y_grid=mu_pub_med,
        random_state=RANDOM_SEED + (0 if model_name == "Spline" else 100),
    )
    logsigma_model, logsigma_pysr_med, logsigma_eqs = fit_pysr_curve(
        q_grid=q_grid,
        y_grid=logsigma_pub_med,
        random_state=RANDOM_SEED + (1 if model_name == "Spline" else 101),
    )
    sigma_pysr_med = np.exp(logsigma_pysr_med)

    # Posterior-draw PySR fits
    mu_draws_sub = subsample_draws(mu_rates, N_POSTERIOR_DRAWS_TO_FIT, seed=RANDOM_SEED + 11)
    logsigma_draws_sub = subsample_draws(logsigma_rates, N_POSTERIOR_DRAWS_TO_FIT, seed=RANDOM_SEED + 22)

    mu_pysr_draw_preds = fit_pysr_draws(
        q_grid=q_grid,
        draws=mu_draws_sub,
        random_state_base=1000 if model_name == "Spline" else 2000,
        quantity_name="mu_chieff",
        model_name=model_name,
        outdir=OUTDIR,
    )
    logsigma_pysr_draw_preds = fit_pysr_draws(
        q_grid=q_grid,
        draws=logsigma_draws_sub,
        random_state_base=3000 if model_name == "Spline" else 4000,
        quantity_name="logsigma_chieff",
        model_name=model_name,
        outdir=OUTDIR,
    )
    sigma_pysr_draw_preds = np.exp(logsigma_pysr_draw_preds)

    # PySR summaries from posterior draws
    mu_pysr_band_med, mu_pysr_band_lo, mu_pysr_band_hi = percentile_summary(mu_pysr_draw_preds)
    sigma_pysr_band_med, sigma_pysr_band_lo, sigma_pysr_band_hi = percentile_summary(sigma_pysr_draw_preds)
    logsigma_pysr_band_med, logsigma_pysr_band_lo, logsigma_pysr_band_hi = percentile_summary(logsigma_pysr_draw_preds)

    # Gradients
    dmu_pub_med = gradient(mu_pub_med, q_grid)
    dmu_pub_lo = gradient(mu_pub_lo, q_grid)
    dmu_pub_hi = gradient(mu_pub_hi, q_grid)

    dlogsigma_pub_med = gradient(logsigma_pub_med, q_grid)
    dlogsigma_pub_lo = gradient(logsigma_pub_lo, q_grid)
    dlogsigma_pub_hi = gradient(logsigma_pub_hi, q_grid)

    dmu_pysr_draws = np.array([gradient(y, q_grid) for y in mu_pysr_draw_preds])
    dlogsigma_pysr_draws = np.array([gradient(y, q_grid) for y in logsigma_pysr_draw_preds])

    dmu_pysr_med = gradient(mu_pysr_med, q_grid)
    dlogsigma_pysr_med = gradient(logsigma_pysr_med, q_grid)

    dmu_pysr_band_med, dmu_pysr_band_lo, dmu_pysr_band_hi = percentile_summary(dmu_pysr_draws)
    dlogsigma_pysr_band_med, dlogsigma_pysr_band_lo, dlogsigma_pysr_band_hi = percentile_summary(dlogsigma_pysr_draws)

    # Optional smoothing for gradient display
    dmu_pub_med_s = maybe_smooth(dmu_pub_med)
    dmu_pub_lo_s = maybe_smooth(dmu_pub_lo)
    dmu_pub_hi_s = maybe_smooth(dmu_pub_hi)

    dlogsigma_pub_med_s = maybe_smooth(dlogsigma_pub_med)
    dlogsigma_pub_lo_s = maybe_smooth(dlogsigma_pub_lo)
    dlogsigma_pub_hi_s = maybe_smooth(dlogsigma_pub_hi)

    dmu_pysr_med_s = maybe_smooth(dmu_pysr_med)
    dmu_pysr_band_lo_s = maybe_smooth(dmu_pysr_band_lo)
    dmu_pysr_band_hi_s = maybe_smooth(dmu_pysr_band_hi)

    dlogsigma_pysr_med_s = maybe_smooth(dlogsigma_pysr_med)
    dlogsigma_pysr_band_lo_s = maybe_smooth(dlogsigma_pysr_band_lo)
    dlogsigma_pysr_band_hi_s = maybe_smooth(dlogsigma_pysr_band_hi)

    model_outdir = os.path.join(OUTDIR, model_name)
    os.makedirs(model_outdir, exist_ok=True)

    mu_eqs.to_csv(os.path.join(model_outdir, "mu_median_equations.csv"), index=False)
    logsigma_eqs.to_csv(os.path.join(model_outdir, "logsigma_median_equations.csv"), index=False)

    np.savez(
        os.path.join(model_outdir, "results.npz"),
        q_grid=q_grid,

        mu_pub_med=mu_pub_med,
        mu_pub_lo=mu_pub_lo,
        mu_pub_hi=mu_pub_hi,

        sigma_pub_med=sigma_pub_med,
        sigma_pub_lo=sigma_pub_lo,
        sigma_pub_hi=sigma_pub_hi,

        logsigma_pub_med=logsigma_pub_med,
        logsigma_pub_lo=logsigma_pub_lo,
        logsigma_pub_hi=logsigma_pub_hi,

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

        dmu_pub_med=dmu_pub_med,
        dmu_pub_lo=dmu_pub_lo,
        dmu_pub_hi=dmu_pub_hi,

        dlogsigma_pub_med=dlogsigma_pub_med,
        dlogsigma_pub_lo=dlogsigma_pub_lo,
        dlogsigma_pub_hi=dlogsigma_pub_hi,

        dmu_pysr_med=dmu_pysr_med,
        dmu_pysr_band_med=dmu_pysr_band_med,
        dmu_pysr_band_lo=dmu_pysr_band_lo,
        dmu_pysr_band_hi=dmu_pysr_band_hi,

        dlogsigma_pysr_med=dlogsigma_pysr_med,
        dlogsigma_pysr_band_med=dlogsigma_pysr_band_med,
        dlogsigma_pysr_band_lo=dlogsigma_pysr_band_lo,
        dlogsigma_pysr_band_hi=dlogsigma_pysr_band_hi,
    )

    best_mu_eq = mu_eqs.sort_values("loss").iloc[0]["equation"]
    best_logsigma_eq = logsigma_eqs.sort_values("loss").iloc[0]["equation"]

    with open(os.path.join(model_outdir, "best_equations.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"mu_chieff(q) = {best_mu_eq}\n")
        f.write(f"log sigma_chieff(q) = {best_logsigma_eq}\n")

    return {
        "q_grid": q_grid,

        "mu_pub_med": mu_pub_med,
        "mu_pub_lo": mu_pub_lo,
        "mu_pub_hi": mu_pub_hi,

        "sigma_pub_med": sigma_pub_med,
        "sigma_pub_lo": sigma_pub_lo,
        "sigma_pub_hi": sigma_pub_hi,

        "logsigma_pub_med": logsigma_pub_med,
        "logsigma_pub_lo": logsigma_pub_lo,
        "logsigma_pub_hi": logsigma_pub_hi,

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

        "dmu_pub_med_s": dmu_pub_med_s,
        "dmu_pub_lo_s": dmu_pub_lo_s,
        "dmu_pub_hi_s": dmu_pub_hi_s,

        "dlogsigma_pub_med_s": dlogsigma_pub_med_s,
        "dlogsigma_pub_lo_s": dlogsigma_pub_lo_s,
        "dlogsigma_pub_hi_s": dlogsigma_pub_hi_s,

        "dmu_pysr_med_s": dmu_pysr_med_s,
        "dmu_pysr_band_lo_s": dmu_pysr_band_lo_s,
        "dmu_pysr_band_hi_s": dmu_pysr_band_hi_s,

        "dlogsigma_pysr_med_s": dlogsigma_pysr_med_s,
        "dlogsigma_pysr_band_lo_s": dlogsigma_pysr_band_lo_s,
        "dlogsigma_pysr_band_hi_s": dlogsigma_pysr_band_hi_s,
    }


# ============================================================
# Plotting
# ============================================================

def plot_top_middle_panels(results: Dict[str, Dict[str, np.ndarray]], outpath: str):
    """
    Figure-11-style top and middle panels:
      left: mu_chieff(q)
      right: sigma_chieff(q)

    For BOTH Spline and Linear:
      - published median + published 90% CI
      - PySR median + PySR 90% CI from symbolic posterior draws
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --------------------------------------------------------
    # Top-style panel: mu_chieff(q)
    # --------------------------------------------------------
    ax = axes[0]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        q = r["q_grid"]
        c = COLORS[model_name]

        # Published band
        ax.fill_between(
            q, r["mu_pub_lo"], r["mu_pub_hi"],
            color=c, alpha=ALPHA_PUBLISHED
        )
        # PySR band
        ax.fill_between(
            q, r["mu_pysr_band_lo"], r["mu_pysr_band_hi"],
            color=c, alpha=ALPHA_PYSR
        )

        # Published median
        ax.plot(
            q, r["mu_pub_med"],
            color=c, lw=2.2, ls=PUBLISHED_LINESTYLE,
            label=f"{model_name} published median"
        )
        # PySR median
        ax.plot(
            q, r["mu_pysr_med"],
            color=c, lw=2.0, ls=PYSR_LINESTYLE,
            label=f"{model_name} PySR median"
        )

    ax.axhline(0, color="k", lw=1, ls=":", alpha=0.7)
    ax.set_xlim(QMIN, 1.0)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\mu_{\chi_\mathrm{eff}}(q)$")
    ax.set_title(r"Mean effective spin vs mass ratio")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=1)

    # --------------------------------------------------------
    # Middle-style panel: sigma_chieff(q)
    # --------------------------------------------------------
    ax = axes[1]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        q = r["q_grid"]
        c = COLORS[model_name]

        # Published band
        ax.fill_between(
            q, r["sigma_pub_lo"], r["sigma_pub_hi"],
            color=c, alpha=ALPHA_PUBLISHED
        )
        # PySR band
        ax.fill_between(
            q, r["sigma_pysr_band_lo"], r["sigma_pysr_band_hi"],
            color=c, alpha=ALPHA_PYSR
        )

        # Published median
        ax.plot(
            q, r["sigma_pub_med"],
            color=c, lw=2.2, ls=PUBLISHED_LINESTYLE,
            label=f"{model_name} published median"
        )
        # PySR median
        ax.plot(
            q, r["sigma_pysr_med"],
            color=c, lw=2.0, ls=PYSR_LINESTYLE,
            label=f"{model_name} PySR median"
        )

    ax.set_xlim(QMIN, 1.0)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\sigma_{\chi_\mathrm{eff}}(q)$")
    ax.set_title(r"Width of effective-spin distribution vs mass ratio")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=1)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_gradient_panels(results: Dict[str, Dict[str, np.ndarray]], outpath: str):
    """
    Gradient plots, with optional smoothing.
    Left: dmu/dq
    Right: dlnsigma/dq

    For BOTH Spline and Linear:
      - published median + published 90% CI
      - PySR median + PySR 90% CI from symbolic posterior draws
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # --------------------------------------------------------
    # dmu/dq
    # --------------------------------------------------------
    ax = axes[0]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        q = r["q_grid"]
        c = COLORS[model_name]

        ax.fill_between(
            q, r["dmu_pub_lo_s"], r["dmu_pub_hi_s"],
            color=c, alpha=ALPHA_PUBLISHED
        )
        ax.fill_between(
            q, r["dmu_pysr_band_lo_s"], r["dmu_pysr_band_hi_s"],
            color=c, alpha=ALPHA_PYSR
        )

        ax.plot(
            q, r["dmu_pub_med_s"],
            color=c, lw=2.2, ls=PUBLISHED_LINESTYLE,
            label=f"{model_name} published median"
        )
        ax.plot(
            q, r["dmu_pysr_med_s"],
            color=c, lw=2.0, ls=PYSR_LINESTYLE,
            label=f"{model_name} PySR median"
        )

    ax.axhline(0, color="k", lw=1, ls=":", alpha=0.7)
    ax.axvline(0.6, color="k", lw=1, ls="--", alpha=0.5)
    ax.set_xlim(QMIN, 1.0)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$d\mu_{\chi_\mathrm{eff}}/dq$")
    ax.set_title(r"Gradient of mean effective spin")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=1)

    # --------------------------------------------------------
    # dlnsigma/dq
    # --------------------------------------------------------
    ax = axes[1]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        q = r["q_grid"]
        c = COLORS[model_name]

        ax.fill_between(
            q, r["dlogsigma_pub_lo_s"], r["dlogsigma_pub_hi_s"],
            color=c, alpha=ALPHA_PUBLISHED
        )
        ax.fill_between(
            q, r["dlogsigma_pysr_band_lo_s"], r["dlogsigma_pysr_band_hi_s"],
            color=c, alpha=ALPHA_PYSR
        )

        ax.plot(
            q, r["dlogsigma_pub_med_s"],
            color=c, lw=2.2, ls=PUBLISHED_LINESTYLE,
            label=f"{model_name} published median"
        )
        ax.plot(
            q, r["dlogsigma_pysr_med_s"],
            color=c, lw=2.0, ls=PYSR_LINESTYLE,
            label=f"{model_name} PySR median"
        )

    ax.axhline(0, color="k", lw=1, ls=":", alpha=0.7)
    ax.axvline(0.6, color="k", lw=1, ls="--", alpha=0.5)
    ax.set_xlim(QMIN, 1.0)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$d\ln\sigma_{\chi_\mathrm{eff}}/dq$")
    ax.set_title(r"Gradient of log width")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=1)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Figure 11 style q-chi_eff plots with published + PySR overlays")
    print("=" * 80)

    results = {}
    for model_name, h5_file in MODEL_FILES.items():
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"Missing file: {h5_file}")
        results[model_name] = analyze_model(model_name, h5_file)

    plot_top_middle_panels(
        results,
        outpath=os.path.join(OUTDIR, "figure11_style_top_middle_pysr_overlay.png"),
    )
    print("Saved: figure11_style_top_middle_pysr_overlay.png")

    plot_gradient_panels(
        results,
        outpath=os.path.join(OUTDIR, "figure11_style_gradients_pysr_overlay.png"),
    )
    print("Saved: figure11_style_gradients_pysr_overlay.png")

    # Save a lightweight cross-model summary
    rows = []
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        rows.append({
            "model_name": model_name,
            "mu_pub_q06": float(np.interp(0.6, r["q_grid"], r["mu_pub_med"])),
            "mu_pysr_q06": float(np.interp(0.6, r["q_grid"], r["mu_pysr_med"])),
            "sigma_pub_q06": float(np.interp(0.6, r["q_grid"], r["sigma_pub_med"])),
            "sigma_pysr_q06": float(np.interp(0.6, r["q_grid"], r["sigma_pysr_med"])),
            "dmu_pub_q06": float(np.interp(0.6, r["q_grid"], r["dmu_pub_med_s"])),
            "dmu_pysr_q06": float(np.interp(0.6, r["q_grid"], r["dmu_pysr_med_s"])),
            "dlogsigma_pub_q06": float(np.interp(0.6, r["q_grid"], r["dlogsigma_pub_med_s"])),
            "dlogsigma_pysr_q06": float(np.interp(0.6, r["q_grid"], r["dlogsigma_pysr_med_s"])),
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "cross_model_summary.csv"), index=False)
    print("Saved: cross_model_summary.csv")

    print(f"\nDone. Outputs written to:\n  {OUTDIR}")


if __name__ == "__main__":
    main()