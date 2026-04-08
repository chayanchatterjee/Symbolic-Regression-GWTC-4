#!/usr/bin/env python3
"""
Replot Figure-14-style z-chi_eff figures WITHOUT rerunning PySR.

What this script does
---------------------
1. Loads existing PySR results from:
      outdir_zchieff_figure14_overlay_with_diagnostics/Spline/results.npz
      outdir_zchieff_figure14_overlay_with_diagnostics/Linear/results.npz

2. Reloads the original GWTC-4 posterior draws from:
      BBHCorr_zchieffSplineCorrelationModel.h5
      BBHCorr_zchieffLinearCorrelationModel.h5

3. Recomputes the GWTC-4 gradient bands properly:
      - for each posterior draw, compute numerical gradient
      - then take 5th/50th/95th percentiles across gradient draws

4. Keeps the PySR medians and PySR posterior-draw bands from the saved NPZ files.

5. Produces:
      - new_gradient_overlay_fixed.png
      - new_top_middle_overlay_fixed.png

"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Tuple

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Existing PySR output directory from previous z-chi_eff run
PYSR_OUTDIR = os.path.join(BASE_DIR, "outdir_zchieff_figure14_overlay_with_diagnostics")

# Original HDF5 files
DATADIR = "."
MODEL_FILES = {
    "Spline": os.path.join(DATADIR, "BBHCorr_zchieffSplineCorrelationModel.h5"),
    "Linear": os.path.join(DATADIR, "BBHCorr_zchieffLinearCorrelationModel.h5"),
}

# Existing saved NPZ files from previous run
NPZ_FILES = {
    "Spline": os.path.join(PYSR_OUTDIR, "Spline", "results.npz"),
    "Linear": os.path.join(PYSR_OUTDIR, "Linear", "results.npz"),
}

# Output directory for replots
OUTDIR = os.path.join(BASE_DIR, "outdir_zchieff_figure14_replot_fixed")
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# Plot config
# ============================================================

ZMIN = 0.0
ZMAX = 1.9
SIGMA_FLOOR = 1e-6

USE_SMOOTHING = True
SAVGOL_WINDOW = 41
SAVGOL_POLYORDER = 3

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 12,
    "figure.dpi": 150,
})

# Same color scheme in both new figures:
# source encoded by color, model encoded by linestyle
GWTC4_COLOR = "#1f77b4"   # blue
PYSR_COLOR  = "#ff7f0e"   # orange

GWTC4_ALPHA = 0.18
PYSR_ALPHA = 0.18

MODEL_LINESTYLES = {
    "Spline": "-",
    "Linear": "--",
}


# ============================================================
# Helpers
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
# Load original GWTC-4 posterior draws
# ============================================================

def load_zchieff_draws(h5_file: str) -> Dict[str, np.ndarray]:
    with h5py.File(h5_file, "r") as f:
        z_grid = np.asarray(f["posterior/rates_on_grids/mu_chieff/positions"][0], dtype=float)
        mu_rates = np.asarray(f["posterior/rates_on_grids/mu_chieff/rates"][:], dtype=float)
        sigma_rates = np.asarray(f["posterior/rates_on_grids/sigma_chieff/rates"][:], dtype=float)

    sigma_rates = sanitize_sigma(sigma_rates)
    z_grid, mu_rates, sigma_rates = trim_z_range(z_grid, mu_rates, sigma_rates)

    logsigma_rates = np.log(sigma_rates)

    return {
        "z_grid": z_grid,
        "mu_rates": mu_rates,
        "sigma_rates": sigma_rates,
        "logsigma_rates": logsigma_rates,
    }


# ============================================================
# Recompute GWTC-4 gradient bands correctly from posterior draws
# ============================================================

def compute_gwtc4_quantities_from_draws(
    z_grid: np.ndarray,
    mu_rates: np.ndarray,
    sigma_rates: np.ndarray,
    logsigma_rates: np.ndarray
):
    # Function-space summaries
    mu_med, mu_lo, mu_hi = percentile_summary(mu_rates)
    sigma_med, sigma_lo, sigma_hi = percentile_summary(sigma_rates)
    logsigma_med, logsigma_lo, logsigma_hi = percentile_summary(logsigma_rates)

    # Proper gradient posterior summaries:
    # differentiate each draw first, then take percentiles
    dmu_draws = np.array([gradient(y, z_grid) for y in mu_rates])
    dlogsigma_draws = np.array([gradient(y, z_grid) for y in logsigma_rates])

    dmu_med, dmu_lo, dmu_hi = percentile_summary(dmu_draws)
    dlogsigma_med, dlogsigma_lo, dlogsigma_hi = percentile_summary(dlogsigma_draws)

    # Smoothed versions for display only
    dmu_med_s = maybe_smooth(dmu_med)
    dmu_lo_s = maybe_smooth(dmu_lo)
    dmu_hi_s = maybe_smooth(dmu_hi)

    dlogsigma_med_s = maybe_smooth(dlogsigma_med)
    dlogsigma_lo_s = maybe_smooth(dlogsigma_lo)
    dlogsigma_hi_s = maybe_smooth(dlogsigma_hi)

    return {
        "mu_med": mu_med,
        "mu_lo": mu_lo,
        "mu_hi": mu_hi,
        "sigma_med": sigma_med,
        "sigma_lo": sigma_lo,
        "sigma_hi": sigma_hi,
        "logsigma_med": logsigma_med,
        "logsigma_lo": logsigma_lo,
        "logsigma_hi": logsigma_hi,
        "dmu_med_s": dmu_med_s,
        "dmu_lo_s": dmu_lo_s,
        "dmu_hi_s": dmu_hi_s,
        "dlogsigma_med_s": dlogsigma_med_s,
        "dlogsigma_lo_s": dlogsigma_lo_s,
        "dlogsigma_hi_s": dlogsigma_hi_s,
    }


# ============================================================
# Load existing PySR outputs (no rerun)
# ============================================================

def load_pysr_results(npz_file: str) -> Dict[str, np.ndarray]:
    d = np.load(npz_file, allow_pickle=True)

    out = {
        "z_grid": d["z_grid"],

        "mu_pysr_med": d["mu_pysr_med"],
        "mu_pysr_band_lo": d["mu_pysr_band_lo"],
        "mu_pysr_band_hi": d["mu_pysr_band_hi"],

        "sigma_pysr_med": d["sigma_pysr_med"],
        "sigma_pysr_band_lo": d["sigma_pysr_band_lo"],
        "sigma_pysr_band_hi": d["sigma_pysr_band_hi"],

        "logsigma_pysr_med": d["logsigma_pysr_med"],
        "logsigma_pysr_band_lo": d["logsigma_pysr_band_lo"],
        "logsigma_pysr_band_hi": d["logsigma_pysr_band_hi"],

        "dmu_pysr_med": d["dmu_pysr_med"],
        "dmu_pysr_band_lo": d["dmu_pysr_band_lo"],
        "dmu_pysr_band_hi": d["dmu_pysr_band_hi"],

        "dlogsigma_pysr_med": d["dlogsigma_pysr_med"],
        "dlogsigma_pysr_band_lo": d["dlogsigma_pysr_band_lo"],
        "dlogsigma_pysr_band_hi": d["dlogsigma_pysr_band_hi"],
    }

    # Smoothed versions for plotting
    out["dmu_pysr_med_s"] = maybe_smooth(out["dmu_pysr_med"])
    out["dmu_pysr_band_lo_s"] = maybe_smooth(out["dmu_pysr_band_lo"])
    out["dmu_pysr_band_hi_s"] = maybe_smooth(out["dmu_pysr_band_hi"])

    out["dlogsigma_pysr_med_s"] = maybe_smooth(out["dlogsigma_pysr_med"])
    out["dlogsigma_pysr_band_lo_s"] = maybe_smooth(out["dlogsigma_pysr_band_lo"])
    out["dlogsigma_pysr_band_hi_s"] = maybe_smooth(out["dlogsigma_pysr_band_hi"])

    return out


# ============================================================
# Merge GWTC-4 and PySR info
# ============================================================

def build_results() -> Dict[str, Dict[str, np.ndarray]]:
    results = {}

    for model_name in ["Spline", "Linear"]:
        if not os.path.exists(MODEL_FILES[model_name]):
            raise FileNotFoundError(f"Missing HDF5 file: {MODEL_FILES[model_name]}")
        if not os.path.exists(NPZ_FILES[model_name]):
            raise FileNotFoundError(f"Missing existing PySR results: {NPZ_FILES[model_name]}")

        raw = load_zchieff_draws(MODEL_FILES[model_name])
        gwtc4 = compute_gwtc4_quantities_from_draws(
            raw["z_grid"], raw["mu_rates"], raw["sigma_rates"], raw["logsigma_rates"]
        )
        pysr = load_pysr_results(NPZ_FILES[model_name])

        z_grid = raw["z_grid"]
        if not np.allclose(z_grid, pysr["z_grid"]):
            raise ValueError(f"z-grid mismatch for {model_name}")

        results[model_name] = {
            "z_grid": z_grid,
            **gwtc4,
            **pysr,
        }

    return results


# ============================================================
# Plot: mean and width vs z
# ============================================================

def plot_top_middle(results: Dict[str, Dict[str, np.ndarray]], outpath: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Mean vs z
    ax = axes[0]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        z = r["z_grid"]
        ls = MODEL_LINESTYLES[model_name]

        ax.fill_between(
            z, r["mu_lo"], r["mu_hi"],
            color=GWTC4_COLOR, alpha=GWTC4_ALPHA
        )
        ax.fill_between(
            z, r["mu_pysr_band_lo"], r["mu_pysr_band_hi"],
            color=PYSR_COLOR, alpha=PYSR_ALPHA
        )

        ax.plot(
            z, r["mu_med"],
            color=GWTC4_COLOR, lw=2.4, ls=ls,
            label=f"{model_name} GWTC-4 median"
        )
        ax.plot(
            z, r["mu_pysr_med"],
            color=PYSR_COLOR, lw=2.4, ls=ls,
            label=f"{model_name} PySR median"
        )

    ax.axhline(0, color="k", lw=1, ls=":", alpha=0.7)
    ax.set_xlim(ZMIN, ZMAX)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\mu_{\chi_\mathrm{eff}}(z)$")
    ax.set_title(r"Mean effective spin vs redshift")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    # Width vs z
    ax = axes[1]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        z = r["z_grid"]
        ls = MODEL_LINESTYLES[model_name]

        ax.fill_between(
            z, r["sigma_lo"], r["sigma_hi"],
            color=GWTC4_COLOR, alpha=GWTC4_ALPHA
        )
        ax.fill_between(
            z, r["sigma_pysr_band_lo"], r["sigma_pysr_band_hi"],
            color=PYSR_COLOR, alpha=PYSR_ALPHA
        )

        ax.plot(
            z, r["sigma_med"],
            color=GWTC4_COLOR, lw=2.4, ls=ls,
            label=f"{model_name} GWTC-4 median"
        )
        ax.plot(
            z, r["sigma_pysr_med"],
            color=PYSR_COLOR, lw=2.4, ls=ls,
            label=f"{model_name} PySR median"
        )

    ax.set_xlim(ZMIN, ZMAX)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\sigma_{\chi_\mathrm{eff}}(z)$")
    ax.set_title(r"Width of effective-spin distribution vs redshift")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def plot_gradients(results: Dict[str, Dict[str, np.ndarray]], outpath: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # dmu/dz
    ax = axes[0]
    for model_name in ["Spline", "Linear"]:
        r = results[model_name]
        z = r["z_grid"]
        ls = MODEL_LINESTYLES[model_name]

        ax.fill_between(
            z, r["dmu_lo_s"], r["dmu_hi_s"],
            color=GWTC4_COLOR, alpha=GWTC4_ALPHA
        )
        ax.fill_between(
            z, r["dmu_pysr_band_lo_s"], r["dmu_pysr_band_hi_s"],
            color=PYSR_COLOR, alpha=PYSR_ALPHA
        )

        ax.plot(
            z, r["dmu_med_s"],
            color=GWTC4_COLOR, lw=2.4, ls=ls,
            label=f"{model_name} GWTC-4 median"
        )
        ax.plot(
            z, r["dmu_pysr_med_s"],
            color=PYSR_COLOR, lw=2.4, ls=ls,
            label=f"{model_name} PySR median"
        )

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
        ls = MODEL_LINESTYLES[model_name]

        ax.fill_between(
            z, r["dlogsigma_lo_s"], r["dlogsigma_hi_s"],
            color=GWTC4_COLOR, alpha=GWTC4_ALPHA
        )
        ax.fill_between(
            z, r["dlogsigma_pysr_band_lo_s"], r["dlogsigma_pysr_band_hi_s"],
            color=PYSR_COLOR, alpha=PYSR_ALPHA
        )

        ax.plot(
            z, r["dlogsigma_med_s"],
            color=GWTC4_COLOR, lw=2.4, ls=ls,
            label=f"{model_name} GWTC-4 median"
        )
        ax.plot(
            z, r["dlogsigma_pysr_med_s"],
            color=PYSR_COLOR, lw=2.4, ls=ls,
            label=f"{model_name} PySR median"
        )

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
# Main
# ============================================================

def main():
    results = build_results()

    top_middle_path = os.path.join(OUTDIR, "new_top_middle_overlay_fixed.png")
    gradients_path = os.path.join(OUTDIR, "new_gradient_overlay_fixed.png")

    plot_top_middle(results, top_middle_path)
    plot_gradients(results, gradients_path)

    print("Saved:")
    print(f"  {top_middle_path}")
    print(f"  {gradients_path}")


if __name__ == "__main__":
    main()