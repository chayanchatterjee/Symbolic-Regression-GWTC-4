import os
import json
import warnings
from dataclasses import dataclass, asdict

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from pysr import PySRRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# User settings
# ============================================================
FILE_VARYQ = "BBHMass_VaryingBetaQs_DominantMode.h5"
OUTDIR = "pysr_q_lowmass_highmass_only"

# Number of posterior draws to fit with PySR for the PySR CI band
N_DRAW_FITS = 150

RNG_SEED = 1234
NITER = 300
POPULATIONS = 30
MAXSIZE = 30
Q_MIN = 1e-3
N_EVAL = 500

REPRODUCIBLE = True
SAVE_INDIVIDUAL_FIGURES = True

# Figure-6-style y-limits from the paper
YLIMS = {
    "dP_dq_lowMassPeak": (1e-3, 1e1),
    "dP_dq_highMassPeak": (1e-3, 1e1),
}

np.random.seed(RNG_SEED)

# ============================================================
# Helpers
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sanitize_positive(arr):
    arr = np.asarray(arr, dtype=float)
    finite_pos = np.isfinite(arr) & (arr > 0)
    if np.any(finite_pos):
        floor = 0.1 * np.min(arr[finite_pos])
    else:
        floor = 1e-30
    out = arr.copy()
    out[~np.isfinite(out) | (out <= 0)] = floor
    return out

def clean_q_and_rates(q, rates):
    q = np.squeeze(np.asarray(q, dtype=float))
    rates = np.asarray(rates, dtype=float)

    if rates.ndim == 1:
        rates = rates[None, :]

    mask = np.isfinite(q) & (q > Q_MIN)
    q = q[mask]
    rates = rates[:, mask]
    rates = sanitize_positive(rates)
    return q, rates

def median_and_ci(rates, axis=0):
    med = np.nanmedian(rates, axis=axis)
    lo = np.nanpercentile(rates, 5, axis=axis)
    hi = np.nanpercentile(rates, 95, axis=axis)
    return med, lo, hi

def save_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2)

def sample_draw_indices(n_draws_total, n_use):
    n_use = min(n_use, n_draws_total)
    return np.random.choice(n_draws_total, size=n_use, replace=False)

# ============================================================
# Null models
# ============================================================
def fit_null_models(q, ylog):
    out = {}

    c = np.mean(ylog)
    yhat = np.full_like(ylog, c)
    out["constant"] = {
        "mse": float(np.mean((ylog - yhat) ** 2)),
        "desc": f"log y = {c:.6g}",
    }

    X = np.vstack([np.ones_like(q), np.log(q)]).T
    beta = np.linalg.lstsq(X, ylog, rcond=None)[0]
    yhat = X @ beta
    out["power_law"] = {
        "mse": float(np.mean((ylog - yhat) ** 2)),
        "desc": f"log y = {beta[0]:.6g} + {beta[1]:.6g} log q",
        "coef": beta.tolist(),
    }

    X = np.vstack([np.ones_like(q), q]).T
    beta = np.linalg.lstsq(X, ylog, rcond=None)[0]
    yhat = X @ beta
    out["linear_q"] = {
        "mse": float(np.mean((ylog - yhat) ** 2)),
        "desc": f"log y = {beta[0]:.6g} + {beta[1]:.6g} q",
        "coef": beta.tolist(),
    }

    X = np.vstack([np.ones_like(q), q, q**2]).T
    beta = np.linalg.lstsq(X, ylog, rcond=None)[0]
    yhat = X @ beta
    out["quadratic_q"] = {
        "mse": float(np.mean((ylog - yhat) ** 2)),
        "desc": f"log y = {beta[0]:.6g} + {beta[1]:.6g} q + {beta[2]:.6g} q^2",
        "coef": beta.tolist(),
    }

    return out

# ============================================================
# PySR
# ============================================================
def build_pysr():
    common = dict(
        niterations=NITER,
        populations=POPULATIONS,
        maxsize=MAXSIZE,
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        binary_operators=["+", "-", "*"],
        unary_operators=["sqrt", "square", "exp"],
        parsimony=1e-4,
        batching=False,
        random_state=RNG_SEED,
        warm_start=False,
        verbosity=0,
    )

    if REPRODUCIBLE:
        return PySRRegressor(
            **common,
            deterministic=True,
            parallelism="serial",
        )
    else:
        return PySRRegressor(
            **common,
            deterministic=False,
            turbo=True,
            procs=0,
        )

def fit_pysr_curve(q, y):
    q = np.asarray(q, dtype=float)
    y = sanitize_positive(y)

    mask = np.isfinite(q) & np.isfinite(y) & (q > Q_MIN)
    q = q[mask]
    y = y[mask]

    ylog = np.log(y)
    X = q.reshape(-1, 1)

    model = build_pysr()
    model.fit(X, ylog)

    ylog_hat = model.predict(X)
    mse = float(np.mean((ylog - ylog_hat) ** 2))

    best = model.get_best()
    equation = str(best["equation"])
    sympy_expr = str(model.sympy())

    nulls = fit_null_models(q, ylog)

    return {
        "model": model,
        "q_fit": q,
        "y_fit": y,
        "ylog_fit": ylog,
        "ylog_hat": ylog_hat,
        "mse": mse,
        "equation": equation,
        "sympy": sympy_expr,
        "null_models": nulls,
    }

# ============================================================
# Diagnostics
# ============================================================
@dataclass
class ShapeDiagnostics:
    q_peak: float
    has_interior_peak: bool
    monotonic_increasing: bool
    monotonic_decreasing: bool
    slope_low: float
    slope_high: float
    rate_q95_over_q80: float
    rises_toward_equal_mass: bool
    peaks_then_declines_near_unity: bool

def compute_shape_diagnostics(q_eval, y_eval):
    q_eval = np.asarray(q_eval, dtype=float)
    y_eval = sanitize_positive(y_eval)

    eps = 1e-300
    logq = np.log(q_eval)
    logy = np.log(np.maximum(y_eval, eps))
    dlogy_dlogq = np.gradient(logy, logq)

    i_peak = int(np.argmax(y_eval))
    q_peak = float(q_eval[i_peak])

    has_interior_peak = bool(q_peak < 0.98 * np.max(q_eval))
    dy = np.diff(y_eval)
    monotonic_increasing = bool(np.all(dy >= -1e-12))
    monotonic_decreasing = bool(np.all(dy <= 1e-12))

    low_mask = (q_eval >= 0.1) & (q_eval <= 0.3)
    high_mask = (q_eval >= 0.8) & (q_eval <= 0.98)

    slope_low = float(np.median(dlogy_dlogq[low_mask])) if np.any(low_mask) else np.nan
    slope_high = float(np.median(dlogy_dlogq[high_mask])) if np.any(high_mask) else np.nan

    f = interp1d(q_eval, y_eval, bounds_error=False, fill_value="extrapolate")
    y80 = float(f(0.80))
    y95 = float(f(0.95))
    ratio = y95 / y80 if y80 > 0 else np.nan

    rises_toward_equal_mass = bool(slope_high > 0 and ratio > 1.0)
    peaks_then_declines_near_unity = bool(has_interior_peak and slope_high < 0)

    return ShapeDiagnostics(
        q_peak=q_peak,
        has_interior_peak=has_interior_peak,
        monotonic_increasing=monotonic_increasing,
        monotonic_decreasing=monotonic_decreasing,
        slope_low=slope_low,
        slope_high=slope_high,
        rate_q95_over_q80=ratio,
        rises_toward_equal_mass=rises_toward_equal_mass,
        peaks_then_declines_near_unity=peaks_then_declines_near_unity,
    )

def summarize_draw_diags(diags):
    q_peak = np.array([d.q_peak for d in diags], dtype=float)
    sl = np.array([d.slope_low for d in diags], dtype=float)
    sh = np.array([d.slope_high for d in diags], dtype=float)
    rr = np.array([d.rate_q95_over_q80 for d in diags], dtype=float)

    if len(diags) == 0:
        return {
            "n_draws": 0,
            "Pr_monotonic_increasing": np.nan,
            "Pr_interior_peak": np.nan,
            "Pr_rises_toward_equal_mass": np.nan,
            "Pr_peaks_then_declines_near_unity": np.nan,
            "q_peak_median": np.nan,
            "q_peak_5_95": [np.nan, np.nan],
            "slope_low_median": np.nan,
            "slope_low_5_95": [np.nan, np.nan],
            "slope_high_median": np.nan,
            "slope_high_5_95": [np.nan, np.nan],
            "rate_q95_over_q80_median": np.nan,
            "rate_q95_over_q80_5_95": [np.nan, np.nan],
        }

    return {
        "n_draws": int(len(diags)),
        "Pr_monotonic_increasing": float(np.mean([d.monotonic_increasing for d in diags])),
        "Pr_interior_peak": float(np.mean([d.has_interior_peak for d in diags])),
        "Pr_rises_toward_equal_mass": float(np.mean([d.rises_toward_equal_mass for d in diags])),
        "Pr_peaks_then_declines_near_unity": float(np.mean([d.peaks_then_declines_near_unity for d in diags])),
        "q_peak_median": float(np.nanmedian(q_peak)),
        "q_peak_5_95": [float(np.nanpercentile(q_peak, 5)), float(np.nanpercentile(q_peak, 95))],
        "slope_low_median": float(np.nanmedian(sl)),
        "slope_low_5_95": [float(np.nanpercentile(sl, 5)), float(np.nanpercentile(sl, 95))],
        "slope_high_median": float(np.nanmedian(sh)),
        "slope_high_5_95": [float(np.nanpercentile(sh, 5)), float(np.nanpercentile(sh, 95))],
        "rate_q95_over_q80_median": float(np.nanmedian(rr)),
        "rate_q95_over_q80_5_95": [float(np.nanpercentile(rr, 5)), float(np.nanpercentile(rr, 95))],
    }

# ============================================================
# Plotting
# ============================================================
def _set_axis_limits(ax, q, dataset_key):
    ax.set_xlim(float(np.min(q)), float(np.max(q)))
    ymin, ymax = YLIMS[dataset_key]
    ax.set_ylim(ymin, ymax)

def plot_dataset(
    outpath,
    dataset_key,
    label,
    q,
    y_med_ref,
    y_lo_ref,
    y_hi_ref,
    y_med_pysr,
    y_lo_pysr,
    y_hi_pysr,
):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.fill_between(q, y_lo_ref, y_hi_ref, alpha=0.25, label="Published 90% CI")
    ax.fill_between(q, y_lo_pysr, y_hi_pysr, alpha=0.20, label="PySR 90% CI")
    ax.plot(q, y_med_ref, lw=2, label="Published median")
    ax.plot(q, y_med_pysr, "--", lw=2, label="Median PySR fit")

    ax.set_yscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("p(q)")
    ax.set_title(label)
    _set_axis_limits(ax, q, dataset_key)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def _panel_plot(ax, dataset_key, result, title):
    q = np.asarray(result["plot_data"]["q"], dtype=float)
    y_med_ref = np.asarray(result["plot_data"]["y_med_ref"], dtype=float)
    y_lo_ref = np.asarray(result["plot_data"]["y_lo_ref"], dtype=float)
    y_hi_ref = np.asarray(result["plot_data"]["y_hi_ref"], dtype=float)
    y_med_pysr = np.asarray(result["plot_data"]["y_med_pysr"], dtype=float)
    y_lo_pysr = np.asarray(result["plot_data"]["y_lo_pysr"], dtype=float)
    y_hi_pysr = np.asarray(result["plot_data"]["y_hi_pysr"], dtype=float)

    ax.fill_between(q, y_lo_ref, y_hi_ref, alpha=0.25, label="Published 90% CI")
    ax.fill_between(q, y_lo_pysr, y_hi_pysr, alpha=0.20, label="PySR 90% CI")
    ax.plot(q, y_med_ref, lw=2, label="Published median")
    ax.plot(q, y_med_pysr, "--", lw=2, label="Median PySR fit")

    ax.set_yscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("p(q)")
    ax.set_title(title)
    _set_axis_limits(ax, q, dataset_key)
    ax.legend(fontsize=8)

def plot_two_panel_figure(outpath, left_key, left_result, left_title,
                          right_key, right_result, right_title, suptitle):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _panel_plot(axes[0], left_key, left_result, left_title)
    _panel_plot(axes[1], right_key, right_result, right_title)
    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Loader
# ============================================================
def load_group_positions_rates(filename, group_key):
    with h5py.File(filename, "r") as f:
        grp = f["posterior"]["rates_on_grids"][group_key]
        q = grp["positions"][()]
        rates = grp["rates"][()]
    return clean_q_and_rates(q, rates)

# ============================================================
# Analysis
# ============================================================
def analyze_dataset(dataset_key, label, q, draws, outdir):
    ensure_dir(outdir)

    q = np.asarray(q, dtype=float)
    draws = np.asarray(draws, dtype=float)

    # Published/reference curves from released posterior grid
    y_med_ref, y_lo_ref, y_hi_ref = median_and_ci(draws, axis=0)

    # Median PySR fit
    med_fit = fit_pysr_curve(q, y_med_ref)
    y_med_pysr = np.exp(med_fit["model"].predict(q.reshape(-1, 1)))

    q_dense = np.linspace(np.min(q), np.max(q), N_EVAL)
    y_pysr_dense = np.exp(med_fit["model"].predict(q_dense.reshape(-1, 1)))
    med_diag = compute_shape_diagnostics(q_dense, y_pysr_dense)

    # Draw-by-draw PySR fits for PySR CI
    idx = sample_draw_indices(draws.shape[0], N_DRAW_FITS)
    draw_diags = []
    draw_rows = []
    pysr_draw_curves = []

    for i in idx:
        y = draws[i]
        try:
            fit = fit_pysr_curve(q, y)

            y_on_q = np.exp(fit["model"].predict(q.reshape(-1, 1)))
            pysr_draw_curves.append(y_on_q)

            y_dense = np.exp(fit["model"].predict(q_dense.reshape(-1, 1)))
            diag = compute_shape_diagnostics(q_dense, y_dense)
            draw_diags.append(diag)

            draw_rows.append({
                "draw_index": int(i),
                "mse_pysr": fit["mse"],
                "equation": fit["equation"],
                "sympy": fit["sympy"],
                "mse_constant": fit["null_models"]["constant"]["mse"],
                "mse_power_law": fit["null_models"]["power_law"]["mse"],
                "mse_linear_q": fit["null_models"]["linear_q"]["mse"],
                "mse_quadratic_q": fit["null_models"]["quadratic_q"]["mse"],
                **asdict(diag),
            })
        except Exception as e:
            draw_rows.append({
                "draw_index": int(i),
                "error": str(e),
            })

    summary = summarize_draw_diags(draw_diags)

    if len(pysr_draw_curves) > 0:
        pysr_draw_curves = np.asarray(pysr_draw_curves, dtype=float)
        y_lo_pysr = np.nanpercentile(pysr_draw_curves, 5, axis=0)
        y_hi_pysr = np.nanpercentile(pysr_draw_curves, 95, axis=0)
        n_success = int(pysr_draw_curves.shape[0])
    else:
        y_lo_pysr = y_med_pysr.copy()
        y_hi_pysr = y_med_pysr.copy()
        n_success = 0

    y_lo_pysr = sanitize_positive(y_lo_pysr)
    y_hi_pysr = sanitize_positive(y_hi_pysr)

    result = {
        "dataset_key": dataset_key,
        "label": label,
        "n_total_draws": int(draws.shape[0]),
        "n_q": int(len(q)),
        "n_successful_pysr_draw_fits": n_success,
        "plot_data": {
            "q": q.tolist(),
            "y_med_ref": y_med_ref.tolist(),
            "y_lo_ref": y_lo_ref.tolist(),
            "y_hi_ref": y_hi_ref.tolist(),
            "y_med_pysr": y_med_pysr.tolist(),
            "y_lo_pysr": y_lo_pysr.tolist(),
            "y_hi_pysr": y_hi_pysr.tolist(),
            "q_dense": q_dense.tolist(),
            "y_pysr_dense": y_pysr_dense.tolist(),
        },
        "median_fit": {
            "mse_pysr": med_fit["mse"],
            "equation": med_fit["equation"],
            "sympy": med_fit["sympy"],
            "mse_constant": med_fit["null_models"]["constant"]["mse"],
            "mse_power_law": med_fit["null_models"]["power_law"]["mse"],
            "mse_linear_q": med_fit["null_models"]["linear_q"]["mse"],
            "mse_quadratic_q": med_fit["null_models"]["quadratic_q"]["mse"],
            "diagnostics": asdict(med_diag),
        },
        "draw_summary": summary,
    }

    save_json(result, os.path.join(outdir, "summary.json"))
    pd.DataFrame(draw_rows).to_csv(os.path.join(outdir, "draw_by_draw_results.csv"), index=False)

    if SAVE_INDIVIDUAL_FIGURES:
        plot_dataset(
            os.path.join(outdir, "median_fit.png"),
            dataset_key,
            label,
            q,
            y_med_ref,
            y_lo_ref,
            y_hi_ref,
            y_med_pysr,
            y_lo_pysr,
            y_hi_pysr,
        )

    return result

# ============================================================
# Main
# ============================================================
def main():
    ensure_dir(OUTDIR)

    all_results = {}

    analyses = {
        "dP_dq_lowMassPeak": "Low-mass peak conditioned p(q)",
        "dP_dq_highMassPeak": "High-mass peak conditioned p(q)",
    }

    for key, label in analyses.items():
        q, draws = load_group_positions_rates(FILE_VARYQ, key)
        res = analyze_dataset(
            dataset_key=key,
            label=label,
            q=q,
            draws=draws,
            outdir=os.path.join(OUTDIR, key),
        )
        all_results[key] = res

    save_json(all_results, os.path.join(OUTDIR, "all_results_summary.json"))

    plot_two_panel_figure(
        os.path.join(OUTDIR, "lowmass_highmass_comparison.png"),
        "dP_dq_lowMassPeak",
        all_results["dP_dq_lowMassPeak"],
        "Low-mass peak conditioned p(q)",
        "dP_dq_highMassPeak",
        all_results["dP_dq_highMassPeak"],
        "High-mass peak conditioned p(q)",
        "Mass-dependent pairing: low-mass vs high-mass peak",
    )

    n_datasets = 2
    total_runs = n_datasets * (1 + N_DRAW_FITS)

    print("\n================ DONE ================\n")
    print(f"Results saved in: {OUTDIR}")
    print(f"Datasets analyzed: {n_datasets}")
    print(f"PySR runs per dataset: {1 + N_DRAW_FITS}")
    print(f"Total PySR runs: {total_runs}")
    print(f"Combined figure: {os.path.join(OUTDIR, 'lowmass_highmass_comparison.png')}\n")

    for name, res in all_results.items():
        med = res["median_fit"]
        diag = med["diagnostics"]
        ds = res["draw_summary"]

        print(name)
        print(f"  PySR median equation: {med['equation']}")
        print(f"  median MSE (PySR):     {med['mse_pysr']:.6g}")
        print(f"  median MSE (powerlaw): {med['mse_power_law']:.6g}")
        print(f"  q_peak_median_fit:     {diag['q_peak']:.4f}")
        print(f"  slope_high_median_fit: {diag['slope_high']:.4f}")
        print(f"  rises to q~1?:         {diag['rises_toward_equal_mass']}")
        print(f"  interior peak?:        {diag['has_interior_peak']}")
        print(f"  successful PySR draw fits: {res['n_successful_pysr_draw_fits']}")
        print(f"  Pr(rises to q~1):      {ds.get('Pr_rises_toward_equal_mass', np.nan):.3f}")
        print(f"  Pr(interior peak):     {ds.get('Pr_interior_peak', np.nan):.3f}")
        print(f"  Pr(peak then decline): {ds.get('Pr_peaks_then_declines_near_unity', np.nan):.3f}")
        print()

if __name__ == "__main__":
    main()