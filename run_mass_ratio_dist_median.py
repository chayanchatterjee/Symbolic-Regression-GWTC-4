import os
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pysr import PySRRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# Inputs: existing outputs from your previous run
# ============================================================
OLD_OUTDIR = Path("pysr_q_lowmass_highmass_only")
NEW_OUTDIR = Path("pysr_q_lowmass_highmass_only_refit_median_only")

LOW_SUMMARY = OLD_OUTDIR / "dP_dq_lowMassPeak" / "summary.json"
HIGH_SUMMARY = OLD_OUTDIR / "dP_dq_highMassPeak" / "summary.json"

# ============================================================
# Settings
# ============================================================
RNG_SEED = 1234
NITER = 1000
POPULATIONS = 80
MAXSIZE = 45

# Mild smoothing of log-median for the refit only
SMOOTH_LOG_MEDIAN = True
SAVGOL_WINDOW = 31   # must be odd
SAVGOL_POLYORDER = 3

# Figure-6-style limits from the paper
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

def load_summary(path):
    with open(path, "r") as f:
        return json.load(f)

def build_weights(q):
    """
    Emphasize the visually important mid/high-q region so the median fit
    matches the GWTC-4 median better where the astrophysical shape is most visible.
    """
    q = np.asarray(q, dtype=float)
    w = np.ones_like(q)

    # modest emphasis in the mid/high-q regime
    w[(q >= 0.35) & (q < 0.55)] = 1.5
    w[(q >= 0.55) & (q < 0.75)] = 2.5
    w[(q >= 0.75)] = 4.0

    # de-emphasize the tiny low-q tail where everything is extremely small
    w[q < 0.12] = 0.2
    w[(q >= 0.12) & (q < 0.25)] = 0.5

    return w

def smooth_log_curve(y):
    y = sanitize_positive(y)
    ylog = np.log(y)

    if not SMOOTH_LOG_MEDIAN:
        return ylog

    window = min(SAVGOL_WINDOW, len(ylog) if len(ylog) % 2 == 1 else len(ylog) - 1)
    if window < 5:
        return ylog
    if window % 2 == 0:
        window -= 1
    if window <= SAVGOL_POLYORDER:
        return ylog

    return savgol_filter(ylog, window_length=window, polyorder=SAVGOL_POLYORDER)

def build_pysr():
    return PySRRegressor(
        niterations=NITER,
        populations=POPULATIONS,
        maxsize=MAXSIZE,
        model_selection="best",
        elementwise_loss="L2DistLoss()",
        binary_operators=["+", "-", "*"],
        unary_operators=["exp", "sqrt", "square", "log"],
        parsimony=1e-6,
        batching=False,
        random_state=RNG_SEED,
        deterministic=True,
        parallelism="serial",
        warm_start=False,
        verbosity=0,
    )

def fit_median_only_pysr(q, y_med):
    q = np.asarray(q, dtype=float)
    y_med = sanitize_positive(y_med)

    # target is log(y); optionally smooth only for the median refit
    ylog_target = smooth_log_curve(y_med)

    weights = build_weights(q)
    X = q.reshape(-1, 1)

    model = build_pysr()
    model.fit(X, ylog_target, weights=weights)

    ylog_pred = model.predict(X)
    y_pred = np.exp(ylog_pred)

    # Report mismatch against the unsmoothed GWTC-4 median
    mse_vs_true_log_median = float(np.mean((np.log(y_med) - ylog_pred) ** 2))

    best = model.get_best()
    equation = str(best["equation"])
    sympy_expr = str(model.sympy())

    return {
        "model": model,
        "y_pred": y_pred,
        "equation": equation,
        "sympy": sympy_expr,
        "mse_vs_true_log_median": mse_vs_true_log_median,
    }

def plot_dataset(
    outpath,
    dataset_key,
    title,
    q,
    y_med_ref,
    y_lo_ref,
    y_hi_ref,
    y_med_pysr_new,
    y_lo_pysr_old,
    y_hi_pysr_old,
):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.fill_between(q, y_lo_ref, y_hi_ref, alpha=0.25, label="GWTC-4 90% CI")
    ax.fill_between(q, y_lo_pysr_old, y_hi_pysr_old, alpha=0.20, label="PySR 90% CI")
    ax.plot(q, y_med_ref, lw=2, label="GWTC-4 median")
    ax.plot(q, y_med_pysr_new, "--", lw=2, label="Median PySR fit")

    ax.set_yscale("log")
    ax.set_xlim(float(np.min(q)), float(np.max(q)))
    ax.set_ylim(*YLIMS[dataset_key])
    ax.set_xlabel("q")
    ax.set_ylabel("p(q)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def plot_two_panel(
    outpath,
    left_key, left_title, left_data,
    right_key, right_title, right_data,
    suptitle,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, dataset_key, title, data in [
        (axes[0], left_key, left_title, left_data),
        (axes[1], right_key, right_title, right_data),
    ]:
        q = data["q"]
        ax.fill_between(q, data["y_lo_ref"], data["y_hi_ref"], alpha=0.25, label="GWTC-4 90% CI")
        ax.fill_between(q, data["y_lo_pysr_old"], data["y_hi_pysr_old"], alpha=0.20, label="PySR 90% CI")
        ax.plot(q, data["y_med_ref"], lw=2, label="GWTC-4 median")
        ax.plot(q, data["y_med_pysr_new"], "--", lw=2, label="Median PySR fit")

        ax.set_yscale("log")
        ax.set_xlim(float(np.min(q)), float(np.max(q)))
        ax.set_ylim(*YLIMS[dataset_key])
        ax.set_xlabel("q")
        ax.set_ylabel("p(q)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)

def process_one(summary_path, dataset_key, title):
    summary = load_summary(summary_path)

    q = np.asarray(summary["plot_data"]["q"], dtype=float)
    y_med_ref = sanitize_positive(summary["plot_data"]["y_med_ref"])
    y_lo_ref = sanitize_positive(summary["plot_data"]["y_lo_ref"])
    y_hi_ref = sanitize_positive(summary["plot_data"]["y_hi_ref"])
    y_lo_pysr_old = sanitize_positive(summary["plot_data"]["y_lo_pysr"])
    y_hi_pysr_old = sanitize_positive(summary["plot_data"]["y_hi_pysr"])

    fit = fit_median_only_pysr(q, y_med_ref)
    y_med_pysr_new = sanitize_positive(fit["y_pred"])

    outdir = NEW_OUTDIR / dataset_key
    ensure_dir(outdir)

    new_summary = {
        "dataset_key": dataset_key,
        "title": title,
        "plot_data": {
            "q": q.tolist(),
            "y_med_ref": y_med_ref.tolist(),
            "y_lo_ref": y_lo_ref.tolist(),
            "y_hi_ref": y_hi_ref.tolist(),
            # old PySR CI reused as requested
            "y_lo_pysr": y_lo_pysr_old.tolist(),
            "y_hi_pysr": y_hi_pysr_old.tolist(),
            # new median-only PySR fit
            "y_med_pysr_refit": y_med_pysr_new.tolist(),
        },
        "median_refit": {
            "equation": fit["equation"],
            "sympy": fit["sympy"],
            "mse_vs_true_log_median": fit["mse_vs_true_log_median"],
            "used_existing_pysr_ci": True,
            "smoothed_log_median_target": bool(SMOOTH_LOG_MEDIAN),
            "weights_description": "low-q downweighted; mid/high-q emphasized",
        },
    }

    with open(outdir / "summary_refit_median_only.json", "w") as f:
        json.dump(new_summary, f, indent=2)

    plot_dataset(
        outdir / "median_fit_refit_only.png",
        dataset_key,
        title,
        q,
        y_med_ref,
        y_lo_ref,
        y_hi_ref,
        y_med_pysr_new,
        y_lo_pysr_old,
        y_hi_pysr_old,
    )

    return {
        "q": q,
        "y_med_ref": y_med_ref,
        "y_lo_ref": y_lo_ref,
        "y_hi_ref": y_hi_ref,
        "y_lo_pysr_old": y_lo_pysr_old,
        "y_hi_pysr_old": y_hi_pysr_old,
        "y_med_pysr_new": y_med_pysr_new,
        "equation": fit["equation"],
        "sympy": fit["sympy"],
        "mse_vs_true_log_median": fit["mse_vs_true_log_median"],
    }

def main():
    ensure_dir(NEW_OUTDIR)

    low = process_one(
        LOW_SUMMARY,
        "dP_dq_lowMassPeak",
        "Low-mass peak conditioned p(q)",
    )

    high = process_one(
        HIGH_SUMMARY,
        "dP_dq_highMassPeak",
        "High-mass peak conditioned p(q)",
    )

    plot_two_panel(
        NEW_OUTDIR / "lowmass_highmass_comparison_refit_median_only.png",
        "dP_dq_lowMassPeak",
        "Low-mass peak conditioned p(q)",
        low,
        "dP_dq_highMassPeak",
        "High-mass peak conditioned p(q)",
        high,
        "Mass-dependent pairing: low-mass vs high-mass peak",
    )

    print("\n================ DONE ================\n")
    print(f"Old PySR CI reused from: {OLD_OUTDIR}")
    print(f"New outputs saved in:    {NEW_OUTDIR}\n")

    print("Low-mass peak:")
    print(f"  equation: {low['equation']}")
    print(f"  MSE vs GWTC-4 median (log-space): {low['mse_vs_true_log_median']:.6g}\n")

    print("High-mass peak:")
    print(f"  equation: {high['equation']}")
    print(f"  MSE vs GWTC-4 median (log-space): {high['mse_vs_true_log_median']:.6g}\n")

if __name__ == "__main__":
    main()