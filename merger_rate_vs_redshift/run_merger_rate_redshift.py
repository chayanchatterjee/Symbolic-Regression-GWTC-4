# Full PySR analysis for GWTC-4 Figure 10:
# Redshift evolution of the BBH comoving merger rate R(z)
#
# This script:
#   1. Loads published GWTC-4 rate-vs-redshift posterior grids for:
#        - PowerLawRedshift
#        - BSplineIID
#   2. Fits PySR to:
#        - the published median curve
#        - many posterior draws
#   3. Produces a Figure-10-style comparison plot with:
#        - GWTC-4 published median
#        - GWTC-4 published 90% CI
#        - PySR posterior median
#        - PySR posterior 90% CI
#   4. Computes diagnostics:
#        - z_peak from symbolic fits
#        - low-z slope gamma = dlnR/dln(1+z)
#        - turnover probability Pr(dR/dz < 0 at z_star)
#        - held-out comparison vs pure (1+z)^kappa
#   5. Adds:
#        - functional-form classification
#        - robustness tests across low-z windows and z_star values
#
# Notes:
#   - The published GWTC-4 paper reports that the BBH merger rate increases
#     with redshift approximately as (1 + z)^kappa with kappa ~ 3.2.
#   - The released HDF5 files use two different group names:
#       PowerLawRedshift: posterior/rates_on_grids/redshift/{positions,rates}
#       BSplineIID:       posterior/rates_on_grids/rate_vs_redshift/{positions,rates}
#

import os
import json
import warnings
import numpy as np
import h5py
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from pysr import PySRRegressor

warnings.filterwarnings("ignore")


# ============================================================
# USER SETTINGS
# ============================================================

DATADIR = "."

POWERLAW_FILE = os.path.join(
    DATADIR,
    "BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5",
)
BSPLINE_FILE = os.path.join(
    DATADIR,
    "BBHMassSpinRedshift_BSplineIID.h5",
)

OUTDIR = "analysis_2_redshift_full"
os.makedirs(OUTDIR, exist_ok=True)

# Figure-10-style axes
# These are chosen to closely match the published Figure 10 style.
# If your local released figure_10.py uses slightly different limits,
# update only these two lines.
FIG10_XLIM = (0.0, 1.5)
FIG10_YLIM = (5.0, 2000.0)

# Main diagnostics
DEFAULT_LOWZ_MIN = 0.1
DEFAULT_LOWZ_MAX = 0.3
DEFAULT_LOWZ_SUMMARY = "mean"   # "mean" or "median"
DEFAULT_ZSTAR = 1.0

# Robustness study
ROBUST_LOWZ_WINDOWS = [
    (0.05, 0.20),
    (0.10, 0.30),
    (0.10, 0.40),
    (0.20, 0.40),
]
ROBUST_ZSTARS = [0.5, 0.75, 1.25]

# PySR controls
FIT_LOG10_RATE = True
SMOOTH_TARGET = True
SAVGOL_WINDOW = 21
SAVGOL_POLY = 3

NITER = 500
MAXSIZE = 28
MAXDEPTH = 12
N_DRAW_FITS = 200

UNARY_OPS = ["exp", "log", "sqrt", "tanh", "abs"]
BINARY_OPS = ["+", "-", "*", "/"]

# Plot styling
ALPHA_GWTC4 = 0.18
ALPHA_PYSR = 0.22
LW_MEDIAN = 2.5
LW_PYSR = 2.7

# Colors
COLOR_PL_GWTC4 = "#1f77b4"   # blue
COLOR_PL_PYSR = "#ff7f0e"    # orange
COLOR_BS_GWTC4 = "#2ca02c"   # green
COLOR_BS_PYSR = "#d62728"    # red

RNG = np.random.default_rng(12345)


# ============================================================
# LOADING
# ============================================================

def load_rate_vs_redshift(filename):
    """
    Handles both released file structures:
      PowerLawRedshift: posterior/rates_on_grids/redshift/{positions,rates}
      BSplineIID:       posterior/rates_on_grids/rate_vs_redshift/{positions,rates}
    """
    candidate_pairs = [
        (
            "posterior/rates_on_grids/rate_vs_redshift/positions",
            "posterior/rates_on_grids/rate_vs_redshift/rates",
        ),
        (
            "posterior/rates_on_grids/redshift/positions",
            "posterior/rates_on_grids/redshift/rates",
        ),
    ]

    with h5py.File(filename, "r") as f:
        for zpath, rpath in candidate_pairs:
            if zpath in f and rpath in f:
                z = np.asarray(f[zpath])
                rates = np.asarray(f[rpath])

                z = np.squeeze(z)
                if z.ndim == 2 and z.shape[0] == 1:
                    z = z[0]

                rates = np.asarray(rates)
                if rates.ndim != 2:
                    raise ValueError(f"{filename}: expected 2D rate array, got {rates.shape}")

                # Ensure shape = (Ndraw, Nz)
                if rates.shape[0] == len(z) and rates.shape[1] != len(z):
                    rates = rates.T

                if rates.shape[-1] != len(z):
                    raise ValueError(
                        f"{filename}: rate shape {rates.shape} incompatible with len(z)={len(z)}"
                    )

                rates = np.clip(rates, 1e-30, None)

                return {
                    "z": z,
                    "rates": rates,
                    "source": f"{zpath} | {rpath}",
                }

    raise KeyError(
        f"Could not find supported redshift-rate datasets in {filename}."
    )


# ============================================================
# BASIC MATH HELPERS
# ============================================================

def smooth_curve(y, window=21, poly=3):
    if len(y) < 7:
        return y.copy()
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < poly + 2:
        return y.copy()
    if window % 2 == 0:
        window -= 1
    return savgol_filter(y, window_length=window, polyorder=poly)

def numerical_derivative(x, y):
    return np.gradient(y, x)

def second_derivative(x, y):
    return np.gradient(np.gradient(y, x), x)

def z_peak_from_curve(z, R):
    return float(z[np.argmax(R)])

def dRdz_at(z, R, z_eval):
    dR_dz = numerical_derivative(z, R)
    idx = np.argmin(np.abs(z - z_eval))
    return float(dR_dz[idx])

def gamma_lowz_window(z, R, zmin=0.1, zmax=0.3, summary="mean"):
    """
    gamma(z) = dlnR / dln(1+z) = ((1+z)/R) * dR/dz
    summarized over a low-z window.
    """
    z = np.asarray(z)
    R = np.asarray(R)

    dR_dz = numerical_derivative(z, R)
    gamma = ((1.0 + z) / R) * dR_dz

    mask = (z >= zmin) & (z <= zmax)
    z_win = z[mask]
    gamma_win = gamma[mask]

    if len(z_win) == 0:
        raise ValueError(f"No grid points in low-z window [{zmin}, {zmax}]")

    if summary == "mean":
        gamma_summary = float(np.mean(gamma_win))
    elif summary == "median":
        gamma_summary = float(np.median(gamma_win))
    else:
        raise ValueError("summary must be 'mean' or 'median'")

    return gamma_summary, z_win, gamma_win


# ============================================================
# SIMPLE BASELINE MODEL
# ============================================================

def powerlaw_model(z, A, kappa):
    return A * (1.0 + z) ** kappa

def fit_pure_powerlaw(z_train, R_train):
    p0 = [float(np.median(R_train)), 2.0]
    pars, _ = curve_fit(
        powerlaw_model,
        z_train,
        R_train,
        p0=p0,
        bounds=([1e-30, -20], [1e30, 20]),
        maxfev=20000,
    )
    return pars

def mse(a, b):
    return float(np.mean((a - b) ** 2))


# ============================================================
# PySR
# ============================================================

def build_pysr():
    return PySRRegressor(
        niterations=NITER,
        binary_operators=BINARY_OPS,
        unary_operators=UNARY_OPS,
        model_selection="best",
        loss="loss(x, y) = (x - y)^2",
        maxsize=MAXSIZE,
        maxdepth=MAXDEPTH,
        parsimony=1e-4,
        population_size=50,
        populations=10,
        turbo=True,
        progress=False,
        verbosity=0,
        random_state=0,
    )

def fit_symbolic_rate(z, R):
    X = z.reshape(-1, 1)

    if FIT_LOG10_RATE:
        y = np.log10(np.clip(R, 1e-30, None))
    else:
        y = R.copy()

    if SMOOTH_TARGET:
        y = smooth_curve(y, SAVGOL_WINDOW, SAVGOL_POLY)

    model = build_pysr()
    model.fit(X, y)
    y_pred = model.predict(X)

    if FIT_LOG10_RATE:
        R_pred = 10 ** y_pred
    else:
        R_pred = y_pred

    return model, R_pred, y, y_pred

def heldout_sr_vs_powerlaw(z, R, train_fraction=0.7):
    n = len(z)
    ntr = max(5, int(train_fraction * n))

    z_train, z_test = z[:ntr], z[ntr:]
    R_train, R_test = R[:ntr], R[ntr:]

    # SR
    sr_model, _, _, _ = fit_symbolic_rate(z_train, R_train)
    if FIT_LOG10_RATE:
        R_sr_test = 10 ** sr_model.predict(z_test.reshape(-1, 1))
        sr_loss = mse(np.log10(R_test), np.log10(R_sr_test))
    else:
        R_sr_test = sr_model.predict(z_test.reshape(-1, 1))
        sr_loss = mse(R_test, R_sr_test)

    # Pure power law
    A, kappa = fit_pure_powerlaw(z_train, R_train)
    R_pl_test = np.clip(powerlaw_model(z_test, A, kappa), 1e-30, None)

    if FIT_LOG10_RATE:
        pl_loss = mse(np.log10(R_test), np.log10(R_pl_test))
    else:
        pl_loss = mse(R_test, R_pl_test)

    return {
        "sr_test_loss": float(sr_loss),
        "powerlaw_test_loss": float(pl_loss),
        "delta_loss_powerlaw_minus_sr": float(pl_loss - sr_loss),
        "powerlaw_A": float(A),
        "powerlaw_kappa": float(kappa),
    }


# ============================================================
# NEW ANALYSES
# ============================================================

def classify_functional_form(z, R, tol_rel_slope=0.02):
    """
    Heuristic classification:
      - monotonic_powerlaw_like
      - peaked
      - plateau_like
      - saturating
      - broken_rise
      - complex
    """
    z = np.asarray(z)
    R = np.asarray(R)

    d1 = numerical_derivative(z, R)
    d2 = second_derivative(z, R)

    slope_scale = np.max(np.abs(d1)) + 1e-12
    near_zero = np.abs(d1) < tol_rel_slope * slope_scale

    # interior peak?
    imax = np.argmax(R)
    interior_peak = (imax > 0) and (imax < len(R) - 1)

    # monotonic?
    monotonic_up = np.all(d1 >= -1e-8)
    monotonic_down = np.all(d1 <= 1e-8)

    # plateau-like: has a long region with near-zero slope
    plateau_fraction = np.mean(near_zero)

    # saturating: rises then slope shrinks toward zero at large z
    tail = d1[int(0.75 * len(d1)):]
    tail_small = np.mean(np.abs(tail)) < 0.15 * slope_scale

    # broken-rise: monotonic up but clear curvature change
    curvature_change = np.sum(np.diff(np.sign(d2)) != 0)

    if interior_peak:
        return "peaked"
    if monotonic_up and tail_small:
        if plateau_fraction > 0.20:
            return "plateau_like"
        return "saturating"
    if monotonic_up and curvature_change >= 1:
        return "broken_rise"
    if monotonic_up:
        return "monotonic_powerlaw_like"
    if monotonic_down:
        return "monotonic_decreasing"
    return "complex"

def robustness_study(z, rate_draws, draw_ids, lowz_windows, zstars):
    """
    Recompute low-z slope and turnover probabilities under
    different analysis choices.
    """
    results = {
        "lowz_windows": [],
        "zstars": [],
    }

    # Symbolic predictions for draws, reused
    symbolic_draws = []
    for idx in draw_ids:
        Rj = np.clip(rate_draws[idx], 1e-30, None)
        try:
            _, Rj_sr, _, _ = fit_symbolic_rate(z, Rj)
            symbolic_draws.append(Rj_sr)
        except Exception:
            continue
    symbolic_draws = np.asarray(symbolic_draws)

    # low-z window robustness
    for (zmin, zmax) in lowz_windows:
        vals = []
        for Rj_sr in symbolic_draws:
            try:
                g, _, _ = gamma_lowz_window(z, Rj_sr, zmin=zmin, zmax=zmax, summary="mean")
                vals.append(g)
            except Exception:
                continue
        vals = np.asarray(vals)

        results["lowz_windows"].append({
            "window": [float(zmin), float(zmax)],
            "n_success": int(len(vals)),
            "median": float(np.median(vals)) if len(vals) else None,
            "p05": float(np.percentile(vals, 5)) if len(vals) else None,
            "p95": float(np.percentile(vals, 95)) if len(vals) else None,
        })

    # z_star robustness
    for zstar in zstars:
        vals = []
        for Rj_sr in symbolic_draws:
            vals.append(dRdz_at(z, Rj_sr, zstar))
        vals = np.asarray(vals)

        results["zstars"].append({
            "zstar": float(zstar),
            "n_success": int(len(vals)),
            "pr_declining": float(np.mean(vals < 0.0)) if len(vals) else None,
            "median_dRdz": float(np.median(vals)) if len(vals) else None,
            "p05_dRdz": float(np.percentile(vals, 5)) if len(vals) else None,
            "p95_dRdz": float(np.percentile(vals, 95)) if len(vals) else None,
        })

    return results


# ============================================================
# MAIN MODEL ANALYSIS
# ============================================================

def analyze_model(label, filename):
    data = load_rate_vs_redshift(filename)
    z = data["z"]
    rate_draws = data["rates"]

    # Published GWTC-4 summaries
    R50 = np.percentile(rate_draws, 50, axis=0)
    R05 = np.percentile(rate_draws, 5, axis=0)
    R95 = np.percentile(rate_draws, 95, axis=0)

    # PySR fit to published median
    sr_med, R_med_sr, _, _ = fit_symbolic_rate(z, R50)

    # Draw-by-draw PySR fits
    n_total = rate_draws.shape[0]
    draw_ids = RNG.choice(n_total, size=min(N_DRAW_FITS, n_total), replace=False)

    z_peak_samples = []
    gamma_samples = []
    dRdz_samples = []
    equations = []
    symbolic_preds = []

    for idx in draw_ids:
        Rj = np.clip(rate_draws[idx], 1e-30, None)
        try:
            sr_j, Rj_sr, _, _ = fit_symbolic_rate(z, Rj)
            symbolic_preds.append(Rj_sr)
            equations.append(str(sr_j.get_best()["equation"]))

            z_peak_samples.append(z_peak_from_curve(z, Rj_sr))

            g, _, _ = gamma_lowz_window(
                z, Rj_sr,
                zmin=DEFAULT_LOWZ_MIN,
                zmax=DEFAULT_LOWZ_MAX,
                summary=DEFAULT_LOWZ_SUMMARY,
            )
            gamma_samples.append(g)

            dRdz_samples.append(dRdz_at(z, Rj_sr, DEFAULT_ZSTAR))
        except Exception as e:
            print(f"[{label}] draw {idx} failed: {e}")

    symbolic_preds = np.asarray(symbolic_preds)
    z_peak_samples = np.asarray(z_peak_samples)
    gamma_samples = np.asarray(gamma_samples)
    dRdz_samples = np.asarray(dRdz_samples)

    if len(symbolic_preds) == 0:
        raise RuntimeError(f"{label}: no successful draw-by-draw PySR fits")

    # PySR posterior summaries from draw-by-draw fits
    Rsr05 = np.percentile(symbolic_preds, 5, axis=0)
    Rsr50 = np.percentile(symbolic_preds, 50, axis=0)
    Rsr95 = np.percentile(symbolic_preds, 95, axis=0)

    # Median-curve diagnostics
    gamma_med, z_gamma_med, gamma_curve_med = gamma_lowz_window(
        z, R_med_sr,
        zmin=DEFAULT_LOWZ_MIN,
        zmax=DEFAULT_LOWZ_MAX,
        summary=DEFAULT_LOWZ_SUMMARY,
    )
    z_peak_med = z_peak_from_curve(z, R_med_sr)
    dRdz_med = dRdz_at(z, R_med_sr, DEFAULT_ZSTAR)

    # Existing tests
    heldout = heldout_sr_vs_powerlaw(z, R50)

    # New tests
    class_med = classify_functional_form(z, R_med_sr)
    class_samples = [classify_functional_form(z, rr) for rr in symbolic_preds]

    # robustness
    robust = robustness_study(
        z,
        rate_draws,
        draw_ids,
        lowz_windows=ROBUST_LOWZ_WINDOWS,
        zstars=ROBUST_ZSTARS,
    )

    return {
        "label": label,
        "filename": filename,
        "source": data["source"],

        # Published GWTC-4
        "z": z,
        "R50": R50,
        "R05": R05,
        "R95": R95,

        # PySR
        "median_equation": str(sr_med.get_best()["equation"]),
        "R_med_sr": R_med_sr,
        "Rsr05": Rsr05,
        "Rsr50": Rsr50,
        "Rsr95": Rsr95,
        "n_symbolic_band_fits": int(len(symbolic_preds)),

        # Existing diagnostics
        "z_peak_median": float(z_peak_med),
        "gamma0_definition": f"{DEFAULT_LOWZ_SUMMARY} of dlnR/dln(1+z) over z in [{DEFAULT_LOWZ_MIN}, {DEFAULT_LOWZ_MAX}]",
        "gamma0_window_value": float(gamma_med),
        "dRdz_at_zstar_median": float(dRdz_med),
        "turnover_probability_at_zstar": float(np.mean(dRdz_samples < 0.0)),
        "heldout_comparison": heldout,

        "z_peak_draws_median": float(np.median(z_peak_samples)),
        "z_peak_draws_5pct": float(np.percentile(z_peak_samples, 5)),
        "z_peak_draws_95pct": float(np.percentile(z_peak_samples, 95)),
        "gamma0_draws_median": float(np.median(gamma_samples)),
        "gamma0_draws_5pct": float(np.percentile(gamma_samples, 5)),
        "gamma0_draws_95pct": float(np.percentile(gamma_samples, 95)),

        # New diagnostics
        "functional_form_median": class_med,
        "functional_form_counts": {
            c: int(sum(cc == c for cc in class_samples))
            for c in sorted(set(class_samples))
        },
        "robustness": robust,

        # Arrays
        "gamma_window_z": z_gamma_med,
        "gamma_window_values_median_fit": gamma_curve_med,
        "z_peak_samples": z_peak_samples,
        "gamma0_samples": gamma_samples,
        "dRdz_samples": dRdz_samples,
        "equations": equations,
    }


# ============================================================
# PLOTTING
# ============================================================

def _clipy(y):
    return np.clip(y, FIG10_YLIM[0] * 0.5, None)

def make_figure10_style_plot(res_pl, res_bs):
    """
    Main requested plot:
      - same Figure 10 style
      - log y-axis
      - published GWTC-4 median + 90% CI
      - PySR posterior median + 90% CI
      - both PowerLawRedshift and BSplineIID
    """
    plt.figure(figsize=(9.2, 6.0))

    # Published GWTC-4 bands
    plt.fill_between(
        res_pl["z"], _clipy(res_pl["R05"]), _clipy(res_pl["R95"]),
        color=COLOR_PL_GWTC4, alpha=ALPHA_GWTC4,
        label="GWTC-4 PowerLawRedshift 90% CI"
    )
    plt.fill_between(
        res_bs["z"], _clipy(res_bs["R05"]), _clipy(res_bs["R95"]),
        color=COLOR_BS_GWTC4, alpha=ALPHA_GWTC4,
        label="GWTC-4 BSplineIID 90% CI"
    )

    # Published GWTC-4 medians
    plt.plot(
        res_pl["z"], _clipy(res_pl["R50"]),
        color=COLOR_PL_GWTC4, lw=LW_MEDIAN,
        label="GWTC-4 PowerLawRedshift median"
    )
    plt.plot(
        res_bs["z"], _clipy(res_bs["R50"]),
        color=COLOR_BS_GWTC4, lw=LW_MEDIAN,
        label="GWTC-4 BSplineIID median"
    )

    # PySR bands
    plt.fill_between(
        res_pl["z"], _clipy(res_pl["Rsr05"]), _clipy(res_pl["Rsr95"]),
        color=COLOR_PL_PYSR, alpha=ALPHA_PYSR,
        label="PySR PowerLawRedshift 90% CI"
    )
    plt.fill_between(
        res_bs["z"], _clipy(res_bs["Rsr05"]), _clipy(res_bs["Rsr95"]),
        color=COLOR_BS_PYSR, alpha=ALPHA_PYSR,
        label="PySR BSplineIID 90% CI"
    )

    # PySR medians
    plt.plot(
        res_pl["z"], _clipy(res_pl["Rsr50"]),
        color=COLOR_PL_PYSR, lw=LW_PYSR, ls="--",
        label="PySR PowerLawRedshift median"
    )
    plt.plot(
        res_bs["z"], _clipy(res_bs["Rsr50"]),
        color=COLOR_BS_PYSR, lw=LW_PYSR, ls="--",
        label="PySR BSplineIID median"
    )

    plt.yscale("log")
    plt.xlim(*FIG10_XLIM)
    plt.ylim(*FIG10_YLIM)

    plt.xlabel("Redshift z", fontsize=14)
    plt.ylabel(r"BBH merger rate $R(z)$", fontsize=14)
    plt.title("Figure 10-style comparison: GWTC-4 and PySR", fontsize=16)
    plt.legend(fontsize=10, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figure10_style_GWTC4_vs_PySR.png"), dpi=220)
    plt.close()

def make_gamma_window_plot(res_pl, res_bs):
    plt.figure(figsize=(8.4, 5.0))
    plt.plot(
        res_pl["gamma_window_z"], res_pl["gamma_window_values_median_fit"],
        color=COLOR_PL_PYSR, lw=2.4, label="PowerLawRedshift median-fit gamma(z)"
    )
    plt.plot(
        res_bs["gamma_window_z"], res_bs["gamma_window_values_median_fit"],
        color=COLOR_BS_PYSR, lw=2.4, label="BSplineIID median-fit gamma(z)"
    )
    plt.axhline(0.0, ls="--", lw=1.2, color="k")
    plt.xlabel("Redshift z")
    plt.ylabel(r"$d\ln R / d\ln(1+z)$")
    plt.title(f"Low-z slope window: z in [{DEFAULT_LOWZ_MIN}, {DEFAULT_LOWZ_MAX}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "compare_gamma_window.png"), dpi=220)
    plt.close()

def make_hist_plot(res_pl, res_bs):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    axs[0].hist(res_pl["z_peak_samples"], bins=18, alpha=0.70, label="PowerLawRedshift")
    axs[0].hist(res_bs["z_peak_samples"], bins=18, alpha=0.70, label="BSplineIID")
    axs[0].set_xlabel(r"$z_{\rm peak}$")
    axs[0].set_ylabel("Count")
    axs[0].legend()

    axs[1].hist(res_pl["gamma0_samples"], bins=18, alpha=0.70, label="PowerLawRedshift")
    axs[1].hist(res_bs["gamma0_samples"], bins=18, alpha=0.70, label="BSplineIID")
    axs[1].set_xlabel(r"window-averaged $d\ln R/d\ln(1+z)$")
    axs[1].set_ylabel("Count")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "compare_diagnostics_hist.png"), dpi=220)
    plt.close()


# ============================================================
# SAVING
# ============================================================

def save_arrays(res):
    np.savez(
        os.path.join(OUTDIR, f"{res['label']}_arrays.npz"),
        z=res["z"],
        R50=res["R50"],
        R05=res["R05"],
        R95=res["R95"],
        R_med_sr=res["R_med_sr"],
        Rsr05=res["Rsr05"],
        Rsr50=res["Rsr50"],
        Rsr95=res["Rsr95"],
        z_peak_samples=res["z_peak_samples"],
        gamma0_samples=res["gamma0_samples"],
        dRdz_samples=res["dRdz_samples"],
        gamma_window_z=res["gamma_window_z"],
        gamma_window_values_median_fit=res["gamma_window_values_median_fit"],
    )

    with open(os.path.join(OUTDIR, f"{res['label']}_equations.txt"), "w") as f:
        f.write("Median-fit equation:\n")
        f.write(res["median_equation"] + "\n\n")
        f.write("Draw-by-draw equations:\n")
        for eq in res["equations"]:
            f.write(eq + "\n")

def write_summary_json(res_pl, res_bs):
    def strip_arrays(d):
        return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

    out = {
        "figure10_style_limits": {
            "xlim": list(FIG10_XLIM),
            "ylim": list(FIG10_YLIM),
        },
        "default_lowz_window": [DEFAULT_LOWZ_MIN, DEFAULT_LOWZ_MAX],
        "default_zstar": DEFAULT_ZSTAR,
        "PowerLawRedshift": strip_arrays(res_pl),
        "BSplineIID": strip_arrays(res_bs),
    }

    with open(os.path.join(OUTDIR, "analysis_summary.json"), "w") as f:
        json.dump(out, f, indent=2)


# ============================================================
# MAIN
# ============================================================

def print_summary(res):
    print(f"{res['label']}:")
    print(f"  file                     = {res['filename']}")
    print(f"  source                   = {res['source']}")
    print(f"  median equation          = {res['median_equation']}")
    print(f"  z_peak_median            = {res['z_peak_median']:.4f}")
    print(f"  gamma0_window_definition = {res['gamma0_definition']}")
    print(f"  gamma0_window_value      = {res['gamma0_window_value']:.4f}")
    print(f"  Pr(dR/dz<0 @ z={DEFAULT_ZSTAR}) = {res['turnover_probability_at_zstar']:.4f}")
    print(f"  heldout delta_loss       = {res['heldout_comparison']['delta_loss_powerlaw_minus_sr']:.6f}")
    print(f"  n_symbolic_band_fits     = {res['n_symbolic_band_fits']}")
    print(f"  functional_form_median   = {res['functional_form_median']}")
    print(f"  functional_form_counts   = {res['functional_form_counts']}")
    print()

def main():
    res_pl = analyze_model("PowerLawRedshift", POWERLAW_FILE)
    res_bs = analyze_model("BSplineIID", BSPLINE_FILE)

    save_arrays(res_pl)
    save_arrays(res_bs)

    make_figure10_style_plot(res_pl, res_bs)
    make_gamma_window_plot(res_pl, res_bs)
    make_hist_plot(res_pl, res_bs)
    write_summary_json(res_pl, res_bs)

    print("\n================ SUMMARY ================\n")
    print_summary(res_pl)
    print_summary(res_bs)
    print(f"Outputs saved in: {OUTDIR}")

if __name__ == "__main__":
    main()