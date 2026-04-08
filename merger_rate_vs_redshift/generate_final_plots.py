# Re-make plots from saved analysis outputs without rerunning PySR.
#
# Expected files in OUTDIR:
#   - analysis_summary.json
#   - PowerLawRedshift_arrays.npz
#   - BSplineIID_arrays.npz


import os
import json
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.pyplot as plt

# Serif / LaTeX-like figure style
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif", "CMU Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "savefig.dpi": 300,
})


# ============================================================
# SETTINGS
# ============================================================

OUTDIR = "analysis_2_redshift_full"

SUMMARY_JSON = os.path.join(OUTDIR, "analysis_summary.json")
PL_NPZ = os.path.join(OUTDIR, "PowerLawRedshift_arrays.npz")
BS_NPZ = os.path.join(OUTDIR, "BSplineIID_arrays.npz")

# colors
COLOR_PL_GWTC4 = "#1f77b4"   # blue
COLOR_PL_PYSR = "#ff7f0e"    # orange
COLOR_BS_GWTC4 = "#2ca02c"   # green
COLOR_BS_PYSR = "#d62728"    # red

ALPHA_GWTC4 = 0.18
ALPHA_PYSR = 0.22
LW_MEDIAN = 2.5
LW_PYSR = 2.7


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_npz(path):
    return dict(np.load(path, allow_pickle=True))

def clip_positive(y, floor=1e-6):
    return np.clip(y, floor, None)

def window_label(window):
    return f"[{window[0]}, {window[1]}]"

def extract_window_stats(model_summary):
    rows = model_summary["robustness"]["lowz_windows"]
    labels = [window_label(row["window"]) for row in rows]
    med = np.array([row["median"] for row in rows], dtype=float)
    p05 = np.array([row["p05"] for row in rows], dtype=float)
    p95 = np.array([row["p95"] for row in rows], dtype=float)
    return labels, med, p05, p95

def extract_zstar_stats(model_summary):
    rows = model_summary["robustness"]["zstars"]
    zstars = np.array([row["zstar"] for row in rows], dtype=float)
    pr_declining = np.array([row["pr_declining"] for row in rows], dtype=float)
    med_dRdz = np.array([row["median_dRdz"] for row in rows], dtype=float)
    p05_dRdz = np.array([row["p05_dRdz"] for row in rows], dtype=float)
    p95_dRdz = np.array([row["p95_dRdz"] for row in rows], dtype=float)
    return zstars, pr_declining, med_dRdz, p05_dRdz, p95_dRdz


# ============================================================
# PLOTS
# ============================================================

def make_figure10_style_plot(summary, pl, bs):
    xlim = summary["figure10_style_limits"]["xlim"]
    ylim = summary["figure10_style_limits"]["ylim"]

    plt.figure(figsize=(9.2, 6.0))

    # GWTC-4 published bands
    plt.fill_between(
        pl["z"], clip_positive(pl["R05"]), clip_positive(pl["R95"]),
        color=COLOR_PL_GWTC4, alpha=ALPHA_GWTC4,
        label="GWTC-4 PowerLawRedshift 90% CI"
    )
    plt.fill_between(
        bs["z"], clip_positive(bs["R05"]), clip_positive(bs["R95"]),
        color=COLOR_BS_GWTC4, alpha=ALPHA_GWTC4,
        label="GWTC-4 BSplineIID 90% CI"
    )

    # GWTC-4 published medians
    plt.plot(
        pl["z"], clip_positive(pl["R50"]),
        color=COLOR_PL_GWTC4, lw=LW_MEDIAN,
        label="GWTC-4 PowerLawRedshift median"
    )
    plt.plot(
        bs["z"], clip_positive(bs["R50"]),
        color=COLOR_BS_GWTC4, lw=LW_MEDIAN,
        label="GWTC-4 BSplineIID median"
    )

    # PySR posterior bands
    if len(pl["Rsr05"]) > 0:
        plt.fill_between(
            pl["z"], clip_positive(pl["Rsr05"]), clip_positive(pl["Rsr95"]),
            color=COLOR_PL_PYSR, alpha=ALPHA_PYSR,
            label="PySR PowerLawRedshift 90% CI"
        )
        plt.plot(
            pl["z"], clip_positive(pl["Rsr50"]),
            color=COLOR_PL_PYSR, lw=LW_PYSR, ls="--",
            label="PySR PowerLawRedshift median"
        )

    if len(bs["Rsr05"]) > 0:
        plt.fill_between(
            bs["z"], clip_positive(bs["Rsr05"]), clip_positive(bs["Rsr95"]),
            color=COLOR_BS_PYSR, alpha=ALPHA_PYSR,
            label="PySR BSplineIID 90% CI"
        )
        plt.plot(
            bs["z"], clip_positive(bs["Rsr50"]),
            color=COLOR_BS_PYSR, lw=LW_PYSR, ls="--",
            label="PySR BSplineIID median"
        )

    plt.yscale("log")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("z", fontsize=14)
    plt.ylabel(r"BBH merger rate $R(z)$", fontsize=14)
    plt.legend(fontsize=10, ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "figure10_style_GWTC4_vs_PySR_remade_new.png"), dpi=220)
    plt.close()


def make_lowz_slope_window_plot(pl, bs):
    plt.figure(figsize=(8.2, 5.0))

    plt.plot(
        pl["gamma_window_z"],
        pl["gamma_window_values_median_fit"],
        color=COLOR_PL_PYSR,
        lw=2.4,
        label="PowerLawRedshift median-fit"
    )
    plt.plot(
        bs["gamma_window_z"],
        bs["gamma_window_values_median_fit"],
        color=COLOR_BS_PYSR,
        lw=2.4,
        label="BSplineIID median-fit"
    )

    plt.axhline(0.0, ls="--", lw=1.2, color="k")
    plt.xlabel("z", fontsize=14)
    plt.ylabel(r"$d\ln R / d\ln(1+z)$", fontsize=14)
    ymax = max(
        np.max(pl["gamma_window_values_median_fit"]),
        np.max(bs["gamma_window_values_median_fit"])
    )
    plt.ylim(2.5, max(3.4, ymax + 0.1))
    plt.legend(fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "compare_gamma_window_remade_new.png"), dpi=220)
    plt.close()


def make_compare_diagnostics_hist(pl, bs):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    axs[0].hist(
        pl["z_peak_samples"], bins=18, alpha=0.70, density=True,
        label="PowerLawRedshift"
    )
    axs[0].hist(
        bs["z_peak_samples"], bins=18, alpha=0.70, density=True,
        label="BSplineIID"
    )
    axs[0].set_xlabel(r"$z_{\rm peak}$")
    axs[0].set_ylabel("Density")
    axs[0].legend()

    axs[1].hist(
        pl["gamma0_samples"], bins=18, alpha=0.70, density=True,
        label="PowerLawRedshift"
    )
    axs[1].hist(
        bs["gamma0_samples"], bins=18, alpha=0.70, density=True,
        label="BSplineIID"
    )
    axs[1].set_xlabel(r"window-averaged $d\ln R/d\ln(1+z)$")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "compare_diagnostics_hist_remade_new.png"), dpi=220)
    plt.close()


def make_lowz_window_robustness_plot(summary):
    pl_sum = summary["PowerLawRedshift"]
    bs_sum = summary["BSplineIID"]

    labels_pl, pl_med, pl_p05, pl_p95 = extract_window_stats(pl_sum)
    labels_bs, bs_med, bs_p05, bs_p95 = extract_window_stats(bs_sum)

    # assume same windows/order
    labels = labels_pl
    x = np.arange(len(labels))

    plt.figure(figsize=(8.6, 5.0))

    plt.plot(x, pl_med, color=COLOR_PL_PYSR, lw=2.4, marker="o", label="PowerLawRedshift")
    plt.fill_between(x, pl_p05, pl_p95, color=COLOR_PL_PYSR, alpha=0.20)

    plt.plot(x, bs_med, color=COLOR_BS_PYSR, lw=2.4, marker="o", label="BSplineIID")
    plt.fill_between(x, bs_p05, bs_p95, color=COLOR_BS_PYSR, alpha=0.20)

    plt.xticks(x, labels, rotation=20)
    plt.xlabel("Low-z window", fontsize=14)
    plt.ylabel(r"$d\ln R / d\ln(1+z)$", fontsize=14)
    plt.legend(fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "robustness_lowz_windows_new.png"), dpi=220)
    plt.close()


def make_zstar_robustness_plot(summary):
    pl_sum = summary["PowerLawRedshift"]
    bs_sum = summary["BSplineIID"]

    z_pl, pr_pl, _, _, _ = extract_zstar_stats(pl_sum)
    z_bs, pr_bs, _, _, _ = extract_zstar_stats(bs_sum)

    plt.figure(figsize=(8.2, 5.0))

    plt.plot(z_pl, pr_pl, color=COLOR_PL_PYSR, lw=2.4, marker="o", label="PowerLawRedshift")
    plt.plot(z_bs, pr_bs, color=COLOR_BS_PYSR, lw=2.4, marker="o", label="BSplineIID")

    plt.ylim(-0.02, 1.02)
    plt.xlabel(r"$z_\star$", fontsize=14)
    plt.ylabel(r"$\Pr(dR/dz < 0 \ \mathrm{at}\ z_\star)$", fontsize=14)
    plt.legend(fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "robustness_zstar_decline_probability_new.png"), dpi=220)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    summary = load_json(SUMMARY_JSON)
    pl = load_npz(PL_NPZ)
    bs = load_npz(BS_NPZ)

    make_figure10_style_plot(summary, pl, bs)
    make_lowz_slope_window_plot(pl, bs)
    make_compare_diagnostics_hist(pl, bs)
    make_lowz_window_robustness_plot(summary)
    make_zstar_robustness_plot(summary)

    print("Saved updated plots in:", OUTDIR)
    print("Created:")
    print("  figure10_style_GWTC4_vs_PySR_remade_new.png")
    print("  compare_gamma_window_remade_new.png")
    print("  compare_diagnostics_hist_remade_new.png")
    print("  robustness_lowz_windows_new.png")
    print("  robustness_zstar_decline_probability_new.png")


if __name__ == "__main__":
    main()