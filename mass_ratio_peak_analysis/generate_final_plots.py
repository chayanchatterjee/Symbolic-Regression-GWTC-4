import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Input / output
# ============================================================
REFIT_OUTDIR = Path("pysr_q_lowmass_highmass_only_refit_median_only")
LOW_SUMMARY = REFIT_OUTDIR / "dP_dq_lowMassPeak" / "summary_refit_median_only.json"
HIGH_SUMMARY = REFIT_OUTDIR / "dP_dq_highMassPeak" / "summary_refit_median_only.json"

SAVEFIG = REFIT_OUTDIR / "lowmass_highmass_comparison_refit_median_only_serif.png"

# ============================================================
# Same axis limits as before
# ============================================================
YLIMS = {
    "dP_dq_lowMassPeak": (1e-3, 1e1),
    "dP_dq_highMassPeak": (1e-3, 1e1),
}

# ============================================================
# Font / style to match the attached plot more closely
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "axes.linewidth": 0.8,
})

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

def load_refit_summary(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_plot_data(summary):
    pdata = summary["plot_data"]
    q = np.asarray(pdata["q"], dtype=float)

    y_med_ref = sanitize_positive(pdata["y_med_ref"])
    y_lo_ref = sanitize_positive(pdata["y_lo_ref"])
    y_hi_ref = sanitize_positive(pdata["y_hi_ref"])

    y_lo_pysr = sanitize_positive(pdata["y_lo_pysr"])
    y_hi_pysr = sanitize_positive(pdata["y_hi_pysr"])

    y_med_pysr = sanitize_positive(pdata["y_med_pysr_refit"])

    return q, y_med_ref, y_lo_ref, y_hi_ref, y_med_pysr, y_lo_pysr, y_hi_pysr

def panel_plot(ax, dataset_key, title, summary):
    q, y_med_ref, y_lo_ref, y_hi_ref, y_med_pysr, y_lo_pysr, y_hi_pysr = extract_plot_data(summary)

    ax.fill_between(q, y_lo_ref, y_hi_ref, alpha=0.25, label="GWTC-4 90% CI")
    ax.fill_between(q, y_lo_pysr, y_hi_pysr, alpha=0.20, label="PySR 90% CI")

    ax.plot(q, y_med_ref, lw=2.4, label="GWTC-4 median")
    ax.plot(q, y_med_pysr, "--", lw=2.4, label="Median PySR fit")

    ax.set_yscale("log")
    ax.set_xlim(float(np.min(q)), float(np.max(q)))
    ax.set_ylim(*YLIMS[dataset_key])

    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$p(q)$")
    ax.set_title(title, pad=8)

    ax.legend(frameon=False, loc="upper left")

def main():
    low_summary = load_refit_summary(LOW_SUMMARY)
    high_summary = load_refit_summary(HIGH_SUMMARY)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=150)

    panel_plot(
        axes[0],
        "dP_dq_lowMassPeak",
        "Low-mass peak conditioned p(q)",
        low_summary,
    )

    panel_plot(
        axes[1],
        "dP_dq_highMassPeak",
        "High-mass peak conditioned p(q)",
        high_summary,
    )

#    fig.suptitle("Mass-dependent pairing: low-mass vs high-mass peak", fontsize=19, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(SAVEFIG, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {SAVEFIG}")

if __name__ == "__main__":
    main()#