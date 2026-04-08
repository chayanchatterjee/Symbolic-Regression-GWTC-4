# Symbolic Regression Analysis of GWTC-4 Population Properties

This repository contains the code, results, and figures for applying symbolic regression (SR) to the posterior population inference products from the LIGO--Virgo--KAGRA [GWTC-4 catalog](https://arxiv.org/abs/2508.18083). Using [PySR](https://github.com/MilesCranmer/PySR), compact closed-form analytic expressions are discovered for four key binary black hole (BBH) population relationships, with full posterior uncertainty propagation via draw-by-draw fitting across 200 individual posterior samples.

## Repository Structure

```
.
├── merger_rate_vs_redshift/       # Analysis 1: R(z)
├── chi_eff_vs_q/                  # Analysis 2: chi_eff(q)
├── chi_eff_vs_redshift/           # Analysis 3: chi_eff(z)
├── mass_ratio_peak_analysis/      # Analysis 4: p(q|peak)
└── README.md
```

---

## Analysis 1: BBH Merger Rate vs. Redshift

**Directory:** `merger_rate_vs_redshift/`

Symbolic expressions are fit to the BBH comoving merger rate R(z) for two GWTC-4 models: **PowerLawRedshift** and **BSplineIID**. PySR is run on 200 individual posterior draws to propagate uncertainty.

**Key results:**
- PowerLawRedshift: low-redshift slope &gamma;<sub>0</sub> = 3.18 (90% CI: 2.32 -- 4.02); 194/200 draws classified as monotonic power-law-like; zero turnover probability at z = 1.5
- BSplineIID: &gamma;<sub>0</sub> = 3.47 (90% CI: 0.98 -- 6.36); 125/200 draws peaked; z<sub>peak</sub> = 1.75 (90% CI: 0.84 -- 2.36); turnover probability 38% at z = 1.5

**Median symbolic expressions:**

| Model | Expression (log<sub>10</sub> R space) |
|---|---|
| PowerLawRedshift | `sqrt(z + (((-0.277*z + 2.634)*z) + 1.664)) - 0.083` |
| BSplineIID | `sqrt(sqrt(z)/(z^2 - 1.526*z + 1.016) + z + 1.614)` |

![Merger rate vs redshift](merger_rate_vs_redshift/figure10_style_GWTC4_vs_PySR_remade_new.png)
*BBH comoving merger rate R(z). Shaded bands: GWTC-4 90% credible intervals. Dashed lines: PySR symbolic fits with 90% CI from 200 draw-by-draw fits.*

![Diagnostics](merger_rate_vs_redshift/compare_diagnostics_hist_remade_new.png)
*Distributions of z<sub>peak</sub> (left) and the low-z logarithmic slope &gamma;<sub>0</sub> (right) from 200 draw-by-draw PySR fits.*

**Files:**
- `run_merger_rate_redshift.py` -- Main analysis script
- `generate_final_plots.py` -- Figure generation
- `analysis_summary.json` -- Full quantitative results
- `PowerLawRedshift_equations.txt` / `BSplineIID_equations.txt` -- All 200 draw-by-draw equations

---

## Analysis 2: Effective Spin vs. Mass Ratio

**Directory:** `chi_eff_vs_q/`

Symbolic expressions are fit to &mu;<sub>&chi;eff</sub>(q) and log &sigma;<sub>&chi;eff</sub>(q) for both the **Linear** and **Spline** GWTC-4 models.

**Key results:**
- The mean &mu;<sub>&chi;eff</sub>(q) is model-dependent: the Linear model yields a simple monotonic decline, while the Spline model favors non-monotonic structure with a sign change in d&mu;/dq near q ~ 0.4
- The narrowing of &sigma;<sub>&chi;eff</sub>(q) toward equal mass is robust across both models (Pr[narrows] = 0.984 for Linear, 0.813 for Spline)
- Analytic gradients d&mu;/dq and d&sigma;/dq are computed exactly from the symbolic expressions

![chi_eff vs q](chi_eff_vs_q/new_top_middle_overlay_fixed.png)
*Top: Mean effective spin &mu;<sub>&chi;eff</sub>(q) and width &sigma;<sub>&chi;eff</sub>(q) for the Spline (blue) and Linear (orange) models with PySR overlays.*

![chi_eff vs q gradients](chi_eff_vs_q/new_gradient_overlay_fixed.png)
*Analytic gradients d&mu;/dq (left) and d(ln&sigma;)/dq (right) computed from the PySR symbolic expressions.*

**Files:**
- `run_chi_eff_vs_q.py` -- Main analysis script
- `generate_final_plots.py` -- Figure generation
- `Linear/` and `Spline/` -- Model-specific results including `best_equations.txt`, equation CSVs, and `results.npz`
- `cross_model_summary.csv` -- Cross-model comparison metrics

---

## Analysis 3: Effective Spin vs. Redshift

**Directory:** `chi_eff_vs_redshift/`

Symbolic expressions are fit to &mu;<sub>&chi;eff</sub>(z) and log &sigma;<sub>&chi;eff</sub>(z) for the **Linear** and **Spline** GWTC-4 models.

**Key results:**
- Linear model: &mu; slope ~ -0.015/z; Pr[broadens with z] = 1.0; median d(ln&sigma;)/dz = 0.96
- Spline model: &mu; slope ~ -0.14/z; Pr[&mu; crosses zero] = 85% at z ~ 0.49; low-z broadening Pr = 0.93
- The broadening of &sigma;<sub>&chi;eff</sub> with redshift is robust across both models, while the mean decline is model-dependent

| Diagnostic | Linear | Spline |
|---|---|---|
| Pr[d&mu;/dz < 0, low z] | 0.61 | 0.75 |
| Pr[&mu; crosses zero] | 0.56 | 0.85 |
| Median zero-crossing z | 0.77 | 0.49 |
| Pr[&sigma; broadens with z] | 1.00 | 0.61 |

![chi_eff vs z](chi_eff_vs_redshift/new_top_middle_overlay_fixed.png)
*Top: Mean effective spin &mu;<sub>&chi;eff</sub>(z) and width &sigma;<sub>&chi;eff</sub>(z) for the Spline and Linear models with PySR overlays.*

![chi_eff vs z gradients](chi_eff_vs_redshift/new_gradient_overlay_fixed.png)
*Analytic gradients d&mu;/dz (left) and d(ln&sigma;)/dz (right). The Linear model shows monotonic broadening; the Spline model shows non-monotonic behavior with a hinge near z = 1.*

**Files:**
- `run_chi_eff_vs_redshift.py` -- Main analysis script
- `generate_final_plots.py` -- Figure generation
- `Linear/` and `Spline/` -- Model-specific results including `report.txt`, diagnostic summaries, equation CSVs, and JSON summaries

---

## Analysis 4: Mass-Ratio Distribution by Mass Peak

**Directory:** `mass_ratio_peak_analysis/`

Symbolic expressions are fit to p(q) conditioned on the low-mass (10 M<sub>&#9737;</sub>) and high-mass (35 M<sub>&#9737;</sub>) peaks of the BBH primary mass distribution.

**Key results:**
- Both peaks strongly favor near-equal-mass pairing (q &rarr; 1)
- The low-mass peak requires a sharp double-exponential cutoff below q ~ 0.2
- The high-mass peak follows a smoother logarithmic decline
- 84% of low-mass draws rise monotonically toward q = 1; 91% of high-mass draws do the same

**Median symbolic expressions (log p space):**

| Peak | Expression |
|---|---|
| Low-mass (10 M<sub>&#9737;</sub>) | `15.2*ln(q) + 104.0*exp(-3326*exp(-39.4*q)) - 12.6*q - 89.8` |
| High-mass (35 M<sub>&#9737;</sub>) | `7.30*ln(q) - exp(1.22*exp(sqrt(q)*(9.78 - 100.2*q))) - [ln(q+0.245)*ln(q)]^2 + 3.08` |

![Mass ratio peaks](mass_ratio_peak_analysis/pysr_q_lowmass_highmass_only_refit_median_only/lowmass_highmass_comparison_refit_median_only_serif.png)
*Conditional mass-ratio distributions p(q) for the low-mass peak (left) and high-mass peak (right). Blue: GWTC-4 90% CI. Orange: PySR 90% CI from draw-by-draw fits.*

**Files:**
- `run_mass_ratio_dist.py` -- Full posterior analysis (200 draws)
- `run_mass_ratio_dist_median.py` -- Refined median-only refit with enhanced weighting
- `generate_final_plots.py` -- Figure generation
- `pysr_q_lowmass_highmass_only/` -- Original draw-by-draw results
- `pysr_q_lowmass_highmass_only_refit_median_only/` -- Refined median fits and comparison figures

---

## Method

All analyses use [PySR](https://github.com/MilesCranmer/PySR) ([Cranmer 2023](https://arxiv.org/abs/2305.01582)) for symbolic regression. PySR performs an evolutionary search over mathematical expressions, returning Pareto-optimal formulae that balance accuracy against complexity. Key configuration across analyses:

- **Operators:** +, -, *, /, exp, log, sqrt, tanh, abs
- **Iterations:** 300--1000
- **Populations:** 50--80
- **Max expression size:** 28--45 nodes
- **Loss:** MSE (in log-space where appropriate)
- **Train/test split:** 70/30
- **Uncertainty propagation:** PySR is run independently on 200 posterior draws, yielding an ensemble of symbolic expressions from which credible intervals and derived quantities are computed

This draw-by-draw approach extends [Wong & Cranmer (2022)](https://arxiv.org/abs/2207.12409), who first applied symbolic regression to GW population inference on the GWTC-3 primary mass spectrum using a single median fit.

## Requirements

```
pysr
numpy
scipy
matplotlib
h5py
sympy
```

## Citation

If you use this code or results, please cite:

```
Chatterjee (2026), "Interpretable Analytic Formulae for GWTC-4 Binary Black Hole
Population Properties via Symbolic Regression", in preparation.
```

## License

This project is released under the MIT License.
