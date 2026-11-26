# Verdant Leaf Marketing Mix Model & Q1 2026 Budget Optimizer (`v3`)

Decision-focused Bayesian Causal MMM and Stochastic Operations Research pipeline for **Verdant Leaf** (`2023-01-02` to `2025-12-22`, $T = 156$ weeks), calibrated to the April–May 2025 Meta 20%-pin-code-clusters-dark geo-holdout experiment.

## Quickstart

```bash
pip install -r requirements.txt
python run_all.py
```

Optional CLI flags for `run_all.py`:
- `--data-dir`: Directory containing `mmm_weekly_data.csv` and `lift_test_results.csv` (default: `./data`)
- `--out-dir`: Output directory for quantitative CSV/JSON/NPZ artifacts (default: `./outputs`)
- `--n-chains`: Number of independent HMC chains per model specification (default: `4`)
- `--n-warmup`: Warmup/adaptation iterations per chain (default: `500`)
- `--n-samples`: Post-warmup posterior draws per chain (default: `750`, i.e., `3,000` total draws)
- `--workers`: Parallel worker processes (default: `4`)
- `--skip-cv`: Skip expanding-window CV & stacking weights (default: `False`)
- `--print-only`: Print full evaluation scores and key output tables from `./outputs/results_summary.json` without re-sampling

## Architecture

1. **`clean_data.py`**: Primary-key deduplication (`2025-03-10`), Q4 2023 Meta $\times 10^5$ Lakhs-to-INR restoration + KS verification, dual physical-vs-financial TV rebate ledger (`2024-11-11`), strictly causal fold-isolated YouTube CPM imputation & scaling (`prepare_fold_data`), and ISO calendar reconciliation (`52 + 53 + 51 = 156` weeks).
2. **`model_mmm.py`**: SVD-exact `v0` reproduction & distortion waterfall; structural causal DAG mediator falsification & Google Search / Amazon Ads / TV identification audits; orthogonalized concave $(r_0, \kappa)$ Bayesian MMM with spline trend, $52.1775$-wk Fourier seasonality, symmetric priors, and AR(1) innovations; exact 20%-geo-dark Meta lift counterfactual operator; multi-chain Hessian-preconditioned NUTS ($\hat{R}$, ESS, 0 divergences); leak-free expanding-window CV; and Yao–Vehtari Bayesian stacking across 6 admissible specifications (screening out pathological convex S-curves).
3. **`optimize_budget.py`**: Smooth Rockafellar–Uryasev Mean–$\text{CVaR}_{10\%}$ optimization with identification-weighted trust-region shrinkage delivering **one reconciled ₹16.00 Cr Q1 2026 plan**, verified across all 3 quarterly guardrail definitions (`51w/4`, `52w/4`, `51w×13/51`), dual 13-week P&L (with full prediction intervals) and 25-week carryover accounting, contribution-profit efficient frontier ($B^*(m, \lambda_{\text{LTV}})$), flighted pacing, and Monte Carlo Bayesian EVPPI next-experiment ranking.
4. **`run_all.py`**: End-to-end 3-stage pipeline runner and structured console evaluation report printer.
