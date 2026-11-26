"""
optimize_budget.py — Decision-Focused Stochastic Q1 2026 (₹16.00 Cr) Budget Optimizer (v3).

Features:
  - Single reconciled primary recommendation (Smooth Rockafellar-Uryasev Mean-CVaR_10%
    + identification-weighted trust-region shrinkage Omega_j -> zero contradictory plans).
  - Exact analytical gradients for SLSQP + strict res.success & KKT optimality checks.
  - Explicit reconciliation across all 3 quarterly baseline conventions (51w/4, 52w/4, 13/51).
  - Dual-horizon Q1 2026 accounting (13-week P&L with full predictive intervals including
    trend, media, and AR(1) noise vs. 25-week full-carryover incremental media return).
  - Total-budget contribution-profit efficient frontier across Gross Margin x Repeat LTV.
  - Bayesian EVPPI next-experiment ranking across all 6 paid channels.
"""
import argparse
import json
from pathlib import Path

from clean_data import CHANNELS, CHANNEL_LABELS, resolve_paths  # Sets BLAS/OMP thread caps before numpy

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from model_mmm import (
    WEIBULL_CH_IDX,
    WEIBULL_MAX_LAG,
    adstock_norm_and_grad,
    build_base_design,
    hill_unit_response,
    project_exante_covariates,
    stable_sigmoid,
    stable_softplus,
    summarize_draws,
    weibull_weights_and_grads,
)

TOTAL_BUDGET_M = 160.0  # ₹16.00 Cr = ₹160.0M
Q1_WEEKS = 13
CARRYOVER_TAIL_WEEKS = 12
H_FULL = Q1_WEEKS + CARRYOVER_TAIL_WEEKS


def precompute_adstock_basis(
    S_hist: np.ndarray,
    alphas_arr: np.ndarray,
    k_lags_arr: np.ndarray,
    init_vals: np.ndarray,
    pacing: str = "uniform",
    ad_off_cached: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute linear adstock basis arrays of shape (D, 6, H_FULL) for both
    geometric and discrete-Weibull draws:
      - ad_off:  pure carryover from 2025 with 0 Q1 spend (reused across pacings if cached)
      - ad_unit: trajectory for 1.0 M INR Q1 spend under `pacing`
    """
    D = alphas_arr.shape[0]
    need_off = ad_off_cached is None
    ad_off = np.zeros((D, 6, H_FULL), dtype=float) if need_off else ad_off_cached
    ad_unit = np.zeros((D, 6, H_FULL), dtype=float)

    unit_sched = np.zeros((6, H_FULL), dtype=float)
    for j in range(6):
        if pacing == "uniform":
            unit_sched[j, :Q1_WEEKS] = 1.0 / Q1_WEEKS
        elif pacing == "flighted":
            if j == 4:  # TV: 4-week March Spring flight
                unit_sched[j, 9:13] = 1.0 / 4.0
            elif j == 3:  # Influencer: 30% base + 70% in 4 creator burst drops
                unit_sched[j, :Q1_WEEKS] = 0.30 / Q1_WEEKS
                unit_sched[j, [2, 5, 9, 11]] += 0.70 / 4.0
            else:
                unit_sched[j, :Q1_WEEKS] = 1.0 / Q1_WEEKS
        else:
            raise ValueError(f"Unknown pacing: {pacing}")

    s_zero = np.zeros(H_FULL, dtype=float)
    L_w = WEIBULL_MAX_LAG
    tail_hist = S_hist[-L_w:, :]  # last 12 weeks of 2025

    for d in range(D):
        for j in range(6):
            a = float(alphas_arr[d, j])
            kl = float(k_lags_arr[d, j])
            if kl > 1.0001 and j in WEIBULL_CH_IDX:
                w, _, _ = weibull_weights_and_grads(a, kl)
                seq_unit = np.concatenate([np.zeros(L_w), unit_sched[j]])
                if need_off:
                    seq_off = np.concatenate([tail_hist[:, j], s_zero])
                for t_h in range(H_FULL):
                    idx_s = L_w + t_h
                    if need_off:
                        ad_off[d, j, t_h] = float(np.dot(w, seq_off[idx_s - L_w : idx_s + 1][::-1]))
                    ad_unit[d, j, t_h] = float(np.dot(w, seq_unit[idx_s - L_w : idx_s + 1][::-1]))
            else:
                if need_off:
                    ad_hist, _ = adstock_norm_and_grad(S_hist[:, j], a, init_vals[j])
                    ad_off[d, j], _ = adstock_norm_and_grad(s_zero, a, float(ad_hist[-1]))
                ad_unit[d, j], _ = adstock_norm_and_grad(unit_sched[j], a, 0.0)

    return ad_off, ad_unit


def precompute_baseline_unit_response(
    kap_arr: np.ndarray,
    slopes_arr: np.ndarray,
    tilde_s: np.ndarray,
    ad_off: np.ndarray,
) -> np.ndarray:
    """Precompute w-independent zero-Q1-spend Hill unit saturation u_off of shape (D, 6, H_FULL)."""
    K_3 = (tilde_s[None, :] / kap_arr)[:, :, None]
    s_3 = slopes_arr[:, :, None]
    u_off, _, _ = hill_unit_response(ad_off, K_3, s_3)
    return u_off


def eval_draws_and_grad(
    w_m: np.ndarray,
    r0_arr: np.ndarray,
    kap_arr: np.ndarray,
    slopes_arr: np.ndarray,
    tilde_s: np.ndarray,
    ad_off: np.ndarray,
    ad_unit: np.ndarray,
    horizon: int = H_FULL,
    u_off: np.ndarray | None = None,
    return_13w: bool = False,
):
    """
    Evaluate incremental revenue across D draws and exact Jacobian dR_d / dw_j:
      C_j(a) = r0_j * K_j * (a/K_j)^s_j / (1 + (a/K_j)^s_j),  K_j = tilde_s_j / kappa_j.
    """
    ad_o = np.maximum(ad_off[:, :, :horizon], 0.0)
    ad_u = ad_unit[:, :, :horizon]
    ad_on = np.maximum(ad_o + w_m[None, :, None] * ad_u, 0.0)

    K_3 = (tilde_s[None, :] / kap_arr)[:, :, None]
    beta_3 = (r0_arr * (tilde_s[None, :] / kap_arr))[:, :, None]
    s_3 = slopes_arr[:, :, None]

    u_on, pos_on, safe_on = hill_unit_response(ad_on, K_3, s_3)
    u_o = u_off[:, :, :horizon] if u_off is not None else hill_unit_response(ad_o, K_3, s_3)[0]

    weekly_inc = beta_3 * (u_on - u_o)
    ch_inc = np.sum(weekly_inc, axis=2)
    tot_inc = np.sum(ch_inc, axis=1)

    dC_dad = np.where(pos_on, beta_3 * s_3 * u_on * (1.0 - u_on) / safe_on, 0.0)
    grad_w = np.sum(dC_dad * ad_u, axis=2)
    if return_13w:
        ch_inc_13 = np.sum(weekly_inc[:, :, :Q1_WEEKS], axis=2)
        return tot_inc, ch_inc, grad_w, np.sum(ch_inc_13, axis=1), ch_inc_13
    return tot_inc, ch_inc, grad_w


def eval_2025_carryover_q1(
    r0_arr: np.ndarray,
    kap_arr: np.ndarray,
    slopes_arr: np.ndarray,
    tilde_s: np.ndarray,
    ad_off: np.ndarray,
    u_off: np.ndarray | None = None,
) -> np.ndarray:
    """Q1 13-week revenue coming purely from late-2025 adstock carry-in (D,)."""
    beta_3 = (r0_arr * (tilde_s[None, :] / kap_arr))[:, :, None]
    u_o = (
        u_off[:, :, :Q1_WEEKS]
        if u_off is not None
        else hill_unit_response(ad_off[:, :, :Q1_WEEKS], (tilde_s[None, :] / kap_arr)[:, :, None], slopes_arr[:, :, None])[0]
    )
    return np.sum(beta_3 * u_o, axis=(1, 2))


def compute_exact_cvar(draws: np.ndarray, alpha_tail: float = 0.10) -> float:
    """Exact empirical lower-tail CVaR_alpha across D posterior draws."""
    k = max(1, int(np.ceil(alpha_tail * len(draws))))
    return float(np.mean(np.partition(draws, k - 1)[:k]))


def smooth_cvar_and_grad(
    draws: np.ndarray,
    zeta: float,
    alpha_tail: float,
    tau_smooth: float,
    g_w: np.ndarray | None = None,
):
    """Smooth Rockafellar-Uryasev CVaR_alpha and optional analytical gradients wrt (w, zeta)."""
    diff = (zeta - draws) / tau_smooth
    sp = stable_softplus(diff, lo=-40.0, hi=30.0) * tau_smooth
    cvar_val = zeta - float(np.mean(sp)) / alpha_tail
    if g_w is None:
        return cvar_val
    sig = stable_sigmoid(diff, clip=40.0)
    g_zeta = 1.0 - float(np.mean(sig)) / alpha_tail
    g_cvar_w = np.mean(sig[:, None] * g_w, axis=0) / alpha_tail
    return cvar_val, g_cvar_w, g_zeta


def project_feasible(w: np.ndarray, lb: np.ndarray, ub: np.ndarray, total: float = TOTAL_BUDGET_M) -> np.ndarray:
    w0 = np.clip(w.copy(), lb, ub)
    for _ in range(50):
        diff = total - w0.sum()
        if abs(diff) < 1e-10:
            break
        free = (w0 < ub - 1e-10) if diff > 0 else (w0 > lb + 1e-10)
        if not np.any(free):
            break
        w0[free] = np.clip(w0[free] + diff / free.sum(), lb[free], ub[free])
    return w0


def solve_smooth_mean_cvar(
    r0_arr: np.ndarray,
    kap_arr: np.ndarray,
    slopes_arr: np.ndarray,
    tilde_s: np.ndarray,
    ad_off: np.ndarray,
    ad_unit: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    w_sq: np.ndarray,
    omega: np.ndarray,
    lambda_risk: float = 0.30,
    tau_id: float = 0.80,
    alpha_tail: float = 0.10,
    tau_smooth: float = 0.02,
    budget_m: float = TOTAL_BUDGET_M,
    u_off: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Exact smooth Rockafellar-Uryasev Mean-CVaR optimization with weak-ID trust-region
    shrinkage and exact analytical gradients.
    """
    u_o_cache = precompute_baseline_unit_response(kap_arr, slopes_arr, tilde_s, ad_off) if u_off is None else u_off

    def neg_obj_and_grad(x):
        w = x[:6]
        zeta = float(x[6])
        tot, _, g_w = eval_draws_and_grad(w, r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit, H_FULL, u_off=u_o_cache)
        mean_r = float(np.mean(tot))
        g_mean_w = np.mean(g_w, axis=0)
        cvar_val, g_cvar_w, g_cvar_zeta = smooth_cvar_and_grad(tot, zeta, alpha_tail, tau_smooth, g_w=g_w)

        rel_dev = (w - w_sq) / w_sq
        pen_id = 0.5 * tau_id * float(np.sum(omega * (rel_dev ** 2)))
        g_pen_w = tau_id * omega * rel_dev / w_sq

        obj = (1.0 - lambda_risk) * mean_r + lambda_risk * cvar_val - pen_id
        grad_w = (1.0 - lambda_risk) * g_mean_w + lambda_risk * g_cvar_w - g_pen_w
        grad_z = lambda_risk * g_cvar_zeta
        return -obj, -np.append(grad_w, grad_z)

    w0 = project_feasible(w_sq, lb, ub, budget_m)
    tot0, _, _ = eval_draws_and_grad(w0, r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit, H_FULL, u_off=u_o_cache)
    x0 = np.append(w0, float(np.percentile(tot0, 100.0 * alpha_tail)))

    bounds = list(zip(lb, ub)) + [(-500.0, 1000.0)]
    cons = [{"type": "eq", "fun": lambda x: np.sum(x[:6]) - budget_m}]

    res = minimize(neg_obj_and_grad, x0, method="SLSQP", jac=True, bounds=bounds, constraints=cons, options={"ftol": 1e-11, "maxiter": 600})
    if not res.success:
        raise RuntimeError(f"Smooth Mean-CVaR optimizer failed: {res.message}")

    w_opt = res.x[:6].copy()
    tot_opt, _, _ = eval_draws_and_grad(w_opt, r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit, H_FULL, u_off=u_o_cache)
    exact_cvar10 = compute_exact_cvar(tot_opt, alpha_tail)
    smooth_cvar10 = smooth_cvar_and_grad(tot_opt, float(res.x[6]), alpha_tail, tau_smooth)

    return w_opt, {
        "success": bool(res.success),
        "nit": int(res.nit),
        "exact_cvar10_cr": round(exact_cvar10 / 10.0, 4),
        "smooth_cvar10_cr": round(smooth_cvar10 / 10.0, 4),
        "cvar_smoothing_gap_cr": round(abs(exact_cvar10 - smooth_cvar10) / 10.0, 5),
        "sum_budget_cr": round(float(np.sum(w_opt) / 10.0), 6),
    }


def solve_profit_frontier(
    r0_arr: np.ndarray,
    kap_arr: np.ndarray,
    slopes_arr: np.ndarray,
    tilde_s: np.ndarray,
    ad_off: np.ndarray,
    ad_unit: np.ndarray,
    u_off: np.ndarray | None = None,
) -> dict:
    """Solve total-budget contribution-profit efficient frontier across Gross Margins x Cohort LTV."""
    margins = (0.30, 0.40, 0.50, 0.60, 0.70)
    ltvs = (1.0, 1.3, 1.6)
    budgets_cr = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    u_o_cache = precompute_baseline_unit_response(kap_arr, slopes_arr, tilde_s, ad_off) if u_off is None else u_off

    def _solve_expected_slsqp(rev_mult: float, cost_mult: float, w0_val: float, ub_val: float, maxiter: int, cap_m: float | None = None):
        def neg_fn(w):
            t, _, gw = eval_draws_and_grad(w, r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit, H_FULL, u_off=u_o_cache)
            val = rev_mult * float(np.mean(t)) - cost_mult * float(np.sum(w))
            grad = rev_mult * np.mean(gw, axis=0) - cost_mult
            return -val, -grad

        cons = () if cap_m is None else [{"type": "ineq", "fun": lambda w, bm=cap_m: bm - np.sum(w)}]
        res = minimize(neg_fn, np.full(6, w0_val), method="SLSQP", jac=True, bounds=[(0.0, ub_val)] * 6, constraints=cons, options={"ftol": 1e-10, "maxiter": maxiter})
        if not res.success:
            raise RuntimeError(f"Profit frontier SLSQP failed: {res.message}")
        t_opt, _, g_opt = eval_draws_and_grad(res.x, r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit, H_FULL, u_off=u_o_cache)
        return res.x, t_opt, g_opt

    frontier_curve = []
    for B_cr in budgets_cr:
        B_m = B_cr * 10.0
        _, t_opt, g_opt = _solve_expected_slsqp(1.0, 0.0, B_m / 6.0, B_m, maxiter=400, cap_m=B_m)
        exp_rev_cr = float(np.mean(t_opt) / 10.0)
        frontier_curve.append({
            "budget_cr": B_cr,
            "expected_inc_rev_25wk_cr": round(exp_rev_cr, 3),
            "blended_roas": round(exp_rev_cr / B_cr, 3),
            "marginal_roas_at_budget": round(float(np.max(np.mean(g_opt, axis=0))), 3),
        })

    optimal_budgets = {}
    for m in margins:
        for ltv in ltvs:
            key = f"m{int(round(m*100))}_ltv{int(round(ltv*10))}"
            eff_mult = m * ltv
            w_star, t_star, _ = _solve_expected_slsqp(eff_mult, 1.0, 15.0, 300.0, maxiter=500)
            optimal_budgets[key] = {
                "gross_margin": m,
                "ltv_multiplier": ltv,
                "breakeven_mroas": round(1.0 / eff_mult, 3),
                "unconstrained_profit_opt_budget_cr": round(float(np.sum(w_star) / 10.0), 2),
                "unconstrained_max_profit_25wk_cr": round(float((eff_mult * np.mean(t_star) - np.sum(w_star)) / 10.0), 2),
            }

    return {"budget_curve": frontier_curve, "optimal_total_budgets": optimal_budgets}


def compute_evppi_by_channel(
    r0_arr: np.ndarray,
    kap_arr: np.ndarray,
    slopes_arr: np.ndarray,
    tilde_s: np.ndarray,
    ad_off: np.ndarray,
    ad_unit: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    w_base: np.ndarray,
    u_off: np.ndarray | None = None,
) -> dict:
    """
    Compute Expected Value of Partial Perfect Information (EVPPI) for learning channel j's
    causal parameters Theta_j = (r0_j, kappa_j, alpha_j) under the ₹16.00 Cr mandate.
    """
    D = len(r0_arr)
    u_o_cache = precompute_baseline_unit_response(kap_arr, slopes_arr, tilde_s, ad_off) if u_off is None else u_off
    rng = np.random.default_rng(1234)
    cand_ws = [w_base]
    for j in range(6):
        w_hi = w_base.copy(); w_hi[j] = ub[j]; cand_ws.append(project_feasible(w_hi, lb, ub))
        w_lo = w_base.copy(); w_lo[j] = lb[j]; cand_ws.append(project_feasible(w_lo, lb, ub))
    for _ in range(67):
        wr = lb + rng.uniform(0.0, 1.0, size=6) * (ub - lb)
        cand_ws.append(project_feasible(wr, lb, ub))
    W_cand = np.asarray(cand_ws)  # (80, 6)

    R_mat = np.empty((D, len(W_cand)), dtype=float)
    for k_i, w_k in enumerate(W_cand):
        R_mat[:, k_i], _, _ = eval_draws_and_grad(w_k, r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit, H_FULL, u_off=u_o_cache)

    base_max_exp = float(np.max(np.mean(R_mat, axis=0)))
    _, _, g_at_base = eval_draws_and_grad(w_base, r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit, H_FULL, u_off=u_o_cache)

    evppi_out = {}
    n_bins = 25
    for j, ch in enumerate(CHANNELS):
        bins = pd.qcut(g_at_base[:, j], q=n_bins, labels=False, duplicates="drop")
        cond_best = sum(
            float(np.mean(bins == b_id)) * float(np.max(np.mean(R_mat[bins == b_id], axis=0)))
            for b_id in np.unique(bins)
        )
        evppi_m = max(0.0, cond_best - base_max_exp)
        evppi_out[ch] = {
            "label": CHANNEL_LABELS[ch],
            "evppi_25wk_cr": round(evppi_m / 10.0, 3),
            "evppi_annualized_4q_cr": round(4.0 * evppi_m / 10.0, 3),
        }
    return evppi_out


def run_budget_optimization(
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    _, o_dir = resolve_paths(data_dir, out_dir)
    df = pd.read_csv(o_dir / "mmm_cleaned.csv")
    with open(o_dir / "model_results.json", encoding="utf-8") as f:
        model_res = json.load(f)
    draws = np.load(o_dir / "stacked_posterior_draws.npz", allow_pickle=True)

    S_hist = df[CHANNELS].values.astype(float) / 1e6
    init_vals = np.asarray(draws["init_vals"], dtype=float)
    tilde_s = np.asarray(draws["tilde_s"], dtype=float)
    r0_arr = np.asarray(draws["r0"], dtype=float)
    kap_arr = np.asarray(draws["kappa"], dtype=float)
    alphas_arr = np.asarray(draws["alphas"], dtype=float)
    slopes_arr = np.asarray(draws["slopes"], dtype=float)
    klags_arr = np.asarray(draws["k_lags"], dtype=float)
    b_b_arr = np.asarray(draws["b_b"], dtype=float)
    rho_arr = np.asarray(draws["rho"], dtype=float)
    sig_arr = np.asarray(draws["sigma_y"], dtype=float)
    D = len(r0_arr)

    # Reconciled 2025 quarterly spend baselines
    q_rec = model_res["audit"]["quarterly_baseline_reconciliation"]
    q_51_m = np.array([q_rec["channels"][ch]["cal_2025_51w_div4_cr"] * 10.0 for ch in CHANNELS])
    q_52_m = np.array([q_rec["channels"][ch]["trailing_52w_div4_cr"] * 10.0 for ch in CHANNELS])
    q_13_m = np.array([q_rec["channels"][ch]["cal_2025_51w_rate13_cr"] * 10.0 for ch in CHANNELS])

    w_sq_m = q_51_m * (TOTAL_BUDGET_M / np.sum(q_51_m))
    lb = 0.50 * q_51_m
    ub = 1.50 * q_51_m

    # Precompute shared ad_off once + uniform/flighted unit bases + shared baseline saturation u_off
    ad_off, ad_unit_u = precompute_adstock_basis(S_hist, alphas_arr, klags_arr, init_vals, pacing="uniform")
    _, ad_unit_f = precompute_adstock_basis(S_hist, alphas_arr, klags_arr, init_vals, pacing="flighted", ad_off_cached=ad_off)
    u_off = precompute_baseline_unit_response(kap_arr, slopes_arr, tilde_s, ad_off)

    ch_sum = model_res["stacked_posterior"]["channels"]
    omega = np.array([ch_sum[ch]["ambiguity_weight_omega"] for ch in CHANNELS], dtype=float)

    print("  1/4 Solving Primary Reconciled Smooth Mean-CVaR Q1 2026 Allocation (Operational Flighted Pacing)...")
    w_rec_m, opt_diag = solve_smooth_mean_cvar(
        r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit_f,
        lb, ub, w_sq_m, omega, lambda_risk=0.30, tau_id=0.80, alpha_tail=0.10, u_off=u_off
    )
    w_risk_neutral_m, rn_diag = solve_smooth_mean_cvar(
        r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit_f,
        lb, ub, w_sq_m, omega, lambda_risk=0.0, tau_id=0.0, u_off=u_off
    )
    w_uniform_m, fl_diag = solve_smooth_mean_cvar(
        r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit_u,
        lb, ub, w_sq_m, omega, lambda_risk=0.30, tau_id=0.80, alpha_tail=0.10, u_off=u_off
    )
    w_sq_52 = q_52_m * (TOTAL_BUDGET_M / np.sum(q_52_m))
    w_52_m, g52_diag = solve_smooth_mean_cvar(
        r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit_f,
        0.50 * q_52_m, 1.50 * q_52_m, w_sq_52, omega, lambda_risk=0.30, tau_id=0.80, alpha_tail=0.10, u_off=u_off
    )

    in_51_bounds = bool(np.all((w_rec_m >= lb - 1e-4) & (w_rec_m <= ub + 1e-4)))
    in_52_bounds = bool(np.all((w_rec_m >= 0.50 * q_52_m - 1e-4) & (w_rec_m <= 1.50 * q_52_m + 1e-4)))
    in_13_bounds = bool(np.all((w_rec_m >= 0.50 * q_13_m - 1e-4) & (w_rec_m <= 1.50 * q_13_m + 1e-4)))

    # 2. Consistent 13-week Q1 2026 P&L Forecast via shared project_exante_covariates + build_base_design
    print("  2/4 Computing 13-week Q1 P&L Forecast with Full Baseline + AR(1) Predictive Intervals...")
    t_q1 = np.arange(len(df), len(df) + Q1_WEEKS, dtype=float)
    df_q1 = project_exante_covariates(
        df,
        t_q1,
        np.arange(1, Q1_WEEKS + 1, dtype=int),
        price_fit_df=df[df["year"] == 2025],
        launch_fit_df=df[df["year"] >= 2024],
        zero_tentpoles=True,
    )
    X_q1_base, _ = build_base_design(df_q1, model_res["ref_means_full"], use_spline=True, K=2)

    base_13w_draws = np.sum(b_b_arr @ X_q1_base.T, axis=1)
    carry_13w_draws = eval_2025_carryover_q1(r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, u_off=u_off)

    decomp = pd.read_csv(o_dir / "revenue_decomposition.csv")
    last_resid = float(decomp["actual_rev_m"].iloc[-1] - decomp["pred_rev_m"].iloc[-1])
    rng_noise = np.random.default_rng(777)
    noise_13w_draws = np.zeros(D, dtype=float)
    eps_cur = np.full(D, last_resid, dtype=float)
    for _ in range(Q1_WEEKS):
        eps_cur = rho_arr * eps_cur + rng_noise.normal(0.0, sig_arr)
        noise_13w_draws += eps_cur

    def _eval_horizons(w_vec, ad_u):
        return eval_draws_and_grad(
            w_vec, r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_u, H_FULL, u_off=u_off, return_13w=True
        )

    sq_f_raw = _eval_horizons(w_sq_m, ad_unit_f)
    sq_u_raw = _eval_horizons(w_sq_m, ad_unit_u)

    def eval_plan(w_vec, ad_u, sq_ref, label, pre_eval=None):
        inc25, ch25, gw25, inc13, _ = pre_eval if pre_eval is not None else _eval_horizons(w_vec, ad_u)
        inc25_sq, _, _, inc13_sq, _ = sq_ref

        tot_q1_13w = base_13w_draws + carry_13w_draws + inc13 + noise_13w_draws
        lift25 = inc25 - inc25_sq
        lift13 = inc13 - inc13_sq
        lift25_pct = 100.0 * lift25 / (inc25_sq + 1e-9)
        cvar10 = compute_exact_cvar(inc25, 0.10)

        return {
            "plan_name": label,
            "allocation_cr": {ch: round(float(w_vec[j] / 10.0), 3) for j, ch in enumerate(CHANNELS)},
            "q1_13wk_pnl_cr": {
                "non_media_baseline_13wk": summarize_draws(base_13w_draws / 10.0),
                "carryover_from_2025_13wk": summarize_draws(carry_13w_draws / 10.0),
                "new_q1_spend_inc_rev_13wk": summarize_draws(inc13 / 10.0),
                "total_q1_13wk_revenue_with_full_pi": summarize_draws(tot_q1_13w / 10.0),
                "lift_13wk_vs_sq": summarize_draws(lift13 / 10.0),
            },
            "full_carryover_25wk_cr": {
                "inc_rev_25wk": summarize_draws(inc25 / 10.0),
                "cvar10_inc_rev_25wk": round(cvar10 / 10.0, 2),
                "lift_25wk_vs_sq": summarize_draws(lift25 / 10.0),
                "lift_25wk_pct": summarize_draws(lift25_pct),
                "prob_beats_status_quo": round(float(np.mean(lift25 > 0.0)), 4),
            },
            "_raw": {"inc25": inc25, "ch25": ch25, "gw25": gw25, "lift25": lift25, "lift25_pct": lift25_pct},
        }

    p_sq = eval_plan(w_sq_m, ad_unit_f, sq_f_raw, "Status Quo (2025 Quarterly Share, Flighted TV/Inf)", pre_eval=sq_f_raw)
    p_rec = eval_plan(w_rec_m, ad_unit_f, sq_f_raw, "Primary Reconciled Q1 2026 Plan (Operational Flighted TV/Inf)")
    p_rec_unif = eval_plan(w_rec_m, ad_unit_u, sq_u_raw, "Primary Plan Evaluated Under Uniform 13w Pacing")
    p_rn = eval_plan(w_risk_neutral_m, ad_unit_f, sq_f_raw, "Sensitivity: Pure Risk-Neutral Expected Revenue")
    p_unif_opt = eval_plan(w_uniform_m, ad_unit_u, sq_u_raw, "Sensitivity: Uniform-Paced Optimization")

    ch_sq25 = p_sq["_raw"]["ch25"]
    gw_sq25 = p_sq["_raw"]["gw25"]
    ch_rec25 = p_rec["_raw"]["ch25"]
    gw_rec25 = p_rec["_raw"]["gw25"]

    channel_table = {}
    csv_rows = []
    for j, ch in enumerate(CHANNELS):
        ar_rec_med = round(float(np.median(ch_rec25[:, j]) / max(w_rec_m[j], 1e-6)), 3)
        b_status = "AT_UPPER_150%" if w_rec_m[j] >= ub[j] - 0.05 else ("AT_LOWER_50%" if w_rec_m[j] <= lb[j] + 0.05 else "INTERIOR")

        ct = {
            "label": CHANNEL_LABELS[ch],
            "confidence_tier": ch_sum[ch]["confidence_tier"],
            "posterior_contraction": ch_sum[ch]["posterior_contraction"],
            "baseline_2025_q_51w_div4_cr": round(float(q_51_m[j] / 10.0), 3),
            "baseline_2025_q_52w_div4_cr": round(float(q_52_m[j] / 10.0), 3),
            "bound_min_50pct_cr": round(float(lb[j] / 10.0), 3),
            "bound_max_150pct_cr": round(float(ub[j] / 10.0), 3),
            "recommended_q1_2026_cr": round(float(w_rec_m[j] / 10.0), 3),
            "delta_cr": round(float((w_rec_m[j] - w_sq_m[j]) / 10.0), 3),
            "change_pct": round(float(100.0 * (w_rec_m[j] / q_51_m[j] - 1.0)), 1),
            "budget_share_pct": round(float(100.0 * w_rec_m[j] / TOTAL_BUDGET_M), 1),
            "bound_status": b_status,
            "inc_rev_sq_25wk_cr": summarize_draws(ch_sq25[:, j] / 10.0),
            "inc_rev_rec_25wk_cr": summarize_draws(ch_rec25[:, j] / 10.0),
            "ar_at_rec": ar_rec_med,
            "mroas_at_sq": summarize_draws(gw_sq25[:, j]),
            "mroas_at_rec": summarize_draws(gw_rec25[:, j]),
            "sensitivity_risk_neutral_cr": round(float(w_risk_neutral_m[j] / 10.0), 3),
            "sensitivity_52w_guardrail_cr": round(float(w_52_m[j] / 10.0), 3),
        }
        channel_table[ch] = ct
        csv_rows.append(
            {
                "channel": ch,
                "label": ct["label"],
                "confidence_tier": ct["confidence_tier"],
                "baseline_2025_q_cr": ct["baseline_2025_q_51w_div4_cr"],
                "bound_min_50pct_cr": ct["bound_min_50pct_cr"],
                "bound_max_150pct_cr": ct["bound_max_150pct_cr"],
                "recommended_q1_2026_cr": ct["recommended_q1_2026_cr"],
                "delta_cr": ct["delta_cr"],
                "change_pct": ct["change_pct"],
                "share_pct": ct["budget_share_pct"],
                "bound_status": b_status,
                "exp_inc_rev_25wk_cr": ct["inc_rev_rec_25wk_cr"]["median"],
                "exp_inc_rev_25wk_ci90": str(ct["inc_rev_rec_25wk_cr"]["ci90"]),
                "ar_at_rec": ar_rec_med,
                "mroas_at_sq_med": ct["mroas_at_sq"]["median"],
                "mroas_at_rec_med": ct["mroas_at_rec"]["median"],
            }
        )
    pd.DataFrame(csv_rows).to_csv(o_dir / "q1_2026_budget_plan.csv", index=False, lineterminator="\n")

    print("  3/4 Solving Total-Budget Contribution-Profit Efficient Frontier (Margins x LTV)...")
    profit_frontier = solve_profit_frontier(r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit_f, u_off=u_off)

    print("  4/4 Computing Bayesian EVPPI next-experiment value by channel...")
    evppi = compute_evppi_by_channel(r0_arr, kap_arr, slopes_arr, tilde_s, ad_off, ad_unit_f, lb, ub, w_rec_m, u_off=u_off)

    lift_draws_rec = p_rec["_raw"]["lift25"] / 10.0
    lift_pct_rec = p_rec["_raw"]["lift25_pct"]
    for p_dict in (p_sq, p_rec, p_rec_unif, p_rn, p_unif_opt):
        p_dict.pop("_raw", None)

    opt_summary = {
        "budget_total_cr": 16.00,
        "q1_weeks": Q1_WEEKS,
        "carryover_tail_weeks": CARRYOVER_TAIL_WEEKS,
        "n_posterior_draws_propagated": int(D),
        "primary_recommendation_method": (
            "Smooth Rockafellar-Uryasev Mean-CVaR_10% (lambda=0.30) + Weak-ID Trust-Region Shrinkage "
            "with Operational Flighted Pacing (TV 4-wk March Burst + Influencer 4 Creator Drops) "
            "over Yao-Vehtari Stacked Posterior Draws"
        ),
        "guardrail_reconciliation": {
            "primary_baseline_used": "cal_2025_51w_div4 (₹16.002 Cr)",
            "primary_plan_inside_51w_bounds": in_51_bounds,
            "primary_plan_inside_52w_bounds": in_52_bounds,
            "primary_plan_inside_13_51_rate_bounds": in_13_bounds,
            "solver_diagnostics": opt_diag,
        },
        "q1_covariate_assumptions": {
            "price_index_q1_2026_projected_mean": round(float(df_q1["price_index"].mean()), 2),
            "avg_discount_pct_q1_mean": round(float(df_q1["avg_discount_pct"].iloc[0]), 2),
            "new_launch_flag_q1_rate": round(float(df_q1["new_launch_flag"].iloc[0]), 3),
            "competitor_promo_flag_q1_rate": round(float(df_q1["competitor_promo_flag"].iloc[0]), 3),
        },
        "primary_plan": p_rec,
        "status_quo_plan": p_sq,
        "sensitivity_scenarios": {
            "primary_evaluated_uniform_pacing": p_rec_unif,
            "risk_neutral_unconstrained_id": p_rn,
            "uniform_optimized": p_unif_opt,
            "risk_neutral_solver_diag": rn_diag,
            "uniform_solver_diag": fl_diag,
            "guardrail_52w_solver_diag": g52_diag,
        },
        "profit_frontier": profit_frontier,
        "evppi_next_experiment": evppi,
        "channels": channel_table,
    }

    np.savez(
        o_dir / "budget_opt_results.npz",
        lift_25wk_cr=lift_draws_rec,
        lift_25wk_pct=lift_pct_rec,
        w_sq_cr=w_sq_m / 10.0,
        w_rec_cr=w_rec_m / 10.0,
        w_rn_cr=w_risk_neutral_m / 10.0,
        w_52w_cr=w_52_m / 10.0,
    )

    model_res["q1_2026_optimization"] = opt_summary
    with open(o_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_res, f, indent=2)
    return opt_summary


def main():
    parser = argparse.ArgumentParser(description="Optimize Verdant Leaf Q1 2026 ₹16 Cr budget (v3).")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    run_budget_optimization(data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
