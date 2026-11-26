"""
run_all.py — End-to-End Reproducible Pipeline for Verdant Leaf Bayesian Causal MMM (v3).

Runs cleanly from a fresh clone with zero edits and prints full evaluation scores & key values:
  python run_all.py
  python run_all.py --print-only
  python run_all.py --data-dir ./data --out-dir ./outputs --n-chains 4 --n-warmup 500 --n-samples 750
"""
import argparse
import json
from pathlib import Path

from clean_data import CHANNELS, clean_mmm_data, resolve_paths
from model_mmm import run_modeling_pipeline
from optimize_budget import run_budget_optimization


def print_evaluation_report(out_dir: Path) -> None:
    """Print all evaluation scores, diagnostics, causal checks, and Q1 2026 values to stdout."""
    with open(out_dir / "results_summary.json", encoding="utf-8") as f:
        res = json.load(f)

    aud = res["audit"]
    anom = aud["anomalies"]
    v0 = res["baseline_diagnosis"]["baseline_full"]
    st = res["stacked_posterior"]
    fit_s = st["in_sample_median_fit"]
    dec = st["revenue_decomposition_summary"]
    lift = res["lift_test_validation"]
    caus = res["causal_and_identification_audits"]
    gs_id = caus["google_search_and_falsification"]
    tv_id = caus["tv_identification"]
    val = res["validation"]
    opt = res["q1_2026_optimization"]

    sep = "=" * 98
    sub = "-" * 98
    print(f"\n{sep}")
    print("VERDANT LEAF BAYESIAN CAUSAL MMM (v3) — EVALUATION SCORES & KEY OUTPUT VALUES")
    print(sep)

    # 1. Data Hygiene & Fit Comparison
    m_err = anom["meta_q4_2023_unit_error"]
    yt_err = anom["youtube_missing_spend"]
    print("\n[1] DATA INTEGRITY & MODEL FIT COMPARISON (LEGACY v0 OLS vs. STACKED v3 BAYESIAN MMM)")
    print(sub)
    print(
        f"  Data Hygiene   : {aud['raw_rows']} raw -> {anom['duplicate_weeks']['post_dedup_rows']} contiguous 7-day weeks "
        f"| Meta Q4 '23 restored ₹{m_err['spend_restored_inr']/1e7:.3f} Cr (KS p={m_err['ks_pval_vs_adjacent_quarters']:.4f})"
    )
    print(
        f"  Imputation/TV  : {yt_err['count']} YouTube NaNs imputed via causal trailing-8 CPM (₹{yt_err['total_imputed_spend_inr']/1e5:.2f}L), "
        f"{yt_err['genuine_dark_weeks_zero_spend_zero_imp']} dark wks kept | TV -₹12.50L rebate dual-tracked"
    )
    print(
        f"  Legacy v0 OLS  : N={v0['n']} | R²={v0['r2']:.4f} | Adj R²={v0['adj_r2']:.4f} | MAPE={v0['mape']:.2f}% | "
        f"RMSE=₹{v0['rmse']/1e6:.3f}M | DW={v0['dw']:.3f} | Ljung-Box p={v0['ljung_box_lag4_pval']:.2e} | kappa={v0['cond_num']:.2e}"
    )
    print(
        f"  Stacked v3 Fit : N=156 | R²={fit_s['r2']:.4f} | MAPE={fit_s['mape_pct']:.2f}% | NRMSE={fit_s['nrmse_pct']:.2f}% | "
        f"DW={fit_s['dw']:.3f} | AR(1) rho={fit_s['rho_median']:+.3f} | sigma_y=₹{fit_s['sigma_y_median']:.3f}M"
    )
    print(
        f"  Revenue Split  : 3-Yr Total ₹{dec['total_3yr_rev_m_inr']/10:.2f} Cr (Paid Media {dec['media_share_3yr_pct']:.2f}% = "
        f"₹{dec['total_3yr_media_contrib_m_inr']/10:.2f} Cr) | 2025 Total ₹{dec['total_2025_rev_m_inr']/10:.2f} Cr "
        f"(Paid Media {dec['media_share_2025_pct']:.2f}% = ₹{dec['total_2025_media_contrib_m_inr']/10:.2f} Cr)"
    )

    # 2. HMC Convergence & Out-of-Sample CV Scores
    print("\n[2] MULTI-CHAIN NUTS CONVERGENCE & LEAK-FREE OUT-OF-SAMPLE VALIDATION SCORES")
    print(sub)
    hdr = f"  {'Spec ID':26s} {'Wt':>5s} {'Rhat':>7s} {'ESS_b':>7s} {'ESS_t':>7s} {'Div':>4s} {'In_R2':>7s} {'OOS_R2':>7s} {'OOS_MAPE':>8s} {'ExAnte':>7s} {'PI90%':>6s} {'ELPD':>8s}"
    print(hdr)
    rc = val.get("rolling_cv", {})
    for sp in res["specification_ensemble"]:
        sid = sp["spec_id"]
        d = sp["diagnostics"]
        ins = sp["in_sample"]
        cv_s = rc.get(sid, {})
        oos_r2 = f"{cv_s['mean_oos_r2']:.4f}" if cv_s else "  n/a "
        oos_mp = f"{cv_s['mean_oos_mape_pct']:.2f}%" if cv_s else "  n/a "
        exa_mp = f"{cv_s['mean_ex_ante_mape_pct']:.2f}%" if cv_s else "  n/a"
        pi90 = f"{cv_s['mean_pi90_coverage_pct']:.1f}%" if cv_s else " n/a "
        elpd = f"{cv_s['total_elpd']:8.2f}" if cv_s else "    n/a "
        print(
            f"  {sid:26s} {sp['stacking_weight']:5.2f} {d['max_rhat']:7.4f} {d['min_ess_bulk']:7.1f} "
            f"{d['min_ess_tail']:7.1f} {d['n_divergent']:4d} {ins['r2']:7.4f} {oos_r2:>7s} {oos_mp:>8s} {exa_mp:>7s} {pi90:>6s} {elpd}"
        )
    h25 = val.get("holdout_2025_m1")
    if h25:
        print(
            f"  Strict 2025 Holdout (M1, train={h25['train_weeks']}w, test={h25['test_weeks']}w): "
            f"OOS R²={h25['oos_r2']:.4f} | Cond MAPE={h25['oos_mape_pct']:.2f}% | RMSE=₹{h25['oos_rmse_m_inr']:.3f}M | "
            f"Ex-Ante MAPE={h25['ex_ante_mape_pct']:.2f}% | 90% PI Cov={h25['pi90_coverage_pct']:.1f}% | ELPD={h25['elpd_sum']:.2f}"
        )

    # 3. Experimental Concordance & Causal Diagnostics
    s1 = lift["stage1_uncalibrated_geo_iroas"]
    s2 = lift["stage2_stacked_calibrated_geo_iroas"]
    rej = caus["dead_end_unconstrained_hill_rejected_by_HG4"]
    bad_org = caus["dead_end_raw_organic_traffic"]
    min_lead_p = min(gs_id["lead_1wk_granger_pvals_hac4"].values())
    print("\n[3] EXPERIMENTAL CALIBRATION & CAUSAL IDENTIFICATION SCORES")
    print(sub)
    print(
        f"  Meta 20%-Geo-Dark Lift : Measured={lift['measured_meta_iroas']:.3f} {lift['measured_meta_90ci']} | "
        f"Stage-1 Uncalibrated={s1['median']:.3f} {s1['ci90']} (p={lift['stage1_concordance_pval']:.4f}) | "
        f"Stage-2 Calibrated={s2['median']:.3f} {s2['ci90']} | Regime A2={lift['regime_a2_reallocated_geo_iroas_map']:.3f}"
    )
    print(
        f"  Identification Checks  : Search demand-driven R²={gs_id['demand_driven_r2_core_drop2']:.4f} "
        f"(exogenous var={gs_id['exogenous_residual_variance_share_pct']:.2f}%) | Min 1-wk lead Granger p={min_lead_p:.4f} | "
        f"Amazon x Sale p={gs_id['amazon_sale_interaction_pval_hac4']:.4f}"
    )
    print(
        f"  TV Robustness & Rejects: 6-Flight LOFO ROAS={tv_id['lofo_tv_avg_roas_range']} (no-festive={tv_id['no_festive_flag_tv_avg_roas']:.3f}, "
        f"net-cost={tv_id['tv_net_cost_rebate_avg_roas']:.3f}) | Bad-control organic Meta iROAS={bad_org['meta_geo_iroas_collapsed']:.3f} | "
        f"Unconstrained Hill Search slope={rej['google_search_slope']:.2f} (mROAS {rej['google_search_mroas_2025']:.2f} > aROAS {rej['google_search_avg_roas']:.2f}, rejected={rej['rejected']})"
    )

    # 4. Channel Performance Table
    print("\n[4] STACKED POSTERIOR CHANNEL ESTIMATES (3,000 POSTERIOR DRAWS)")
    print(sub)
    print(f"  {'Channel':15s} {'Confidence Tier':27s} {'Contr':>6s} {'Alpha':>6s} {'v0_OLS':>7s} {'3Yr Avg ROAS [90% CI]':>24s} {'2025 mROAS [90% CI]':>23s}")
    for ch in CHANNELS:
        c = st["channels"][ch]
        v0_c = v0["coefs"][ch]["coef"]
        ar = c["avg_roas_3yr"]
        mr = c["mroas_2025"]
        ar_s = f"{ar['median']:5.3f} [{ar['ci90'][0]:4.2f}, {ar['ci90'][1]:4.2f}]"
        mr_s = f"{mr['median']:5.3f} [{mr['ci90'][0]:4.2f}, {mr['ci90'][1]:4.2f}]"
        print(
            f"  {c['label']:15s} {c['confidence_tier']:27s} {c['posterior_contraction']:6.3f} "
            f"{c['alpha']['median']:6.3f} {v0_c:7.2f} {ar_s:>24s} {mr_s:>23s}"
        )

    # 5. Q1 2026 Budget Plan, P&L, Profit Frontier & EVPPI
    gr = opt["guardrail_reconciliation"]
    sdiag = gr["solver_diagnostics"]
    p_rec = opt["primary_plan"]
    p13 = p_rec["q1_13wk_pnl_cr"]
    p25 = p_rec["full_carryover_25wk_cr"]
    sq25 = opt["status_quo_plan"]["full_carryover_25wk_cr"]
    print("\n[5] Q1 2026 RECONCILED BUDGET ALLOCATION (₹16.000 Cr), FORECAST, PROFIT FRONTIER & EVPPI")
    print(sub)
    print(f"  {'Channel':15s} {'Base51w/4':>9s} {'[50%, 150%]':>15s} {'Rec Q1 Cr':>10s} {'Chg %':>7s} {'Share':>6s} {'25w Inc Rev [90% CI]':>24s} {'mROAS@Rec':>10s} {'Status':>13s}")
    for ch in CHANNELS:
        oc = opt["channels"][ch]
        ir = oc["inc_rev_rec_25wk_cr"]
        b_rng = f"[{oc['bound_min_50pct_cr']:4.2f}, {oc['bound_max_150pct_cr']:4.2f}]"
        ir_s = f"₹{ir['median']:5.2f} [{ir['ci90'][0]:4.2f}, {ir['ci90'][1]:5.2f}]"
        print(
            f"  {oc['label']:15s} ₹{oc['baseline_2025_q_51w_div4_cr']:7.3f} {b_rng:>15s} "
            f"₹{oc['recommended_q1_2026_cr']:8.3f} {oc['change_pct']:+6.1f}% {oc['budget_share_pct']:5.1f}% "
            f"{ir_s:>24s} {oc['mroas_at_rec']['median']:9.3f}x {oc['bound_status']:>13s}"
        )
    print(
        f"  Guardrail & Solver : Sum=₹{sdiag['sum_budget_cr']:.3f} Cr | 51w/4={gr['primary_plan_inside_51w_bounds']} | "
        f"52w/4={gr['primary_plan_inside_52w_bounds']} | 13/51={gr['primary_plan_inside_13_51_rate_bounds']} | "
        f"SLSQP nit={sdiag['nit']} | CVaR gap={sdiag['cvar_smoothing_gap_cr']} Cr"
    )
    tot13 = p13["total_q1_13wk_revenue_with_full_pi"]
    l25 = p25["lift_25wk_vs_sq"]
    lp25 = p25["lift_25wk_pct"]
    print(
        f"  13-Wk Q1 Total Rev : ₹{tot13['median']:.3f} Cr (90% PI {tot13['ci90']}) = "
        f"Baseline ₹{p13['non_media_baseline_13wk']['median']:.3f} + 2025 Carry-In ₹{p13['carryover_from_2025_13wk']['median']:.3f} + "
        f"New Q1 Media ₹{p13['new_q1_spend_inc_rev_13wk']['median']:.3f} Cr"
    )
    print(
        f"  25-Wk Media Return : ₹{p25['inc_rev_25wk']['median']:.3f} Cr (90% PI {p25['inc_rev_25wk']['ci90']}) | "
        f"Lift vs SQ: +₹{l25['median']:.3f} Cr ({lp25['median']:+.2f}%, 90% PI {l25['ci90']}) | "
        f"P(Beats SQ)={p25['prob_beats_status_quo']*100:.1f}% | CVaR_10%: ₹{sq25['cvar10_inc_rev_25wk']:.2f} -> ₹{p25['cvar10_inc_rev_25wk']:.2f} Cr"
    )

    ob = opt["profit_frontier"]["optimal_total_budgets"]
    pf_items = [
        f"40%x1.0=₹{ob['m40_ltv10']['unconstrained_profit_opt_budget_cr']:.2f}Cr",
        f"50%x1.0=₹{ob['m50_ltv10']['unconstrained_profit_opt_budget_cr']:.2f}Cr",
        f"50%x1.3=₹{ob['m50_ltv13']['unconstrained_profit_opt_budget_cr']:.2f}Cr",
        f"60%x1.3=₹{ob['m60_ltv13']['unconstrained_profit_opt_budget_cr']:.2f}Cr",
        f"60%x1.6=₹{ob['m60_ltv16']['unconstrained_profit_opt_budget_cr']:.2f}Cr",
    ]
    print("  Profit-Opt Budget B*: " + " | ".join(pf_items))

    ev_sorted = sorted(opt["evppi_next_experiment"].values(), key=lambda x: x["evppi_annualized_4q_cr"], reverse=True)
    ev_str = " | ".join(f"{e['label']}: ₹{e['evppi_annualized_4q_cr']:.3f} Cr/yr" for e in ev_sorted)
    print(f"  EVPPI Experiment Rk : {ev_str}")
    print(f"{sep}\n")


def main(
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_samples: int = 750,
    workers: int = 4,
    skip_cv: bool = False,
    print_only: bool = False,
):
    d_dir, o_dir = resolve_paths(data_dir, out_dir)
    if print_only:
        print_evaluation_report(o_dir)
        return

    print(f"Data directory  : {d_dir}")
    print(f"Output directory: {o_dir}\n")

    print("Step 1/3: Split-Safe Data Hygiene & Calendar Reconciliation...")
    clean_bundle = clean_mmm_data(data_dir=d_dir, out_dir=o_dir, return_bundle=True)

    print(f"Step 2/3: Bayesian Causal MMM, Multi-Chain HMC ({n_chains}x{n_samples} draws), Leak-Free CV & Stacking...")
    run_modeling_pipeline(
        data_dir=d_dir,
        out_dir=o_dir,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        workers=workers,
        skip_cv=skip_cv,
        clean_bundle=clean_bundle,
    )

    print("Step 3/3: Smooth Rockafellar-Uryasev Q1 2026 Optimization, Profit Frontier & EVPPI...")
    run_budget_optimization(data_dir=d_dir, out_dir=o_dir)

    print_evaluation_report(o_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full Verdant Leaf v3 Bayesian MMM pipeline end-to-end.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing raw CSV files (default: ./data)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: ./outputs)")
    parser.add_argument("--n-chains", type=int, default=4, help="HMC chains per specification (default: 4)")
    parser.add_argument("--n-warmup", type=int, default=500, help="HMC warmup iterations per chain (default: 500)")
    parser.add_argument("--n-samples", type=int, default=750, help="Posterior draws per chain (default: 750)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes (default: 4)")
    parser.add_argument("--skip-cv", action="store_true", help="Skip expanding-window CV (default: False)")
    parser.add_argument("--print-only", action="store_true", help="Print evaluation scores from existing outputs without re-fitting")
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        n_chains=args.n_chains,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        workers=args.workers,
        skip_cv=args.skip_cv,
        print_only=args.print_only,
    )
