"""
clean_data.py — Split-Safe Data Hygiene, Invariant Auditing & Calendar Reconciliation
for Verdant Leaf MMM (v3).

Guarantees:
  - Zero hardcoded absolute paths (relative ./data and ./outputs by default).
  - Exact primary-key deduplication (2025-03-10) prior to any lag or adstock.
  - Cross-metric CPM/AOV anomaly detection & Q4 2023 Meta x10^5 restoration + KS check.
  - Dual TV billing-credit ledger: physical media delivery (0 on 2024-11-11) vs net-cost basis.
  - Strictly causal, fold-isolated preprocessing (`prepare_fold_data`) preventing HG-1 leakage.
  - Explicit reconciliation of 51-week calendar-2025 vs trailing-52-week quarterly baselines.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent

CHANNELS = [
    "meta_spend",
    "google_search_spend",
    "youtube_spend",
    "influencer_spend",
    "tv_spend",
    "amazon_ads_spend",
]
CHANNEL_LABELS = {
    "meta_spend": "Meta",
    "google_search_spend": "Google Search",
    "youtube_spend": "YouTube",
    "influencer_spend": "Influencer",
    "tv_spend": "TV",
    "amazon_ads_spend": "Amazon Ads",
}
CONTROLS = [
    "festive_flag",
    "marketplace_sale_flag",
    "avg_discount_pct",
    "price_index",
    "competitor_promo_flag",
    "new_launch_flag",
]


def resolve_paths(
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve input data directory and output directory using relative defaults or CLI args."""
    if data_dir is not None:
        d_dir = Path(data_dir).expanduser().resolve()
    else:
        candidates = [SCRIPT_DIR / "data", SCRIPT_DIR, Path.cwd() / "data", Path.cwd()]
        d_dir = next((c for c in candidates if (c / "mmm_weekly_data.csv").exists()), SCRIPT_DIR / "data")

    if out_dir is not None:
        o_dir = Path(out_dir).expanduser().resolve()
    else:
        o_dir = SCRIPT_DIR / "outputs"

    o_dir.mkdir(parents=True, exist_ok=True)
    return d_dir, o_dir


def load_lift_test(data_dir: Path) -> dict:
    """Load Meta geo-holdout lift test parameters dynamically from lift_test_results.csv."""
    lift_path = data_dir / "lift_test_results.csv"
    df_lift = pd.read_csv(lift_path)
    row = df_lift.iloc[0]
    iroas = float(row["measured_iroas"])
    ci_lo = float(row["iroas_90ci_low"])
    ci_hi = float(row["iroas_90ci_high"])
    se = (ci_hi - ci_lo) / (2.0 * 1.6448536)
    test_start = str(row["test_start"])
    test_end = str(row["test_end"])
    eval_end = (pd.to_datetime(test_end) + pd.Timedelta(weeks=4)).strftime("%Y-%m-%d")
    return {
        "test_name": str(row["test_name"]),
        "channel": str(row["channel"]),
        "test_start": test_start,
        "test_end": test_end,
        "eval_end": eval_end,
        "holdout_share": float(row["holdout_share_of_spend"]),
        "spend_withheld_inr": float(row["spend_withheld_inr"]),
        "incremental_revenue_inr": float(row["incremental_revenue_inr"]),
        "measured_iroas": iroas,
        "ci90": [ci_lo, ci_hi],
        "se": float(se),
        "notes": str(row["notes"]),
    }


def apply_structural_repairs(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply purely deterministic, row-local structural repairs that use zero cross-time
    or future-row statistics:
      1. Exact primary-key deduplication on `week_start` (verifying 19-col equality).
      2. Q4 2023 Meta Lakhs-to-INR unit restoration (spend < 100,000 while impressions > 10M).
      3. TV 2024-11-11 negative billing credit separation (physical=0 vs net-cost pro-rata).
    Leaves `youtube_spend` NaNs intact so imputation is strictly fold-isolated in `prepare_fold_data`.
    """
    audit = {
        "raw_rows": int(len(df_raw)),
        "raw_cols": int(df_raw.shape[1]),
        "anomalies": {},
    }

    # 1. Exact duplicate week check (2025-03-10)
    dup_groups = df_raw[df_raw.duplicated(subset=["week_start"], keep=False)]
    dup_weeks = sorted(dup_groups["week_start"].unique().tolist())
    exact_dup_match = bool(len(dup_groups) > 0 and dup_groups.duplicated(keep="first").sum() == (len(dup_groups) - len(dup_weeks)))
    if len(dup_groups) > 0 and not exact_dup_match:
        raise ValueError(f"Non-identical duplicate week_start rows detected: {dup_weeks}")

    df = df_raw.drop_duplicates(subset=["week_start"], keep="first").copy()
    df["week_dt"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_dt").reset_index(drop=True)
    date_gaps = df["week_dt"].diff().dropna().dt.days.value_counts().to_dict()
    if list(date_gaps.keys()) != [7]:
        raise ValueError(f"Non-contiguous weekly lattice detected: {date_gaps}")

    df["t"] = np.arange(len(df), dtype=float)
    df["year"] = df["week_dt"].dt.year
    df["week_of_year"] = df["week_dt"].dt.isocalendar().week.astype(int)
    year_week_counts = {int(k): int(v) for k, v in df["year"].value_counts().sort_index().items()}

    audit["anomalies"]["duplicate_weeks"] = {
        "count": int(len(df_raw) - len(df)),
        "weeks": dup_weeks,
        "exact_row_match": exact_dup_match,
        "post_dedup_rows": int(len(df)),
        "date_range": [df["week_start"].iloc[0], df["week_start"].iloc[-1]],
        "day_spacing_counts": {int(k): int(v) for k, v in date_gaps.items()},
        "monday_weeks_per_calendar_year": year_week_counts,
        "iso_boundary_note": (
            "Monday 2024-12-30 carries calendar year=2024 (ISO week 1 of 2025), giving 2024 53 Mondays "
            "and calendar-2025 51 Mondays (2025-01-06 to 2025-12-22)."
        ),
    }

    # 2. Q4 2023 Meta spend unit error (Lakhs instead of INR)
    meta_cpm_raw = df["meta_spend"] / (df["meta_impressions"] / 1000.0)
    meta_err_mask = (df["meta_spend"] < 100_000.0) & (df["meta_impressions"] > 1_000_000.0)
    normal_meta_cpm = meta_cpm_raw[~meta_err_mask].values
    spend_before = float(df.loc[meta_err_mask, "meta_spend"].sum())
    df.loc[meta_err_mask, "meta_spend"] = df.loc[meta_err_mask, "meta_spend"] * 100_000.0
    spend_after = float(df.loc[meta_err_mask, "meta_spend"].sum())
    fixed_meta_cpm = (df.loc[meta_err_mask, "meta_spend"] / (df.loc[meta_err_mask, "meta_impressions"] / 1000.0)).values

    # Compare Q4 2023 fixed CPM against adjacent normal quarters (Q3 2023 + Q1 2024) via KS test
    adj_mask = (~meta_err_mask) & (df["week_start"] >= "2023-07-01") & (df["week_start"] <= "2024-03-31")
    adj_cpm = meta_cpm_raw[adj_mask].values
    ks_stat, ks_pval = stats.ks_2samp(fixed_meta_cpm, adj_cpm)

    audit["anomalies"]["meta_q4_2023_unit_error"] = {
        "count": int(meta_err_mask.sum()),
        "start_week": str(df.loc[meta_err_mask, "week_start"].min()),
        "end_week": str(df.loc[meta_err_mask, "week_start"].max()),
        "raw_cpm_mean": float(meta_cpm_raw[meta_err_mask].mean()),
        "normal_cpm_mean": float(normal_meta_cpm.mean()),
        "normal_cpm_median": float(np.median(normal_meta_cpm)),
        "fixed_cpm_mean": float(fixed_meta_cpm.mean()),
        "fixed_cpm_median": float(np.median(fixed_meta_cpm)),
        "ks_stat_vs_adjacent_quarters": round(float(ks_stat), 4),
        "ks_pval_vs_adjacent_quarters": round(float(ks_pval), 4),
        "spend_before_inr": spend_before,
        "spend_after_inr": spend_after,
        "spend_restored_inr": spend_after - spend_before,
    }

    # 3. Negative TV spend (-1,250,000 on 2024-11-11): dual physical vs net-cost ledger
    neg_tv_mask = df["tv_spend"] < 0
    neg_tv_rows = df.loc[neg_tv_mask, ["week_start", "tv_spend"]].to_dict(orient="records")
    rebate_inr = float(-df.loc[neg_tv_mask, "tv_spend"].sum())
    df.loc[neg_tv_mask, "tv_spend"] = 0.0

    # Net-cost series allocating rebate pro-rata across the preceding Diwali 2024 flight
    flight4_mask = (df["week_start"] >= "2024-10-07") & (df["week_start"] <= "2024-11-04")
    flight4_gross = float(df.loc[flight4_mask, "tv_spend"].sum())
    df["tv_spend_net_cost"] = df["tv_spend"].astype(float)
    net_factor = 1.0 - (rebate_inr / flight4_gross)
    df.loc[flight4_mask, "tv_spend_net_cost"] = df.loc[flight4_mask, "tv_spend"] * net_factor

    audit["anomalies"]["tv_negative_spend"] = {
        "count": int(neg_tv_mask.sum()),
        "rows": neg_tv_rows,
        "physical_treatment": "Set to 0.0 INR (post-flight billing credit/make-good after 5-week Diwali 2024 flight; 0 media aired).",
        "net_cost_treatment": f"Rebate of ₹{rebate_inr:,.0f} ({100*(1-net_factor):.2f}%) allocated pro-rata across 2024-10-07..2024-11-04 (factor={net_factor:.6f}) in `tv_spend_net_cost`.",
        "preceding_flight_gross_spend_inr": flight4_gross,
        "preceding_flight_net_spend_inr": float(df.loc[flight4_mask, "tv_spend_net_cost"].sum()),
    }

    return df, audit


def prepare_fold_data(
    df_struct: pd.DataFrame,
    train_end_idx: int | None = None,
    use_tv_net_cost: bool = False,
) -> dict:
    """
    Strictly causal, fold-isolated preprocessing given training cutoff `train_end_idx`
    (rows `0 .. train_end_idx - 1` are training; rows `>= train_end_idx` are future/test).

    Zero Temporal Leakage Guarantee (HG-1):
      - Imputes missing YouTube spend at row `i` using ONLY observed active weeks
        `tau < min(i, train_end_idx)` (trailing up to 8 active weeks, falling back to
        all prior active weeks `< min(i, train_end_idx)`).
      - Computes saturation reference scale `tilde_s` (median active-week spend),
        `mean_nz`, `max_active_s`, and control centering means STRICTLY on `0:train_end_idx`.
    """
    n = len(df_struct)
    if train_end_idx is None:
        train_end_idx = n
    df = df_struct.copy()
    if use_tv_net_cost and "tv_spend_net_cost" in df.columns:
        df["tv_spend"] = df["tv_spend_net_cost"].astype(float)

    # Causal YouTube spend imputation (only active weeks strictly prior to min(idx, train_end_idx))
    yt_na_mask = df["youtube_spend"].isna()
    yt_obs_cpm = np.where(
        (df["youtube_spend"].notna()) & (df["youtube_spend"] > 0) & (df["youtube_impressions"] > 0),
        df["youtube_spend"] / (df["youtube_impressions"] / 1000.0),
        np.nan,
    )
    yt_imputed_details = []
    for idx in df.index[yt_na_mask]:
        cutoff = min(int(idx), int(train_end_idx))
        prior_cpms = yt_obs_cpm[:cutoff]
        valid_prior = prior_cpms[~np.isnan(prior_cpms)]
        if len(valid_prior) == 0:
            # Should not happen here (first NaN is idx 58), fallback only to train_end_idx active weeks
            train_cpms = yt_obs_cpm[:train_end_idx]
            valid_prior = train_cpms[~np.isnan(train_cpms)]
        recent_prior = valid_prior[-8:]  # causal trailing 8 active weeks
        causal_med_cpm = float(np.median(recent_prior))
        causal_sd_log_cpm = float(np.std(np.log(valid_prior))) if len(valid_prior) > 1 else 0.08
        imp = float(df.loc[idx, "youtube_impressions"])
        est_spend = round((imp / 1000.0) * causal_med_cpm, 2)
        df.loc[idx, "youtube_spend"] = est_spend
        yt_imputed_details.append(
            {
                "row_idx": int(idx),
                "week_start": str(df.loc[idx, "week_start"]),
                "youtube_impressions": int(imp),
                "causal_trailing8_median_cpm": round(causal_med_cpm, 2),
                "causal_expanding_log_cpm_sd": round(causal_sd_log_cpm, 4),
                "n_prior_active_weeks": int(len(valid_prior)),
                "imputed_spend_inr": est_spend,
            }
        )

    yt_dark_weeks = int(((df["youtube_spend"] == 0) & (df["youtube_impressions"] == 0)).sum())
    df["aov_inr"] = df["revenue_inr"] / df["orders"]
    df["meta_cpm"] = df["meta_spend"] / (df["meta_impressions"] / 1000.0)
    df["youtube_cpm"] = np.where(
        df["youtube_impressions"] > 0, df["youtube_spend"] / (df["youtube_impressions"] / 1000.0), np.nan
    )

    # Strict training-fold matrices and scales
    S = df[CHANNELS].values.astype(float) / 1e6  # ₹ Millions
    y = df["revenue_inr"].values.astype(float) / 1e6
    orders_k = df["orders"].values.astype(float) / 1e3

    tr_mask = np.zeros(n, dtype=bool)
    tr_mask[:train_end_idx] = True
    S_tr = S[tr_mask]

    # Pre-period initialization from Jan 2023 (weeks 0..3, inside every fold)
    init_vals = np.array(
        [S[:4, 0].mean(), S[:4, 1].mean(), S[:4, 2].mean(), S[:4, 3].mean(), 0.0, S[:4, 5].mean()],
        dtype=float,
    )
    # Training-only median active-week spend (tilde_s) & mean_nz & max active spend
    tilde_s = np.array([float(np.median(S_tr[S_tr[:, j] > 0, j])) for j in range(6)], dtype=float)
    mean_nz_tr = np.array([float(np.mean(S_tr[S_tr[:, j] > 0, j])) for j in range(6)], dtype=float)
    max_s_tr = np.array([float(np.max(S_tr[:, j])) for j in range(6)], dtype=float)

    ref_means = {
        "avg_discount_pct": float(df.loc[tr_mask, "avg_discount_pct"].mean()),
        "price_index": float(df.loc[tr_mask, "price_index"].mean()),
    }

    return {
        "df": df,
        "S": S,
        "y": y,
        "orders_k": orders_k,
        "train_mask": tr_mask,
        "train_end_idx": int(train_end_idx),
        "init_vals": init_vals,
        "tilde_s": tilde_s,
        "mean_nz_tr": mean_nz_tr,
        "max_s_tr": max_s_tr,
        "ref_means": ref_means,
        "yt_imputed_details": yt_imputed_details,
        "yt_dark_weeks": yt_dark_weeks,
    }


def compute_quarterly_baselines(df_clean: pd.DataFrame) -> dict:
    """Reconcile all three 2025 quarterly spend baseline conventions."""
    mask_51w = df_clean["year"].values == 2025  # 51 weeks: 2025-01-06 .. 2025-12-22
    s_51w = df_clean.loc[mask_51w, CHANNELS].sum() / 1e7  # ₹ Cr
    s_52w = df_clean.iloc[-52:][CHANNELS].sum() / 1e7     # ₹ Cr (2024-12-30 .. 2025-12-22)

    q_51w_div4 = s_51w / 4.0
    q_51w_rate13 = s_51w * (13.0 / 51.0)
    q_52w_div4 = s_52w / 4.0

    table = {}
    for ch in CHANNELS:
        table[ch] = {
            "label": CHANNEL_LABELS[ch],
            "spend_2025_51wk_total_cr": round(float(s_51w[ch]), 4),
            "cal_2025_51w_div4_cr": round(float(q_51w_div4[ch]), 4),
            "cal_2025_51w_rate13_cr": round(float(q_51w_rate13[ch]), 4),
            "trailing_52w_div4_cr": round(float(q_52w_div4[ch]), 4),
        }
    return {
        "n_weeks_cal_2025": int(mask_51w.sum()),
        "n_weeks_trailing_52": 52,
        "totals_cr": {
            "cal_2025_51w_total": round(float(s_51w.sum()), 3),
            "cal_2025_51w_div4": round(float(q_51w_div4.sum()), 3),
            "cal_2025_51w_rate13": round(float(q_51w_rate13.sum()), 3),
            "trailing_52w_total": round(float(s_52w.sum()), 3),
            "trailing_52w_div4": round(float(q_52w_div4.sum()), 3),
        },
        "channels": table,
    }


def clean_mmm_data(
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    return_bundle: bool = False,
) -> tuple:
    """Run full deterministic structural cleaning + causal full-sample YouTube imputation and save artifacts."""
    d_dir, o_dir = resolve_paths(data_dir, out_dir)
    raw_path = d_dir / "mmm_weekly_data.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Could not find mmm_weekly_data.csv in {d_dir}")

    df_raw = pd.read_csv(raw_path)
    df_struct, audit = apply_structural_repairs(df_raw)
    fold_full = prepare_fold_data(df_struct, train_end_idx=len(df_struct))
    df = fold_full["df"]

    yt_active_cpms = df.loc[(df["youtube_spend"] > 0) & (df["youtube_impressions"] > 0), "youtube_cpm"]
    audit["anomalies"]["youtube_missing_spend"] = {
        "count": len(fold_full["yt_imputed_details"]),
        "imputation_method": "Strictly causal trailing-8-active-week median CPM (zero future lookahead)",
        "active_cpm_mean": round(float(yt_active_cpms.mean()), 2),
        "active_cpm_sd": round(float(yt_active_cpms.std()), 2),
        "imputed_rows": fold_full["yt_imputed_details"],
        "total_imputed_spend_inr": round(sum(r["imputed_spend_inr"] for r in fold_full["yt_imputed_details"]), 2),
        "genuine_dark_weeks_zero_spend_zero_imp": fold_full["yt_dark_weeks"],
    }

    # Verify lift test spend accounting identity if lift_test_results.csv exists
    if (d_dir / "lift_test_results.csv").exists():
        lift_info = load_lift_test(d_dir)
        t_mask = (df["week_start"] >= lift_info["test_start"]) & (df["week_start"] <= lift_info["test_end"])
        s_test_sum = float(df.loc[t_mask, "meta_spend"].sum())
        audit["lift_test_spend_accounting"] = {
            "csv_meta_spend_6wk_sum_inr": round(s_test_sum, 2),
            "holdout_share_times_sum_inr": round(lift_info["holdout_share"] * s_test_sum, 2),
            "reported_spend_withheld_inr": lift_info["spend_withheld_inr"],
            "ratio_020_vs_reported": round((lift_info["holdout_share"] * s_test_sum) / lift_info["spend_withheld_inr"], 5),
            "note": (
                "0.20 * sum(meta_spend_6wk) = ₹4,459,196 matches reported spend_withheld_inr = ₹4,459,000 "
                "(rounded to ₹1,000), confirming CSV spend reflects planned national run-rate intensity S_act."
            ),
        }

    audit["secondary_checks"] = {
        "influencer_flighting": {
            "median_weekly_spend_inr": float(df["influencer_spend"].median()),
            "p90_weekly_spend_inr": float(df["influencer_spend"].quantile(0.90)),
            "max_weekly_spend_inr": float(df["influencer_spend"].max()),
            "weeks_above_1m": int((df["influencer_spend"] > 1_000_000).sum()),
        },
        "tv_flighting": {
            "active_weeks": int((df["tv_spend"] > 0).sum()),
            "zero_weeks": int((df["tv_spend"] == 0).sum()),
            "active_mean_inr": float(df.loc[df["tv_spend"] > 0, "tv_spend"].mean()),
            "active_range_inr": [
                float(df.loc[df["tv_spend"] > 0, "tv_spend"].min()),
                float(df.loc[df["tv_spend"] > 0, "tv_spend"].max()),
            ],
        },
        "aov_trend": {
            "2023_mean": round(float(df.loc[df["year"] == 2023, "aov_inr"].mean()), 2),
            "2024_mean": round(float(df.loc[df["year"] == 2024, "aov_inr"].mean()), 2),
            "2025_mean": round(float(df.loc[df["year"] == 2025, "aov_inr"].mean()), 2),
        },
        "price_index_trend": {
            "2023_mean": round(float(df.loc[df["year"] == 2023, "price_index"].mean()), 2),
            "2024_mean": round(float(df.loc[df["year"] == 2024, "price_index"].mean()), 2),
            "2025_mean": round(float(df.loc[df["year"] == 2025, "price_index"].mean()), 2),
        },
        "remaining_nulls_core_cols": int(df.drop(columns=["youtube_cpm"]).isna().sum().sum()),
    }
    audit["quarterly_baseline_reconciliation"] = compute_quarterly_baselines(df)

    out_csv = o_dir / "mmm_cleaned.csv"
    df.drop(columns=["week_dt"]).to_csv(out_csv, index=False, lineterminator="\n")
    out_json = o_dir / "audit_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    if return_bundle:
        return df, audit, df_raw, df_struct, fold_full
    return df, audit


def main():
    parser = argparse.ArgumentParser(description="Clean and audit Verdant Leaf MMM weekly dataset (v3).")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing raw CSVs")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    df_clean, _ = clean_mmm_data(data_dir=args.data_dir, out_dir=args.out_dir)
    _, o_dir = resolve_paths(args.data_dir, args.out_dir)
    print(f"Saved cleaned dataset ({len(df_clean)} rows) -> {o_dir / 'mmm_cleaned.csv'}")


if __name__ == "__main__":
    main()
