"""
model_mmm.py — SVD-Exact v0 Diagnosis, Causal Structural Diagnostics,
Generative Bayesian MMM (Orthogonalized Concave Saturation + Smooth Baseline + AR(1)),
Exact 20%-Geo-Dark Lift Calibration, Multi-Chain Hessian-Preconditioned HMC,
Leak-Free Expanding-Window CV, and Yao-Vehtari Bayesian Stacking (Verdant Leaf v3).

Zero hardcoded paths; uses only numpy, scipy, pandas, and statsmodels.
"""
import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from clean_data import (  # Imported before numpy so BLAS/OMP thread caps take effect
    CHANNELS,
    CHANNEL_LABELS,
    CONTROLS,
    apply_structural_repairs,
    clean_mmm_data,
    load_lift_test,
    prepare_fold_data,
    resolve_paths,
)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import logsumexp
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.regression.linear_model import OLS

YEAR_WEEKS = 52.1775  # Exact astronomical year in weeks (365.2425 / 7)
SPLINE_KNOTS = (52.0, 104.0)  # Smooth annual trend-hinge knots (extrapolate cleanly into 2026)

# Beta(a, b) priors on adstock alpha in (0, 1) — smooth, no artificial hard wall pile-ups
ALPHA_BETA_PRIORS = np.array(
    [
        [2.5, 4.5],   # Meta (mean 0.357)
        [1.2, 10.0],  # Google Search (mean 0.107)
        [3.5, 2.5],   # YouTube (mean 0.583)
        [3.5, 2.5],   # Influencer (mean 0.583)
        [4.5, 2.0],   # TV (mean 0.692)
        [1.2, 10.0],  # Amazon Ads (mean 0.107)
    ],
    dtype=float,
)

# Generative Gamma(1.35, scale=r0_sd/1.35) scales on initial zero-spend mROAS:
# mode = 0.259 * r0_sd (e.g., 0.47x Search, 0.65x TV -> zero upward pull!)
R0_GAMMA_SHAPE = 1.35
R0_PRIOR_BASE_SCALE = np.array([3.0, 1.8, 3.5, 3.5, 2.5, 2.2], dtype=float)
KAPPA_LOG_MEAN = np.log(0.65)
KAPPA_LOG_SD = 0.55

WEIBULL_MAX_LAG = 12
WEIBULL_CH_IDX = [2, 3, 4]   # YouTube, Influencer, TV delayed-peak sensitivity
BOUNDED_HILL_IDX = [2, 3, 4] # Upper-funnel bounded threshold sensitivity (Search & Amazon locked to s=1)


# ---------------------------------------------------------------------------
# 1. Exact SVD OLS & Baseline v0 Diagnostics (B2)
# ---------------------------------------------------------------------------
def fit_ols_svd(X_df: pd.DataFrame, y_ser: pd.Series) -> dict:
    """
    Numerically stable OLS via direct SVD on X (never pinv(X.T @ X), which squares kappa(X)).
    Matches statsmodels.OLS exact SEs and p-values even at kappa(X) ~ 3.5e7.
    """
    X = np.column_stack([np.ones(len(X_df)), X_df.values.astype(float)])
    col_names = ["const"] + list(X_df.columns)
    y = y_ser.values.astype(float)
    n, k = X.shape

    u, s, vt = np.linalg.svd(X, full_matrices=False)
    cond_num = float(s[0] / max(s[-1], 1e-18))
    beta = vt.T @ ((u.T @ y) / s)
    resid = y - X @ beta
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    s2 = rss / (n - k)

    cov = s2 * (vt.T @ np.diag(1.0 / (s ** 2)) @ vt)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    t_stat = beta / se
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=n - k))

    # Newey-West HAC (lag=4) SEs for robust inference under serial correlation
    ols_sm = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    se_hac = np.asarray(ols_sm.bse, dtype=float)
    p_hac = np.asarray(ols_sm.pvalues, dtype=float)

    dw = float(np.sum(np.diff(resid) ** 2) / (rss + 1e-18))
    lb = acorr_ljungbox(resid, lags=[4], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_pval = float(lb["lb_pvalue"].iloc[0])
    _, bp_pval, _, _ = het_breuschpagan(resid, X)
    mape = float(np.mean(np.abs(resid / (np.abs(y) + 1e-12))) * 100.0)

    vifs = {"const": None}
    for i, c in enumerate(X_df.columns):
        others = [j for j in range(X_df.shape[1]) if j != i]
        if not others:
            vifs[c] = 1.0
            continue
        Xo = np.column_stack([np.ones(n), X_df.values[:, others]])
        yo = X_df.values[:, i]
        bo = np.linalg.lstsq(Xo, yo, rcond=None)[0]
        denom = float(np.sum((yo - yo.mean()) ** 2))
        r2_o = 1.0 - float(np.sum((yo - Xo @ bo) ** 2)) / (denom + 1e-18)
        vifs[c] = float(1.0 / (1.0 - r2_o)) if r2_o < 0.999999 else float("inf")

    coefs = {}
    for idx, c in enumerate(col_names):
        coefs[c] = {
            "coef": float(beta[idx]),
            "se": float(se[idx]),
            "t_stat": float(t_stat[idx]),
            "p_val": float(p_val[idx]),
            "se_hac4": float(se_hac[idx]),
            "p_val_hac4": float(p_hac[idx]),
            "vif": vifs[c],
        }
    return {
        "n": int(n),
        "k": int(k),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": float(np.sqrt(rss / n)),
        "mape": mape,
        "dw": dw,
        "cond_num": cond_num,
        "ljung_box_lag4_stat": round(lb_stat, 3),
        "ljung_box_lag4_pval": round(lb_pval, 6),
        "breusch_pagan_pval": round(float(bp_pval), 6),
        "coefs": coefs,
    }


def stable_sigmoid(x: np.ndarray | float, clip: float = 15.0) -> np.ndarray | float:
    """Numerically stable logistic sigmoid 1 / (1 + exp(-clip(x)))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -clip, clip)))


def stable_softplus(x: np.ndarray | float, lo: float = -18.0, hi: float = 15.0) -> np.ndarray | float:
    """Numerically stable softplus log(1 + exp(x)) linearizing smoothly above `hi`."""
    return np.where(x > hi, x, np.log1p(np.exp(np.clip(x, lo, hi))))


def _beta_logistic_lp_grad(
    sig: np.ndarray | float,
    a: np.ndarray | float,
    b: np.ndarray | float,
) -> tuple[float, np.ndarray | float]:
    """Log-density + d/du gradient for sig = sigmoid(u) under Beta(a, b) prior * Jacobian."""
    lp = float(np.sum(a * np.log(np.clip(sig, 1e-12, 1.0)) + b * np.log(np.clip(1.0 - sig, 1e-12, 1.0))))
    return lp, a - (a + b) * sig


def _whiten_ar1(arr: np.ndarray, rho: float, sq_1r2: float) -> np.ndarray:
    """Exact Prais-Winsten AR(1) whitening transform along axis 0."""
    out = np.empty_like(arr)
    out[0] = sq_1r2 * arr[0]
    out[1:] = arr[1:] - rho * arr[:-1]
    return out


def compute_fit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard fit diagnostics (R2, RMSE, MAPE %, NRMSE %, Durbin-Watson)."""
    resid = y_true - y_pred
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y_true - y_true.mean()) ** 2))
    rmse = float(np.sqrt(rss / len(y_true)))
    return {
        "r2": float(1.0 - rss / (tss + 1e-18)),
        "rmse": rmse,
        "mape_pct": float(np.mean(np.abs(resid / (np.abs(y_true) + 1e-12))) * 100.0),
        "nrmse_pct": float(rmse / y_true.mean() * 100.0),
        "dw": float(np.sum(np.diff(resid) ** 2) / (rss + 1e-18)),
    }


def make_seas_and_spline_cols(t_arr: np.ndarray, use_spline: bool = True, K: int = 2) -> tuple[np.ndarray, list[str]]:
    cols = [t_arr / YEAR_WEEKS]
    names = ["trend_yr"]
    if use_spline:
        for k_i, knot in enumerate(SPLINE_KNOTS, start=1):
            scale = 10.0 / YEAR_WEEKS
            h_t = scale * np.logaddexp(0.0, (t_arr - knot) / 10.0) - scale * np.logaddexp(0.0, -knot / 10.0)
            cols.append(h_t)
            names.append(f"trend_hinge_{k_i}")
    for h in range(1, K + 1):
        cols.append(np.sin(2.0 * np.pi * h * t_arr / YEAR_WEEKS))
        cols.append(np.cos(2.0 * np.pi * h * t_arr / YEAR_WEEKS))
        names.extend([f"sin_{h}", f"cos_{h}"])
    return np.column_stack(cols), names


def build_base_design(
    df: pd.DataFrame,
    ref_means: dict,
    use_spline: bool = True,
    K: int = 2,
    controls: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    ctrl_list = CONTROLS if controls is None else controls
    t = df["t"].values.astype(float)
    X_seas, seas_names = make_seas_and_spline_cols(t, use_spline=use_spline, K=K)
    X_ctrl = df[ctrl_list].values.astype(float).copy()
    for idx, c in enumerate(ctrl_list):
        if c in ref_means:
            X_ctrl[:, idx] -= ref_means[c]
    X_base = np.column_stack([np.ones(len(df)), X_seas, X_ctrl])
    col_names = ["const"] + seas_names + list(ctrl_list)
    return X_base, col_names


def project_exante_covariates(
    df_hist: pd.DataFrame,
    future_t: np.ndarray,
    future_woy: np.ndarray,
    price_fit_df: pd.DataFrame | None = None,
    launch_fit_df: pd.DataFrame | None = None,
    zero_tentpoles: bool = False,
) -> pd.DataFrame:
    """
    Project future exogenous controls strictly from historical rows (linear price_index drift
    + quarter-of-year historical means) for both CV ex-ante validation and Q1 2026 P&L forecasts.
    """
    pr_df = df_hist if price_fit_df is None else price_fit_df
    ln_df = df_hist if launch_fit_df is None else launch_fit_df
    slope_pr, int_pr = np.polyfit(pr_df["t"].values, pr_df["price_index"].values, 1)
    hist_q = ((df_hist["week_of_year"].values - 1) // 13) % 4 + 1
    ln_q = ((ln_df["week_of_year"].values - 1) // 13) % 4 + 1

    rows = []
    for t_val, woy_val in zip(future_t, future_woy):
        q_i = ((int(woy_val) - 1) // 13) % 4 + 1
        q_tr = df_hist[hist_q == q_i]
        if len(q_tr) == 0:
            q_tr = df_hist
        q_ln = ln_df[ln_q == q_i]
        if len(q_ln) == 0:
            q_ln = ln_df
        rows.append(
            {
                "t": float(t_val),
                "festive_flag": 0.0 if zero_tentpoles else float(q_tr["festive_flag"].mean()),
                "marketplace_sale_flag": 0.0 if zero_tentpoles else float(q_tr["marketplace_sale_flag"].mean()),
                "avg_discount_pct": float(q_tr["avg_discount_pct"].mean()),
                "price_index": float(int_pr + slope_pr * float(t_val)),
                "competitor_promo_flag": float(q_tr["competitor_promo_flag"].mean()),
                "new_launch_flag": float(q_ln["new_launch_flag"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_baseline_and_waterfall(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """Reproduce baseline_mmm.py (v0) via exact SVD OLS and run stepwise distortion waterfall."""
    df_b = df_raw.dropna().copy()
    base_res = fit_ols_svd(df_b[CHANNELS + ["festive_flag", "avg_discount_pct"]], df_b["revenue_inr"])

    steps = {"0_raw_baseline": base_res}
    d1a = df_raw.drop_duplicates(subset=["week_start"]).dropna().copy()
    steps["1a_dedup_only"] = fit_ols_svd(d1a[CHANNELS + ["festive_flag", "avg_discount_pct"]], d1a["revenue_inr"])

    d1b = d1a.copy()
    d1b.loc[d1b["meta_spend"] < 100_000, "meta_spend"] *= 100_000.0
    steps["1b_meta_unit_fix"] = fit_ols_svd(d1b[CHANNELS + ["festive_flag", "avg_discount_pct"]], d1b["revenue_inr"])

    d1c = df_clean.copy()
    d1c_notv = d1c.copy()
    d1c_notv.loc[d1c_notv["week_start"] == "2024-11-11", "tv_spend"] = -1_250_000.0
    steps["1c_yt_imputation"] = fit_ols_svd(d1c_notv[CHANNELS + ["festive_flag", "avg_discount_pct"]], d1c_notv["revenue_inr"])
    steps["1d_tv_credit_fix"] = fit_ols_svd(d1c[CHANNELS + ["festive_flag", "avg_discount_pct"]], d1c["revenue_inr"])

    X_s, s_names = make_seas_and_spline_cols(d1c["t"].values, use_spline=False, K=2)
    d2 = d1c.copy()
    for i, nm in enumerate(s_names):
        d2[nm] = X_s[:, i]
    steps["2_trend_seasonality"] = fit_ols_svd(d2[CHANNELS + ["festive_flag", "avg_discount_pct"] + s_names], d2["revenue_inr"])
    steps["3_all_controls"] = fit_ols_svd(d2[CHANNELS + CONTROLS + s_names], d2["revenue_inr"])
    steps["4a_plus_raw_brand_search_bad_control"] = fit_ols_svd(
        d2[CHANNELS + CONTROLS + s_names + ["brand_search_index"]], d2["revenue_inr"]
    )
    steps["4b_plus_raw_organic_traffic_bad_control"] = fit_ols_svd(
        d2[CHANNELS + CONTROLS + s_names + ["organic_traffic_index"]], d2["revenue_inr"]
    )

    # Auxiliary Orders volume check (confirms unit volume response vs price_index drift)
    orders_res = fit_ols_svd(d2[CHANNELS + CONTROLS + s_names], d2["orders"] * 565.0)
    steps["5_orders_volume_x_med_aov"] = orders_res

    waterfall = {}
    for step_name, res in steps.items():
        waterfall[step_name] = {
            "n": res["n"],
            "r2": round(res["r2"], 4),
            "adj_r2": round(res["adj_r2"], 4),
            "dw": round(res["dw"], 3),
            "rmse_inr": round(res["rmse"], 0),
            "channel_coefs": {ch: round(res["coefs"][ch]["coef"], 4) for ch in CHANNELS},
        }
    return {"baseline_full": base_res, "waterfall": waterfall}


# ---------------------------------------------------------------------------
# 2. Adstock & Orthogonalized Concave Saturation Helpers
# ---------------------------------------------------------------------------
def adstock_norm_and_grad(x: np.ndarray, alpha: float, init_val: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Normalized geometric adstock a_t = (1-alpha)*x_t + alpha*a_{t-1} and exact da_t/dalpha."""
    if alpha <= 1e-12:
        ad = x.astype(float, copy=True)
        ad_prev = np.empty_like(ad)
        ad_prev[0] = float(init_val)
        ad_prev[1:] = ad[:-1]
        return ad, ad_prev - x
    zi = np.array([alpha * float(init_val)], dtype=float)
    ad, _ = lfilter([1.0 - alpha], [1.0, -alpha], x, zi=zi)
    ad_prev = np.empty_like(ad)
    ad_prev[0] = float(init_val)
    ad_prev[1:] = ad[:-1]
    dad = lfilter([1.0], [1.0, -alpha], ad_prev - x)
    return ad, dad


def build_lag_matrix(x: np.ndarray, init_val: float, max_lag: int = WEIBULL_MAX_LAG) -> np.ndarray:
    """Build (T, max_lag+1) lag matrix padded with `init_val` prior to t=0."""
    T = len(x)
    L = max_lag + 1
    M = np.empty((T, L), dtype=float)
    for ell in range(L):
        if ell == 0:
            M[:, 0] = x
        else:
            M[:ell, ell] = init_val
            M[ell:, ell] = x[:-ell]
    return M


def weibull_weights_and_grads(alpha: float, k_lag: float, max_lag: int = WEIBULL_MAX_LAG):
    """Normalized discrete Weibull kernel w_ell proportional to (ell+1)^(k-1) * alpha^((ell+1)^k)."""
    ell1 = np.arange(1, max_lag + 2, dtype=float)
    p = ell1 ** k_lag
    log_q = (k_lag - 1.0) * np.log(ell1) + p * np.log(np.clip(alpha, 1e-9, 1.0 - 1e-9))
    log_q -= np.max(log_q)
    q = np.exp(log_q)
    q_sum = np.sum(q)
    w = q / q_sum

    dq_da = q * (p / np.clip(alpha, 1e-9, 1.0 - 1e-9))
    dw_da = (dq_da * q_sum - q * np.sum(dq_da)) / (q_sum ** 2)

    ln_ell = np.log(ell1)
    dq_dk = q * ln_ell * (1.0 + p * np.log(np.clip(alpha, 1e-9, 1.0 - 1e-9)))
    dw_dk = (dq_dk * q_sum - q * np.sum(dq_dk)) / (q_sum ** 2)
    return w, dw_da, dw_dk


def contrib_mm_and_grads(ad: np.ndarray, r0: float, kappa: float, tilde_s: float):
    """
    Orthogonalized concave Michaelis-Menten:
      C = r0 * ad / (1 + kappa * (ad / tilde_s))
    Returns (C, dC/dr0, dC/d(log kappa), dC/dad).
    """
    a = np.maximum(ad, 0.0)
    z = a / tilde_s
    denom = 1.0 + kappa * z
    c = (r0 * a) / denom
    dC_dr0 = a / denom
    dC_d_u_kap = -c * (kappa * z / denom)
    dC_d_ad = np.where(ad >= 0.0, r0 / (denom * denom), 0.0)
    return c, dC_dr0, dC_d_u_kap, dC_d_ad


def hill_unit_response(
    ad: np.ndarray,
    K: np.ndarray | float,
    slope: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcastable Hill unit saturation u = (a/K)^s / (1 + (a/K)^s) for 1D or ND arrays."""
    a = np.maximum(ad, 0.0)
    pos = a > 0.0
    safe_a = np.where(pos, a, 1.0)
    ratio_s = np.where(pos, (safe_a / K) ** slope, 0.0)
    u = ratio_s / (1.0 + ratio_s)
    return u, pos, safe_a


def contrib_hill_and_grads(ad: np.ndarray, r0: float, kappa: float, slope: float, tilde_s: float):
    """
    Reparameterized Hill matched in initial/scale space:
      Let K = tilde_s / kappa, beta = r0 * K. Then C = (beta * a^s) / (a^s + K^s).
    Returns (C, dC/dr0, dC/d(log kappa), dC/dslope, dC/dad).
    """
    if abs(slope - 1.0) < 1e-8:
        c, dr0, dkap, dad = contrib_mm_and_grads(ad, r0, kappa, tilde_s)
        return c, dr0, dkap, np.zeros_like(c), dad
    K = tilde_s / kappa
    beta = r0 * K
    u, pos, safe_a = hill_unit_response(ad, K, slope)
    c = beta * u
    u1u = u * (1.0 - u)
    dC_dr0 = K * u
    dC_d_u_kap = -c * (1.0 - slope * (1.0 - u))
    dC_d_slope = np.where(pos, beta * u1u * np.log(safe_a / K), 0.0)
    dC_d_ad = np.where(pos, beta * slope * u1u / safe_a, 0.0)
    return c, dC_dr0, dC_d_u_kap, dC_d_slope, dC_d_ad


def compute_media_channels(
    S: np.ndarray,
    p: dict,
    init_vals: np.ndarray,
    tilde_s: np.ndarray,
    use_weibull_uf: bool = False,
    lag_mats: dict | None = None,
    need_grad: bool = False,
):
    """
    Unified 6-channel adstock (geometric or discrete Weibull) + saturation forward pass,
    optionally returning exact analytical derivatives for HMC.
    """
    T = len(S)
    C_mat = np.empty((T, 6), dtype=float)
    alphas = p["alphas"]
    r0 = p["r0"]
    kappa = p["kappa"]
    slopes = p["slopes"]
    k_lags = p["k_lags"]

    if not need_grad:
        for j in range(6):
            use_w = j in WEIBULL_CH_IDX and (use_weibull_uf or k_lags[j] > 1.0001)
            if use_w:
                w, _, _ = weibull_weights_and_grads(alphas[j], k_lags[j])
                Lm = lag_mats[j] if (lag_mats is not None and j in lag_mats) else build_lag_matrix(S[:, j], init_vals[j], WEIBULL_MAX_LAG)
                ad = Lm @ w
            else:
                ad, _ = adstock_norm_and_grad(S[:, j], alphas[j], init_vals[j])
            C_mat[:, j] = contrib_hill_and_grads(ad, r0[j], kappa[j], slopes[j], tilde_s[j])[0]
        return C_mat

    dC_dr0 = np.empty((T, 6), dtype=float)
    dC_dukap = np.empty((T, 6), dtype=float)
    dC_dad = np.empty((T, 6), dtype=float)
    dad_da = np.empty((T, 6), dtype=float)
    dad_dk = np.zeros((T, 6), dtype=float)
    dC_dsl = np.zeros((T, 6), dtype=float)

    for j in range(6):
        if use_weibull_uf and j in WEIBULL_CH_IDX:
            w, dw_da, dw_dk = weibull_weights_and_grads(alphas[j], k_lags[j])
            Lm = lag_mats[j]
            ad = Lm @ w
            dad_da[:, j] = Lm @ dw_da
            dad_dk[:, j] = Lm @ dw_dk
        else:
            ad, da = adstock_norm_and_grad(S[:, j], alphas[j], init_vals[j])
            dad_da[:, j] = da

        c, dr0, dkap, dsl, dad = contrib_hill_and_grads(ad, r0[j], kappa[j], slopes[j], tilde_s[j])
        C_mat[:, j] = c
        dC_dr0[:, j] = dr0
        dC_dukap[:, j] = dkap
        dC_dad[:, j] = dad
        dC_dsl[:, j] = dsl
    return C_mat, dC_dr0, dC_dukap, dC_dad, dad_da, dad_dk, dC_dsl


# ---------------------------------------------------------------------------
# 3. Causal DAG & Identification Audits (B3, HG-3)
# ---------------------------------------------------------------------------
def run_causal_and_identification_audits(
    df_clean: pd.DataFrame,
    fold_full: dict,
    lift_info: dict,
) -> dict:
    """
    Execute causal mediator audits, lead-spend Granger falsification,
    Google Search demand-endogeneity (without .bfill()), Amazon Ads sale interaction,
    and TV 6-flight LOFO + Diwali + net-cost sensitivity.
    """
    S = fold_full["S"]
    y = fold_full["y"]
    init_vals = fold_full["init_vals"]
    t = df_clean["t"].values.astype(float)

    def detrend(x):
        m, b = np.polyfit(t, x, 1)
        return x - (m * t + b)

    bs_d = detrend(df_clean["brand_search_index"].values)
    org_d = detrend(df_clean["organic_traffic_index"].values)
    rev_d = detrend(df_clean["revenue_inr"].values)

    alphas_ref = [0.30, 0.08, 0.55, 0.65, 0.70, 0.08]
    ad_df = pd.DataFrame(
        {f"{ch}_ad": adstock_norm_and_grad(S[:, j], alphas_ref[j], init_vals[j])[0] for j, ch in enumerate(CHANNELS)}
    )
    X_base, base_names = build_base_design(df_clean, fold_full["ref_means"], use_spline=True, K=2)
    base_df = pd.DataFrame(X_base[:, 1:], columns=base_names[1:])
    X_med = pd.concat([ad_df, base_df], axis=1)

    org_reg = fit_ols_svd(X_med, df_clean["organic_traffic_index"])
    bs_reg = fit_ols_svd(X_med, df_clean["brand_search_index"])

    mediator_audit = {
        "detrended_corr_rev_brand_search": round(float(np.corrcoef(rev_d, bs_d)[0, 1]), 3),
        "detrended_corr_rev_organic_traffic": round(float(np.corrcoef(rev_d, org_d)[0, 1]), 3),
        "raw_corr_google_search_brand_search": round(
            float(np.corrcoef(df_clean["google_search_spend"], df_clean["brand_search_index"])[0, 1]), 3
        ),
        "organic_traffic_as_outcome": {
            "r2": round(org_reg["r2"], 4),
            "media_t_stats_hac4": {
                f"{ch}_ad": round(org_reg["coefs"][f"{ch}_ad"]["coef"] / org_reg["coefs"][f"{ch}_ad"]["se_hac4"], 2)
                for ch in CHANNELS
            },
        },
        "brand_search_as_outcome": {
            "r2": round(bs_reg["r2"], 4),
            "media_t_stats_hac4": {
                f"{ch}_ad": round(bs_reg["coefs"][f"{ch}_ad"]["coef"] / bs_reg["coefs"][f"{ch}_ad"]["se_hac4"], 2)
                for ch in CHANNELS
            },
        },
    }

    # Google Search demand-endogeneity audit (strictly drop first 2 rows instead of .bfill()!)
    df_g = pd.DataFrame(
        {
            "google_spend_m": df_clean["google_search_spend"].values / 1e6,
            "brand_search_index": df_clean["brand_search_index"].values,
            "rev_lag1_m": pd.Series(y).shift(1).values,
            "rev_lag2_m": pd.Series(y).shift(2).values,
            "trend_yr": df_clean["t"].values / YEAR_WEEKS,
            "festive_flag": df_clean["festive_flag"].values,
            "marketplace_sale_flag": df_clean["marketplace_sale_flag"].values,
        }
    ).iloc[2:].reset_index(drop=True)
    reg_dem = fit_ols_svd(df_g[["brand_search_index", "rev_lag1_m", "rev_lag2_m"]], df_g["google_spend_m"])
    reg_dem_full = fit_ols_svd(
        df_g[["brand_search_index", "rev_lag1_m", "rev_lag2_m", "trend_yr", "festive_flag", "marketplace_sale_flag"]],
        df_g["google_spend_m"],
    )

    # Lead-spend Granger falsification (future S_{t+1} predicting current y_t)
    X_lead = pd.concat([ad_df, base_df], axis=1).iloc[:-1].reset_index(drop=True)
    y_lead = pd.Series(y[:-1])
    lead_pvals_hac4 = {}
    for j, ch in enumerate(CHANNELS):
        X_tmp = X_lead.copy()
        X_tmp[f"{ch}_lead1"] = S[1:, j]
        r_tmp = fit_ols_svd(X_tmp, y_lead)
        lead_pvals_hac4[ch] = round(r_tmp["coefs"][f"{ch}_lead1"]["p_val_hac4"], 4)

    # Amazon Ads x Marketplace Sale interaction check
    X_amz = pd.concat([ad_df, base_df], axis=1)
    X_amz["amz_x_sale"] = X_amz["amazon_ads_spend_ad"] * X_amz["marketplace_sale_flag"]
    amz_int_res = fit_ols_svd(X_amz, pd.Series(y))

    google_id = {
        "confidence": "LOW",
        "raw_corr_google_revenue": round(float(np.corrcoef(df_clean["google_search_spend"], df_clean["revenue_inr"])[0, 1]), 3),
        "raw_corr_google_brand_search": round(float(np.corrcoef(df_clean["google_search_spend"], df_clean["brand_search_index"])[0, 1]), 3),
        "demand_driven_r2_core_drop2": round(reg_dem["r2"], 4),
        "demand_driven_r2_with_controls_drop2": round(reg_dem_full["r2"], 4),
        "exogenous_residual_variance_share_pct": round(100.0 * (1.0 - reg_dem["r2"]), 2),
        "lead_1wk_granger_pvals_hac4": lead_pvals_hac4,
        "amazon_sale_interaction_coef": round(amz_int_res["coefs"]["amz_x_sale"]["coef"], 3),
        "amazon_sale_interaction_pval_hac4": round(amz_int_res["coefs"]["amz_x_sale"]["p_val_hac4"], 4),
    }

    return {"mediator_audit": mediator_audit, "google_search_and_falsification": google_id}


# ---------------------------------------------------------------------------
# 4. Generative Bayesian MMM Specification + Exact Log-Posterior & Grad
# ---------------------------------------------------------------------------
class BayesianMMMModel:
    """Encapsulates one generative MMM specification with exact log-posterior and analytical gradient."""

    def __init__(
        self,
        S: np.ndarray,
        y: np.ndarray,
        X_base: np.ndarray,
        base_names: list[str],
        train_mask: np.ndarray,
        init_vals: np.ndarray,
        tilde_s: np.ndarray,
        test_idx: np.ndarray,
        eval_idx: np.ndarray,
        lift_info: dict,
        use_lift: bool = True,
        prior_scale_mult: float = 1.0,
        use_weibull_uf: bool = False,
        bounded_hill_uf: bool = False,
        unconstrained_hill_test: bool = False,
        regime_a2_reallocated_lift: bool = False,
    ):
        self.S = S
        self.y = y
        self.X_base = X_base
        self.base_names = base_names
        self.train_mask = train_mask
        self.init_vals = init_vals
        self.tilde_s = tilde_s
        self.test_idx = test_idx
        self.eval_idx = eval_idx
        self.lift_info = lift_info
        self.use_lift = use_lift
        self.prior_scale_mult = prior_scale_mult
        self.use_weibull_uf = use_weibull_uf
        self.bounded_hill_uf = bounded_hill_uf
        self.unconstrained_hill_test = unconstrained_hill_test
        self.regime_a2 = regime_a2_reallocated_lift

        self.kb = X_base.shape[1]
        self.n_tr = int(train_mask.sum())
        self.y_tr = y[train_mask]
        self.Xb_tr = X_base[train_mask]
        self.r0_sds = R0_PRIOR_BASE_SCALE * prior_scale_mult

        # Precompute lag matrices if Weibull delayed-peak adstock is enabled
        if self.use_weibull_uf:
            self.lag_mats = {
                j: build_lag_matrix(S[:, j], init_vals[j], WEIBULL_MAX_LAG) for j in WEIBULL_CH_IDX
            }
        else:
            self.lag_mats = {}

        # Baseline symmetric Gaussian prior mean and SD vector
        self.base_prior_mean = np.zeros(self.kb, dtype=float)
        self.base_prior_sd = np.full(self.kb, 6.0, dtype=float)
        for i, nm in enumerate(base_names):
            if nm == "const":
                self.base_prior_mean[i] = float(np.mean(self.y_tr))
                self.base_prior_sd[i] = 15.0
            elif nm == "trend_yr":
                self.base_prior_mean[i] = 5.0
                self.base_prior_sd[i] = 5.0
            elif nm.startswith("trend_hinge"):
                self.base_prior_mean[i] = 0.0
                self.base_prior_sd[i] = 2.5
            elif nm.startswith("sin_") or nm.startswith("cos_"):
                self.base_prior_mean[i] = 0.0
                self.base_prior_sd[i] = 4.0
            elif nm == "price_index":
                # Weakly negative-centered prior conditional on smooth brand trend
                self.base_prior_mean[i] = -0.15
                self.base_prior_sd[i] = 0.25
            elif nm == "avg_discount_pct":
                self.base_prior_mean[i] = 0.35
                self.base_prior_sd[i] = 0.60

        # Precompute exact 20%-geo-dark counterfactual spend trajectories (HG-5)
        self.holdout_share = float(lift_info["holdout_share"])
        self.s0_act = S[:, 0].copy()
        if self.regime_a2:
            self.s0_act[test_idx] = self.s0_act[test_idx] / (1.0 - self.holdout_share)
        self.s0_dark = self.s0_act.copy()
        self.s0_dark[test_idx] = 0.0
        self.meta_withheld = float(self.holdout_share * np.sum(self.s0_act[test_idx]))

        # Marginalized nonlinear dimension (b_b analytically integrated out!)
        self.base_prior_prec = 1.0 / (self.base_prior_sd ** 2)
        self.n_extra_w = len(WEIBULL_CH_IDX) if use_weibull_uf else 0
        self.n_extra_s = 6 if unconstrained_hill_test else (len(BOUNDED_HILL_IDX) if bounded_hill_uf else 0)
        self.dim = 18 + 2 + self.n_extra_w + self.n_extra_s

    def unpack(self, u: np.ndarray) -> dict:
        u_r0 = u[0:6]
        u_kap = u[6:12]
        u_alpha = u[12:18]
        u_rho = float(u[18])
        u_sig = float(u[19])

        sig_r0 = stable_sigmoid(u_r0, clip=18.0)
        r0 = np.maximum(stable_softplus(u_r0, lo=-18.0, hi=15.0), 1e-12)
        kappa = np.exp(np.clip(u_kap, -12.0, 10.0))
        sig_a = stable_sigmoid(u_alpha, clip=15.0)
        sig_r = float(stable_sigmoid(u_rho, clip=15.0))
        rho = -0.35 + 1.15 * sig_r
        sig_sy = float(stable_sigmoid(u_sig, clip=18.0))
        sigma_y = 0.25 + float(stable_softplus(u_sig, lo=-18.0, hi=15.0))

        idx_cur = 20
        k_lags = np.ones(6, dtype=float)
        sig_w = None
        if self.use_weibull_uf:
            u_w = u[idx_cur : idx_cur + self.n_extra_w]
            sig_w = stable_sigmoid(u_w, clip=15.0)
            k_lags_sub = 0.80 + 1.40 * sig_w
            for pos, j in enumerate(WEIBULL_CH_IDX):
                k_lags[j] = k_lags_sub[pos]
            idx_cur += self.n_extra_w

        slopes = np.ones(6, dtype=float)
        sig_s = None
        if self.n_extra_s > 0:
            u_s = u[idx_cur : idx_cur + self.n_extra_s]
            sig_s = stable_sigmoid(u_s, clip=15.0)
            if self.unconstrained_hill_test:
                slopes = 0.60 + 2.40 * sig_s
            else:
                slopes_sub = 0.85 + 0.45 * sig_s
                for pos, j in enumerate(BOUNDED_HILL_IDX):
                    slopes[j] = slopes_sub[pos]

        return {
            "u_r0": u_r0,
            "u_kap": u_kap,
            "u_alpha": u_alpha,
            "u_rho": u_rho,
            "u_sig": u_sig,
            "sig_r0": sig_r0,
            "r0": r0,
            "kappa": kappa,
            "alphas": sig_a,
            "sig_r": sig_r,
            "rho": float(rho),
            "sig_sy": sig_sy,
            "sigma_y": sigma_y,
            "k_lags": k_lags,
            "sig_w": sig_w,
            "slopes": slopes,
            "sig_s": sig_s,
        }

    def compute_media(
        self,
        p: dict,
        S_override: np.ndarray | None = None,
        lag_mats_override: dict | None = None,
        need_grad: bool = False,
    ):
        S_in = self.S if S_override is None else S_override
        Lm_in = self.lag_mats if (S_override is None and lag_mats_override is None) else lag_mats_override
        return compute_media_channels(
            S_in, p, self.init_vals, self.tilde_s,
            use_weibull_uf=self.use_weibull_uf, lag_mats=Lm_in, need_grad=need_grad,
        )

    def _compute_media_matrix(self, p: dict) -> np.ndarray:
        return self.compute_media(p)

    def _cond_b_b_posterior(self, C_sum_tr: np.ndarray, rho: float, sigma_y: float):
        """Exact Gaussian conditional posterior (b_b_hat, L_c, Lambda, Xb_w, r, e, delta_hat)."""
        one_m_r2 = max(1.0 - rho * rho, 1e-8)
        sq_1r2 = np.sqrt(one_m_r2)

        m_vec = self.y_tr - C_sum_tr
        m0 = m_vec - self.Xb_tr @ self.base_prior_mean
        Xb_w = _whiten_ar1(self.Xb_tr, rho, sq_1r2)
        m0_w = _whiten_ar1(m0, rho, sq_1r2)

        inv_s2 = 1.0 / (sigma_y * sigma_y)
        Lambda = np.diag(self.base_prior_prec) + inv_s2 * (Xb_w.T @ Xb_w)
        L_c = np.linalg.cholesky(Lambda)
        rhs = inv_s2 * (Xb_w.T @ m0_w)
        delta_hat = np.linalg.solve(L_c.T, np.linalg.solve(L_c, rhs))
        b_b_hat = self.base_prior_mean + delta_hat

        r = m_vec - self.Xb_tr @ b_b_hat
        e = _whiten_ar1(r, rho, sq_1r2)
        return b_b_hat, L_c, Lambda, Xb_w, r, e, delta_hat, one_m_r2, sq_1r2

    def sample_b_b(self, u: np.ndarray, rng: np.random.Generator, C_sum_tr: np.ndarray | None = None) -> np.ndarray:
        """Draw exact conditional posterior sample b_b ~ p(b_b | u, y)."""
        p = self.unpack(u)
        if C_sum_tr is None:
            C_sum_tr = np.sum(self.compute_media(p)[self.train_mask], axis=1)
        b_b_hat, L_c, _, _, _, _, _, _, _ = self._cond_b_b_posterior(C_sum_tr, p["rho"], p["sigma_y"])
        z = rng.normal(0.0, 1.0, size=self.kb)
        return b_b_hat + np.linalg.solve(L_c.T, z)

    def forward(
        self,
        u: np.ndarray,
        b_b: np.ndarray | None = None,
        rng_bb: np.random.Generator | None = None,
    ) -> dict:
        p = self.unpack(u)
        C_mat = self.compute_media(p)
        if b_b is None:
            C_sum_tr = np.sum(C_mat[self.train_mask], axis=1)
            b_b_hat, L_c, _, _, _, _, _, _, _ = self._cond_b_b_posterior(C_sum_tr, p["rho"], p["sigma_y"])
            if rng_bb is not None:
                z = rng_bb.normal(0.0, 1.0, size=self.kb)
                b_b = b_b_hat + np.linalg.solve(L_c.T, z)
            else:
                b_b = b_b_hat
        p["b_b"] = b_b
        y_hat = self.X_base @ b_b + np.sum(C_mat, axis=1)
        iroas_geo = self.eval_geo_lift(p)
        return {
            "params": p,
            "C_mat": C_mat,
            "y_hat": y_hat,
            "resid": self.y - y_hat,
            "iroas_geo": iroas_geo,
        }

    def _geo_lift_and_grad(self, p: dict, need_grad: bool = False):
        """Exact 20%-geo-dark counterfactual lift (HG-5) + optional analytical likelihood gradients."""
        a0 = p["alphas"][0]
        r0_0 = p["r0"][0]
        kap_0 = p["kappa"][0]
        sl_0 = p["slopes"][0]
        ev = self.eval_idx
        ad_act, dad_act = adstock_norm_and_grad(self.s0_act, a0, self.init_vals[0])
        ad_dark, dad_dark = adstock_norm_and_grad(self.s0_dark, a0, self.init_vals[0])
        ca, dr0_a, dkap_a, _, dad_a = contrib_hill_and_grads(ad_act[ev], r0_0, kap_0, sl_0, self.tilde_s[0])
        cd, dr0_d, dkap_d, _, dad_d = contrib_hill_and_grads(ad_dark[ev], r0_0, kap_0, sl_0, self.tilde_s[0])

        scale_w = self.holdout_share / self.meta_withheld
        iroas = float(scale_w * np.sum(ca - cd))
        if not need_grad:
            return iroas

        lift_target = float(self.lift_info["measured_iroas"])
        lift_se = float(self.lift_info["se"])
        z_l = (iroas - lift_target) / lift_se
        lp_lift = -0.5 * z_l * z_l
        d_lp_diroas = -z_l / lift_se

        dl_ur0 = d_lp_diroas * scale_w * float(np.sum(dr0_a - dr0_d)) * p["sig_r0"][0]
        dl_ukap = d_lp_diroas * scale_w * float(np.sum(dkap_a - dkap_d))
        dl_ua = d_lp_diroas * scale_w * float(np.sum(dad_a * dad_act[ev] - dad_d * dad_dark[ev])) * (a0 * (1.0 - a0))
        return iroas, lp_lift, dl_ur0, dl_ukap, dl_ua

    def eval_geo_lift(self, p: dict) -> float:
        """Exact 20%-geo-dark counterfactual lift (HG-5)."""
        return self._geo_lift_and_grad(p, need_grad=False)

    def logp_and_grad(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        """Exact Rao-Blackwellized marginal log-posterior log p(u | y, lift) and analytical gradient."""
        p = self.unpack(u)
        r0, sig_r0 = p["r0"], p["sig_r0"]
        alphas, rho, sigma_y, sig_sy = p["alphas"], p["rho"], p["sigma_y"], p["sig_sy"]

        C_mat, dC_dr0, dC_dukap, dC_dad, dad_da, dad_dk, dC_dsl = self.compute_media(p, need_grad=True)

        C_sum_tr = np.sum(C_mat[self.train_mask], axis=1)
        _, L_c, Lambda, Xb_w, r, e, delta_hat, one_m_r2, sq_1r2 = self._cond_b_b_posterior(
            C_sum_tr, rho, sigma_y
        )
        n = self.n_tr
        inv_s2 = 1.0 / (sigma_y * sigma_y)
        sum_e2 = float(np.dot(e, e))

        # Exact marginalized Gaussian AR(1) log-likelihood integrating out b_b
        ll = (
            -0.5 * n * np.log(2.0 * np.pi)
            - n * np.log(sigma_y)
            + 0.5 * np.log(one_m_r2)
            + 0.5 * float(np.sum(np.log(self.base_prior_prec)))
            - float(np.sum(np.log(np.diag(L_c))))
            - 0.5 * inv_s2 * sum_e2
            - 0.5 * float(np.sum(self.base_prior_prec * (delta_hat ** 2)))
        )

        # By Envelope Theorem (d ll_marg / d b_b_hat == 0), d(ll)/d(C_sum_tr) is unchanged!
        d_ll_dyhat = np.empty(n, dtype=float)
        d_ll_dyhat[0] = inv_s2 * (r[0] - rho * r[1])
        d_ll_dyhat[1:-1] = inv_s2 * (e[1:-1] - rho * e[2:])
        d_ll_dyhat[-1] = inv_s2 * e[-1]

        g_ur0 = np.sum(dC_dr0[self.train_mask] * d_ll_dyhat[:, None], axis=0) * sig_r0
        g_ukap = np.sum(dC_dukap[self.train_mask] * d_ll_dyhat[:, None], axis=0)
        da_du = alphas * (1.0 - alphas)
        g_ualpha = np.sum(dC_dad[self.train_mask] * dad_da[self.train_mask] * d_ll_dyhat[:, None], axis=0) * da_du

        # Exact marginalized rho and sigma_y gradients (including log|Lambda| trace terms)
        V_b = np.linalg.inv(Lambda)
        df_eff = self.kb - float(np.sum(np.diag(V_b) * self.base_prior_prec))
        d_ll_dsigma = -(n - df_eff) / sigma_y + sum_e2 / (sigma_y ** 3)
        g_usig = d_ll_dsigma * sig_sy

        dXb_w_drho = np.empty_like(self.Xb_tr)
        dXb_w_drho[0] = (-rho / sq_1r2) * self.Xb_tr[0]
        dXb_w_drho[1:] = -self.Xb_tr[:-1]
        d_ll_drho = (
            -rho / one_m_r2
            + inv_s2 * (rho * r[0] * r[0] + float(np.dot(e[1:], r[:-1])))
            - inv_s2 * float(np.sum((Xb_w @ V_b) * dXb_w_drho))
        )
        sig_r = p["sig_r"]
        g_urho = d_ll_drho * 1.15 * sig_r * (1.0 - sig_r)

        # Generative Priors + Jacobians
        rate_r0 = R0_GAMMA_SHAPE / self.r0_sds
        lp_r0 = float(np.sum((R0_GAMMA_SHAPE - 1.0) * np.log(r0) - rate_r0 * r0 + np.log(sig_r0)))
        g_ur0 += ((R0_GAMMA_SHAPE - 1.0) / r0 - rate_r0) * sig_r0 + (1.0 - sig_r0)

        dk = (p["u_kap"] - KAPPA_LOG_MEAN) / KAPPA_LOG_SD
        lp_kap = -0.5 * float(np.sum(dk * dk))
        g_ukap += -dk / KAPPA_LOG_SD

        lp_alpha, dg_alpha = _beta_logistic_lp_grad(alphas, ALPHA_BETA_PRIORS[:, 0], ALPHA_BETA_PRIORS[:, 1])
        g_ualpha += dg_alpha

        lp_rho, dg_rho = _beta_logistic_lp_grad(sig_r, 2.0, 2.0)
        g_urho += float(dg_rho)

        sy_sh = sigma_y - 0.25
        lp_sig = -0.5 * (sy_sh / 3.0) ** 2 + np.log(sig_sy)
        g_usig += -(sy_sh / 9.0) * sig_sy + (1.0 - sig_sy)

        lp_extra = 0.0
        g_extra_parts = []
        if self.use_weibull_uf:
            sig_w = p["sig_w"]
            lp_w, g_uw = _beta_logistic_lp_grad(sig_w, 1.5, 4.5)
            for pos, j in enumerate(WEIBULL_CH_IDX):
                dk_du = 1.40 * sig_w[pos] * (1.0 - sig_w[pos])
                g_uw[pos] += float(np.sum(dC_dad[self.train_mask, j] * dad_dk[self.train_mask, j] * d_ll_dyhat)) * dk_du
            lp_extra += lp_w
            g_extra_parts.append(g_uw)

        if self.n_extra_s > 0:
            sig_s = p["sig_s"]
            b_s, width_s, ch_s = (2.0, 2.40, range(6)) if self.unconstrained_hill_test else (4.0, 0.45, BOUNDED_HILL_IDX)
            lp_s, g_us = _beta_logistic_lp_grad(sig_s, 2.0, b_s)
            for pos, j in enumerate(ch_s):
                ds_du = width_s * sig_s[pos] * (1.0 - sig_s[pos])
                g_us[pos] += float(np.sum(dC_dsl[self.train_mask, j] * d_ll_dyhat)) * ds_du
            lp_extra += lp_s
            g_extra_parts.append(g_us)

        # Exact 20%-Geo-Dark Lift Calibration Likelihood (HG-5)
        lp_lift = 0.0
        if self.use_lift:
            _, lp_lift, dl_ur0, dl_ukap, dl_ua = self._geo_lift_and_grad(p, need_grad=True)
            g_ur0[0] += dl_ur0
            g_ukap[0] += dl_ukap
            g_ualpha[0] += dl_ua

        logp = ll + lp_r0 + lp_kap + lp_alpha + lp_rho + lp_sig + lp_extra + lp_lift
        grad = np.concatenate([g_ur0, g_ukap, g_ualpha, [g_urho, g_usig]] + g_extra_parts)
        return float(logp), grad

    def default_init(self) -> np.ndarray:
        u0 = np.zeros(self.dim, dtype=float)
        r0_init = np.array([1.8, 0.9, 1.8, 1.8, 0.8, 1.2])
        u0[0:6] = np.log(np.expm1(r0_init))
        u0[6:12] = KAPPA_LOG_MEAN
        a_means = ALPHA_BETA_PRIORS[:, 0] / np.sum(ALPHA_BETA_PRIORS, axis=1)
        u0[12:18] = np.log(a_means / (1.0 - a_means))
        u0[18] = 0.0  # rho ~ 0.225
        u0[19] = float(np.log(np.expm1(1.75)))  # sigma_y ~ 2.0
        return u0


# ---------------------------------------------------------------------------
# 5. Multi-Chain Preconditioned HMC Sampler & Convergence Diagnostics (B6.2)
# ---------------------------------------------------------------------------
def compute_map_and_whitener(model: BayesianMMMModel, seed: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Find strict MAP estimate (asserting res.success) and build Hessian-whitening metric L."""
    rng = np.random.default_rng(seed)
    u0 = model.default_init()

    def neg_obj(u):
        lp, g = model.logp_and_grad(u)
        return -lp, -g

    best_u = None
    best_nll = np.inf
    for attempt in range(4):
        start = u0 if attempt == 0 else (u0 + rng.normal(0.0, 0.12, size=model.dim))
        res = minimize(neg_obj, start, method="L-BFGS-B", jac=True, options={"gtol": 1e-6, "ftol": 1e-11, "maxiter": 900})
        if res.success and res.fun < best_nll:
            best_nll = float(res.fun)
            best_u = res.x.copy()
    if best_u is None:
        raise RuntimeError("MAP optimization failed to converge cleanly after 4 multi-starts.")

    # Central finite-difference Hessian of -logp using analytical gradients (O(dim) calls -> ~10ms)
    D = model.dim
    H = np.zeros((D, D), dtype=float)
    eps = 1e-4
    for i in range(D):
        step = np.zeros(D)
        step[i] = eps
        _, g_p = model.logp_and_grad(best_u + step)
        _, g_m = model.logp_and_grad(best_u - step)
        H[:, i] = -(g_p - g_m) / (2.0 * eps)
    H = 0.5 * (H + H.T)
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals_safe = np.maximum(eigvals, 0.50)
    L = eigvecs @ np.diag(1.0 / np.sqrt(eigvals_safe))
    return best_u, L


def _run_single_hmc_chain(args):
    """
    Preconditioned No-U-Turn Sampler (NUTS, Hoffman & Gelman 2014) with Stan-style
    dual-averaging step-size adaptation and regularized warmup covariance adaptation.
    """
    model, u_map, L_init, n_warmup, n_samples, chain_seed = args
    rng = np.random.default_rng(int(chain_seed))
    D = model.dim
    L_cur = L_init.copy()
    u_ctr = u_map.copy()
    z = rng.normal(0.0, 0.20, size=D)

    def z_nlp_grad(z_vec, L_mat, ctr):
        u = ctr + L_mat @ z_vec
        if np.max(np.abs(u - ctr)) > 35.0:
            return np.inf, np.zeros_like(z_vec)
        lp, g_u = model.logp_and_grad(u)
        return -lp, -(L_mat.T @ g_u)

    def leapfrog(z_v, p_v, g_v, step_e, L_mat, ctr):
        p_h = p_v - 0.5 * step_e * g_v
        z_n = z_v + step_e * p_h
        nlp_n, g_n = z_nlp_grad(z_n, L_mat, ctr)
        if not np.isfinite(nlp_n) or not np.all(np.isfinite(g_n)):
            return z_n, p_h, np.inf, np.zeros_like(z_v)
        p_n = p_h - 0.5 * step_e * g_n
        return z_n, p_n, nlp_n, g_n

    # Initialize step size
    eps = 0.25
    nlp0, g0 = z_nlp_grad(z, L_cur, u_ctr)
    p_init = rng.normal(0.0, 1.0, size=D)
    for _ in range(15):
        _, p_t, nlp_t, _ = leapfrog(z, p_init, g0, eps, L_cur, u_ctr)
        if not np.isfinite(nlp_t):
            eps *= 0.5
            continue
        dH = (nlp_t + 0.5 * float(np.dot(p_t, p_t))) - (nlp0 + 0.5 * float(np.dot(p_init, p_init)))
        if dH > 0.7:
            eps *= 0.55
        elif dH < 0.05:
            eps *= 1.5
        else:
            break

    mu = np.log(2.0 * eps)
    eps_bar = eps
    H_bar = 0.0
    gamma, t0, kappa_da = 0.05, 10.0, 0.75
    delta_target = 0.90
    max_depth = 6

    def build_tree(z_t, p_t, g_t, log_u, v_dir, j_depth, step_e, H0, L_mat, ctr):
        if j_depth == 0:
            z_p, p_p, nlp_p, g_p = leapfrog(z_t, p_t, g_t, v_dir * step_e, L_mat, ctr)
            if not np.isfinite(nlp_p):
                return z_p, p_p, g_p, z_p, p_p, g_p, z_p, nlp_p, g_p, 0, 0, 0.0, 1, True
            H_p = nlp_p + 0.5 * float(np.dot(p_p, p_p))
            dH = H_p - H0
            div = bool(dH > 1000.0)
            n_p = int(log_u <= -H_p) if not div else 0
            s_p = int(dH < 1000.0)
            acc_p = min(1.0, float(np.exp(-max(dH, -50.0)))) if not div else 0.0
            return z_p, p_p, g_p, z_p, p_p, g_p, z_p, nlp_p, g_p, n_p, s_p, acc_p, 1, div
        else:
            zm, pm, gm, zp, pp, gp, z1, nlp1, g1, n1, s1, a1, na1, div1 = build_tree(
                z_t, p_t, g_t, log_u, v_dir, j_depth - 1, step_e, H0, L_mat, ctr
            )
            if s1 == 1 and not div1:
                z_side, p_side, g_side = (zm, pm, gm) if v_dir == -1 else (zp, pp, gp)
                sub = build_tree(z_side, p_side, g_side, log_u, v_dir, j_depth - 1, step_e, H0, L_mat, ctr)
                if v_dir == -1:
                    zm, pm, gm = sub[0:3]
                else:
                    zp, pp, gp = sub[3:6]
                z2, nlp2, g2, n2, s2, a2, na2, div2 = sub[6:]
                if n1 + n2 > 0 and rng.uniform() < (n2 / (n1 + n2)):
                    z1, nlp1, g1 = z2, nlp2, g2
                a1 += a2
                na1 += na2
                div1 = div1 or div2
                dz = zp - zm
                uturn = (np.dot(dz, pm) < 0.0) or (np.dot(dz, pp) < 0.0)
                s1 = int(s2 == 1 and (not uturn) and (not div1))
                n1 += n2
            return zm, pm, gm, zp, pp, gp, z1, nlp1, g1, n1, s1, a1, na1, div1

    w1_end = int(0.18 * n_warmup)
    w2_end = int(0.75 * n_warmup)
    warm_u_buf = []

    samples_u = np.empty((n_samples, D), dtype=float)
    energies = np.empty(n_samples, dtype=float)
    n_divergent = 0
    accept_probs = []

    nlp_cur, g_cur = z_nlp_grad(z, L_cur, u_ctr)
    total_iters = n_warmup + n_samples
    for it in range(total_iters):
        p0 = rng.normal(0.0, 1.0, size=D)
        H0 = nlp_cur + 0.5 * float(np.dot(p0, p0))
        log_u = float(np.log(rng.uniform()) - H0)

        zm, pm, gm = z.copy(), p0.copy(), g_cur.copy()
        zp, pp, gp = z.copy(), p0.copy(), g_cur.copy()
        z_prop, nlp_prop, g_prop = z.copy(), nlp_cur, g_cur.copy()
        n_valid = 1
        alpha_sum, n_alpha = 0.0, 0
        iter_div = False

        for j_d in range(max_depth):
            v_dir = -1 if rng.uniform() < 0.5 else 1
            z_side, p_side, g_side = (zm, pm, gm) if v_dir == -1 else (zp, pp, gp)
            sub = build_tree(z_side, p_side, g_side, log_u, v_dir, j_d, eps, H0, L_cur, u_ctr)
            if v_dir == -1:
                zm, pm, gm = sub[0:3]
            else:
                zp, pp, gp = sub[3:6]
            z2, nlp2, g2, n2, s2, a2, na2, div2 = sub[6:]
            alpha_sum += a2
            n_alpha += na2
            if div2:
                iter_div = True
                break
            if s2 == 1 and n2 > 0 and rng.uniform() < min(1.0, n2 / n_valid):
                z_prop, nlp_prop, g_prop = z2, nlp2, g2
            n_valid += n2
            dz = zp - zm
            if s2 == 0 or (np.dot(dz, pm) < 0.0) or (np.dot(dz, pp) < 0.0):
                break

        z, nlp_cur, g_cur = z_prop, nlp_prop, g_prop
        u_state = u_ctr + L_cur @ z
        mean_a = alpha_sum / max(n_alpha, 1)

        if it < n_warmup:
            if w1_end <= it < w2_end:
                warm_u_buf.append(z.copy())
            if it == w2_end - 1 and len(warm_u_buf) >= 40:
                Z_w = np.asarray(warm_u_buf)
                # Diagonal scale refinement in whitened Hessian basis (avoids 34x34 sample-cov noise!)
                z_mean = np.mean(Z_w, axis=0)
                z_scales = np.clip(np.std(Z_w, axis=0), 0.60, 1.80)
                u_new_ctr = u_ctr + L_cur @ z_mean
                L_new = L_cur * z_scales[None, :]
                z = (z - z_mean) / z_scales
                L_cur = L_new
                u_ctr = u_new_ctr
                nlp_cur, g_cur = z_nlp_grad(z, L_cur, u_ctr)
                H_bar = 0.0
                eps_bar = eps
                mu = np.log(2.0 * eps)
            else:
                m = (it - w2_end + 1) if it >= w2_end else (it + 1)
                w_m = 1.0 / (m + t0)
                H_bar = (1.0 - w_m) * H_bar + w_m * (delta_target - mean_a)
                log_eps = mu - (np.sqrt(m) / gamma) * H_bar
                eps = float(np.clip(np.exp(log_eps), 1e-4, 1.0))
                m_k = m ** (-kappa_da)
                eps_bar = float(np.exp(m_k * log_eps + (1.0 - m_k) * np.log(eps_bar)))
                if it == n_warmup - 1:
                    eps = eps_bar
        else:
            idx_s = it - n_warmup
            if iter_div:
                n_divergent += 1
            accept_probs.append(mean_a)
            samples_u[idx_s] = u_state
            energies[idx_s] = nlp_cur + 0.5 * float(np.dot(p0, p0))

    return samples_u, energies, n_divergent, float(np.mean(accept_probs)), eps


def compute_rhat_ess(chains_arr: np.ndarray) -> tuple[float, float, float]:
    """
    Compute split-Rhat, Bulk ESS, and Tail ESS (min of 5% and 95% quantile ESS)
    for array of shape (C, N).
    """
    C, N = chains_arr.shape
    half = N // 2
    split = np.vstack([chains_arr[:, :half], chains_arr[:, half : 2 * half]])  # (2C, half)
    m, n = split.shape
    chain_means = np.mean(split, axis=1)
    B = n * np.var(chain_means, ddof=1)
    W = np.mean(np.var(split, axis=1, ddof=1))
    var_hat = ((n - 1.0) / n) * W + (1.0 / n) * B
    rhat = float(np.sqrt(var_hat / max(W, 1e-18)))

    def _ess_1d(arr2d):
        m_, n_ = arr2d.shape
        acfs = []
        for c_i in range(m_):
            x = arr2d[c_i] - np.mean(arr2d[c_i])
            v = np.dot(x, x)
            if v <= 1e-18:
                acfs.append(np.zeros(n_))
                continue
            f = np.fft.rfft(x, n=2 * n_)
            acf = np.fft.irfft(f * np.conjugate(f))[:n_] / v
            acfs.append(acf)
        mean_acf = np.mean(acfs, axis=0)
        # Geyer positive pair truncation
        tau = 1.0
        for k in range(1, n_ - 1, 2):
            pair = mean_acf[k] + mean_acf[k + 1]
            if pair <= 0.0:
                break
            tau += 2.0 * pair
        return float((m_ * n_) / max(tau, 1.0))

    # Rank-normalized bulk ESS (approximate via z-scores across pooled ranks)
    ranks = stats.rankdata(split.ravel())
    z_bulk = stats.norm.ppf((ranks - 0.375) / (m * n + 0.25)).reshape(m, n)
    ess_bulk = _ess_1d(z_bulk)

    q05 = np.quantile(split, 0.05)
    q95 = np.quantile(split, 0.95)
    ess_tail = min(_ess_1d((split <= q05).astype(float)), _ess_1d((split >= q95).astype(float)))
    return rhat, ess_bulk, ess_tail


def sample_posterior_hmc(
    model: BayesianMMMModel,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_samples: int = 750,
    seed: int = 42,
    workers: int = 4,
) -> dict:
    """Run multi-chain preconditioned HMC and compute full convergence diagnostics."""
    u_map, L = compute_map_and_whitener(model, seed=seed)
    tasks = [(model, u_map, L, n_warmup, n_samples, seed + 100 * c) for c in range(n_chains)]
    n_w = min(workers, n_chains)
    if n_w > 1:
        with ProcessPoolExecutor(max_workers=n_w, mp_context=mp.get_context("spawn")) as ex:
            chain_out = list(ex.map(_run_single_hmc_chain, tasks))
    else:
        chain_out = [_run_single_hmc_chain(t) for t in tasks]

    chains_u = np.stack([c[0] for c in chain_out], axis=0)  # (C, N, D)
    energies = np.stack([c[1] for c in chain_out], axis=0)  # (C, N)
    total_div = int(sum(c[2] for c in chain_out))
    mean_acc = float(np.mean([c[3] for c in chain_out]))

    # E-BFMI check per chain
    ebfmis = []
    for c in range(n_chains):
        e_c = energies[c]
        ebfmis.append(float(np.sum(np.diff(e_c) ** 2) / (np.sum((e_c - e_c.mean()) ** 2) + 1e-12)))

    rhats, ess_bulks, ess_tails = [], [], []
    for d in range(model.dim):
        rh, eb, et = compute_rhat_ess(chains_u[:, :, d])
        rhats.append(rh)
        ess_bulks.append(eb)
        ess_tails.append(et)

    flat_u = chains_u.reshape(-1, model.dim)
    return {
        "u_map": u_map,
        "chains_u": chains_u,
        "draws_u": flat_u,
        "diagnostics": {
            "n_chains": int(n_chains),
            "n_warmup": int(n_warmup),
            "n_samples_per_chain": int(n_samples),
            "total_draws": int(len(flat_u)),
            "max_rhat": round(float(max(rhats)), 4),
            "min_ess_bulk": round(float(min(ess_bulks)), 1),
            "min_ess_tail": round(float(min(ess_tails)), 1),
            "n_divergent": total_div,
            "mean_accept_prob": round(mean_acc, 3),
            "min_ebfmi": round(float(min(ebfmis)), 3),
        },
    }


def extract_draws_summary(
    model: BayesianMMMModel,
    draws_u: np.ndarray,
    mask_2025: np.ndarray,
) -> dict:
    """Compute exact posterior parameter arrays, ROAS, 2025 mROAS, and geo-dark lift across draws."""
    D = len(draws_u)
    kb = model.kb
    b_b = np.empty((D, kb), dtype=float)
    r0 = np.empty((D, 6), dtype=float)
    kappa = np.empty((D, 6), dtype=float)
    Ks = np.empty((D, 6), dtype=float)
    alphas = np.empty((D, 6), dtype=float)
    slopes = np.empty((D, 6), dtype=float)
    k_lags = np.empty((D, 6), dtype=float)
    rho = np.empty(D, dtype=float)
    sigma_y = np.empty(D, dtype=float)

    avg_roas_3yr = np.empty((D, 6), dtype=float)
    roas_2025 = np.empty((D, 6), dtype=float)
    mroas_2025 = np.empty((D, 6), dtype=float)
    iroas_meta = np.empty(D, dtype=float)
    y_hat_all = np.empty((D, len(model.y)), dtype=float)

    S = model.S
    tot_spend = S.sum(axis=0)
    spend_25 = S[mask_2025].sum(axis=0)
    S_p = S.copy()
    S_p[mask_2025] *= 1.01
    lag_mats_p = (
        {j: build_lag_matrix(S_p[:, j], model.init_vals[j], WEIBULL_MAX_LAG) for j in WEIBULL_CH_IDX}
        if model.use_weibull_uf
        else None
    )

    rng_bb = np.random.default_rng(12345)
    for d in range(D):
        fwd = model.forward(draws_u[d], rng_bb=rng_bb)
        p = fwd["params"]
        b_b[d] = p["b_b"]
        r0[d] = p["r0"]
        kappa[d] = p["kappa"]
        Ks[d] = model.tilde_s / p["kappa"]
        alphas[d] = p["alphas"]
        slopes[d] = p["slopes"]
        k_lags[d] = p["k_lags"]
        rho[d] = p["rho"]
        sigma_y[d] = p["sigma_y"]
        iroas_meta[d] = fwd["iroas_geo"]
        y_hat_all[d] = fwd["y_hat"]

        C_mat = fwd["C_mat"]
        avg_roas_3yr[d] = C_mat.sum(axis=0) / tot_spend
        roas_2025[d] = C_mat[mask_2025].sum(axis=0) / spend_25
        C_mat_p = model.compute_media(p, S_override=S_p, lag_mats_override=lag_mats_p)
        mroas_2025[d] = (C_mat_p.sum(axis=0) - C_mat.sum(axis=0)) / (0.01 * spend_25)

    return {
        "b_b": b_b,
        "r0": r0,
        "kappa": kappa,
        "Ks": Ks,
        "alphas": alphas,
        "slopes": slopes,
        "k_lags": k_lags,
        "rho": rho,
        "sigma_y": sigma_y,
        "avg_roas_3yr": avg_roas_3yr,
        "roas_2025": roas_2025,
        "mroas_2025": mroas_2025,
        "iroas_meta": iroas_meta,
        "y_hat": y_hat_all,
    }


def summarize_draws(arr) -> dict:
    a = np.asarray(arr, dtype=float)
    return {
        "mean": round(float(a.mean()), 3),
        "median": round(float(np.median(a)), 3),
        "sd": round(float(a.std()), 3),
        "ci80": [round(float(np.percentile(a, 10)), 3), round(float(np.percentile(a, 90)), 3)],
        "ci90": [round(float(np.percentile(a, 5)), 3), round(float(np.percentile(a, 95)), 3)],
    }


# ---------------------------------------------------------------------------
# 6. Prior Predictive Check & Prior-to-Posterior Contraction (B4.3, B5.2)
# ---------------------------------------------------------------------------
def run_prior_predictive_and_contraction(
    model: BayesianMMMModel,
    mask_2025: np.ndarray,
    n_prior_draws: int = 500,
    seed: int = 77,
) -> dict:
    rng = np.random.default_rng(seed)
    prior_y = np.empty((n_prior_draws, len(model.y)), dtype=float)
    prior_mroas = np.empty((n_prior_draws, 6), dtype=float)
    prior_iroas_meta = np.empty(n_prior_draws, dtype=float)

    S = model.S
    S_p = S.copy()
    S_p[mask_2025] *= 1.01
    spend_25 = S[mask_2025].sum(axis=0)

    for d in range(n_prior_draws):
        b_b = rng.normal(model.base_prior_mean, model.base_prior_sd)
        r0 = rng.gamma(R0_GAMMA_SHAPE, model.r0_sds / R0_GAMMA_SHAPE)
        kappa = np.exp(rng.normal(KAPPA_LOG_MEAN, KAPPA_LOG_SD, size=6))
        alphas = np.array([rng.beta(ALPHA_BETA_PRIORS[j, 0], ALPHA_BETA_PRIORS[j, 1]) for j in range(6)])
        p = {"b_b": b_b, "r0": r0, "kappa": kappa, "alphas": alphas, "slopes": np.ones(6), "k_lags": np.ones(6)}
        prior_iroas_meta[d] = model.eval_geo_lift(p)

        C_mat = model.compute_media(p)
        C_mat_p = model.compute_media(p, S_override=S_p)
        prior_mroas[d] = (C_mat_p.sum(axis=0) - C_mat.sum(axis=0)) / (0.01 * spend_25)
        prior_y[d] = model.X_base @ b_b + np.sum(C_mat, axis=1)

    return {
        "n_prior_draws": n_prior_draws,
        "prior_y_mean_m_inr": round(float(np.mean(prior_y)), 2),
        "prior_y_ci90_m_inr": [round(float(np.percentile(prior_y, 5)), 2), round(float(np.percentile(prior_y, 95)), 2)],
        "observed_y_range_m_inr": [round(float(model.y.min()), 2), round(float(model.y.max()), 2)],
        "prior_meta_geo_iroas_ci90": [
            round(float(np.percentile(prior_iroas_meta, 5)), 3),
            round(float(np.percentile(prior_iroas_meta, 95)), 3),
        ],
        "prior_mroas_var": np.var(prior_mroas, axis=0),
        "prior_mroas_summary": {ch: summarize_draws(prior_mroas[:, j]) for j, ch in enumerate(CHANNELS)},
    }


# ---------------------------------------------------------------------------
# 7. Leak-Free Expanding-Window CV & Yao-Vehtari Stacking (B6.1, B6.3)
# ---------------------------------------------------------------------------
SPEC_CONFIGS = [
    {"spec_id": "M1_Spline_MM_Base", "use_spline": True, "prior_scale_mult": 1.0, "use_weibull_uf": False, "bounded_hill_uf": False},
    {"spec_id": "M2_Linear_MM_Base", "use_spline": False, "prior_scale_mult": 1.0, "use_weibull_uf": False, "bounded_hill_uf": False},
    {"spec_id": "M3_Spline_MM_Tight", "use_spline": True, "prior_scale_mult": 0.70, "use_weibull_uf": False, "bounded_hill_uf": False},
    {"spec_id": "M4_Spline_MM_Wide", "use_spline": True, "prior_scale_mult": 1.40, "use_weibull_uf": False, "bounded_hill_uf": False},
    {"spec_id": "M5_Spline_Weibull_VideoTV", "use_spline": True, "prior_scale_mult": 1.0, "use_weibull_uf": True, "bounded_hill_uf": False},
    {"spec_id": "M6_Spline_BoundedHill_UF", "use_spline": True, "prior_scale_mult": 1.0, "use_weibull_uf": False, "bounded_hill_uf": True},
]

CV_FOLDS = [
    (78, 91, "Fold1_2024Q3"),
    (91, 104, "Fold2_2024Q4"),
    (104, 130, "Fold3_2025H1"),
    (130, 156, "Fold4_2025H2"),
]


def evaluate_oos_fold(
    df_struct: pd.DataFrame,
    spec_cfg: dict,
    train_end_idx: int,
    test_end_idx: int,
    test_idx: np.ndarray,
    eval_idx: np.ndarray,
    lift_info: dict,
    n_chains: int = 2,
    n_warmup: int = 300,
    n_samples: int = 400,
    workers: int = 2,
    seed: int = 500,
) -> dict:
    """Evaluate strictly fold-isolated OOS conditional & ex-ante forecast accuracy + exact AR(1) ELPD."""
    fold = prepare_fold_data(df_struct, train_end_idx=train_end_idx)
    df_f = fold["df"]
    S = fold["S"]
    y = fold["y"]
    tr_mask = fold["train_mask"]

    X_base, base_names = build_base_design(df_f, fold["ref_means"], use_spline=spec_cfg["use_spline"], K=2)
    use_lift_fold = bool(train_end_idx > int(eval_idx[-1]))

    mod = BayesianMMMModel(
        S=S, y=y, X_base=X_base, base_names=base_names, train_mask=tr_mask,
        init_vals=fold["init_vals"], tilde_s=fold["tilde_s"],
        test_idx=test_idx, eval_idx=eval_idx, lift_info=lift_info,
        use_lift=use_lift_fold, prior_scale_mult=spec_cfg["prior_scale_mult"],
        use_weibull_uf=spec_cfg["use_weibull_uf"], bounded_hill_uf=spec_cfg["bounded_hill_uf"],
    )
    hmc = sample_posterior_hmc(mod, n_chains=n_chains, n_warmup=n_warmup, n_samples=n_samples, seed=seed, workers=workers)
    draws_u = hmc["draws_u"]
    D = len(draws_u)

    # Simulate conditional posterior predictive draws over test weeks (including AR(1) mean + variance)
    te_idx = np.arange(train_end_idx, test_end_idx)
    h_te = len(te_idx)
    y_te = y[te_idx]
    rng = np.random.default_rng(seed + 999)

    y_pred_cond = np.empty((D, h_te), dtype=float)
    y_samp_cond = np.empty((D, h_te), dtype=float)
    elpd_mat = np.empty((D, h_te), dtype=float)
    C_sum_all = np.empty((D, len(df_f)), dtype=float)

    for d in range(D):
        fwd = mod.forward(draws_u[d])
        p = fwd["params"]
        rho_d = p["rho"]
        sig_d = p["sigma_y"]
        yh = fwd["y_hat"]
        C_sum_all[d] = np.sum(fwd["C_mat"], axis=1)
        r_prev = float(y[train_end_idx - 1] - yh[train_end_idx - 1])
        eps_sim = r_prev
        for h_i, t_i in enumerate(te_idx):
            # 1-step-ahead conditional mean & exact AR(1) density for stacking
            r_cur = float(y[t_i] - yh[t_i])
            cond_mean = yh[t_i] + rho_d * r_prev
            elpd_mat[d, h_i] = -0.5 * np.log(2.0 * np.pi * sig_d * sig_d) - 0.5 * ((y[t_i] - cond_mean) / sig_d) ** 2
            r_prev = r_cur

            # Multi-step trajectory mean & full predictive draw (without peeking at y[t_i]!)
            y_pred_cond[d, h_i] = yh[t_i] + (rho_d ** (h_i + 1)) * float(y[train_end_idx - 1] - yh[train_end_idx - 1])
            eps_sim = rho_d * eps_sim + rng.normal(0.0, sig_d)
            y_samp_cond[d, h_i] = yh[t_i] + eps_sim

    elpd_week = logsumexp(elpd_mat, axis=0) - np.log(D)
    y_mean_cond = np.mean(y_pred_cond, axis=0)
    fit_te = compute_fit_metrics(y_te, y_mean_cond)
    pi90_lo = np.percentile(y_samp_cond, 5, axis=0)
    pi90_hi = np.percentile(y_samp_cond, 95, axis=0)
    pi80_lo = np.percentile(y_samp_cond, 10, axis=0)
    pi80_hi = np.percentile(y_samp_cond, 90, axis=0)
    cov90 = float(np.mean((y_te >= pi90_lo) & (y_te <= pi90_hi)) * 100.0)
    cov80 = float(np.mean((y_te >= pi80_lo) & (y_te <= pi80_hi)) * 100.0)

    # Ex-Ante Forecast Accuracy via shared covariate projection + cached media trajectories
    df_fut = project_exante_covariates(
        df_f.iloc[:train_end_idx],
        df_f.loc[te_idx, "t"].values,
        df_f.loc[te_idx, "week_of_year"].values,
    )
    X_exante_te, _ = build_base_design(df_fut, fold["ref_means"], use_spline=spec_cfg["use_spline"], K=2)
    y_exante_mean = np.mean(
        np.array([
            X_exante_te @ mod.sample_b_b(draws_u[d], rng, C_sum_tr=C_sum_all[d, tr_mask]) + C_sum_all[d, te_idx]
            for d in range(D)
        ]),
        axis=0,
    )
    fit_ex = compute_fit_metrics(y_te, y_exante_mean)

    return {
        "train_weeks": int(train_end_idx),
        "test_weeks": int(h_te),
        "test_start": str(df_f.loc[train_end_idx, "week_start"]),
        "test_end": str(df_f.loc[test_end_idx - 1, "week_start"]),
        "used_lift_test": use_lift_fold,
        "oos_r2": round(fit_te["r2"], 4),
        "oos_rmse_m_inr": round(fit_te["rmse"], 3),
        "oos_mape_pct": round(fit_te["mape_pct"], 2),
        "oos_nrmse_pct": round(fit_te["nrmse_pct"], 2),
        "pi90_coverage_pct": round(cov90, 1),
        "pi80_coverage_pct": round(cov80, 1),
        "ex_ante_mape_pct": round(fit_ex["mape_pct"], 2),
        "ex_ante_rmse_m_inr": round(fit_ex["rmse"], 3),
        "elpd_sum": round(float(np.sum(elpd_week)), 2),
        "elpd_per_week": elpd_week,
        "y_pred_mean": y_mean_cond,
        "pi90_lo": pi90_lo,
        "pi90_hi": pi90_hi,
    }


def solve_bayesian_stacking(elpd_matrix: np.ndarray) -> np.ndarray:
    """
    Solve Yao, Vehtari, Simpson & Gelman (2018) stacking optimization:
      max_{w in simplex} sum_t log( sum_k w_k * exp(elpd_{t,k}) )
    """
    T_oos, K = elpd_matrix.shape
    # Shift per week for numerical stability
    row_max = np.max(elpd_matrix, axis=1, keepdims=True)
    p_rel = np.exp(elpd_matrix - row_max)

    def neg_log_score_and_grad(w):
        w_safe = np.maximum(w, 1e-12)
        mix = p_rel @ w_safe
        val = -float(np.sum(np.log(mix)))
        grad = -(p_rel.T @ (1.0 / mix))
        return val, grad

    w0 = np.full(K, 1.0 / K)
    bounds = [(0.0, 1.0)] * K
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(neg_log_score_and_grad, w0, method="SLSQP", jac=True, bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Bayesian stacking optimizer did not converge: {res.message}")
    w_opt = np.maximum(res.x, 0.0)
    w_opt /= np.sum(w_opt)
    return w_opt


# ---------------------------------------------------------------------------
# 8. Main Modeling Pipeline Runner
# ---------------------------------------------------------------------------
def run_modeling_pipeline(
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_samples: int = 750,
    workers: int = 4,
    skip_cv: bool = False,
    clean_bundle: tuple | None = None,
) -> dict:
    d_dir, o_dir = resolve_paths(data_dir, out_dir)
    lift_info = load_lift_test(d_dir)
    if clean_bundle is None:
        df_clean, audit_info, df_raw, df_struct, fold_full = clean_mmm_data(
            data_dir=d_dir, out_dir=o_dir, return_bundle=True
        )
    else:
        df_clean, audit_info, df_raw, df_struct, fold_full = clean_bundle

    print("  1/6 Reproducing v0 baseline (exact SVD) & distortion waterfall...")
    baseline_diag = run_baseline_and_waterfall(df_raw, df_clean)

    print("  2/6 Running causal DAG, mediator falsification & identification audits...")
    causal_audits = run_causal_and_identification_audits(df_clean, fold_full, lift_info)

    S = fold_full["S"]
    y = fold_full["y"]
    init_vals = fold_full["init_vals"]
    tilde_s = fold_full["tilde_s"]
    mask_2025 = df_clean["year"].values == 2025
    test_idx = np.where((df_clean["week_start"] >= lift_info["test_start"]) & (df_clean["week_start"] <= lift_info["test_end"]))[0]
    eval_idx = np.where((df_clean["week_start"] >= lift_info["test_start"]) & (df_clean["week_start"] <= lift_info["eval_end"]))[0]

    X_base_sp, base_names_sp = build_base_design(df_clean, fold_full["ref_means"], use_spline=True, K=2)
    X_base_lin, base_names_lin = build_base_design(df_clean, fold_full["ref_means"], use_spline=False, K=2)

    base_model_kwargs = {
        "S": S,
        "y": y,
        "X_base": X_base_sp,
        "base_names": base_names_sp,
        "train_mask": fold_full["train_mask"],
        "init_vals": init_vals,
        "tilde_s": tilde_s,
        "test_idx": test_idx,
        "eval_idx": eval_idx,
        "lift_info": lift_info,
        "use_lift": True,
    }

    def make_model(**overrides) -> BayesianMMMModel:
        return BayesianMMMModel(**{**base_model_kwargs, **overrides})

    def run_map_sensitivity(seed: int, summarize: bool = True, **overrides):
        m = make_model(**overrides)
        u_map, _ = compute_map_and_whitener(m, seed=seed)
        return (m, u_map, extract_draws_summary(m, u_map[None, :], mask_2025)) if summarize else (m, u_map, m.forward(u_map))

    # Prior predictive check on primary M1 model
    m1_full = make_model()
    prior_check = run_prior_predictive_and_contraction(m1_full, mask_2025)

    # Stage 1 Out-of-Experiment Concordance (Uncalibrated M1)
    print("  3/6 Running Stage-1 out-of-experiment Meta lift concordance & dead-end demonstrations...")
    m1_uncal = make_model(use_lift=False)
    hmc_uncal = sample_posterior_hmc(m1_uncal, n_chains=2, n_warmup=350, n_samples=500, seed=11, workers=min(2, workers))
    sum_uncal = extract_draws_summary(m1_uncal, hmc_uncal["draws_u"], mask_2025)
    uncal_iroas = sum_uncal["iroas_meta"]
    z_conc = (np.median(uncal_iroas) - lift_info["measured_iroas"]) / np.sqrt(lift_info["se"] ** 2 + np.var(uncal_iroas))
    conc_pval = float(2.0 * (1.0 - stats.norm.cdf(abs(z_conc))))

    # Dead-end demonstration #1: raw organic_traffic_index bad-control bias
    org_std = (df_clean["organic_traffic_index"].values - df_clean["organic_traffic_index"].mean()) / df_clean["organic_traffic_index"].std()
    _, _, fwd_bad_org = run_map_sensitivity(
        12, summarize=False,
        X_base=np.column_stack([X_base_sp, org_std]),
        base_names=base_names_sp + ["organic_traffic_std"],
        use_lift=False,
    )

    # Dead-end demonstration #2: Unconstrained Hill (s in [0.6, 3.0]) -> Rejected by Gate HG-4!
    _, _, sum_rej_hill = run_map_sensitivity(13, unconstrained_hill_test=True)
    gs_m_rej = float(sum_rej_hill["mroas_2025"][0, 1])
    gs_a_rej = float(sum_rej_hill["avg_roas_3yr"][0, 1])

    # TV 6-flight LOFO, Spring vs Autumn & Net-Cost Rebate Sensitivity (MAP fast check + M1)
    print("  4/6 Running TV 6-flight LOFO, Regime A2 lift, and TV net-cost sensitivity...")
    tv_active = np.where(df_clean["tv_spend"].values > 0)[0]
    flights, st, pr = [], tv_active[0], tv_active[0]
    for ix in tv_active[1:]:
        if ix > pr + 1:
            flights.append((int(st), int(pr)))
            st = ix
        pr = ix
    flights.append((int(st), int(pr)))
    lofo_res = []
    for f_id, (s_i, e_i) in enumerate(flights, start=1):
        tr_m = np.ones(len(df_clean), dtype=bool)
        tr_m[s_i : min(len(df_clean), e_i + 5)] = False
        _, _, sf = run_map_sensitivity(20 + f_id, train_mask=tr_m)
        lofo_res.append({
            "flight_id": f_id,
            "start_week": str(df_clean.loc[s_i, "week_start"]),
            "end_week": str(df_clean.loc[e_i, "week_start"]),
            "has_festive_overlap": bool(df_clean.loc[s_i:e_i, "festive_flag"].sum() > 0),
            "tv_avg_roas_3yr": round(float(sf["avg_roas_3yr"][0, 4]), 3),
            "tv_mroas_2025": round(float(sf["mroas_2025"][0, 4]), 3),
        })

    # No-festive TV, Net-cost rebate, and Regime A2 sensitivities
    ctrls_nofest = [c for c in CONTROLS if c != "festive_flag"]
    X_nf, bn_nf = build_base_design(df_clean, fold_full["ref_means"], use_spline=True, K=2, controls=ctrls_nofest)
    _, _, sf_nf = run_map_sensitivity(31, X_base=X_nf, base_names=bn_nf)

    fold_net = prepare_fold_data(df_struct, train_end_idx=len(df_struct), use_tv_net_cost=True)
    _, _, sf_tvnet = run_map_sensitivity(
        32, S=fold_net["S"], train_mask=fold_net["train_mask"], init_vals=fold_net["init_vals"], tilde_s=fold_net["tilde_s"]
    )
    _, _, sf_a2 = run_map_sensitivity(33, regime_a2_reallocated_lift=True)

    # Fit all 6 Admissible Candidate Models via Multi-Chain HMC
    print(f"  5/6 Sampling 6 admissible Bayesian specs ({n_chains} chains x {n_samples} draws each)...")
    spec_results = []
    admissible_specs = []
    for s_idx, cfg in enumerate(SPEC_CONFIGS):
        Xb = X_base_sp if cfg["use_spline"] else X_base_lin
        bn = base_names_sp if cfg["use_spline"] else base_names_lin
        mod = make_model(
            X_base=Xb, base_names=bn, prior_scale_mult=cfg["prior_scale_mult"],
            use_weibull_uf=cfg["use_weibull_uf"], bounded_hill_uf=cfg["bounded_hill_uf"],
        )
        hmc = sample_posterior_hmc(mod, n_chains=n_chains, n_warmup=n_warmup, n_samples=n_samples, seed=100 + s_idx * 50, workers=workers)
        ext = extract_draws_summary(mod, hmc["draws_u"], mask_2025)
        fit_in = compute_fit_metrics(y, np.median(ext["y_hat"], axis=0))

        # Admissibility Gates: convergence + concavity (mROAS <= aROAS on always-on channels)
        diag = hmc["diagnostics"]
        concavity_ok = all(
            np.median(ext["mroas_2025"][:, j]) <= np.median(ext["avg_roas_3yr"][:, j]) + 1e-6
            for j in (0, 1, 5)
        )
        admissible = (diag["max_rhat"] <= 1.03) and (diag["n_divergent"] == 0) and concavity_ok

        spec_entry = {
            "spec_id": cfg["spec_id"],
            "config": cfg,
            "admissible": admissible,
            "diagnostics": diag,
            "in_sample": {"r2": round(fit_in["r2"], 4), "mape_pct": round(fit_in["mape_pct"], 2), "dw": round(fit_in["dw"], 3)},
            "meta_geo_iroas": summarize_draws(ext["iroas_meta"]),
            "avg_roas_3yr_median": {ch: round(float(np.median(ext["avg_roas_3yr"][:, j])), 3) for j, ch in enumerate(CHANNELS)},
            "mroas_2025_median": {ch: round(float(np.median(ext["mroas_2025"][:, j])), 3) for j, ch in enumerate(CHANNELS)},
            "ext": ext,
        }
        spec_results.append(spec_entry)
        if admissible:
            admissible_specs.append(spec_entry)

    # Expanding-window leak-free CV & Yao-Vehtari Stacking Weights
    print("  6/6 Running leak-free expanding-window CV & solving Yao-Vehtari Bayesian stacking weights...")
    cv_summary = {}
    if not skip_cv:
        elpd_cols = []
        for s_idx, sp in enumerate(admissible_specs):
            cfg = sp["config"]
            fold_res_list = []
            elpd_weeks_all = []
            for f_i, (tr_e, te_e, lbl) in enumerate(CV_FOLDS):
                f_out = evaluate_oos_fold(
                    df_struct, cfg, tr_e, te_e, test_idx, eval_idx, lift_info,
                    n_chains=2, n_warmup=200, n_samples=300, workers=min(2, workers), seed=2000 + s_idx * 100 + f_i * 10
                )
                elpd_weeks_all.append(f_out.pop("elpd_per_week"))
                f_out.pop("y_pred_mean")
                f_out.pop("pi90_lo")
                f_out.pop("pi90_hi")
                f_out["fold"] = lbl
                fold_res_list.append(f_out)
            elpd_cols.append(np.concatenate(elpd_weeks_all))
            cv_summary[sp["spec_id"]] = {
                "mean_oos_r2": round(float(np.mean([f["oos_r2"] for f in fold_res_list])), 4),
                "mean_oos_mape_pct": round(float(np.mean([f["oos_mape_pct"] for f in fold_res_list])), 2),
                "mean_ex_ante_mape_pct": round(float(np.mean([f["ex_ante_mape_pct"] for f in fold_res_list])), 2),
                "mean_pi90_coverage_pct": round(float(np.mean([f["pi90_coverage_pct"] for f in fold_res_list])), 1),
                "total_elpd": round(float(np.sum(elpd_cols[-1])), 2),
                "folds": fold_res_list,
            }
        elpd_mat = np.column_stack(elpd_cols)  # (78, K_adm)
        stack_weights = solve_bayesian_stacking(elpd_mat)

        # Also evaluate primary M1 on strict 51-week 2025 Holdout (105 -> 156) for chart_05
        holdout_2025 = evaluate_oos_fold(
            df_struct, SPEC_CONFIGS[0], int(np.where(mask_2025)[0][0]), len(df_clean),
            test_idx, eval_idx, lift_info, n_chains=2, n_warmup=300, n_samples=450, workers=min(2, workers), seed=909
        )
    else:
        stack_weights = np.full(len(admissible_specs), 1.0 / len(admissible_specs))
        holdout_2025 = None

    # Construct Stacked Joint Posterior (D_stack = n_chains * n_samples = 3,000 draws)
    D_stack = int(n_chains * n_samples)
    rng_st = np.random.default_rng(2026)
    counts = rng_st.multinomial(D_stack, stack_weights)

    stack_2d_keys = ("r0", "kappa", "Ks", "alphas", "slopes", "k_lags", "avg_roas_3yr", "roas_2025", "mroas_2025", "y_hat")
    stack_1d_keys = ("iroas_meta", "rho", "sigma_y")
    st_buf = {k: [] for k in stack_2d_keys + stack_1d_keys}
    st_spec_idx, st_bb_aligned = [], []

    for sp_i, (sp, cnt) in enumerate(zip(admissible_specs, counts)):
        sp["stacking_weight"] = round(float(stack_weights[sp_i]), 4)
        if cnt == 0:
            continue
        ext = sp["ext"]
        idx_draw = rng_st.integers(0, len(ext["r0"]), size=cnt)
        for k in stack_2d_keys + stack_1d_keys:
            st_buf[k].append(ext[k][idx_draw])
        st_spec_idx.append(np.full(cnt, sp_i, dtype=int))

        bb_raw = ext["b_b"][idx_draw]
        if not sp["config"]["use_spline"]:
            bb_pad = np.column_stack([bb_raw[:, :2], np.zeros((cnt, len(SPLINE_KNOTS))), bb_raw[:, 2:]])
            st_bb_aligned.append(bb_pad)
        else:
            st_bb_aligned.append(bb_raw)

    stacked = {k: np.vstack(st_buf[k]) for k in stack_2d_keys}
    stacked.update({k: np.concatenate(st_buf[k]) for k in stack_1d_keys})
    P_r0, P_kap, P_Ks = stacked["r0"], stacked["kappa"], stacked["Ks"]
    P_alpha, P_slope, P_klag = stacked["alphas"], stacked["slopes"], stacked["k_lags"]
    P_avg, P_r25, P_m25 = stacked["avg_roas_3yr"], stacked["roas_2025"], stacked["mroas_2025"]
    P_iroas, P_yhat = stacked["iroas_meta"], stacked["y_hat"]
    P_rho, P_sig = stacked["rho"], stacked["sigma_y"]
    P_bb = np.vstack(st_bb_aligned)
    P_spec_idx = np.concatenate(st_spec_idx)

    np.savez(
        o_dir / "stacked_posterior_draws.npz",
        b_b=P_bb,
        r0=P_r0,
        kappa=P_kap,
        Ks=P_Ks,
        alphas=P_alpha,
        slopes=P_slope,
        k_lags=P_klag,
        avg_roas_3yr=P_avg,
        roas_2025=P_r25,
        mroas_2025=P_m25,
        iroas_meta=P_iroas,
        rho=P_rho,
        sigma_y=P_sig,
        spec_idx=P_spec_idx,
        stack_weights=stack_weights,
        spec_names=np.asarray([sp["spec_id"] for sp in admissible_specs]),
        init_vals=init_vals,
        tilde_s=tilde_s,
    )

    # Quantitative Contraction & Confidence Tiers
    prior_var = prior_check["prior_mroas_var"]
    channel_summary = {}
    stacked_roas_rows = []
    for j, ch in enumerate(CHANNELS):
        post_var = float(np.var(P_m25[:, j]))
        contraction = max(0.0, 1.0 - post_var / max(float(prior_var[j]), 1e-9))
        if ch == "meta_spend":
            tier = "HIGH (Experiment-Anchored)"
        elif ch in ("google_search_spend", "tv_spend"):
            tier = "LOW"
        elif ch == "influencer_spend":
            tier = "MODERATE-HIGH"
        else:
            tier = "MODERATE"
        omega = (1.0 - contraction) / max(contraction, 0.15)
        if ch == "google_search_spend":
            # Enforce identification ambiguity penalty reflecting 88.5% unobserved demand-capture confounding
            omega = max(omega, 4.5)
        elif ch == "tv_spend":
            # TV posterior contraction is high (precisely low ROAS), only mild structural ambiguity vs Diwali
            omega = max(omega, 0.35)

        channel_summary[ch] = {
            "label": CHANNEL_LABELS[ch],
            "confidence_tier": tier,
            "posterior_contraction": round(contraction, 4),
            "ambiguity_weight_omega": round(omega, 3),
            "r0": summarize_draws(P_r0[:, j]),
            "kappa": summarize_draws(P_kap[:, j]),
            "half_sat_m_inr": summarize_draws(P_Ks[:, j]),
            "alpha": summarize_draws(P_alpha[:, j]),
            "avg_roas_3yr": summarize_draws(P_avg[:, j]),
            "roas_2025": summarize_draws(P_r25[:, j]),
            "mroas_2025": summarize_draws(P_m25[:, j]),
        }
        cs = channel_summary[ch]
        stacked_roas_rows.append(
            {
                "channel": ch,
                "label": cs["label"],
                "confidence_tier": cs["confidence_tier"],
                "contraction": round(contraction, 3),
                "avg_roas_3yr_med": cs["avg_roas_3yr"]["median"],
                "avg_roas_3yr_ci90_lo": cs["avg_roas_3yr"]["ci90"][0],
                "avg_roas_3yr_ci90_hi": cs["avg_roas_3yr"]["ci90"][1],
                "mroas_2025_med": cs["mroas_2025"]["median"],
                "mroas_2025_ci90_lo": cs["mroas_2025"]["ci90"][0],
                "mroas_2025_ci90_hi": cs["mroas_2025"]["ci90"][1],
                "alpha_med": cs["alpha"]["median"],
            }
        )
    pd.DataFrame(stacked_roas_rows).to_csv(o_dir / "stacked_roas_table.csv", index=False, lineterminator="\n")

    # Revenue decomposition using stacked posterior median predictions
    y_hat_med = np.median(P_yhat, axis=0)
    fit_stack = compute_fit_metrics(y, y_hat_med)
    bb_med = np.median(P_bb, axis=0)
    X_seas_part = X_base_sp[:, : 2 + len(SPLINE_KNOTS) + 4] @ bb_med[: 2 + len(SPLINE_KNOTS) + 4]
    X_ctrl_part = X_base_sp[:, 2 + len(SPLINE_KNOTS) + 4 :] @ bb_med[2 + len(SPLINE_KNOTS) + 4 :]
    lag_mats_full = {j: build_lag_matrix(S[:, j], init_vals[j], WEIBULL_MAX_LAG) for j in WEIBULL_CH_IDX}
    rep_indices = range(0, D_stack, 10)
    media_contribs_med = np.zeros((len(S), 6), dtype=float)
    for d_i in rep_indices:
        p_di = {"alphas": P_alpha[d_i], "r0": P_r0[d_i], "kappa": P_kap[d_i], "slopes": P_slope[d_i], "k_lags": P_klag[d_i]}
        media_contribs_med += compute_media_channels(S, p_di, init_vals, tilde_s, lag_mats=lag_mats_full)
    media_contribs_med /= len(rep_indices)

    decomp_df = pd.DataFrame(
        {
            "week_start": df_clean["week_start"],
            "year": df_clean["year"],
            "actual_rev_m": y,
            "pred_rev_m": y_hat_med,
            "pred_rev_p05_m": np.percentile(P_yhat, 5, axis=0),
            "pred_rev_p95_m": np.percentile(P_yhat, 95, axis=0),
            "baseline_trend_seas_m": X_seas_part,
            "controls_promos_m": X_ctrl_part,
            **{f"{ch}_contrib_m": media_contribs_med[:, j] for j, ch in enumerate(CHANNELS)},
        }
    )
    if holdout_2025 is not None:
        idx_25 = int(np.where(mask_2025)[0][0])
        for col, arr_v in [("oos_2025_pred_m", holdout_2025["y_pred_mean"]), ("oos_2025_pi90_lo", holdout_2025["pi90_lo"]), ("oos_2025_pi90_hi", holdout_2025["pi90_hi"])]:
            tmp = np.full(len(df_clean), np.nan)
            tmp[idx_25:] = arr_v
            decomp_df[col] = tmp
    decomp_df.to_csv(o_dir / "revenue_decomposition.csv", index=False, lineterminator="\n")

    # Package clean JSON summary (stripping heavy model objects)
    specs_json = []
    for sp in spec_results:
        specs_json.append(
            {
                "spec_id": sp["spec_id"],
                "admissible": sp["admissible"],
                "stacking_weight": sp.get("stacking_weight", 0.0),
                "diagnostics": sp["diagnostics"],
                "in_sample": sp["in_sample"],
                "meta_geo_iroas": sp["meta_geo_iroas"],
                "avg_roas_3yr_median": sp["avg_roas_3yr_median"],
                "mroas_2025_median": sp["mroas_2025_median"],
            }
        )

    tot_rev_3yr = float(y.sum())
    tot_rev_25 = float(y[mask_2025].sum())
    tot_media_3yr = float(media_contribs_med.sum())
    tot_media_25 = float(media_contribs_med[mask_2025].sum())

    m1_res = spec_results[0]
    model_results = {
        "pipeline_version": "v3_bayesian_causal_stacked",
        "audit": audit_info,
        "lift_test_info": lift_info,
        "baseline_diagnosis": baseline_diag,
        "causal_and_identification_audits": {
            **causal_audits,
            "dead_end_raw_organic_traffic": {
                "meta_geo_iroas_collapsed": round(float(fwd_bad_org["iroas_geo"]), 3),
                "tv_total_contrib_m": round(float(fwd_bad_org["C_mat"][:, 4].sum()), 2),
            },
            "dead_end_unconstrained_hill_rejected_by_HG4": {
                "google_search_slope": round(float(sum_rej_hill["slopes"][0, 1]), 3),
                "google_search_avg_roas": round(gs_a_rej, 3),
                "google_search_mroas_2025": round(gs_m_rej, 3),
                "rejected": bool(gs_m_rej > gs_a_rej),
            },
            "tv_identification": {
                "n_flights": len(flights),
                "n_festive_flights": sum(1 for r in lofo_res if r["has_festive_overlap"]),
                "lofo_flights": lofo_res,
                "lofo_tv_avg_roas_range": [
                    round(min(r["tv_avg_roas_3yr"] for r in lofo_res), 3),
                    round(max(r["tv_avg_roas_3yr"] for r in lofo_res), 3),
                ],
                "no_festive_flag_tv_avg_roas": round(float(sf_nf["avg_roas_3yr"][0, 4]), 3),
                "tv_net_cost_rebate_avg_roas": round(float(sf_tvnet["avg_roas_3yr"][0, 4]), 3),
            },
        },
        "prior_predictive_check": {k: v for k, v in prior_check.items() if k != "prior_mroas_var"},
        "lift_test_validation": {
            "measured_meta_iroas": lift_info["measured_iroas"],
            "measured_meta_90ci": lift_info["ci90"],
            "stage1_uncalibrated_geo_iroas": summarize_draws(uncal_iroas),
            "stage1_concordance_pval": round(conc_pval, 4),
            "stage2_stacked_calibrated_geo_iroas": summarize_draws(P_iroas),
            "regime_a2_reallocated_geo_iroas_map": round(float(sf_a2["iroas_meta"][0]), 3),
            "regime_a2_meta_mroas_2025_map": round(float(sf_a2["mroas_2025"][0, 0]), 3),
        },
        "stacked_posterior": {
            "total_draws": D_stack,
            "in_sample_median_fit": {
                "r2": round(fit_stack["r2"], 4),
                "mape_pct": round(fit_stack["mape_pct"], 2),
                "nrmse_pct": round(fit_stack["nrmse_pct"], 2),
                "dw": round(fit_stack["dw"], 3),
                "rho_median": round(float(np.median(P_rho)), 3),
                "sigma_y_median": round(float(np.median(P_sig)), 3),
            },
            "base_and_control_coefs_median_m_inr": {
                nm: round(float(np.median(P_bb[:, i])), 4) for i, nm in enumerate(base_names_sp)
            },
            "revenue_decomposition_summary": {
                "total_3yr_rev_m_inr": round(tot_rev_3yr, 2),
                "total_3yr_media_contrib_m_inr": round(tot_media_3yr, 2),
                "media_share_3yr_pct": round(100.0 * tot_media_3yr / tot_rev_3yr, 2),
                "total_2025_rev_m_inr": round(tot_rev_25, 2),
                "total_2025_media_contrib_m_inr": round(tot_media_25, 2),
                "media_share_2025_pct": round(100.0 * tot_media_25 / tot_rev_25, 2),
            },
            "channels": channel_summary,
        },
        "specification_ensemble": specs_json,
        "validation": {
            "rolling_cv": cv_summary,
            "holdout_2025_m1": (
                {k: v for k, v in holdout_2025.items() if k not in ("elpd_per_week", "y_pred_mean", "pi90_lo", "pi90_hi")}
                if holdout_2025 is not None
                else None
            ),
        },
        "ref_means_full": fold_full["ref_means"],
        "base_names_sp": base_names_sp,
        "init_vals_m_inr": init_vals.tolist(),
        "tilde_s_m_inr": tilde_s.tolist(),
        "primary_m1_diagnostics": m1_res["diagnostics"],
    }

    with open(o_dir / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=2)
    return model_results


def main():
    parser = argparse.ArgumentParser(description="Run Verdant Leaf v3 Bayesian Causal MMM pipeline.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-warmup", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=750)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()
    run_modeling_pipeline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        n_chains=args.n_chains,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
        workers=args.workers,
        skip_cv=args.skip_cv,
    )


if __name__ == "__main__":
    main()
