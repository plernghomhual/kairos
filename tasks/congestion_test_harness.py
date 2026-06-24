"""Kairos congestion-vs-impact data-sufficiency test harness.

Purpose: decide whether a desk's own gateway/FIX telemetry carries congestion
signal that is INDEPENDENT of the desk's own order size and market volatility.

Status (2026-06-10): instrument validated on synthetic ground truth.
  - Recovers an independent congestion effect when one exists (t ~ 90).
  - Refuses a false positive when congestion is merely a proxy of size (t ~ 0).

To run the real thesis test, replace build_synthetic() with a FIX-log parser that
emits, per (venue, minute): own order size, realized volatility, gateway dwell /
session-stress, and the desk's own slippage. Then call run(real_frame).
The statistics in ols()/run() are unchanged — only the data source changes.
"""

from __future__ import annotations

import numpy as np


def ols(X: np.ndarray, y: np.ndarray):
    """Classical OLS. Returns (beta, t_stats, r2, rss). X must include an intercept column."""
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    rss = float(resid @ resid)
    sigma2 = rss / (n - k)
    se = np.sqrt(np.diag(sigma2 * XtXi))
    t = beta / se
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss
    return beta, t, r2, rss


def build_synthetic(n_days: int = 10, news_per_day: int = 2, congestion_mode: str = "independent", seed: int = 7):
    """Synthetic FIX-session frame with KNOWN ground truth, for instrument validation.

    congestion_mode="independent": gateway dwell carries a real causal effect on
        slippage that own-size/volatility do not explain. Test MUST detect it.
    congestion_mode="proxy": gateway dwell is just a noisy echo of own order size,
        with zero independent effect. Test MUST reject it (false-positive guard).
    """
    r = np.random.default_rng(seed)
    rows = []
    for _ in range(n_days):
        news_minutes = set(r.choice(390, news_per_day, replace=False).tolist())
        for m in range(390):  # 390 trading minutes per day
            open_close = (m < 30) or (m > 360)  # session edges are structurally stressed
            stress = (1.2 if open_close else 0.0) + (2.0 if m in news_minutes else 0.0)
            vol = abs(0.8 + 0.6 * r.standard_normal() + stress)
            size = 0.5 * vol + 0.5 * abs(r.standard_normal())  # desk trades bigger in vol (collinearity)
            if congestion_mode == "independent":
                congestion = 0.5 * stress + 1.0 * abs(r.standard_normal())
                true_c = 0.9
            else:  # proxy
                congestion = 1.3 * size + 0.15 * r.standard_normal()
                true_c = 0.0
            own_impact = 0.7 * size + 0.6 * vol
            slippage = own_impact + true_c * congestion + 0.4 * r.standard_normal()
            rows.append((size, vol, congestion, slippage))
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def run(
    size: np.ndarray,
    vol: np.ndarray,
    cong: np.ndarray,
    slip: np.ndarray,
    t_breach: float = 3.5,
    min_incremental_r2: float = 0.01,
    label: str = "",
):
    """Decompose slippage. Tests whether congestion adds explanatory power beyond size+vol."""
    n = len(slip)
    ones = np.ones(n)
    Xr = np.column_stack([ones, size, vol])  # reduced: own-impact only
    _, _, r2_r, rss_r = ols(Xr, slip)
    Xf = np.column_stack([ones, size, vol, cong])  # full: + gateway congestion
    beta, t, r2_f, rss_f = ols(Xf, slip)
    F = ((rss_r - rss_f) / 1) / (rss_f / (n - 4))  # partial F for congestion
    has_signal = abs(t[3]) > t_breach and (r2_f - r2_r) > min_incremental_r2
    print(f"[{label}]")
    print(f"  reduced R^2 (size+vol)    = {r2_r:.4f}")
    print(f"  full    R^2 (+congestion) = {r2_f:.4f}")
    print(f"  incremental R^2           = {r2_f - r2_r:.4f}")
    print(f"  congestion coef           = {beta[3]:+.3f}")
    print(f"  congestion t-stat         = {t[3]:+.2f}   (|t|>{t_breach} = breach)")
    print(f"  partial F                 = {F:.1f}")
    print(
        f"  VERDICT                   = "
        f"{'SIGNAL: independent congestion effect' if has_signal else 'NO independent signal (redundant/proxy)'}\n"
    )
    return has_signal


if __name__ == "__main__":
    print("=== TEST INSTRUMENT VALIDATION (synthetic ground truth) ===\n")
    run(*build_synthetic(congestion_mode="independent"), label="independent")  # must PASS
    run(*build_synthetic(congestion_mode="proxy"), label="proxy")  # must REJECT
