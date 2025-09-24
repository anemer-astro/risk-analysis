#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Analysis & Stress Testing (Step-by-step project)
====================================================

This script computes portfolio risk metrics and diagnostics, designed to pair with
the portfolio optimization repo. 

Features
--------
1) Data & Portfolio:
   - Load prices from CSV (or download via yfinance if requested)
   - Use equal weights OR read weights from CSV (e.g., data/optimal_weights.csv)
   - Optionally choose a column from weights CSV (MaxSharpe_Weight, MinVariance_Weight, etc.)

2) Risk Metrics (daily or weekly):
   - Historical VaR and Expected Shortfall (ES/CVaR) at 95% and 99%
   - Parametric Normal VaR/ES using sample mean & std
   - Monte Carlo VaR/ES via multivariate normal simulation using asset covariances

3) VaR Backtesting:
   - Rolling historical VaR estimation (e.g., 252-day window)
   - Exceptions count and Kupiec POF test (proportion of failures)

4) Drawdowns:
   - Max drawdown & duration
   - Drawdown curve

5) Stress Testing:
   - Historical scans for worst 1-day, worst N-day windows
   - Hypothetical shocks by asset category (simple mapping) to estimate portfolio P/L

Outputs
-------
- data/risk_report.csv
- data/var_backtest.csv
- data/stress_hypothetical.csv
- data/stress_historical.csv
- figures/ret_hist_var.png
- figures/rolling_var_exceedances.png
- figures/drawdown_curve.png

Usage
-----
# Using equal weights from prices in data/prices.csv (daily returns)
python risk_analysis.py --prices data/prices.csv --alpha 0.95 --alpha-hi 0.99

# Using weights from data/optimal_weights.csv (select column)
python risk_analysis.py --prices data/prices.csv --weights data/optimal_weights.csv \
    --weights-col MaxSharpe_Weight --alpha 0.95 --alpha-hi 0.99

# Weekly returns and Monte Carlo with 100k sims
python risk_analysis.py --prices data/prices.csv --weekly --mc 100000

# Download prices directly (if yfinance available) for a quick demo
python risk_analysis.py --download --tickers "VTI,VEA,VWO,AGG,TLT,TIP,VNQ,DBC" --start 2015-01-01

Notes
-----
- Install: pip install numpy pandas scipy matplotlib yfinance
- Internet may be blocked in some environments; --download will then fail, but CSV mode works.
"""

import os
import math
import argparse
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

TRADING_DAYS = 252

# ------------------------- I/O & Utilities -------------------------

def ensure_dirs(outdir: str):
    """Create the default output folders if missing."""
    os.makedirs(os.path.join(outdir, "data"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)

def load_prices(prices_csv: str) -> pd.DataFrame:
    """
    Load prices from CSV and align to business days.
    The file must contain Adj Close-like columns for each asset.
    """
    if not os.path.exists(prices_csv):
        raise FileNotFoundError(f"Missing {prices_csv}. Provide --prices or use --download.")
    df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    df = df.asfreq("B").ffill().bfill().dropna(how="all", axis=1)
    return df

def download_prices(tickers: str, start: str, end: Optional[str]) -> pd.DataFrame:
    if not YF_OK:
        raise RuntimeError("yfinance not available in this environment. Use --prices to load a CSV.")
    tlist = [t.strip() for t in tickers.split(",") if t.strip()]
    df = yf.download(tlist, start=start, end=end, auto_adjust=False)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df[tlist].asfreq("B").ffill().bfill()
    return df

def compute_log_returns(prices: pd.DataFrame, weekly: bool=False) -> pd.DataFrame:
    """
    Compute (optionally weekly) log returns.
    Weekly returns are resampled to W-FRI and summed to match log compounding.
    """
    rets = np.log(prices / prices.shift(1)).dropna(how="all")
    if weekly:
        rets = rets.resample("W-FRI").sum().dropna(how="all")
    return rets

def load_weights(weights_csv: Optional[str],
                 assets: pd.Index,
                 weights_col: Optional[str]=None) -> np.ndarray:
    if weights_csv is None:
        # equal weights
        w = np.ones(len(assets)) / len(assets)
        return w
    df = pd.read_csv(weights_csv)
    # expected columns: Asset, <one or more weight columns>
    if "Asset" not in df.columns:
        raise ValueError("weights CSV must have an 'Asset' column.")
    df = df.set_index("Asset").reindex(assets)
    if weights_col is None:
        # take the first non-Asset column by default
        candidates = [c for c in df.columns if "Weight" in c or c != "Asset"]
        if not candidates:
            raise ValueError("Could not infer weight column. Pass --weights-col explicitly.")
        weights_col = candidates[0]
    if weights_col not in df.columns:
        raise ValueError(f"Column '{weights_col}' not in weights CSV. Available: {list(df.columns)}")
    w = df[weights_col].values.astype(float)
    if np.any(np.isnan(w)):
        raise ValueError("Weights contain NaN after aligning to assets. Check Asset names.")
    s = w.sum()
    if s <= 0:
        raise ValueError("Sum of weights must be positive.")
    w = np.maximum(0.0, w) / s
    return w

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.array(weights, dtype=float)
    # Align columns just in case
    if len(w) != returns.shape[1]:
        raise ValueError("Weights length does not match number of assets in returns.")
    pr = (returns * w).sum(axis=1)
    return pr

# ------------------------- VaR & ES (core) -------------------------
#VaR answers the question:"With X% confidence, what is the maximum loss we expect over a given period?"
#Expected Shortfall (ES) — also called CVaR — answers:"If we end up in that worst α-tail, what is our average loss?"
def hist_var_es(losses: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Historical VaR/ES: losses is array of positive 'loss' numbers (e.g., -returns).
    VaR_alpha = empirical percentile of loss at 'alpha' (e.g., 0.95).
    ES_alpha  = average loss beyond VaR_alpha.
    """
    if len(losses) < 10:
        return np.nan, np.nan
    var = np.quantile(losses, alpha)
    es = losses[losses >= var].mean() if np.any(losses >= var) else var
    return float(var), float(es)

#Aspect -	Historical VaR -	Normal (Parametric) VaR
#Assumption - Uses actual empirical distribution of past losses - Assumes losses are normally distributed
#Shape -	Can capture skew, fat tails, multimodality -	Symmetric, thin-tailed

def norm_var_es(mu: float, sigma: float, alpha: float) -> Tuple[float, float]:
    """
    Parametric Normal VaR/ES for LOSS distribution where loss = -return.
    If returns ~ N(mu, sigma^2), then loss L = -R ~ N(-mu, sigma^2).
    VaR_alpha(L) = q_alpha(L) = -mu + z_alpha*sigma
    ES_alpha(L)  = E[L | L >= VaR_alpha] = -mu + sigma * (phi(z_alpha) / (1-alpha))
    """
    if sigma <= 0 or not np.isfinite(sigma):
        return np.nan, np.nan
    z = norm.ppf(alpha)
    var = -mu + z * sigma
    es  = -mu + sigma * (norm.pdf(z) / (1.0 - alpha))
    return float(var), float(es)

#It simulates many draws of asset returns from a multivariate normal with mean mu_vec and covariance cov, 
# turns each draw into a portfolio return using weights, converts those to losses (=-return), 
# then computes VaR and ES from the simulated loss distribution (using the same empirical formulas as hist_var_es).

def mc_var_es(mu_vec: np.ndarray, cov: np.ndarray, weights: np.ndarray,
              alpha: float, n_sims: int=100000, seed: int=7) -> Tuple[float, float]:
    """
    Monte Carlo VaR/ES using multivariate normal for asset returns.
    """
    rng = np.random.default_rng(seed)# reproducible randomness
    n = len(mu_vec)
    # Cholesky factorization to impose the desired covariance
    C = np.linalg.cholesky(cov + 1e-12*np.eye(n))
    Z = rng.standard_normal((n_sims, n))# Independent standard normal samples
    # Correlated returns: each row is one scenario of asset returns
    sims = (Z @ C.T) + mu_vec  
    p_rets = sims @ weights# Map asset returns to portfolio returns
    losses = -p_rets
    return hist_var_es(losses, alpha)

# ----------------------- VaR Backtesting ---------------------------
#In practice, VaR is often backtested by comparing realized daily losses to the rolling VaR estimate.
#If you see too many breaches (exceptions), your VaR model is underestimating risk.
#If you see too few breaches, it’s too conservative.

def rolling_hist_var(losses: pd.Series, window: int, alpha: float) -> pd.Series:
    """
    Compute rolling historical VaR on loss series (loss = -return), in-sample.
    """
    def q(x):
        return np.quantile(x, alpha)
    return losses.rolling(window=window, min_periods=window).apply(q, raw=True)

#if your VaR model is correct:Out of 252 trading days, you should see about 
#0.05×252≈12 exceptions (days when actual loss > VaR).
#The Kupiec test checks: Does the observed frequency of breaches match the expected frequency?

def kupiec_pof_test(losses: pd.Series, var_series: pd.Series, alpha: float) -> Tuple[int, float, float]:
    """
    Kupiec Proportion of Failures (POF) test.
    H0: exception probability = (1 - alpha).
    Returns: (num_exceptions, exception_rate, p_value)
    """
    mask = var_series.notna()
    L = losses[mask]
    V = var_series[mask]
    exceptions = (L >= V)  # losses exceeding VaR (breach)
    x = int(exceptions.sum())
    n = int(exceptions.shape[0])
    if n == 0:
        return 0, np.nan, np.nan

    p = 1 - alpha
    # Likelihood ratio statistic
    pi_hat = x / n if n > 0 else 0.0
    # Avoid log(0)
    def logp(k, n, p):
        if p <= 0 or p >= 1:
            return -np.inf
        return (n - k) * np.log(1 - p) + k * np.log(p)

    lr = -2.0 * (logp(x, n, p) - logp(x, n, max(min(pi_hat, 1-1e-12), 1e-12)))
    pval = 1 - chi2.cdf(lr, df=1)#If p-value is small (<0.05), Your VaR model does not match the observed exception rate.
    return x, (x / n), float(pval)

# ----------------------------- Drawdowns ---------------------------
#The function computes the drawdown curve of a cumulative return (or cumulative wealth) series.
#cum = cumulative portfolio value or return index (e.g., starting at 1.0 and growing over time).
#dd = a series showing the percentage drop from the previous peak at each point in time.
def drawdown_curve(cum: pd.Series) -> pd.Series:
    roll_max = cum.cummax()
    dd = cum / roll_max - 1.0
    return dd

# ------------------------- Stress Testing --------------------------
#Takes your asset names (assets, a pandas Index) and portfolio weights (weights).
#Looks up a shock (a one-period return, e.g. -0.20 for −20%) for each asset (defaulting to DEFAULT_SHOCKS).
#Computes stress P&L (as a return) by summing weight_i × shock_i across assets.
#Returns a single number (e.g., -0.062 means −6.2% portfolio P&L under that scenario).

DEFAULT_SHOCKS: Dict[str, float] = {
    # Simple illustrative shocks (percentage returns)
    # Equities
    "VTI": -0.20, "SPY": -0.20, "VEA": -0.18, "VWO": -0.22, "IWM": -0.25,
    # Bonds (approximate price drops for a rate shock)
    "AGG": -0.04, "BND": -0.04, "TLT": -0.08, "IEF": -0.05, "LQD": -0.06, "HYG": -0.09,
    # Real assets
    "VNQ": -0.12, "DBC": -0.10, "GLD": +0.05,
    # Cash-like
    "BIL": +0.00
}

def hypothetical_stress_pnl(
    assets: pd.Index, weights: np.ndarray,
    shocks: Optional[Dict[str, float]] = None,
    prefer_exact: bool = True
) -> Tuple[float, Optional[float]]:
    shocks = shocks or DEFAULT_SHOCKS
    # normalize once
    norm_keys = {str(k).strip().lower(): v for k, v in shocks.items()}
    pnl_ret = 0.0

    for i, name in enumerate(assets):
        nm = str(name).strip()
        nm_l = nm.lower()
        shock = None
        if prefer_exact:
            # exact first
            if nm_l in norm_keys:
                shock = norm_keys[nm_l]
            else:
                # then prefix/suffix/substring
                for k, v in norm_keys.items():
                    if nm_l.startswith(k) or nm_l.endswith(k) or (k in nm_l):
                        shock = v
                        break
        else:
            for k, v in norm_keys.items():
                if k == nm_l or nm_l.startswith(k) or nm_l.endswith(k) or (k in nm_l):
                    shock = v
                    break

        pnl_ret += weights[i] * (shock if shock is not None else 0.0)

    return float(pnl_ret)


def historical_worst_windows(returns: pd.Series, horizons=(1, 5, 10, 20)) -> pd.DataFrame:
    rows = []
    for h in horizons:
        rolled = returns.rolling(h).sum()
        idx = rolled.idxmin()
        val = rolled.min()
        rows.append({"horizon_days": h, "start": (idx - pd.tseries.offsets.BDay(h-1)) if pd.notna(idx) else None,
                     "end": idx, "return": float(val) if pd.notna(val) else np.nan})
    return pd.DataFrame(rows)

# ------------------------------- Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", type=str, default="data/prices.csv")
    ap.add_argument("--download", action="store_true", help="Download prices via yfinance instead of reading CSV.")
    ap.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers for --download mode.")
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--weights", type=str, default=None, help="CSV with columns: Asset, <Weight columns>")
    ap.add_argument("--weights-col", type=str, default=None, help="Column name in weights CSV to use.")
    ap.add_argument("--alpha", type=float, default=0.95, help="Lower VaR/ES confidence (e.g., 0.95).")
    ap.add_argument("--alpha-hi", type=float, default=0.99, help="Higher VaR/ES confidence (e.g., 0.99).")
    ap.add_argument("--mc", type=int, default=100000, help="Monte Carlo simulations.")
    ap.add_argument("--bt-window", type=int, default=252, help="Rolling window for VaR backtest.")
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    ensure_dirs(args.outdir)
    data_dir = os.path.join(args.outdir, "data")
    fig_dir = os.path.join(args.outdir, "figures")

    # Load or download prices
    if args.download:
        if not args.tickers:
            raise ValueError("--tickers must be provided for --download mode.")
        prices = download_prices(args.tickers, args.start, args.end)
        prices.to_csv(os.path.join(data_dir, "prices.csv"))
    else:
        prices = load_prices(args.prices)

    # Returns
    returns = compute_log_returns(prices, weekly=args.weekly)
    periods = 52 if args.weekly else TRADING_DAYS

    # Weights
    weights = load_weights(args.weights, assets=returns.columns, weights_col=args.weights_col)

    # Portfolio returns and cumulative growth
    p_ret = portfolio_returns(returns, weights)
    p_loss = -p_ret
    cum = np.exp(p_ret.cumsum())

    # Summary stats
    mu = p_ret.mean() * periods
    sigma = p_ret.std() * np.sqrt(periods)
    skew = p_ret.skew()
    kurt = p_ret.kurt()

    # VaR/ES (Historical)
    var95_h, es95_h = hist_var_es(p_loss.values, args.alpha)
    var99_h, es99_h = hist_var_es(p_loss.values, args.alpha_hi)

    # VaR/ES (Parametric Normal)
    var95_n, es95_n = norm_var_es(p_ret.mean(), p_ret.std(), args.alpha)
    var99_n, es99_n = norm_var_es(p_ret.mean(), p_ret.std(), args.alpha_hi)

    # VaR/ES (Monte Carlo)
    mu_vec = returns.mean().values
    cov = returns.cov().values
    var95_mc, es95_mc = mc_var_es(mu_vec, cov, weights, args.alpha, n_sims=args.mc, seed=7)
    var99_mc, es99_mc = mc_var_es(mu_vec, cov, weights, args.alpha_hi, n_sims=args.mc, seed=13)

    # Save risk report
    risk_rows = [
        ["Annualized Mean Return", mu],
        ["Annualized Volatility", sigma],
        ["Skewness (daily/weekly)", skew],
        ["Excess Kurtosis (daily/weekly)", kurt],
        [f"Hist VaR {int(args.alpha*100)}%", var95_h],
        [f"Hist ES  {int(args.alpha*100)}%", es95_h],
        [f"Hist VaR {int(args.alpha_hi*100)}%", var99_h],
        [f"Hist ES  {int(args.alpha_hi*100)}%", es99_h],
        [f"Normal VaR {int(args.alpha*100)}%", var95_n],
        [f"Normal ES  {int(args.alpha*100)}%", es95_n],
        [f"Normal VaR {int(args.alpha_hi*100)}%", var99_n],
        [f"Normal ES  {int(args.alpha_hi*100)}%", es99_n],
        [f"MC VaR {int(args.alpha*100)}%", var95_mc],
        [f"MC ES  {int(args.alpha*100)}%", es95_mc],
        [f"MC VaR {int(args.alpha_hi*100)}%", var99_mc],
        [f"MC ES  {int(args.alpha_hi*100)}%", es99_mc],
    ]
    risk_df = pd.DataFrame(risk_rows, columns=["Metric", "Value"])
    risk_df.to_csv(os.path.join(data_dir, "risk_report.csv"), index=False)

    # VaR Backtest (rolling historical)
    var_roll = rolling_hist_var(p_loss, window=args.bt_window, alpha=args.alpha)
    exc = (p_loss >= var_roll).astype(float)
    x, rate, pval = kupiec_pof_test(p_loss, var_roll, args.alpha)
    back_df = pd.DataFrame({
        "loss": p_loss,
        "var_hist": var_roll,
        "exceed": exc
    })
    back_df.to_csv(os.path.join(data_dir, "var_backtest.csv"))
    # Print brief backtest summary
    print(f"Kupiec POF (alpha={args.alpha:.2f}): exceptions={x}, rate={rate:.4f}, p-value={pval:.4f}")

    # Drawdown analysis
    dd = drawdown_curve(cum)
    mdd = dd.min()
    # Simple duration proxy: longest below-0 run length
    below = (dd < 0).astype(int)
    runs = []
    run = 0
    for b in below:
        if b == 1:
            run += 1
        else:
            runs.append(run)
            run = 0
    runs.append(run)
    dd_dur = int(max(runs) if runs else 0)

    # Historical worst windows
    worst = historical_worst_windows(p_ret, horizons=(1,5,10,20))
    worst.to_csv(os.path.join(data_dir, "stress_historical.csv"), index=False)

    # Hypothetical stress
    hyp_pnl = hypothetical_stress_pnl(returns.columns, weights, shocks=None)
    hyp_df = pd.DataFrame([{"scenario": "DefaultShock", "portfolio_return": hyp_pnl}])
    hyp_df.to_csv(os.path.join(data_dir, "stress_hypothetical.csv"), index=False)

    # ------------------------------ Plots ------------------------------
    # 1) Histogram with VaR lines
    plt.figure(figsize=(8,6))
    plt.hist(p_ret, bins=60, alpha=0.7)
    # VaR lines (convert loss VaR back to return quantiles: VaR is positive loss => return = -VaR)
    plt.axvline(-var95_h, linestyle="--", linewidth=2, label=f"Hist VaR {int(args.alpha*100)}%")
    plt.axvline(-var99_h, linestyle="--", linewidth=2, label=f"Hist VaR {int(args.alpha_hi*100)}%")
    plt.xlabel("Portfolio Return")
    plt.ylabel("Frequency")
    plt.title("Portfolio Return Distribution with Historical VaR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ret_hist_var.png"), dpi=150)
    plt.close()

    # 2) Rolling VaR & exceedances
    plt.figure(figsize=(10,6))
    plt.plot(p_ret.index, p_ret.values, linewidth=1, label="Portfolio Return")
    plt.plot(var_roll.index, -var_roll.values, linewidth=2, label=f"Rolling -VaR {int(args.alpha*100)}%")
    # Exceedances: mark points where p_loss >= var_roll
    ex_idx = back_df.index[back_df["exceed"] > 0]
    plt.scatter(ex_idx, p_ret.loc[ex_idx], s=15, marker="x", label="Exceedance")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.title("Rolling Historical VaR & Exceedances")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rolling_var_exceedances.png"), dpi=150)
    plt.close()

    # 3) Drawdown curve
    plt.figure(figsize=(10,4))
    plt.plot(dd.index, dd.values, linewidth=1)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title(f"Drawdown Curve (Max DD={mdd:.2%}, Longest Duration={dd_dur} days)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "drawdown_curve.png"), dpi=150)
    plt.close()

    # Print summary to console
    print("\n=== Risk Summary ===")
    print(risk_df.to_string(index=False))
    print(f"\nMax Drawdown: {mdd:.2%} | Longest Drawdown Duration: {dd_dur} days")
    print("\nWorst historical windows (h-day cumulative returns):")
    print(worst.to_string(index=False))
    print("\nHypothetical shock portfolio return (DefaultShock): {:.2%}".format(hyp_pnl))
    print("\nSaved:")
    print(" - data/risk_report.csv")
    print(" - data/var_backtest.csv")
    print(" - data/stress_hypothetical.csv")
    print(" - data/stress_historical.csv")
    print(" - figures/ret_hist_var.png")
    print(" - figures/rolling_var_exceedances.png")
    print(" - figures/drawdown_curve.png")

if __name__ == "__main__":
    main()
