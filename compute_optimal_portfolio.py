"""
Compute the max-Sharpe optimal factor portfolio (USA, vw_cap) and save to JSON.
Optionally plot cumulative return vs aggregate option open interest.

Optimizer: expanding window (60-month min, 12-month rebalance),
Ledoit-Wolf covariance, long-only weights capped at 10% per factor.

Output JSON at intermediary_results/max_sharpe_portfolio.json with structure:
    {
        "params": {...},
        "returns": [{"date": "YYYY-MM-DD", "ret_opt": ..., "ret_ew": ..., "n_factors": ...}, ...]
    }

Usage:
    .venv/Scripts/python compute_optimal_portfolio.py
"""

from config import PATH
from data import load_jkp_factor_returns, load_optionm_agg

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize as sp_minimize

# =========================================================================
#  CONSTANTS
# =========================================================================

START_DATE = "1985-01-01"
PLOT_START = "1990-01-01"
MIN_OBS = 60
REBAL_FREQ = 12
MAX_WEIGHT = 0.10
JSON_PATH = PATH["INTERMEDIARY_RESULTS"] / "max_sharpe_portfolio.json"
PDF_PATH = PATH["PROJECT_ROOT"] / "max_sharpe_vs_oi.pdf"


# =========================================================================
#  MAX-SHARPE PORTFOLIO OPTIMIZER
# =========================================================================

def _max_sharpe_weights(mu, cov, max_w=MAX_WEIGHT):
    """
    Long-only maximum-Sharpe-ratio portfolio weights with an upper bound.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns (length n).
    cov : np.ndarray
        Covariance matrix (n x n).
    max_w : float
        Maximum weight per asset.

    Returns
    -------
    np.ndarray of weights (length n), summing to 1.
    """
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, max_w)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-12:
            return 0.0
        return -port_ret / port_vol

    res = sp_minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if res.success:
        return res.x
    return w0


def build_max_sharpe_portfolio(df, min_obs=MIN_OBS, rebal=REBAL_FREQ,
                               window=None, return_weights=False):
    """
    Build a max-Sharpe portfolio from factor returns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date, name, ret_signed.
    min_obs : int
        Minimum number of observations before optimizing.
    rebal : int
        Rebalance every *rebal* months.
    window : int or None
        If None, use an expanding window (all data up to month t).
        If an int, use a rolling window of *window* months.
    return_weights : bool
        If True, also return a list of weight records (one dict per
        rebalance date) alongside the returns DataFrame.

    Returns
    -------
    pd.DataFrame with columns: date, ret_opt, ret_ew, n_factors
    If return_weights=True, returns (pd.DataFrame, list of dicts).
    """
    wide = (
        df.pivot_table(index="date", columns="name", values="ret_signed")
        .sort_index()
    )

    dates = wide.index
    records = []
    weight_records = [] if return_weights else None
    weights = None
    last_rebal = -rebal

    for i in range(len(dates)):
        row = wide.iloc[i]
        available = row.dropna()

        if len(available) < 2:
            continue

        ret_ew = available.mean()

        if i < min_obs:
            records.append({
                "date": dates[i], "ret_opt": ret_ew,
                "ret_ew": ret_ew, "n_factors": len(available),
            })
            continue

        if (i - last_rebal) >= rebal or weights is None:
            if window is not None:
                start_idx = max(0, i - window)
                hist = wide.iloc[start_idx:i].dropna(axis=1, how="all")
            else:
                hist = wide.iloc[:i].dropna(axis=1, how="all")
            common = hist.columns.intersection(available.index)
            hist = hist[common].dropna()

            if len(hist) >= min_obs and hist.shape[1] >= 2:
                try:
                    lw = LedoitWolf().fit(hist.values)
                    cov = lw.covariance_
                    mu = hist.mean().values
                    w = _max_sharpe_weights(mu, cov)
                    weights = pd.Series(w, index=hist.columns)
                    last_rebal = i
                    if return_weights:
                        weight_records.append({'date': dates[i], 'weights': dict(zip(hist.columns, w))})
                except Exception:
                    weights = None
            else:
                weights = None

        if weights is not None:
            common = weights.index.intersection(available.index)
            if len(common) >= 2:
                w_aligned = weights[common]
                w_aligned = w_aligned / w_aligned.sum()
                ret_opt = (available[common] * w_aligned).sum()
            else:
                ret_opt = ret_ew
        else:
            ret_opt = ret_ew

        records.append({
            "date": dates[i], "ret_opt": ret_opt,
            "ret_ew": ret_ew, "n_factors": len(available),
        })

    if return_weights:
        return pd.DataFrame(records), weight_records
    return pd.DataFrame(records)


# =========================================================================
#  COMPUTE & SAVE
# =========================================================================

def compute_and_save():
    """
    Build max-Sharpe portfolio from USA vw_cap factor returns and save to JSON.

    Returns
    -------
    pd.DataFrame with columns: date, ret_opt, ret_ew, n_factors
    """
    print("[max_sharpe] Loading vw_cap factor returns ...")
    fr = load_jkp_factor_returns(weighting="vw_cap")
    fr = fr[(fr["location"] == "usa") & (fr["date"] >= START_DATE)].copy()
    fr["ret_signed"] = fr["ret"] * fr["direction"]

    print("[max_sharpe] Building max-Sharpe portfolio ...")
    port = build_max_sharpe_portfolio(fr)
    port["date"] = pd.to_datetime(port["date"])

    # Compute cumulative return for summary
    cumret = (1 + port["ret_opt"]).cumprod()
    print(f"[max_sharpe] {len(port)} months, "
          f"{port['date'].min().strftime('%Y-%m')} to "
          f"{port['date'].max().strftime('%Y-%m')}, "
          f"total return: {cumret.iloc[-1]:.2f}x")

    # Build JSON-serializable output
    records = port.copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    output = {
        "params": {
            "location": "usa",
            "weighting": "vw_cap",
            "optimizer": "max_sharpe",
            "covariance": "ledoit_wolf",
            "start_date": START_DATE,
            "min_obs": MIN_OBS,
            "rebal_freq_months": REBAL_FREQ,
            "max_weight_per_factor": MAX_WEIGHT,
        },
        "returns": records.to_dict(orient="records"),
    }

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[max_sharpe] Saved -> {JSON_PATH}")

    return port


# =========================================================================
#  PLOT: CUMULATIVE RETURN vs OPTION OPEN INTEREST
# =========================================================================

def plot_vs_oi(port):
    """
    Dual y-axis chart: max-Sharpe cumulative return (left) vs
    aggregate option open interest (right). Saves PDF.

    Parameters
    ----------
    port : pd.DataFrame
        Output of compute_and_save / build_max_sharpe_portfolio
        with columns: date, ret_opt.
    """
    # Cumulative return (trim to plot start)
    port_idx = port.set_index("date").sort_index()
    port_idx = port_idx[port_idx.index >= PLOT_START]
    cumret = (1 + port_idx["ret_opt"]).cumprod()

    # Option open interest
    print("[plot] Loading OptionMetrics aggregated data ...")
    om = load_optionm_agg()
    oi = om.groupby("ym")["total_oi"].sum().sort_index()
    oi = oi[oi.index >= PLOT_START]

    # ── figure ──────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(14, 7))

    c_opt = "#1f77b4"
    c_oi = "#2ca02c"

    # Left axis: cumulative return
    ln1 = ax1.plot(cumret.index, cumret.values,
                   color=c_opt, linewidth=2.0,
                   label="Max-Sharpe Factor Portfolio")
    ax1.set_yscale("log")
    ax1.set_ylabel("Cumulative Return (log scale)", fontsize=12,
                   color=c_opt)
    ax1.tick_params(axis="y", labelcolor=c_opt)
    ax1.set_xlabel("Date", fontsize=12)

    # Auto y-ticks
    opt_log_min = np.log10(cumret.min()) - 0.05
    opt_log_max = np.log10(cumret.max()) + 0.10
    tick_candidates = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5,
                       10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0]
    tick_vals = [v for v in tick_candidates
                 if 10 ** opt_log_min <= v <= 10 ** opt_log_max]
    if tick_vals:
        ax1.set_yticks(tick_vals)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x:.0f}x" if x >= 1 else f"{x:.2f}x"))
    ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax1.set_ylim(10 ** opt_log_min, 10 ** opt_log_max)

    # Right axis: option open interest
    ax2 = ax1.twinx()
    oi_billions = oi / 1e9
    ax2.fill_between(oi.index, 0, oi_billions.values,
                     color=c_oi, alpha=0.15)
    ln2 = ax2.plot(oi.index, oi_billions.values,
                   color=c_oi, linewidth=1.0, alpha=0.7,
                   label="Total Option OI")
    ax2.set_ylabel("Option Open Interest (bn contracts)", fontsize=12,
                   color=c_oi)
    ax2.tick_params(axis="y", labelcolor=c_oi)

    # Common formatting
    ax1.set_xlim(pd.Timestamp(PLOT_START),
                 max(cumret.index.max(), oi.index.max()))
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax1.set_title(
        "Max-Sharpe Factor Portfolio vs Option Open Interest",
        fontsize=14, fontweight="bold", loc="left")

    ax1.grid(True, alpha=0.2)

    # Combined legend
    handles = ln1 + ln2
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels,
               loc="upper left", fontsize=10, framealpha=0.9)

    ax1.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved -> {PDF_PATH}")


# =========================================================================
#  CLI
# =========================================================================

if __name__ == "__main__":
    port = compute_and_save()
    plot_vs_oi(port)
    print("[max_sharpe] Done.")
