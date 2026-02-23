"""
CPD analysis on the OPTIMAL (minimum-variance) portfolio by country.

Two-step workflow:
    1. compute_all_portfolios()  -- heavy lifting, saves results to JSON
    2. plot_optimal_portfolios() -- reads JSON, produces PDF (instant)

The JSON cache lives at:  intermediary_results/optimal_portfolios.json

Usage:
    .venv/Scripts/python cpd_optimal_portfolio.py              (compute + plot)
    .venv/Scripts/python cpd_optimal_portfolio.py --plot-only   (plot only)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize as sp_minimize
from config import PATH
from data import load_jkp_factor_returns
from util_locals.factor_themes import THEME_MAP_13

# =========================================================================
#  CONSTANTS
# =========================================================================

START_DATE = "1985-01-01"
BREAK_DATE = "2000-01-01"
MIN_OBS = 60
REBAL_FREQ = 12
MAX_WEIGHT = 0.10

COUNTRY_NAMES = {
    "usa": "United States",
    "gbr": "United Kingdom",
    "jpn": "Japan",
    "deu": "Germany",
    "fra": "France",
    "ita": "Italy",
    "aus": "Australia",
    "hkg": "Hong Kong",
    "swe": "Sweden",
}
ALL_COUNTRIES = list(COUNTRY_NAMES.keys())

CACHE_PATH = PATH["INTERMEDIARY_RESULTS"] / "optimal_portfolios.json"


# =========================================================================
#  PORTFOLIO CONSTRUCTION
# =========================================================================

def _mvp_weights(cov, max_w=MAX_WEIGHT):
    """Long-only minimum-variance portfolio weights with an upper bound."""
    n = cov.shape[0]
    w0 = np.ones(n) / n
    bounds = [(0.0, max_w)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = sp_minimize(
        lambda w: w @ cov @ w,
        w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if res.success:
        return res.x
    return w0


def build_optimal_portfolio(df, min_obs=MIN_OBS, rebal=REBAL_FREQ):
    """
    Build an expanding-window minimum-variance portfolio from factor returns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date, name, ret_signed.

    Returns
    -------
    pd.DataFrame with columns: date, ret_opt, ret_ew, n_factors
    """
    wide = (
        df.pivot_table(index="date", columns="name", values="ret_signed")
        .sort_index()
    )

    dates = wide.index
    records = []
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
            hist = wide.iloc[:i].dropna(axis=1, how="all")
            common = hist.columns.intersection(available.index)
            hist = hist[common].dropna()

            if len(hist) >= min_obs and hist.shape[1] >= 2:
                try:
                    lw = LedoitWolf().fit(hist.values)
                    cov = lw.covariance_
                    w = _mvp_weights(cov)
                    weights = pd.Series(w, index=hist.columns)
                    last_rebal = i
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

    return pd.DataFrame(records)


# =========================================================================
#  CPD
# =========================================================================

def cpd_break_date(rets, min_segment=24):
    """Sup-Wald change-point detection on a return series."""
    rets = rets.dropna()
    if len(rets) < 2 * min_segment:
        return pd.NaT, np.nan
    values = rets.values
    n = len(values)
    best_t, best_idx = 0, min_segment
    for i in range(min_segment, n - min_segment):
        pre, post = values[:i], values[i:]
        diff = post.mean() - pre.mean()
        se = np.sqrt(pre.var(ddof=1) / len(pre) + post.var(ddof=1) / len(post))
        if se > 0:
            t = abs(diff / se)
            if t > best_t:
                best_t, best_idx = t, i
    return rets.index[best_idx], best_t


# =========================================================================
#  STEP 1: COMPUTE & CACHE
# =========================================================================

def compute_all_portfolios():
    """
    Build optimal portfolios for all countries, run CPD, and save to JSON.

    Returns
    -------
    dict with keys:
        'portfolios' : {country_code: [{date, ret_opt, ret_ew, n_factors}, ...]}
        'cpd_dates'  : {country_code: {date, t_stat}}
        'country_names' : {country_code: full_name}
        'start_date' : str
        'break_date' : str
    """
    print("[opt_port] Loading vw_cap factor returns ...")
    fr = load_jkp_factor_returns(weighting="vw_cap")
    fr = fr[fr["date"] >= START_DATE].copy()
    fr["theme"] = fr["name"].map(THEME_MAP_13)
    fr["ret_signed"] = fr["ret"] * fr["direction"]

    result = {
        "portfolios": {},
        "cpd_dates": {},
        "country_names": COUNTRY_NAMES,
        "start_date": START_DATE,
        "break_date": BREAK_DATE,
    }

    for cc in ALL_COUNTRIES:
        cname = COUNTRY_NAMES[cc]
        print(f"[opt_port] {cname} ({cc}) ...", end=" ", flush=True)

        cdf = fr[fr["location"] == cc].copy()
        if len(cdf) == 0:
            print("no data, skipping")
            continue

        n_fac = cdf["name"].nunique()
        print(f"{n_fac} factors ...", end=" ", flush=True)

        port = build_optimal_portfolio(cdf)
        if len(port) < MIN_OBS:
            print("too few observations, skipping")
            continue

        port_indexed = port.set_index("date").sort_index()
        cpd_date, cpd_t = cpd_break_date(port_indexed["ret_opt"])
        print(f"CPD={cpd_date}")

        # Store as list of dicts (JSON-serializable)
        port_records = port.copy()
        port_records["date"] = port_records["date"].dt.strftime("%Y-%m-%d")
        result["portfolios"][cc] = port_records.to_dict(orient="records")
        result["cpd_dates"][cc] = {
            "date": cpd_date.strftime("%Y-%m-%d") if pd.notna(cpd_date) else None,
            "t_stat": float(cpd_t) if not np.isnan(cpd_t) else None,
        }

    # Save
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[opt_port] Cached -> {CACHE_PATH}")

    return result


def load_cached_portfolios():
    """Load pre-computed portfolios from JSON cache."""
    with open(CACHE_PATH, "r") as f:
        raw = json.load(f)

    # Convert back to DataFrames
    all_ports = {}
    for cc, records in raw["portfolios"].items():
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        all_ports[cc] = df

    cpd_dates = {}
    for cc, cpd_info in raw["cpd_dates"].items():
        d = pd.Timestamp(cpd_info["date"]) if cpd_info["date"] else pd.NaT
        t = cpd_info["t_stat"] if cpd_info["t_stat"] else np.nan
        cpd_dates[cc] = (d, t)

    return all_ports, cpd_dates, raw["country_names"]


# =========================================================================
#  STEP 2: PLOT (reads from cache — instant)
# =========================================================================

def plot_optimal_portfolios(all_ports, cpd_dates, country_names,
                            start_date=START_DATE, break_date=BREAK_DATE,
                            pdf_path=None):
    """
    Plot cumulative returns + correlation matrices from cached data.

    Parameters
    ----------
    all_ports : dict  {cc: pd.DataFrame with ret_opt, ret_ew, n_factors}
    cpd_dates : dict  {cc: (pd.Timestamp, float)}
    country_names : dict  {cc: str}
    start_date, break_date : str
    pdf_path : Path or None  (defaults to PROJECT_ROOT / cpd_optimal_portfolio.pdf)
    """
    if pdf_path is None:
        pdf_path = PATH["PROJECT_ROOT"] / "cpd_optimal_portfolio.pdf"

    # Wide return matrix
    ret_wide = pd.DataFrame({
        country_names[cc]: port["ret_opt"]
        for cc, port in all_ports.items()
    }).sort_index()

    # Colors
    n = len(all_ports)
    cmap = plt.colormaps.get_cmap("tab20")
    colors = {cc: cmap(i / max(n - 1, 1)) for i, cc in enumerate(all_ports.keys())}

    with PdfPages(pdf_path) as pdf_out:

        # ── PAGE 1: All countries cumulative returns ──────────────────
        fig, ax = plt.subplots(figsize=(16, 9))

        cum_all = []
        for cc, port in all_ports.items():
            cname = country_names[cc]
            cum = (1 + port["ret_opt"]).cumprod()
            cum_all.append(cum)
            lw = 2.5 if cc == "usa" else 1.0
            alpha = 1.0 if cc == "usa" else 0.7
            ax.plot(cum.index, cum.values, color=colors[cc],
                    linewidth=lw, alpha=alpha, label=cname)

        for cc, (cpd_date, cpd_t) in cpd_dates.items():
            if pd.notna(cpd_date):
                ax.axvline(cpd_date, color=colors[cc],
                           linewidth=0.8, alpha=0.4, linestyle=":")

        ax.set_yscale("log")

        # Auto-detect data range and choose readable ticks
        all_vals = np.concatenate([c.values for c in cum_all])
        data_min, data_max = all_vals.min(), all_vals.max()
        # Add 5% padding in log space
        log_min = np.log10(data_min) - 0.05
        log_max = np.log10(data_max) + 0.05
        ax.set_ylim(10 ** log_min, 10 ** log_max)

        # Build ticks within the data range
        # Fine-grained candidates: every 10pp from -50% to +200%
        candidates = [v / 100 + 1 for v in range(-50, 210, 10)]  # 0.50 .. 3.0
        tick_vals = [v for v in candidates
                     if 10 ** log_min <= v <= 10 ** log_max]
        ax.set_yticks(tick_vals)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x * 100 - 100:+,.0f}%"
                                  if x != 1.0 else "0%"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylabel("Cumulative Return", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_title(
            "Optimal Factor Portfolios (Min-Variance, Ledoit-Wolf) "
            "-- All Countries",
            fontsize=14, fontweight="bold", loc="left")
        ax.legend(fontsize=7, ncol=4, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(pd.Timestamp(start_date), ret_wide.index.max())
        ax.tick_params(labelsize=9)

        fig.tight_layout()
        pdf_out.savefig(fig)
        plt.close(fig)

        # ── PAGE 2: Vol-standardized cumulative returns ────────────
        fig, ax = plt.subplots(figsize=(16, 9))

        for cc, port in all_ports.items():
            cname = country_names[cc]
            rets = port["ret_opt"].copy()
            vol = rets.std()
            if vol > 0:
                rets_std = rets / vol
            else:
                rets_std = rets
            cum_std = rets_std.cumsum()
            lw = 2.5 if cc == "usa" else 1.0
            alpha = 1.0 if cc == "usa" else 0.7
            ax.plot(cum_std.index, cum_std.values, color=colors[cc],
                    linewidth=lw, alpha=alpha, label=cname)

        for cc, (cpd_date, cpd_t) in cpd_dates.items():
            if pd.notna(cpd_date):
                ax.axvline(cpd_date, color=colors[cc],
                           linewidth=0.8, alpha=0.4, linestyle=":")

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_ylabel("Cumulative Vol-Standardized Return", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_title(
            "Optimal Factor Portfolios (Vol-Standardized) "
            "-- All Countries",
            fontsize=14, fontweight="bold", loc="left")
        ax.legend(fontsize=7, ncol=4, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(pd.Timestamp(start_date), ret_wide.index.max())
        ax.tick_params(labelsize=9)

        fig.tight_layout()
        pdf_out.savefig(fig)
        plt.close(fig)

        # ── PAGE 3: Rolling mean return (36-month) ─────────────────
        fig, ax = plt.subplots(figsize=(16, 9))
        roll_window = 36

        for cc, port in all_ports.items():
            cname = country_names[cc]
            rolling_mean = port["ret_opt"].rolling(roll_window).mean() * 12
            lw = 2.5 if cc == "usa" else 1.0
            alpha = 1.0 if cc == "usa" else 0.7
            ax.plot(rolling_mean.index, rolling_mean.values * 100,
                    color=colors[cc],
                    linewidth=lw, alpha=alpha, label=cname)

        for cc, (cpd_date, cpd_t) in cpd_dates.items():
            if pd.notna(cpd_date):
                ax.axvline(cpd_date, color=colors[cc],
                           linewidth=0.8, alpha=0.4, linestyle=":")

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"
                                  if x != 0 else "0%"))
        ax.set_ylabel("Annualized Return (36-month rolling avg)", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_title(
            "Optimal Factor Portfolios (36-Month Rolling Mean Return) "
            "-- All Countries",
            fontsize=14, fontweight="bold", loc="left")
        ax.legend(fontsize=7, ncol=4, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(pd.Timestamp(start_date), ret_wide.index.max())
        ax.tick_params(labelsize=9)

        fig.tight_layout()
        pdf_out.savefig(fig)
        plt.close(fig)

        # ── PAGE 4: Correlation matrix PRE-2000 ──────────────────────
        pre = ret_wide[ret_wide.index < break_date].dropna(axis=1, how="all")
        pre = pre.dropna(axis=1, thresh=24)
        corr_pre = pre.corr()

        fig, ax = plt.subplots(figsize=(14, 11))
        im = ax.imshow(corr_pre.values, cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="equal")
        ax.set_xticks(range(len(corr_pre.columns)))
        ax.set_yticks(range(len(corr_pre.columns)))
        ax.set_xticklabels(corr_pre.columns, rotation=45, ha="right",
                           fontsize=8)
        ax.set_yticklabels(corr_pre.columns, fontsize=8)

        for i in range(len(corr_pre)):
            for j in range(len(corr_pre)):
                val = corr_pre.iloc[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

        ax.set_title(
            f"Correlation of Optimal Portfolio Returns -- "
            f"Pre-2000  ({pre.index.min().strftime('%Y-%m')} to "
            f"{pre.index.max().strftime('%Y-%m')})",
            fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
        fig.tight_layout()
        pdf_out.savefig(fig)
        plt.close(fig)

        # ── PAGE 5: Correlation matrix POST-2000 ─────────────────────
        post = ret_wide[ret_wide.index >= break_date].dropna(axis=1,
                                                              how="all")
        post = post.dropna(axis=1, thresh=24)
        corr_post = post.corr()

        fig, ax = plt.subplots(figsize=(14, 11))
        im = ax.imshow(corr_post.values, cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="equal")
        ax.set_xticks(range(len(corr_post.columns)))
        ax.set_yticks(range(len(corr_post.columns)))
        ax.set_xticklabels(corr_post.columns, rotation=45, ha="right",
                           fontsize=8)
        ax.set_yticklabels(corr_post.columns, fontsize=8)

        for i in range(len(corr_post)):
            for j in range(len(corr_post)):
                val = corr_post.iloc[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

        ax.set_title(
            f"Correlation of Optimal Portfolio Returns -- "
            f"Post-2000  ({post.index.min().strftime('%Y-%m')} to "
            f"{post.index.max().strftime('%Y-%m')})",
            fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
        fig.tight_layout()
        pdf_out.savefig(fig)
        plt.close(fig)

    print(f"[opt_port] Saved -> {pdf_path}")


# =========================================================================
#  CLI
# =========================================================================

if __name__ == "__main__":
    import sys

    plot_only = "--plot-only" in sys.argv

    if plot_only and CACHE_PATH.exists():
        print("[opt_port] Loading from cache (--plot-only) ...")
        all_ports, cpd_dates, country_names = load_cached_portfolios()
    else:
        if plot_only:
            print("[opt_port] No cache found, computing from scratch ...")
        compute_all_portfolios()
        all_ports, cpd_dates, country_names = load_cached_portfolios()

    plot_optimal_portfolios(all_ports, cpd_dates, country_names)
    print("[opt_port] Done.")
