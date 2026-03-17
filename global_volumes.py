"""
Global volumes report: non-US / US trading volume ratio vs max-Sharpe portfolios.

Computes aggregate non-US / US dollar volume ratio (2Y rolling average,
from iShares ETFs on Yahoo Finance), then correlates with each country's
max-Sharpe 2Y rolling Sharpe, monthly return, and 12M rolling return.

Output PDF (global_volumes.pdf):
  Page 1:  Heatmap of correlations grouped by continent
  Page 2:  World map color-coded by 12M return correlation
  Pages 3+: Per-country 2-panel charts (ratio vs cumret, ratio vs Sharpe)

Usage:
    .venv/Scripts/python global_volumes.py
"""

from config import PATH
from data import load_jkp_factor_returns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr
import geopandas as gpd
import yfinance as yf

# =========================================================================
#  CONSTANTS
# =========================================================================

CACHE_PATH = PATH["INTERMEDIARY_RESULTS"] / "max_sharpe_all_countries.parquet"
PDF_PATH = PATH["PROJECT_ROOT"] / "global_volumes.pdf"
START = "1996-03-01"
END = "2024-12-31"
ROLLING_SHARPE_WINDOW = 24
ROLLING_VOL_WINDOW = 24
TOP_N = 20
START_DATE = "1970-01-01"

# iShares country ETFs -- ticker : (JKP code, label)
COUNTRY_ETFS = {
    'SPY':  ('usa', 'United States'),
    'EWJ':  ('jpn', 'Japan'),
    'EWY':  ('kor', 'South Korea'),
    'EWH':  ('hkg', 'Hong Kong'),
    'EWT':  ('twn', 'Taiwan'),
    'EWA':  ('aus', 'Australia'),
    'EWU':  ('gbr', 'United Kingdom'),
    'EWM':  ('mys', 'Malaysia'),
    'EWD':  ('swe', 'Sweden'),
    'EWG':  ('deu', 'Germany'),
    'EWQ':  ('fra', 'France'),
    'EWC':  ('can', 'Canada'),
    'EWS':  ('sgp', 'Singapore'),
    'FXI':  ('chn', 'China'),
    'INDA': ('ind', 'India'),
    'THD':  ('tha', 'Thailand'),
    'EIDO': ('idn', 'Indonesia'),
    'EPOL': ('pol', 'Poland'),
    'TUR':  ('tur', 'Turkiye'),
    'VNM':  ('vnm', 'Vietnam'),
}

US_TICKER = "SPY"

COUNTRY_NAMES = {
    'usa': 'United States', 'jpn': 'Japan', 'chn': 'China',
    'ind': 'India', 'twn': 'Taiwan', 'gbr': 'United Kingdom',
    'kor': 'South Korea', 'hkg': 'Hong Kong', 'aus': 'Australia',
    'can': 'Canada', 'mys': 'Malaysia', 'deu': 'Germany',
    'fra': 'France', 'vnm': 'Vietnam', 'pol': 'Poland',
    'tha': 'Thailand', 'sgp': 'Singapore', 'swe': 'Sweden',
    'idn': 'Indonesia', 'tur': 'Turkiye',
}

CONTINENT = {
    'United States': 'Americas', 'Canada': 'Americas',
    'United Kingdom': 'Europe', 'Germany': 'Europe',
    'France': 'Europe', 'Sweden': 'Europe',
    'Poland': 'Europe', 'Turkiye': 'Europe',
    'Japan': 'Asia-Pacific', 'China': 'Asia-Pacific',
    'India': 'Asia-Pacific', 'South Korea': 'Asia-Pacific',
    'Hong Kong': 'Asia-Pacific', 'Taiwan': 'Asia-Pacific',
    'Australia': 'Asia-Pacific', 'Malaysia': 'Asia-Pacific',
    'Singapore': 'Asia-Pacific', 'Thailand': 'Asia-Pacific',
    'Indonesia': 'Asia-Pacific', 'Vietnam': 'Asia-Pacific',
}

CONTINENT_ORDER = ['Americas', 'Europe', 'Asia-Pacific']

# Natural Earth country-name mapping
NAME_TO_NE = {
    'United States': 'United States of America',
    'South Korea': 'South Korea',
    'Turkiye': 'Turkey',
}


# =========================================================================
#  DATA
# =========================================================================

def fetch_volume_ratio() -> pd.Series:
    """
    Compute non-US / US dollar volume ratio (2Y rolling average).

    Downloads monthly ETF data from Yahoo Finance, filters to complete
    series, computes aggregate non-US dollar volume / US dollar volume,
    and smooths with a 2Y rolling average.
    """
    tickers = list(COUNTRY_ETFS.keys())
    print(f"  Downloading {len(tickers)} ETFs from Yahoo Finance ...")

    raw = yf.download(tickers, start=START, end=END,
                      interval="1mo", progress=False)

    close = raw["Close"]
    volume = raw["Volume"]
    dollar_vol = close * volume

    # Filter to complete series
    cutoff = pd.Timestamp(START)
    complete = dollar_vol.dropna(axis=1, how="all")
    ok_cols = []
    for col in complete.columns:
        s = complete[col].dropna()
        if len(s) > 0 and s.index.min() <= cutoff + pd.DateOffset(months=2):
            ok_cols.append(col)

    dropped = set(complete.columns) - set(ok_cols)
    if dropped:
        names = sorted(COUNTRY_ETFS[t][1] for t in dropped
                       if t in COUNTRY_ETFS)
        print(f"  Dropped (start after {START}): {', '.join(names)}")

    dv = complete.loc[complete.index >= cutoff, ok_cols].dropna()
    kept = sorted(COUNTRY_ETFS[t][1] for t in dv.columns
                  if t in COUNTRY_ETFS)
    print(f"  Kept {len(dv.columns)} markets from "
          f"{dv.index.min().strftime('%Y-%m')} ({', '.join(kept)})")

    # Compute ratio
    non_us = dv.drop(columns=[US_TICKER], errors="ignore").sum(axis=1)
    us = dv[US_TICKER]
    ratio = (non_us / us).rolling(ROLLING_VOL_WINDOW).mean().dropna()
    ratio.index = ratio.index.to_period("M").to_timestamp("M")

    print(f"  Ratio: {ratio.index.min().strftime('%Y-%m')} to "
          f"{ratio.index.max().strftime('%Y-%m')} ({len(ratio)} months)")
    return ratio


def load_portfolios():
    """Load cached max-Sharpe portfolio returns for all countries."""
    df = pd.read_parquet(CACHE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return {loc: grp.set_index("date").sort_index()
            for loc, grp in df.groupby("location")}


def select_top_countries(portfolios, n=TOP_N):
    """Select top N countries by stock count."""
    fr = load_jkp_factor_returns(weighting="vw_cap")
    fr = fr[fr["date"] >= START_DATE].copy()
    latest = fr["date"].max() - pd.DateOffset(years=1)
    recent = fr[fr["date"] >= latest]
    proxy = recent.groupby("location")["n_stocks"].max()
    proxy = proxy[proxy.index.isin(portfolios.keys())]
    return proxy.nlargest(n).index.tolist()


def compute_correlations(ratio, portfolios, top20):
    """
    Compute correlations between the volume ratio and each country's
    max-Sharpe 2Y Sharpe, monthly return, and 12M rolling return.

    Returns
    -------
    results : pd.DataFrame
        Indexed by country name with correlation columns and p-values.
    country_ports : dict
        {jkp_code: (label, port_df)} for per-country plotting.
    """
    rows = []
    country_ports = {}

    for loc in top20:
        label = COUNTRY_NAMES.get(loc, loc.upper())
        port = portfolios[loc]

        common = ratio.index.intersection(port.index)
        if len(common) < 24:
            print(f"  [{label}] SKIPPED (only {len(common)} overlap months)")
            continue

        ratio_c = ratio.reindex(common)
        port_c = port.reindex(common)

        # 2Y rolling Sharpe
        rm = port_c["ret_opt"].rolling(ROLLING_SHARPE_WINDOW).mean()
        rs = port_c["ret_opt"].rolling(ROLLING_SHARPE_WINDOW).std()
        sharpe = ((rm / rs) * np.sqrt(12)).dropna()
        common_sh = ratio_c.index.intersection(sharpe.index)

        # 12M rolling return
        annual = port_c["ret_opt"].rolling(12).apply(
            lambda x: (1 + x).prod() - 1, raw=True).dropna()
        common_ann = ratio_c.index.intersection(annual.index)

        # Correlations
        if len(common_sh) >= 24:
            r_sh, p_sh = pearsonr(ratio_c[common_sh], sharpe[common_sh])
        else:
            r_sh, p_sh = np.nan, np.nan

        r_ret, p_ret = pearsonr(ratio_c[common], port_c.loc[common, "ret_opt"])

        if len(common_ann) >= 24:
            r_ann, p_ann = pearsonr(ratio_c[common_ann], annual[common_ann])
        else:
            r_ann, p_ann = np.nan, np.nan

        print(f"  [{label:<20s}] {len(common):3d} months  "
              f"Sharpe={r_sh:+.2f}  Ret={r_ret:+.2f}  12M={r_ann:+.2f}")

        rows.append({
            "Country": label,
            "Corr w/ 2Y Sharpe": r_sh,
            "Corr w/ Monthly Ret": r_ret,
            "Corr w/ 12M Ret": r_ann,
            "p_sharpe": p_sh,
            "p_ret": p_ret,
            "p_ann": p_ann,
            "n": len(common),
        })
        country_ports[loc] = (label, port)

    results = pd.DataFrame(rows).set_index("Country")

    # Sort by continent then abs Sharpe
    results["continent"] = results.index.map(CONTINENT)
    continent_cat = pd.CategoricalDtype(
        categories=CONTINENT_ORDER, ordered=True)
    results["continent"] = results["continent"].astype(continent_cat)
    results = results.sort_values(
        ["continent", "Corr w/ 2Y Sharpe"],
        ascending=[True, False],
        key=lambda x: x.abs() if x.name == "Corr w/ 2Y Sharpe" else x,
    )

    return results, country_ports


# =========================================================================
#  PLOTTING
# =========================================================================

def _plot_heatmap(results: pd.DataFrame, pdf) -> None:
    """Page 1: heatmap of correlations grouped by continent."""
    metrics = ["Corr w/ 2Y Sharpe", "Corr w/ Monthly Ret", "Corr w/ 12M Ret"]

    results = results.copy()
    results["continent"] = results.index.map(CONTINENT)

    n_continents = len(CONTINENT_ORDER)
    fig, axes = plt.subplots(n_continents, 1, figsize=(8, 14),
                             gridspec_kw={"height_ratios": [
                                 max(1, sum(results["continent"] == c))
                                 for c in CONTINENT_ORDER
                             ]})

    for ax_idx, cont in enumerate(CONTINENT_ORDER):
        ax = axes[ax_idx]
        sub = results[results["continent"] == cont].copy()
        sub = sub.sort_values("Corr w/ 2Y Sharpe",
                              key=lambda x: x.abs(), ascending=False)
        data = sub[metrics]

        n = len(data)
        if n == 0:
            ax.set_visible(False)
            continue

        im = ax.imshow(data.values, cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="auto")

        pcols = ["p_sharpe", "p_ret", "p_ann"]
        for i in range(n):
            for j in range(len(metrics)):
                val = data.iloc[i, j]
                pval = sub[pcols[j]].iloc[i]
                if np.isnan(val):
                    txt = "--"
                    color = "gray"
                else:
                    stars = ("***" if pval < 0.01
                             else "**" if pval < 0.05
                             else "*" if pval < 0.10 else "")
                    txt = f"{val:.2f}{stars}"
                    color = "white" if abs(val) > 0.45 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color)

        ax.set_xticks(range(len(metrics)))
        if ax_idx == n_continents - 1:
            ax.set_xticklabels(metrics, fontsize=9)
        else:
            ax.set_xticklabels([])
        ax.set_yticks(range(n))
        ax.set_yticklabels(data.index, fontsize=9)

        ax.set_ylabel(cont, fontsize=11, fontweight="bold", rotation=0,
                      labelpad=80, va="center")

    fig.suptitle(
        "Corr(Non-US/US Volume Ratio, Max-Sharpe Performance)",
        fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.5, 0.005, "* p<0.10   ** p<0.05   *** p<0.01",
             ha="center", fontsize=9, style="italic")

    cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.03)
    cbar.set_label("Pearson Correlation", fontsize=9)

    fig.subplots_adjust(top=0.93, bottom=0.05, hspace=0.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_world_map(results: pd.DataFrame, pdf) -> None:
    """Page 2: world choropleth colored by 12M return correlation."""
    ne_url = ("https://naciscdn.org/naturalearth/110m/cultural/"
              "ne_110m_admin_0_countries.zip")
    world = gpd.read_file(ne_url)

    # Build lookup
    ne_corr = {}
    corr_by_country = {}
    for country in results.index:
        val = results.loc[country, "Corr w/ 12M Ret"]
        corr_by_country[country] = val
        ne_name = NAME_TO_NE.get(country, country)
        ne_corr[ne_name] = val

    world["corr"] = world["ADMIN"].map(ne_corr)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Base map
    world.plot(ax=ax, color="#e0e0e0", edgecolor="white", linewidth=0.3)

    # Countries with data
    has_data = world.dropna(subset=["corr"])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    has_data.plot(ax=ax, column="corr", cmap="RdBu_r", norm=norm,
                  edgecolor="white", linewidth=0.5, legend=False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=25)
    cbar.set_label("Correlation with Non-US/US Volume Ratio", fontsize=10)

    # Annotate small countries
    annotations = {
        "Hong Kong": (114.2, 22.3),
        "Singapore": (103.8, 1.35),
        "Taiwan": (121.0, 23.7),
    }
    for country, (lon, lat) in annotations.items():
        if country in corr_by_country:
            val = corr_by_country[country]
            ax.annotate(
                f"{country}\n{val:+.2f}",
                xy=(lon, lat),
                xytext=(lon + 15, lat - 10),
                fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="->", color="black",
                                linewidth=0.7),
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="gray", alpha=0.8),
            )

    # Labels on larger countries
    label_coords = {
        "United States": (-100, 40),
        "Canada": (-100, 55),
        "United Kingdom": (-2, 54),
        "Germany": (10, 51),
        "France": (2, 47),
        "Sweden": (16, 62),
        "Poland": (20, 52),
        "Turkiye": (35, 39),
        "Japan": (138, 37),
        "China": (105, 35),
        "India": (79, 22),
        "South Korea": (128, 36),
        "Australia": (134, -25),
        "Malaysia": (109, 3),
        "Thailand": (101, 15),
        "Indonesia": (118, -3),
        "Vietnam": (107, 16),
    }
    for country, (lon, lat) in label_coords.items():
        if country in corr_by_country:
            val = corr_by_country[country]
            color = "white" if abs(val) > 0.45 else "black"
            ax.text(lon, lat, f"{val:+.2f}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)

    ax.set_title(
        "Corr(Non-US/US Volume Ratio, 12M Rolling Return) by Country",
        fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(-170, 180)
    ax.set_ylim(-60, 85)
    ax.set_axis_off()

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_country(country_label: str, ratio: pd.Series,
                  port: pd.DataFrame, pdf) -> None:
    """Two-panel chart for one country."""
    overlap = ratio.index.intersection(port.index)
    if len(overlap) < 24:
        return

    ratio_c = ratio.reindex(overlap)
    port_c = port.reindex(overlap)

    cumret = (1 + port_c["ret_opt"]).cumprod()
    rm = port_c["ret_opt"].rolling(ROLLING_SHARPE_WINDOW).mean()
    rs = port_c["ret_opt"].rolling(ROLLING_SHARPE_WINDOW).std()
    sharpe = ((rm / rs) * np.sqrt(12)).dropna()

    c_vol = "#1f77b4"
    c_cum = "#d62728"
    c_sh = "#2ca02c"

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Panel 1: ratio + cumret
    ax1 = axes[0]
    ln1 = ax1.plot(ratio_c.index, ratio_c.values, color=c_vol, linewidth=1.5,
                   label="Non-US / US Volume Ratio")
    ax1.set_ylabel("Non-US / US Dollar Volume", fontsize=10, color=c_vol)
    ax1.tick_params(axis="y", labelcolor=c_vol)

    ax1r = ax1.twinx()
    ln2 = ax1r.plot(cumret.index, cumret.values, color=c_cum, linewidth=1.5,
                    label=f"{country_label} Cumulative Return")
    ax1r.set_yscale("log")
    vals = cumret.values
    if vals.min() > 0:
        log_lo = np.log10(vals.min()) - 0.05
        log_hi = np.log10(vals.max()) + 0.10
        ticks = [v for v in [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5,
                              7.5, 10, 15, 20, 30, 50]
                 if 10 ** log_lo <= v <= 10 ** log_hi]
        if ticks:
            ax1r.set_yticks(ticks)
        ax1r.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, _: f"{x:.0f}x" if x >= 1 else f"{x:.2f}x"))
        ax1r.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax1r.set_ylabel("Cumulative Return (log)", fontsize=10, color=c_cum)
    ax1r.tick_params(axis="y", labelcolor=c_cum)

    handles = ln1 + ln2
    ax1.legend(handles, [h.get_label() for h in handles],
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, fontsize=9, framealpha=0.9)
    ax1.set_title(f"{country_label}: Non-US/US Volume Ratio "
                  f"vs Max-Sharpe Portfolio",
                  fontsize=12, fontweight="bold", loc="left", pad=18)
    ax1.grid(True, alpha=0.2)

    # Panel 2: ratio + rolling Sharpe
    ax2 = axes[1]
    ln3 = ax2.plot(ratio_c.index, ratio_c.values, color=c_vol, linewidth=1.5,
                   label="Non-US / US Volume Ratio")
    ax2.set_ylabel("Non-US / US Dollar Volume", fontsize=10, color=c_vol)
    ax2.tick_params(axis="y", labelcolor=c_vol)

    ax2r = ax2.twinx()
    ln4 = ax2r.plot(sharpe.index, sharpe.values, color=c_sh, linewidth=1.5,
                    label=f"{country_label} 2Y Rolling Sharpe")
    ax2r.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax2r.set_ylabel("Rolling Sharpe (2Y, ann.)", fontsize=10, color=c_sh)
    ax2r.tick_params(axis="y", labelcolor=c_sh)

    handles = ln3 + ln4
    ax2.legend(handles, [h.get_label() for h in handles],
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Date", fontsize=10)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_minor_locator(mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a in axes:
        a.tick_params(labelsize=9)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
#  CLI
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Global Volumes Report")
    print("=" * 60)

    # 1. Load portfolios & select top 20
    print("\n[1/4] Loading portfolios ...")
    portfolios = load_portfolios()
    top20 = select_top_countries(portfolios, TOP_N)
    print(f"  Top {TOP_N}: {', '.join(c.upper() for c in top20)}")

    # 2. Compute volume ratio
    print("\n[2/4] Computing non-US / US volume ratio ...")
    ratio = fetch_volume_ratio()

    # 3. Correlations for each country
    print("\n[3/4] Computing correlations per country ...")
    results, country_ports = compute_correlations(ratio, portfolios, top20)
    print(f"\n  {len(results)} countries with sufficient data")

    # 4. Generate PDF
    print("\n[4/4] Generating PDF ...")
    with PdfPages(PDF_PATH) as pdf:
        # Page 1: heatmap
        _plot_heatmap(results, pdf)

        # Page 2: world map
        print("  World map ...")
        _plot_world_map(results, pdf)

        # Pages 3+: per-country charts grouped by continent
        for cont in CONTINENT_ORDER:
            sub = results[results["continent"] == cont]
            for label in sub.index:
                loc = next(k for k, v in country_ports.items()
                           if v[0] == label)
                _, port = country_ports[loc]
                _plot_country(label, ratio, port, pdf)

    print(f"Saved -> {PDF_PATH}")

    # Summary table
    print("\nSummary:")
    for cont in CONTINENT_ORDER:
        sub = results[results["continent"] == cont]
        if sub.empty:
            continue
        print(f"\n  {cont}")
        print(f"  {'Country':<20s} {'Sharpe':>8s} {'MonthRet':>8s} "
              f"{'12M Ret':>8s} {'n':>5s}")
        print(f"  {'-' * 50}")
        for _, row in sub.iterrows():
            print(f"  {row.name:<20s} {row['Corr w/ 2Y Sharpe']:+8.3f} "
                  f"{row['Corr w/ Monthly Ret']:+8.3f} "
                  f"{row['Corr w/ 12M Ret']:+8.3f} {row['n']:5.0f}")

    print("\nDone.")
