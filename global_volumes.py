"""
Global volumes report: DM and EM volume ratios vs max-Sharpe portfolios.

Computes two aggregate volume ratios (2Y rolling average):
  - Developed Markets (ex-US) / US
  - Emerging Markets / US
using iShares country ETF dollar volumes from Yahoo Finance.

Correlates each ratio with every country's max-Sharpe 2Y rolling Sharpe,
monthly return, and 12M rolling return.

Output PDF (global_volumes.pdf):
  Page 1:  Heatmap -- DM (ex-US) / US ratio correlations (ETF countries)
  Page 2:  World map -- DM (ex-US) / US ratio, Sharpe (all countries)
  Page 3:  Heatmap -- EM / US ratio correlations (ETF countries)
  Page 4:  World map -- EM / US ratio, Sharpe (all countries)
  Pages 5+: Per-country 2-panel charts (ETF countries)

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
START_DATE = "1970-01-01"

# iShares country ETFs -- ticker : (JKP code, label)
COUNTRY_ETFS = {
    # Developed Markets
    'SPY':  ('usa', 'United States'),
    'EWJ':  ('jpn', 'Japan'),
    'EWH':  ('hkg', 'Hong Kong'),
    'EWA':  ('aus', 'Australia'),
    'EWU':  ('gbr', 'United Kingdom'),
    'EWD':  ('swe', 'Sweden'),
    'EWG':  ('deu', 'Germany'),
    'EWQ':  ('fra', 'France'),
    'EWC':  ('can', 'Canada'),
    'EWS':  ('sgp', 'Singapore'),
    'EWI':  ('ita', 'Italy'),
    'EWP':  ('esp', 'Spain'),
    # Emerging Markets
    'EWY':  ('kor', 'South Korea'),
    'EWT':  ('twn', 'Taiwan'),
    'EWM':  ('mys', 'Malaysia'),
    'FXI':  ('chn', 'China'),
    'INDA': ('ind', 'India'),
    'THD':  ('tha', 'Thailand'),
    'EIDO': ('idn', 'Indonesia'),
    'EPOL': ('pol', 'Poland'),
    'TUR':  ('tur', 'Turkiye'),
    'VNM':  ('vnm', 'Vietnam'),
    'GREK': ('grc', 'Greece'),
}

US_TICKER = "SPY"

# MSCI classification (for ETF-based volume split)
MARKET_CLASS = {
    'SPY': 'DM', 'EWJ': 'DM', 'EWH': 'DM', 'EWA': 'DM', 'EWU': 'DM',
    'EWD': 'DM', 'EWG': 'DM', 'EWQ': 'DM', 'EWC': 'DM', 'EWS': 'DM',
    'EWI': 'DM', 'EWP': 'DM',
    'EWY': 'EM', 'EWT': 'EM', 'EWM': 'EM', 'FXI': 'EM', 'INDA': 'EM',
    'THD': 'EM', 'EIDO': 'EM', 'EPOL': 'EM', 'TUR': 'EM', 'VNM': 'EM',
    'GREK': 'EM',
}

# Full country name lookup (all 54 JKP countries)
COUNTRY_NAMES = {
    'are': 'UAE', 'arg': 'Argentina', 'aus': 'Australia', 'aut': 'Austria',
    'bel': 'Belgium', 'bgd': 'Bangladesh', 'bra': 'Brazil', 'can': 'Canada',
    'che': 'Switzerland', 'chl': 'Chile', 'chn': 'China', 'col': 'Colombia',
    'deu': 'Germany', 'dnk': 'Denmark', 'egy': 'Egypt', 'esp': 'Spain',
    'fin': 'Finland', 'fra': 'France', 'gbr': 'United Kingdom',
    'grc': 'Greece', 'hkg': 'Hong Kong', 'idn': 'Indonesia',
    'ind': 'India', 'irl': 'Ireland', 'isr': 'Israel', 'ita': 'Italy',
    'jpn': 'Japan', 'kor': 'South Korea', 'kwt': 'Kuwait',
    'mar': 'Morocco', 'mex': 'Mexico', 'mys': 'Malaysia',
    'nga': 'Nigeria', 'nld': 'Netherlands', 'nor': 'Norway',
    'nzl': 'New Zealand', 'omn': 'Oman', 'pak': 'Pakistan',
    'per': 'Peru', 'phl': 'Philippines', 'pol': 'Poland',
    'prt': 'Portugal', 'qat': 'Qatar', 'rus': 'Russia',
    'sau': 'Saudi Arabia', 'sgp': 'Singapore', 'swe': 'Sweden',
    'tha': 'Thailand', 'tur': 'Turkiye', 'twn': 'Taiwan',
    'usa': 'United States', 'ven': 'Venezuela', 'vnm': 'Vietnam',
    'zaf': 'South Africa',
}

CONTINENT = {
    # Americas
    'United States': 'Americas', 'Canada': 'Americas',
    'Argentina': 'Americas', 'Brazil': 'Americas', 'Chile': 'Americas',
    'Colombia': 'Americas', 'Mexico': 'Americas', 'Peru': 'Americas',
    'Venezuela': 'Americas',
    # Europe
    'United Kingdom': 'Europe', 'Germany': 'Europe', 'France': 'Europe',
    'Sweden': 'Europe', 'Poland': 'Europe', 'Turkiye': 'Europe',
    'Italy': 'Europe', 'Spain': 'Europe', 'Greece': 'Europe',
    'Austria': 'Europe', 'Belgium': 'Europe', 'Denmark': 'Europe',
    'Finland': 'Europe', 'Ireland': 'Europe', 'Netherlands': 'Europe',
    'Norway': 'Europe', 'Portugal': 'Europe', 'Russia': 'Europe',
    'Switzerland': 'Europe',
    # Asia-Pacific
    'Japan': 'Asia-Pacific', 'China': 'Asia-Pacific',
    'India': 'Asia-Pacific', 'South Korea': 'Asia-Pacific',
    'Hong Kong': 'Asia-Pacific', 'Taiwan': 'Asia-Pacific',
    'Australia': 'Asia-Pacific', 'Malaysia': 'Asia-Pacific',
    'Singapore': 'Asia-Pacific', 'Thailand': 'Asia-Pacific',
    'Indonesia': 'Asia-Pacific', 'Vietnam': 'Asia-Pacific',
    'Bangladesh': 'Asia-Pacific', 'New Zealand': 'Asia-Pacific',
    'Pakistan': 'Asia-Pacific', 'Philippines': 'Asia-Pacific',
    # Middle East & Africa
    'UAE': 'Middle East & Africa', 'Egypt': 'Middle East & Africa',
    'Israel': 'Middle East & Africa', 'Kuwait': 'Middle East & Africa',
    'Morocco': 'Middle East & Africa', 'Nigeria': 'Middle East & Africa',
    'Oman': 'Middle East & Africa', 'Qatar': 'Middle East & Africa',
    'Saudi Arabia': 'Middle East & Africa',
    'South Africa': 'Middle East & Africa',
}

CONTINENT_ORDER = ['Americas', 'Europe', 'Asia-Pacific',
                   'Middle East & Africa']

# JKP country name -> Natural Earth ADMIN name
NAME_TO_NE = {
    'United States': 'United States of America',
    'South Korea': 'South Korea',
    'Turkiye': 'Turkey',
    'UAE': 'United Arab Emirates',
    'Russia': 'Russia',
}

# Coordinates for text labels on larger countries
LABEL_COORDS = {
    'United States': (-100, 40), 'Canada': (-100, 55),
    'Argentina': (-64, -35), 'Brazil': (-52, -12),
    'Chile': (-71, -33), 'Colombia': (-73, 4),
    'Mexico': (-102, 24), 'Peru': (-75, -10),
    'Venezuela': (-66, 8),
    'United Kingdom': (-2, 54), 'Germany': (10, 51),
    'France': (2, 47), 'Sweden': (16, 62),
    'Poland': (20, 52), 'Turkiye': (35, 39),
    'Italy': (12, 43), 'Spain': (-4, 40), 'Greece': (22, 39),
    'Austria': (14, 47), 'Belgium': (4, 50.5),
    'Denmark': (10, 56), 'Finland': (26, 64),
    'Ireland': (-8, 53), 'Netherlands': (5, 52),
    'Norway': (10, 62), 'Portugal': (-8, 39.5),
    'Russia': (90, 60), 'Switzerland': (8, 47),
    'Japan': (138, 37), 'China': (105, 35),
    'India': (79, 22), 'South Korea': (128, 36),
    'Australia': (134, -25), 'Indonesia': (118, -3),
    'Thailand': (101, 15), 'Vietnam': (107, 16),
    'Pakistan': (69, 30), 'Philippines': (122, 12),
    'Bangladesh': (90, 24), 'New Zealand': (173, -42),
    'Egypt': (30, 27), 'Morocco': (-6, 32),
    'Nigeria': (8, 10), 'South Africa': (25, -30),
    'Saudi Arabia': (45, 24), 'Israel': (35, 31),
}

# Small countries that need arrow annotations
SMALL_ANNOTATIONS = {
    'Hong Kong': (114.2, 22.3),
    'Singapore': (103.8, 1.35),
    'Taiwan': (121.0, 23.7),
    'Kuwait': (47.5, 29.3),
    'Qatar': (51.2, 25.3),
    'Oman': (57.5, 21.5),
    'Malaysia': (109, 3),
}


# =========================================================================
#  DATA
# =========================================================================

def fetch_volume_ratios() -> tuple[pd.Series, pd.Series]:
    """
    Compute DM (ex-US) / US and EM / US dollar volume ratios (2Y rolling avg).

    Returns (dm_ratio, em_ratio).
    """
    tickers = list(COUNTRY_ETFS.keys())
    print(f"  Downloading {len(tickers)} ETFs from Yahoo Finance ...")

    raw = yf.download(tickers, start=START, end=END,
                      interval="1mo", progress=False)

    close = raw["Close"]
    volume = raw["Volume"]
    dollar_vol = close * volume

    us_vol = dollar_vol[US_TICKER]

    dm_tickers = [t for t, c in MARKET_CLASS.items()
                  if c == "DM" and t != US_TICKER]
    em_tickers = [t for t, c in MARKET_CLASS.items() if c == "EM"]

    # DM (ex-US)
    dm_available = [t for t in dm_tickers if t in dollar_vol.columns]
    dm_vol = dollar_vol[dm_available].sum(axis=1, min_count=1)
    dm_ratio_raw = (dm_vol / us_vol).dropna()
    dm_ratio = dm_ratio_raw.rolling(ROLLING_VOL_WINDOW).mean().dropna()
    dm_ratio.index = dm_ratio.index.to_period("M").to_timestamp("M")

    dm_names = sorted(COUNTRY_ETFS[t][1] for t in dm_available)
    print(f"  DM (ex-US): {len(dm_available)} ETFs "
          f"({', '.join(dm_names)})")
    print(f"    {dm_ratio.index.min().strftime('%Y-%m')} to "
          f"{dm_ratio.index.max().strftime('%Y-%m')} ({len(dm_ratio)} months)")

    # EM
    em_available = [t for t in em_tickers if t in dollar_vol.columns]
    em_vol = dollar_vol[em_available].sum(axis=1, min_count=1)
    em_ratio_raw = (em_vol / us_vol).dropna()
    em_ratio = em_ratio_raw.rolling(ROLLING_VOL_WINDOW).mean().dropna()
    em_ratio.index = em_ratio.index.to_period("M").to_timestamp("M")

    em_names = sorted(COUNTRY_ETFS[t][1] for t in em_available)
    print(f"  EM: {len(em_available)} ETFs "
          f"({', '.join(em_names)})")
    print(f"    {em_ratio.index.min().strftime('%Y-%m')} to "
          f"{em_ratio.index.max().strftime('%Y-%m')} ({len(em_ratio)} months)")

    return dm_ratio, em_ratio


def load_portfolios():
    """Load cached max-Sharpe portfolio returns for all countries."""
    df = pd.read_parquet(CACHE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return {loc: grp.set_index("date").sort_index()
            for loc, grp in df.groupby("location")}


def get_etf_countries(portfolios):
    """Return JKP codes that have both a portfolio and an ETF (for heatmaps)."""
    etf_codes = {v[0] for v in COUNTRY_ETFS.values()}
    return [loc for loc in portfolios if loc in etf_codes and loc != "usa"]


def get_all_countries(portfolios):
    """Return all JKP codes except USA (for world maps)."""
    return [loc for loc in portfolios if loc != "usa"]


def compute_correlations(ratio, portfolios, countries, ratio_label):
    """
    Compute correlations between a volume ratio and each country's
    max-Sharpe 2Y Sharpe, monthly return, and 12M rolling return.
    """
    rows = []
    country_ports = {}

    for loc in countries:
        label = COUNTRY_NAMES.get(loc, loc.upper())
        port = portfolios[loc]

        common = ratio.index.intersection(port.index)
        if len(common) < 24:
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

def _plot_heatmap(results: pd.DataFrame, ratio_label: str, pdf) -> None:
    """Heatmap of correlations grouped by continent."""
    metrics = ["Corr w/ 2Y Sharpe", "Corr w/ Monthly Ret", "Corr w/ 12M Ret"]

    results = results.copy()
    results["continent"] = results.index.map(CONTINENT)
    present = [c for c in CONTINENT_ORDER
               if (results["continent"] == c).any()]

    n_groups = len(present)
    counts = [max(1, sum(results["continent"] == c)) for c in present]
    fig, axes = plt.subplots(n_groups, 1, figsize=(8, 14),
                             gridspec_kw={"height_ratios": counts})
    if n_groups == 1:
        axes = [axes]

    for ax_idx, cont in enumerate(present):
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
        if ax_idx == n_groups - 1:
            ax.set_xticklabels(metrics, fontsize=9)
        else:
            ax.set_xticklabels([])
        ax.set_yticks(range(n))
        ax.set_yticklabels(data.index, fontsize=9)

        ax.set_ylabel(cont, fontsize=11, fontweight="bold", rotation=0,
                      labelpad=80, va="center")

    fig.suptitle(
        f"Corr({ratio_label} / US Volume, Max-Sharpe Performance)",
        fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.5, 0.005, "* p<0.10   ** p<0.05   *** p<0.01",
             ha="center", fontsize=9, style="italic")

    cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.03)
    cbar.set_label("Pearson Correlation", fontsize=9)

    fig.subplots_adjust(top=0.93, bottom=0.05, hspace=0.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_world_map(results: pd.DataFrame, ratio_label: str,
                    metric: str, metric_label: str,
                    pdf, *, world: gpd.GeoDataFrame) -> None:
    """World choropleth colored by a given metric."""
    ne_corr = {}
    corr_by_country = {}
    for country in results.index:
        val = results.loc[country, metric]
        if np.isnan(val):
            continue
        corr_by_country[country] = val
        ne_name = NAME_TO_NE.get(country, country)
        ne_corr[ne_name] = val

    w = world.copy()
    w["corr"] = w["ADMIN"].map(ne_corr)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Base map
    w.plot(ax=ax, color="#e0e0e0", edgecolor="white", linewidth=0.3)

    # Countries with data
    has_data = w.dropna(subset=["corr"])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    has_data.plot(ax=ax, column="corr", cmap="RdBu_r", norm=norm,
                  edgecolor="white", linewidth=0.5, legend=False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, aspect=25)
    cbar.set_label(f"Corr with {ratio_label} / US Volume Ratio",
                   fontsize=10)

    # Small countries: arrow annotations
    for country, (lon, lat) in SMALL_ANNOTATIONS.items():
        if country in corr_by_country:
            val = corr_by_country[country]
            ax.annotate(
                f"{country}\n{val:+.2f}",
                xy=(lon, lat),
                xytext=(lon + 15, lat - 10),
                fontsize=6, ha="center",
                arrowprops=dict(arrowstyle="->", color="black",
                                linewidth=0.7),
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="gray", alpha=0.8),
            )

    # Larger countries: text labels
    for country, (lon, lat) in LABEL_COORDS.items():
        if country in corr_by_country and country not in SMALL_ANNOTATIONS:
            val = corr_by_country[country]
            color = "white" if abs(val) > 0.45 else "black"
            ax.text(lon, lat, f"{val:+.2f}", ha="center", va="center",
                    fontsize=6, fontweight="bold", color=color)

    ax.set_title(
        f"Corr({ratio_label} / US Volume Ratio, {metric_label})",
        fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(-170, 180)
    ax.set_ylim(-60, 85)
    ax.set_axis_off()

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_country(country_label: str, ratio: pd.Series,
                  ratio_label: str, port: pd.DataFrame, pdf) -> None:
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

    vol_label = f"{ratio_label} / US Volume Ratio"

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Panel 1: ratio + cumret
    ax1 = axes[0]
    ln1 = ax1.plot(ratio_c.index, ratio_c.values, color=c_vol, linewidth=1.5,
                   label=vol_label)
    ax1.set_ylabel(vol_label, fontsize=10, color=c_vol)
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
    ax1.set_title(f"{country_label}: {ratio_label}/US Volume Ratio "
                  f"vs Max-Sharpe Portfolio",
                  fontsize=12, fontweight="bold", loc="left", pad=18)
    ax1.grid(True, alpha=0.2)

    # Panel 2: ratio + rolling Sharpe
    ax2 = axes[1]
    ln3 = ax2.plot(ratio_c.index, ratio_c.values, color=c_vol, linewidth=1.5,
                   label=vol_label)
    ax2.set_ylabel(vol_label, fontsize=10, color=c_vol)
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
    print("Global Volumes Report (DM / EM split)")
    print("=" * 60)

    # 1. Load portfolios
    print("\n[1/4] Loading portfolios ...")
    portfolios = load_portfolios()
    etf_countries = get_etf_countries(portfolios)
    all_countries = get_all_countries(portfolios)
    print(f"  {len(all_countries)} total countries, "
          f"{len(etf_countries)} with ETFs")

    # 2. Compute volume ratios
    print("\n[2/4] Computing DM and EM volume ratios ...")
    dm_ratio, em_ratio = fetch_volume_ratios()

    # 3. Correlations -- all countries (for world maps)
    print("\n[3/4] Computing correlations ...")

    print("\n  --- DM (ex-US) / US  [all countries] ---")
    dm_results_all, dm_ports = compute_correlations(
        dm_ratio, portfolios, all_countries, "DM (ex-US)")
    print(f"\n  {len(dm_results_all)} countries")

    print("\n  --- EM / US  [all countries] ---")
    em_results_all, em_ports = compute_correlations(
        em_ratio, portfolios, all_countries, "EM")
    print(f"\n  {len(em_results_all)} countries")

    # Subset for heatmaps (ETF countries only)
    etf_labels = {COUNTRY_NAMES[loc] for loc in etf_countries}
    dm_results_etf = dm_results_all[dm_results_all.index.isin(etf_labels)]
    em_results_etf = em_results_all[em_results_all.index.isin(etf_labels)]

    # 4. Generate PDF
    print("\n[4/4] Generating PDF ...")

    ne_url = ("https://naciscdn.org/naturalearth/110m/cultural/"
              "ne_110m_admin_0_countries.zip")
    print("  Loading world shapefile ...")
    world = gpd.read_file(ne_url)

    all_ports = {**dm_ports, **em_ports}

    # JKP code -> market class
    jkp_class = {}
    for ticker, (jkp, _) in COUNTRY_ETFS.items():
        jkp_class[jkp] = MARKET_CLASS.get(ticker, "EM")

    with PdfPages(PDF_PATH) as pdf:
        # DM block
        _plot_heatmap(dm_results_etf, "DM (ex-US)", pdf)
        print("  World map (DM, Sharpe) ...")
        _plot_world_map(dm_results_all, "DM (ex-US)",
                        metric="Corr w/ 2Y Sharpe",
                        metric_label="2Y Rolling Sharpe",
                        pdf=pdf, world=world)

        # EM block
        _plot_heatmap(em_results_etf, "EM", pdf)
        print("  World map (EM, Sharpe) ...")
        _plot_world_map(em_results_all, "EM",
                        metric="Corr w/ 2Y Sharpe",
                        metric_label="2Y Rolling Sharpe",
                        pdf=pdf, world=world)

        # Per-country charts (ETF countries only)
        for cont in CONTINENT_ORDER:
            dm_sub = dm_results_etf[dm_results_etf["continent"] == cont]
            for label in dm_sub.index:
                loc = next(k for k, v in all_ports.items()
                           if v[0] == label)
                _, port = all_ports[loc]
                cls = jkp_class.get(loc, "EM")
                ratio = dm_ratio if cls == "DM" else em_ratio
                r_label = "DM (ex-US)" if cls == "DM" else "EM"
                _plot_country(label, ratio, r_label, port, pdf)

            em_sub = em_results_etf[em_results_etf["continent"] == cont]
            for label in em_sub.index:
                if label in dm_sub.index:
                    continue
                loc = next(k for k, v in all_ports.items()
                           if v[0] == label)
                _, port = all_ports[loc]
                cls = jkp_class.get(loc, "EM")
                ratio = dm_ratio if cls == "DM" else em_ratio
                r_label = "DM (ex-US)" if cls == "DM" else "EM"
                _plot_country(label, ratio, r_label, port, pdf)

    print(f"Saved -> {PDF_PATH}")

    # Summary
    for tag, res in [("DM (ex-US) / US", dm_results_all),
                     ("EM / US", em_results_all)]:
        print(f"\n{tag}:")
        for cont in CONTINENT_ORDER:
            sub = res[res["continent"] == cont]
            if sub.empty:
                continue
            print(f"\n  {cont}")
            print(f"  {'Country':<20s} {'Sharpe':>8s} {'MonthRet':>8s} "
                  f"{'12M Ret':>8s} {'n':>5s}")
            print(f"  {'-' * 50}")
            for _, row in sub.iterrows():
                print(f"  {row.name:<20s} "
                      f"{row['Corr w/ 2Y Sharpe']:+8.3f} "
                      f"{row['Corr w/ Monthly Ret']:+8.3f} "
                      f"{row['Corr w/ 12M Ret']:+8.3f} {row['n']:5.0f}")

    print("\nDone.")
