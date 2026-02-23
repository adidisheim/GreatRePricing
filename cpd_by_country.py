"""
CPD analysis by country vs USA comparison (top 20 by number of stocks).

Produces a single multi-page PDF (cpd_by_country.pdf). Each page shows
8 panels in a 4×2 layout:

    LEFT column (country X):           RIGHT column (USA):
    ┌──────────────────┬──────────────────┐
    │ CPD histogram    │ CPD histogram    │
    ├──────────────────┼──────────────────┤
    │ Spaghetti cumret │ Spaghetti cumret │
    ├──────────────────┼──────────────────┤
    │ Dot plot         │ Dot plot         │
    ├──────────────────┼──────────────────┤
    │ Theme cumret     │ Theme cumret     │
    └──────────────────┴──────────────────┘

Uses vw_cap weighting, post-1970 data, 13-theme classification.

Usage:
    .venv/bin/python cpd_by_country.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from config import PATH
from data import load_jkp_factor_returns
from util_locals.factor_themes import THEME_MAP_13, THEMES_13, THEME_COLORS_13
import shutil


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

START_DATE = "1970-01-01"
THEME_COLORS = THEME_COLORS_13
themes_ordered = THEMES_13

COUNTRY_NAMES = {
    "jpn": "Japan", "chn": "China", "ind": "India",
    "twn": "Taiwan", "gbr": "United Kingdom", "kor": "South Korea",
    "hkg": "Hong Kong", "aus": "Australia", "can": "Canada",
    "mys": "Malaysia", "deu": "Germany", "fra": "France", "vnm": "Vietnam",
    "pol": "Poland", "tha": "Thailand", "sgp": "Singapore", "swe": "Sweden",
    "pak": "Pakistan", "idn": "Indonesia",
}

# Top 20 minus USA (USA goes on the right side for comparison)
TOP_COUNTRIES = list(COUNTRY_NAMES.keys())


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def cpd_break_date(rets, min_segment=24):
    """Sup-Wald change-point detection."""
    rets = rets.dropna()
    if len(rets) < 2 * min_segment:
        return pd.NaT, np.nan
    dates = rets.index
    values = rets.values
    n = len(values)
    best_t = 0
    best_idx = min_segment
    for i in range(min_segment, n - min_segment):
        pre = values[:i]
        post = values[i:]
        mean_diff = post.mean() - pre.mean()
        se = np.sqrt(pre.var(ddof=1) / len(pre) + post.var(ddof=1) / len(post))
        if se > 0:
            t = abs(mean_diff / se)
            if t > best_t:
                best_t = t
                best_idx = i
    return dates[best_idx], best_t


def compute_cpd(df):
    """Compute CPD for every factor in the DataFrame."""
    factors = df["name"].unique()
    records = []
    for f in factors:
        sub = df[df["name"] == f].set_index("date")["ret_signed"].sort_index()
        if len(sub) < 48:
            continue
        theme = df.loc[df["name"] == f, "theme"].iloc[0]
        cpd_date, cpd_t = cpd_break_date(sub)
        records.append({"factor": f, "theme": theme,
                        "cpd_date": cpd_date, "cpd_t": cpd_t})
    stats_df = pd.DataFrame(records)
    cpd_df = stats_df.dropna(subset=["cpd_date"]).copy()
    if len(cpd_df) > 0:
        cpd_df["cpd_year"] = cpd_df["cpd_date"].dt.year
    return stats_df, cpd_df


def draw_column(ax_hist, ax_spag, ax_dots, ax_cumret,
                cdf, stats_df, cpd_df, label, is_right=False):
    """Draw the 4 panels for one country into the given axes."""
    date_min = cdf["date"].min()
    date_max = cdf["date"].max()
    year_min = date_min.year
    n_factors = stats_df["factor"].nunique()
    n_cpd = len(cpd_df)

    # ── 1. Stacked histogram ────────────────────────────────────────────
    bins = np.arange(max(1970, year_min), 2026, 1)
    if len(cpd_df) > 0 and len(bins) > 1:
        bottom = np.zeros(len(bins) - 1)
        for theme in themes_ordered:
            sub = cpd_df[cpd_df["theme"] == theme]["cpd_year"]
            if len(sub) == 0:
                continue
            counts, _ = np.histogram(sub, bins=bins)
            ax_hist.bar(bins[:-1] + 0.5, counts, width=0.85, bottom=bottom,
                        color=THEME_COLORS[theme], label=theme, alpha=0.85,
                        edgecolor="white", linewidth=0.2)
            bottom += counts

    ax_hist.set_ylabel("# Factors", fontsize=8)
    ax_hist.set_title(f"{label}  ({n_factors} factors, {n_cpd} CPDs)",
                      fontsize=9, fontweight="bold", loc="left")
    if not is_right:
        ax_hist.legend(fontsize=4.5, ncol=2, loc="upper left", framealpha=0.9)
    ax_hist.grid(True, alpha=0.2, axis="y")
    ax_hist.set_xlim(1970, 2025)
    ax_hist.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax_hist.tick_params(labelsize=7)

    # ── 2. Spaghetti cumret ─────────────────────────────────────────────
    for theme in themes_ordered:
        factor_list = stats_df[stats_df["theme"] == theme]["factor"].values
        if len(factor_list) == 0:
            continue
        color = THEME_COLORS[theme]
        for j, f in enumerate(factor_list):
            sub = cdf[cdf["name"] == f].set_index("date")["ret_signed"].sort_index()
            cumret = (1 + sub).cumprod()
            lbl = theme if j == 0 else None
            ax_spag.plot(cumret.index, cumret.values, color=color,
                         linewidth=0.3, alpha=0.3, label=lbl)

    ax_spag.set_yscale("log")
    ax_spag.set_ylabel("Cumulative Return", fontsize=8)
    ax_spag.set_title("Factor Cumulative Returns", fontsize=9,
                      fontweight="bold", loc="left")
    if not is_right:
        ax_spag.legend(fontsize=4.5, ncol=2, loc="upper left", framealpha=0.9)
    ax_spag.grid(True, alpha=0.2)
    ax_spag.set_xlim(pd.Timestamp("1970-01-01"), date_max)
    ax_spag.tick_params(labelsize=7)

    # ── 3. Dot plot ─────────────────────────────────────────────────────
    if len(cpd_df) > 0:
        theme_medians = cpd_df.groupby("theme")["cpd_date"].median().sort_values()
        themes_sorted = [t for t in theme_medians.index if t in themes_ordered]

        for i, theme in enumerate(themes_sorted):
            sub = cpd_df[cpd_df["theme"] == theme]
            dates = sub["cpd_date"].values
            y_jit = np.full(len(dates), i) + np.random.uniform(-0.2, 0.2,
                                                                len(dates))
            ax_dots.scatter(dates, y_jit, c="#CC4444", s=10, alpha=0.5,
                            edgecolors="none", zorder=3)
            med = sub["cpd_date"].median()
            if pd.notna(med):
                ax_dots.scatter([med], [i], c="#2266AA", s=55, marker="D",
                                edgecolors="black", linewidths=0.5, zorder=5)

        ax_dots.set_yticks(range(len(themes_sorted)))
        ax_dots.set_yticklabels(themes_sorted, fontsize=6)
        ax_dots.invert_yaxis()
    else:
        ax_dots.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                     transform=ax_dots.transAxes, fontsize=9, color="grey")

    ax_dots.set_title("Break Dates by Theme", fontsize=9,
                      fontweight="bold", loc="left")
    ax_dots.grid(True, alpha=0.2, axis="x")
    ax_dots.set_xlim(pd.Timestamp("1970-01-01"), pd.Timestamp("2025-01-01"))
    ax_dots.tick_params(labelsize=7)

    # ── 4. Theme cumret ─────────────────────────────────────────────────
    theme_rets = (
        cdf.groupby(["date", "theme"])["ret_signed"]
        .mean()
        .reset_index()
        .pivot(index="date", columns="theme", values="ret_signed")
        .sort_index()
    )

    if len(cpd_df) > 0:
        plot_themes = themes_sorted
    else:
        plot_themes = [t for t in themes_ordered if t in theme_rets.columns]

    for theme in plot_themes:
        if theme not in theme_rets.columns:
            continue
        cumret = (1 + theme_rets[theme]).cumprod()
        color = THEME_COLORS.get(theme, "#999999")
        ax_cumret.plot(cumret.index, cumret.values, color=color,
                       linewidth=1.2, alpha=0.85, label=theme)

    ax_cumret.set_yscale("log")
    ax_cumret.set_ylabel("Cumulative Return", fontsize=8)
    ax_cumret.set_title("Theme Cumulative Returns", fontsize=9,
                        fontweight="bold", loc="left")
    if not is_right:
        ax_cumret.legend(fontsize=4.5, ncol=2, loc="upper left", framealpha=0.9)
    ax_cumret.grid(True, alpha=0.2)
    ax_cumret.set_xlim(pd.Timestamp("1970-01-01"), date_max)
    ax_cumret.tick_params(labelsize=7)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

print("[cpd_country] Loading vw_cap factor returns ...")
fr = load_jkp_factor_returns(weighting="vw_cap")
fr = fr[fr["date"] >= START_DATE].copy()

# 13-theme mapping + sign convention
fr["theme"] = fr["name"].map(THEME_MAP_13)
fr["ret_signed"] = fr["ret"] * fr["direction"]

# ── Pre-compute USA (reused on every page) ──────────────────────────────
print("[cpd_country] Computing USA CPD (reference) ...")
usa_df = fr[fr["location"] == "usa"].copy()
usa_stats, usa_cpd = compute_cpd(usa_df)
print(f"[cpd_country] USA: {usa_stats['factor'].nunique()} factors, "
      f"{len(usa_cpd)} CPDs")

# ── Build PDF ───────────────────────────────────────────────────────────
local_dir = PATH["FINAL_RESULTS"] / "figures"
local_dir.mkdir(parents=True, exist_ok=True)
pdf_path = local_dir / "cpd_by_country.pdf"

with PdfPages(pdf_path) as pdf:
    for cc in TOP_COUNTRIES:
        cname = COUNTRY_NAMES[cc]
        print(f"[cpd_country] {cname} ({cc}) ...", end=" ", flush=True)

        cdf = fr[fr["location"] == cc].copy()
        if len(cdf) == 0:
            print("no data, skipping")
            continue

        n_fac = cdf["name"].nunique()
        print(f"{n_fac} factors", end=" ", flush=True)

        c_stats, c_cpd = compute_cpd(cdf)
        print(f"→ {len(c_cpd)} CPDs")

        # 4 rows × 2 columns: left = country, right = USA
        fig, axes = plt.subplots(4, 2, figsize=(18, 14))

        # Left column: country
        draw_column(axes[0, 0], axes[1, 0], axes[2, 0], axes[3, 0],
                    cdf, c_stats, c_cpd,
                    label=f"{cname} ({cc.upper()})",
                    is_right=False)

        # Right column: USA
        draw_column(axes[0, 1], axes[1, 1], axes[2, 1], axes[3, 1],
                    usa_df, usa_stats, usa_cpd,
                    label="United States (USA)",
                    is_right=True)

        fig.suptitle(f"{cname}  vs  United States",
                     fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

print(f"\n[cpd_country] Saved → {pdf_path}")

# Copy to Overleaf
overleaf_dir = PATH["OVERLEAF"] / "figures"
overleaf_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(pdf_path, overleaf_dir / "cpd_by_country.pdf")
print(f"[cpd_country] Copied → {overleaf_dir / 'cpd_by_country.pdf'}")

print("[cpd_country] Done.")
