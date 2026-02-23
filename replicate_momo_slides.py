"""
Replicate CPD results from momo_slides.pdf (slides 4 & 5) using JKP
factor returns.

For each weighting scheme (vw, ew, vw_cap) produces a 2-page PDF:
    Page 1 – "CPDs on JKP Factors" (slide 4):
             Left:  stacked histogram of CPD break dates by theme
             Right: spaghetti cumulative returns (all 153 factors, by theme)
    Page 2 – "CPDs on JKP Themes" (slide 5):
             Left:  dot plot of break dates (dots = factors, diamonds = median)
             Right: cumulative returns by theme (one line per theme)

Output files:
    cpd_jkp_factors_vw.pdf      (value-weighted)
    cpd_jkp_factors_ew.pdf      (equal-weighted)
    cpd_jkp_factors_vw_cap.pdf  (value-weighted capped)

Usage:
    .venv/bin/python replicate_momo_slides.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from config import PATH
from data import load_jkp_factor_returns, load_jkp_factor_details
from util_locals.factor_themes import THEME_MAP_13, THEMES_13, THEME_COLORS_13
import shutil


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

START_DATE = "1974-01-01"
BREAK_YEAR = 2002
THEME_COLORS = THEME_COLORS_13
themes_ordered = THEMES_13

WEIGHTINGS = ["vw", "ew", "vw_cap"]
WEIGHTING_LABELS = {"vw": "Value-Weighted", "ew": "Equal-Weighted",
                    "vw_cap": "Value-Weighted (Capped)"}


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def cpd_break_date(rets, min_segment=24):
    """
    Simple change-point detection: find the date that maximizes the
    absolute t-stat for difference in means (sup-Wald).
    """
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


def compute_factor_stats(usa):
    """Compute CPD break date for every factor."""
    factors = usa["name"].unique()
    records = []
    for f in factors:
        sub = usa[usa["name"] == f].set_index("date")["ret_signed"].sort_index()
        theme = usa.loc[usa["name"] == f, "theme"].iloc[0]
        cpd_date, cpd_t = cpd_break_date(sub)
        records.append({
            "factor": f, "theme": theme,
            "cpd_date": cpd_date, "cpd_t": cpd_t,
        })

    stats_df = pd.DataFrame(records)
    cpd_df = stats_df.dropna(subset=["cpd_date"]).copy()
    cpd_df["cpd_year"] = cpd_df["cpd_date"].dt.year
    return stats_df, cpd_df


def make_cpd_pdf(usa, stats_df, cpd_df, wt, wt_label):
    """Create the 2-page CPD PDF (slides 4 & 5) for a given weighting."""
    # Theme-level returns
    theme_rets = (
        usa.groupby(["date", "theme"])["ret_signed"]
        .mean()
        .reset_index()
        .pivot(index="date", columns="theme", values="ret_signed")
        .sort_index()
    )

    local_dir = PATH["FINAL_RESULTS"] / "figures"
    local_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = f"cpd_jkp_factors_{wt}.pdf"
    pdf_path = local_dir / pdf_name

    with PdfPages(pdf_path) as pdf:

        # ── PAGE 1: "CPDs on JKP Factors" (slide 4) ──────────────────────
        fig, (ax_hist, ax_spaghetti) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: stacked histogram (year-by-year)
        bins = np.arange(1974, 2026, 1)
        bottom = np.zeros(len(bins) - 1)
        for theme in themes_ordered:
            sub = cpd_df[cpd_df["theme"] == theme]["cpd_year"]
            counts, _ = np.histogram(sub, bins=bins)
            ax_hist.bar(bins[:-1] + 0.5, counts, width=0.85, bottom=bottom,
                        color=THEME_COLORS[theme], label=theme, alpha=0.85,
                        edgecolor="white", linewidth=0.2)
            bottom += counts

        ax_hist.set_xlabel("Estimated Break Date", fontsize=11)
        ax_hist.set_ylabel("Number of Factors", fontsize=11)
        ax_hist.set_title("Distribution of CPD Break Dates", fontsize=12,
                          fontweight="bold", loc="left")
        ax_hist.legend(fontsize=6, ncol=2, loc="upper left", framealpha=0.9)
        ax_hist.grid(True, alpha=0.2, axis="y")
        ax_hist.set_xlim(1974, 2025)
        ax_hist.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax_hist.xaxis.set_minor_locator(mticker.MultipleLocator(1))

        # Right: spaghetti cumulative returns
        for theme in themes_ordered:
            factor_list = stats_df[stats_df["theme"] == theme]["factor"].values
            color = THEME_COLORS[theme]
            for j, f in enumerate(factor_list):
                sub = usa[usa["name"] == f].set_index("date")["ret_signed"].sort_index()
                cumret = (1 + sub).cumprod()
                label = theme if j == 0 else None
                ax_spaghetti.plot(cumret.index, cumret.values, color=color,
                                  linewidth=0.5, alpha=0.4, label=label)

        ax_spaghetti.set_yscale("log")
        ax_spaghetti.set_ylabel("Cumulative Return ($1 invested)", fontsize=11)
        ax_spaghetti.set_title("Individual Factor Cumulative Returns", fontsize=12,
                               fontweight="bold", loc="left")
        ax_spaghetti.legend(fontsize=7, ncol=1, loc="upper left", framealpha=0.9)
        ax_spaghetti.grid(True, alpha=0.2)
        ax_spaghetti.set_xlim(pd.Timestamp(START_DATE), usa["date"].max())

        fig.suptitle(f"CPDs on JKP Factors — {wt_label}",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # ── PAGE 2: "CPDs on JKP Themes" (slide 5) ──────────────────────
        fig, (ax_dots, ax_cumret) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: dot plot
        theme_medians = cpd_df.groupby("theme")["cpd_date"].median().sort_values()
        themes_sorted = [t for t in theme_medians.index if t in themes_ordered]

        for i, theme in enumerate(themes_sorted):
            sub = cpd_df[cpd_df["theme"] == theme]
            dates = sub["cpd_date"].values

            y_jitter = np.full(len(dates), i) + np.random.uniform(-0.2, 0.2,
                                                                   len(dates))
            ax_dots.scatter(dates, y_jitter, c="#CC4444", s=18, alpha=0.5,
                            edgecolors="none", zorder=3)

            med = sub["cpd_date"].median()
            if pd.notna(med):
                ax_dots.scatter([med], [i], c="#2266AA", s=100, marker="D",
                                edgecolors="black", linewidths=0.8, zorder=5)

        ax_dots.set_yticks(range(len(themes_sorted)))
        ax_dots.set_yticklabels(themes_sorted, fontsize=10)
        ax_dots.set_xlabel("Estimated Break Date", fontsize=11)
        ax_dots.set_title("Break Dates by Theme", fontsize=12,
                          fontweight="bold", loc="left")
        ax_dots.grid(True, alpha=0.2, axis="x")
        ax_dots.set_xlim(pd.Timestamp("1974-01-01"), pd.Timestamp("2025-01-01"))
        ax_dots.invert_yaxis()

        # Right: theme-level cumulative returns
        for theme in themes_sorted:
            if theme not in theme_rets.columns:
                continue
            cumret = (1 + theme_rets[theme]).cumprod()
            color = THEME_COLORS.get(theme, "#999999")
            ax_cumret.plot(cumret.index, cumret.values, color=color,
                           linewidth=1.8, alpha=0.85, label=theme)

        ax_cumret.set_yscale("log")
        ax_cumret.set_ylabel("Cumulative Return ($1 invested)", fontsize=11)
        ax_cumret.set_title("Theme-Level Cumulative Returns", fontsize=12,
                            fontweight="bold", loc="left")
        ax_cumret.legend(fontsize=8, ncol=1, loc="upper left", framealpha=0.9)
        ax_cumret.grid(True, alpha=0.2)
        ax_cumret.set_xlim(pd.Timestamp(START_DATE), usa["date"].max())

        fig.suptitle(f"CPDs on JKP Themes — {wt_label}",
                     fontsize=15, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[momo] Saved → {pdf_path}")

    # Copy to Overleaf
    overleaf_dir = PATH["OVERLEAF"] / "figures"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, overleaf_dir / pdf_name)
    print(f"[momo] Copied → {overleaf_dir / pdf_name}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN LOOP OVER WEIGHTINGS
# ═══════════════════════════════════════════════════════════════════════════

print("[momo] Loading factor details ...")
details = load_jkp_factor_details()

for wt in WEIGHTINGS:
    wt_label = WEIGHTING_LABELS[wt]
    print(f"\n{'='*70}")
    print(f"  WEIGHTING: {wt_label}  ({wt})")
    print(f"{'='*70}")

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"[momo] Loading JKP factor returns ({wt}) ...")
    fr = load_jkp_factor_returns(weighting=wt)

    usa = fr[(fr["location"] == "usa") & (fr["date"] >= START_DATE)].copy()
    print(f"[momo] USA (≥1974): {len(usa):,} rows, "
          f"{usa['name'].nunique()} factors, "
          f"{usa['date'].min().date()} to {usa['date'].max().date()}")

    # 13-theme mapping
    usa["theme"] = usa["name"].map(THEME_MAP_13)
    unmapped = usa[usa["theme"].isna()]["name"].unique()
    if len(unmapped) > 0:
        print(f"[momo] WARNING: {len(unmapped)} factors unmapped: "
              f"{list(unmapped[:10])}")

    # Sign convention
    usa["ret_signed"] = usa["ret"] * usa["direction"]

    # ── Compute stats ─────────────────────────────────────────────────────
    print(f"[momo] Computing CPD for {usa['name'].nunique()} factors ...")
    stats_df, cpd_df = compute_factor_stats(usa)
    print(f"[momo] CPD dates for {len(cpd_df)} factors.")

    # ── Plots ─────────────────────────────────────────────────────────────
    make_cpd_pdf(usa, stats_df, cpd_df, wt, wt_label)

print("\n[momo] All done.")
