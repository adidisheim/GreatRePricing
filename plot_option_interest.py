"""
Plot option market interest over time: total volume & total open interest.

Panels (rows):
    - All stocks  (full OptionMetrics universe, no CRSP/JKP merge needed)
    - Small stocks (JKP size_grp ∈ {nano, micro, small})
    - Big stocks   (JKP size_grp ∈ {large, mega})
    - High momentum (top quintile of JKP ret_12_1 each month)

Within each panel, two line series:
    - Short-dated options  (DTE ≤ 30 days)
    - Long-dated options   (DTE > 180 days)

Output:
    Two multi-panel figures saved via save_figure:
        option_volume_by_group   (total volume)
        option_oi_by_group       (total open interest)

Usage:
    .venv/bin/python plot_option_interest.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from config import PATH
from data import load_optionm_agg, load_jkp, load_crsp
from util_locals.save import save_figure


# ── 1. Load data ─────────────────────────────────────────────────────────────

print("[plot] Loading OptionMetrics ...")
optm = load_optionm_agg()

print("[plot] Loading CRSP ...")
crsp = load_crsp()

print("[plot] Loading JKP ...")
jkp = load_jkp("us")


# ── 2. Build permno link for OptionMetrics ───────────────────────────────────

# Map OptionMetrics cusip → CRSP permno (via ncusip)
crsp_map = crsp.dropna(subset=["ncusip"]).copy()
crsp_map["cusip8"] = crsp_map["ncusip"].str[:8]
crsp_map = crsp_map.sort_values("date").drop_duplicates(subset=["cusip8"], keep="last")
crsp_map = crsp_map[["cusip8", "permno"]]

optm["cusip8"] = optm["cusip"].str[:8]
optm = optm.merge(crsp_map, on="cusip8", how="left")


# ── 3. Build JKP monthly characteristics (size_grp, momentum quintile) ───────

# JKP end-of-month → first-of-next-month to match OptionMetrics ym convention
# OptionMetrics ym is 1st of month; JKP eom is end-of-month
# We align by rounding both to month periods
jkp["ym"] = jkp["eom"] + pd.offsets.MonthBegin(0)  # snap to beginning of same month
# Actually, JKP eom is end-of-month characteristics known at that date,
# and they predict the *next* month.  So a characteristic at 2020-01-31
# applies to the stock in February 2020.  We want to tag OptionMetrics
# months with the *lagged* JKP characteristics (known at start of month).
# shift eom forward by 1 month to align
jkp["ym"] = jkp["eom"] + pd.DateOffset(months=1)
jkp["ym"] = jkp["ym"].values.astype("datetime64[M]").astype("datetime64[ns]")

# Momentum quintile within each month
jkp["mom_q"] = jkp.groupby("ym")["ret_12_1"].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
)

# Keep only what we need
jkp_slim = jkp[["permno", "ym", "size_grp", "mom_q", "me"]].dropna(
    subset=["permno"]
).copy()
jkp_slim["permno"] = jkp_slim["permno"].astype(float)


# ── 4. Merge OptionMetrics with JKP ─────────────────────────────────────────

# OptionMetrics ym is already datetime
optm["ym_dt"] = optm["ym"].values.astype("datetime64[M]").astype("datetime64[ns]")
jkp_slim["ym_dt"] = jkp_slim["ym"].values.astype("datetime64[M]").astype("datetime64[ns]")

optm_jkp = optm.merge(
    jkp_slim,
    left_on=["permno", "ym_dt"],
    right_on=["permno", "ym_dt"],
    how="left",
    suffixes=("", "_jkp"),
)

print(f"[plot] OptionMetrics rows: {len(optm):,}")
print(f"[plot] Matched with JKP:   {optm_jkp['size_grp'].notna().sum():,} "
      f"({optm_jkp['size_grp'].notna().mean():.1%})")


# ── 5. Aggregation helper ───────────────────────────────────────────────────

def aggregate_ts(df, dte_val, metric):
    """
    Aggregate a metric (total_volume or total_oi) by month for a given
    DTE bucket across all cp_flag and moneyness.

    Returns a Series indexed by ym.
    """
    sub = df[df["dte_bucket"] == dte_val]
    return sub.groupby("ym")[metric].sum()


def build_panel_data(df, metric):
    """
    For a given subset df, build a DataFrame with columns:
        ym, short, long
    for the chosen metric.
    """
    short = aggregate_ts(df, "short", metric).rename("short")
    long_ = aggregate_ts(df, "long", metric).rename("long")
    out = pd.concat([short, long_], axis=1).sort_index()
    return out


# ── 6. Define stock groups ───────────────────────────────────────────────────

groups = {}

# All stocks: use full optm (no JKP merge needed)
groups["All stocks"] = optm

# Small stocks: size_grp in {nano, micro, small}
groups["Small stocks"] = optm_jkp[
    optm_jkp["size_grp"].isin(["nano", "micro", "small"])
]

# Big stocks: size_grp in {large, mega}
groups["Big stocks"] = optm_jkp[
    optm_jkp["size_grp"].isin(["large", "mega"])
]

# High momentum: top quintile (mom_q == 4)
groups["High momentum"] = optm_jkp[optm_jkp["mom_q"] == 4]


# ── 7. Build time series for each group ─────────────────────────────────────

metrics = {
    "total_volume": "Total Option Volume (Contracts)",
    "total_oi":     "Total Open Interest (Contracts)",
}

panel_data = {}  # metric → group_name → DataFrame(ym, short, long)

for metric in metrics:
    panel_data[metric] = {}
    for gname, gdf in groups.items():
        panel_data[metric][gname] = build_panel_data(gdf, metric)
        print(f"[plot]   {metric} / {gname}: "
              f"{len(panel_data[metric][gname])} months")


# ── 8. Plot ──────────────────────────────────────────────────────────────────

COLOR_SHORT = "#E24A33"   # warm red
COLOR_LONG  = "#348ABD"   # steel blue

group_order = ["All stocks", "Small stocks", "Big stocks", "High momentum"]

for metric, ylabel in metrics.items():
    fig, axes = plt.subplots(
        len(group_order), 1,
        figsize=(10, 3.0 * len(group_order)),
        sharex=True,
    )

    for i, gname in enumerate(group_order):
        ax = axes[i]
        ts = panel_data[metric][gname]

        # Smooth with 12-month rolling mean for readability
        ts_smooth = ts.rolling(12, min_periods=6).mean()

        ax.plot(
            ts_smooth.index, ts_smooth["short"],
            color=COLOR_SHORT, linewidth=1.5, label="Short-dated (≤30d)",
        )
        ax.plot(
            ts_smooth.index, ts_smooth["long"],
            color=COLOR_LONG, linewidth=1.5, label="Long-dated (>180d)",
        )

        ax.set_title(gname, fontsize=12, fontweight="bold", loc="left")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6
                                  else f"{x/1e3:.0f}K" if x >= 1e3
                                  else f"{x:.0f}")
        )
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylabel, fontsize=9)

        if i == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    axes[-1].set_xlabel("")
    fig.suptitle(
        ylabel + " — Short vs Long Maturity",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_name = "option_volume_by_group" if "volume" in metric else "option_oi_by_group"
    save_figure(fig, save_name)
    plt.close(fig)

print("\n[plot] Done.")
