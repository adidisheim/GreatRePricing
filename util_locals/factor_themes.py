"""
13-theme classification for the 153 JKP factors.

Matches the granular theme grouping used in the momo_slides analysis:
    Momentum, Short-Term Reversal, Value, Low Leverage, Profitability,
    Profit Growth, Quality, Investment, Accruals, Debt Issuance,
    Low Risk, Size, Seasonality.

Usage:
    from util_locals.factor_themes import THEME_MAP_13, get_theme_13
    theme = get_theme_13("ret_12_1")  # → "Momentum"
"""

# factor_name → 13-theme
THEME_MAP_13 = {
    # ── Momentum ─────────────────────────────────────────────────────────
    "ret_6_1":           "Momentum",
    "ret_12_1":          "Momentum",
    "resff3_6_1":        "Momentum",
    "resff3_12_1":       "Momentum",
    "prc_highprc_252d":  "Momentum",
    "niq_su":            "Momentum",
    "saleq_su":          "Momentum",
    "tax_gr1a":          "Momentum",
    "ni_inc8q":          "Momentum",

    # ── Short-Term Reversal ──────────────────────────────────────────────
    "ret_1_0":           "Short-Term Reversal",
    "ret_3_1":           "Short-Term Reversal",
    "ret_9_1":           "Short-Term Reversal",
    "ret_12_7":          "Short-Term Reversal",
    "ret_60_12":         "Short-Term Reversal",

    # ── Value (valuation ratios) ─────────────────────────────────────────
    "be_me":             "Value",
    "at_me":             "Value",
    "bev_mev":           "Value",
    "ni_me":             "Value",
    "fcf_me":            "Value",
    "div12m_me":         "Value",
    "eqpo_me":           "Value",
    "eqnpo_me":          "Value",
    "sale_me":           "Value",
    "ival_me":           "Value",
    "ebitda_mev":        "Value",
    "ocf_me":            "Value",
    "netdebt_me":        "Value",
    "eq_dur":            "Value",

    # ── Low Leverage (leverage / distress) ───────────────────────────────
    "debt_me":           "Low Leverage",
    "at_be":             "Low Leverage",
    "kz_index":          "Low Leverage",
    "o_score":           "Low Leverage",
    "z_score":           "Low Leverage",
    "cash_at":           "Low Leverage",

    # ── Profitability ────────────────────────────────────────────────────
    "gp_at":             "Profitability",
    "gp_atl1":           "Profitability",
    "op_at":             "Profitability",
    "op_atl1":           "Profitability",
    "ope_be":            "Profitability",
    "ope_bel1":          "Profitability",
    "cop_at":            "Profitability",
    "cop_atl1":          "Profitability",
    "ebit_sale":         "Profitability",
    "ebit_bev":          "Profitability",
    "niq_at":            "Profitability",
    "niq_be":            "Profitability",
    "at_turnover":       "Profitability",
    "sale_bev":          "Profitability",
    "f_score":           "Profitability",
    "pi_nix":            "Profitability",
    "ocf_at":            "Profitability",
    "opex_at":           "Profitability",
    "sale_emp_gr1":      "Profitability",

    # ── Profit Growth ────────────────────────────────────────────────────
    "niq_at_chg1":       "Profit Growth",
    "niq_be_chg1":       "Profit Growth",
    "saleq_gr1":         "Profit Growth",
    "sale_gr1":          "Profit Growth",
    "sale_gr3":          "Profit Growth",
    "ocf_at_chg1":       "Profit Growth",

    # ── Quality ──────────────────────────────────────────────────────────
    "qmj":               "Quality",
    "qmj_growth":        "Quality",
    "qmj_prof":          "Quality",
    "qmj_safety":        "Quality",
    "mispricing_mgmt":   "Quality",
    "mispricing_perf":   "Quality",
    "ni_be":             "Quality",
    "ni_ar1":            "Quality",
    "earnings_variability": "Quality",
    "ni_ivol":           "Quality",
    "ocfq_saleq_std":    "Quality",
    "age":               "Quality",
    "corr_1260d":        "Quality",

    # ── Investment (real investment / asset growth) ───────────────────────
    "at_gr1":            "Investment",
    "capex_abn":         "Investment",
    "capx_gr1":          "Investment",
    "capx_gr2":          "Investment",
    "capx_gr3":          "Investment",
    "inv_gr1":           "Investment",
    "inv_gr1a":          "Investment",
    "ppeinv_gr1a":       "Investment",
    "be_gr1a":           "Investment",
    "lnoa_gr1a":         "Investment",
    "ncoa_gr1a":         "Investment",
    "ncol_gr1a":         "Investment",
    "nncoa_gr1a":        "Investment",
    "noa_gr1a":          "Investment",
    "emp_gr1":           "Investment",
    "rd5_at":            "Investment",
    "rd_me":             "Investment",
    "rd_sale":           "Investment",
    "tangibility":       "Investment",

    # ── Accruals ─────────────────────────────────────────────────────────
    "oaccruals_at":      "Accruals",
    "oaccruals_ni":      "Accruals",
    "taccruals_at":      "Accruals",
    "taccruals_ni":      "Accruals",
    "cowc_gr1a":         "Accruals",
    "coa_gr1a":          "Accruals",
    "col_gr1a":          "Accruals",
    "nfna_gr1a":         "Accruals",
    "fnl_gr1a":          "Accruals",
    "lti_gr1a":          "Accruals",
    "sti_gr1a":          "Accruals",
    "noa_at":            "Accruals",
    "dgp_dsale":         "Accruals",
    "dsale_dinv":        "Accruals",
    "dsale_drec":        "Accruals",
    "dsale_dsga":        "Accruals",

    # ── Debt Issuance ────────────────────────────────────────────────────
    "chcsho_12m":        "Debt Issuance",
    "dbnetis_at":        "Debt Issuance",
    "eqnetis_at":        "Debt Issuance",
    "netis_at":          "Debt Issuance",
    "eqnpo_12m":         "Debt Issuance",
    "debt_gr3":          "Debt Issuance",

    # ── Low Risk (beta, vol, skewness) ───────────────────────────────────
    "beta_60m":          "Low Risk",
    "beta_dimson_21d":   "Low Risk",
    "betabab_1260d":     "Low Risk",
    "betadown_252d":     "Low Risk",
    "ivol_capm_21d":     "Low Risk",
    "ivol_capm_252d":    "Low Risk",
    "ivol_ff3_21d":      "Low Risk",
    "ivol_hxz4_21d":     "Low Risk",
    "rvol_21d":          "Low Risk",
    "rskew_21d":         "Low Risk",
    "iskew_capm_21d":    "Low Risk",
    "iskew_ff3_21d":     "Low Risk",
    "iskew_hxz4_21d":    "Low Risk",
    "coskew_21d":        "Low Risk",
    "rmax1_21d":         "Low Risk",
    "rmax5_21d":         "Low Risk",
    "rmax5_rvol_21d":    "Low Risk",

    # ── Size (market cap, liquidity, trading) ────────────────────────────
    "market_equity":     "Size",
    "dolvol_126d":       "Size",
    "dolvol_var_126d":   "Size",
    "ami_126d":          "Size",
    "bidaskhl_21d":      "Size",
    "prc":               "Size",
    "turnover_126d":     "Size",
    "turnover_var_126d": "Size",
    "zero_trades_21d":   "Size",
    "zero_trades_126d":  "Size",
    "zero_trades_252d":  "Size",
    "aliq_at":           "Size",
    "aliq_mat":          "Size",

    # ── Seasonality ──────────────────────────────────────────────────────
    "seas_1_1an":        "Seasonality",
    "seas_1_1na":        "Seasonality",
    "seas_2_5an":        "Seasonality",
    "seas_2_5na":        "Seasonality",
    "seas_6_10an":       "Seasonality",
    "seas_6_10na":       "Seasonality",
    "seas_11_15an":      "Seasonality",
    "seas_11_15na":      "Seasonality",
    "seas_16_20an":      "Seasonality",
    "seas_16_20na":      "Seasonality",
}

# 13 theme names in display order
THEMES_13 = [
    "Momentum", "Short-Term Reversal", "Value", "Low Leverage",
    "Profitability", "Profit Growth", "Quality",
    "Investment", "Accruals", "Debt Issuance",
    "Low Risk", "Size", "Seasonality",
]

# Colors for 13 themes
THEME_COLORS_13 = {
    "Momentum":             "#E24A33",
    "Short-Term Reversal":  "#C75050",
    "Value":                "#348ABD",
    "Low Leverage":         "#6699CC",
    "Profitability":        "#988ED5",
    "Profit Growth":        "#B07DD5",
    "Quality":              "#8B6CAF",
    "Investment":           "#777777",
    "Accruals":             "#999999",
    "Debt Issuance":        "#555555",
    "Low Risk":             "#8EBA42",
    "Size":                 "#66A61E",
    "Seasonality":          "#FBC15E",
}


def get_theme_13(factor_name: str) -> str:
    """Look up the 13-theme classification for a JKP factor."""
    return THEME_MAP_13.get(factor_name, "Unknown")
