"""
WRDS data downloads.

Each function downloads raw data from WRDS and saves it to RAW_DATA.
These are meant to be run once (or rarely) — the data.py loaders handle
day-to-day caching via the reload pattern.

Usage:
    .venv/bin/python wrds_import.py crsp          # download CRSP monthly
    .venv/bin/python wrds_import.py optionm       # download OptionMetrics agg (1996-2025, ~3h)
    .venv/bin/python wrds_import.py --list         # list available downloads
    .venv/bin/python wrds_import.py --all          # download everything

On first run, wrds will prompt for your WRDS username and password
and cache credentials for future use.
"""

import sys
import wrds
import pandas as pd
from config import PATH


def get_connection() -> wrds.Connection:
    """Open a WRDS connection (prompts for credentials on first use)."""
    return wrds.Connection()


# ══════════════════════════════════════════════════════════════════════════════
#  CRSP MONTHLY
# ══════════════════════════════════════════════════════════════════════════════

def download_crsp_monthly() -> None:
    """
    Download CRSP monthly stock file for common shares (shrcd 10, 11).

    Pulls from crsp.msf (monthly stock file) joined with crsp.msenames
    (security names / identifiers).

    Variables:
        - permno, permco          firm & security identifiers
        - date                    end-of-month date
        - ret, retx               return (with / without dividends)
        - prc, altprc              price (negative = bid/ask avg)
        - vol                      trading volume (in shares)
        - shrout                   shares outstanding (in thousands)
        - bid, ask                 closing bid & ask
        - cfacpr, cfacshr          cumulative adjustment factors
        - shrcd, exchcd            share code, exchange code
        - siccd, naics             SIC & NAICS industry codes
        - hsiccd                   historical SIC code
        - ncusip                   8-digit historical CUSIP (for linking)
        - ticker, comnam           ticker symbol, company name
    """
    print("[wrds_import] Downloading CRSP monthly ...")

    query = """
        SELECT
            a.permno, a.permco, a.date,
            a.ret, a.retx,
            a.prc, a.altprc,
            a.vol,
            a.shrout,
            a.bid, a.ask,
            a.cfacpr, a.cfacshr,
            b.shrcd, b.exchcd,
            b.siccd, b.naics,
            b.hsiccd,
            b.ncusip,
            b.ticker, b.comnam
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
            ON a.permno = b.permno
            AND a.date >= b.namedt
            AND a.date <= b.nameendt
        WHERE b.shrcd IN (10, 11)
    """

    db = get_connection()
    df = db.raw_sql(query, date_cols=["date"])
    db.close()

    # ── basic cleaning ────────────────────────────────────────────────────
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    # market cap (in $thousands since shrout is in thousands)
    df["mktcap"] = df["prc"].abs() * df["shrout"]

    # spread
    df["spread"] = df["ask"] - df["bid"]
    df.loc[df["spread"] < 0, "spread"] = None

    out = PATH["RAW_DATA"] / "crsp_monthly.parquet"
    PATH["RAW_DATA"].mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[wrds_import] CRSP monthly: {len(df):,} rows → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  JKP CHARACTERISTICS  (Jensen, Kelly & Pedersen)
# ══════════════════════════════════════════════════════════════════════════════

# Countries to download.  Add more lists here to extend coverage.
JKP_COUNTRY_SETS = {
    "us": ["USA"],
    # "developed": None,   # None → auto-detect from JKP country_classification.xlsx
    # "all": None,         # could pull everything
}

# URL for the 153-characteristic list (abr_jkp column)
_JKP_FACTOR_DETAILS_URL = (
    "https://github.com/bkelly-lab/jkp-data/raw/main/data/factor_details.xlsx"
)
# URL for country classification (if we ever need developed / emerging)
_JKP_COUNTRY_CLASS_URL = (
    "https://github.com/bkelly-lab/jkp-data/raw/main/data/country_classification.xlsx"
)


def _get_jkp_char_names() -> list[str]:
    """Fetch the 153 JKP characteristic abbreviations from GitHub."""
    chars = pd.read_excel(_JKP_FACTOR_DETAILS_URL)
    return chars.loc[chars["abr_jkp"].notna(), "abr_jkp"].tolist()


def _resolve_jkp_countries(country_set: str) -> list[str]:
    """
    Return a list of excntry codes for the requested set.

    Parameters
    ----------
    country_set : str
        Key into JKP_COUNTRY_SETS.  If the value is a list, use it directly.
        If None, download the classification file and pick 'developed'.
    """
    preset = JKP_COUNTRY_SETS.get(country_set)
    if preset is not None:
        return preset

    # fall back to the JKP country classification
    countries = pd.read_excel(_JKP_COUNTRY_CLASS_URL)
    if country_set == "developed":
        return countries.loc[
            countries["msci_development"] == "developed", "excntry"
        ].tolist()
    else:
        # "all" or anything else → return every country
        return countries["excntry"].tolist()


def download_jkp_characteristics(country_set: str = "us") -> None:
    """
    Download JKP stock-level characteristics from WRDS (contrib.global_factor).

    Downloads the 153 published characteristics plus identifiers and the
    1-month-ahead excess return.

    Parameters
    ----------
    country_set : str
        Which countries to pull.  Default 'us'.
        Add entries to JKP_COUNTRY_SETS to extend.
    """
    print(f"[wrds_import] Downloading JKP characteristics ({country_set}) ...")

    # resolve characteristic columns
    char_names = _get_jkp_char_names()
    print(f"[wrds_import]   {len(char_names)} characteristics")

    # resolve countries
    excntry_list = _resolve_jkp_countries(country_set)
    country_sql = ", ".join(f"'{c}'" for c in excntry_list)
    print(f"[wrds_import]   Countries: {excntry_list}")

    # build query
    id_cols = "id, eom, excntry, gvkey, permno, size_grp, me, ret_exc_lead1m"
    char_cols = ", ".join(char_names)

    query = f"""
        SELECT {id_cols}, {char_cols}
        FROM contrib.global_factor
        WHERE common = 1
          AND exch_main = 1
          AND primary_sec = 1
          AND obs_main = 1
          AND excntry IN ({country_sql})
    """

    db = get_connection()
    df = db.raw_sql(query, date_cols=["eom"])
    db.close()

    df = df.sort_values(["permno", "eom"]).reset_index(drop=True)

    # save with country set in filename so US / developed / all don't collide
    out = PATH["RAW_DATA"] / f"jkp_characteristics_{country_set}.parquet"
    PATH["RAW_DATA"].mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[wrds_import] JKP ({country_set}): {len(df):,} rows, "
          f"{len(df.columns)} cols → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONMETRICS – aggregated option volume & open interest
# ══════════════════════════════════════════════════════════════════════════════

# Year range for OptionMetrics opprcd tables
_OM_YEAR_START = 1996
_OM_YEAR_END   = 2025


def _optionm_agg_query(year: int) -> str:
    """
    SQL query that aggregates OptionMetrics contract-level data into
    secid × month × call/put × DTE-bucket × moneyness-bucket.

    Joins with secprd for underlying price to compute moneyness.
    Filters: impl_volatility > 0, non-null/non-zero underlying price.

    Aggregated columns:
        total_volume, total_oi, n_obs, avg_iv, avg_abs_delta
    """
    return f"""
        SELECT
            b.secid,
            DATE_TRUNC('month', b.date)::date  AS ym,
            b.cp_flag,
            CASE
                WHEN (b.exdate - b.date) <= 30  THEN 'short'
                WHEN (b.exdate - b.date) <= 180 THEN 'medium'
                ELSE 'long'
            END AS dte_bucket,
            CASE
                WHEN b.cp_flag = 'C'
                     AND b.strike_price / 1000.0 / ABS(a.close) < 0.95 THEN 'ITM'
                WHEN b.cp_flag = 'C'
                     AND b.strike_price / 1000.0 / ABS(a.close) > 1.05 THEN 'OTM'
                WHEN b.cp_flag = 'P'
                     AND b.strike_price / 1000.0 / ABS(a.close) > 1.05 THEN 'ITM'
                WHEN b.cp_flag = 'P'
                     AND b.strike_price / 1000.0 / ABS(a.close) < 0.95 THEN 'OTM'
                ELSE 'ATM'
            END AS moneyness,
            SUM(b.volume)            AS total_volume,
            SUM(b.open_interest)     AS total_oi,
            COUNT(*)                 AS n_obs,
            AVG(b.impl_volatility)   AS avg_iv,
            AVG(ABS(b.delta))        AS avg_abs_delta
        FROM optionm.secprd{year} AS a
        INNER JOIN optionm.opprcd{year} AS b
            ON a.secid = b.secid AND a.date = b.date
        WHERE a.close IS NOT NULL
          AND a.close != 0
          AND b.impl_volatility > 0
        GROUP BY
            b.secid,
            DATE_TRUNC('month', b.date)::date,
            b.cp_flag,
            CASE
                WHEN (b.exdate - b.date) <= 30  THEN 'short'
                WHEN (b.exdate - b.date) <= 180 THEN 'medium'
                ELSE 'long'
            END,
            CASE
                WHEN b.cp_flag = 'C'
                     AND b.strike_price / 1000.0 / ABS(a.close) < 0.95 THEN 'ITM'
                WHEN b.cp_flag = 'C'
                     AND b.strike_price / 1000.0 / ABS(a.close) > 1.05 THEN 'OTM'
                WHEN b.cp_flag = 'P'
                     AND b.strike_price / 1000.0 / ABS(a.close) > 1.05 THEN 'ITM'
                WHEN b.cp_flag = 'P'
                     AND b.strike_price / 1000.0 / ABS(a.close) < 0.95 THEN 'OTM'
                ELSE 'ATM'
            END
    """


def download_optionm_agg() -> None:
    """
    Download aggregated option volume & open interest from OptionMetrics.

    Loops year-by-year over opprcd{year} joined with secprd{year}.
    Aggregates by: secid × month × call/put × DTE bucket × moneyness.

    Also downloads the secnmd identifier table for secid→CUSIP mapping
    (needed later to link to CRSP permno via CUSIP).

    Output files in RAW_DATA:
        - optionm_agg.parquet       (aggregated option data)
        - optionm_secnmd.parquet    (secid → CUSIP/ticker mapping)
    """
    import time

    PATH["RAW_DATA"].mkdir(parents=True, exist_ok=True)

    db = get_connection()

    # ── 1. Download secnmd (identifier mapping, ~272K rows) ──────────────
    print("[wrds_import] Downloading optionm.secnmd ...")
    secnmd = db.raw_sql(
        "SELECT secid, effect_date, cusip, ticker, issuer, sic "
        "FROM optionm.secnmd",
        date_cols=["effect_date"],
    )
    secnmd_out = PATH["RAW_DATA"] / "optionm_secnmd.parquet"
    secnmd.to_parquet(secnmd_out, index=False)
    print(f"[wrds_import]   secnmd: {len(secnmd):,} rows → {secnmd_out}")

    # ── 2. Loop over years, aggregate, concatenate ───────────────────────
    frames = []
    for year in range(_OM_YEAR_START, _OM_YEAR_END + 1):
        print(f"[wrds_import]   opprcd{year} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            df_year = db.raw_sql(_optionm_agg_query(year), date_cols=["ym"])
            elapsed = time.time() - t0
            print(f"{len(df_year):>10,} rows  ({elapsed:.0f}s)")
            frames.append(df_year)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"SKIPPED ({e})  ({elapsed:.0f}s)")

    db.close()

    # ── 3. Concatenate and save ──────────────────────────────────────────
    df = pd.concat(frames, ignore_index=True)

    # Downcast secid to int
    df["secid"] = df["secid"].astype("Int64")

    df = df.sort_values(["secid", "ym", "cp_flag", "dte_bucket", "moneyness"])
    df = df.reset_index(drop=True)

    out = PATH["RAW_DATA"] / "optionm_agg.parquet"
    df.to_parquet(out, index=False)
    print(f"[wrds_import] OptionMetrics aggregated: {len(df):,} rows, "
          f"{df['secid'].nunique():,} secids → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRY – add new download functions here
# ══════════════════════════════════════════════════════════════════════════════

DOWNLOADS = {
    "crsp": ("CRSP monthly stock file (shrcd 10/11)", download_crsp_monthly),
    "jkp":  ("JKP 153 characteristics (US)", lambda: download_jkp_characteristics("us")),
    "optionm": ("OptionMetrics aggregated volume/OI (1996-2025)", download_optionm_agg),
    # "jkp_developed": ("JKP 153 characteristics (developed)", lambda: download_jkp_characteristics("developed")),
    # "compustat":     ("Compustat annual fundamentals", download_compustat),
    # "ff":            ("Fama-French factors", download_ff_factors),
}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print("Usage: .venv/bin/python wrds_import.py [dataset ...] [--all] [--list]")
        print()
        for key, (desc, _) in DOWNLOADS.items():
            print(f"  {key:15s}  {desc}")
        return

    if "--list" in args:
        for key, (desc, _) in DOWNLOADS.items():
            print(f"  {key:15s}  {desc}")
        return

    targets = list(DOWNLOADS.keys()) if "--all" in args else args

    for name in targets:
        if name.startswith("-"):
            continue
        if name not in DOWNLOADS:
            print(f"[wrds_import] Unknown dataset: '{name}'.  Use --list to see options.")
            continue
        _, func = DOWNLOADS[name]
        func()

    print("[wrds_import] Done.")


if __name__ == "__main__":
    main()
