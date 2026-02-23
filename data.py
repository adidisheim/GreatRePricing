"""
Data loaders.

Convention: every loader function follows the same pattern:

    def load_<dataset>(reload: bool = False) -> pd.DataFrame:
        - reload=True  → read from RAW_DATA, process, save to PROCESSED_DATA
        - reload=False → read the cached file from PROCESSED_DATA

Cached files use parquet (large / columnar) or pickle (complex objects).

Usage:
    from data import load_crsp, load_compustat   # (once you implement them)
"""

import pandas as pd
from pathlib import Path
from config import PATH


# ── helper ────────────────────────────────────────────────────────────────────

def _cache_path(name: str, fmt: str = "parquet") -> Path:
    """Return the path for a cached processed dataset."""
    return PATH["PROCESSED_DATA"] / f"{name}.{fmt}"


def _load_or_reload(
    name: str,
    raw_loader: callable,
    fmt: str = "parquet",
    reload: bool = False,
) -> pd.DataFrame:
    """
    Generic reload-or-cache wrapper.

    Parameters
    ----------
    name : str
        Dataset identifier (used as filename stem).
    raw_loader : callable
        A zero-argument function that returns a DataFrame from raw data.
    fmt : str
        'parquet' or 'pickle'.
    reload : bool
        If True, call raw_loader and overwrite the cache.
    """
    cache = _cache_path(name, fmt)

    if not reload and cache.exists():
        print(f"[data] Loading cached {cache.name}")
        if fmt == "parquet":
            return pd.read_parquet(cache)
        else:
            return pd.read_pickle(cache)

    print(f"[data] Building {name} from raw data ...")
    df = raw_loader()

    if fmt == "parquet":
        df.to_parquet(cache, index=False)
    else:
        df.to_pickle(cache)

    print(f"[data] Cached → {cache}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ADD YOUR LOADERS BELOW
#  Follow the pattern:
#
#   def load_my_dataset(reload: bool = False) -> pd.DataFrame:
#       def _raw():
#           df = pd.read_csv(PATH['RAW_DATA'] / 'my_file.csv')
#           # ... cleaning / processing ...
#           return df
#       return _load_or_reload('my_dataset', _raw, fmt='parquet', reload=reload)
# ══════════════════════════════════════════════════════════════════════════════


# ── CRSP monthly ──────────────────────────────────────────────────────────────

def load_crsp(reload: bool = False) -> pd.DataFrame:
    """
    Load CRSP monthly stock file (common shares, shrcd 10/11).

    Raw source: RAW_DATA/crsp_monthly.parquet  (created by wrds_import.py)
    Cache:      PROCESSED_DATA/crsp.parquet

    Columns after processing:
        permno, permco, date, ret, retx, prc, altprc, vol, shrout,
        bid, ask, cfacpr, cfacshr, shrcd, exchcd, siccd, naics,
        hsiccd, ticker, comnam, mktcap, spread
    """
    def _raw():
        raw_file = PATH["RAW_DATA"] / "crsp_monthly.parquet"
        if not raw_file.exists():
            raise FileNotFoundError(
                f"Raw CRSP file not found at {raw_file}.\n"
                "Run:  .venv/bin/python wrds_import.py crsp"
            )
        df = pd.read_parquet(raw_file)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["permno", "date"]).reset_index(drop=True)
        return df

    return _load_or_reload("crsp", _raw, fmt="parquet", reload=reload)


# ── JKP characteristics ──────────────────────────────────────────────────────

def load_jkp(country_set: str = "us", reload: bool = False) -> pd.DataFrame:
    """
    Load JKP stock-level characteristics (153 published factors).

    Parameters
    ----------
    country_set : str
        Which country set to load ('us', 'developed', 'all', ...).
        Must match the suffix used by wrds_import.py.
    reload : bool
        If True, rebuild the cache from the raw parquet.

    Raw source: RAW_DATA/jkp_characteristics_{country_set}.parquet
                (created by wrds_import.py)
    Cache:      PROCESSED_DATA/jkp_{country_set}.parquet

    Key columns:
        id, eom, excntry, gvkey, permno, size_grp, me,
        ret_exc_lead1m  (1-month-ahead excess return),
        + 153 characteristic columns
    """
    cache_name = f"jkp_{country_set}"

    def _raw():
        raw_file = PATH["RAW_DATA"] / f"jkp_characteristics_{country_set}.parquet"
        if not raw_file.exists():
            raise FileNotFoundError(
                f"Raw JKP file not found at {raw_file}.\n"
                f"Run:  .venv/bin/python wrds_import.py jkp"
            )
        df = pd.read_parquet(raw_file)
        df["eom"] = pd.to_datetime(df["eom"])
        df = df.sort_values(["permno", "eom"]).reset_index(drop=True)
        return df

    return _load_or_reload(cache_name, _raw, fmt="parquet", reload=reload)


# ── JKP factor returns ───────────────────────────────────────────────────────

def load_jkp_factor_returns(weighting: str = "vw", reload: bool = False) -> pd.DataFrame:
    """
    Load JKP long-short factor returns (all countries, all factors, monthly).

    Downloaded from jkpfactors.com.

    Parameters
    ----------
    weighting : str
        Portfolio weighting scheme: 'vw' (value-weighted),
        'ew' (equal-weighted), or 'vw_cap' (value-weighted capped).
    reload : bool
        If True, rebuild the cache from the raw CSV.

    Raw source: RAW_DATA/jkp_factor_returns_all_{weighting}.csv
    Cache:      PROCESSED_DATA/jkp_factor_returns_{weighting}.parquet

    Columns:
        location   – country code (e.g. 'usa', 'deu', ...)
        name       – factor abbreviation (153 factors)
        freq       – 'monthly'
        weighting  – 'vw', 'ew', or 'vw_cap'
        direction  – sign convention (1 or -1)
        n_stocks   – number of stocks in the portfolio
        n_stocks_min – minimum stocks in any leg
        date       – end-of-month date
        ret        – long-short portfolio return
    """
    valid = ("vw", "ew", "vw_cap")
    if weighting not in valid:
        raise ValueError(f"weighting must be one of {valid}, got '{weighting}'")

    cache_name = f"jkp_factor_returns_{weighting}"

    def _raw():
        raw_file = PATH["RAW_DATA"] / f"jkp_factor_returns_all_{weighting}.csv"
        if not raw_file.exists():
            raise FileNotFoundError(
                f"Raw JKP factor returns not found at {raw_file}.\n"
                "Download from https://jkpfactors.com/ and place in RAW_DATA."
            )
        df = pd.read_csv(raw_file)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["location", "name", "date"]).reset_index(drop=True)
        return df

    return _load_or_reload(cache_name, _raw, fmt="parquet", reload=reload)


# ── OptionMetrics aggregated volume / open interest ──────────────────────────

def load_optionm_agg(reload: bool = False) -> pd.DataFrame:
    """
    Load aggregated option volume & open interest from OptionMetrics.

    Raw source: RAW_DATA/optionm_agg.parquet     (created by wrds_import.py)
                RAW_DATA/optionm_secnmd.parquet   (secid → CUSIP mapping)
    Cache:      PROCESSED_DATA/optionm_agg.parquet

    Granularity: secid × month × call/put × DTE bucket × moneyness bucket

    Columns after processing:
        secid           OptionMetrics security ID (underlying)
        ym              year-month (datetime)
        cp_flag         'C' or 'P'
        dte_bucket      'short' (≤30d), 'medium' (31-180d), 'long' (>180d)
        moneyness       'ITM', 'ATM', 'OTM'  (relative to call/put convention)
        total_volume    sum of daily contract volume in the bucket
        total_oi        sum of daily open interest in the bucket
        n_obs           number of contract-day observations aggregated
        avg_iv          average implied volatility in the bucket
        avg_abs_delta   average |delta| in the bucket
        cusip           8-digit CUSIP from secnmd (for linking to CRSP)
        ticker          ticker symbol from secnmd

    To merge with CRSP (permno), join on cusip:
        crsp = load_crsp()
        # CRSP's ncusip is 8-digit historical CUSIP
        merged = optionm.merge(crsp[['permno','date','ncusip']].drop_duplicates(),
                               left_on=['cusip','ym'], right_on=['ncusip','date'],
                               how='left')
    """
    def _raw():
        raw_file = PATH["RAW_DATA"] / "optionm_agg.parquet"
        if not raw_file.exists():
            raise FileNotFoundError(
                f"Raw OptionMetrics file not found at {raw_file}.\n"
                "Run:  .venv/bin/python wrds_import.py optionm"
            )
        df = pd.read_parquet(raw_file)
        df["ym"] = pd.to_datetime(df["ym"])

        # ── attach CUSIP/ticker from secnmd for downstream permno linking ──
        # Take the latest CUSIP/ticker per secid (most identifiers are
        # stable; the few that change mid-life we take the final value).
        secnmd_file = PATH["RAW_DATA"] / "optionm_secnmd.parquet"
        if secnmd_file.exists():
            secnmd = pd.read_parquet(secnmd_file)
            secnmd["effect_date"] = pd.to_datetime(secnmd["effect_date"])
            secnmd["secid"] = secnmd["secid"].astype("Int64")
            # Keep the latest record per secid
            secnmd = (secnmd.sort_values("effect_date")
                      .drop_duplicates(subset=["secid"], keep="last"))
            secnmd_map = secnmd[["secid", "cusip", "ticker"]].copy()
            df = df.merge(secnmd_map, on="secid", how="left")

        df = df.sort_values(["secid", "ym", "cp_flag", "dte_bucket", "moneyness"])
        df = df.reset_index(drop=True)
        return df

    return _load_or_reload("optionm_agg", _raw, fmt="parquet", reload=reload)


# ── JKP factor details (theme mapping) ───────────────────────────────────

def load_jkp_factor_details(reload: bool = False) -> pd.DataFrame:
    """
    Load JKP factor details (theme/group mapping, direction, citations).

    Raw source: RAW_DATA/jkp_factor_details.xlsx
                (downloaded from github.com/bkelly-lab/jkp-data)
    Cache:      PROCESSED_DATA/jkp_factor_details.parquet

    Columns after processing (153 published factors only):
        name            JKP factor abbreviation (matches factor returns 'name')
        name_full       descriptive factor name
        theme           factor theme (Momentum, Value, Profitability,
                        Investment, Intangibles, Trading Frictions, New)
        direction       sign convention (1 or -1)
        cite            original paper citation
    """
    def _raw():
        raw_file = PATH["RAW_DATA"] / "jkp_factor_details.xlsx"
        if not raw_file.exists():
            raise FileNotFoundError(
                f"JKP factor details not found at {raw_file}.\n"
                "Download from https://github.com/bkelly-lab/jkp-data/raw/"
                "main/data/factor_details.xlsx and place in RAW_DATA."
            )
        df = pd.read_excel(raw_file)
        # Keep only the 153 published JKP factors
        df = df.dropna(subset=["abr_jkp"]).copy()
        df = df[["abr_jkp", "name_new", "group", "direction", "cite"]].copy()
        df = df.rename(columns={
            "abr_jkp": "name",
            "name_new": "name_full",
            "group": "theme",
        })
        df["theme"] = df["theme"].str.strip().str.title()
        df = df.sort_values("name").reset_index(drop=True)
        return df

    return _load_or_reload("jkp_factor_details", _raw, fmt="parquet", reload=reload)
