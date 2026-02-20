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
