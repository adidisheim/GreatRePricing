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


# ── example (uncomment / adapt when data is ready) ────────────────────────────
# def load_crsp(reload: bool = False) -> pd.DataFrame:
#     """Load CRSP monthly stock file."""
#     def _raw():
#         df = pd.read_csv(PATH['RAW_DATA'] / 'crsp_monthly.csv', parse_dates=['date'])
#         df = df[df['shrcd'].isin([10, 11])]  # common shares only
#         df = df[df['exchcd'].isin([1, 2, 3])]  # NYSE/AMEX/NASDAQ
#         return df
#     return _load_or_reload('crsp', _raw, fmt='parquet', reload=reload)
