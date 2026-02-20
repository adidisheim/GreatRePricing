"""
Portfolio construction utilities.

Add project-specific portfolio helpers here:
    - Quantile sorts (single / double)
    - Value- and equal-weighted returns
    - Long-short spread construction
    - etc.
"""

import pandas as pd
import numpy as np


def quantile_sort(
    df: pd.DataFrame,
    sort_var: str,
    ret_var: str,
    n_portfolios: int = 5,
    weight_var: str = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Single-sort portfolio formation.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with at least (date_col, sort_var, ret_var).
    sort_var : str
        Column used for sorting into quantiles.
    ret_var : str
        Column with the forward return.
    n_portfolios : int
        Number of quantile buckets.
    weight_var : str, optional
        Column for value weighting.  None → equal weight.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame with columns [date, portfolio, ret].
    """

    def _assign(group):
        group = group.dropna(subset=[sort_var])
        group["portfolio"] = pd.qcut(
            group[sort_var], n_portfolios, labels=False, duplicates="drop"
        )
        group["portfolio"] += 1  # 1-indexed
        return group

    df = df.groupby(date_col, group_keys=False).apply(_assign)

    if weight_var is not None:
        def _wavg(g):
            w = g[weight_var]
            return np.average(g[ret_var], weights=w) if w.sum() > 0 else np.nan

        result = (
            df.groupby([date_col, "portfolio"])
            .apply(_wavg)
            .rename("ret")
            .reset_index()
        )
    else:
        result = (
            df.groupby([date_col, "portfolio"])[ret_var]
            .mean()
            .rename("ret")
            .reset_index()
        )

    return result
