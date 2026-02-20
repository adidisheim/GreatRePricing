"""
Statistical helper functions.

Add project-specific statistics utilities here:
    - Newey-West standard errors
    - Fama-MacBeth regressions
    - t-statistics with clustered SEs
    - etc.
"""

import numpy as np
import pandas as pd


def newey_west_t(y: pd.Series, lags: int = None) -> tuple:
    """
    Newey-West t-statistic for the mean of a time series.

    Parameters
    ----------
    y : pd.Series
        Time series of returns or residuals.
    lags : int, optional
        Number of lags.  Defaults to int(4*(len(y)/100)^(2/9)).

    Returns
    -------
    (mean, t_stat, se)
    """
    y = y.dropna()
    n = len(y)
    if lags is None:
        lags = int(4 * (n / 100) ** (2 / 9))

    mean = y.mean()
    demean = y.values - mean

    # Bartlett kernel
    gamma_0 = np.dot(demean, demean) / n
    weighted_sum = 0.0
    for j in range(1, lags + 1):
        gamma_j = np.dot(demean[j:], demean[:-j]) / n
        weighted_sum += 2 * (1 - j / (lags + 1)) * gamma_j

    nw_var = (gamma_0 + weighted_sum) / n
    se = np.sqrt(nw_var)
    t_stat = mean / se if se > 0 else np.nan

    return mean, t_stat, se
