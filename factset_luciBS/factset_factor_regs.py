"""
Factor-level dIO panel construction and panel regressions.

Point 1 (US): For each factor, compute the value-weighted change in IO for
the long and short legs (tercile sorts on JKP characteristics). The factor-level
dIO is VW_dIO_long - VW_dIO_short. Regress factor returns on factor dIO using
PanelOLS at quarterly, semiannual, and annual frequencies.

Point 2 (all countries): Use US-computed dIO as regressor for all-country
factor returns. The entity dimension is factor x country.

Usage:
  cd GreatRePricing
  PYTHONPATH=. .venv/Scripts/python.exe factset_luciBS/factset_factor_regs.py
"""

from config import PATH
import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from linearmodels.panel import PanelOLS, PooledOLS

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
PANEL_CACHE = PATH['INTERMEDIARY_RESULTS'] / 'factor_dio_panel.parquet'
RESULTS_CACHE = PATH['INTERMEDIARY_RESULTS'] / 'factor_dio_reg_results.pkl'

META_COLS = ['id', 'eom', 'excntry', 'gvkey', 'permno', 'size_grp', 'me',
             'ret_exc_lead1m']

# ══════════════════════════════════════════════════════════════════════════════
# PART A: BUILD FACTOR-LEVEL dIO PANEL
# ══════════════════════════════════════════════════════════════════════════════

def load_ferreira_us_io() -> pd.DataFrame:
    """Load Ferreira IO for US stocks, keyed by (factset_entity_id, rquarter)."""
    print("  Loading Ferreira ownership (US) ...")
    df = pd.read_parquet(
        PATH['RAW_DATA'] / 'ferreira_ownership.parquet',
        columns=['factset_entity_id', 'quarter', 'rquarter', 'sec_country', 'io', 'io_for', 'io_from_US']
    )
    df = df[df['sec_country'] == 'US'].copy()
    df['rquarter'] = pd.to_datetime(df['rquarter'])
    df = df.sort_values(['factset_entity_id', 'rquarter']).reset_index(drop=True)

    # Compute dio = io(t) - io(t-1) within entity
    df['io_lag'] = df.groupby('factset_entity_id')['io'].shift(1)
    df['dio'] = df['io'] - df['io_lag']
    df = df.dropna(subset=['dio'])

    df['io_for'] = df['io_for'].fillna(0)
    df['io_from_US'] = df['io_from_US'].fillna(0)
    # Compute foreign share of US IO: what fraction of the stock's US IO is "foreign-directed"
    # For US stocks: io_from_US = US institutions' holdings. io_for = foreign institutions' holdings.
    # US foreign share = io_for / io (fraction of total IO that is foreign)
    df['io_for_share'] = df['io_for'] / df['io'].replace(0, np.nan)
    df['io_for_share'] = df['io_for_share'].fillna(0)

    print(f"    {len(df):,} entity-quarter obs with dIO")
    return df[['factset_entity_id', 'rquarter', 'io', 'dio', 'io_for', 'io_from_US', 'io_for_share']].copy()


def load_crosswalk() -> pd.DataFrame:
    """Load factset_entity_id -> permno crosswalk."""
    cw = pd.read_parquet(PATH['PROCESSED_DATA'] / 'factset_crsp_crosswalk.parquet')
    # Keep one row per permno (some permnos map to multiple entities  -- rare)
    cw = cw.drop_duplicates(subset='permno', keep='first')
    return cw[['factset_entity_id', 'permno']].copy()


def load_factor_details() -> pd.DataFrame:
    """Load factor direction info."""
    return pd.read_parquet(PATH['PROCESSED_DATA'] / 'jkp_factor_details.parquet')


def quarter_end_to_rquarter(eom: pd.Timestamp) -> pd.Timestamp:
    """Map a quarter-end eom date to the *next* quarter's rquarter.

    The JKP sort date (eom at quarter t-1 end) maps to Ferreira quarter t.
    E.g., eom=2004-12-31 is the sort for Q1 2005 -> rquarter=2005-03-31.
    """
    return eom + pd.offsets.QuarterEnd(1)


def build_factor_dio_panel(reload: bool = False) -> pd.DataFrame:
    """
    Build the factor-level dIO panel for US stocks.

    For each factor f and quarter t:
      1. Sort stocks into terciles using characteristic f at eom = quarter(t-1) end
      2. Match stocks to Ferreira IO via crosswalk (permno -> factset_entity_id)
      3. Compute VW dIO for long and short legs
      4. factor_dio = VW_dIO_long - VW_dIO_short

    Returns
    -------
    pd.DataFrame with columns: factor, location, quarter_date, factor_dio, factor_ret
    """
    if not reload and PANEL_CACHE.exists():
        print("Loading cached factor dIO panel ...")
        panel = pd.read_parquet(PANEL_CACHE)
        panel['quarter_date'] = pd.to_datetime(panel['quarter_date'])
        print(f"  {len(panel):,} rows ({panel['factor'].nunique()} factors, "
              f"{panel['quarter_date'].nunique()} quarters)")
        return panel

    print("Building factor-level dIO panel from scratch ...")
    t0 = time.time()

    # Load supporting data
    ferreira = load_ferreira_us_io()
    crosswalk = load_crosswalk()
    details = load_factor_details()
    factor_names = details['name'].tolist()
    direction_map = details.set_index('name')['direction'].to_dict()

    # ── Load factor returns (US) and aggregate to quarterly ────────────────
    print("  Loading JKP factor returns (US) ...")
    fr_all = pd.read_parquet(PATH['PROCESSED_DATA'] / 'jkp_factor_returns_vw_cap.parquet')
    fr_usa = fr_all[fr_all['location'] == 'usa'].copy()
    fr_usa['date'] = pd.to_datetime(fr_usa['date'])
    # Sign-correct returns
    fr_usa['ret_signed'] = fr_usa['ret'] * fr_usa['direction']
    # Quarterly aggregation: sum monthly returns within each quarter
    fr_usa['quarter_date'] = fr_usa['date'] + pd.offsets.QuarterEnd(0)
    fr_q = (fr_usa.groupby(['name', 'quarter_date'])['ret_signed']
            .sum().reset_index().rename(columns={'ret_signed': 'factor_ret'}))

    # ── Merge Ferreira with crosswalk to get permno-level IO ──────────────
    print("  Merging Ferreira with crosswalk ...")
    io_data = ferreira.merge(crosswalk, on='factset_entity_id', how='inner')
    # permno is int in crosswalk; JKP permno is Float64
    io_data['permno'] = io_data['permno'].astype('Int64')
    print(f"    {len(io_data):,} permno-quarter obs with dIO")

    # ── Determine which quarter-end dates to use for sorting ──────────────
    # Ferreira quarters available
    ferreira_quarters = sorted(ferreira['rquarter'].unique())
    # For each Ferreira quarter t, the sort date is the previous quarter end
    # E.g., for rquarter 2005-03-31, sort date = 2004-12-31
    # Build mapping: sort_eom -> rquarter
    sort_quarters = []
    for rq in ferreira_quarters:
        rq_ts = pd.Timestamp(rq)
        sort_eom = rq_ts - pd.offsets.QuarterEnd(1)
        sort_quarters.append({'sort_eom': sort_eom, 'rquarter': rq_ts})
    sort_q_df = pd.DataFrame(sort_quarters)

    # ── Process factors in batches to reduce parquet I/O ──────────────────
    BATCH_SIZE = 20
    n_factors = len(factor_names)
    n_batches = (n_factors + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Processing {n_factors} factors in {n_batches} batches ...")
    all_rows = []

    # Pre-index io_data by rquarter for fast lookup
    io_by_quarter = {rq: grp[['permno', 'dio', 'io', 'io_for', 'io_for_share']].copy()
                     for rq, grp in io_data.groupby('rquarter')}
    sort_eom_list = sort_q_df[['sort_eom', 'rquarter']].values.tolist()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, n_factors)
        batch_factors = factor_names[batch_start:batch_end]
        elapsed = time.time() - t0
        print(f"    Batch {batch_idx+1}/{n_batches} "
              f"[{batch_start+1}-{batch_end}/{n_factors}] ({elapsed:.0f}s)")

        # Load batch columns from parquet
        cols_needed = ['permno', 'eom', 'me'] + batch_factors
        try:
            chars = pd.read_parquet(
                PATH['RAW_DATA'] / 'jkp_characteristics_us.parquet',
                columns=cols_needed
            )
        except Exception as e:
            print(f"      ERROR loading batch: {e}")
            continue

        chars['eom'] = pd.to_datetime(chars['eom'])
        chars['permno'] = chars['permno'].astype('Int64')

        # Keep only quarter-end months (3, 6, 9, 12) for sorting
        chars = chars[chars['eom'].dt.month.isin([3, 6, 9, 12])].copy()

        # Process each factor in the batch
        for factor in batch_factors:
            direction = direction_map[factor]

            # Subset to rows with non-null characteristic and me
            fchars = chars[['permno', 'eom', 'me', factor]].dropna(
                subset=[factor, 'me']).copy()
            if len(fchars) == 0:
                continue

            # For each sort_eom, assign terciles
            for sort_eom, rquarter in sort_eom_list:
                sort_eom = pd.Timestamp(sort_eom)
                rquarter = pd.Timestamp(rquarter)

                ch = fchars[fchars['eom'] == sort_eom]
                if len(ch) < 30:
                    continue

                # Rank into terciles
                try:
                    tercile = pd.qcut(ch[factor], 3, labels=[1, 2, 3]).astype(int)
                except ValueError:
                    # Too many duplicate values for qcut
                    continue

                if direction == 1:
                    long_t, short_t = 3, 1
                else:
                    long_t, short_t = 1, 3

                long_mask = tercile == long_t
                short_mask = tercile == short_t
                long_stocks = ch.loc[long_mask, ['permno', 'me']]
                short_stocks = ch.loc[short_mask, ['permno', 'me']]

                # Match to Ferreira IO at this quarter
                io_q = io_by_quarter.get(rquarter)
                if io_q is None:
                    continue

                long_merged = long_stocks.merge(io_q, on='permno', how='inner')
                if len(long_merged) < 5:
                    continue
                short_merged = short_stocks.merge(io_q, on='permno', how='inner')
                if len(short_merged) < 5:
                    continue

                vw_dio_long = np.average(long_merged['dio'],
                                         weights=long_merged['me'])
                vw_dio_short = np.average(short_merged['dio'],
                                          weights=short_merged['me'])
                vw_io_long = np.average(long_merged['io'],
                                        weights=long_merged['me'])
                vw_io_short = np.average(short_merged['io'],
                                         weights=short_merged['me'])
                vw_io_for_long = np.average(long_merged['io_for'],
                                            weights=long_merged['me'])
                vw_io_for_short = np.average(short_merged['io_for'],
                                             weights=short_merged['me'])
                vw_fs_long = np.average(long_merged['io_for_share'],
                                        weights=long_merged['me'])
                vw_fs_short = np.average(short_merged['io_for_share'],
                                         weights=short_merged['me'])

                all_rows.append({
                    'factor': factor,
                    'quarter_date': rquarter,
                    'factor_dio': vw_dio_long - vw_dio_short,
                    'factor_io': vw_io_long - vw_io_short,
                    'factor_io_for': vw_io_for_long - vw_io_for_short,
                    'factor_io_for_share': vw_fs_long - vw_fs_short,
                    'n_long': len(long_merged),
                    'n_short': len(short_merged),
                })

        del chars  # free memory

    # ── Assemble panel ────────────────────────────────────────────────────
    print("  Assembling panel ...")
    panel = pd.DataFrame(all_rows)
    panel['location'] = 'usa'

    # Merge with quarterly factor returns
    panel = panel.merge(fr_q, left_on=['factor', 'quarter_date'],
                        right_on=['name', 'quarter_date'], how='left')
    panel = panel.drop(columns=['name'], errors='ignore')

    # Save
    panel.to_parquet(PANEL_CACHE, index=False)
    elapsed = time.time() - t0
    print(f"  Panel built: {len(panel):,} rows, {panel['factor'].nunique()} factors, "
          f"{panel['quarter_date'].nunique()} quarters ({elapsed:.0f}s)")
    print(f"  Cached -> {PANEL_CACHE}")

    return panel


# ══════════════════════════════════════════════════════════════════════════════
# PART A2: EXTEND PANEL TO ALL COUNTRIES (Point 2)
# ══════════════════════════════════════════════════════════════════════════════

def build_all_countries_panel(us_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build all-countries panel by pairing US-computed dIO with each country's
    factor returns.

    The US dIO is used as the regressor for every country because it measures
    the delta in IO ownership in the equities traded to obtain the factor.

    Parameters
    ----------
    us_panel : pd.DataFrame
        US factor-level dIO panel (from build_factor_dio_panel).

    Returns
    -------
    pd.DataFrame with columns: factor, location, quarter_date, factor_dio,
                                factor_ret, entity_id
    """
    print("Building all-countries panel ...")

    # Load all-country factor returns
    fr_all = pd.read_parquet(PATH['PROCESSED_DATA'] / 'jkp_factor_returns_vw_cap.parquet')
    fr_all['date'] = pd.to_datetime(fr_all['date'])
    fr_all['ret_signed'] = fr_all['ret'] * fr_all['direction']
    fr_all['quarter_date'] = fr_all['date'] + pd.offsets.QuarterEnd(0)

    # Quarterly aggregation per (location, factor)
    fr_q = (fr_all.groupby(['location', 'name', 'quarter_date'])['ret_signed']
            .sum().reset_index().rename(columns={'name': 'factor',
                                                 'ret_signed': 'factor_ret'}))

    # US dIO and IO level keyed by (factor, quarter_date)
    keep_cols = ['factor', 'quarter_date', 'factor_dio']
    for extra in ['factor_io', 'factor_io_for', 'factor_io_for_share']:
        if extra in us_panel.columns:
            keep_cols.append(extra)
    us_dio = us_panel[keep_cols].copy()

    # Cross-join: for each country's factor returns, attach the US dIO
    panel = fr_q.merge(us_dio, on=['factor', 'quarter_date'], how='inner')

    # Create composite entity ID for panel regressions
    panel['entity_id'] = panel['factor'] + '_' + panel['location']

    n_countries = panel['location'].nunique()
    n_factors = panel['factor'].nunique()
    print(f"  All-countries panel: {len(panel):,} rows, "
          f"{n_factors} factors, {n_countries} countries, "
          f"{panel['quarter_date'].nunique()} quarters")

    return panel


# ══════════════════════════════════════════════════════════════════════════════
# FREQUENCY AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_to_frequency(panel: pd.DataFrame, freq: str,
                           entity_col: str = 'factor') -> pd.DataFrame:
    """
    Aggregate quarterly panel to semiannual or annual frequency.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have columns: [entity_col, 'quarter_date', 'factor_dio', 'factor_ret']
    freq : str
        'Q' (quarterly, passthrough), 'S' (semiannual), 'A' (annual)
    entity_col : str
        Column identifying the entity (e.g. 'factor' or 'entity_id')

    Returns
    -------
    pd.DataFrame with columns: [entity_col, 'date', 'factor_dio', 'factor_ret']
    """
    keep = [entity_col, 'quarter_date', 'factor_dio', 'factor_ret']
    for extra in panel.columns:
        if extra.startswith('factor_io') and extra not in keep:
            keep.append(extra)
    df = panel[keep].copy()
    df = df.dropna(subset=['factor_dio', 'factor_ret'])

    if freq == 'Q':
        df = df.rename(columns={'quarter_date': 'date'})
        return df

    df['year'] = df['quarter_date'].dt.year

    if freq == 'S':
        # H1 = Q1+Q2 (months 3,6), H2 = Q3+Q4 (months 9,12)
        df['half'] = np.where(df['quarter_date'].dt.month <= 6, 1, 2)
        grp = df.groupby([entity_col, 'year', 'half'])
        agg_d = {'factor_ret': 'sum', 'factor_dio': 'mean', 'n_q': ('factor_dio', 'count')}
        agg = grp.agg(
            factor_ret=('factor_ret', 'sum'),
            factor_dio=('factor_dio', 'mean'),
            n_q=('factor_dio', 'count'),
            **{c: (c, 'mean') for c in df.columns
               if c.startswith('factor_io') and c in df.columns}
        ).reset_index()
        # Keep only complete semesters (2 quarters)
        agg = agg[agg['n_q'] == 2].drop(columns='n_q')
        # Construct a date for the period end
        agg['date'] = pd.to_datetime(
            agg['year'].astype(str) + '-' +
            np.where(agg['half'] == 1, '06-30', '12-31')
        )
        agg = agg.drop(columns=['year', 'half'])
        return agg

    if freq == 'A':
        grp = df.groupby([entity_col, 'year'])
        agg = grp.agg(
            factor_ret=('factor_ret', 'sum'),
            factor_dio=('factor_dio', 'mean'),
            n_q=('factor_dio', 'count'),
            **{c: (c, 'mean') for c in df.columns
               if c.startswith('factor_io') and c in df.columns}
        ).reset_index()
        # Keep only complete years (4 quarters)
        agg = agg[agg['n_q'] == 4].drop(columns='n_q')
        agg['date'] = pd.to_datetime(agg['year'].astype(str) + '-12-31')
        agg = agg.drop(columns='year')
        return agg

    raise ValueError(f"Unknown freq: {freq}")


# ══════════════════════════════════════════════════════════════════════════════
# PART B: REGRESSIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_panel_regression(panel_freq: pd.DataFrame, entity_col: str,
                         entity_fe: bool = True, time_fe: bool = False,
                         iv_col: str = 'factor_dio',
                         iv_cols: list = None) -> dict:
    """
    Run PanelOLS: factor_ret ~ iv_col(s) with specified FEs.

    If iv_cols is provided (list), use multiple regressors and return
    per-variable results. Otherwise use single iv_col.

    Returns
    -------
    dict with keys per IV: coef, tstat, pval + r2_within, nobs, n_entities
    """
    if iv_cols is None:
        iv_cols = [iv_col]

    df = panel_freq[[entity_col, 'date'] + iv_cols + ['factor_ret']].dropna().copy()

    if len(df) < 20:
        result = {iv: {'coef': np.nan, 'tstat': np.nan, 'pval': np.nan} for iv in iv_cols}
        result['r2_within'] = np.nan
        result['nobs'] = len(df)
        result['n_entities'] = 0
        if len(iv_cols) == 1:
            # Backward compat: flat dict
            return {'coef': np.nan, 'tstat': np.nan, 'pval': np.nan,
                    'r2_within': np.nan, 'nobs': len(df), 'n_entities': 0}
        return result

    df = df.set_index([entity_col, 'date'])

    if not entity_fe and not time_fe:
        mod = PooledOLS(df['factor_ret'], df[iv_cols], check_rank=False)
    else:
        fe_kw = {}
        if entity_fe and time_fe:
            fe_kw = {'entity_effects': True, 'time_effects': True}
        elif entity_fe:
            fe_kw = {'entity_effects': True}
        elif time_fe:
            fe_kw = {'time_effects': True}
        mod = PanelOLS(df['factor_ret'], df[iv_cols], check_rank=False, **fe_kw)
    res = mod.fit(cov_type='clustered', cluster_entity=True)

    r2 = res.rsquared_within if hasattr(res, 'rsquared_within') else res.rsquared

    if len(iv_cols) == 1:
        iv = iv_cols[0]
        return {
            'coef': res.params[iv], 'tstat': res.tstats[iv], 'pval': res.pvalues[iv],
            'r2_within': r2, 'nobs': res.nobs,
            'n_entities': df.index.get_level_values(0).nunique(),
        }
    else:
        result = {}
        for iv in iv_cols:
            result[iv] = {'coef': res.params[iv], 'tstat': res.tstats[iv], 'pval': res.pvalues[iv]}
        result['r2_within'] = r2
        result['nobs'] = res.nobs
        result['n_entities'] = df.index.get_level_values(0).nunique()
        return result


def run_all_regressions(us_panel: pd.DataFrame,
                        all_panel: pd.DataFrame) -> dict:
    """
    Run all regression specifications for Points 1 and 2.

    Returns a nested dict:
      results[point][freq][fe_spec] = regression result dict
    """
    results = {}

    # ── Point 1: US only ──────────────────────────────────────────────────
    print("\n=== Point 1: US regressions ===")
    results['P1'] = {}
    for freq in ['Q', 'S', 'A']:
        results['P1'][freq] = {}
        panel_freq = aggregate_to_frequency(us_panel, freq, entity_col='factor')
        for fe_label, fe_kw in [('FE_entity', dict(entity_fe=True, time_fe=False)),
                                ('FE_entity_time', dict(entity_fe=True, time_fe=True))]:
            res = run_panel_regression(panel_freq, entity_col='factor', **fe_kw)
            results['P1'][freq][fe_label] = res
            stars = '***' if res['pval'] < 0.01 else '**' if res['pval'] < 0.05 else '*' if res['pval'] < 0.10 else ''
            print(f"  {freq} | {fe_label:20s} | coef={res['coef']:+.4f} | "
                  f"t={res['tstat']:+.2f}{stars} | R2w={res['r2_within']:.4f} | "
                  f"N={res['nobs']}")

    # ── Add lags to panels ──────────────────────────────────────────────
    print("  Adding factor IO lags...")
    for panel, ecol in [(us_panel, 'factor'), (all_panel, 'entity_id')]:
        panel.sort_values([ecol, 'quarter_date'], inplace=True)
        for col in ['factor_io', 'factor_io_for', 'factor_io_for_share']:
            if col in panel.columns:
                panel[f'{col}_lag1'] = panel.groupby(ecol)[col].shift(1)
                panel[f'{col}_lag2'] = panel.groupby(ecol)[col].shift(2)

    # ── Point 1b: US only (IO level + foreign IO level + lags) ────────
    print("\n=== Point 1b: US regressions (IO + foreign IO levels + lags) ===")
    results['P1b'] = {}
    P1b_IVS = ['factor_io', 'factor_io_lag1', 'factor_io_lag2',
               'factor_io_for', 'factor_io_for_lag1', 'factor_io_for_lag2',
               'factor_io_for_share', 'factor_io_for_share_lag1', 'factor_io_for_share_lag2']
    for freq in ['Q']:
        results['P1b'][freq] = {}
        panel_freq = aggregate_to_frequency(us_panel, freq, entity_col='factor')
        for fe_label, fe_kw in [('FE_none', dict(entity_fe=False, time_fe=False)),
                                ('FE_entity', dict(entity_fe=True, time_fe=False)),
                                ('FE_entity_time', dict(entity_fe=True, time_fe=True))]:
            res = run_panel_regression(panel_freq, entity_col='factor', iv_cols=P1b_IVS, **fe_kw)
            results['P1b'][freq][fe_label] = res
            print(f"  {freq} | {fe_label:20s} | R2w={res['r2_within']:.4f} | N={res['nobs']}")
            for iv in P1b_IVS:
                c, t, p = res[iv]['coef'], res[iv]['tstat'], res[iv]['pval']
                s = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                print(f"    {iv}: {c:+.4f}{s} (t={t:+.2f})")

    # ── Point 2: All countries (IO level + foreign IO level) ────────────
    print("\n=== Point 2: All-countries regressions (IO + foreign IO levels) ===")
    results['P2'] = {}
    P2_IVS = ['factor_io', 'factor_io_lag1', 'factor_io_lag2',
               'factor_io_for', 'factor_io_for_lag1', 'factor_io_for_lag2',
               'factor_io_for_share', 'factor_io_for_share_lag1', 'factor_io_for_share_lag2']
    for freq in ['Q', 'S', 'A']:
        results['P2'][freq] = {}
        panel_freq = aggregate_to_frequency(all_panel, freq, entity_col='entity_id')
        for fe_label, fe_kw in [('FE_entity', dict(entity_fe=True, time_fe=False)),
                                ('FE_entity_time', dict(entity_fe=True, time_fe=True))]:
            res = run_panel_regression(panel_freq, entity_col='entity_id', iv_cols=P2_IVS, **fe_kw)
            results['P2'][freq][fe_label] = res
            print(f"  {freq} | {fe_label:20s} | R2w={res['r2_within']:.4f} | N={res['nobs']}")
            for iv in P2_IVS:
                c, t, p = res[iv]['coef'], res[iv]['tstat'], res[iv]['pval']
                s = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                print(f"    {iv}: {c:+.4f}{s} (t={t:+.2f})")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART C: STOCK-LEVEL CONTRIBUTION TO OPTIMAL PORTFOLIO (Points 3-4)
# ══════════════════════════════════════════════════════════════════════════════

STOCK_PANEL_CACHE = PATH['INTERMEDIARY_RESULTS'] / 'stock_contribution_panel.parquet'
WEIGHTS_CACHE = PATH['INTERMEDIARY_RESULTS'] / 'max_sharpe_roll48_6_weights.parquet'
STOCK_REG_CACHE = PATH['INTERMEDIARY_RESULTS'] / 'stock_contribution_reg_results.pkl'

# Minimum absolute portfolio weight to consider a factor "active" at a rebalance
WEIGHT_THRESHOLD = 0.001


def build_factor_weights_panel(reload: bool = False) -> pd.DataFrame:
    """
    Build a panel of max-Sharpe portfolio factor weights over time (US, vw_cap,
    rolling 48-month window, 6-month rebalance).

    Returns
    -------
    pd.DataFrame with columns: date, factor, weight
        One row per (rebalance date, factor) with non-zero weight.
    """
    if not reload and WEIGHTS_CACHE.exists():
        print("Loading cached factor weights ...")
        wdf = pd.read_parquet(WEIGHTS_CACHE)
        wdf['date'] = pd.to_datetime(wdf['date'])
        print(f"  {len(wdf):,} rows, {wdf['date'].nunique()} rebalance dates")
        return wdf

    print("Building factor weight panel from scratch ...")
    from data import load_jkp_factor_returns
    from compute_optimal_portfolio import build_max_sharpe_portfolio

    fr = load_jkp_factor_returns(weighting='vw_cap')
    fr_us = fr[fr['location'] == 'usa'].copy()
    fr_us['ret_signed'] = fr_us['ret'] * fr_us['direction']

    _, weight_records = build_max_sharpe_portfolio(
        fr_us, min_obs=48, rebal=6, window=48, return_weights=True
    )

    # Convert list of dicts to long DataFrame
    rows = []
    for rec in weight_records:
        dt = rec['date']
        for factor, w in rec['weights'].items():
            if abs(w) > 1e-10:
                rows.append({'date': dt, 'factor': factor, 'weight': w})
    wdf = pd.DataFrame(rows)
    wdf['date'] = pd.to_datetime(wdf['date'])

    wdf.to_parquet(WEIGHTS_CACHE, index=False)
    print(f"  Factor weights panel: {len(wdf):,} rows, "
          f"{wdf['date'].nunique()} rebalance dates")
    print(f"  Cached -> {WEIGHTS_CACHE}")
    return wdf


def _get_active_weights_at_quarter(wdf: pd.DataFrame, quarter_end: pd.Timestamp,
                                    threshold: float = WEIGHT_THRESHOLD) -> dict:
    """
    Get the portfolio factor weights active at a given quarter end.

    Uses the most recent rebalance date on or before quarter_end.

    Returns
    -------
    dict : {factor_name: weight} for factors with |weight| > threshold
    """
    eligible = wdf[wdf['date'] <= quarter_end]
    if len(eligible) == 0:
        return {}
    latest_date = eligible['date'].max()
    snap = eligible[eligible['date'] == latest_date]
    return {row['factor']: row['weight']
            for _, row in snap.iterrows()
            if abs(row['weight']) > threshold}


def build_stock_contribution_panel(reload: bool = False) -> pd.DataFrame:
    """
    Build stock-level contribution panel (US only).

    For each stock i in quarter t:
      - contribution_i = W_i * ret_i
        where W_i = sum_f (w_f * w_if)
              w_f = optimal portfolio weight for factor f
              w_if = VW weight of stock i within factor f's long/short leg
              ret_i = stock's excess return in quarter t

    Returns
    -------
    pd.DataFrame with columns:
        permno, quarter_date, contribution, stock_weight, ret_exc,
        dio, io, factset_entity_id
    """
    if not reload and STOCK_PANEL_CACHE.exists():
        print("Loading cached stock contribution panel ...")
        panel = pd.read_parquet(STOCK_PANEL_CACHE)
        panel['quarter_date'] = pd.to_datetime(panel['quarter_date'])
        print(f"  {len(panel):,} rows ({panel['permno'].nunique()} stocks, "
              f"{panel['quarter_date'].nunique()} quarters)")
        return panel

    print("\n" + "=" * 60)
    print("Building stock contribution panel from scratch ...")
    print("=" * 60)
    t0 = time.time()

    # ── Load supporting data ──────────────────────────────────────────────
    wdf = build_factor_weights_panel(reload=False)
    ferreira = load_ferreira_us_io()
    crosswalk = load_crosswalk()
    details = load_factor_details()
    direction_map = details.set_index('name')['direction'].to_dict()

    # Merge Ferreira with crosswalk to get permno-level IO
    io_data = ferreira.merge(crosswalk, on='factset_entity_id', how='inner')
    io_data['permno'] = io_data['permno'].astype('Int64')

    # Pre-index io_data by rquarter for fast lookup
    io_by_quarter = {}
    for rq, grp in io_data.groupby('rquarter'):
        io_by_quarter[rq] = grp.set_index('permno')[['dio', 'io', 'factset_entity_id']]

    # ── Determine quarters to process ─────────────────────────────────────
    ferreira_quarters = sorted(io_data['rquarter'].unique())
    min_weight_date = wdf['date'].min()
    ferreira_quarters = [rq for rq in ferreira_quarters
                         if pd.Timestamp(rq) >= min_weight_date]
    print(f"  Processing {len(ferreira_quarters)} quarters "
          f"({pd.Timestamp(ferreira_quarters[0]).strftime('%Y-%m') if ferreira_quarters else 'none'} to "
          f"{pd.Timestamp(ferreira_quarters[-1]).strftime('%Y-%m') if ferreira_quarters else 'none'})")

    # ── Pre-load returns data (permno, eom, ret_exc_lead1m) ONCE ──────────
    # This avoids re-reading the 2.6 GB file every quarter.
    print("  Pre-loading returns column from JKP characteristics ...")
    ret_all = pd.read_parquet(
        PATH['RAW_DATA'] / 'jkp_characteristics_us.parquet',
        columns=['permno', 'eom', 'ret_exc_lead1m']
    )
    ret_all['eom'] = pd.to_datetime(ret_all['eom'])
    ret_all['permno'] = ret_all['permno'].astype('Int64')
    # Pre-compute quarterly returns: for each quarter ending at rquarter,
    # sum ret_exc_lead1m from the 3 months (sort_eom, sort_eom+1m, sort_eom+2m)
    # assign quarter = eom + QuarterEnd(1) so that eom in Q4 -> next Q1, etc.
    # Actually: ret_exc_lead1m at eom is the return for eom+1month.
    # For rquarter (e.g. 2005-03-31), the 3 months are Jan, Feb, Mar 2005.
    # ret for Jan 2005 is ret_exc_lead1m at eom=2004-12-31
    # ret for Feb 2005 is ret_exc_lead1m at eom=2005-01-31
    # ret for Mar 2005 is ret_exc_lead1m at eom=2005-02-28
    # So the 3 eom months are: rquarter - QuarterEnd(1), + MonthEnd(1), + MonthEnd(2)
    # i.e. sort_eom, sort_eom+1m, sort_eom+2m -- which maps to rquarter
    # Assign each eom to the rquarter it contributes to:
    ret_all['rquarter'] = ret_all['eom'] + pd.offsets.QuarterEnd(1)
    qret_all = (ret_all.dropna(subset=['ret_exc_lead1m'])
                .groupby(['permno', 'rquarter'])['ret_exc_lead1m']
                .sum().reset_index()
                .rename(columns={'ret_exc_lead1m': 'ret_exc_q'}))
    # Index by rquarter for fast lookup
    qret_by_quarter = {rq: grp.set_index('permno')['ret_exc_q']
                       for rq, grp in qret_all.groupby('rquarter')}
    del ret_all, qret_all
    print(f"    Quarterly returns precomputed for {len(qret_by_quarter)} quarters")

    # ── Identify all unique factors needed across all quarters ────────────
    all_active_factors = set()
    quarter_weights = {}  # rquarter -> active_weights dict
    for rq in ferreira_quarters:
        rq = pd.Timestamp(rq)
        aw = _get_active_weights_at_quarter(wdf, rq)
        if aw:
            quarter_weights[rq] = aw
            all_active_factors.update(aw.keys())
    all_active_factors = sorted(all_active_factors)
    print(f"  {len(all_active_factors)} unique factors needed across all quarters")

    # ── Load characteristics ONCE for all needed sort dates ───────────────
    sort_eom_dates = set()
    for rq in quarter_weights:
        sort_eom_dates.add(rq - pd.offsets.QuarterEnd(1))
    sort_eom_dates = sorted(sort_eom_dates)

    print(f"  Loading characteristics for {len(sort_eom_dates)} sort dates ...")
    cols_needed = ['permno', 'eom', 'me'] + all_active_factors
    cols_needed = list(dict.fromkeys(cols_needed))

    chars_full = pd.read_parquet(
        PATH['RAW_DATA'] / 'jkp_characteristics_us.parquet',
        columns=cols_needed
    )
    chars_full['eom'] = pd.to_datetime(chars_full['eom'])
    chars_full['permno'] = chars_full['permno'].astype('Int64')
    # Drop rows with NA permno (gvkey-only rows without CRSP match)
    chars_full = chars_full.dropna(subset=['permno'])
    # Keep only quarter-end months and relevant sort dates
    chars_full = chars_full[chars_full['eom'].isin(sort_eom_dates)].copy()
    # Drop duplicates: keep first row per (permno, eom)
    chars_full = chars_full.drop_duplicates(subset=['permno', 'eom'], keep='first')

    # Index by eom for fast lookup
    chars_by_eom = {eom: grp.reset_index(drop=True)
                    for eom, grp in chars_full.groupby('eom')}
    del chars_full
    print(f"  Characteristics loaded for {len(chars_by_eom)} sort dates")

    # ── Quarter-by-quarter processing ─────────────────────────────────────
    all_stock_rows = []

    for qi, rq in enumerate(ferreira_quarters):
        rq = pd.Timestamp(rq)
        sort_eom = rq - pd.offsets.QuarterEnd(1)

        active_weights = quarter_weights.get(rq)
        if not active_weights:
            continue
        n_active = len(active_weights)

        if (qi + 1) % 10 == 0 or qi == 0:
            elapsed = time.time() - t0
            print(f"    Quarter {qi+1}/{len(ferreira_quarters)} "
                  f"({rq.strftime('%Y-%m')}) | {n_active} active factors | "
                  f"{elapsed:.0f}s")

        chars = chars_by_eom.get(sort_eom)
        if chars is None or len(chars) == 0:
            continue

        # Get precomputed quarterly returns
        qret_series = qret_by_quarter.get(rq)
        if qret_series is None:
            continue
        qret = qret_series.reset_index()
        qret.columns = ['permno', 'ret_exc_q']

        # ── Compute stock-level optimal portfolio weight W_i ──────────────
        # For each active factor, do tercile sort and compute VW weights
        # within legs, then sum across factors using vectorized operations.
        # Accumulate in a Series indexed by permno.
        weight_series = pd.Series(dtype='float64')  # permno -> cumulative W_i

        for factor, w_f in active_weights.items():
            direction = direction_map.get(factor, 1)

            # Get characteristic for this factor at sort_eom
            fcol = chars[['permno', 'me', factor]].dropna(subset=[factor, 'me']).copy()
            if len(fcol) < 30:
                continue

            # Tercile sort
            try:
                tercile = pd.qcut(fcol[factor], 3, labels=[1, 2, 3]).astype(int)
            except ValueError:
                continue

            if direction == 1:
                long_t, short_t = 3, 1
            else:
                long_t, short_t = 1, 3

            long_mask = tercile == long_t
            short_mask = tercile == short_t

            long_me = fcol.loc[long_mask, 'me']
            short_me = fcol.loc[short_mask, 'me']

            if len(long_me) == 0 or len(short_me) == 0:
                continue

            long_total = long_me.sum()
            short_total = short_me.sum()

            if long_total <= 0 or short_total <= 0:
                continue

            # Vectorized: compute w_f * (me_i / sum_leg) for long/short
            long_contrib = pd.Series(
                (w_f * long_me.values / long_total),
                index=fcol.loc[long_mask, 'permno'].values
            )
            short_contrib = pd.Series(
                (-w_f * short_me.values / short_total),
                index=fcol.loc[short_mask, 'permno'].values
            )

            # Accumulate
            factor_contrib = pd.concat([long_contrib, short_contrib])
            # Group by permno in case a stock appears in both (shouldn't happen, but safe)
            factor_contrib = factor_contrib.groupby(level=0).sum()
            weight_series = weight_series.add(factor_contrib, fill_value=0.0)

        if len(weight_series) == 0:
            continue

        # Convert to DataFrame
        sw_df = pd.DataFrame({
            'permno': weight_series.index,
            'stock_weight': weight_series.values
        })
        sw_df['permno'] = sw_df['permno'].astype('Int64')

        # Merge with quarterly returns
        sw_df = sw_df.merge(qret, on='permno', how='inner')

        # Contribution = W_i * ret_i
        sw_df['contribution'] = sw_df['stock_weight'] * sw_df['ret_exc_q']

        # Match to Ferreira dIO
        io_q = io_by_quarter.get(rq)
        if io_q is not None:
            sw_df = sw_df.merge(
                io_q[['dio', 'io', 'factset_entity_id']],
                left_on='permno', right_index=True, how='inner'
            )
        else:
            continue

        if len(sw_df) == 0:
            continue

        sw_df['quarter_date'] = rq
        sw_df = sw_df.rename(columns={'ret_exc_q': 'ret_exc'})
        all_stock_rows.append(
            sw_df[['permno', 'quarter_date', 'contribution', 'stock_weight',
                   'ret_exc', 'dio', 'io', 'factset_entity_id']].copy()
        )

    # ── Assemble panel ────────────────────────────────────────────────────
    if not all_stock_rows:
        print("  WARNING: No stock contribution rows produced!")
        return pd.DataFrame()

    panel = pd.concat(all_stock_rows, ignore_index=True)
    panel.to_parquet(STOCK_PANEL_CACHE, index=False)
    elapsed = time.time() - t0
    print(f"\n  Stock contribution panel: {len(panel):,} rows, "
          f"{panel['permno'].nunique()} stocks, "
          f"{panel['quarter_date'].nunique()} quarters ({elapsed:.0f}s)")
    print(f"  Cached -> {STOCK_PANEL_CACHE}")
    return panel


def aggregate_stock_panel_to_frequency(panel: pd.DataFrame,
                                        freq: str) -> pd.DataFrame:
    """
    Aggregate stock-quarter panel to semiannual or annual frequency.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have columns: [permno, quarter_date, contribution, dio]
    freq : str
        'Q' (passthrough), 'S' (semiannual), 'A' (annual)

    Returns
    -------
    pd.DataFrame with columns: [permno, date, contribution, dio]
    """
    df = panel[['permno', 'quarter_date', 'contribution', 'dio']].dropna().copy()

    if freq == 'Q':
        df = df.rename(columns={'quarter_date': 'date'})
        return df

    df['year'] = df['quarter_date'].dt.year

    if freq == 'S':
        df['half'] = np.where(df['quarter_date'].dt.month <= 6, 1, 2)
        grp = df.groupby(['permno', 'year', 'half'])
        agg = grp.agg(
            contribution=('contribution', 'sum'),
            dio=('dio', 'mean'),
            n_q=('dio', 'count')
        ).reset_index()
        agg = agg[agg['n_q'] == 2].drop(columns='n_q')
        agg['date'] = pd.to_datetime(
            agg['year'].astype(str) + '-' +
            np.where(agg['half'] == 1, '06-30', '12-31')
        )
        return agg[['permno', 'date', 'contribution', 'dio']]

    if freq == 'A':
        grp = df.groupby(['permno', 'year'])
        agg = grp.agg(
            contribution=('contribution', 'sum'),
            dio=('dio', 'mean'),
            n_q=('dio', 'count')
        ).reset_index()
        agg = agg[agg['n_q'] == 4].drop(columns='n_q')
        agg['date'] = pd.to_datetime(agg['year'].astype(str) + '-12-31')
        return agg[['permno', 'date', 'contribution', 'dio']]

    raise ValueError(f"Unknown freq: {freq}")


def run_stock_panel_regression(panel_freq: pd.DataFrame,
                                entity_col: str = 'permno',
                                entity_fe: bool = True,
                                time_fe: bool = False) -> dict:
    """
    Run PanelOLS: contribution ~ dio with specified FEs.

    Parameters
    ----------
    panel_freq : pd.DataFrame
        Must have [entity_col, 'date', 'contribution', 'dio'].
    entity_col : str
        Column to use as the entity index.
    entity_fe : bool
        Include entity fixed effects.
    time_fe : bool
        Include time fixed effects.

    Returns
    -------
    dict with keys: coef, tstat, pval, r2_within, nobs, n_entities
    """
    df = panel_freq[[entity_col, 'date', 'contribution', 'dio']].dropna().copy()

    if len(df) < 20:
        return {'coef': np.nan, 'tstat': np.nan, 'pval': np.nan,
                'r2_within': np.nan, 'nobs': len(df), 'n_entities': 0}

    # Ensure entity_col is suitable for panel index (convert permno to str)
    df[entity_col] = df[entity_col].astype(str)
    df = df.set_index([entity_col, 'date'])

    fe_kw = {}
    if entity_fe and time_fe:
        fe_kw = {'entity_effects': True, 'time_effects': True}
    elif entity_fe:
        fe_kw = {'entity_effects': True}
    elif time_fe:
        fe_kw = {'time_effects': True}

    mod = PanelOLS(df['contribution'], df[['dio']], check_rank=False, **fe_kw)
    res = mod.fit(cov_type='clustered', cluster_entity=True)

    return {
        'coef': res.params['dio'],
        'tstat': res.tstats['dio'],
        'pval': res.pvalues['dio'],
        'r2_within': res.rsquared_within,
        'nobs': res.nobs,
        'n_entities': df.index.get_level_values(0).nunique(),
    }


def run_stock_contribution_regressions(stock_panel: pd.DataFrame) -> dict:
    """
    Run all regression specifications for Points 3 and 4.

    Point 3 (US): stock-level contribution ~ dIO
      - Entity = permno (stock FE)
      - Specs: Stock FE | Stock + Time FE
      - Frequencies: Q, S, A

    Point 4 (all countries): requires all-country JKP characteristics.
      Currently only US is available, so Point 4 is skipped with a message.

    Returns
    -------
    dict: results[point][freq][fe_spec] = regression result dict
    """
    results = {}

    # ── Point 3: US stock-level regressions ───────────────────────────────
    print("\n=== Point 3: US stock-level contribution regressions ===")
    results['P3'] = {}
    for freq in ['Q', 'S', 'A']:
        results['P3'][freq] = {}
        panel_freq = aggregate_stock_panel_to_frequency(stock_panel, freq)
        for fe_label, fe_kw in [
            ('FE_stock', dict(entity_fe=True, time_fe=False)),
            ('FE_stock_time', dict(entity_fe=True, time_fe=True)),
        ]:
            res = run_stock_panel_regression(
                panel_freq, entity_col='permno', **fe_kw
            )
            results['P3'][freq][fe_label] = res
            stars = ('***' if res['pval'] < 0.01 else
                     '**' if res['pval'] < 0.05 else
                     '*' if res['pval'] < 0.10 else '')
            print(f"  {freq} | {fe_label:20s} | coef={res['coef']:+.6f} | "
                  f"t={res['tstat']:+.2f}{stars} | R2w={res['r2_within']:.4f} | "
                  f"N={res['nobs']} | stocks={res['n_entities']}")

    # ── Point 4: All-countries stock-level regressions ────────────────────
    all_chars_path = PATH['RAW_DATA'] / 'jkp_characteristics_all.parquet'
    if all_chars_path.exists():
        print("\n=== Point 4: All-countries stock-level contribution regressions ===")
        print("  [NOT YET IMPLEMENTED — all-country characteristics file found but "
              "code for multi-country stock contributions is pending]")
        results['P4'] = {}
    else:
        print("\n=== Point 4: SKIPPED ===")
        print(f"  All-country JKP characteristics not found at:")
        print(f"    {all_chars_path}")
        print("  Point 4 requires jkp_characteristics_all.parquet for non-US "
              "stock-level sorts.")
        results['P4'] = {}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PART E: LaTeX REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_cell(coef, tstat, pval):
    """Format a regression coefficient cell for LaTeX.

    Uses 4 decimal places by default, but switches to scientific notation
    when the coefficient would round to 0.0000.
    """
    stars = '^{***}' if pval < 0.01 else '^{**}' if pval < 0.05 else '^{*}' if pval < 0.1 else ''
    sign = '$-$' if coef < 0 else ''
    ac = abs(coef)
    if ac < 0.00005 and ac > 0:
        # Use scientific notation for very small coefficients
        # e.g. 1.79e-05 -> "1.79\text{e-}05"
        exp = int(np.floor(np.log10(ac)))
        mantissa = ac / (10 ** exp)
        coef_str = f'{sign}{mantissa:.2f}\\text{{e}}{exp}${stars}$'
    else:
        coef_str = f'{sign}{ac:.4f}${stars}$'
    return coef_str, f'({tstat:.2f})'


def _fmt_r2(val):
    """Format R-squared for LaTeX."""
    if np.isnan(val):
        return '--'
    if abs(val) < 0.00005:
        return '$<$0.0001' if val >= 0 else '$-$0.0000'
    return f'{val:.4f}'


def _fmt_n(val):
    """Format sample size with comma separator."""
    return f'{int(val):,}'


def _build_factor_table(results_point, fe_keys, beta_label, caption, label):
    """
    Build a LaTeX table for factor-level regressions (Points 1-2).

    Parameters
    ----------
    results_point : dict
        results[point] with keys Q, S, A, each containing fe_keys.
    fe_keys : tuple of (str, str)
        (entity_fe_key, entity_time_fe_key) in the results dict.
    beta_label : str
        LaTeX label for the coefficient row.
    caption : str
        Table caption.
    label : str
        Table label.

    Returns
    -------
    str : LaTeX table code.
    """
    fe_entity_key, fe_time_key = fe_keys
    freqs = ['Q', 'S', 'A']

    # Collect cells: entity FE columns, then entity+time FE columns
    coef_cells = []
    tstat_cells = []
    r2_cells = []
    n_cells = []

    for fe_key in [fe_entity_key, fe_time_key]:
        for freq in freqs:
            res = results_point[freq][fe_key]
            c, t = _fmt_cell(res['coef'], res['tstat'], res['pval'])
            coef_cells.append(c)
            tstat_cells.append(t)
            r2_cells.append(_fmt_r2(res['r2_within']))
            n_cells.append(_fmt_n(res['nobs']))

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{tab:{label}}}')
    lines.append(r'\footnotesize')
    lines.append(r'\begin{tabular}{@{}l ccc ccc @{}}')
    lines.append(r'\toprule')
    lines.append(r' & \multicolumn{3}{c}{Entity FE} & \multicolumn{3}{c}{Entity + Time FE} \\')
    lines.append(r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}')
    lines.append(r' & Q & S & A & Q & S & A \\')
    lines.append(r'\midrule')
    lines.append(f'{beta_label} & ' + ' & '.join(coef_cells) + r' \\')
    lines.append(r' & ' + ' & '.join(tstat_cells) + r' \\[2pt]')
    lines.append(r'$R^2_w$ & ' + ' & '.join(r2_cells) + r' \\')
    lines.append(r'$N$ & ' + ' & '.join(n_cells) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def _build_stock_table(results_point, fe_keys, beta_label, caption, label):
    """
    Build a LaTeX table for stock-level regressions (Points 3-4).
    Same structure as factor tables but with Stock FE labels.
    """
    fe_stock_key, fe_time_key = fe_keys
    freqs = ['Q', 'S', 'A']

    coef_cells = []
    tstat_cells = []
    r2_cells = []
    n_cells = []

    for fe_key in [fe_stock_key, fe_time_key]:
        for freq in freqs:
            res = results_point[freq][fe_key]
            c, t = _fmt_cell(res['coef'], res['tstat'], res['pval'])
            coef_cells.append(c)
            tstat_cells.append(t)
            r2_cells.append(_fmt_r2(res['r2_within']))
            n_cells.append(_fmt_n(res['nobs']))

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{tab:{label}}}')
    lines.append(r'\footnotesize')
    lines.append(r'\begin{tabular}{@{}l ccc ccc @{}}')
    lines.append(r'\toprule')
    lines.append(r' & \multicolumn{3}{c}{Stock FE} & \multicolumn{3}{c}{Stock + Time FE} \\')
    lines.append(r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}')
    lines.append(r' & Q & S & A & Q & S & A \\')
    lines.append(r'\midrule')
    lines.append(f'{beta_label} & ' + ' & '.join(coef_cells) + r' \\')
    lines.append(r' & ' + ' & '.join(tstat_cells) + r' \\[2pt]')
    lines.append(r'$R^2_w$ & ' + ' & '.join(r2_cells) + r' \\')
    lines.append(r'$N$ & ' + ' & '.join(n_cells) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def append_latex_sections(results: dict):
    """
    Append four new sections to factset.tex using regression results.

    Reads the existing .tex file, strips \\end{document}, appends sections
    for Points 1-4, re-adds \\end{document}, and compiles with pdflatex.
    """
    import subprocess

    tex_path = PATH['OVERLEAF'] / 'factset.tex'
    print(f"\n{'='*60}")
    print("Appending LaTeX sections to factset.tex")
    print(f"{'='*60}")

    # Read existing file
    with open(tex_path, 'r', encoding='utf-8') as f:
        tex_content = f.read()

    # Strip any previously appended sections (idempotent re-runs)
    MARKER = '%%% BEGIN AUTO-GENERATED FACTOR/STOCK IO SECTIONS %%%'
    if MARKER in tex_content:
        tex_content = tex_content[:tex_content.index(MARKER)].rstrip()

    # Strip \end{document}
    tex_content = tex_content.replace(r'\end{document}', '').rstrip()

    sec1 = []  # Section 1 (US factor-level) removed

    # ── Section 2: Factor-Level IO and US Factor Returns ────────────────
    sec2 = []
    sec2.append('')
    sec2.append(r'\clearpage')
    sec2.append(r'\section{Factor-Level IO and US Factor Returns}')
    sec2.append('')
    sec2.append(
        r'For each JKP factor, we compute the value-weighted IO level, foreign IO level, '
        r'and foreign share of IO (foreign IO / total IO) for the long and short legs, '
        r'and take their difference (long minus short). All variables are included with two '
        r'quarterly lags. Standard errors are clustered at the factor level.'
    )
    sec2.append('')

    iv_list = ['factor_io', 'factor_io_lag1', 'factor_io_lag2',
               'factor_io_for', 'factor_io_for_lag1', 'factor_io_for_lag2',
               'factor_io_for_share', 'factor_io_for_share_lag1', 'factor_io_for_share_lag2']
    iv_labels = {
        'factor_io': r'$\beta_{\text{IO}_t}$',
        'factor_io_lag1': r'$\beta_{\text{IO}_{t-1}}$',
        'factor_io_lag2': r'$\beta_{\text{IO}_{t-2}}$',
        'factor_io_for': r'$\beta_{\text{ForIO}_t}$',
        'factor_io_for_lag1': r'$\beta_{\text{ForIO}_{t-1}}$',
        'factor_io_for_lag2': r'$\beta_{\text{ForIO}_{t-2}}$',
        'factor_io_for_share': r'$\beta_{\text{ForShare}_t}$',
        'factor_io_for_share_lag1': r'$\beta_{\text{ForShare}_{t-1}}$',
        'factor_io_for_share_lag2': r'$\beta_{\text{ForShare}_{t-2}}$',
    }
    fe_keys_us = [('FE_none', 'No FE'), ('FE_entity', 'Factor FE'), ('FE_entity_time', 'Factor + Time FE')]

    if 'P1b' in results and results['P1b']:
        P1b = results['P1b']
        sec2.append(r'\begin{table}[htbp]')
        sec2.append(r'\centering')
        sec2.append(r'\caption{Factor-level IO, Foreign IO, and Foreign Share (quarterly, levels, two lags) vs.\ US factor returns.}')
        sec2.append(r'\label{tab:reg_factor_us_level}')
        sec2.append(r'\footnotesize')
        sec2.append(r'\begin{tabular}{@{}l ccc @{}}')
        sec2.append(r'\toprule')
        sec2.append(r' & No FE & Factor FE & Factor + Time FE \\')
        sec2.append(r'\midrule')

        for iv in iv_list:
            coefs_r, tstats_r = [], []
            for fe_key, _ in fe_keys_us:
                r = P1b['Q'][fe_key]
                c = r[iv]['coef']; t = r[iv]['tstat']; p = r[iv]['pval']
                if np.isnan(c):
                    coefs_r.append('---'); tstats_r.append('')
                else:
                    stars = '^{***}' if p < 0.01 else '^{**}' if p < 0.05 else '^{*}' if p < 0.1 else ''
                    sign = '$-$' if c < 0 else ''
                    coefs_r.append(f'{sign}{abs(c):.4f}${stars}$')
                    tstats_r.append(f'({t:.2f})')
            sec2.append(f'{iv_labels[iv]} & {" & ".join(coefs_r)} \\\\')
            sec2.append(f' & {" & ".join(tstats_r)} \\\\[2pt]')

        r2_r, n_r = [], []
        for fe_key, _ in fe_keys_us:
            r = P1b['Q'][fe_key]
            r2_r.append(f'{r["r2_within"]:.4f}')
            n_r.append(f'{r["nobs"]:,}')
        sec2.append(f'$R^2_w$ & {" & ".join(r2_r)} \\\\')
        sec2.append(f'$N$ & {" & ".join(n_r)} \\\\')
        sec2.append(r'\bottomrule')
        sec2.append(r'\end{tabular}')
        sec2.append(r'\end{table}')

    # ── Section 3: Stock-Level IO and US Optimal Portfolio Contributions ──
    sec3 = []
    sec3.append('')
    sec3.append(r'\clearpage')
    sec3.append(r'\section{Stock-Level IO and US Optimal Portfolio Contributions}')
    sec3.append('')
    sec3.append(
        r'For each US stock, we compute its contribution to the rolling 48-month '
        r'max-Sharpe factor portfolio. The contribution of stock $i$ in quarter $t$ '
        r'is $w_i \times r_i$, where '
        r'$w_i = \sum_f (w_f \times w_{if})$ '
        r'aggregates the stock\textquotesingle s value-weighted position across all '
        r'factor portfolios ($w_f$ = portfolio weight on factor $f$, $w_{if}$ = VW '
        r'weight of stock $i$ in factor $f$\textquotesingle s long/short leg). '
        r'The independent variable is $\Delta\text{IO}$ from Ferreira--Matos. '
        r'Standard errors are clustered at the stock level.'
    )
    sec3.append('')

    if 'P3' in results and results['P3']:
        table3 = _build_stock_table(
            results['P3'],
            fe_keys=('FE_stock', 'FE_stock_time'),
            beta_label=r'$\beta_{\Delta\text{IO}}$',
            caption=r'Stock-level $\Delta$IO and US optimal portfolio contributions (panel regression).',
            label='reg_stock_us',
        )
        sec3.append(table3)
    else:
        sec3.append(r'\textit{Stock-level regression results not available.}')

    # ── Section 4: Stock-Level IO and Country Optimal Portfolio Contributions
    sec4 = []
    sec4.append('')
    sec4.append(r'\clearpage')
    sec4.append(r'\section{Stock-Level IO and Country Optimal Portfolio Contributions}')
    sec4.append('')

    if 'P4' in results and results['P4']:
        sec4.append(
            r'We extend the stock-level analysis to all countries in the JKP dataset. '
            r'For each country, the stock-level contribution and $\Delta\text{IO}$ are '
            r'computed as in the US case.'
        )
        sec4.append('')
        table4 = _build_stock_table(
            results['P4'],
            fe_keys=('FE_stock', 'FE_stock_time'),
            beta_label=r'$\beta_{\Delta\text{IO}}$',
            caption=r'Stock-level $\Delta$IO and country optimal portfolio contributions (panel regression).',
            label='reg_stock_global',
        )
        sec4.append(table4)
    else:
        sec4.append(
            r'This analysis requires all-country JKP characteristics data '
            r'(\texttt{jkp\_characteristics\_all.parquet}), which is not currently '
            r'available. Once the data is obtained, the stock-level contribution '
            r'regressions can be extended to all countries following the same '
            r'methodology as Section~\ref{tab:reg_stock_us}.'
        )

    # ── Section 5: IO level + Holder Foreign Share level (no lags) ──────
    sec5 = []
    sec5.append('')
    sec5.append(r'\clearpage')
    sec5.append(r'\section{Stock-Level IO and Holder Foreign Share}')
    sec5.append('')
    sec5.append(
        r'We regress stock contributions to the portfolio return on institutional ownership (IO), '
        r'average foreign portfolio share of the stock\textquotesingle s holders (HFS), and foreign IO. '
        r'All variables are in levels with two lags. '
        r'HFS is the value-weighted average, across all institutions holding stock $i$ '
        r'at quarter $t$, of each institution\textquotesingle s foreign portfolio share. '
        r'Standard errors are clustered at the stock level.'
    )
    sec5.append('')

    if 'P5' in results and results['P5']:
        P5 = results['P5']
        iv_order = ['io', 'io_lag1', 'io_lag2',
                     'holder_foreign_share', 'holder_foreign_share_lag1', 'holder_foreign_share_lag2',
                     'io_for', 'io_for_lag1', 'io_for_lag2']
        iv_labels = {
            'io': r'IO$_t$', 'io_lag1': r'IO$_{t-1}$', 'io_lag2': r'IO$_{t-2}$',
            'holder_foreign_share': r'HFS$_t$', 'holder_foreign_share_lag1': r'HFS$_{t-1}$',
            'holder_foreign_share_lag2': r'HFS$_{t-2}$',
            'io_for': r'ForIO$_t$', 'io_for_lag1': r'ForIO$_{t-1}$', 'io_for_lag2': r'ForIO$_{t-2}$',
        }
        fe_specs = ['No FE', 'Stock FE', 'Time FE', 'Stock+Time FE']

        sec5.append(r'\begin{table}[htbp]')
        sec5.append(r'\centering')
        sec5.append(r'\caption{Return contribution (quarterly, levels): IO, HFS, and Foreign IO with two lags.}')
        sec5.append(r'\label{tab:stock_io_hfs}')
        sec5.append(r'\footnotesize')
        sec5.append(r'\begin{tabular}{@{}l cccc @{}}')
        sec5.append(r'\toprule')
        sec5.append(r' & No FE & Stock FE & Time FE & Stock+Time FE \\')
        sec5.append(r'\midrule')

        for iv in iv_order:
            row_label = f'$\\beta_{{\\text{{{iv_labels[iv]}}}}}$'
            coefs_r, tstats_r = [], []
            for fe in fe_specs:
                r = P5[fe]
                c, t, p = r['coefs'].get(iv, (np.nan, np.nan, np.nan))
                if np.isnan(c):
                    coefs_r.append('---'); tstats_r.append('')
                else:
                    stars = '^{***}' if p < 0.01 else '^{**}' if p < 0.05 else '^{*}' if p < 0.1 else ''
                    sign = '$-$' if c < 0 else ''
                    ac = abs(c)
                    if ac < 0.0001 and ac > 0:
                        exp = int(np.floor(np.log10(ac)))
                        mantissa = ac / 10 ** exp
                        coefs_r.append(f'{sign}{mantissa:.2f}\\text{{e}}{exp}${stars}$')
                    else:
                        coefs_r.append(f'{sign}{ac:.4f}${stars}$')
                    tstats_r.append(f'({t:.2f})')
            sec5.append(f'{row_label} & {" & ".join(coefs_r)} \\\\')
            sec5.append(f' & {" & ".join(tstats_r)} \\\\[2pt]')

        r2_r, n_r = [], []
        for fe in fe_specs:
            r = P5[fe]
            r2_r.append(f'{r["r2"]:.4f}' if not np.isnan(r['r2']) else '---')
            n_r.append(f'{r["n"]:,}' if r['n'] > 0 else '---')
        sec5.append(f'$R^2_w$ & {" & ".join(r2_r)} \\\\')
        sec5.append(f'$N$ & {" & ".join(n_r)} \\\\')
        sec5.append(r'\bottomrule')
        sec5.append(r'\end{tabular}')
        sec5.append(r'\end{table}')

    # P6: first-differenced DV table
    if 'P6' in results and results['P6']:
        P6 = results['P6']
        sec5.append(r'\begin{table}[htbp]')
        sec5.append(r'\centering')
        sec5.append(r'\caption{$\Delta$ Return contribution (quarterly, levels): IO, HFS, and Foreign IO with two lags.}')
        sec5.append(r'\label{tab:stock_d_ret_level}')
        sec5.append(r'\footnotesize')
        sec5.append(r'\begin{tabular}{@{}l cccc @{}}')
        sec5.append(r'\toprule')
        sec5.append(r' & No FE & Stock FE & Time FE & Stock+Time FE \\')
        sec5.append(r'\midrule')

        for iv in iv_order:
            row_label = f'$\\beta_{{\\text{{{iv_labels[iv]}}}}}$'
            coefs_r, tstats_r = [], []
            for fe in fe_specs:
                r = P6[fe]
                c, t, p = r['coefs'].get(iv, (np.nan, np.nan, np.nan))
                if np.isnan(c):
                    coefs_r.append('---'); tstats_r.append('')
                else:
                    stars = '^{***}' if p < 0.01 else '^{**}' if p < 0.05 else '^{*}' if p < 0.1 else ''
                    sign = '$-$' if c < 0 else ''
                    ac = abs(c)
                    if ac < 0.0001 and ac > 0:
                        exp = int(np.floor(np.log10(ac)))
                        mantissa = ac / 10 ** exp
                        coefs_r.append(f'{sign}{mantissa:.2f}\\text{{e}}{exp}${stars}$')
                    else:
                        coefs_r.append(f'{sign}{ac:.4f}${stars}$')
                    tstats_r.append(f'({t:.2f})')
            sec5.append(f'{row_label} & {" & ".join(coefs_r)} \\\\')
            sec5.append(f' & {" & ".join(tstats_r)} \\\\[2pt]')

        r2_r, n_r = [], []
        for fe in fe_specs:
            r = P6[fe]
            r2_r.append(f'{r["r2"]:.4f}' if not np.isnan(r['r2']) else '---')
            n_r.append(f'{r["n"]:,}' if r['n'] > 0 else '---')
        sec5.append(f'$R^2_w$ & {" & ".join(r2_r)} \\\\')
        sec5.append(f'$N$ & {" & ".join(n_r)} \\\\')
        sec5.append(r'\bottomrule')
        sec5.append(r'\end{tabular}')
        sec5.append(r'\end{table}')

    else:
        sec5.append(r'\textit{Results not available (requires holder\_foreign\_share data).}')

    # ── Assemble and write ────────────────────────────────────────────────
    new_sections = '\n'.join(sec1 + sec2 + sec3 + sec4 + sec5)
    final_tex = (tex_content + '\n\n'
                 + MARKER + '\n'
                 + new_sections + '\n\n'
                 + r'\end{document}' + '\n')

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(final_tex)
    print(f"  Updated: {tex_path}")

    # ── Compile with pdflatex (twice for cross-references) ────────────────
    overleaf_dir = str(PATH['OVERLEAF'])
    for run_num in [1, 2]:
        print(f"  pdflatex pass {run_num} ...")
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', 'factset.tex'],
            cwd=overleaf_dir, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"    WARNING: pdflatex returned code {result.returncode}")
            # Print last 20 lines of log for debugging
            log_lines = result.stdout.strip().split('\n')
            for line in log_lines[-20:]:
                print(f"    {line}")
        else:
            print(f"    Pass {run_num} OK")

    print(f"  PDF output: {PATH['OVERLEAF'] / 'factset.pdf'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()

    # ── Part A: Build factor-level panels ─────────────────────────────────
    us_panel = build_factor_dio_panel(reload=False)
    all_panel = build_all_countries_panel(us_panel)

    # ── Part B: Factor-level regressions (Points 1-2) ─────────────────────
    results = run_all_regressions(us_panel, all_panel)

    with open(RESULTS_CACHE, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nFactor-level regression results saved -> {RESULTS_CACHE}")

    # ── Part C: Stock contribution panel (Point 3) ────────────────────────
    stock_panel = build_stock_contribution_panel(reload=False)

    # ── Part D: Stock-level regressions (Points 3-4) ──────────────────────
    if len(stock_panel) > 0:
        stock_results = run_stock_contribution_regressions(stock_panel)

        with open(STOCK_REG_CACHE, 'wb') as f:
            pickle.dump(stock_results, f)
        print(f"\nStock-level regression results saved -> {STOCK_REG_CACHE}")

        # Merge all results into one dict for convenience
        results.update(stock_results)
    else:
        print("\nWARNING: Stock contribution panel is empty, skipping regressions.")
        stock_results = {}

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"Factor panel saved: {PANEL_CACHE}")
    print(f"Factor results saved: {RESULTS_CACHE}")
    if len(stock_panel) > 0:
        print(f"Stock panel saved: {STOCK_PANEL_CACHE}")
        print(f"Stock results saved: {STOCK_REG_CACHE}")
    print(f"{'='*60}")

    # Print key coefficients
    p1_q_fe = results['P1']['Q']['FE_entity']
    print(f"\nKey sanity check - P1 Quarterly (Factor FE only):")
    print(f"  coef = {p1_q_fe['coef']:+.6f}")
    print(f"  t    = {p1_q_fe['tstat']:+.4f}")
    print(f"  p    = {p1_q_fe['pval']:.6f}")
    print(f"  N    = {p1_q_fe['nobs']}")

    if 'P3' in results and 'Q' in results['P3'] and 'FE_stock' in results['P3']['Q']:
        p3_q_fe = results['P3']['Q']['FE_stock']
        print(f"\nKey sanity check - P3 Quarterly (Stock FE only):")
        print(f"  coef = {p3_q_fe['coef']:+.6f}")
        print(f"  t    = {p3_q_fe['tstat']:+.4f}")
        print(f"  p    = {p3_q_fe['pval']:.6f}")
        print(f"  N    = {p3_q_fe['nobs']}")
        print(f"  stocks = {p3_q_fe['n_entities']}")

    # ── Part E: IO + HFS + ForeignIO levels with 2 lags (Point 5) ─────────
    print("\nPart E: IO + HFS + ForeignIO levels regression...")
    HFS_MAPPED = PATH['INTERMEDIARY_RESULTS'] / 'holder_foreign_share_mapped.parquet'
    if HFS_MAPPED.exists() and len(stock_panel) > 0:
        from linearmodels.panel import PooledOLS
        # Load and merge HFS
        hfs = pd.read_parquet(HFS_MAPPED)
        hfs['quarter_date'] = pd.to_datetime(hfs['quarter_date'])
        sp5 = stock_panel.merge(hfs, on=['permno', 'quarter_date'], how='left')
        sp5 = sp5.sort_values(['permno', 'quarter_date'])

        # IO lags
        sp5['io_lag1'] = sp5.groupby('permno')['io'].shift(1)
        sp5['io_lag2'] = sp5.groupby('permno')['io'].shift(2)
        # HFS lags
        sp5['holder_foreign_share_lag1'] = sp5.groupby('permno')['holder_foreign_share'].shift(1)
        sp5['holder_foreign_share_lag2'] = sp5.groupby('permno')['holder_foreign_share'].shift(2)

        # Foreign IO from Ferreira
        fer_for = pd.read_parquet(PATH['RAW_DATA'] / 'ferreira_ownership.parquet',
                                  columns=['factset_entity_id', 'rquarter', 'sec_country', 'io_for'])
        fer_for = fer_for[fer_for['sec_country'] == 'US'].copy()
        fer_for['rquarter'] = pd.to_datetime(fer_for['rquarter'])
        sp5 = sp5.merge(fer_for[['factset_entity_id', 'rquarter', 'io_for']],
                        left_on=['factset_entity_id', 'quarter_date'],
                        right_on=['factset_entity_id', 'rquarter'], how='left').drop(columns='rquarter')
        sp5 = sp5.sort_values(['permno', 'quarter_date'])
        sp5['io_for_lag1'] = sp5.groupby('permno')['io_for'].shift(1)
        sp5['io_for_lag2'] = sp5.groupby('permno')['io_for'].shift(2)

        ivs = ['io', 'io_lag1', 'io_lag2',
               'holder_foreign_share', 'holder_foreign_share_lag1', 'holder_foreign_share_lag2',
               'io_for', 'io_for_lag1', 'io_for_lag2']

        P5 = {}
        tcol = 'quarter_date'
        for fe_label, ent_fe, time_fe in [
            ('No FE', False, False),
            ('Stock FE', True, False),
            ('Time FE', False, True),
            ('Stock+Time FE', True, True),
        ]:
            sub = sp5[['permno', tcol, 'contribution'] + ivs].dropna()
            if len(sub) < 50:
                P5[fe_label] = {'coefs': {iv: (np.nan, np.nan, np.nan) for iv in ivs}, 'r2': np.nan, 'n': 0}
                continue
            sub = sub.set_index(['permno', tcol])
            if not ent_fe and not time_fe:
                mod = PooledOLS(sub['contribution'], sub[ivs], check_rank=False)
            else:
                mod = PanelOLS(sub['contribution'], sub[ivs],
                               entity_effects=ent_fe, time_effects=time_fe, check_rank=False)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            coefs = {iv: (res.params[iv], res.tstats[iv], res.pvalues[iv]) for iv in ivs}
            r2 = res.rsquared_within if hasattr(res, 'rsquared_within') else res.rsquared
            P5[fe_label] = {'coefs': coefs, 'r2': r2, 'n': int(res.nobs)}
            print(f"  {fe_label}: N={int(res.nobs):,}")
            for iv in ivs:
                c, t, p = coefs[iv]
                s = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                print(f"    {iv}: {c:.6f}{s} (t={t:.2f})")

        results['P5'] = P5

        # ── Part E2: First-differenced return contribution ───────────────────
        print("\nPart E2: First-differenced return contribution...")
        sp5 = sp5.sort_values(['permno', 'quarter_date'])
        sp5['d_contribution'] = sp5.groupby('permno')['contribution'].diff()

        P6 = {}
        for fe_label, ent_fe, time_fe in [
            ('No FE', False, False),
            ('Stock FE', True, False),
            ('Time FE', False, True),
            ('Stock+Time FE', True, True),
        ]:
            sub = sp5[['permno', tcol, 'd_contribution'] + ivs].dropna()
            if len(sub) < 50:
                P6[fe_label] = {'coefs': {iv: (np.nan, np.nan, np.nan) for iv in ivs}, 'r2': np.nan, 'n': 0}
                continue
            sub = sub.set_index(['permno', tcol])
            if not ent_fe and not time_fe:
                mod = PooledOLS(sub['d_contribution'], sub[ivs], check_rank=False)
            else:
                mod = PanelOLS(sub['d_contribution'], sub[ivs],
                               entity_effects=ent_fe, time_effects=time_fe, check_rank=False)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            coefs = {iv: (res.params[iv], res.tstats[iv], res.pvalues[iv]) for iv in ivs}
            r2 = res.rsquared_within if hasattr(res, 'rsquared_within') else res.rsquared
            P6[fe_label] = {'coefs': coefs, 'r2': r2, 'n': int(res.nobs)}
            print(f"  {fe_label}: N={int(res.nobs):,}")
            for iv in ivs:
                c, t, p = coefs[iv]
                s = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                print(f"    {iv}: {c:.6f}{s} (t={t:.2f})")

        results['P6'] = P6
    else:
        print("  Skipped (no HFS data or no stock panel)")

    # ══════════════════════════════════════════════════════════════════════════
    # PART F: APPEND LaTeX SECTIONS TO factset.tex
    # ══════════════════════════════════════════════════════════════════════════
    append_latex_sections(results)
