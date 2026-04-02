"""
Institutional Ownership and Factor Portfolio Returns — Full Report.

Portfolio: Rolling 48-month max-Sharpe (6-month rebalance, Ledoit-Wolf, 10% cap).
IO metric: Foreign Share (FS) = foreign $ / total $.

Usage:
  cd GreatRePricing
  PYTHONPATH=. .venv/Scripts/python.exe factset_luciBS/factset_report.py
"""

from config import PATH
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from linearmodels.panel import PanelOLS
from util_locals.save import save_figure
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_parquet(PATH['RAW_DATA'] / 'ferreira_ownership.parquet')
df['date'] = pd.to_datetime(df['rquarter'])
df = df.sort_values(['factset_entity_id', 'quarter'])

# Rolling-window max-Sharpe portfolios (48m window, 6m rebalance)
ROLL_CACHE = PATH['INTERMEDIARY_RESULTS'] / 'max_sharpe_roll48_6.parquet'
if ROLL_CACHE.exists():
    print("  Loading cached rolling-window portfolios...")
    ms = pd.read_parquet(ROLL_CACHE)
    ms['date'] = pd.to_datetime(ms['date'])
else:
    print("  Building rolling-window (48m/6m) portfolios for all countries...")
    from data import load_jkp_factor_returns
    from compute_optimal_portfolio import build_max_sharpe_portfolio
    fr = load_jkp_factor_returns(weighting='vw_cap')
    fr['ret_signed'] = fr['ret'] * fr['direction']
    roll_frames = []
    for loc in sorted(fr['location'].unique()):
        fr_loc = fr[fr['location'] == loc].copy()
        n_f = fr_loc['name'].nunique()
        n_m = fr_loc['date'].nunique()
        if n_f < 5 or n_m < 48:
            continue
        print(f"    [{loc.upper()}] {n_f} factors, {n_m} months ...", end="", flush=True)
        port = build_max_sharpe_portfolio(fr_loc, min_obs=48, rebal=6, window=48)
        port['date'] = pd.to_datetime(port['date'])
        if len(port) < 24:
            print(f" only {len(port)} months, skipping")
            continue
        cumret = (1 + port['ret_opt']).cumprod().iloc[-1]
        print(f" {len(port)} mo, {cumret:.2f}x")
        port['location'] = loc
        roll_frames.append(port)
    ms = pd.concat(roll_frames, ignore_index=True)
    ms.to_parquet(ROLL_CACHE, index=False)
    print(f"  Cached -> {ROLL_CACHE}")

# US IO type aggregates
us_type_agg = pd.read_parquet(PATH['RAW_DATA'] / 'ferreira_us_type_agg.parquet')
us_type_by_country = pd.read_parquet(PATH['RAW_DATA'] / 'ferreira_us_type_by_country.parquet')

io_from_cols = sorted([c for c in df.columns if c.startswith('io_from_') and c != 'io_from_OT'])
io_type_cols = sorted([c for c in df.columns if c.startswith('io_type_') and c != 'io_type_banks'])
INST_COUNTRIES = {c.replace('io_from_', '') for c in io_from_cols}
TYPE_NAMES = sorted([t for t in us_type_agg[us_type_agg['cat_name'] != 'all']['cat_name'].unique() if t != 'banks'])

# Exclude banks
print("  Excluding banks from US IO...")
if 'io_type_banks' in df.columns:
    df['io_from_US'] = (df['io_from_US'] - df['io_type_banks']).clip(lower=0)

banks_agg = us_type_agg[us_type_agg['cat_name'] == 'banks'].set_index('quarter')
all_agg = us_type_agg[us_type_agg['cat_name'] == 'all'].set_index('quarter').copy()
all_agg['io_usd_us'] -= banks_agg['io_usd_us'].reindex(all_agg.index, fill_value=0)
all_agg['io_usd_foreign'] -= banks_agg['io_usd_foreign'].reindex(all_agg.index, fill_value=0)
all_agg = all_agg.reset_index(); all_agg['cat_name'] = 'all'
us_type_agg = pd.concat([all_agg, us_type_agg[~us_type_agg['cat_name'].isin(['all', 'banks'])]], ignore_index=True)

banks_bc = us_type_by_country[us_type_by_country['cat_name'] == 'banks'].set_index(['quarter', 'sec_country'])
all_bc = us_type_by_country[us_type_by_country['cat_name'] == 'all'].set_index(['quarter', 'sec_country']).copy()
all_bc['io_usd'] -= banks_bc['io_usd'].reindex(all_bc.index, fill_value=0)
all_bc = all_bc.reset_index(); all_bc['cat_name'] = 'all'
us_type_by_country = pd.concat([all_bc, us_type_by_country[~us_type_by_country['cat_name'].isin(['all', 'banks'])]], ignore_index=True)

iso2_to_3 = {
    'AE':'are','AR':'arg','AT':'aut','AU':'aus','BE':'bel','BR':'bra','CA':'can',
    'CH':'che','CL':'chl','CN':'chn','CO':'col','CZ':'cze','DE':'deu','DK':'dnk',
    'EG':'egy','ES':'esp','FI':'fin','FR':'fra','GB':'gbr','GR':'grc','HK':'hkg',
    'HU':'hun','ID':'idn','IE':'irl','IL':'isr','IN':'ind','IT':'ita','JP':'jpn',
    'KR':'kor','KW':'kwt','LU':'lux','MA':'mar','MX':'mex','MY':'mys','NL':'nld',
    'NO':'nor','NZ':'nzl','PK':'pak','PE':'per','PH':'phl','PL':'pol','PT':'prt',
    'QA':'qat','RO':'rou','RU':'rus','SA':'sau','SE':'swe','SG':'sgp','TH':'tha',
    'TR':'tur','TW':'twn','US':'usa','ZA':'zaf',
}
iso2_to_name = {
    'AU':'Australia','BE':'Belgium','CA':'Canada','CH':'Switzerland','CN':'China',
    'DE':'Germany','DK':'Denmark','ES':'Spain','FI':'Finland','FR':'France',
    'GB':'United Kingdom','HK':'Hong Kong','IE':'Ireland','IN':'India','IT':'Italy',
    'JP':'Japan','KR':'South Korea','NL':'Netherlands','NO':'Norway','SE':'Sweden',
    'SG':'Singapore','US':'United States','ZA':'South Africa',
}
CONTINENT = {
    'AU':'Asia-Pac','BE':'Europe','CA':'Americas','CH':'Europe','CN':'Asia-Pac',
    'DE':'Europe','DK':'Europe','ES':'Europe','FI':'Europe','FR':'Europe',
    'GB':'Europe','HK':'Asia-Pac','IE':'Europe','IN':'Asia-Pac','IT':'Europe',
    'JP':'Asia-Pac','KR':'Asia-Pac','NL':'Europe','NO':'Europe','SE':'Europe',
    'SG':'Asia-Pac','US':'Americas','ZA':'Africa',
}
TYPE_LABELS = {'all':'All US (ex banks)','hedge':'HF/VC/PE','insurance':'Insurance',
               'inv_adv':'Inv.~Advisors','inv_co':'Inv.~Companies','pension':'Pension'}

ms_locs = set(ms['location'].unique())
COUNTRIES = sorted([c for c in INST_COUNTRIES if iso2_to_3.get(c) in ms_locs])
print(f"  {len(COUNTRIES)} countries, {len(TYPE_NAMES)} IO types")

# Preprocessing
print("Preprocessing...")
for col in io_from_cols:
    df[f'd_{col}'] = df.groupby('factset_entity_id')[col].diff()
df['w'] = df['own_mktcap']
for col in io_from_cols:
    for p in [col, f'd_{col}']:
        df[f'w_{p}'] = df[p] * df['w']
agg_dict = {'w': 'sum'}
for wc in [c for c in df.columns if c.startswith('w_')]:
    agg_dict[wc] = 'sum'
cda = df.groupby(['sec_country', 'date']).agg(agg_dict).reset_index()
gda = df.groupby('date').agg(agg_dict)

# Sharpe winsorization (use rolling portfolios)
_sr = []
for loc, g in ms.groupby('location'):
    g = g.set_index('date')['ret_opt'].sort_index()
    sr = (g.resample('YE').mean() / g.resample('YE').std()) * np.sqrt(12)
    _sr.extend(sr.dropna().values)
SR_LO, SR_HI = np.percentile(_sr, [1, 99])
print(f"  Sharpe winsorization: [{SR_LO:.2f}, {SR_HI:.2f}]")

SUBSAMPLE_START = '2005-01-01'

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def int_to_dt(q):
    y, qq = divmod(int(q), 100)
    return pd.Timestamp(y, qq * 3, 1) + pd.offsets.MonthEnd(0)

def get_returns(c):
    iso3 = iso2_to_3.get(c)
    rets = ms[ms['location'] == iso3].set_index('date')['ret_opt'].sort_index()
    if len(rets) < 24: return None
    ret_q = rets.resample('QE').sum()
    rc = rets.to_frame()
    rc['half'] = rc.index.year * 10 + np.where(rc.index.month <= 6, 1, 2)
    ret_s = rc.groupby('half')['ret_opt'].sum()
    ret_s.index = ret_s.index.map(lambda h: pd.Timestamp(h//10, 6 if h%10==1 else 12, 30 if h%10==1 else 31))
    ret_a = rets.resample('YE').sum()
    sharpe_a = ((rets.resample('YE').mean() / rets.resample('YE').std()) * np.sqrt(12)).clip(SR_LO, SR_HI)
    return ret_q, ret_s, ret_a, sharpe_a

def get_returns_sub(c, start):
    iso3 = iso2_to_3.get(c)
    rets = ms[ms['location'] == iso3].set_index('date')['ret_opt'].sort_index()
    rets = rets[rets.index >= start]
    if len(rets) < 24: return None
    ret_q = rets.resample('QE').sum()
    rc = rets.to_frame()
    rc['half'] = rc.index.year * 10 + np.where(rc.index.month <= 6, 1, 2)
    ret_s = rc.groupby('half')['ret_opt'].sum()
    ret_s.index = ret_s.index.map(lambda h: pd.Timestamp(h//10, 6 if h%10==1 else 12, 30 if h%10==1 else 31))
    ret_a = rets.resample('YE').sum()
    sharpe_a = ((rets.resample('YE').mean() / rets.resample('YE').std()) * np.sqrt(12)).clip(SR_LO, SR_HI)
    return ret_q, ret_s, ret_a, sharpe_a

def corr_single(series, ret_q, ret_s, ret_a, sharpe_a):
    s = series.dropna(); s.index = pd.to_datetime(s.index)
    mq = pd.DataFrame({'m': s}).join(ret_q.rename('ret'), how='inner').dropna()
    ms_ = pd.DataFrame({'m': s.loc[s.index.month.isin([6,12])]}).join(ret_s.rename('ret'), how='inner').dropna()
    ma = pd.DataFrame({'m': s.loc[s.index.month==12]}).join(ret_a.rename('ret'), how='inner').join(sharpe_a.rename('sr'), how='inner').dropna()
    if len(mq) < 8 or len(ma) < 5: return None
    return [mq[['m','ret']].corr().iloc[0,1],
            ms_[['m','ret']].corr().iloc[0,1] if len(ms_)>=4 else np.nan,
            ma[['m','ret']].corr().iloc[0,1], ma[['m','sr']].corr().iloc[0,1]]

def fmt(v):
    if np.isnan(v): return '---'
    return f'$-${abs(v):.2f}' if v < 0 else f'{v:.2f}'

def fmt_cell(c, t, p):
    stars = '^{***}' if p<0.01 else '^{**}' if p<0.05 else '^{*}' if p<0.1 else ''
    sign = '$-$' if c < 0 else ''
    return f'{sign}{abs(c):.3f}${stars}$', f'({t:.2f})'

def get_aggs(c):
    h = cda[cda['sec_country']==c].set_index('date').drop(columns='sec_country').sort_index()
    f = gda.sort_index().sub(h, fill_value=0)
    return h, f

def agg_ret(rets_df, freq):
    parts = []
    for loc, g in rets_df.groupby('location'):
        g = g.set_index('date')['ret_opt'].sort_index()
        if freq=='Q': r = g.resample('QE').sum()
        elif freq=='S':
            rc = g.to_frame(); rc['half'] = rc.index.year*10 + np.where(rc.index.month<=6,1,2)
            r = rc.groupby('half')['ret_opt'].sum()
            r.index = r.index.map(lambda h: pd.Timestamp(h//10, 6 if h%10==1 else 12, 30 if h%10==1 else 31))
        elif freq=='A': r = g.resample('YE').sum()
        r = r.rename('ret'); r.index.name = 'date'
        parts.append(r.to_frame().assign(location=loc).reset_index())
    return pd.concat(parts, ignore_index=True)

def agg_sr(rets_df):
    parts = []
    for loc, g in rets_df.groupby('location'):
        g = g.set_index('date')['ret_opt'].sort_index()
        sr = ((g.resample('YE').mean()/g.resample('YE').std())*np.sqrt(12)).clip(SR_LO, SR_HI)
        sr = sr.rename('sharpe'); sr.index.name = 'date'
        parts.append(sr.to_frame().assign(location=loc).reset_index())
    return pd.concat(parts, ignore_index=True)

fe_specs = [('Country FE', dict(entity_effects=True, time_effects=False)),
            ('Country + Time FE', dict(entity_effects=True, time_effects=True))]

def run_regs(p, ms_data, sr_data, var_cols):
    results = {}
    for fe_label, fe_kw in fe_specs:
        results[fe_label] = []
        for freq, dep_col, dep_fn in [
            ('Q','ret',lambda: agg_ret(ms_data,'Q')), ('S','ret',lambda: agg_ret(ms_data,'S')),
            ('A','ret',lambda: agg_ret(ms_data,'A')), ('A','sharpe',lambda: sr_data.copy())]:
            dep = dep_fn()
            if freq=='Q': pp = p.copy()
            elif freq=='S': pp = p[p['date'].dt.month.isin([6,12])].copy()
            elif freq=='A': pp = p[p['date'].dt.month==12].copy()
            pp = pp.merge(dep, left_on=['iso3','date'], right_on=['location','date'], how='inner')
            pp = pp.dropna(subset=var_cols+[dep_col])
            for v in var_cols:
                s = pp[v].std()
                if s > 0: pp[v] = pp[v] / s
            if len(pp) < 30:
                results[fe_label].append({v:(np.nan,np.nan,np.nan) for v in var_cols})
                continue
            pp = pp.set_index(['iso3','date'])
            mod = PanelOLS(pp[dep_col], pp[var_cols], check_rank=False, **fe_kw)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            row = {v:(res.params[v], res.tstats[v], res.pvalues[v]) for v in var_cols}
            row['_r2'] = res.rsquared_within; row['_n'] = res.nobs
            results[fe_label].append(row)
    return results

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: IO figures
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 1: IO figures...")

# IO in USD billions
us_s = df[df['sec_country']=='US']; nonus_s = df[df['sec_country']!='US']
io_us = df.groupby('date').apply(lambda g: (g['io_from_US']*g['w']).sum()/1e3)
nonus_cols = [c for c in io_from_cols if c != 'io_from_US']
io_nonus = df.groupby('date').apply(lambda g: sum((g[c]*g['w']).sum() for c in nonus_cols)/1e3)

fig, ax = plt.subplots(figsize=(10,5), dpi=150)
ax.plot(io_us.index, io_us.values, label='US-based institutions', lw=2, color='#2c7bb6')
ax.plot(io_nonus.index, io_nonus.values, label='Non-US institutions', lw=2, color='#d7191c')
ax.set_ylabel('USD billions'); ax.set_title('Total IO by Institution Domicile', fontweight='bold')
ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.3)
ax.set_xlim(io_us.index[0], io_us.index[-1])
fig.tight_layout(); save_figure(fig, 'factset_io_usd_ts'); plt.close()

# IO as % of mktcap
pct_us = us_s.groupby('date').apply(lambda g: (g['io']*g['w']).sum()/g['w'].sum())
pct_nonus = nonus_s.groupby('date').apply(lambda g: (g['io']*g['w']).sum()/g['w'].sum())
fig, ax = plt.subplots(figsize=(10,5), dpi=150)
ax.plot(pct_us.index, pct_us.values, label='US equities', lw=2, color='#2c7bb6')
ax.plot(pct_nonus.index, pct_nonus.values, label='Non-US equities', lw=2, color='#d7191c')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylabel('IO (% of mktcap)'); ax.set_title('VW Institutional Ownership by Stock Domicile', fontweight='bold')
ax.legend(loc='upper left'); ax.grid(axis='y', alpha=0.3)
ax.set_xlim(pct_us.index[0], pct_us.index[-1])
fig.tight_layout(); save_figure(fig, 'factset_io_pct_ts'); plt.close()

# IO % by type
type_ts = {}
for tc in io_type_cols:
    tname = tc.replace('io_type_', '')
    type_ts[tname] = us_s.groupby('date').apply(lambda g: (g[tc]*g['w']).sum()/g['w'].sum())
fig, ax = plt.subplots(figsize=(10,5), dpi=150)
colors = plt.cm.Set2(np.linspace(0, 1, len(type_ts)))
for (tname, ts), color in zip(type_ts.items(), colors):
    ax.plot(ts.index, ts.values, label=TYPE_LABELS.get(tname, tname).replace('~',' '), lw=2, color=color)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylabel('IO (% of mktcap)'); ax.set_title('US Equity IO by Institution Type', fontweight='bold')
ax.legend(loc='upper left', ncol=2, fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.set_xlim(ts.index[0], ts.index[-1])
fig.tight_layout(); save_figure(fig, 'factset_io_type_ts'); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Compute Foreign Share metrics
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: Computing Foreign Share...")

# US IO types: FS = foreign $ / total $
fs_us_type = {}
for cat in ['all'] + TYPE_NAMES:
    sub = us_type_agg[us_type_agg['cat_name'] == cat].sort_values('quarter')
    total = sub.set_index('quarter')['io_usd_us'] + sub.set_index('quarter')['io_usd_foreign']
    fs = sub.set_index('quarter')['io_usd_foreign'] / total.replace(0, np.nan)
    fs.index = fs.index.map(int_to_dt)
    fs_us_type[cat] = fs

# Domestic institutions: FS = foreign $ / total $
fs_country = {}
for c in COUNTRIES:
    h, f = get_aggs(c)
    if len(h) < 20: continue
    ifc = f'io_from_{c}'
    dom_home = h[f'w_{ifc}']
    dom_foreign = f[f'w_{ifc}']
    fs = dom_foreign / (dom_home + dom_foreign).replace(0, np.nan)
    fs = fs.replace([np.inf, -np.inf], np.nan)
    fs.index = pd.to_datetime(fs.index)
    fs_country[c] = fs

print(f"  FS: {len(fs_us_type)} US types, {len(fs_country)} countries")

# FS time series figure
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
s = fs_us_type['all'].dropna()
ax.plot(s.index, s.values, lw=3, color='black', label='All US', zorder=10)
colors_fs = plt.cm.Set2(np.linspace(0, 1, len(TYPE_NAMES)))
for tname, color in zip(TYPE_NAMES, colors_fs):
    s = fs_us_type[tname].dropna()
    ax.plot(s.index, s.values, label=TYPE_LABELS.get(tname, tname).replace('~', ' '), lw=1.5, color=color, alpha=0.8)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylabel('Foreign Share'); ax.set_title('US Institutional Foreign Allocation Share by Type', fontweight='bold')
ax.legend(loc='upper left', ncol=2, fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.set_xlim(s.index[0], s.index[-1])
fig.tight_layout(); save_figure(fig, 'factset_fs_type_ts'); plt.close()

# FS bar chart
bar_cats = ['all'] + TYPE_NAMES
bar_labels = [TYPE_LABELS.get(c, c).replace('~', ' ') for c in bar_cats]
avg_pre, avg_post = [], []
for cat in bar_cats:
    s = fs_us_type[cat].dropna()
    pre = s[s.index < '2005-01-01']; post = s[s.index >= '2005-01-01']
    avg_pre.append(pre.mean() if len(pre) > 0 else 0)
    avg_post.append(post.mean() if len(post) > 0 else 0)
x = np.arange(len(bar_cats)); w = 0.35
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.bar(x - w/2, avg_pre, w, label='1999--2004', color='#2c7bb6', alpha=0.85)
ax.bar(x + w/2, avg_post, w, label='2005--2025', color='#d7191c', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(bar_labels, rotation=30, ha='right')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylabel('Foreign Share'); ax.set_title('Average Foreign Allocation Share by US Institution Type', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
for i, (v1, v2) in enumerate(zip(avg_pre, avg_post)):
    ax.text(i - w/2, v1 + 0.002, f'{v1:.1%}', ha='center', va='bottom', fontsize=8)
    ax.text(i + w/2, v2 + 0.002, f'{v2:.1%}', ha='center', va='bottom', fontsize=8)
fig.tight_layout(); save_figure(fig, 'factset_fs_bar'); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Correlations — US FS vs US max-Sharpe
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3: US FS correlations...")
us_ret = get_returns('US')
us_ret_sub = get_returns_sub('US', SUBSAMPLE_START)

corr_us_full, corr_us_sub = [], []
for cat in ['all'] + TYPE_NAMES:
    cf = corr_single(fs_us_type[cat], *us_ret)
    cs = corr_single(fs_us_type[cat][fs_us_type[cat].index >= SUBSAMPLE_START], *us_ret_sub)
    lab = TYPE_LABELS.get(cat, cat)
    if cf: corr_us_full.append((lab, *cf))
    if cs: corr_us_sub.append((lab, *cs))
    if cf:
        print(f"  {cat:<12} Full: Q={cf[0]:.3f} A={cf[2]:.3f} | 2005+: Q={cs[0]:.3f} A={cs[2]:.3f}" if cs else f"  {cat:<12} Full: Q={cf[0]:.3f} A={cf[2]:.3f} | 2005+: N/A")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Correlations — US FS vs each foreign country
# ══════════════════════════════════════════════════════════════════════════════
print("Step 4: US FS vs foreign countries...")
corr_foreign_full, corr_foreign_sub = {}, {}
for c in COUNTRIES:
    if c == 'US': continue
    ret = get_returns(c)
    ret_s2 = get_returns_sub(c, SUBSAMPLE_START)
    if ret is None: continue
    rows_f, rows_s = [], []
    for cat in ['all'] + TYPE_NAMES:
        rf = corr_single(fs_us_type[cat], *ret)
        if rf: rows_f.append((TYPE_LABELS.get(cat, cat), *rf))
        if ret_s2:
            rs = corr_single(fs_us_type[cat][fs_us_type[cat].index >= SUBSAMPLE_START], *ret_s2)
            if rs: rows_s.append((TYPE_LABELS.get(cat, cat), *rs))
    if rows_f: corr_foreign_full[c] = rows_f
    if rows_s: corr_foreign_sub[c] = rows_s
print(f"  Full: {len(corr_foreign_full)} countries, 2005+: {len(corr_foreign_sub)} countries")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Panel regressions — domestic FS
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 5: Panel regressions (domestic FS)...")
panel_base_rows = []
for c in COUNTRIES:
    if c == 'US' or c not in fs_country: continue
    iso3 = iso2_to_3[c]
    tmp = pd.DataFrame({'foreign_share': fs_country[c]})
    tmp['iso3'] = iso3
    tmp = tmp.reset_index()
    panel_base_rows.append(tmp)

panel_base = pd.concat(panel_base_rows, ignore_index=True)
ms_pb = ms[ms['location'].isin(panel_base['iso3'].unique())]
sr_pb = agg_sr(ms_pb)
n_countries = panel_base['iso3'].nunique()
print(f"  {len(panel_base)} rows, {n_countries} countries")

reg_base = run_regs(panel_base, ms_pb, sr_pb, ['foreign_share'])

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Panel + US FS on country X
# ══════════════════════════════════════════════════════════════════════════════
print("Step 6: Panel + US FS controls...")
reg_us = {}
for cat in ['all'] + TYPE_NAMES:
    sub = us_type_by_country[us_type_by_country['cat_name'] == cat]
    us_total = us_type_agg[(us_type_agg['cat_name'] == cat)].set_index('quarter')
    us_total_usd = us_total['io_usd_us'] + us_total['io_usd_foreign']

    panel_rows = []
    for c in COUNTRIES:
        if c == 'US' or c not in fs_country: continue
        iso3 = iso2_to_3[c]
        us_on_x = sub[sub['sec_country'] == c].set_index('quarter')['io_usd']
        share_x = us_on_x / us_total_usd.replace(0, np.nan)
        share_x = share_x.replace([np.inf, -np.inf], np.nan)
        share_x.index = share_x.index.map(int_to_dt)

        tmp = pd.DataFrame({'foreign_share': fs_country[c], 'us_share_x': share_x})
        tmp.index.name = 'date'
        tmp['iso3'] = iso3
        tmp = tmp.reset_index()
        panel_rows.append(tmp)

    if not panel_rows: continue
    panel = pd.concat(panel_rows, ignore_index=True)
    ms_p = ms[ms['location'].isin(panel['iso3'].unique())]
    sr_p = agg_sr(ms_p)
    reg_us[cat] = run_regs(panel, ms_p, sr_p, ['foreign_share', 'us_share_x'])
    print(f"  {cat}: {panel['iso3'].nunique()} countries")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Write LaTeX
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 7: Writing LaTeX...")
L = []
L.append(r'\documentclass[11pt]{article}')
L.append(r'\usepackage[margin=0.85in]{geometry}')
L.append(r'\usepackage{graphicx,booktabs,amsmath,caption}')
L.append(r'\captionsetup{font=small, labelfont=bf}')
L.append(r'\title{\large Institutional Ownership and Factor Portfolio Returns}')
L.append(r'\author{Luciano Somoza}')
L.append(r'\date{\today}')
L.append(r'\begin{document}')
L.append(r'\maketitle')

# ── Data section ─────────────────────────────────────────────────────────────
L.append(r'\section{Data and Variable Construction}')
L.append('')
L.append(
    r'We use the Ferreira--Matos--Pires (2020) institutional ownership dataset constructed from '
    r'FactSet Ownership V5, covering 53 MSCI ACWI countries at quarterly frequency (Q1 1999--Q1 2025). '
    r'For each firm-quarter, institutional ownership (IO) is decomposed by the institution\textquotesingle s '
    r'country of domicile (25 countries) and, for US-domiciled institutions, by type: '
    r'investment advisors, investment companies (mutual funds/ETFs), hedge funds/VC/PE, '
    r'pension/endowments, banks, and insurance.'
)
L.append('')
L.append(
    r'Our main variable of interest is the \textbf{foreign allocation share} ($\text{FS}$), '
    r'defined for each US institution type $g$ and quarter $t$ as:'
)
L.append(r'\[')
L.append(r'\text{FS}_{g,t} = \frac{\sum_{j \in \text{foreign}} \text{IO}^g_{j,t} \times \text{mktcap}_{j,t}}')
L.append(r'{\sum_{k \in \text{all}} \text{IO}^g_{k,t} \times \text{mktcap}_{k,t}}')
L.append(r'\]')
L.append(
    r'A higher $\text{FS}$ means US institutions hold a larger share of their total portfolio in '
    r'foreign stocks. For the panel regressions, the country-specific US variable '
    r'$\text{FS}^{\text{US}}_i$ is the share of total US institutional holdings allocated to '
    r'country $i$\textquotesingle s stocks.'
)
L.append('')
L.append(
    r'We correlate these metrics with the returns and Sharpe ratios of country-specific max-Sharpe '
    r'factor portfolios optimized over JKP factors using a \textbf{rolling 48-month estimation window} '
    r'with 6-month rebalancing, Ledoit-Wolf covariance, and long-only weights capped at 10\%. '
    r'Annual Sharpe ratios are winsorized at the 1st and 99th percentiles.'
)
L.append('')

# ── Figures ──────────────────────────────────────────────────────────────────
L.append(r'\section{Figures}')
L.append('')
for fn, cap, lab in [
    ('factset_io_usd_ts', 'Total IO in USD billions by institution domicile.', 'fig:io_usd'),
    ('factset_io_pct_ts', r'VW IO as \% of market cap by stock domicile.', 'fig:io_pct'),
    ('factset_io_type_ts', r'US equity IO by institution type (\% of market cap).', 'fig:io_type'),
    ('factset_fs_type_ts', r'Foreign Allocation Share by US institution type (black = all US).', 'fig:fs_type'),
    ('factset_fs_bar', r'Average Foreign Allocation Share by US institution type, before and after 2005.', 'fig:fs_bar'),
]:
    L.append(r'\begin{figure}[htbp]')
    L.append(r'\centering')
    L.append(f'\\includegraphics[width=0.95\\textwidth]{{figures/{fn}.pdf}}')
    L.append(f'\\caption{{{cap}}}\\label{{{lab}}}')
    L.append(r'\end{figure}')
    L.append('')

# ── Correlation section ──────────────────────────────────────────────────────
L.append(r'\clearpage')
L.append(r'\section{Correlation Analysis}')
L.append('')
L.append(
    r'Table~\ref{tab:corr_us_fs} reports correlations between $\text{FS}$ for each US '
    r'institution type and the US max-Sharpe portfolio.'
)
L.append('')

# Table: US FS correlations
L.append(r'\begin{table}[htbp]')
L.append(r'\centering')
L.append(r'\caption{Foreign Share: correlation with US max-Sharpe by IO type.}')
L.append(r'\label{tab:corr_us_fs}')
L.append(r'\footnotesize')
L.append(r'\begin{tabular}{@{}l rrrr rrrr @{}}')
L.append(r'\toprule')
L.append(r' & \multicolumn{4}{c}{Full sample (1999--2025)} & \multicolumn{4}{c}{2005--2025} \\')
L.append(r'\cmidrule(lr){2-5} \cmidrule(lr){6-9}')
L.append(r'IO Type & Q & S & A & SR & Q & S & A & SR \\')
L.append(r'\midrule')
for rf, rs in zip(corr_us_full, corr_us_sub):
    L.append(f'{rf[0]} & {fmt(rf[1])} & {fmt(rf[2])} & {fmt(rf[3])} & {fmt(rf[4])} & {fmt(rs[1])} & {fmt(rs[2])} & {fmt(rs[3])} & {fmt(rs[4])} \\\\')
L.append(r'\bottomrule')
L.append(r'\end{tabular}')
L.append(r'\end{table}')
L.append('')

# Table: US FS vs foreign countries
L.append(r'\begin{table}[htbp]')
L.append(r'\centering')
L.append(r'\caption{US Foreign Share (all US inst): correlation with each foreign country max-Sharpe. SR winsorized.}')
L.append(r'\label{tab:corr_foreign_fs}')
L.append(r'\scriptsize')
L.append(r'\begin{tabular}{@{}ll rrrr rrrr @{}}')
L.append(r'\toprule')
L.append(r' & & \multicolumn{4}{c}{Full sample} & \multicolumn{4}{c}{2005--2025} \\')
L.append(r'\cmidrule(lr){3-6} \cmidrule(lr){7-10}')
L.append(r'Region & Country & Q & S & A & SR & Q & S & A & SR \\')
L.append(r'\midrule')
sorted_c = sorted(corr_foreign_full.keys(), key=lambda cc: (CONTINENT.get(cc,'ZZZ'), cc))
prev = None
for cc in sorted_c:
    cont = CONTINENT.get(cc, '?')
    if cont != prev:
        if prev is not None: L.append(r'\midrule')
        prev = cont
    all_f = [r for r in corr_foreign_full.get(cc, []) if r[0] == TYPE_LABELS['all']]
    all_s = [r for r in corr_foreign_sub.get(cc, []) if r[0] == TYPE_LABELS['all']]
    if all_f:
        rf = all_f[0]
        if all_s:
            rs = all_s[0]
            L.append(f'{cont} & {iso2_to_name.get(cc,cc)} & {fmt(rf[1])} & {fmt(rf[2])} & {fmt(rf[3])} & {fmt(rf[4])} & {fmt(rs[1])} & {fmt(rs[2])} & {fmt(rs[3])} & {fmt(rs[4])} \\\\')
        else:
            L.append(f'{cont} & {iso2_to_name.get(cc,cc)} & {fmt(rf[1])} & {fmt(rf[2])} & {fmt(rf[3])} & {fmt(rf[4])} & --- & --- & --- & --- \\\\')
L.append(r'\bottomrule')
L.append(r'\end{tabular}')
L.append(r'\end{table}')
L.append('')

# Cross-country type tables
all_types = ['all'] + TYPE_NAMES
type_hdrs = [TYPE_LABELS.get(t, t) for t in all_types]
n_types = len(all_types)

def write_country_type_table(caption, label, corr_dict, val_idx):
    L.append(r'\begin{table}[htbp]')
    L.append(r'\centering')
    L.append(f'\\caption{{{caption}}}')
    L.append(f'\\label{{{label}}}')
    L.append(r'\scriptsize')
    L.append(f'\\begin{{tabular}}{{@{{}}ll {"r" * n_types} @{{}}}}')
    L.append(r'\toprule')
    L.append(f'Region & Country & {" & ".join(type_hdrs)} \\\\')
    L.append(r'\midrule')
    sorted_cc = sorted(corr_dict.keys(), key=lambda cc: (CONTINENT.get(cc, 'ZZZ'), cc))
    prev_ = None
    all_vals = {cat: [] for cat in all_types}
    for cc in sorted_cc:
        cont = CONTINENT.get(cc, '?')
        if cont != prev_:
            if prev_ is not None: L.append(r'\midrule')
            prev_ = cont
        vals = []
        for cat in all_types:
            row = [r for r in corr_dict[cc] if r[0] == TYPE_LABELS.get(cat, cat)]
            if row:
                v = row[0][val_idx]; vals.append(fmt(v))
                if not np.isnan(v): all_vals[cat].append(abs(v))
            else: vals.append('---')
        L.append(f'{cont} & {iso2_to_name.get(cc, cc)} & {" & ".join(vals)} \\\\')
    L.append(r'\midrule')
    mabs = [fmt(np.mean(all_vals[cat]) if all_vals[cat] else np.nan) for cat in all_types]
    L.append(f' & Mean $|\\rho|$ & {" & ".join(mabs)} \\\\')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\end{table}')
    L.append('')

L.append(
    r'Tables~\ref{tab:corr_type_fs_aret}--\ref{tab:corr_type_fs_asr_sub} decompose the '
    r'cross-country correlations by US institution type using $\text{FS}$.'
)
L.append('')

write_country_type_table('Full sample: US Foreign Share by IO type vs.\\ country annual max-Sharpe return.', 'tab:corr_type_fs_aret', corr_foreign_full, 3)
write_country_type_table('Full sample: US Foreign Share by IO type vs.\\ country annual Sharpe ratio (winsorized).', 'tab:corr_type_fs_asr', corr_foreign_full, 4)
write_country_type_table('2005--2025: US Foreign Share by IO type vs.\\ country annual max-Sharpe return.', 'tab:corr_type_fs_aret_sub', corr_foreign_sub, 3)
write_country_type_table('2005--2025: US Foreign Share by IO type vs.\\ country annual Sharpe ratio (winsorized).', 'tab:corr_type_fs_asr_sub', corr_foreign_sub, 4)

# ── Regression section ───────────────────────────────────────────────────────
L.append(r'\clearpage')
L.append(r'\section{Panel Regressions}')
L.append('')
L.append(
    f'We estimate panel regressions on {n_countries} non-US countries. '
    r'The dependent variable $y_{i,t}$ is either the max-Sharpe portfolio return for country $i$ '
    r'(at quarterly, semiannual, or annual frequency) or the annualized within-year Sharpe ratio. '
    r'All independent variables are standardized by their pooled standard deviation. '
    r'Standard errors are clustered by country. Two specifications are reported: '
    r'country fixed effects and country + time fixed effects.'
)
L.append('')
L.append(
    r'Table~\ref{tab:reg_base} uses a single domestic regressor: the foreign allocation share of '
    r'country $i$\textquotesingle s own institutions.'
)
L.append('')
L.append(
    r'Tables~\ref{tab:reg_us_all}--\ref{tab:reg_us_pension} augment the baseline with a '
    r'US-specific control: $\text{FS}^{\text{US}}_i$, the share of total US institutional '
    r'holdings allocated to country $i$\textquotesingle s stocks. We report separate regressions '
    r'for each US institution type.'
)
L.append('')

def write_reg_table(caption, label, reg, var_cols, var_labels):
    L.append(r'\begin{table}[htbp]')
    L.append(r'\centering')
    L.append(f'\\caption{{{caption}}}')
    L.append(f'\\label{{{label}}}')
    L.append(r'\footnotesize')
    col_spec = ' '.join(['cccc'] * len(fe_specs))
    L.append(f'\\begin{{tabular}}{{@{{}}l {col_spec} @{{}}}}')
    L.append(r'\toprule')
    hdr = ' & '.join([f'\\multicolumn{{4}}{{c}}{{{fl}}}' for fl, _ in fe_specs])
    L.append(f' & {hdr} \\\\')
    cmi = []; col = 2
    for _ in fe_specs: cmi.append(f'\\cmidrule(lr){{{col}-{col+3}}}'); col += 4
    L.append(' '.join(cmi))
    L.append(f' & {" & ".join(["Q & S & A & SR"] * len(fe_specs))} \\\\')
    L.append(r'\midrule')
    for v, vl in zip(var_cols, var_labels):
        coefs, tstats = [], []
        for fl, _ in fe_specs:
            for row in reg[fl]:
                cc, t, p = row[v]
                if np.isnan(cc): coefs.append('---'); tstats.append('')
                else:
                    cs, ts = fmt_cell(cc, t, p); coefs.append(cs); tstats.append(ts)
        L.append(f'${vl}$ & {" & ".join(coefs)} \\\\')
        L.append(f' & {" & ".join(tstats)} \\\\[2pt]')
    r2s, ns = [], []
    for fl, _ in fe_specs:
        for row in reg[fl]:
            r2s.append(f'{row.get("_r2",np.nan):.4f}' if '_r2' in row else '---')
            ns.append(f'{row.get("_n",0):,}' if '_n' in row else '---')
    L.append(f'$R^2_w$ & {" & ".join(r2s)} \\\\')
    L.append(f'$N$ & {" & ".join(ns)} \\\\')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\end{table}')
    L.append('')

write_reg_table(
    f'Panel: domestic FS ({n_countries} non-US countries). SE clustered by country.',
    'tab:reg_base', reg_base,
    ['foreign_share'], [r'\beta_{\text{FS dom}}'])

for cat in ['all'] + TYPE_NAMES:
    if cat not in reg_us: continue
    write_reg_table(
        f'Adding US FS on country $i$ ({TYPE_LABELS.get(cat,cat)}). '
        f'US FS$_i$ = US \\$ on country $i$ / total US \\$.',
        f'tab:reg_us_{cat}', reg_us[cat],
        ['foreign_share', 'us_share_x'],
        [r'\beta_{\text{FS dom}}', r'\beta_{\text{FS}^{\text{US}}_i}'])

L.append(r'\end{document}')

tex_path = PATH['OVERLEAF'] / 'factset.tex'
tex_path.write_text('\n'.join(L), encoding='utf-8')
print(f"  Written to {tex_path}")

subprocess.run(['pdflatex', '-interaction=nonstopmode', 'factset.tex'],
               cwd=str(PATH['OVERLEAF']), capture_output=True)
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'factset.tex'],
               cwd=str(PATH['OVERLEAF']), capture_output=True)
print("  Compiled factset.pdf")
print("\nDone.")
