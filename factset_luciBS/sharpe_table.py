"""
Stock-level Sharpe contribution regression: IO level + HFS level with 2 lags.
DV = w_i * r_i / sigma_p (contribution to Sharpe ratio).

Usage:
  cd GreatRePricing
  PYTHONPATH=. .venv/Scripts/python.exe factset_luciBS/sharpe_table.py
"""
from config import PATH
import pandas as pd
import numpy as np
import subprocess
import warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS, PooledOLS

# ── Load and build Sharpe contribution ───────────────────────────────────────
print("Loading stock contribution panel...")
sp = pd.read_parquet(PATH['INTERMEDIARY_RESULTS'] / 'stock_contribution_panel.parquet')
sp['quarter_date'] = pd.to_datetime(sp['quarter_date'])

print(f"  Contribution panel: {len(sp):,} rows")

# ── Add IO lags ──────────────────────────────────────────────────────────────
sp = sp.sort_values(['permno', 'quarter_date'])
sp['io_lag1'] = sp.groupby('permno')['io'].shift(1)
sp['io_lag2'] = sp.groupby('permno')['io'].shift(2)
sp['dio_lag1'] = sp.groupby('permno')['dio'].shift(1)
sp['dio_lag2'] = sp.groupby('permno')['dio'].shift(2)

# ── Merge HFS + lags ────────────────────────────────────────────────────────
hfs = pd.read_parquet(PATH['INTERMEDIARY_RESULTS'] / 'holder_foreign_share_mapped.parquet')
hfs['quarter_date'] = pd.to_datetime(hfs['quarter_date'])
sp = sp.merge(hfs, on=['permno', 'quarter_date'], how='left')
sp = sp.sort_values(['permno', 'quarter_date'])
sp['holder_foreign_share_lag1'] = sp.groupby('permno')['holder_foreign_share'].shift(1)
sp['holder_foreign_share_lag2'] = sp.groupby('permno')['holder_foreign_share'].shift(2)
sp['d_hfs'] = sp.groupby('permno')['holder_foreign_share'].diff()
sp['d_hfs_lag1'] = sp.groupby('permno')['d_hfs'].shift(1)
sp['d_hfs_lag2'] = sp.groupby('permno')['d_hfs'].shift(2)

print(f"  HFS non-null: {sp['holder_foreign_share'].notna().sum():,} / {len(sp):,}")

# ── Foreign IO (from Ferreira) + lags ────────────────────────────────────────
print("Adding foreign IO...")
fer_for = pd.read_parquet(PATH['RAW_DATA'] / 'ferreira_ownership.parquet',
                          columns=['factset_entity_id', 'rquarter', 'sec_country', 'io_for'])
fer_for = fer_for[fer_for['sec_country'] == 'US'].copy()
fer_for['rquarter'] = pd.to_datetime(fer_for['rquarter'])
sp = sp.merge(fer_for[['factset_entity_id', 'rquarter', 'io_for']],
              left_on=['factset_entity_id', 'quarter_date'],
              right_on=['factset_entity_id', 'rquarter'], how='left').drop(columns='rquarter')
sp = sp.sort_values(['permno', 'quarter_date'])
sp['io_for_lag1'] = sp.groupby('permno')['io_for'].shift(1)
sp['io_for_lag2'] = sp.groupby('permno')['io_for'].shift(2)
print(f"  io_for non-null: {sp['io_for'].notna().sum():,}")

# ── Regressions ──────────────────────────────────────────────────────────────
all_ivs = ['dio', 'dio_lag1', 'dio_lag2',
           'd_hfs', 'd_hfs_lag1', 'd_hfs_lag2',
           'io_for', 'io_for_lag1', 'io_for_lag2']


def agg_to_freq(df, freq):
    if freq == 'Q':
        return df, 'quarter_date'
    df = df.copy()
    if freq == 'S':
        df['period'] = df['quarter_date'].dt.year * 10 + np.where(df['quarter_date'].dt.month <= 6, 1, 2)
    else:
        df['period'] = df['quarter_date'].dt.year
    agg_d = {'contribution': 'sum'}
    for c in all_ivs + ['dio', 'dio_lag1', 'dio_lag2', 'd_hfs', 'd_hfs_lag1', 'd_hfs_lag2',
                         'dio_x_dhfs', 'dio_lag1_x_dhfs_lag1', 'dio_lag2_x_dhfs_lag2']:
        if c in df.columns and c not in agg_d:
            agg_d[c] = 'mean'
    grp = df.groupby(['permno', 'period']).agg(agg_d).reset_index()
    if freq == 'S':
        grp['date'] = grp['period'].apply(lambda p: pd.Timestamp(p // 10, 6 if p % 10 == 1 else 12, 30))
    else:
        grp['date'] = grp['period'].apply(lambda p: pd.Timestamp(p, 12, 31))
    return grp, 'date'


# ── Sharpe contribution ──────────────────────────────────────────────────────
print("Computing Sharpe contribution...")
port_ret = sp.groupby('quarter_date')['contribution'].sum().sort_index()
roll_vol = port_ret.rolling(8, min_periods=4).std()
roll_vol.name = 'port_vol'
sp = sp.merge(roll_vol.reset_index(), on='quarter_date', how='left')
sp['sharpe_contribution'] = sp['contribution'] / sp['port_vol'].replace(0, np.nan)

all_ivs_level = ['io', 'io_lag1', 'io_lag2',
                 'holder_foreign_share', 'holder_foreign_share_lag1', 'holder_foreign_share_lag2',
                 'io_for', 'io_for_lag1', 'io_for_lag2']

FE_COMBOS = [
    ('No FE', False, False),
    ('Stock FE', True, False),
    ('Time FE', False, True),
    ('Stock+Time FE', True, True),
]

print("\nRunning regressions (quarterly, all FE combos)...")
results_ret = {}
results_sharpe = {}

for spec_name, dep_var, res_dict in [('RETURN CONTRIBUTION', 'contribution', results_ret),
                                      ('SHARPE CONTRIBUTION', 'sharpe_contribution', results_sharpe)]:
    print(f"\n  --- {spec_name} ---")
    all_ivs = all_ivs_level
    pdf, tcol = sp.copy(), 'quarter_date'
    for fe_label, ent_fe, time_fe in FE_COMBOS:
        sub = pdf[['permno', tcol, dep_var] + all_ivs].dropna()
        if len(sub) < 50:
            res_dict[fe_label] = {
                'coefs': {iv: (np.nan, np.nan, np.nan) for iv in all_ivs},
                'r2': np.nan, 'n': 0
            }
            continue
        sub = sub.set_index(['permno', tcol])
        if not ent_fe and not time_fe:
            mod = PooledOLS(sub[dep_var], sub[all_ivs], check_rank=False)
        else:
            mod = PanelOLS(sub[dep_var], sub[all_ivs],
                           entity_effects=ent_fe, time_effects=time_fe, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        coefs = {iv: (res.params[iv], res.tstats[iv], res.pvalues[iv]) for iv in all_ivs}
        r2 = res.rsquared_within if hasattr(res, 'rsquared_within') else res.rsquared
        res_dict[fe_label] = {'coefs': coefs, 'r2': r2, 'n': int(res.nobs)}
        print(f"    {fe_label}: N={int(res.nobs):,}")
        for iv in all_ivs:
            c, t, p = coefs[iv]
            s = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            print(f"      {iv}: {c:.6f}{s} (t={t:.2f})")

# ── Generate LaTeX ───────────────────────────────────────────────────────────
print("\nGenerating LaTeX...")


def fmt_cell(c, t, p):
    if np.isnan(c):
        return '---', ''
    stars = '^{***}' if p < 0.01 else '^{**}' if p < 0.05 else '^{*}' if p < 0.1 else ''
    sign = '$-$' if c < 0 else ''
    ac = abs(c)
    if ac < 0.001 and ac > 0:
        exp = int(np.floor(np.log10(ac)))
        mantissa = ac / 10 ** exp
        return f'{sign}{mantissa:.2f}\\text{{e}}{exp}${stars}$', f'({t:.2f})'
    return f'{sign}{ac:.4f}${stars}$', f'({t:.2f})'


IV_LABELS_LEVEL = {
    'io': r'IO$_t$', 'io_lag1': r'IO$_{t-1}$', 'io_lag2': r'IO$_{t-2}$',
    'holder_foreign_share': r'HFS$_t$', 'holder_foreign_share_lag1': r'HFS$_{t-1}$', 'holder_foreign_share_lag2': r'HFS$_{t-2}$',
    'io_for': r'ForIO$_t$', 'io_for_lag1': r'ForIO$_{t-1}$', 'io_for_lag2': r'ForIO$_{t-2}$',
}
FE_SPECS = ['No FE', 'Stock FE', 'Time FE', 'Stock+Time FE']

L = []


def build_table(ivs, iv_labels, res_dict, caption, label):
    T = []
    T.append(r'\begin{table}[htbp]')
    T.append(r'\centering')
    T.append(f'\\caption{{{caption}}}')
    T.append(f'\\label{{{label}}}')
    T.append(r'\footnotesize')
    T.append(r'\begin{tabular}{@{}l cccc @{}}')
    T.append(r'\toprule')
    T.append(r' & No FE & Stock FE & Time FE & Stock+Time FE \\')
    T.append(r'\midrule')
    for iv in ivs:
        row_label = f'$\\beta_{{\\text{{{iv_labels[iv]}}}}}$'
        coefs_r, tstats_r = [], []
        for fe in FE_SPECS:
            r = res_dict[fe]
            c, t, p = r['coefs'][iv]
            cs, ts = fmt_cell(c, t, p)
            coefs_r.append(cs)
            tstats_r.append(ts)
        T.append(f'{row_label} & {" & ".join(coefs_r)} \\\\')
        T.append(f' & {" & ".join(tstats_r)} \\\\[2pt]')
    r2_r, n_r = [], []
    for fe in FE_SPECS:
        r = res_dict[fe]
        r2_r.append(f'{r["r2"]:.4f}' if not np.isnan(r['r2']) else '---')
        n_r.append(f'{r["n"]:,}' if r['n'] > 0 else '---')
    T.append(f'$R^2_w$ & {" & ".join(r2_r)} \\\\')
    T.append(f'$N$ & {" & ".join(n_r)} \\\\')
    T.append(r'\bottomrule')
    T.append(r'\end{tabular}')
    T.append(r'\end{table}')
    T.append('')
    return T


L.extend(build_table(all_ivs_level, IV_LABELS_LEVEL, results_ret,
    r'Return contribution (quarterly, levels): IO, HFS, and Foreign IO with two lags.',
    'tab:stock_ret_level'))
L.extend(build_table(all_ivs_level, IV_LABELS_LEVEL, results_sharpe,
    r'Sharpe contribution (quarterly, levels): IO, HFS, and Foreign IO with two lags.',
    'tab:stock_sharpe_level'))

# ── Update factset.tex ───────────────────────────────────────────────────────
print("Updating factset.tex...")
tex_path = PATH['OVERLEAF'] / 'factset.tex'
tex = tex_path.read_text(encoding='utf-8')

old = r'\section{Stock-Level IO and Holder Foreign Share}'
idx = tex.find(old)
next_break = tex.find(r'\end{document}', idx)

new_sec = []
new_sec.append(r'\section{Stock-Level IO and Holder Foreign Share}')
new_sec.append('')
new_sec.append(
    r'We regress stock contributions on institutional ownership (IO), '
    r'average foreign portfolio share of the stock\textquotesingle s holders (HFS), and foreign IO. '
    r'All variables are in levels with two lags. Two dependent variables are considered: '
    r'the return contribution ($w_i \times r_i$) and the Sharpe contribution '
    r'($w_i \times r_i / \sigma_p$, where $\sigma_p$ is the rolling 8-quarter portfolio volatility). '
    r'Standard errors are clustered at the stock level.'
)
new_sec.append('')
new_sec.extend(L)

tex_new = tex[:idx] + '\n'.join(new_sec) + '\n\n' + tex[next_break:]
tex_path.write_text(tex_new, encoding='utf-8')

subprocess.run(['pdflatex', '-interaction=nonstopmode', 'factset.tex'],
               cwd=str(PATH['OVERLEAF']), capture_output=True)
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'factset.tex'],
               cwd=str(PATH['OVERLEAF']), capture_output=True)
print("  Compiled factset.pdf")

# ── Render as PNG ────────────────────────────────────────────────────────────
import tempfile, shutil
from pathlib import Path
tmpdir = tempfile.mkdtemp()
standalone = (r'\documentclass[border=5pt]{standalone}' + '\n'
              + r'\usepackage{booktabs,amsmath}' + '\n'
              + r'\begin{document}' + '\n'
              + '\n'.join(L) + '\n'
              + r'\end{document}')
(Path(tmpdir) / 'table.tex').write_text(standalone, encoding='utf-8')
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'table.tex'],
               cwd=tmpdir, capture_output=True)
table_pdf = Path(tmpdir) / 'table.pdf'
if table_pdf.exists():
    import fitz
    doc = fitz.open(str(table_pdf))
    pix = doc[0].get_pixmap(dpi=300)
    out_png = PATH['PROJECT_ROOT'] / 'table_combined.png'
    pix.save(str(out_png))
    doc.close()
    print(f"  Saved PNG: {out_png}")
shutil.rmtree(tmpdir, ignore_errors=True)

print("\nDone.")
