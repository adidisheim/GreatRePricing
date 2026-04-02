"""
Re-run Table 16 (stock-level contribution regressions) split by IO type + lagged IO.
Replaces Section "Stock-Level IO and US Optimal Portfolio Contributions" in factset.tex.

Usage:
  cd GreatRePricing
  PYTHONPATH=. .venv/Scripts/python.exe factset_luciBS/rerun_table16.py
"""
from config import PATH
import pandas as pd
import numpy as np
import pickle
import subprocess
import warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS, PooledOLS

# ── Load stock contribution panel ────────────────────────────────────────────
print("Loading stock contribution panel...")
sp = pd.read_parquet(PATH['INTERMEDIARY_RESULTS'] / 'stock_contribution_panel.parquet')
sp['quarter_date'] = pd.to_datetime(sp['quarter_date'])

# ── Load Ferreira IO by type and merge ───────────────────────────────────────
print("Loading Ferreira IO by type...")
io_type_cols = ['io_type_hedge', 'io_type_insurance', 'io_type_inv_adv',
                'io_type_inv_co', 'io_type_pension']
fer = pd.read_parquet(PATH['RAW_DATA'] / 'ferreira_ownership.parquet',
                      columns=['factset_entity_id', 'rquarter', 'sec_country',
                               'primary_sic_code', 'io_for'] + io_type_cols)
fer = fer[fer['sec_country'] == 'US'].copy()
fer['rquarter'] = pd.to_datetime(fer['rquarter'])
fer = fer.sort_values(['factset_entity_id', 'rquarter'])

# Compute delta IO for each type + delta foreign IO
for col in io_type_cols:
    fer[f'd_{col}'] = fer.groupby('factset_entity_id')[col].diff()
fer['d_io_for'] = fer.groupby('factset_entity_id')['io_for'].diff()

# SIC 2-digit industry code (time-invariant per firm: take mode)
fer['sic2'] = fer['primary_sic_code'].str[:2]
sic2_map = fer.groupby('factset_entity_id')['sic2'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '99')

# Merge with stock panel
merge_cols = (['factset_entity_id', 'rquarter', 'io_for', 'd_io_for']
              + [f'd_{c}' for c in io_type_cols])
sp = sp.merge(
    fer[merge_cols],
    left_on=['factset_entity_id', 'quarter_date'],
    right_on=['factset_entity_id', 'rquarter'],
    how='left'
).drop(columns='rquarter')

# Add SIC2 industry
sp['sic2'] = sp['factset_entity_id'].map(sic2_map).fillna('99')

# ── Add 1-quarter lag of each dIO + foreign IO ──────────────────────────────
print("Adding lagged dIO...")
sp = sp.sort_values(['permno', 'quarter_date'])
for col in io_type_cols:
    dcol = f'd_{col}'
    sp[f'{dcol}_lag1'] = sp.groupby('permno')[dcol].shift(1)
sp['dio_lag1'] = sp.groupby('permno')['dio'].shift(1)
sp['dio_lag2'] = sp.groupby('permno')['dio'].shift(2)
sp['io_lag1'] = sp.groupby('permno')['io'].shift(1)
sp['io_for_lag1'] = sp.groupby('permno')['io_for'].shift(1)
sp['d_io_for_lag1'] = sp.groupby('permno')['d_io_for'].shift(1)

# ── Merge holder_foreign_share (avg foreign portfolio share of holders) ──────
print("Merging holder_foreign_share...")
hfs = pd.read_parquet(PATH['INTERMEDIARY_RESULTS'] / 'holder_foreign_share_mapped.parquet')
hfs['quarter_date'] = pd.to_datetime(hfs['quarter_date'])
sp = sp.merge(hfs, on=['permno', 'quarter_date'], how='left')
sp = sp.sort_values(['permno', 'quarter_date'])
sp['holder_foreign_share_lag1'] = sp.groupby('permno')['holder_foreign_share'].shift(1)
sp['holder_foreign_share_lag2'] = sp.groupby('permno')['holder_foreign_share'].shift(2)
print(f"  holder_foreign_share non-null: {sp['holder_foreign_share'].notna().sum():,} / {len(sp):,}")

print(f"  Panel: {len(sp):,} rows")

# ── Helpers ──────────────────────────────────────────────────────────────────
TYPE_LABELS = {
    'dio': 'All IO',
    'd_io_type_hedge': 'HF/VC/PE',
    'd_io_type_insurance': 'Insurance',
    'd_io_type_inv_adv': 'Inv. Advisors',
    'd_io_type_inv_co': 'Inv. Companies',
    'd_io_type_pension': 'Pension',
}


def agg_to_freq(df, freq):
    df = df.copy()
    if freq == 'Q':
        return df, 'quarter_date'
    elif freq == 'S':
        df['period'] = df['quarter_date'].dt.year * 10 + np.where(df['quarter_date'].dt.month <= 6, 1, 2)
    elif freq == 'A':
        df['period'] = df['quarter_date'].dt.year

    agg_d = {'contribution': 'sum'}
    iv_cols = [c for c in df.columns if c.startswith('d_io_type') or c.startswith('io_for')
               or c.startswith('d_io_for') or c.startswith('holder_foreign_share')
               or c.startswith('d_hfs')
               or c in ['dio', 'dio_lag1', 'dio_lag2', 'io', 'io_lag1'] or c.endswith('_lag1') or c.endswith('_lag2')]
    for c in iv_cols:
        if c in df.columns:
            agg_d[c] = 'mean'
    # Keep sic2 (take first since it's time-invariant)
    if 'sic2' in df.columns:
        agg_d['sic2'] = 'first'

    grp = df.groupby(['permno', 'period']).agg(agg_d).reset_index()
    if freq == 'S':
        grp['date'] = grp['period'].apply(lambda p: pd.Timestamp(p // 10, 6 if p % 10 == 1 else 12, 30))
    else:
        grp['date'] = grp['period'].apply(lambda p: pd.Timestamp(p, 12, 31))
    return grp, 'date'


def run_reg(df, dep, ivs, ecol='permno', tcol='quarter_date',
            entity_fe=True, time_fe=False, ind_time_fe=False):
    """
    Run PanelOLS.
    If ind_time_fe=True, demean by industry×time (SIC2×quarter) instead of entity/time FE.
    Requires 'sic2' column in df.
    """
    if ind_time_fe:
        # Create industry×time group and use it as entity dimension (no separate time FE)
        sub = df[['permno', 'sic2', tcol, dep] + ivs].dropna(subset=[dep] + ivs).copy()
        sub['ind_time'] = sub['sic2'].astype(str) + '_' + sub[tcol].astype(str)
        if len(sub) < 50:
            return {iv: (np.nan, np.nan, np.nan) for iv in ivs}, np.nan, 0
        sub = sub.set_index(['ind_time', 'permno'])
        mod = PanelOLS(sub[dep], sub[ivs], entity_effects=True, time_effects=False, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
    else:
        needed = list(set([ecol, tcol, dep] + ivs))
        sub = df[needed].dropna(subset=[dep] + ivs)
        if len(sub) < 50:
            return {iv: (np.nan, np.nan, np.nan) for iv in ivs}, np.nan, 0
        sub = sub.set_index([ecol, tcol])
        if not entity_fe and not time_fe:
            mod = PooledOLS(sub[dep], sub[ivs], check_rank=False)
        else:
            mod = PanelOLS(sub[dep], sub[ivs], entity_effects=entity_fe, time_effects=time_fe, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
    coefs = {iv: (res.params[iv], res.tstats[iv], res.pvalues[iv]) for iv in ivs}
    return coefs, res.rsquared_within, int(res.nobs)


# ── Run regressions ──────────────────────────────────────────────────────────
print("\nRunning regressions by IO type...")
results = {}
for io_col, label in [('dio', 'All IO')] + [(f'd_{c}', TYPE_LABELS[f'd_{c}']) for c in io_type_cols]:
    lag_col = f'{io_col}_lag1'
    ivs = [io_col, lag_col]
    print(f"  {label}...")

    for freq in ['Q', 'S', 'A']:
        pdf, tcol = agg_to_freq(sp, freq)
        for fe_label, ife in [('Stock FE', False), ('Ind xTime FE', True)]:
            coefs, r2, n = run_reg(pdf, 'contribution', ivs, ecol='permno', tcol=tcol,
                                   entity_fe=True, time_fe=False, ind_time_fe=ife)
            results[(io_col, freq, fe_label)] = {'coefs': coefs, 'r2': r2, 'n': n}
            c0, t0, p0 = coefs[io_col]
            c1, t1, p1 = coefs[lag_col]
            if not np.isnan(c0):
                s0 = '***' if p0 < 0.01 else '**' if p0 < 0.05 else '*' if p0 < 0.1 else ''
                s1 = '***' if p1 < 0.01 else '**' if p1 < 0.05 else '*' if p1 < 0.1 else ''
                print(f"    {freq} {fe_label}: c={c0:.6f}{s0} (t={t0:.2f}), lag={c1:.6f}{s1} (t={t1:.2f}), N={n:,}")

# ── Kitchen-sink regression: IO + holder foreign share (both with lag) ────────
print("\n  Combined: IO + holder foreign share...")
all_ivs = ['io', 'holder_foreign_share']

for freq in ['Q', 'S', 'A']:
    pdf, tcol = agg_to_freq(sp, freq)
    for fe_label, ent_fe, time_fe, ife in [
        ('No FE', False, False, False),
        ('Stock FE', True, False, False),
    ]:
        coefs, r2, n = run_reg(pdf, 'contribution', all_ivs, ecol='permno', tcol=tcol,
                               entity_fe=ent_fe, time_fe=time_fe, ind_time_fe=ife)
        results[('combined', freq, fe_label)] = {'coefs': coefs, 'r2': r2, 'n': n}
        print(f"    {freq} {fe_label}: N={n:,}, R2={r2:.6f}")
        for iv in all_ivs:
            c, t, p = coefs[iv]
            if not np.isnan(c):
                s = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                print(f"      {iv}: {c:.6f}{s} (t={t:.2f})")

# Save
with open(PATH['INTERMEDIARY_RESULTS'] / 'stock_contribution_bytype_reg_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# ── Generate LaTeX ───────────────────────────────────────────────────────────
print("\nGenerating LaTeX...")


def fmt_cell(c, t, p):
    if np.isnan(c):
        return '---', ''
    stars = '^{***}' if p < 0.01 else '^{**}' if p < 0.05 else '^{*}' if p < 0.1 else ''
    sign = '$-$' if c < 0 else ''
    ac = abs(c)
    if ac < 0.0001 and ac > 0:
        exp = int(np.floor(np.log10(ac)))
        mantissa = ac / 10 ** exp
        coef_str = f'{sign}{mantissa:.2f}\\text{{e}}{exp}${stars}$'
    else:
        coef_str = f'{sign}{ac:.4f}${stars}$'
    return coef_str, f'({t:.2f})'


# Build one table per IO type
L = []
for io_col, label in [('dio', 'All IO')] + [(f'd_{c}', TYPE_LABELS[f'd_{c}']) for c in io_type_cols]:
    lag_col = f'{io_col}_lag1'
    safe = label.replace(' ', '_').replace('.', '').replace('/', '_').lower()
    L.append(r'\begin{table}[htbp]')
    L.append(r'\centering')
    L.append(f'\\caption{{Stock contribution vs.\\ $\\Delta$IO ({label}): contemporaneous + lagged.}}')
    L.append(f'\\label{{tab:stock_bytype_{safe}}}')
    L.append(r'\footnotesize')
    L.append(r'\begin{tabular}{@{}l ccc ccc @{}}')
    L.append(r'\toprule')
    L.append(r' & \multicolumn{3}{c}{Stock FE} & \multicolumn{3}{c}{Ind xTime FE} \\')
    L.append(r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}')
    L.append(r' & Q & S & A & Q & S & A \\')
    L.append(r'\midrule')

    # Contemporaneous row
    coefs_r, tstats_r = [], []
    for freq in ['Q', 'S', 'A']:
        for fe in ['Stock FE', 'Ind xTime FE']:
            r = results[(io_col, freq, fe)]
            c, t, p = r['coefs'][io_col]
            cs, ts = fmt_cell(c, t, p)
            coefs_r.append(cs)
            tstats_r.append(ts)
    L.append(f'$\\beta_{{\\Delta\\text{{IO}}_t}}$ & {" & ".join(coefs_r)} \\\\')
    L.append(f' & {" & ".join(tstats_r)} \\\\[2pt]')

    # Lagged row
    coefs_r, tstats_r = [], []
    for freq in ['Q', 'S', 'A']:
        for fe in ['Stock FE', 'Ind xTime FE']:
            r = results[(io_col, freq, fe)]
            c, t, p = r['coefs'][lag_col]
            cs, ts = fmt_cell(c, t, p)
            coefs_r.append(cs)
            tstats_r.append(ts)
    L.append(f'$\\beta_{{\\Delta\\text{{IO}}_{{t-1}}}}$ & {" & ".join(coefs_r)} \\\\')
    L.append(f' & {" & ".join(tstats_r)} \\\\[2pt]')

    # R2 and N
    r2_r, n_r = [], []
    for freq in ['Q', 'S', 'A']:
        for fe in ['Stock FE', 'Ind xTime FE']:
            r = results[(io_col, freq, fe)]
            r2_r.append(f'{r["r2"]:.4f}' if not np.isnan(r['r2']) else '---')
            n_r.append(f'{r["n"]:,}' if r['n'] > 0 else '---')
    L.append(f'$R^2_w$ & {" & ".join(r2_r)} \\\\')
    L.append(f'$N$ & {" & ".join(n_r)} \\\\')
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\end{table}')
    L.append('')

# ── Combined table: IO + holder foreign share, 3 FE specs ─────────────────────
ks_iv_order = ['io', 'holder_foreign_share']
KS_LABELS = {
    'io': r'IO',
    'holder_foreign_share': 'HFS',
}
FE_SPECS_KS = ['No FE', 'Stock FE']

L.append(r'\begin{table}[htbp]')
L.append(r'\centering')
L.append(r'\caption{Stock contribution: IO level and Holder Foreign Share (HFS) level, no lags.}')
L.append(r'\label{tab:stock_combined}')
L.append(r'\footnotesize')
L.append(r'\begin{tabular}{@{}l ccc ccc @{}}')
L.append(r'\toprule')
L.append(r' & \multicolumn{3}{c}{No FE} & \multicolumn{3}{c}{Stock FE} \\')
L.append(r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}')
L.append(r' & Q & S & A & Q & S & A \\')
L.append(r'\midrule')

for iv in ks_iv_order:
    row_label = f'$\\beta_{{\\text{{{KS_LABELS[iv]}}}}}$'
    coefs_r, tstats_r = [], []
    for freq in ['Q', 'S', 'A']:
        for fe in FE_SPECS_KS:
            r = results[('combined', freq, fe)]
            c, t, p = r['coefs'][iv]
            cs, ts = fmt_cell(c, t, p)
            coefs_r.append(cs)
            tstats_r.append(ts)
    L.append(f'{row_label} & {" & ".join(coefs_r)} \\\\')
    L.append(f' & {" & ".join(tstats_r)} \\\\[2pt]')

r2_r, n_r = [], []
for freq in ['Q', 'S', 'A']:
    for fe in FE_SPECS_KS:
        r = results[('combined', freq, fe)]
        r2_r.append(f'{r["r2"]:.4f}' if not np.isnan(r['r2']) else '---')
        n_r.append(f'{r["n"]:,}' if r['n'] > 0 else '---')
L.append(f'$R^2_w$ & {" & ".join(r2_r)} \\\\')
L.append(f'$N$ & {" & ".join(n_r)} \\\\')
L.append(r'\bottomrule')
L.append(r'\end{tabular}')
L.append(r'\end{table}')
L.append('')

# ── Append to factset.tex ────────────────────────────────────────────────────
print("Updating factset.tex...")
tex_path = PATH['OVERLEAF'] / 'factset.tex'
tex = tex_path.read_text(encoding='utf-8')

old_header = r'\section{Stock-Level IO and US Optimal Portfolio Contributions}'
idx = tex.find(old_header)
# Find next \clearpage or \end{document}
search_after = idx + len(old_header)
next_break = tex.find(r'\clearpage', search_after)
if next_break == -1:
    next_break = tex.find(r'\end{document}', search_after)

new_section = []
new_section.append(r'\section{Stock-Level IO and US Optimal Portfolio Contributions}')
new_section.append('')
new_section.append(
    r'For each US stock, we compute its contribution to the rolling 48-month max-Sharpe factor portfolio. '
    r'The contribution of stock $i$ in quarter $t$ is $w_i \times r_i$, where $w_i = \sum_f (w_f \times w_{if})$ '
    r'aggregates the stock\textquotesingle s value-weighted position across all factor portfolios. '
    r'We regress stock contributions on the contemporaneous and one-quarter-lagged change in institutional ownership '
    r'($\Delta\text{IO}_t$ and $\Delta\text{IO}_{t-1}$), separately for total IO and each US institution type '
    r'(HF/VC/PE, insurance, investment advisors, investment companies, pension). '
    r'Standard errors are clustered at the stock level. '
    r'The final table reports a joint regression with all five IO types (contemporaneous and lagged) '
    r'plus stock-level foreign IO (level, contemporaneous and lagged) as regressors. '
    r'Two fixed-effects specifications are reported: stock FE and industry$\times$time FE '
    r'(SIC 2-digit $\times$ quarter, demeaning within each industry-quarter group).'
)
new_section.append('')
new_section.extend(L)

tex_new = tex[:idx] + '\n'.join(new_section) + '\n' + tex[next_break:]
tex_path.write_text(tex_new, encoding='utf-8')

# Compile
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'factset.tex'],
               cwd=str(PATH['OVERLEAF']), capture_output=True)
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'factset.tex'],
               cwd=str(PATH['OVERLEAF']), capture_output=True)
print("  Compiled factset.pdf")

# ── Render the combined table as a standalone PDF ────────────────────────────
print("Rendering combined table as standalone PDF...")
import tempfile, shutil
from pathlib import Path
tmpdir = tempfile.mkdtemp()
last_table_start = None
for i in range(len(L) - 1, -1, -1):
    if L[i] == r'\begin{table}[htbp]':
        last_table_start = i
        break
if last_table_start is not None:
    table_lines = L[last_table_start:]
    standalone_tex = (r'\documentclass[border=5pt]{standalone}' + '\n'
                      + r'\usepackage{booktabs,amsmath,graphicx}' + '\n'
                      + r'\begin{document}' + '\n'
                      + '\n'.join(table_lines) + '\n'
                      + r'\end{document}')
    (Path(tmpdir) / 'table.tex').write_text(standalone_tex, encoding='utf-8')
    subprocess.run(['pdflatex', '-interaction=nonstopmode', 'table.tex'],
                   cwd=tmpdir, capture_output=True)
    table_pdf = Path(tmpdir) / 'table.pdf'
    out_pdf = PATH['PROJECT_ROOT'] / 'table_combined.pdf'
    if table_pdf.exists():
        shutil.copy(table_pdf, out_pdf)
        print(f"  Saved: {out_pdf}")
        # Try rendering to PNG via matplotlib
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import fitz  # PyMuPDF
            doc = fitz.open(str(out_pdf))
            page = doc[0]
            pix = page.get_pixmap(dpi=300)
            out_png = PATH['PROJECT_ROOT'] / 'table_combined.png'
            pix.save(str(out_png))
            doc.close()
            print(f"  Saved PNG: {out_png}")
        except ImportError:
            print("  (No PyMuPDF for PNG conversion, sending PDF)")
    shutil.rmtree(tmpdir, ignore_errors=True)

print("\nDone.")
