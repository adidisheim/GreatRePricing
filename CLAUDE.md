# GreatRePricing – Project Conventions

This file is read by every Claude instance working on this repo.
Follow these rules strictly.

---

## Project structure

```
GreatRePricing/
├── CLAUDE.md              ← you are here
├── config.py              ← machine-specific paths (auto-detected)
├── data.py                ← all data loaders (reload pattern)
├── util_locals/           ← shared utility functions
│   ├── __init__.py
│   ├── stats.py           ← statistical helpers
│   ├── portfolio.py       ← portfolio sorts & construction
│   └── save.py            ← dual-save (local + Overleaf)
├── scripts/               ← shell & slurm scripts
├── processed_data/        ← cached processed datasets (git-ignored)
├── intermediary_results/  ← intermediate outputs (git-ignored)
└── final_results/         ← figures & tables (git-ignored)
    ├── figures/
    └── tables/
```

## 1. Path configuration (`config.py`)

- All paths are centralized in `config.py` via the `PATH` dict.
- Machine detection uses `os.getlogin()@socket.gethostname()`.
- **Never hard-code absolute paths anywhere else.** Always use:
  ```python
  from config import PATH
  PATH['RAW_DATA'] / 'my_file.csv'
  ```
- Available keys: `PROJECT_ROOT`, `OVERLEAF`, `RAW_DATA`, `PROCESSED_DATA`,
  `INTERMEDIARY_RESULTS`, `FINAL_RESULTS`.
- When a new coder or machine joins, they add their entry to
  `_MACHINE_CONFIGS` in `config.py`.

## 2. Data loading (`data.py`)

**Every data loader must follow the reload pattern:**

```python
def load_my_dataset(reload: bool = False) -> pd.DataFrame:
    def _raw():
        df = pd.read_csv(PATH['RAW_DATA'] / 'file.csv')
        # ... cleaning ...
        return df
    return _load_or_reload('my_dataset', _raw, fmt='parquet', reload=reload)
```

- `reload=False` (default) → loads from `processed_data/` cache (fast).
- `reload=True` → reads raw data, processes, saves cache, then returns.
- Use `parquet` for large tabular data, `pickle` for complex objects.
- Raw data lives in the shared Dropbox (`PATH['RAW_DATA']`).
  Processed caches are local and git-ignored.

## 3. Saving figures and tables (`util_locals/save.py`)

**Every figure and table must be saved twice** – locally and to Overleaf:

```python
from util_locals.save import save_figure, save_table

# Figures
fig, ax = plt.subplots()
ax.plot(...)
save_figure(fig, "my_plot")           # saves .pdf to final_results/figures/ AND Overleaf/figures/

# Tables
save_table(df, "summary_stats")       # saves .tex to final_results/tables/ AND Overleaf/tables/
```

- Default figure format is PDF (change via `fmt='png'` if needed).
- Never call `fig.savefig()` or `df.to_latex()` directly – always use the
  wrappers so both destinations are hit.

## 4. Utility functions (`util_locals/`)

- **stats.py** – statistical tests, standard errors, regressions.
- **portfolio.py** – portfolio sorts, return computation.
- **save.py** – dual-save for figures and tables.
- Add new modules as needed (e.g., `cleaning.py`, `plotting.py`).
- Keep functions focused and well-documented with docstrings.

## 5. Coding conventions

- **Python 3.10+**.
- Use `pathlib.Path` for all file paths, never string concatenation.
- Imports at the top of each file; `from config import PATH` is always first.
- Use `pandas` + `numpy` as the core stack.
- Prefer vectorized pandas/numpy operations over loops.
- Type hints in function signatures are encouraged but not mandatory.
- Docstrings: NumPy-style (`Parameters`, `Returns`).

## 6. Git hygiene

- Never commit data files (`.csv`, `.parquet`, `.pkl`, `.pickle`).
- Never commit figures or LaTeX outputs – they are reproducible.
- The `.gitignore` covers `processed_data/`, `intermediary_results/`,
  `final_results/`, and common data formats.
- Commit messages: short imperative (`Add CRSP loader`, `Fix NW t-stat`).
- Each coder works on their own branch for large features; merge to `main`
  via PR or after coordination.

## 7. Adding a new machine

Run this to get your machine id:
```bash
python -c "import socket,os; print(f'{os.getlogin()}@{socket.gethostname()}')"
```
Then add your entry to `_MACHINE_CONFIGS` in `config.py` with your
`overleaf` and `raw_data` paths.

## 8. Important rules for Claude instances

- Always read `config.py` before modifying paths.
- Always use `save_figure` / `save_table` – never save to only one location.
- Always use the `_load_or_reload` pattern for new data loaders.
- Do not create files in the repo root unless discussed – use the existing
  directory structure.
- When adding a new utility function, put it in the appropriate
  `util_locals/` module.
- Keep this file (`CLAUDE.md`) up to date when conventions change.
