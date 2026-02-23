"""
Dual-save utilities: every figure / table is saved both locally and to Overleaf.

Usage:
    from util_locals.save import save_figure, save_table

    fig, ax = plt.subplots()
    ax.plot(...)
    save_figure(fig, "my_plot")

    save_table(df, "summary_stats")
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import PATH


# ── figures ──────────────────────────────────────────────────────────────────

def save_figure(
    fig: plt.Figure,
    name: str,
    local_dir: str = "final_results",
    subfolder: str = "figures",
    fmt: str = "pdf",
    dpi: int = 300,
    tight: bool = True,
    **savefig_kwargs,
) -> None:
    """
    Save a matplotlib figure to *both* a local directory and Overleaf.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        File stem (no extension).
    local_dir : str
        Key into PATH dict for the local destination (default: FINAL_RESULTS).
    subfolder : str
        Sub-folder inside both local and Overleaf dirs.
    fmt : str
        File format (pdf, png, ...).
    dpi : int
        Resolution for raster formats.
    tight : bool
        Whether to use bbox_inches='tight'.
    """
    filename = f"{name}.{fmt}"

    # local save
    local_path = PATH[local_dir.upper()] / subfolder
    local_path.mkdir(parents=True, exist_ok=True)
    local_file = local_path / filename

    kwargs = dict(dpi=dpi, **savefig_kwargs)
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(local_file, **kwargs)

    # overleaf save
    overleaf_path = PATH["OVERLEAF"] / subfolder
    overleaf_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_file, overleaf_path / filename)

    print(f"[save_figure] {filename} -> {local_file}")
    print(f"[save_figure] {filename} -> {overleaf_path / filename}")


# ── LaTeX tables ─────────────────────────────────────────────────────────────

def save_table(
    df: pd.DataFrame,
    name: str,
    local_dir: str = "final_results",
    subfolder: str = "tables",
    caption: str = "",
    label: str = "",
    float_format: str = "%.3f",
    **to_latex_kwargs,
) -> None:
    """
    Save a DataFrame as a .tex table to *both* a local directory and Overleaf.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        File stem (no extension).
    local_dir : str
        Key into PATH dict for the local destination.
    subfolder : str
        Sub-folder inside both local and Overleaf dirs.
    caption : str
        LaTeX caption.
    label : str
        LaTeX label (auto-prefixed with tab: if not provided).
    float_format : str
        Format string for floats.
    """
    if label == "":
        label = f"tab:{name}"

    filename = f"{name}.tex"

    latex_str = df.to_latex(
        caption=caption or None,
        label=label,
        float_format=float_format,
        **to_latex_kwargs,
    )

    # local save
    local_path = PATH[local_dir.upper()] / subfolder
    local_path.mkdir(parents=True, exist_ok=True)
    local_file = local_path / filename
    local_file.write_text(latex_str)

    # overleaf save
    overleaf_path = PATH["OVERLEAF"] / subfolder
    overleaf_path.mkdir(parents=True, exist_ok=True)
    (overleaf_path / filename).write_text(latex_str)

    print(f"[save_table] {filename} -> {local_file}")
    print(f"[save_table] {filename} -> {overleaf_path / filename}")
