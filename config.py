"""
Project-wide path configuration.

Detects the current machine via socket.gethostname() + os.getlogin() and sets
all paths accordingly.  Each coder should fill in their own block below.

Usage:
    from config import PATH
    df.to_parquet(PATH['PROCESSED_DATA'] / 'my_dataset.parquet')
"""

import socket
import os
from pathlib import Path

# ── project root (this file lives at the repo root) ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── machine detection ────────────────────────────────────────────────────────
_hostname = socket.gethostname()
_user = os.getlogin()
_machine_id = f"{_user}@{_hostname}"


def _build_paths(overleaf: str, raw_data: str) -> dict:
    """Return the full path dict given the two machine-specific paths."""
    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "OVERLEAF": Path(overleaf),
        "RAW_DATA": Path(raw_data),
        "PROCESSED_DATA": PROJECT_ROOT / "processed_data",
        "INTERMEDIARY_RESULTS": PROJECT_ROOT / "intermediary_results",
        "FINAL_RESULTS": PROJECT_ROOT / "final_results",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CODER-SPECIFIC CONFIGURATION
#  Add your hostname(s) / username(s) below.
#  Run  python -c "import socket,os; print(f'{os.getlogin()}@{socket.gethostname()}')"
#  to find your machine_id.
# ══════════════════════════════════════════════════════════════════════════════

_MACHINE_CONFIGS = {
    # ── Adi ────────────────────────────────────────────────────────────────
    "adidisheim@UML-FNQ2JDW1GV": {
        "overleaf": "/Users/adidisheim/Dropbox/Apps/Overleaf/GreatRePricing",
        "raw_data": "/Users/adidisheim/Dropbox/Melbourne/research/GreatRePricing_data",
    },
    # ── Mo ─────────────────────────────────────────────────────────────────
    # Run:  python -c "import socket,os; print(f'{os.getlogin()}@{socket.gethostname()}')"
    # Then add your entry here, e.g.:
    # "mo@Mos-MacBook-Pro.local": {
    #     "overleaf": "/Users/mo/Dropbox/Apps/Overleaf/GreatRePricing",
    #     "raw_data": "/Users/mo/Dropbox/Melbourne/research/GreatRePricing_data",
    # },

    # ── Luciano ────────────────────────────────────────────────────────────
    # "luciano@lucianos-machine": {
    #     "overleaf": "/Users/luciano/Dropbox/Apps/Overleaf/GreatRePricing",
    #     "raw_data": "/Users/luciano/Dropbox/Melbourne/research/GreatRePricing_data",
    # },
}

# ── resolve paths ─────────────────────────────────────────────────────────────
if _machine_id in _MACHINE_CONFIGS:
    _cfg = _MACHINE_CONFIGS[_machine_id]
    PATH = _build_paths(overleaf=_cfg["overleaf"], raw_data=_cfg["raw_data"])
else:
    raise EnvironmentError(
        f"Unknown machine: '{_machine_id}'.\n"
        f"Please add your config to _MACHINE_CONFIGS in config.py.\n"
        f"Run:  python -c \"import socket,os; print(f'{{os.getlogin()}}@{{socket.gethostname()}}')\"  "
        f"to get your machine id."
    )

# ── convenience: ensure local dirs exist ──────────────────────────────────────
for _key in ("PROCESSED_DATA", "INTERMEDIARY_RESULTS", "FINAL_RESULTS"):
    PATH[_key].mkdir(parents=True, exist_ok=True)
