# GreatRePricing

## Quick start

```bash
# 1. Clone & set up venv
git clone git@github.com:adidisheim/GreatRePricing.git
cd GreatRePricing
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Add your machine to config.py
python3 -c "import socket,os; print(f'{os.getlogin()}@{socket.gethostname()}')"
# Then add your entry to _MACHINE_CONFIGS in config.py

# 3. Download raw data from WRDS (requires WRDS account)
.venv/bin/python wrds_import.py --all       # download everything
.venv/bin/python wrds_import.py --list      # see available datasets
```

## Data downloads

| Command | Description |
|---|---|
| `.venv/bin/python wrds_import.py crsp` | CRSP monthly stock file (shrcd 10/11) |
| `.venv/bin/python wrds_import.py jkp` | JKP 153 characteristics (US) |
| `.venv/bin/python wrds_import.py --all` | All of the above |

On first run, WRDS will prompt for your username and password and cache them.

## Using the data

```python
from data import load_crsp, load_jkp

crsp = load_crsp()          # cached, fast
jkp  = load_jkp()           # cached, fast

crsp = load_crsp(reload=True)   # rebuild from raw
```

## Project conventions

See [CLAUDE.md](CLAUDE.md) for full details on paths, coding style, and rules.
