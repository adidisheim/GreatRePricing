"""
WRDS data downloads.

Each function downloads raw data from WRDS and saves it to RAW_DATA.
These are meant to be run once (or rarely) — the data.py loaders handle
day-to-day caching via the reload pattern.

Usage:
    .venv/bin/python wrds_import.py crsp          # download CRSP monthly
    .venv/bin/python wrds_import.py --list         # list available downloads
    .venv/bin/python wrds_import.py --all          # download everything

On first run, wrds will prompt for your WRDS username and password
and cache credentials for future use.
"""

import sys
import wrds
import pandas as pd
from config import PATH


def get_connection() -> wrds.Connection:
    """Open a WRDS connection (prompts for credentials on first use)."""
    return wrds.Connection()


# ══════════════════════════════════════════════════════════════════════════════
#  CRSP MONTHLY
# ══════════════════════════════════════════════════════════════════════════════

def download_crsp_monthly() -> None:
    """
    Download CRSP monthly stock file for common shares (shrcd 10, 11).

    Pulls from crsp.msf (monthly stock file) joined with crsp.msenames
    (security names / identifiers).

    Variables:
        - permno, permco          firm & security identifiers
        - date                    end-of-month date
        - ret, retx               return (with / without dividends)
        - prc, altprc              price (negative = bid/ask avg)
        - vol                      trading volume (in shares)
        - shrout                   shares outstanding (in thousands)
        - bid, ask                 closing bid & ask
        - cfacpr, cfacshr          cumulative adjustment factors
        - shrcd, exchcd            share code, exchange code
        - siccd, naics             SIC & NAICS industry codes
        - hsiccd                   historical SIC code
        - ticker, comnam           ticker symbol, company name
    """
    print("[wrds_import] Downloading CRSP monthly ...")

    query = """
        SELECT
            a.permno, a.permco, a.date,
            a.ret, a.retx,
            a.prc, a.altprc,
            a.vol,
            a.shrout,
            a.bid, a.ask,
            a.cfacpr, a.cfacshr,
            b.shrcd, b.exchcd,
            b.siccd, b.naics,
            b.hsiccd,
            b.ticker, b.comnam
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
            ON a.permno = b.permno
            AND a.date >= b.namedt
            AND a.date <= b.nameendt
        WHERE b.shrcd IN (10, 11)
    """

    db = get_connection()
    df = db.raw_sql(query, date_cols=["date"])
    db.close()

    # ── basic cleaning ────────────────────────────────────────────────────
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    # market cap (in $thousands since shrout is in thousands)
    df["mktcap"] = df["prc"].abs() * df["shrout"]

    # spread
    df["spread"] = df["ask"] - df["bid"]
    df.loc[df["spread"] < 0, "spread"] = None

    out = PATH["RAW_DATA"] / "crsp_monthly.parquet"
    PATH["RAW_DATA"].mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[wrds_import] CRSP monthly: {len(df):,} rows → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRY – add new download functions here
# ══════════════════════════════════════════════════════════════════════════════

DOWNLOADS = {
    "crsp": ("CRSP monthly stock file (shrcd 10/11)", download_crsp_monthly),
    # "compustat": ("Compustat annual fundamentals", download_compustat),
    # "ff":        ("Fama-French factors", download_ff_factors),
}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print("Usage: .venv/bin/python wrds_import.py [dataset ...] [--all] [--list]")
        print()
        for key, (desc, _) in DOWNLOADS.items():
            print(f"  {key:15s}  {desc}")
        return

    if "--list" in args:
        for key, (desc, _) in DOWNLOADS.items():
            print(f"  {key:15s}  {desc}")
        return

    targets = list(DOWNLOADS.keys()) if "--all" in args else args

    for name in targets:
        if name.startswith("-"):
            continue
        if name not in DOWNLOADS:
            print(f"[wrds_import] Unknown dataset: '{name}'.  Use --list to see options.")
            continue
        _, func = DOWNLOADS[name]
        func()

    print("[wrds_import] Done.")


if __name__ == "__main__":
    main()
