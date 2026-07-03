"""One-time backfill: load OptionsDX EOD text files into chain_snapshots.

Usage:  py src\\backfill.py
Safe to re-run: primary key + INSERT OR REPLACE makes it idempotent.
"""

import time
from pathlib import Path

import pandas as pd

from db import get_connection, insert_rows, row_count

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "optionsdx"
TICKER = "SPY"


def parse_file(path: Path) -> list[dict]:
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    # Header names come as '[QUOTE_DATE]' -> strip brackets/whitespace
    df.columns = [c.strip().strip("[]").strip() for c in df.columns]

    rows: list[dict] = []
    for side, prefix in (("C", "C_"), ("P", "P_")):
        sub = pd.DataFrame({
            "snapshot_date": df["QUOTE_DATE"].astype(str).str.strip(),
            "ticker": TICKER,
            "expiry": df["EXPIRE_DATE"].astype(str).str.strip(),
            "strike": pd.to_numeric(df["STRIKE"], errors="coerce"),
            "option_type": side,
            "volume": pd.to_numeric(df[prefix + "VOLUME"], errors="coerce").fillna(0).astype(int),
            "open_interest": None,  # not provided in OptionsDX free files
            "implied_vol": pd.to_numeric(df[prefix + "IV"], errors="coerce"),
            "last_price": pd.to_numeric(df[prefix + "LAST"], errors="coerce"),
            "spot_price": pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce"),
            "source": "optionsdx",
        })
        sub = sub.dropna(subset=["strike", "spot_price"])
        # NaN -> None so SQLite stores proper NULLs
        sub = sub.astype(object).where(pd.notna(sub), None)
        rows.extend(sub.to_dict("records"))
    return rows


def main() -> None:
    files = sorted(RAW_DIR.rglob("*.txt"))
    if not files:
        print(f"No .txt files found under {RAW_DIR}")
        return
    print(f"Found {len(files)} files to load.\n")

    conn = get_connection()
    start = time.time()
    try:
        for i, path in enumerate(files, 1):
            t0 = time.time()
            rows = parse_file(path)
            written = insert_rows(conn, rows)
            print(f"[{i:>2}/{len(files)}] {path.name}: {written:,} rows in {time.time()-t0:.1f}s")
        total = row_count(conn)
        dates = conn.execute(
            "SELECT MIN(snapshot_date), MAX(snapshot_date), COUNT(DISTINCT snapshot_date) "
            "FROM chain_snapshots"
        ).fetchone()
        print(f"\nDone in {(time.time()-start)/60:.1f} min.")
        print(f"Database now: {total:,} rows, {dates[2]} trading days, {dates[0]} -> {dates[1]}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()