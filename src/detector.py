"""Signal engine: bucket-normalized volume anomaly detection.

For a given snapshot date, classify every contract into a
(moneyness bucket x DTE bucket) cell and compare its volume against
that cell's trailing history. Output: the most anomalous contracts.

Usage:
    py src\\detector.py              -> latest date in DB
    py src\\detector.py 2023-03-10   -> specific date
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "options.db"

LOOKBACK_DAYS = 60          # trailing window for the baseline
MIN_HISTORY = 30            # need at least this many days to score
Z_THRESHOLD = 3.0           # flag contracts above this z-score
MIN_VOLUME = 500            # ignore tiny-volume noise

MONEYNESS_BINS = [0, 0.85, 0.95, 0.99, 1.01, 1.05, 1.15, np.inf]
MONEYNESS_LABELS = ["deep OTM put-side", "OTM 5-15%", "near OTM 1-5%", "ATM ±1%",
                    "near OTM 1-5% (up)", "OTM 5-15% (up)", "deep OTM call-side"]
DTE_BINS = [-1, 1, 7, 30, 90, np.inf]
DTE_LABELS = ["0-1d", "2-7d", "8-30d", "31-90d", ">90d"]


def load_window(conn, end_date: str, lookback: int) -> pd.DataFrame:
    dates = pd.read_sql(
        "SELECT DISTINCT snapshot_date FROM chain_snapshots "
        "WHERE snapshot_date <= ? ORDER BY snapshot_date DESC LIMIT ?",
        conn, params=[end_date, lookback + 1],
    )["snapshot_date"].tolist()
    if not dates:
        raise SystemExit(f"No data on or before {end_date}")
    placeholders = ",".join("?" * len(dates))
    df = pd.read_sql(
        f"SELECT snapshot_date, expiry, strike, option_type, volume, "
        f"implied_vol, spot_price FROM chain_snapshots "
        f"WHERE snapshot_date IN ({placeholders})",
        conn, params=dates, parse_dates=["snapshot_date", "expiry"],
    )
    return df


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dte"] = (df["expiry"] - df["snapshot_date"]).dt.days
    df["moneyness"] = df["strike"] / df["spot_price"]
    df["m_bucket"] = pd.cut(df["moneyness"], MONEYNESS_BINS, labels=MONEYNESS_LABELS)
    df["dte_bucket"] = pd.cut(df["dte"], DTE_BINS, labels=DTE_LABELS)
    df["bucket"] = (df["option_type"].astype(str) + "|"
                    + df["m_bucket"].astype(str) + "|"
                    + df["dte_bucket"].astype(str))
    return df


def detect(conn, target_date: str) -> pd.DataFrame:
    df = add_buckets(load_window(conn, target_date, LOOKBACK_DAYS))

    actual_target = df["snapshot_date"].max()
    today = df[df["snapshot_date"] == actual_target].copy()
    history = df[df["snapshot_date"] < actual_target]

    n_days = history["snapshot_date"].nunique()
    if n_days < MIN_HISTORY:
        raise SystemExit(f"Only {n_days} days of history before {target_date}; "
                         f"need {MIN_HISTORY}.")

    # Baseline: per-contract volume distribution within each bucket,
    # built from all contract-days in the trailing window
    base = history.groupby("bucket", observed=True)["volume"].agg(
        mu="mean", sigma="std", n="count")
    today = today.join(base, on="bucket")

    today["z_score"] = (today["volume"] - today["mu"]) / today["sigma"].replace(0, np.nan)
    flags = today[
        (today["z_score"] >= Z_THRESHOLD)
        & (today["volume"] >= MIN_VOLUME)
        & (today["n"] >= 100)          # bucket must have a real sample
    ].sort_values("z_score", ascending=False)

    cols = ["snapshot_date", "option_type", "strike", "expiry", "dte",
            "moneyness", "bucket", "volume", "mu", "z_score", "implied_vol"]
    return flags[cols], actual_target, n_days


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = conn.execute(
            "SELECT MAX(snapshot_date) FROM chain_snapshots").fetchone()[0]

    flags, actual, n_days = detect(conn, target)
    conn.close()

    print(f"\n=== Anomaly report for {actual.date()} "
          f"(baseline: {n_days} trailing days) ===")
    print(f"Flagged contracts (z >= {Z_THRESHOLD}, volume >= {MIN_VOLUME}): {len(flags)}\n")
    if flags.empty:
        print("No anomalies today — a quiet, normal day in the chain.")
        return
    out = flags.head(20).copy()
    out["expiry"] = out["expiry"].dt.date
    out["snapshot_date"] = out["snapshot_date"].dt.date
    out["moneyness"] = out["moneyness"].round(3)
    out["mu"] = out["mu"].round(0)
    out["z_score"] = out["z_score"].round(1)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()