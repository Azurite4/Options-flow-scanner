"""Volatility metrics from stored implied vols.

Two daily metrics:
  - atm_iv:  mean IV of near-ATM contracts, 20-45 DTE  (our homemade VIX)
  - skew:    IV of ~5% OTM puts minus ~5% OTM calls, 20-45 DTE (fear gauge)

Usage:
    py src\\vol_metrics.py             -> print metrics for latest date
    py src\\vol_metrics.py --history   -> compute all dates -> data/vol_metrics.csv
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "options.db"
METRICS_CSV = ROOT / "data" / "vol_metrics.csv"

IV_MIN, IV_MAX = 0.03, 3.0        # sanity filter: kill stale/broken quotes
DTE_LO, DTE_HI = 20, 45           # the "standard" measurement window
ATM_BAND = 0.01                   # ATM = within 1% of spot
OTM_TARGET, OTM_BAND = 0.05, 0.02 # skew wings: 5% +/- 2% from spot


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["implied_vol"] = pd.to_numeric(df["implied_vol"], errors="coerce")
    df = df[(df["implied_vol"] > IV_MIN) & (df["implied_vol"] < IV_MAX)]
    df = df[(df["dte"] >= DTE_LO) & (df["dte"] <= DTE_HI)]
    return df


def day_metrics(day: pd.DataFrame) -> dict:
    """Compute atm_iv and skew for one day's chain (needs dte + moneyness cols)."""
    d = _clean(day)
    out = {"atm_iv": np.nan, "put_iv_5otm": np.nan,
           "call_iv_5otm": np.nan, "skew": np.nan}
    if d.empty:
        return out

    atm = d[d["moneyness"].between(1 - ATM_BAND, 1 + ATM_BAND)]
    if len(atm) >= 4:
        out["atm_iv"] = float(atm["implied_vol"].mean())

    puts = d[(d["option_type"] == "P")
             & d["moneyness"].between(1 - OTM_TARGET - OTM_BAND,
                                      1 - OTM_TARGET + OTM_BAND)]
    calls = d[(d["option_type"] == "C")
              & d["moneyness"].between(1 + OTM_TARGET - OTM_BAND,
                                       1 + OTM_TARGET + OTM_BAND)]
    if len(puts) >= 2 and len(calls) >= 2:
        out["put_iv_5otm"] = float(puts["implied_vol"].mean())
        out["call_iv_5otm"] = float(calls["implied_vol"].mean())
        out["skew"] = out["put_iv_5otm"] - out["call_iv_5otm"]
    return out


def _load_all() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT snapshot_date, expiry, strike, option_type, implied_vol, spot_price "
        "FROM chain_snapshots WHERE implied_vol IS NOT NULL",
        conn, parse_dates=["snapshot_date", "expiry"])
    conn.close()
    df["dte"] = (df["expiry"] - df["snapshot_date"]).dt.days
    df["moneyness"] = df["strike"] / df["spot_price"]
    return df


def build_history() -> pd.DataFrame:
    df = _load_all()
    records = []
    for d, day in df.groupby("snapshot_date", observed=True):
        m = day_metrics(day)
        records.append({"date": pd.Timestamp(d).date().isoformat(),
                        **{k: (round(v, 4) if pd.notna(v) else None)
                           for k, v in m.items()}})
    out = pd.DataFrame(records).sort_values("date")
    out.to_csv(METRICS_CSV, index=False)
    print(f"Wrote {len(out)} days -> {METRICS_CSV}")
    print("\nHighest ATM IV days (should be the crisis dates):")
    print(out.nlargest(8, "atm_iv")[["date", "atm_iv", "skew"]].to_string(index=False))
    print("\nSteepest skew days (fear without necessarily high vol):")
    print(out.nlargest(8, "skew")[["date", "atm_iv", "skew"]].to_string(index=False))
    return out


def load_metrics() -> pd.DataFrame | None:
    if not METRICS_CSV.exists():
        return None
    return pd.read_csv(METRICS_CSV, parse_dates=["date"])


if __name__ == "__main__":
    if "--history" in sys.argv:
        build_history()
    else:
        df = _load_all()
        latest = df["snapshot_date"].max()
        m = day_metrics(df[df["snapshot_date"] == latest])
        print(f"{latest.date()}:  ATM IV = {m['atm_iv']:.3f}   "
              f"skew = {m['skew']:.4f}  "
              f"(put {m['put_iv_5otm']:.3f} / call {m['call_iv_5otm']:.3f})")