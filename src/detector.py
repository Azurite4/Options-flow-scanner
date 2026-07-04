"""Signal engine v2: robust, liquidity-weighted options volume anomalies.

Changes from v1:
  - Universe restricted to OTM / near-ATM contracts (speculation zone).
  - Robust baselines: median + MAD instead of mean + std, so single
    whale trades can't poison a bucket.
  - Ranking by EXCESS VOLUME (contracts above baseline), with robust-z
    as a filter, so dead buckets can't top the table with z=68 noise.
  - Moneyness labels defined by distance-from-spot (correct for both
    calls and puts).
  - Daily composite score: today's total excess volume as a fraction
    of a normal day's volume. This is what the event study consumes.

Usage:
    py src\\detector.py                          -> report, latest date
    py src\\detector.py 2023-03-10               -> report, given date
    py src\\detector.py --daily-scores           -> score every date,
                                                    write data/daily_scores.csv
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "options.db"
SCORES_CSV = ROOT / "data" / "daily_scores.csv"

LOOKBACK_DAYS = 60
MIN_HISTORY = 30
ROBUST_Z_MIN = 5.0        # filter: contract must be this many robust-sigmas up
MIN_VOLUME = 500
MIN_BUCKET_N = 100        # baseline sample size required
MAD_SCALE = 1.4826        # makes MAD comparable to a std deviation

# Universe: OTM and near-ATM only. Calls: strike >= 0.99*spot.
# Puts: strike <= 1.01*spot. Both capped at 15% from spot.
OTM_TOL = 0.01
MAX_DIST = 0.15

DIST_BINS = [-0.001, 0.01, 0.05, 0.15]
DIST_LABELS = ["ATM <1%", "OTM 1-5%", "OTM 5-15%"]
DTE_BINS = [-1, 1, 7, 30, 90, np.inf]
DTE_LABELS = ["0-1d", "2-7d", "8-30d", "31-90d", ">90d"]


def load_universe(conn, start_date: str | None = None) -> pd.DataFrame:
    """Load the speculation-zone universe: OTM/near-ATM, nonzero volume."""
    q = """
    SELECT snapshot_date, expiry, strike, option_type, volume, implied_vol, spot_price
    FROM chain_snapshots
    WHERE volume > 0
      AND ((option_type = 'C' AND strike >= spot_price * (1 - ?))
        OR (option_type = 'P' AND strike <= spot_price * (1 + ?)))
      AND strike BETWEEN spot_price * (1 - ?) AND spot_price * (1 + ?)
    """
    params = [OTM_TOL, OTM_TOL, MAX_DIST, MAX_DIST]
    if start_date:
        q += " AND snapshot_date >= ?"
        params.append(start_date)
    df = pd.read_sql(q, conn, params=params,
                     parse_dates=["snapshot_date", "expiry"])
    df["dte"] = (df["expiry"] - df["snapshot_date"]).dt.days
    # Distance from spot, signed by direction of the bet, bucketed by magnitude
    df["dist"] = np.where(df["option_type"] == "C",
                          df["strike"] / df["spot_price"] - 1,
                          1 - df["strike"] / df["spot_price"])
    df["dist_bucket"] = pd.cut(df["dist"].clip(lower=0), DIST_BINS, labels=DIST_LABELS)
    df["dte_bucket"] = pd.cut(df["dte"], DTE_BINS, labels=DTE_LABELS)
    df["bucket"] = (df["option_type"].astype(str) + "|"
                    + df["dist_bucket"].astype(str) + "|"
                    + df["dte_bucket"].astype(str))
    return df.dropna(subset=["dist_bucket", "dte_bucket"])


def bucket_baseline(history: pd.DataFrame) -> pd.DataFrame:
    """Robust per-bucket baseline: median, scaled MAD, sample size."""
    def mad(x):
        return np.median(np.abs(x - np.median(x)))
    g = history.groupby("bucket", observed=True)["volume"]
    iv = history[(history["implied_vol"] > 0.03) & (history["implied_vol"] < 3.0)]
    base = pd.DataFrame({
        "med": g.median(),
        "mad": g.apply(mad) * MAD_SCALE,
        "n": g.count(),
        "iv_med": iv.groupby("bucket", observed=True)["implied_vol"].median(),
    })
    # Floor MAD so buckets where >half of volumes are identical don't divide by 0
    base["mad"] = base["mad"].clip(lower=1.0)
    return base


def score_day(today: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    t = today.join(base, on="bucket")
    t["robust_z"] = (t["volume"] - t["med"]) / t["mad"]
    t["excess_vol"] = (t["volume"] - t["med"]).clip(lower=0)
    t["iv_vs_normal"] = t["implied_vol"] / t["iv_med"]
    return t


def flags_for_day(scored: pd.DataFrame) -> pd.DataFrame:
    f = scored[
        (scored["robust_z"] >= ROBUST_Z_MIN)
        & (scored["volume"] >= MIN_VOLUME)
        & (scored["n"] >= MIN_BUCKET_N)
    ]
    return f.sort_values("excess_vol", ascending=False)


def composite(scored: pd.DataFrame, history: pd.DataFrame) -> dict:
    """Daily composite: excess volume as a fraction of a normal day's volume."""
    norm = history.groupby("snapshot_date", observed=True)["volume"].sum().median()
    out = {"score": scored["excess_vol"].sum() / norm}
    for side, name in (("C", "call_score"), ("P", "put_score")):
        out[name] = scored.loc[scored["option_type"] == side, "excess_vol"].sum() / norm
    return out


def run_report(target: str | None) -> None:
    conn = sqlite3.connect(DB_PATH)
    if target is None:
        target = conn.execute("SELECT MAX(snapshot_date) FROM chain_snapshots").fetchone()[0]
    dates = pd.read_sql(
        "SELECT DISTINCT snapshot_date d FROM chain_snapshots WHERE snapshot_date <= ? "
        "ORDER BY d DESC LIMIT ?", conn, params=[target, LOOKBACK_DAYS + 1])["d"].tolist()
    df = load_universe(conn, start_date=min(dates))
    conn.close()

    df = df[df["snapshot_date"] <= target]
    actual = df["snapshot_date"].max()
    today = df[df["snapshot_date"] == actual]
    history = df[df["snapshot_date"] < actual]
    n_days = history["snapshot_date"].nunique()
    if n_days < MIN_HISTORY:
        raise SystemExit(f"Only {n_days} days of history; need {MIN_HISTORY}.")

    scored = score_day(today, bucket_baseline(history))
    comp = composite(scored, history)
    flags = flags_for_day(scored)

    print(f"\n=== Anomaly report v2: {actual.date()} (baseline {n_days} days) ===")
    print(f"Composite score: {comp['score']:.3f}   "
          f"(calls {comp['call_score']:.3f} / puts {comp['put_score']:.3f})")
    print(f"Flagged contracts: {len(flags)}\n")
    if flags.empty:
        print("Quiet day — nothing unusual in the speculation zone.")
        return
    out = flags.head(15)[["option_type", "strike", "expiry", "dte", "dist",
                          "volume", "med", "robust_z", "excess_vol", "implied_vol"]].copy()
    out["expiry"] = out["expiry"].dt.date
    out["dist"] = (out["dist"] * 100).round(1)
    out = out.rename(columns={"dist": "pct_from_spot"})
    out[["med", "robust_z", "excess_vol"]] = out[["med", "robust_z", "excess_vol"]].round(1)
    print(out.to_string(index=False))


def run_daily_scores() -> None:
    """Composite score for every scoreable date -> data/daily_scores.csv."""
    conn = sqlite3.connect(DB_PATH)
    df = load_universe(conn)
    conn.close()

    dates = sorted(df["snapshot_date"].unique())
    print(f"Scoring {len(dates) - MIN_HISTORY} days "
          f"(universe: {len(df):,} contract-days)...")

    records = []
    grouped = dict(tuple(df.groupby("snapshot_date", observed=True)))
    for i in range(MIN_HISTORY, len(dates)):
        window = dates[max(0, i - LOOKBACK_DAYS):i]
        history = pd.concat([grouped[d] for d in window], ignore_index=True)
        today = grouped[dates[i]]
        scored = score_day(today, bucket_baseline(history))
        comp = composite(scored, history)
        records.append({
            "date": pd.Timestamp(dates[i]).date().isoformat(),
            "spot": float(today["spot_price"].iloc[0]),
            "n_flags": int(len(flags_for_day(scored))),
            **{k: round(v, 4) for k, v in comp.items()},
        })
        if (i - MIN_HISTORY) % 100 == 0:
            print(f"  {records[-1]['date']}  score={records[-1]['score']:.3f}")

    out = pd.DataFrame(records)
    out.to_csv(SCORES_CSV, index=False)
    print(f"\nWrote {len(out)} daily scores -> {SCORES_CSV}")
    print("\nTop 10 most anomalous days in history:")
    print(out.nlargest(10, "score")[["date", "score", "call_score", "put_score", "n_flags"]]
          .to_string(index=False))


if __name__ == "__main__":
    if "--daily-scores" in sys.argv:
        run_daily_scores()
    else:
        run_report(sys.argv[1] if len(sys.argv) > 1 else None)