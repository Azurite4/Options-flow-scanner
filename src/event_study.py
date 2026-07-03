"""Event study: do high anomaly-score days predict SPY's future behavior?

Tests, over 2020-2023:
  1. Direction: are forward returns after high-score days different?
  2. Magnitude: are forward moves BIGGER after high-score days (vol prediction)?
  3. Tilt: does put_score - call_score predict direction?

Methodology: top-decile score days vs the rest; Welch t-tests plus a
bootstrap (shuffle event labels 10,000x) that doesn't trust normality.

Usage:  py src\\event_study.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "data" / "daily_scores.csv"

HORIZONS = [1, 3, 5]
TOP_PCT = 0.90          # "event" = score in top 10% of trailing distribution
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def bootstrap_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided p-value for difference in means via label shuffling."""
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n = len(a)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        RNG.shuffle(pooled)
        diffs[i] = pooled[:n].mean() - pooled[n:].mean()
    return float((np.abs(diffs) >= abs(observed)).mean())


def analyze(df: pd.DataFrame, event_col: str, label: str) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    ev, base = df[df[event_col]], df[~df[event_col]]
    print(f"Event days: {len(ev)}   Baseline days: {len(base)}")

    for h in HORIZONS:
        r = f"fwd_ret_{h}d"
        a, b = ev[r].dropna().values, base[r].dropna().values
        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        p_boot = bootstrap_pvalue(a.copy(), b.copy())
        print(f"\n  {h}-day forward RETURN (direction):")
        print(f"    event mean {a.mean()*100:+.3f}%  vs baseline {b.mean()*100:+.3f}%   "
              f"t={t:+.2f}  p(t)={p_t:.3f}  p(boot)={p_boot:.3f}")

        aa, ab = np.abs(a), np.abs(b)
        t2, p_t2 = stats.ttest_ind(aa, ab, equal_var=False)
        p_boot2 = bootstrap_pvalue(aa.copy(), ab.copy())
        print(f"  {h}-day forward |RETURN| (magnitude/volatility):")
        print(f"    event mean {aa.mean()*100:.3f}%  vs baseline {ab.mean()*100:.3f}%   "
              f"t={t2:+.2f}  p(t)={p_t2:.3f}  p(boot)={p_boot2:.3f}")


def main() -> None:
    df = pd.read_csv(SCORES, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    for h in HORIZONS:
        df[f"fwd_ret_{h}d"] = df["spot"].shift(-h) / df["spot"] - 1
        gap = (df["date"].shift(-h) - df["date"]).dt.days
        df.loc[gap > h * 4, f"fwd_ret_{h}d"] = np.nan   # void returns across data gaps

    # Event definition uses an EXPANDING trailing quantile — no lookahead:
    # a day only counts as an event relative to scores known by then.
    df["threshold"] = df["score"].expanding(min_periods=100).quantile(TOP_PCT).shift(1)
    df["event"] = df["score"] > df["threshold"]

    # Put-tilt: puts dominating the anomaly flow
    df["tilt"] = df["put_score"] - df["call_score"]
    df["tilt_thresh"] = df["tilt"].expanding(min_periods=100).quantile(TOP_PCT).shift(1)
    df["put_tilt_event"] = df["tilt"] > df["tilt_thresh"]

    usable = df.dropna(subset=["threshold"]).copy()
    print(f"Sample: {usable['date'].min().date()} -> {usable['date'].max().date()}"
          f"  ({len(usable)} days after warm-up)")

    analyze(usable, "event", "TEST 1: HIGH COMPOSITE SCORE days (top decile)")
    analyze(usable, "put_tilt_event", "TEST 2: PUT-TILTED anomaly days (top decile of put-call tilt)")

    # Sub-period robustness: does anything survive outside the 2021 mania?
    for name, lo, hi in [("2020-2021", "2020-01-01", "2021-12-31"),
                         ("2022-2023", "2022-01-01", "2023-12-31")]:
        sub = usable[(usable["date"] >= lo) & (usable["date"] <= hi)]
        if sub["event"].sum() >= 10:
            analyze(sub, "event", f"ROBUSTNESS: high-score days, {name} only")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS:")
    print("  p < 0.05 on BOTH p(t) and p(boot) = worth taking seriously.")
    print("  But we ran ~18 tests: expect ~1 false positive at p=0.05.")
    print("  A real effect should survive sub-periods and multiple horizons.")
    print("  Magnitude (|return|) significance with direction insignificance")
    print("  = 'anomalies predict MOVEMENT, not direction' — still useful.")
    print("=" * 70)


if __name__ == "__main__":
    main()