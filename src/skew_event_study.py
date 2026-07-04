"""Event study #2: does put-call skew predict SPY's future behavior?

Pre-declared hypotheses (before looking at results):
  A. LEVEL:  days with unusually steep skew (top decile, expanding threshold)
  B. CHANGE: days where skew steepened unusually fast over the prior 5 days
Each tested on forward returns (direction) and |returns| (magnitude)
at 1, 3, 5 day horizons, pooled and by sub-period.

Prior: academic literature (e.g. Xing/Zhang/Zhao) finds modest predictive
content in skew for returns — unlike the volume study, a positive here
would not be theoretically surprising. A null is still a fine answer.

Usage:  py src\\skew_event_study.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
VOL_CSV = ROOT / "data" / "vol_metrics.csv"
SCORES_CSV = ROOT / "data" / "daily_scores.csv"   # for spot prices

HORIZONS = [1, 3, 5]
TOP_PCT = 0.90
N_BOOT = 10_000
RNG = np.random.default_rng(42)


def bootstrap_pvalue(a: np.ndarray, b: np.ndarray) -> float:
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
    if len(ev) < 10:
        print("  Too few event days to test meaningfully — skipping.")
        return

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


def expanding_event(s: pd.Series, min_periods: int = 100) -> pd.Series:
    """Top-decile flag using only information available at the time."""
    thresh = s.expanding(min_periods=min_periods).quantile(TOP_PCT).shift(1)
    return s > thresh


def main() -> None:
    vol = pd.read_csv(VOL_CSV, parse_dates=["date"])
    scores = pd.read_csv(SCORES_CSV, parse_dates=["date"])[["date", "spot"]]
    df = vol.merge(scores, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["skew"]).reset_index(drop=True)

    # Forward returns with the gap guard (lesson learned the hard way)
    for h in HORIZONS:
        df[f"fwd_ret_{h}d"] = df["spot"].shift(-h) / df["spot"] - 1
        gap = (df["date"].shift(-h) - df["date"]).dt.days
        df.loc[gap > h * 4, f"fwd_ret_{h}d"] = np.nan

    # Hypothesis A: skew LEVEL. Same gap guard logic applies to the change calc.
    df["event_level"] = expanding_event(df["skew"])

    # Hypothesis B: skew CHANGE over 5 trading days (void across gaps)
    df["skew_chg_5d"] = df["skew"] - df["skew"].shift(5)
    back_gap = (df["date"] - df["date"].shift(5)).dt.days
    df.loc[back_gap > 20, "skew_chg_5d"] = np.nan
    df["event_change"] = expanding_event(df["skew_chg_5d"].fillna(-np.inf))

    usable = df[df["skew"].expanding(min_periods=100).count().shift(1).notna()].copy()
    usable = usable.iloc[100:]  # drop warm-up
    print(f"Sample: {usable['date'].min().date()} -> {usable['date'].max().date()}"
          f"  ({len(usable)} days after warm-up)")
    print(f"Pre-declared: 2 hypothesis families x 3 horizons x 2 outcomes "
          f"(+ sub-periods) — interpret p-values accordingly.")

    analyze(usable, "event_level", "HYPOTHESIS A: STEEP SKEW days (top decile, level)")
    analyze(usable, "event_change", "HYPOTHESIS B: RAPID STEEPENING days (top decile, 5d change)")

    for name, lo, hi in [("2020-2021", "2020-01-01", "2021-12-31"),
                         ("2022-2023", "2022-01-01", "2023-12-31")]:
        sub = usable[(usable["date"] >= lo) & (usable["date"] <= hi)]
        analyze(sub, "event_level", f"ROBUSTNESS A: steep skew, {name}")
        analyze(sub, "event_change", f"ROBUSTNESS B: steepening, {name}")

    print("\n" + "=" * 70)
    print("HOW TO READ THIS:")
    print("  Prior from the literature: skew has MODEST predictive content for")
    print("  returns. A weak positive here is plausible; a null is informative.")
    print("  ~36 tests total incl. sub-periods: at p=0.05 expect ~2 by chance.")
    print("  Believe only what shows up at multiple horizons AND sub-periods,")
    print("  or a pooled result with p well below 0.01.")
    print("=" * 70)


if __name__ == "__main__":
    main()