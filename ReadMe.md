# Options Flow Scanner

A tool that scans SPY options activity for statistically unusual volume and
tells you how today compares to every day since 2020. Built from scratch as
my first quant project, with the goal of actually testing whether "unusual
options activity" predicts anything — instead of just assuming it does.

## What it does

Every trading day it collects the full SPY option chain (every strike and
expiry, with volume, open interest and implied volatility) and stores it in
a local SQLite database. A detector then compares each contract's volume
against contracts of the same type — same side (call/put), similar distance
from the spot price, similar days to expiry — over the trailing 60 trading
days, using robust statistics (median + MAD instead of mean + std, so one
big block trade can't distort a baseline). Contracts trading far above their
normal level get flagged, and the whole day gets a composite anomaly score.

There's a Streamlit GUI: pick any date on a calendar, scan it, see the
flagged contracts, and export an Excel report that shows where that day
ranks against the full history (percentile and all-time rank).

## The data problem this solves

Free options data has no history — yfinance only shows you today's chain,
and today's expiring contracts disappear from it within hours. So the tool
builds its own history: a daily collector going forward, plus a one-time
backfill of about 9 million rows (2020–2023) parsed from OptionsDX free
end-of-day files. One caveat: the free historical files don't include open
interest, so OI-based signals only accumulate from the live collection
onward.

## Does it predict anything? (the actual research)

This was the point of the project. I ran an event study over 2020–2023:
take the top-decile anomaly-score days (defined using only information
available at the time — expanding threshold, no lookahead), and test whether
SPY's forward returns over 1, 3 and 5 days look any different from normal
days. Welch t-tests plus a 10,000-iteration bootstrap, because options
volume is heavily fat-tailed and I didn't want to trust normality.

**Result: no directional edge.** High anomaly-score days do not predict
which way SPY goes, at any horizon, in any sub-period. Put-heavy anomaly
days don't predict drops either.

Two things made this study better than my first attempt at it:

1. **I caught my own false positive.** The first run showed a "significant"
   volatility effect (p ≈ 0.01). It turned out to be a bug: my forward
   returns were computed on row offsets, so days at the edge of a 2.5-year
   data gap were getting "5-day returns" of +55% measured against prices
   from years later. After fixing the gap handling, the result died
   completely. The most exciting number in the output was an artifact.
2. **One marginal result survived** — days where put anomalies dominate
   call anomalies show slightly larger absolute moves over the next 5 days
   (p ≈ 0.03 on both tests). But it only appears at one horizon out of
   three, and I ran ~18 tests total, so one hit at p ≈ 0.03 is roughly what
   chance owes me. I'm treating it as a hypothesis, not a finding. Since
   the collector now runs daily, it will accumulate true out-of-sample
   evidence on 2026 data that didn't exist when the hypothesis was formed.

**So what is the tool for?** It's a barometer, not a crystal ball. It
reliably fires on real events — validated against the SVB collapse
(March 2023), the COVID crash weeks, and the Jan 2021 meme-stock mania,
which is the most anomalous period in the entire sample. It tells you
*that* and *where* the options market is doing something historically
unusual. It does not tell you what happens next, and the event study is
the receipt for why I won't claim otherwise.

## Known limitations

End-of-day data only. No trade direction (bought vs sold), no intraday
timing — the things paid flow services sell. This measures attention,
not "smart money."
The trailing baseline adapts during long crises: March 2020 scores high
early, then drifts down as the crash itself becomes the baseline. A
trailing-window score measures surprise vs the recent past, not absolute
chaos.
Data gap from 2024-01 to 2026-06 (free historical data ends, live
collection began). Recent dates are currently benchmarked partly against
late-2023 conditions; the app warns when this happens. It self-heals
after ~3 months of daily collection.
No open interest before 2026-07 (not in the free historical files).
Some stale-quote noise on quiet days (contracts with near-zero IV)
Still slips through the filters. On the fix list.

## Project structure
src/
db.py           SQLite schema and helpers
collector.py    Daily yfinance chain collector
backfill.py     One-time OptionsDX historical file parser
detector.py     Anomaly scoring engine + daily composite scores
event_study.py  The statistical validation
report.py       Excel report generator
app.py          Streamlit GUI
main.py           One-click launcher

Data files are not in the repo (gitignored — too big, and the historical
files are licensed to my account). To rebuild: run `src/db.py`, download
the free SPY EOD files from optionsdx.com into `data/optionsdx/`, run
`src/backfill.py`, then `src/detector.py --daily-scores`.

### Second study: does put-call skew predict? (spoiler: it's regime-dependent)

Since the volume study came back null, I tested a signal with an actual
academic prior: put-call skew (the price gap between 5% OTM puts and calls —
the market's crash-insurance premium). Two pre-declared hypotheses: steep
skew *levels* and rapid skew *steepening*, same bootstrap framework, same
gap guards.

The raw result looked spectacular: steep-skew days in 2020–2021 were
followed by strongly positive returns and higher volatility (p < 0.01
across horizons). But the sub-period discipline caught it again — the
effect is completely absent in 2022–2023, the level-based signal barely
fires outside the high-vol era (4 event days in two years, untestable),
and fear days cluster so heavily that 28 "independent" events are really
a handful of episodes. The honest conclusion: during the post-COVID
recovery, buying after fear spikes paid — but so did buying after
everything. That's a regime story, not an edge.

Both open hypotheses (put-tilt volatility, skew-level returns) are frozen
and tracked out-of-sample in the app as daily collection accumulates data
that didn't exist when they were formed.