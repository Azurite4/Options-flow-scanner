# Options Flow Scanner

An end-of-day options activity scanner for SPY. Every trading day it collects the full option chain (volume, open interest, implied volatility for every strike and expiry), stores it in a local database, and flags strikes showing statistically unusual activity relative to their own history.

## Why

Free options data has no history - you can only see today's chain. This tool solves that by building its own historical database, one snapshot per day, which then serves as the baseline for detecting anomalies (eg. a strike trading several standard deviations above its normal volume, or volume far exceeding open interest, which suggests new positioning).

## How it works

1. **Collector** - pulls the SPY option chain after market close and appends it to a SQLite database
2. **Detector** - compares today's snapshot against the stored baseline and flags anomalies
3. **Validation** (planned) - backtests whether flagged anomalies have any predictive power for forward returns

## Status

Early development. Currently building the collector.
