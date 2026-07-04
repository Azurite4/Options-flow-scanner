"""Options Flow Scanner — local GUI.

Run with:  py -m streamlit run src\\app.py   (or just run main.py)
"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st

from vol_metrics import build_history
from collector import fetch_chain_rows, TICKERS
from db import get_connection, insert_rows
from detector import (run_daily_scores, load_universe, bucket_baseline, score_day,
                      flags_for_day, composite, LOOKBACK_DAYS, MIN_HISTORY)
from report import load_day, build_report
from vol_surface import (load_chain_day, build_grid, surface_stats,
                         plotly_surface, term_structure_fig, smile_fig)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "options.db"
ROOT_SCORES = Path(__file__).resolve().parent.parent / "data" / "daily_scores.csv"
ROOT_VOL = Path(__file__).resolve().parent.parent / "data" / "vol_metrics.csv"

st.set_page_config(page_title="Options Flow Scanner", layout="wide")
st.title("Options Flow Scanner")


@st.cache_data
def get_dates() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    d = [r[0] for r in conn.execute(
        "SELECT DISTINCT snapshot_date FROM chain_snapshots ORDER BY snapshot_date")]
    conn.close()
    return d

@st.cache_data(show_spinner=False)
def get_grid(date_str: str):
    return build_grid(load_chain_day(date_str))

dates = get_dates()
d_min, d_max = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
n_2023_and_earlier = sum(1 for d in dates if d <= "2023-12-31")
n_recent = len(dates) - n_2023_and_earlier

tab_scan, tab_hyp, tab_help = st.tabs(
    ["🔍 Scanner", "🧪 Hypothesis Tracker", "📖 Instructions & Data Coverage"])
# ============================= HELP TAB =============================
with tab_help:
    st.header("What this tool does")
    st.markdown(
        "Every option contract's volume is compared against contracts of the "
        "same type (call/put, distance from spot, days to expiry) over the "
        "trailing 60 trading days, using robust statistics. Contracts trading "
        "far above their bucket's normal level get flagged, and the whole day "
        "gets a **composite anomaly score** you can compare against history.\n\n"
        "⚠️ **What it is not:** a prediction engine. Our event study over "
        "2020–2023 found **no evidence** that high anomaly scores predict "
        "SPY's direction. Treat this as a *barometer* of unusual activity, "
        "not a buy/sell signal."
        "The 🧪 Hypothesis Tracker tab shows the two pre-registered hypotheses from our event studies accumulating out-of-sample evidence as daily collection grows."
    )

    st.header("Which date should I pick?")
    st.markdown(
        f"""
| Situation | What to select |
|---|---|
| **It's a trading day, after ~2:30 PM MT** (market closed) | Today's date — but only if the collector has run today. Run `py src\\collector.py` first if unsure. |
| **It's a trading day, before market close** | The **previous trading day**. Today's volume is still accumulating and would be misleading. |
| **Weekend or market holiday** | The **last trading day** (the calendar auto-snaps back for you). |
| **Researching a past event** | Any date from the historical archive — try 2023-03-10 (SVB collapse). |
"""
    )

    st.header("Data coverage")
    st.markdown(
        f"""
- **Historical archive:** {n_2023_and_earlier} trading days, {dates[0]} → 2023-12-29
  (source: OptionsDX end-of-day files — volume and IV, **no open interest**)
- **Gap:** 2024-01 → 2026-06 (no free data available; fillable later with paid data)
- **Live collection:** {n_recent} day(s) so far, from 2026-07 onward
  (source: yfinance daily snapshots — includes open interest)

**Total: {len(dates)} trading days.**
"""
    )

    st.header("How benchmarking against history works")
    st.markdown(
        """
Two layers of comparison happen on every scan:

1. **Trailing baseline (the flags):** each contract vs. its bucket's last 60
   trading days. For recent dates, that trailing window currently reaches back
   across the data gap into late-2023 — the app shows an amber warning when
   this happens. The composite score is a *ratio* (excess volume ÷ a normal
   day's total volume), so it adjusts for overall market growth reasonably
   well, but cross-era comparisons deserve some skepticism. **This fixes
   itself automatically:** after ~60 trading days of daily collection
   (~3 months), baselines become purely current-era data.

2. **All-time percentile (the Excel report):** the day's composite score is
   ranked against every scored day since 2020 — that's what the *Historical
   Context* sheet shows. A day in the 95th percentile is behaving like the
   SVB collapse or COVID-crash era; a day at the 40th is business as usual.

The daily workflow after market close: (1) collect data — the button appears
automatically when the database is behind; (2) open **Maintenance** and hit
*Refresh score & volatility history* so the new day gets scored; (3) run the
scan. Steps 1–2 only matter for new days; historical dates are already scored.
"""
    )
# ========================= HYPOTHESIS TRACKER =========================

with tab_hyp:
    st.header("Out-of-sample hypothesis tracker")
    st.markdown(
        "Two marginal results from the 2020–2023 studies were **pre-registered as "
        "hypotheses**, not findings. Every day of live collection adds genuinely "
        "out-of-sample evidence — data that did not exist when the hypotheses were "
        "formed. This page accumulates that evidence. No peeking-based tweaks allowed: "
        "the definitions below are frozen."
    )
    OOS_START = "2026-07-01"

    sc = pd.read_csv(ROOT_SCORES, parse_dates=["date"]) if ROOT_SCORES.exists() else None
    vm = pd.read_csv(ROOT_VOL, parse_dates=["date"]) if ROOT_VOL.exists() else None

    def fwd5(df):
        df = df.sort_values("date").reset_index(drop=True)
        df["fwd_ret_5d"] = df["spot"].shift(-5) / df["spot"] - 1
        gap = (df["date"].shift(-5) - df["date"]).dt.days
        df.loc[gap > 20, "fwd_ret_5d"] = pd.NA
        return df

    st.subheader("H1 — Put-tilted anomaly days precede larger 5-day moves")
    st.caption("Definition (frozen): put_score − call_score in the top decile of the "
               "expanding historical distribution. Prediction: larger |5-day return| than baseline.")
    if sc is None:
        st.info("daily_scores.csv not found — run the history refresh in Maintenance.")
    else:
        d = fwd5(sc.copy())
        d["tilt"] = d["put_score"] - d["call_score"]
        d["thresh"] = d["tilt"].expanding(min_periods=100).quantile(0.90).shift(1)
        oos = d[(d["date"] >= OOS_START)]
        ev = oos[oos["tilt"] > oos["thresh"]].dropna(subset=["fwd_ret_5d"])
        base = oos[oos["tilt"] <= oos["thresh"]].dropna(subset=["fwd_ret_5d"])
        c1, c2, c3 = st.columns(3)
        c1.metric("OOS event days so far", len(ev))
        c2.metric("Event mean |5d move|",
                  f"{abs(ev['fwd_ret_5d']).mean()*100:.2f}%" if len(ev) else "—")
        c3.metric("Baseline mean |5d move|",
                  f"{abs(base['fwd_ret_5d']).mean()*100:.2f}%" if len(base) else "—")
        if len(ev) < 15:
            st.caption(f"Verdict pending — need ~15+ event days for a meaningful test "
                       f"(likely 6–12 months of collection).")

    st.subheader("H2 — Steep/steepening skew precedes positive returns (regime question)")
    st.caption("Definition (frozen): skew or its 5-day change in the top decile, expanding "
               "threshold. 2020–21 said yes strongly; 2022–23 said no. Out-of-sample data "
               "is the tiebreaker.")
    if vm is None or sc is None:
        st.info("vol_metrics.csv not found — run the history refresh in Maintenance.")
    else:
        d = vm.merge(sc[["date", "spot"]], on="date", how="inner")
        d = fwd5(d.dropna(subset=["skew"]).copy())
        d["thresh"] = d["skew"].expanding(min_periods=100).quantile(0.90).shift(1)
        oos = d[d["date"] >= OOS_START]
        ev = oos[oos["skew"] > oos["thresh"]].dropna(subset=["fwd_ret_5d"])
        base = oos[oos["skew"] <= oos["thresh"]].dropna(subset=["fwd_ret_5d"])
        c1, c2, c3 = st.columns(3)
        c1.metric("OOS steep-skew days so far", len(ev))
        c2.metric("Event mean 5d return",
                  f"{ev['fwd_ret_5d'].mean()*100:+.2f}%" if len(ev) else "—")
        c3.metric("Baseline mean 5d return",
                  f"{base['fwd_ret_5d'].mean()*100:+.2f}%" if len(base) else "—")
        if len(ev) < 15:
            st.caption("Verdict pending — evidence accumulates with every collection day.")

# ============================ SCANNER TAB ============================
with tab_scan:
    st.caption(f"Database: {len(dates)} trading days, {dates[0]} → {dates[-1]}")

# ---------- Freshness check & one-click collection ----------
    @st.cache_data(ttl=3600)
    def last_expected_trading_day() -> date:
        """Most recent NYSE session whose close has already passed."""
        now_et = pd.Timestamp.now(tz="America/New_York")
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=now_et.date() - timedelta(days=10),
                              end_date=now_et.date())
        closed = sched[sched["market_close"] <= now_et]
        return closed.index[-1].date()

    latest_in_db = date.fromisoformat(dates[-1])
    expected = last_expected_trading_day()

    if latest_in_db < expected:
        nyse_now = pd.Timestamp.now(tz="America/New_York")
        sched_today = mcal.get_calendar("NYSE").schedule(
            start_date=nyse_now.date(), end_date=nyse_now.date())
        market_open = (not sched_today.empty
                       and sched_today["market_open"].iloc[0] <= nyse_now
                       <= sched_today["market_close"].iloc[0])

        st.warning(f"Database's latest snapshot is **{latest_in_db}**, but the most "
                   f"recent completed trading day is **{expected}**.")
        if market_open:
            st.caption("⚠️ The market is currently open — collecting now stores "
                       "partial-day volume. Best practice: collect after 2:00 PM MT.")

        if st.button("📡 Collect latest data now"):
            with st.spinner("Pulling option chains from yfinance (1–3 min)..."):
                conn = get_connection()
                try:
                    for tkr in TICKERS:
                        rows = fetch_chain_rows(tkr)
                        insert_rows(conn, rows)
                        st.write(f"{tkr}: {len(rows):,} contracts stored")
                finally:
                    conn.close()
            get_dates.clear()               # invalidate the cached date list
            st.success("Collection complete — refreshing...")
            st.rerun()
    else:
        st.caption(f"✅ Data is up to date (latest snapshot: {latest_in_db}).")
    # -------------------------------------------------------------

    with st.expander("🔧 Maintenance — refresh history tables"):
        st.markdown(
            "The percentile rankings and volatility context in reports come from two "
            "pre-computed tables (`daily_scores.csv` and `vol_metrics.csv`). **Run this "
            "after collecting new data** so new days get scored and included in the "
            "history. Takes a few minutes — it re-scores every day since 2020."
        )
        if st.button("Refresh score & volatility history"):
            with st.spinner("Rebuilding daily anomaly scores (this is the slow part)..."):
                run_daily_scores()
            with st.spinner("Rebuilding volatility metrics..."):
                build_history()
            get_dates.clear()
            st.success("History tables refreshed — new reports will include the latest days.")

    picked = st.date_input("Pick a date", value=d_max, min_value=d_min, max_value=d_max)
    target = max((d for d in dates if d <= picked.isoformat()), default=None)

    if target is None:
        st.error("No data on or before that date.")
        st.stop()
    elif target != picked.isoformat():
        st.info(f"No snapshot for {picked} (weekend, holiday, or data gap). "
                f"Using the most recent trading day before it: **{target}**")

    if st.button(f"Run scan for {target}", type="primary"):
        with st.spinner("Scoring chain against trailing baseline..."):
            conn = sqlite3.connect(DB_PATH)
            uni = load_universe(conn)
            conn.close()

            uni = uni[uni["snapshot_date"] <= target]
            actual = uni["snapshot_date"].max()
            today = uni[uni["snapshot_date"] == actual]
            history = uni[uni["snapshot_date"] < actual]
            history = history[history["snapshot_date"].isin(
                sorted(history["snapshot_date"].unique())[-LOOKBACK_DAYS:])]

            n_days = history["snapshot_date"].nunique()
            if n_days < MIN_HISTORY:
                st.error(f"Only {n_days} days of usable history before {target}; "
                         f"need {MIN_HISTORY}.")
                st.stop()

            hist_dates = pd.to_datetime(sorted(history["snapshot_date"].unique()))
            span = int((pd.Timestamp(actual) - hist_dates.min()).days)

            scored = score_day(today, bucket_baseline(history))
            st.session_state["scan"] = {
                "target": target,
                "comp": composite(scored, history),
                "flags": flags_for_day(scored),
                "span": span,
                "n_days": n_days,
            }
            st.session_state.pop("report_path", None)

    scan = st.session_state.get("scan")
    if scan is not None and scan["target"] != target:
        st.caption(f"Showing nothing yet for {target} — hit Run scan. "
                   f"(Last scan was {scan['target']}.)")
    elif scan is not None:
        comp, flags = scan["comp"], scan["flags"]

        if scan["span"] > LOOKBACK_DAYS * 3:
            st.warning(f"Baseline spans {scan['span']} calendar days due to the gap in "
                       f"the database — this date is compared against older market "
                       f"conditions (see Instructions tab). Read scores with caution.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Composite score", f"{comp['score']:.3f}")
        c2.metric("Call score", f"{comp['call_score']:.3f}")
        c3.metric("Put score", f"{comp['put_score']:.3f}")
        c4.metric("Flagged contracts", len(flags))

        st.subheader("Top anomalies")
        if flags.empty:
            st.info("Quiet day — nothing unusual in the speculation zone.")
        else:
            show = flags.head(25)[["option_type", "strike", "expiry", "dte", "dist",
                                   "volume", "med", "robust_z", "excess_vol"]].copy()
            show["expiry"] = show["expiry"].dt.date.astype(str)
            show["dist"] = (show["dist"] * 100).round(1)
            show = show.rename(columns={"dist": "% from spot"})
            st.dataframe(show, use_container_width=True, hide_index=True)

        st.subheader("Implied volatility surface")
        try:
            grid = get_grid(target)
            sstats = surface_stats(grid)
            sc1, sc2 = st.columns(2)
            slope = sstats.get("term_slope")
            sc1.metric("Term-structure slope",
                       f"{slope:+.3f}" if slope is not None else "n/a",
                       delta="INVERTED — stress" if slope is not None and slope < 0
                             else "normal", delta_color="inverse")
            lw = sstats.get("left_wing")
            sc2.metric("Left-wing steepness (~1m)",
                       f"{lw:+.3f}" if lw is not None else "n/a")

            abs_scale = st.toggle("Absolute scale (compare across dates)", value=True,
                                  help="On: fixed IV scale — crisis days tower, calm days "
                                       "sit flat. Off: surface fills the frame.")
            st.plotly_chart(plotly_surface(grid, target, absolute_scale=abs_scale),
                            use_container_width=True)
            t1, t2 = st.columns(2)
            t1.plotly_chart(term_structure_fig(grid, target), use_container_width=True)
            t2.plotly_chart(smile_fig(grid, target), use_container_width=True)
            st.caption("Drag to rotate. The 2D slices below cut through the 3D: term "
                       "structure (left) inverts under stress; the smile (right) shows "
                       "the crash-insurance premium on the left wing.")
        except Exception as exc:
            st.info(f"Surface unavailable for this date: {exc}")

        st.subheader("Excel report")
        if st.button("📄 Generate & archive Excel report"):
            with st.spinner("Generating Excel report (10-20s)..."):
                try:
                    path = build_report(load_day(target))
                    st.session_state["report_path"] = str(path)
                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")

        rp = st.session_state.get("report_path")
        if rp:
            st.success(f"Report saved to:  `{rp}`")
            with open(rp, "rb") as f:
                st.download_button("Download report", f, file_name=Path(rp).name,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")