"""Generate a dated Excel data report into reports/ each run."""

import sqlite3
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render charts without opening windows
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils.dataframe import dataframe_to_rows

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "options.db"
REPORTS = ROOT / "reports"
SCORES_CSV = ROOT / "data" / "daily_scores.csv"

HEADER_FILL = PatternFill("solid", start_color="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF", name="Arial")
TITLE_FONT = Font(bold=True, size=14, name="Arial")
BODY_FONT = Font(name="Arial")


def load_day(target_date: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    if target_date is None:
        target_date = conn.execute(
            "SELECT MAX(snapshot_date) FROM chain_snapshots").fetchone()[0]
    df = pd.read_sql(
        "SELECT * FROM chain_snapshots WHERE snapshot_date = ?",
        conn, params=[target_date], parse_dates=["snapshot_date", "expiry"],
    )
    conn.close()
    if df.empty:
        raise ValueError(f"No data for {target_date} (not a trading day in the DB?)")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["dte"] = (df["expiry"] - df["snapshot_date"]).dt.days
    df["moneyness"] = df["strike"] / df["spot_price"]
    df["vol_oi_ratio"] = df["volume"] / df["open_interest"].replace(0, pd.NA)
    return df


def make_charts(df: pd.DataFrame, outdir: Path) -> dict[str, Path]:
    paths = {}

    fig, ax = plt.subplots(figsize=(7, 3.5))
    buckets = pd.cut(df["dte"], bins=[-1, 1, 7, 30, 90, 10_000],
                     labels=["0-1d", "2-7d", "8-30d", "31-90d", ">90d"])
    df.groupby(buckets, observed=True)["volume"].sum().plot.bar(ax=ax, color="#1F3864")
    ax.set_title("Volume by days to expiry"); ax.set_xlabel(""); ax.set_ylabel("Contracts")
    fig.tight_layout()
    paths["dte"] = outdir / "_chart_dte.png"; fig.savefig(paths["dte"], dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    near = df[(df["moneyness"].between(0.9, 1.1)) & (df["dte"] <= 30)]
    for opt, color in (("C", "#2E7D32"), ("P", "#C62828")):
        sub = near[near["option_type"] == opt].groupby("strike")["volume"].sum()
        ax.plot(sub.index, sub.values, label="Calls" if opt == "C" else "Puts", color=color)
    ax.axvline(df["spot_price"].iloc[0], ls="--", color="gray", label="Spot")
    ax.set_title("Volume by strike (±10% of spot, ≤30 DTE)")
    ax.set_xlabel("Strike"); ax.set_ylabel("Contracts"); ax.legend()
    fig.tight_layout()
    paths["strike"] = outdir / "_chart_strike.png"; fig.savefig(paths["strike"], dpi=110); plt.close(fig)

    return paths


def styled_header(ws, row: int, headers: list[str]) -> None:
    for col, h in enumerate(headers, start=1):
        cell = ws.cell(row=row, column=col, value=h)
        cell.fill = HEADER_FILL; cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")

def load_scores() -> pd.DataFrame | None:
    if not SCORES_CSV.exists():
        return None
    return pd.read_csv(SCORES_CSV, parse_dates=["date"])

def build_report(df: pd.DataFrame) -> Path:
    REPORTS.mkdir(exist_ok=True)
    snap = df["snapshot_date"].iloc[0].date().isoformat()
    out = REPORTS / f"report_{snap}.xlsx"

    wb = Workbook()

    # ---- Sheet 1: Summary ----
    ws = wb.active; ws.title = "Summary"
    ws["A1"] = f"SPY Options Chain Report — {snap}"; ws["A1"].font = TITLE_FONT

    call_vol = int(df.loc[df.option_type == "C", "volume"].sum())
    put_vol = int(df.loc[df.option_type == "P", "volume"].sum())
    stats = [
        ("Spot price", round(float(df["spot_price"].iloc[0]), 2)),
        ("Total contracts", len(df)),
        ("Total volume", int(df["volume"].sum())),
        ("Call volume", call_vol),
        ("Put volume", put_vol),
        ("Put/Call ratio", round(put_vol / call_vol, 3)),
        ("Contracts with volume > OI",
         int((df["volume"] > df["open_interest"]).sum())
         if df["open_interest"].notna().any() else "n/a (no OI in source)"),
        ("Expiries", int(df["expiry"].nunique())),
    ]
    styled_header(ws, 3, ["Metric", "Value"])
    for i, (name, val) in enumerate(stats, start=4):
        ws.cell(row=i, column=1, value=name).font = BODY_FONT
        c = ws.cell(row=i, column=2, value=val); c.font = BODY_FONT
        c.number_format = "#,##0" if isinstance(val, int) else "#,##0.00"
    ws.column_dimensions["A"].width = 30; ws.column_dimensions["B"].width = 16

    # ---- Sheet 2: Charts ----
    ws2 = wb.create_sheet("Charts")
    charts = make_charts(df, REPORTS)
    ws2.add_image(XLImage(str(charts["dte"])), "B2")
    ws2.add_image(XLImage(str(charts["strike"])), "B22")

    # ---- Sheet 3: Top activity ----
    ws3 = wb.create_sheet("Top Activity")
    cols = ["option_type", "strike", "expiry", "dte", "volume",
            "open_interest", "vol_oi_ratio", "implied_vol"]
    top = df.nlargest(25, "volume")[cols].copy()
    top["expiry"] = top["expiry"].dt.date.astype(str)
    top["vol_oi_ratio"] = top["vol_oi_ratio"].astype(float).round(2)
    top["implied_vol"] = top["implied_vol"].round(4)
    styled_header(ws3, 1, cols)
    for r_idx, row in enumerate(dataframe_to_rows(top, index=False, header=False), start=2):
        for c_idx, val in enumerate(row, start=1):
            ws3.cell(row=r_idx, column=c_idx, value=val).font = BODY_FONT
    for col_letter in "ABCDEFGH":
        ws3.column_dimensions[col_letter].width = 14
    ws3.freeze_panes = "A2"

# ---- Sheet 4: Historical context ----
    scores = load_scores()
    if scores is not None and (scores["date"] == pd.Timestamp(snap)).any():
        ws4 = wb.create_sheet("Historical Context")
        ws4["A1"] = f"How does {snap} compare to history?"
        ws4["A1"].font = TITLE_FONT

        row_today = scores.loc[scores["date"] == pd.Timestamp(snap)].iloc[0]
        pct = (scores["score"] <= row_today["score"]).mean() * 100
        pct_put = (scores["put_score"] <= row_today["put_score"]).mean() * 100
        pct_call = (scores["call_score"] <= row_today["call_score"]).mean() * 100
        rank = int((scores["score"] > row_today["score"]).sum()) + 1

        ctx = [
            ("Composite score", round(float(row_today["score"]), 3)),
            ("Percentile vs all history", f"{pct:.1f}%"),
            ("Rank (1 = most anomalous ever)", f"{rank} of {len(scores)}"),
            ("Call score percentile", f"{pct_call:.1f}%"),
            ("Put score percentile", f"{pct_put:.1f}%"),
            ("History covers", f"{scores['date'].min().date()} to {scores['date'].max().date()}"),
        ]
        styled_header(ws4, 3, ["Metric", "Value"])
        for i, (name, val) in enumerate(ctx, start=4):
            ws4.cell(row=i, column=1, value=name).font = BODY_FONT
            ws4.cell(row=i, column=2, value=val).font = BODY_FONT
        ws4.column_dimensions["A"].width = 34
        ws4.column_dimensions["B"].width = 26

        # Score-through-time chart with today marked
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(scores["date"], scores["score"], lw=0.7, color="#1F3864")
        ax.axhline(scores["score"].quantile(0.90), ls="--", color="#C62828",
                   lw=1, label="90th percentile")
        ax.scatter([row_today["date"]], [row_today["score"]], color="#C62828",
                   zorder=5, s=60, label=f"This report ({snap})")
        ax.set_title("Daily composite anomaly score, full history")
        ax.legend(); fig.tight_layout()
        chart_path = REPORTS / "_chart_history.png"
        fig.savefig(chart_path, dpi=110); plt.close(fig)
        ws4.add_image(XLImage(str(chart_path)), "D3")
        # register for cleanup alongside the other temp charts
        charts["history"] = chart_path

    wb.save(out)
    for p in charts.values():  # clean up temp chart images
        p.unlink(missing_ok=True)
    return out


if __name__ == "__main__":
    df = load_day()
    path = build_report(df)
    print(f"Report written: {path}")