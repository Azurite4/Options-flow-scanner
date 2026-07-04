"""Implied volatility surface: build a clean moneyness x DTE grid from the
stored chain, and render it as a heatmap (report) or interactive 3D (GUI).

Usage:
    py src\\vol_surface.py               -> surface stats, latest date
    py src\\vol_surface.py 2020-03-16    -> peak-COVID surface stats
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "options.db"

IV_MIN, IV_MAX = 0.03, 3.0
MONEY_EDGES = np.arange(0.85, 1.16, 0.025)      # 0.85 .. 1.15 in 2.5% steps
DTE_EDGES = [0, 8, 15, 30, 60, 91, 183, 366]
DTE_LABELS = ["<1w", "1-2w", "2-4w", "1-2m", "2-3m", "3-6m", "6-12m"]
MIN_CELL_N = 2                                   # need >=2 quotes per cell


def load_chain_day(target: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT snapshot_date, expiry, strike, option_type, implied_vol, spot_price "
        "FROM chain_snapshots WHERE snapshot_date = ? AND implied_vol IS NOT NULL",
        conn, params=[target], parse_dates=["snapshot_date", "expiry"])
    conn.close()
    if df.empty:
        raise ValueError(f"No IV data for {target}")
    df["dte"] = (df["expiry"] - df["snapshot_date"]).dt.days
    df["moneyness"] = df["strike"] / df["spot_price"]
    df = df[(df["implied_vol"] > IV_MIN) & (df["implied_vol"] < IV_MAX)]
    df = df[(df["dte"] >= 1) & (df["dte"] <= 365)]
    df = df[df["moneyness"].between(MONEY_EDGES[0], MONEY_EDGES[-1])]
    # OTM-only convention: puts below spot, calls above (the liquid wings)
    df = df[((df["option_type"] == "P") & (df["moneyness"] <= 1.0))
            | ((df["option_type"] == "C") & (df["moneyness"] > 1.0))]
    return df


def build_grid(day: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows = DTE bucket, cols = moneyness bucket, values = median IV."""
    d = day.copy()
    d["m_bin"] = pd.cut(d["moneyness"], MONEY_EDGES)
    d["t_bin"] = pd.cut(d["dte"], DTE_EDGES, labels=DTE_LABELS)
    g = d.groupby(["t_bin", "m_bin"], observed=True)["implied_vol"]
    grid = g.median().unstack()
    counts = g.count().unstack()
    grid = grid.where(counts >= MIN_CELL_N)
    grid.columns = [f"{c.left:.3f}-{c.right:.3f}" for c in grid.columns]
    return grid


def surface_stats(grid: pd.DataFrame) -> dict:
    """Summary descriptors of the surface shape."""
    money_mids = np.array([(float(c.split("-")[0]) + float(c.split("-")[1])) / 2
                           for c in grid.columns])
    atm_col = grid.columns[np.argmin(np.abs(money_mids - 1.0))]
    atm_by_term = grid[atm_col]

    stats = {}
    # Term structure slope: long-dated ATM IV minus short-dated ATM IV.
    # Negative = inverted = "the danger is NOW".
    short = atm_by_term.iloc[:2].dropna()
    long = atm_by_term.iloc[-2:].dropna()
    if len(short) and len(long):
        stats["term_slope"] = float(long.mean() - short.mean())
    # Left-wing steepness at ~1 month: 10-12.5% OTM put IV minus ATM IV
    row = grid.loc["1-2m"] if "1-2m" in grid.index else grid.iloc[len(grid) // 2]
    left = row.iloc[:3].dropna()
    atm_val = row[atm_col]
    if len(left) and pd.notna(atm_val):
        stats["left_wing"] = float(left.mean() - atm_val)
    stats["atm_1to2m"] = float(atm_val) if pd.notna(atm_val) else None
    return stats

def heatmap_png(grid: pd.DataFrame, target: str, out_path: Path) -> Path:
    """Static heatmap for the Excel report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.8))
    data = grid.values * 100
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([c.split("-")[0] for c in grid.columns], fontsize=7)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index, fontsize=8)
    ax.set_xlabel("Moneyness (strike / spot)", fontsize=8)
    ax.set_ylabel("Time to expiry", fontsize=8)
    ax.set_title(f"Implied volatility surface — {target} (IV %, red = expensive)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plotly_surface(grid: pd.DataFrame, target: str, absolute_scale: bool = True):
    """Interactive 3D surface. absolute_scale=True pins the z-axis and colors
    to a fixed 0-100% IV range so surfaces are comparable across dates;
    False lets the day fill the frame (shape detail, no comparability)."""
    import plotly.graph_objects as go

    money_mids = [(float(c.split("-")[0]) + float(c.split("-")[1])) / 2
                  for c in grid.columns]
    z = grid.values * 100
    surf = dict(
        z=z, x=money_mids, y=list(range(len(grid.index))),
        colorscale="RdYlGn", reversescale=True,
        colorbar=dict(title="IV %"), connectgaps=True,
    )
    if absolute_scale:
        surf["cmin"], surf["cmax"] = 0, 60   # fixed color meaning across all dates

    fig = go.Figure(data=[go.Surface(**surf)])
    scene = dict(
        xaxis_title="Moneyness (strike/spot)",
        yaxis=dict(title="Expiry", tickvals=list(range(len(grid.index))),
                   ticktext=list(grid.index)),
        zaxis_title="IV %",
        camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.9)),
    )
    if absolute_scale:
        scene["zaxis"] = dict(title="IV %", range=[0, 100])
    fig.update_layout(title=f"Implied volatility surface — {target}"
                            + ("  (absolute scale)" if absolute_scale else "  (auto scale)"),
                      scene=scene, height=560, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def term_structure_fig(grid: pd.DataFrame, target: str):
    """2D slice: ATM IV vs expiry. Upward slope = normal; downward = stress."""
    import plotly.graph_objects as go

    money_mids = np.array([(float(c.split("-")[0]) + float(c.split("-")[1])) / 2
                           for c in grid.columns])
    atm_col = grid.columns[np.argmin(np.abs(money_mids - 1.0))]
    s = grid[atm_col] * 100
    fig = go.Figure(go.Scatter(x=list(grid.index), y=s.values,
                               mode="lines+markers", line=dict(color="#1F3864")))
    fig.update_layout(title=f"ATM term structure — {target}",
                      xaxis_title="Expiry", yaxis_title="ATM IV %",
                      yaxis=dict(range=[0, max(40, float(s.max()) * 1.15)]),
                      height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def smile_fig(grid: pd.DataFrame, target: str, row: str = "1-2m"):
    """2D slice: IV vs moneyness at one expiry bucket. The skew/smile view."""
    import plotly.graph_objects as go

    if row not in grid.index:
        row = grid.index[len(grid.index) // 2]
    money_mids = [(float(c.split("-")[0]) + float(c.split("-")[1])) / 2
                  for c in grid.columns]
    s = grid.loc[row] * 100
    fig = go.Figure(go.Scatter(x=money_mids, y=s.values,
                               mode="lines+markers", line=dict(color="#C62828")))
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"Skew/smile at {row} — {target}",
                      xaxis_title="Moneyness (strike/spot)", yaxis_title="IV %",
                      yaxis=dict(range=[0, max(40, float(s.max()) * 1.15)]),
                      height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        conn = sqlite3.connect(DB_PATH)
        target = conn.execute("SELECT MAX(snapshot_date) FROM chain_snapshots").fetchone()[0]
        conn.close()
    grid = build_grid(load_chain_day(target))
    print(f"\nIV surface grid for {target} (median IV per cell):\n")
    print((grid * 100).round(1).to_string())
    s = surface_stats(grid)
    print(f"\nTerm-structure slope (long minus short ATM): {s.get('term_slope', float('nan')):+.4f}"
          f"   ({'INVERTED - stress' if s.get('term_slope', 0) < 0 else 'normal upward'})")
    print(f"Left-wing steepness (~1m, deep puts over ATM): {s.get('left_wing', float('nan')):+.4f}")