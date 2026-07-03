"""Data quality report: what does our collected chain actually look like?"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "options.db"


def load(date: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM chain_snapshots"
    if date:
        q += f" WHERE snapshot_date = '{date}'"
    df = pd.read_sql(q, conn, parse_dates=["snapshot_date", "expiry"])
    conn.close()
    return df


def report(df: pd.DataFrame) -> None:
    d = df["snapshot_date"].dt.date.unique()
    print(f"=== Data quality report: {list(d)} ===\n")

    spot = df["spot_price"].iloc[0]
    print(f"Spot price: {spot:.2f}")
    print(f"Total contracts: {len(df):,}")
    print(f"  Calls: {len(df[df.option_type == 'C']):,}   Puts: {len(df[df.option_type == 'P']):,}")
    print(f"Expiries: {df['expiry'].nunique()}  "
          f"(nearest {df['expiry'].min().date()}, furthest {df['expiry'].max().date()})")
    print(f"Strike range: {df['strike'].min():.0f} – {df['strike'].max():.0f}\n")

    # Derived features the signals will use
    df = df.copy()
    df["dte"] = (df["expiry"] - df["snapshot_date"]).dt.days
    df["moneyness"] = df["strike"] / df["spot_price"]

    total_vol = df["volume"].sum()
    call_vol = df.loc[df.option_type == "C", "volume"].sum()
    put_vol = df.loc[df.option_type == "P", "volume"].sum()
    print(f"Total volume: {total_vol:,}")
    print(f"  Call volume: {call_vol:,}   Put volume: {put_vol:,}")
    print(f"  Put/Call ratio: {put_vol / call_vol:.3f}\n")

    zero_vol = (df["volume"] == 0).mean()
    print(f"Contracts with ZERO volume: {zero_vol:.1%}  <- expect a large number, most strikes never trade")
    print(f"Contracts with volume > OI: {(df['volume'] > df['open_interest']).sum():,}"
          "  <- 'fresh positioning' candidates\n")

    print("Volume concentration by days-to-expiry:")
    dte_buckets = pd.cut(df["dte"], bins=[-1, 1, 7, 30, 90, 10_000],
                         labels=["0-1d", "2-7d", "8-30d", "31-90d", ">90d"])
    vol_by_dte = df.groupby(dte_buckets, observed=True)["volume"].sum()
    for bucket, v in vol_by_dte.items():
        print(f"  {bucket:>6}: {v:>12,}  ({v / total_vol:.1%})")

    print("\nTop 10 contracts by volume today:")
    top = df.nlargest(10, "volume")[
        ["option_type", "strike", "expiry", "dte", "volume", "open_interest", "implied_vol"]
    ]
    top["expiry"] = top["expiry"].dt.date
    print(top.to_string(index=False))

    print("\nSanity checks:")
    print(f"  Missing implied_vol: {df['implied_vol'].isna().mean():.1%}")
    print(f"  Missing last_price:  {df['last_price'].isna().mean():.1%}")
    print(f"  Negative/zero strikes: {(df['strike'] <= 0).sum()}")
    print(f"  Expiries in the past:  {(df['dte'] < 0).sum()}")


if __name__ == "__main__":
    report(load())