"""Daily options chain collector.

Pulls the full option chain for each ticker from yfinance and stores
one row per contract in the chain_snapshots table.

Run after market close:  py src\\collector.py
"""

from datetime import datetime, date

import pandas as pd
import yfinance as yf

from db import get_connection, insert_rows, row_count

TICKERS = ["SPY"]  # add "QQQ" etc. here later


def fetch_chain_rows(ticker: str) -> list[dict]:
    """Fetch all expiries for one ticker, return normalized row dicts."""
    tk = yf.Ticker(ticker)

    # Spot price — needed later for moneyness calculations
    hist = tk.history(period="1d")
    if hist.empty:
        raise RuntimeError(f"No price history returned for {ticker}")
    spot = float(hist["Close"].iloc[-1])

    snapshot_date = hist.index[-1].date().isoformat() 
    expiries = tk.options  # tuple of 'YYYY-MM-DD' strings
    if not expiries:
        raise RuntimeError(f"No option expiries returned for {ticker}")

    rows: list[dict] = []
    for expiry in expiries:
        try:
            chain = tk.option_chain(expiry)
        except Exception as exc:  # one bad expiry shouldn't kill the run
            print(f"  WARN: failed to fetch {ticker} {expiry}: {exc}")
            continue

        for opt_type, frame in (("C", chain.calls), ("P", chain.puts)):
            for rec in frame.itertuples(index=False):
                rows.append({
                    "snapshot_date": snapshot_date,
                    "ticker": ticker,
                    "expiry": expiry,
                    "strike": float(rec.strike),
                    "option_type": opt_type,
                    "volume": int(rec.volume) if pd.notna(rec.volume) else 0,
                    "open_interest": int(rec.openInterest) if pd.notna(rec.openInterest) else 0,
                    "implied_vol": float(rec.impliedVolatility) if pd.notna(rec.impliedVolatility) else None,
                    "last_price": float(rec.lastPrice) if pd.notna(rec.lastPrice) else None,
                    "spot_price": spot,
                    "source": "yfinance",
                })
    return rows


def main() -> None:
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{started}] Collector starting for: {', '.join(TICKERS)}")

    conn = get_connection()
    try:
        for ticker in TICKERS:
            print(f"Fetching {ticker} option chain...")
            rows = fetch_chain_rows(ticker)
            written = insert_rows(conn, rows)
            print(f"  {ticker}: wrote {written} contract rows")
        print(f"Done. Total rows in database: {row_count(conn)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()