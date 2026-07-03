"""Database setup and helpers for the options flow scanner."""

import sqlite3
from pathlib import Path

# Database lives in data/ regardless of where the script is run from
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "options.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chain_snapshots (
    snapshot_date TEXT NOT NULL,     -- 'YYYY-MM-DD' the day this data describes
    ticker        TEXT NOT NULL,     -- e.g. 'SPY'
    expiry        TEXT NOT NULL,     -- 'YYYY-MM-DD' option expiration date
    strike        REAL NOT NULL,
    option_type   TEXT NOT NULL,     -- 'C' or 'P'
    volume        INTEGER,           -- contracts traded that day
    open_interest INTEGER,           -- open contracts at day end
    implied_vol   REAL,              -- implied volatility (decimal, e.g. 0.18)
    last_price    REAL,              -- option's last traded price
    spot_price    REAL,              -- underlying price at snapshot time
    source        TEXT,              -- 'yfinance' or 'optionsdx' (data lineage)
    PRIMARY KEY (snapshot_date, ticker, expiry, strike, option_type)
);

CREATE INDEX IF NOT EXISTS idx_snap_date ON chain_snapshots (snapshot_date);
CREATE INDEX IF NOT EXISTS idx_ticker_date ON chain_snapshots (ticker, snapshot_date);
"""


def get_connection() -> sqlite3.Connection:
    """Open a connection, creating the database and schema if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def insert_rows(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert contract rows. Re-inserting the same contract/date updates it.

    Each row dict must have keys matching the column names.
    Returns number of rows written.
    """
    sql = """
    INSERT OR REPLACE INTO chain_snapshots
    (snapshot_date, ticker, expiry, strike, option_type,
     volume, open_interest, implied_vol, last_price, spot_price, source)
    VALUES (:snapshot_date, :ticker, :expiry, :strike, :option_type,
            :volume, :open_interest, :implied_vol, :last_price, :spot_price, :source)
    """
    with conn:  # auto-commit / rollback on error
        conn.executemany(sql, rows)
    return len(rows)


def row_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chain_snapshots").fetchone()[0]


if __name__ == "__main__":
    # Smoke test: create the DB, insert one fake row, count, clean up
    conn = get_connection()
    test_row = {
        "snapshot_date": "2026-01-01", "ticker": "TEST", "expiry": "2026-02-01",
        "strike": 100.0, "option_type": "C", "volume": 10, "open_interest": 5,
        "implied_vol": 0.2, "last_price": 1.5, "spot_price": 99.0, "source": "test",
    }
    insert_rows(conn, [test_row])
    print("Rows in database:", row_count(conn))
    conn.execute("DELETE FROM chain_snapshots WHERE ticker = 'TEST'")
    conn.commit()
    print("Test row cleaned up. Database ready at:", DB_PATH)
    conn.close()