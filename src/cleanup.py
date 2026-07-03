"""One-off: remove rows stamped with the wrong (non-trading) date."""
import sqlite3
from pathlib import Path

db = Path(__file__).resolve().parent.parent / "data" / "options.db"
conn = sqlite3.connect(db)
deleted = conn.execute(
    "DELETE FROM chain_snapshots WHERE snapshot_date = '2026-07-03'"
).rowcount
conn.commit()
print(f"Deleted {deleted} rows")
print("Remaining:", conn.execute(
    "SELECT snapshot_date, COUNT(*) FROM chain_snapshots GROUP BY snapshot_date"
).fetchall())
conn.close()