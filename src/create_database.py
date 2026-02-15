"""One-time database creation and initial historical data load (5 years)."""

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

CT = timezone(timedelta(hours=-6))

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from config import (
    DATA_DIR,
    DB_PATH,
    FRED_API_KEY,
    FRED_INDICATORS,
    HISTORY_YEARS,
    MARKET_INDICATORS,
    STATUS_PATH,
)

load_dotenv(DATA_DIR.parent / ".env")

# Re-read after dotenv in case it was loaded from .env
import os

FRED_KEY = os.environ.get("FRED_API_KEY", "") or FRED_API_KEY


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes."""
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS fred_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_id TEXT NOT NULL,
            date DATE NOT NULL,
            value REAL,
            category TEXT NOT NULL,
            indicator_name TEXT NOT NULL,
            UNIQUE(series_id, date)
        );

        CREATE INDEX IF NOT EXISTS idx_fred_series ON fred_indicators(series_id);
        CREATE INDEX IF NOT EXISTS idx_fred_date ON fred_indicators(date);
        CREATE INDEX IF NOT EXISTS idx_fred_category ON fred_indicators(category);

        CREATE TABLE IF NOT EXISTS market_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            indicator_name TEXT NOT NULL,
            UNIQUE(ticker, date)
        );

        CREATE INDEX IF NOT EXISTS idx_market_ticker ON market_indicators(ticker);
        CREATE INDEX IF NOT EXISTS idx_market_date ON market_indicators(date);

        CREATE TABLE IF NOT EXISTS indicator_metadata (
            series_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            source TEXT NOT NULL,
            frequency TEXT,
            unit TEXT,
            description TEXT,
            last_updated DATE
        );
        """
    )
    conn.commit()
    print("[OK] Schema created.")


def populate_metadata(conn: sqlite3.Connection) -> None:
    """Insert metadata for every indicator."""
    cur = conn.cursor()
    rows = []

    for category, indicators in FRED_INDICATORS.items():
        for ind in indicators:
            rows.append(
                (
                    ind["series_id"],
                    ind["name"],
                    category,
                    "fred",
                    ind["freq"],
                    ind["unit"],
                    ind.get("description", ""),
                    None,
                )
            )

    for mkt in MARKET_INDICATORS:
        rows.append(
            (
                mkt["ticker"],
                mkt["name"],
                mkt["category"],
                "yfinance",
                "daily",
                "USD",
                "",
                None,
            )
        )

    cur.executemany(
        """INSERT OR REPLACE INTO indicator_metadata
           (series_id, name, category, source, frequency, unit, description, last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    print(f"[OK] Metadata inserted for {len(rows)} indicators.")


def load_fred_data(conn: sqlite3.Connection) -> int:
    """Fetch historical data from FRED and insert into fred_indicators."""
    if not FRED_KEY:
        print("[WARN] FRED_API_KEY not set — skipping FRED data load.")
        return 0

    fred = Fred(api_key=FRED_KEY)
    start = datetime.now(CT) - timedelta(days=365 * HISTORY_YEARS)
    total = 0

    for category, indicators in FRED_INDICATORS.items():
        for ind in indicators:
            sid = ind["series_id"]
            try:
                series = fred.get_series(sid, observation_start=start)
                if series is None or series.empty:
                    print(f"  [WARN] No data for {sid}")
                    continue

                rows = []
                for date, value in series.items():
                    if pd.notna(value):
                        rows.append(
                            (sid, date.strftime("%Y-%m-%d"), float(value), category, ind["name"])
                        )

                conn.executemany(
                    """INSERT OR REPLACE INTO fred_indicators
                       (series_id, date, value, category, indicator_name)
                       VALUES (?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.commit()

                # Update metadata last_updated
                if rows:
                    conn.execute(
                        "UPDATE indicator_metadata SET last_updated = ? WHERE series_id = ?",
                        (rows[-1][1], sid),
                    )
                    conn.commit()

                total += len(rows)
                print(f"  [OK] {sid:25s} ({ind['name']:35s}) — {len(rows):>6,} rows")

            except Exception as e:
                print(f"  [ERR] {sid}: {e}")

    return total


def load_market_data(conn: sqlite3.Connection) -> int:
    """Fetch historical market data from yfinance and insert into market_indicators."""
    start = (datetime.now(CT) - timedelta(days=365 * HISTORY_YEARS)).strftime("%Y-%m-%d")
    total = 0

    for mkt in MARKET_INDICATORS:
        ticker = mkt["ticker"]
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if df is None or df.empty:
                print(f"  [WARN] No data for {ticker}")
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            rows = []
            for date, row in df.iterrows():
                rows.append(
                    (
                        ticker,
                        date.strftime("%Y-%m-%d"),
                        float(row.get("Open", 0) or 0),
                        float(row.get("High", 0) or 0),
                        float(row.get("Low", 0) or 0),
                        float(row.get("Close", 0) or 0),
                        int(row.get("Volume", 0) or 0),
                        mkt["name"],
                    )
                )

            conn.executemany(
                """INSERT OR REPLACE INTO market_indicators
                   (ticker, date, open, high, low, close, volume, indicator_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()

            # Update metadata
            if rows:
                conn.execute(
                    "UPDATE indicator_metadata SET last_updated = ? WHERE series_id = ?",
                    (rows[-1][1], ticker),
                )
                conn.commit()

            total += len(rows)
            print(f"  [OK] {ticker:15s} ({mkt['name']:25s}) — {len(rows):>6,} rows")

        except Exception as e:
            print(f"  [ERR] {ticker}: {e}")

    return total


def write_status(fred_rows: int, market_rows: int) -> None:
    """Write data_status.json with load summary."""
    status = {
        "last_full_load": datetime.now(CT).isoformat(),
        "last_update": datetime.now(CT).isoformat(),
        "fred_rows_loaded": fred_rows,
        "market_rows_loaded": market_rows,
        "history_years": HISTORY_YEARS,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))
    print(f"[OK] Status written to {STATUS_PATH}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        print(f"[INFO] Existing database found at {DB_PATH}")
        resp = input("Delete and recreate? (y/N): ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)
        DB_PATH.unlink()

    print(f"\n{'='*60}")
    print("  Macro Dashboard — Initial Database Setup")
    print(f"{'='*60}\n")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        create_schema(conn)
        populate_metadata(conn)

        print("\n--- Loading FRED data (this may take a few minutes) ---")
        fred_rows = load_fred_data(conn)

        print("\n--- Loading Market data from Yahoo Finance ---")
        market_rows = load_market_data(conn)

        write_status(fred_rows, market_rows)

        print(f"\n{'='*60}")
        print(f"  Done! FRED: {fred_rows:,} rows | Market: {market_rows:,} rows")
        print(f"  Database: {DB_PATH}")
        print(f"{'='*60}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
