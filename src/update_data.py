"""Daily incremental data updater for macro dashboard.

Reads data_status.json to determine last update, fetches new data from
FRED and yfinance, upserts into the database, and writes update_result.json
for GitHub Actions integration.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

CT = timezone(timedelta(hours=-6))

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from config import (
    DATA_DIR,
    DB_PATH,
    FRED_INDICATORS,
    MARKET_INDICATORS,
    STATUS_PATH,
)

load_dotenv(DATA_DIR.parent / ".env")
FRED_KEY = os.environ.get("FRED_API_KEY", "")


def get_last_update() -> datetime:
    """Read last update timestamp from status file."""
    if STATUS_PATH.exists():
        status = json.loads(STATUS_PATH.read_text())
        return datetime.fromisoformat(status["last_update"])
    # Default: fetch last 30 days if no status
    return datetime.now(CT) - timedelta(days=30)


def update_fred(conn: sqlite3.Connection, since: datetime) -> dict:
    """Fetch new FRED data since last update."""
    if not FRED_KEY:
        print("[WARN] FRED_API_KEY not set — skipping FRED update.")
        return {"status": "skipped", "rows": 0}

    fred = Fred(api_key=FRED_KEY)
    # Fetch a bit before 'since' to catch revisions
    start = since - timedelta(days=7)
    total = 0
    errors = []

    for category, indicators in FRED_INDICATORS.items():
        for ind in indicators:
            sid = ind["series_id"]
            try:
                series = fred.get_series(sid, observation_start=start)
                if series is None or series.empty:
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

                if rows:
                    conn.execute(
                        "UPDATE indicator_metadata SET last_updated = ? WHERE series_id = ?",
                        (rows[-1][1], sid),
                    )

                total += len(rows)
                print(f"  [OK] {sid:25s} — {len(rows)} new/updated rows")

            except Exception as e:
                errors.append(f"{sid}: {e}")
                print(f"  [ERR] {sid}: {e}")

    conn.commit()
    return {"status": "ok", "rows": total, "errors": errors}


def update_markets(conn: sqlite3.Connection, since: datetime) -> dict:
    """Fetch new market data from yfinance since last update."""
    start = (since - timedelta(days=3)).strftime("%Y-%m-%d")
    total = 0
    errors = []

    for mkt in MARKET_INDICATORS:
        ticker = mkt["ticker"]
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue

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

            if rows:
                conn.execute(
                    "UPDATE indicator_metadata SET last_updated = ? WHERE series_id = ?",
                    (rows[-1][1], ticker),
                )

            total += len(rows)
            print(f"  [OK] {ticker:15s} — {len(rows)} new/updated rows")

        except Exception as e:
            errors.append(f"{ticker}: {e}")
            print(f"  [ERR] {ticker}: {e}")

    conn.commit()
    return {"status": "ok", "rows": total, "errors": errors}


def main() -> None:
    if not DB_PATH.exists():
        print(f"[ERR] Database not found at {DB_PATH}")
        print("Run create_database.py first.")
        return

    last = get_last_update()
    print(f"\n{'='*60}")
    print(f"  Macro Dashboard — Daily Update")
    print(f"  Last update: {last.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        print("--- Updating FRED data ---")
        fred_result = update_fred(conn, last)

        print("\n--- Updating Market data ---")
        market_result = update_markets(conn, last)

        # Write status
        now = datetime.now(CT)
        status = {
            "last_update": now.isoformat(),
            "fred_rows_loaded": fred_result["rows"],
            "market_rows_loaded": market_result["rows"],
        }
        if STATUS_PATH.exists():
            existing = json.loads(STATUS_PATH.read_text())
            existing.update(status)
            status = existing
        else:
            status["last_full_load"] = now.isoformat()

        STATUS_PATH.write_text(json.dumps(status, indent=2))

        # Write update_result.json for GitHub Actions
        result = {
            "timestamp": now.isoformat(),
            "fred": fred_result,
            "markets": market_result,
            "total_new_rows": fred_result["rows"] + market_result["rows"],
            "has_new_data": (fred_result["rows"] + market_result["rows"]) > 0,
        }
        result_path = DATA_DIR / "update_result.json"
        result_path.write_text(json.dumps(result, indent=2))

        print(f"\n{'='*60}")
        print(f"  Done! FRED: {fred_result['rows']} rows | Market: {market_result['rows']} rows")
        print(f"{'='*60}\n")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
