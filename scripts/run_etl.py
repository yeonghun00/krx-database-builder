#!/usr/bin/env python3
"""
Unified ETL Runner

Single entry point for all 4 ETL pipelines:
  1. Prices (daily_prices)
  2. Index constituents (index_constituents)
  3. Delisted stocks (delisted_stocks)
  4. Financials (financial_periods, financial_items_*)

Modes:
  update   - Auto-detect stale data and fetch only what's missing
  backfill - Load historical data for a date range

Usage:
  python3 scripts/run_etl.py update [--markets kospi,kosdaq] [--workers 4] [--skip prices index delisted financial]
  python3 scripts/run_etl.py backfill --start-date 20100101 --end-date 20251231 [--markets kospi,kosdaq] [--workers 4] [--skip prices index delisted financial]
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "etl"))

from config import load_config
from etl.clean_etl import CleanETLPipeline
from etl.krx_api import KRXAPI
from etl.index_constituents_etl import KRXIndexConstituentsDirect
from etl.delisted_stocks_etl import (
    download_delisted_stocks,
    create_database_table,
    insert_delisted_stocks_to_db,
)
from etl.financial_etl import FinancialDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_etl")

PIPELINES = ["prices", "index", "delisted", "financial"]

# Marker file for tracking processed financial ZIPs
FINANCIAL_MARKER = PROJECT_ROOT / "data" / "raw_financial" / ".processed_files"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trading_weekdays(start: datetime, end: datetime) -> List[str]:
    """Return list of YYYYMMDD strings for weekdays in [start, end]."""
    dates = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return dates


def _monthly_first_days(start: datetime, end: datetime) -> List[str]:
    """Return list of YYYY-MM-DD strings for 1st of each month in [start, end]."""
    dates = []
    cur = start.replace(day=1)
    if cur < start:
        # move to next month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return dates


def _next_month(dt: datetime) -> datetime:
    """Return 1st of next month."""
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1)
    return dt.replace(month=dt.month + 1, day=1)


def _get_processed_financial_files() -> Set[str]:
    """Read the set of already-processed financial ZIP filenames."""
    if FINANCIAL_MARKER.exists():
        return set(FINANCIAL_MARKER.read_text().strip().splitlines())
    return set()


def _save_processed_financial_files(files: Set[str]):
    """Persist the set of processed financial ZIP filenames."""
    FINANCIAL_MARKER.parent.mkdir(parents=True, exist_ok=True)
    FINANCIAL_MARKER.write_text("\n".join(sorted(files)) + "\n")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def print_status(db_path: str, raw_financial_dir: Path, skip: List[str]):
    """Print a status table showing each pipeline's current state."""
    print("\n" + "=" * 60)
    print("  ETL Pipeline Status")
    print("=" * 60)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Prices ---
    if "prices" not in skip:
        try:
            cursor.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily_prices")
            row = cursor.fetchone()
            min_d, max_d, n_dates = row if row else (None, None, 0)
            print(f"\n  Prices")
            print(f"    Date range : {min_d or 'N/A'} ~ {max_d or 'N/A'}")
            print(f"    Trading days: {n_dates or 0}")
            if max_d:
                gap = _trading_weekdays(
                    datetime.strptime(max_d, "%Y%m%d") + timedelta(days=1),
                    datetime.now(),
                )
                print(f"    Gap (est)  : {len(gap)} weekdays to fetch")
        except sqlite3.OperationalError:
            print("\n  Prices")
            print("    Table not found (will be created)")

    # --- Index constituents ---
    if "index" not in skip:
        try:
            cursor.execute("SELECT MAX(date), COUNT(DISTINCT date) FROM index_constituents")
            row = cursor.fetchone()
            max_d, n_months = row if row else (None, 0)
            print(f"\n  Index Constituents")
            print(f"    Latest date : {max_d or 'N/A'}")
            print(f"    Months stored: {n_months or 0}")
            if max_d:
                latest_dt = datetime.strptime(max_d, "%Y-%m-%d")
                months_gap = _monthly_first_days(_next_month(latest_dt), datetime.now())
                print(f"    Gap (est)   : {len(months_gap)} months to fetch")
        except sqlite3.OperationalError:
            print("\n  Index Constituents")
            print("    Table not found (will be created)")

    # --- Delisted stocks ---
    if "delisted" not in skip:
        try:
            cursor.execute("SELECT COUNT(*), MAX(downloaded_at) FROM delisted_stocks")
            row = cursor.fetchone()
            cnt, last_dl = row if row else (0, None)
            print(f"\n  Delisted Stocks")
            print(f"    Records     : {cnt or 0}")
            print(f"    Last updated: {last_dl or 'never'}")
        except sqlite3.OperationalError:
            print("\n  Delisted Stocks")
            print("    Table not found (will be created)")

    # --- Financials ---
    if "financial" not in skip:
        raw_dir = raw_financial_dir
        all_zips = sorted(raw_dir.glob("*.zip")) if raw_dir.exists() else []
        all_zips = [f for f in all_zips if "_CE_" not in f.name]
        processed = _get_processed_financial_files()
        new_zips = [f for f in all_zips if f.name not in processed]

        try:
            cursor.execute("SELECT COUNT(*) FROM financial_periods")
            periods = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            periods = 0

        print(f"\n  Financials")
        print(f"    Periods in DB : {periods}")
        print(f"    ZIP files     : {len(all_zips)} total, {len(new_zips)} new")

    conn.close()
    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def run_prices_update(db_path: str, config: dict, markets: List[str], workers: int):
    """Update prices: fetch from MAX(date)+1 to today, skip existing."""
    logger.info("--- Prices: update ---")
    pipeline = CleanETLPipeline(db_path)
    api = KRXAPI(config["api"]["auth_key"], config.get("api", {}))

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM daily_prices")
        row = cursor.fetchone()
        max_date = row[0] if row and row[0] else None
        conn.close()
    except sqlite3.OperationalError:
        max_date = None

    if not max_date:
        logger.warning("No existing price data found. Use 'backfill' mode to load historical data.")
        return {"days_processed": 0, "records": 0}

    start_dt = datetime.strptime(max_date, "%Y%m%d") + timedelta(days=1)
    end_dt = datetime.now()

    if start_dt > end_dt:
        logger.info("Prices already up to date.")
        return {"days_processed": 0, "records": 0}

    dates_to_fetch = _trading_weekdays(start_dt, end_dt)

    # Filter out dates that already exist
    existing = pipeline.get_existing_dates(dates_to_fetch[0], dates_to_fetch[-1]) if dates_to_fetch else set()
    dates_to_fetch = [d for d in dates_to_fetch if d not in existing]

    if not dates_to_fetch:
        logger.info("Prices already up to date (all weekdays present).")
        return {"days_processed": 0, "records": 0}

    logger.info(f"Fetching prices for {len(dates_to_fetch)} days ({dates_to_fetch[0]} ~ {dates_to_fetch[-1]})")

    days_ok = 0
    total_records = 0
    for date_str in dates_to_fetch:
        try:
            market_data = api.fetch_data_for_date_parallel(date_str, markets, is_backfill=False)
            raw_data = []
            for records in market_data.values():
                raw_data.extend(records)
            if raw_data:
                result = pipeline.process_data(raw_data)
                total_records += result["prices_processed"]
                days_ok += 1
                logger.info(f"  {date_str}: {result['prices_processed']} records")
            else:
                logger.info(f"  {date_str}: no data (holiday?)")
        except Exception as e:
            logger.error(f"  {date_str}: error - {e}")

    pipeline.close()
    return {"days_processed": days_ok, "records": total_records}


def run_prices_backfill(db_path: str, config: dict, markets: List[str], workers: int,
                        start_date: str, end_date: str):
    """Backfill prices for a date range, skipping existing."""
    logger.info("--- Prices: backfill ---")
    pipeline = CleanETLPipeline(db_path)
    api = KRXAPI(config["api"]["auth_key"], config.get("api", {}))

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    all_dates = _trading_weekdays(start_dt, end_dt)

    existing = pipeline.get_existing_dates(start_date, end_date)
    dates_to_fetch = [d for d in all_dates if d not in existing]

    if not dates_to_fetch:
        logger.info("All dates already present. Nothing to backfill.")
        return {"days_processed": 0, "records": 0}

    logger.info(f"Backfilling {len(dates_to_fetch)} / {len(all_dates)} days")

    days_ok = 0
    total_records = 0
    for date_str in dates_to_fetch:
        try:
            market_data = api.fetch_data_for_date_parallel(date_str, markets, is_backfill=True)
            raw_data = []
            for records in market_data.values():
                raw_data.extend(records)
            if raw_data:
                result = pipeline.process_data(raw_data)
                total_records += result["prices_processed"]
                days_ok += 1
                if days_ok % 20 == 0:
                    logger.info(f"  Progress: {days_ok}/{len(dates_to_fetch)} days")
            else:
                logger.info(f"  {date_str}: no data (holiday?)")
        except Exception as e:
            logger.error(f"  {date_str}: error - {e}")

    pipeline.close()
    return {"days_processed": days_ok, "records": total_records}


def run_index_update(config: dict, workers: int):
    """Update index constituents from MAX(date)+1 month to now."""
    logger.info("--- Index Constituents: update ---")
    processor = KRXIndexConstituentsDirect()
    processor.update(strategy="skip", max_workers=workers)
    return {"status": "done"}


def run_index_backfill(config: dict, workers: int, start_date: str, end_date: str):
    """Backfill index constituents monthly for a date range."""
    logger.info("--- Index Constituents: backfill ---")
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    iso_start = start_dt.strftime("%Y-%m-%d")
    processor = KRXIndexConstituentsDirect()
    processor.backfill(start_date=iso_start, max_workers=workers)
    return {"status": "done"}


def run_delisted(db_path: str):
    """Download and refresh delisted stocks (idempotent full refresh)."""
    logger.info("--- Delisted Stocks: refresh ---")
    df, name_d = download_delisted_stocks()
    if df is None:
        logger.error("Failed to download delisted stocks")
        return {"status": "error", "records": 0}

    create_database_table(db_path)
    insert_delisted_stocks_to_db(df, db_path)
    logger.info(f"Loaded {len(df)} delisted stocks")
    return {"status": "done", "records": len(df)}


def run_financial_update(db_path: str, raw_dir: Path):
    """Process only new (unprocessed) financial ZIP files."""
    logger.info("--- Financials: update ---")
    if not raw_dir.exists():
        logger.warning(f"Raw financial directory not found: {raw_dir}")
        return {"files_processed": 0}

    all_zips = sorted(raw_dir.glob("*.zip"))
    all_zips = [f for f in all_zips if "_CE_" not in f.name]

    processed = _get_processed_financial_files()
    new_zips = [f for f in all_zips if f.name not in processed]

    if not new_zips:
        logger.info("No new financial ZIP files to process.")
        return {"files_processed": 0}

    logger.info(f"Processing {len(new_zips)} new financial ZIP files")

    loader = FinancialDataLoader(str(db_path), str(raw_dir))
    loader.connect()
    loader.create_tables()

    files_ok = 0
    total_items = 0
    for zip_path in new_zips:
        try:
            periods, items = loader.process_file(zip_path)
            total_items += items
            files_ok += 1
            processed.add(zip_path.name)
            logger.info(f"  {zip_path.name}: {items} items")
        except Exception as e:
            logger.error(f"  {zip_path.name}: error - {e}")

    loader.close()

    _save_processed_financial_files(processed)
    return {"files_processed": files_ok, "total_items": total_items}


def run_financial_backfill(db_path: str, raw_dir: Path):
    """Process all financial ZIP files (ignoring marker)."""
    logger.info("--- Financials: backfill ---")
    if not raw_dir.exists():
        logger.warning(f"Raw financial directory not found: {raw_dir}")
        return {"files_processed": 0}

    loader = FinancialDataLoader(str(db_path), str(raw_dir))
    loader.connect()
    loader.create_tables()

    try:
        stats = loader.process_all()
        logger.info(f"Processed {stats['files_processed']} files, {stats['total_items']} items")
    finally:
        loader.close()

    # Mark all as processed
    all_zips = sorted(raw_dir.glob("*.zip"))
    all_zips = [f for f in all_zips if "_CE_" not in f.name]
    _save_processed_financial_files({f.name for f in all_zips})

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified ETL runner for KRX stock data pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipelines: prices, index, delisted, financial

Examples:
  python3 scripts/run_etl.py update
  python3 scripts/run_etl.py update --markets kospi,kosdaq --workers 4
  python3 scripts/run_etl.py update --skip index financial
  python3 scripts/run_etl.py backfill --start-date 20200101 --end-date 20201231
""",
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # -- update --
    p_update = sub.add_parser("update", help="Auto-detect gaps and fetch missing data")
    p_update.add_argument("--markets", default="kospi,kosdaq",
                          help="Comma-separated markets (default: kospi,kosdaq)")
    p_update.add_argument("--workers", type=int, default=4,
                          help="Parallel workers for index scraping (default: 4)")
    p_update.add_argument("--skip", nargs="*", default=[], choices=PIPELINES,
                          help="Pipelines to skip")

    # -- backfill --
    p_backfill = sub.add_parser("backfill", help="Load historical data for a date range")
    p_backfill.add_argument("--start-date", required=True,
                            help="Start date YYYYMMDD")
    p_backfill.add_argument("--end-date", required=True,
                            help="End date YYYYMMDD")
    p_backfill.add_argument("--markets", default="kospi,kosdaq",
                            help="Comma-separated markets (default: kospi,kosdaq)")
    p_backfill.add_argument("--workers", type=int, default=4,
                            help="Parallel workers for index scraping (default: 4)")
    p_backfill.add_argument("--skip", nargs="*", default=[], choices=PIPELINES,
                            help="Pipelines to skip")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    markets = [m.strip() for m in args.markets.split(",")]
    skip = set(args.skip)

    config_path = PROJECT_ROOT / "config.json"
    config = load_config(str(config_path))
    db_path = config.get("database", {}).get("path", "krx_stock_data.db")
    raw_financial_dir = PROJECT_ROOT / "data" / "raw_financial"

    # Print status before running
    print_status(db_path, raw_financial_dir, skip)

    t0 = time.time()
    results = {}

    if args.mode == "update":
        if "prices" not in skip:
            results["prices"] = run_prices_update(db_path, config, markets, args.workers)
        if "index" not in skip:
            results["index"] = run_index_update(config, args.workers)
        if "delisted" not in skip:
            results["delisted"] = run_delisted(db_path)
        if "financial" not in skip:
            results["financial"] = run_financial_update(db_path, raw_financial_dir)

    elif args.mode == "backfill":
        if "prices" not in skip:
            results["prices"] = run_prices_backfill(
                db_path, config, markets, args.workers,
                args.start_date, args.end_date,
            )
        if "index" not in skip:
            results["index"] = run_index_backfill(
                config, args.workers, args.start_date, args.end_date,
            )
        if "delisted" not in skip:
            results["delisted"] = run_delisted(db_path)
        if "financial" not in skip:
            results["financial"] = run_financial_backfill(db_path, raw_financial_dir)

    elapsed = time.time() - t0

    # Print summary
    print("\n" + "=" * 60)
    print("  ETL Results Summary")
    print("=" * 60)
    for name, res in results.items():
        print(f"\n  {name}:")
        for k, v in res.items():
            print(f"    {k}: {v}")
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
