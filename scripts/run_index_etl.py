#!/usr/bin/env python3
"""
Main runner for Market Index ETL Pipeline

Usage:
    # Check status
    python run_index_etl.py --status

    # Backfill historical data
    python run_index_etl.py --backfill --start-date 20200101 --end-date 20241231

    # Daily update
    python run_index_etl.py --daily-update

    # Force reprocess
    python run_index_etl.py --backfill --start-date 20240101 --end-date 20240131 --force

    # Fetch specific index types only
    python run_index_etl.py --backfill --start-date 20240101 --end-date 20240131 --index-types kospi_index,kosdaq_index
"""

import argparse
import logging
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.krx_api import KRXAPI
from etl.index_etl import IndexETLPipeline
from config import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_trading_dates(start_date: str, end_date: str) -> List[str]:
    """
    Generate list of trading dates (weekdays only).

    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format

    Returns:
        List of date strings in YYYYMMDD format
    """
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')

    dates = []
    current = start
    while current <= end:
        # Skip weekends
        if current.weekday() < 5:
            dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    return dates


def save_progress(db_path: str, progress_data: dict):
    """Save progress to JSON file for resume capability."""
    progress_file = f"{db_path}.index_progress.json"
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        logger.info(f"Progress saved to {progress_file}")
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")


def load_progress(db_path: str) -> dict:
    """Load progress from JSON file."""
    progress_file = f"{db_path}.index_progress.json"
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load progress: {e}")
    return {}


def cleanup_progress(db_path: str):
    """Remove progress file after successful completion."""
    progress_file = f"{db_path}.index_progress.json"
    if os.path.exists(progress_file):
        os.remove(progress_file)
        logger.info("Progress file cleaned up")


def backfill_index_data(
    api: KRXAPI,
    pipeline: IndexETLPipeline,
    start_date: str,
    end_date: str,
    index_types: List[str] = None,
    force: bool = False,
    db_path: str = 'krx_stock_data.db'
):
    """
    Backfill historical index data.

    Args:
        api: KRXAPI instance
        pipeline: IndexETLPipeline instance
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        index_types: List of index types to fetch (None = all)
        force: Force reprocess existing dates
        db_path: Database path for progress tracking
    """
    # Default to all index types
    if index_types is None:
        index_types = ['kospi_index', 'kosdaq_index', 'bond_index', 'govt_bond', 'derivatives']

    dates = generate_trading_dates(start_date, end_date)
    logger.info(f"Backfilling {len(dates)} trading dates from {start_date} to {end_date}")
    logger.info(f"Index types: {', '.join(index_types)}")

    # Get existing dates if not forcing
    existing_dates: Set[str] = set()
    if not force:
        existing_dates = pipeline.get_existing_dates(start_date, end_date, 'market_indices')
        dates_to_process = [d for d in dates if d not in existing_dates]
        logger.info(f"Skipping {len(existing_dates)} dates with existing data")
    else:
        dates_to_process = dates
        logger.info("Force mode: processing all dates")

    if not dates_to_process:
        logger.info("No dates to process. Use --force to reprocess existing data.")
        return

    logger.info(f"Processing {len(dates_to_process)} dates...")

    processed_count = 0
    total_records = {t: 0 for t in index_types}

    for i, date in enumerate(dates_to_process):
        try:
            logger.info(f"[{i+1}/{len(dates_to_process)}] Processing {date}")

            # Fetch all index data for this date
            index_data = api.fetch_index_data_parallel(date, index_types, is_backfill=True)

            # Process the data
            results = pipeline.process_all_index_data(index_data)

            # Accumulate counts
            for idx_type, count in results.items():
                total_records[idx_type] = total_records.get(idx_type, 0) + count

            record_summary = ', '.join(f"{k}: {v}" for k, v in results.items() if v > 0)
            logger.info(f"  Processed: {record_summary}")

            processed_count += 1

            # Save progress every 10 dates
            if processed_count % 10 == 0:
                progress_data = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'last_processed_date': date,
                    'processed_count': processed_count,
                    'total_records': total_records,
                    'index_types': index_types
                }
                save_progress(db_path, progress_data)

        except Exception as e:
            logger.error(f"Error processing {date}: {e}")
            # Save progress on error
            progress_data = {
                'start_date': start_date,
                'end_date': end_date,
                'last_processed_date': date,
                'processed_count': processed_count,
                'total_records': total_records,
                'index_types': index_types,
                'error': str(e)
            }
            save_progress(db_path, progress_data)
            continue

    # Summary
    logger.info("=" * 60)
    logger.info(f"Backfill completed. Processed {processed_count} dates.")
    for idx_type, count in total_records.items():
        logger.info(f"  {idx_type}: {count:,} records")

    # Clean up progress file if completed successfully
    if processed_count == len(dates_to_process):
        cleanup_progress(db_path)


def daily_update(
    api: KRXAPI,
    pipeline: IndexETLPipeline,
    date: str = None,
    index_types: List[str] = None,
    force: bool = False
):
    """
    Run daily update for index data.

    Args:
        api: KRXAPI instance
        pipeline: IndexETLPipeline instance
        date: Specific date (default: yesterday)
        index_types: List of index types to fetch (None = all)
        force: Force reprocess even if data exists
    """
    # Default to yesterday
    if date is None:
        yesterday = datetime.now() - timedelta(days=1)
        date = yesterday.strftime('%Y%m%d')

    # Check if weekday
    dt = datetime.strptime(date, '%Y%m%d')
    if dt.weekday() >= 5:
        logger.info(f"{date} is a weekend, skipping")
        return

    # Default to all index types
    if index_types is None:
        index_types = ['kospi_index', 'kosdaq_index', 'bond_index', 'govt_bond', 'derivatives']

    # Check if data already exists
    if not force and pipeline.check_date_exists(date, 'market_indices'):
        logger.info(f"Data already exists for {date}. Use --force to reprocess.")
        return

    logger.info(f"Running daily update for {date}")
    logger.info(f"Index types: {', '.join(index_types)}")

    try:
        # Fetch all index data
        index_data = api.fetch_index_data_parallel(date, index_types, is_backfill=False)

        # Process the data
        results = pipeline.process_all_index_data(index_data)

        # Summary
        record_summary = ', '.join(f"{k}: {v}" for k, v in results.items() if v > 0)
        logger.info(f"Processed: {record_summary}")

    except Exception as e:
        logger.error(f"Error processing {date}: {e}")
        raise


def show_status(pipeline: IndexETLPipeline):
    """Display status of all index tables."""
    stats = pipeline.get_stats()

    print("\n" + "=" * 60)
    print("MARKET INDEX DATA STATUS")
    print("=" * 60)

    for table, info in stats.items():
        print(f"\n{table}:")
        print(f"  Records:      {info['count']:,}")
        print(f"  Unique dates: {info['unique_dates']:,}")
        if info['min_date'] and info['max_date']:
            print(f"  Date range:   {info['min_date']} to {info['max_date']}")
        else:
            print(f"  Date range:   No data")

    # Show sample index names
    print("\n" + "-" * 60)
    print("SAMPLE INDEX NAMES")
    print("-" * 60)

    kospi_indices = pipeline.get_market_index_names('KOSPI')
    if kospi_indices:
        print(f"\nKOSPI Indices ({len(kospi_indices)} total):")
        for name in kospi_indices[:5]:
            print(f"  - {name}")
        if len(kospi_indices) > 5:
            print(f"  ... and {len(kospi_indices) - 5} more")

    kosdaq_indices = pipeline.get_market_index_names('KOSDAQ')
    if kosdaq_indices:
        print(f"\nKOSDAQ Indices ({len(kosdaq_indices)} total):")
        for name in kosdaq_indices[:5]:
            print(f"  - {name}")
        if len(kosdaq_indices) > 5:
            print(f"  ... and {len(kosdaq_indices) - 5} more")

    bond_indices = pipeline.get_bond_index_names()
    if bond_indices:
        print(f"\nBond Indices ({len(bond_indices)} total):")
        for name in bond_indices[:5]:
            print(f"  - {name}")

    deriv_indices = pipeline.get_derivative_index_names()
    if deriv_indices:
        print(f"\nDerivative Indices ({len(deriv_indices)} total):")
        for name in deriv_indices[:5]:
            print(f"  - {name}")
        if len(deriv_indices) > 5:
            print(f"  ... and {len(deriv_indices) - 5} more")

    print("\n" + "=" * 60)


def validate_data(pipeline: IndexETLPipeline):
    """Run data validation checks."""
    results = pipeline.validate_data()

    print("\n" + "=" * 60)
    print("DATA VALIDATION RESULTS")
    print("=" * 60)

    for check, value in results.items():
        if check == 'validation_passed':
            continue
        status = "OK" if value == 0 else f"WARNING: {value}"
        print(f"  {check}: {status}")

    print("-" * 60)
    if results['validation_passed']:
        print("Overall: PASSED")
    else:
        print("Overall: WARNINGS FOUND")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Market Index ETL Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show status
  python run_index_etl.py --status

  # Backfill 2020-2024
  python run_index_etl.py --backfill --start-date 20200101 --end-date 20241231

  # Daily update
  python run_index_etl.py --daily-update

  # Specific index types only
  python run_index_etl.py --backfill --start-date 20240101 --end-date 20240131 \\
      --index-types kospi_index,kosdaq_index

  # Force reprocess
  python run_index_etl.py --backfill --start-date 20240101 --end-date 20240131 --force

Available index types:
  kospi_index  - KOSPI market indices
  kosdaq_index - KOSDAQ market indices
  bond_index   - Bond indices (KRX, KTB, etc.)
  govt_bond    - Government bond market data
  derivatives  - Derivatives indices (futures, options, VIX)
"""
    )

    parser.add_argument('--db-path', default='krx_stock_data.db',
                        help='Database path (default: krx_stock_data.db)')

    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--status', action='store_true',
                            help='Show status of index data')
    mode_group.add_argument('--backfill', action='store_true',
                            help='Run historical backfill')
    mode_group.add_argument('--daily-update', action='store_true',
                            help='Run daily update')
    mode_group.add_argument('--validate', action='store_true',
                            help='Run data validation')
    mode_group.add_argument('--optimize', action='store_true',
                            help='Optimize database')

    # Backfill arguments
    parser.add_argument('--start-date', type=str,
                        help='Start date for backfill (YYYYMMDD)')
    parser.add_argument('--end-date', type=str,
                        help='End date for backfill (YYYYMMDD)')
    parser.add_argument('--date', type=str,
                        help='Specific date for daily update (YYYYMMDD)')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess existing dates')
    parser.add_argument('--index-types', type=str,
                        help='Comma-separated list of index types to fetch')

    args = parser.parse_args()

    # Parse index types if specified
    index_types = None
    if args.index_types:
        index_types = [t.strip() for t in args.index_types.split(',')]
        valid_types = ['kospi_index', 'kosdaq_index', 'bond_index', 'govt_bond', 'derivatives']
        invalid = [t for t in index_types if t not in valid_types]
        if invalid:
            parser.error(f"Invalid index types: {invalid}. Valid types: {valid_types}")

    # Initialize pipeline
    with IndexETLPipeline(args.db_path) as pipeline:
        pipeline.init_tables()

        if args.status:
            show_status(pipeline)
            return

        if args.validate:
            validate_data(pipeline)
            return

        if args.optimize:
            print("Optimizing database...")
            pipeline.optimize_database()
            print("Done.")
            return

        # Load config and initialize API for backfill/update operations
        if args.backfill or args.daily_update:
            try:
                config = load_config()
                api = KRXAPI(config['api']['auth_key'], config.get('api', {}))
            except FileNotFoundError:
                parser.error("Configuration file not found. Create config.json with API key.")
            except KeyError:
                parser.error("API auth_key not found in config.json")

        if args.backfill:
            if not args.start_date or not args.end_date:
                parser.error("--backfill requires --start-date and --end-date")

            # Validate date format
            try:
                datetime.strptime(args.start_date, '%Y%m%d')
                datetime.strptime(args.end_date, '%Y%m%d')
            except ValueError:
                parser.error("Invalid date format. Use YYYYMMDD.")

            backfill_index_data(
                api=api,
                pipeline=pipeline,
                start_date=args.start_date,
                end_date=args.end_date,
                index_types=index_types,
                force=args.force,
                db_path=args.db_path
            )

        elif args.daily_update:
            daily_update(
                api=api,
                pipeline=pipeline,
                date=args.date,
                index_types=index_types,
                force=args.force
            )

        else:
            parser.print_help()


if __name__ == '__main__':
    main()
