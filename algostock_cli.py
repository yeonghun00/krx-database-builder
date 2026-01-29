#!/usr/bin/env python3
"""
AlgoStock CLI - Unified command-line interface for KRX stock data management and analysis.

Combines ETL operations and stock screening into a single tool with subcommands:
    algostock etl backfill    - Run historical data backfill
    algostock etl update      - Run daily update or catch-up
    algostock etl status      - Show database status
    algostock etl validate    - Validate data integrity
    algostock etl cleanup     - Clean up old data

    algostock analyze price   - Screen by price increase
    algostock analyze value   - Screen by trading value
    algostock analyze combined - Combined screening

    algostock quick           - Quick screening for current year
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Set up logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def validate_date(date_str: str) -> str:
    """Validate date format (YYYYMMDD)."""
    try:
        datetime.strptime(date_str, '%Y%m%d')
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Expected YYYYMMDD")


def validate_percentile(percentile: str) -> int:
    """Validate percentile value (1-100)."""
    try:
        value = int(percentile)
        if 1 <= value <= 100:
            return value
        raise argparse.ArgumentTypeError(f"Percentile must be between 1 and 100, got {value}")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid percentile: {percentile}. Must be an integer")


def validate_markets(markets_str: str) -> list:
    """Validate markets list."""
    valid_markets = ['kospi', 'kosdaq', 'kodex']
    markets = [m.strip().lower() for m in markets_str.split(',')]

    for market in markets:
        if market not in valid_markets:
            raise argparse.ArgumentTypeError(
                f"Invalid market: {market}. Valid markets: {', '.join(valid_markets)}"
            )
    return markets


# =============================================================================
# ETL Commands
# =============================================================================

def cmd_etl_status(args):
    """Show ETL status."""
    from clean_etl import CleanETLPipeline

    pipeline = CleanETLPipeline(args.db_path)
    status = pipeline.get_status()

    print("\n=== Database Status ===")
    print(f"Stocks: {status['stocks']:,}")
    print(f"Daily Prices: {status['daily_prices']:,}")
    if status['date_range'][0] and status['date_range'][1]:
        print(f"Date Range: {status['date_range'][0]} to {status['date_range'][1]}")
    else:
        print("Date Range: No data")
    print(f"Total Records: {status['total_records']:,}")


def cmd_etl_validate(args):
    """Validate data integrity."""
    from clean_etl import CleanETLPipeline

    pipeline = CleanETLPipeline(args.db_path)
    validation = pipeline.validate_data()

    print("\n=== Validation Results ===")
    print(f"Orphaned Prices: {validation['orphaned_prices']}")
    print(f"Duplicate Prices: {validation['duplicate_prices']}")
    print(f"Validation Passed: {'Yes' if validation['validation_passed'] else 'No'}")

    if not validation['validation_passed']:
        sys.exit(1)


def cmd_etl_cleanup(args):
    """Clean up old data."""
    from clean_etl import CleanETLPipeline

    logger = logging.getLogger(__name__)
    logger.info(f"Cleaning up data older than {args.days} days...")

    pipeline = CleanETLPipeline(args.db_path)
    pipeline.cleanup_old_data(days_to_keep=args.days)

    logger.info("Cleanup completed!")


def cmd_etl_verify(args):
    """Verify backfill completeness - check for gaps in data."""
    import sqlite3

    conn = sqlite3.connect(args.db_path)
    cursor = conn.cursor()

    print("\n" + "=" * 70)
    print("BACKFILL VERIFICATION REPORT")
    print("=" * 70)

    # Get date range
    cursor.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily_prices")
    min_date, max_date, total_days = cursor.fetchone()

    print(f"\nDate Range: {min_date} to {max_date}")
    print(f"Total Trading Days in DB: {total_days:,}")

    # Calculate expected trading days (approximate: ~250 per year)
    from datetime import datetime
    start_dt = datetime.strptime(min_date, '%Y%m%d')
    end_dt = datetime.strptime(max_date, '%Y%m%d')
    years = (end_dt - start_dt).days / 365.25
    expected_days = int(years * 250)
    print(f"Expected Trading Days (~250/year): ~{expected_days:,}")

    # Check market coverage by year
    print("\n" + "-" * 70)
    print("MARKET COVERAGE BY YEAR")
    print("-" * 70)
    cursor.execute("""
        SELECT SUBSTR(date, 1, 4) as year,
               SUM(CASE WHEN market_type = 'kospi' THEN 1 ELSE 0 END) as kospi,
               SUM(CASE WHEN market_type = 'kosdaq' THEN 1 ELSE 0 END) as kosdaq,
               SUM(CASE WHEN market_type = 'kodex' THEN 1 ELSE 0 END) as kodex,
               COUNT(DISTINCT date) as trading_days
        FROM daily_prices
        GROUP BY year
        ORDER BY year
    """)

    print(f"{'Year':<6} {'KOSPI':>10} {'KOSDAQ':>10} {'KODEX':>10} {'Days':>8} {'Status':<15}")
    print("-" * 70)

    missing_markets = []
    for row in cursor.fetchall():
        year, kospi, kosdaq, kodex, days = row
        status = "OK"
        missing = []
        if kospi == 0:
            missing.append('kospi')
        if kosdaq == 0:
            missing.append('kosdaq')
        if kodex == 0:
            missing.append('kodex')

        if missing:
            status = f"MISSING: {','.join(missing)}"
            missing_markets.append((year, missing))

        print(f"{year:<6} {kospi:>10,} {kosdaq:>10,} {kodex:>10,} {days:>8} {status:<15}")

    # Check for anomalies in daily record counts
    print("\n" + "-" * 70)
    print("DAILY RECORD COUNT ANOMALIES (< 1000 records)")
    print("-" * 70)

    cursor.execute("""
        SELECT date, COUNT(*) as cnt,
               SUM(CASE WHEN market_type = 'kospi' THEN 1 ELSE 0 END) as kospi,
               SUM(CASE WHEN market_type = 'kosdaq' THEN 1 ELSE 0 END) as kosdaq,
               SUM(CASE WHEN market_type = 'kodex' THEN 1 ELSE 0 END) as kodex
        FROM daily_prices
        GROUP BY date
        HAVING COUNT(*) < 1000
        ORDER BY date
    """)

    anomalies = cursor.fetchall()
    dates_to_fix = []

    if anomalies:
        print(f"Found {len(anomalies)} dates with low record counts:")
        print(f"{'Date':<12} {'Total':>8} {'KOSPI':>8} {'KOSDAQ':>8} {'KODEX':>8} {'Missing'}")
        print("-" * 70)
        for date, cnt, kospi, kosdaq, kodex in anomalies[:30]:  # Show max 30
            missing = []
            if kospi == 0:
                missing.append('kospi')
            if kosdaq == 0:
                missing.append('kosdaq')
            if kodex == 0:
                missing.append('kodex')
            if missing:
                dates_to_fix.append((date, missing))
            print(f"{date:<12} {cnt:>8,} {kospi:>8,} {kosdaq:>8,} {kodex:>8,} {','.join(missing)}")

        if len(anomalies) > 30:
            print(f"... and {len(anomalies) - 30} more dates")
    else:
        print("No anomalies found (all days have >= 1,000 records)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if not missing_markets and not dates_to_fix:
        print("✓ Backfill appears COMPLETE - all markets covered")
    else:
        if missing_markets:
            print(f"⚠ Years with missing markets:")
            for year, markets in missing_markets:
                print(f"   {year}: missing {', '.join(markets)}")

        if dates_to_fix:
            print(f"⚠ {len(dates_to_fix)} dates need fixing")

    conn.close()

    # Fix mode
    if args.fix and dates_to_fix:
        print("\n" + "=" * 70)
        print("FIXING MISSING DATA")
        print("=" * 70)

        from clean_etl import CleanETLPipeline
        from krx_api import KRXAPI
        from config import load_config

        config_dict = load_config()
        api = KRXAPI(config_dict['api']['auth_key'], config_dict.get('api', {}))
        pipeline = CleanETLPipeline(args.db_path)

        # Group dates by missing markets
        fix_summary = {}
        for date, missing in dates_to_fix:
            key = tuple(sorted(missing))
            if key not in fix_summary:
                fix_summary[key] = []
            fix_summary[key].append(date)

        total_fixed = 0
        for markets, dates in fix_summary.items():
            markets_list = list(markets)
            print(f"\nFetching {', '.join(markets_list)} for {len(dates)} dates...")

            for date in dates:
                try:
                    market_data = api.fetch_data_for_date_parallel(date, markets_list, is_backfill=True)

                    raw_data = []
                    for market, records in market_data.items():
                        raw_data.extend(records)

                    if raw_data:
                        result = pipeline.process_data(raw_data)
                        print(f"  {date}: +{result['prices_processed']} records")
                        total_fixed += result['prices_processed']
                    else:
                        print(f"  {date}: no data (holiday?)")

                except Exception as e:
                    print(f"  {date}: error - {e}")

        print(f"\nFixed {total_fixed} total records")

    elif args.fix and not dates_to_fix:
        print("\nNothing to fix!")


def cmd_etl_backfill(args):
    """Run historical data backfill."""
    from clean_etl import CleanETLPipeline
    from krx_api import KRXAPI
    from config import load_config

    pipeline = CleanETLPipeline(args.db_path)

    print(f"Starting backfill from {args.start_date} to {args.end_date}")
    print(f"Markets: {', '.join(args.markets)}")
    if args.force:
        print("Force mode enabled - will reprocess existing data")

    # Load config and initialize API
    config_dict = load_config()
    api = KRXAPI(config_dict['api']['auth_key'], config_dict.get('api', {}))

    # Check for resume capability
    progress_data = pipeline.load_progress()
    start_date_str = args.start_date

    if progress_data and not args.force:
        print(f"Found existing progress: {progress_data.get('progress_percentage', 0):.1f}% complete")
        resume = input("Resume from last processed date? (y/n): ").lower().strip()
        if resume == 'y':
            last_date = progress_data.get('last_processed_date')
            if last_date:
                resume_date = datetime.strptime(last_date, '%Y%m%d') + timedelta(days=1)
                start_date_str = resume_date.strftime('%Y%m%d')
                print(f"Resuming from {start_date_str}")

    # Get progress information
    progress = pipeline.get_backfill_progress(start_date_str, args.end_date)
    print(f"\nTotal trading days: {progress['total_trading_days']}")
    print(f"Already processed: {progress['processed_dates']}")
    print(f"Remaining: {progress['remaining_dates']}")
    print(f"Progress: {progress['progress_percentage']:.1f}%")

    # Determine dates to process
    if args.force:
        dates_to_process = sorted(progress['remaining_dates_list'] + progress['processed_dates_list'])
    else:
        dates_to_process = progress['remaining_dates_list']
        if not dates_to_process:
            print("All dates already processed. Use --force to reprocess.")
            return

    print(f"\nProcessing {len(dates_to_process)} dates...")

    processed_dates = 0
    total_records = 0

    try:
        for date_str in dates_to_process:
            market_data = api.fetch_data_for_date_parallel(date_str, args.markets, is_backfill=True)

            if market_data:
                raw_data = []
                for market, records in market_data.items():
                    raw_data.extend(records)

                if raw_data:
                    result = pipeline.process_data(raw_data)
                    total_records += result['prices_processed']
                    processed_dates += 1
                    print(f"Processed {result['prices_processed']} records for {date_str}")

                    # Save progress every 10 dates
                    if processed_dates % 10 == 0:
                        pipeline.save_progress({
                            'start_date': args.start_date,
                            'end_date': args.end_date,
                            'last_processed_date': date_str,
                            'processed_dates': processed_dates,
                            'total_records': total_records,
                            'markets': args.markets
                        })
            else:
                print(f"No data for {date_str}")

        print(f"\nBackfill completed: {processed_dates} dates, {total_records} records")

        # Clean up progress file if done
        final_progress = pipeline.get_backfill_progress(args.start_date, args.end_date)
        if final_progress['remaining_dates'] == 0:
            import os
            progress_file = f"{args.db_path}.progress.json"
            if os.path.exists(progress_file):
                os.remove(progress_file)

    except Exception as e:
        print(f"Error during backfill: {e}")
        pipeline.save_progress({
            'start_date': args.start_date,
            'end_date': args.end_date,
            'last_processed_date': date_str if 'date_str' in locals() else None,
            'processed_dates': processed_dates,
            'total_records': total_records,
            'markets': args.markets,
            'error': str(e)
        })
        print("Progress saved - you can resume later")
        sys.exit(1)


def cmd_etl_update(args):
    """Run daily update or catch-up from last date."""
    from clean_etl import CleanETLPipeline
    from krx_api import KRXAPI
    from config import load_config

    pipeline = CleanETLPipeline(args.db_path)
    config_dict = load_config()
    api = KRXAPI(config_dict['api']['auth_key'], config_dict.get('api', {}))

    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y%m%d')

    # Determine dates to process
    if args.catchup:
        # Catch-up mode: find last date in DB and fill gaps to yesterday
        status = pipeline.get_status()
        if status['date_range'][1]:
            last_date_str = status['date_range'][1]
            last_date = datetime.strptime(last_date_str, '%Y%m%d')
            start_date = last_date + timedelta(days=1)
        else:
            # No data in DB, start from 30 days ago
            start_date = yesterday - timedelta(days=30)
            last_date_str = "none"

        if start_date > yesterday:
            print(f"Database is up to date (last: {last_date_str})")
            return

        print(f"Catch-up mode: {start_date.strftime('%Y%m%d')} → {yesterday_str}")
        print(f"Last date in DB: {last_date_str}")

        # Generate list of dates to process (skip weekends)
        dates_to_process = []
        current = start_date
        while current <= yesterday:
            if current.weekday() < 5:  # Skip weekends
                dates_to_process.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)

        if not dates_to_process:
            print("No trading days to process")
            return

        print(f"Processing {len(dates_to_process)} trading days...")
        print(f"Markets: {', '.join(args.markets)}")

    elif args.date:
        # Specific date mode
        dates_to_process = [args.date]
        print(f"Running update for {args.date}")
        print(f"Markets: {', '.join(args.markets)}")

    else:
        # Default: yesterday only
        dates_to_process = [yesterday_str]
        print(f"Running daily update for {yesterday_str}")
        print(f"Markets: {', '.join(args.markets)}")

    # Process each date
    total_records = 0
    processed_days = 0

    try:
        for date_str in dates_to_process:
            # Check if data exists (skip unless force)
            if not args.force and pipeline.check_date_exists(date_str):
                print(f"Skipping {date_str} (exists)")
                continue

            market_data = api.fetch_data_for_date_parallel(
                date_str, args.markets, is_backfill=args.catchup
            )

            raw_data = []
            for market, records in market_data.items():
                raw_data.extend(records)

            if raw_data:
                result = pipeline.process_data(raw_data)
                total_records += result['prices_processed']
                processed_days += 1
                print(f"Processed {result['prices_processed']} records for {date_str}")
            else:
                print(f"No data for {date_str} (holiday?)")

        if args.catchup:
            print(f"\nCatch-up complete: {processed_days} days, {total_records} records")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# =============================================================================
# Analysis Commands
# =============================================================================

def cmd_analyze_price(args):
    """Screen by price increase."""
    from analyzer.screeners import StockScreener

    screener = StockScreener(args.db_path)

    print(f"Screening for top {args.percentile}% by price increase...")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Markets: {', '.join(args.markets)}")

    results = screener.top_price_increase(
        args.start_date, args.end_date, args.markets, args.percentile
    )

    if len(results) == 0:
        print("No results found.")
        return

    print(f"\nFound {len(results)} stocks:")
    print(results.to_string())

    if args.export:
        filename = screener.export_results(results, "price_screening")
        print(f"\nExported to: {filename}")


def cmd_analyze_value(args):
    """Screen by trading value."""
    from analyzer.screeners import StockScreener

    screener = StockScreener(args.db_path)

    print(f"Screening for top {args.percentile}% by trading value...")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Markets: {', '.join(args.markets)}")

    results = screener.top_value(
        args.start_date, args.end_date, args.markets, args.percentile
    )

    if len(results) == 0:
        print("No results found.")
        return

    print(f"\nFound {len(results)} stocks:")
    print(results.to_string())

    if args.export:
        filename = screener.export_results(results, "value_screening")
        print(f"\nExported to: {filename}")


def cmd_analyze_combined(args):
    """Combined screening by price and value."""
    from analyzer.screeners import StockScreener

    screener = StockScreener(args.db_path)

    print(f"Combined screening: top {args.price_percentile}% price AND top {args.value_percentile}% value")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Markets: {', '.join(args.markets)}")

    results = screener.combined_screen(
        args.start_date, args.end_date, args.markets,
        args.price_percentile, args.value_percentile
    )

    if len(results) == 0:
        print("No results found.")
        return

    print(f"\nFound {len(results)} stocks meeting both criteria:")
    print(results.to_string())

    if args.export:
        filename = screener.export_results(results, "combined_screening")
        print(f"\nExported to: {filename}")


def cmd_quick(args):
    """Quick screening for current year."""
    from analyzer.screeners import StockScreener

    screener = StockScreener(args.db_path)

    current_year = datetime.now().year
    start_date = f"{current_year}0101"
    end_date = f"{current_year}1231"

    print(f"Quick screening for {current_year}")
    print(f"Top {args.percentile}% by both price increase and trading value")
    print(f"Markets: {', '.join(args.markets)}")

    results = screener.combined_screen(
        start_date, end_date, args.markets, args.percentile, args.percentile
    )

    if len(results) == 0:
        print("No results found.")
        return

    print(f"\nFound {len(results)} top performers for {current_year}:")
    print(results.to_string())

    if args.export:
        filename = screener.export_results(results, f"quick_screen_{current_year}")
        print(f"\nExported to: {filename}")


# =============================================================================
# Backtest Commands
# =============================================================================

def cmd_backtest(args):
    """Run backtesting on combined screening strategy."""
    from analyzer.backtester import Backtester

    print(f"Running backtest...")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Stocks: {args.n_stocks}, Hold: {args.hold_months}m, Rebalance: {args.rebalance_months}m")
    print(f"Weighting: {args.weighting}, Sort by: {args.sort_by}")
    print(f"Markets: {', '.join(args.markets)}")
    print()

    with Backtester(args.db_path) as bt:
        results = bt.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            n_stocks=args.n_stocks,
            holding_months=args.hold_months,
            rebalance_months=args.rebalance_months,
            weighting=args.weighting,
            sort_by=args.sort_by,
            price_percentile=args.price_percentile,
            value_percentile=args.value_percentile,
            markets=args.markets
        )

        # Print report
        bt.print_report(results)

        # Export if requested
        if args.export:
            files = bt.export_results(results, "backtest")
            print(f"\nExported to:")
            for name, path in files.items():
                print(f"  {name}: {path}")


# =============================================================================
# Main CLI Setup
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='algostock',
        description='AlgoStock - KRX Stock Data Management and Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ETL Operations
  algostock etl status                              Show database status
  algostock etl backfill -s 20200101 -e 20251231    Backfill historical data
  algostock etl update                              Update yesterday's data
  algostock etl update --catchup                    Auto catch-up to yesterday
  algostock etl validate                            Validate data integrity

  # Stock Analysis
  algostock analyze price -s 20250101 -e 20251231 -p 1    Top 1% by price
  algostock analyze value -s 20250101 -e 20251231 -p 5    Top 5% by value
  algostock analyze combined -s 20250101 -e 20251231      Combined screening

  # Quick Analysis
  algostock quick -p 5                              Top 5% for current year

  # Backtesting
  algostock backtest -s 20150101 -e 20241231 -n 5   Backtest top 5 stocks
  algostock backtest -s 20150101 -e 20241231 -n 10 --hold-months 6 --rebalance-months 6 -w value
        """
    )

    # Global arguments
    parser.add_argument('--db-path', default='krx_stock_data.db', help='Database path')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # -------------------------------------------------------------------------
    # ETL Commands
    # -------------------------------------------------------------------------
    etl_parser = subparsers.add_parser('etl', help='ETL operations')
    etl_subparsers = etl_parser.add_subparsers(dest='etl_command', help='ETL commands')

    # etl status
    etl_status = etl_subparsers.add_parser('status', help='Show database status')
    etl_status.set_defaults(func=cmd_etl_status)

    # etl validate
    etl_validate = etl_subparsers.add_parser('validate', help='Validate data integrity')
    etl_validate.set_defaults(func=cmd_etl_validate)

    # etl cleanup
    etl_cleanup = etl_subparsers.add_parser('cleanup', help='Clean up old data')
    etl_cleanup.add_argument('--days', type=int, default=365, help='Days to keep')
    etl_cleanup.set_defaults(func=cmd_etl_cleanup)

    # etl verify
    etl_verify = etl_subparsers.add_parser('verify', help='Verify backfill completeness')
    etl_verify.add_argument('--fix', action='store_true',
                            help='Automatically fix missing market data')
    etl_verify.set_defaults(func=cmd_etl_verify)

    # etl backfill
    etl_backfill = etl_subparsers.add_parser('backfill', help='Run historical backfill')
    etl_backfill.add_argument('-s', '--start-date', type=validate_date, required=True,
                              help='Start date (YYYYMMDD)')
    etl_backfill.add_argument('-e', '--end-date', type=validate_date, required=True,
                              help='End date (YYYYMMDD)')
    etl_backfill.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                              help='Markets (comma-separated)')
    etl_backfill.add_argument('--force', action='store_true', help='Force reprocess existing data')
    etl_backfill.set_defaults(func=cmd_etl_backfill)

    # etl update
    etl_update = etl_subparsers.add_parser('update', help='Run daily update or catch-up')
    etl_update.add_argument('--date', type=validate_date, help='Specific date (YYYYMMDD)')
    etl_update.add_argument('--catchup', action='store_true',
                            help='Auto catch-up from last DB date to yesterday')
    etl_update.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                            help='Markets (comma-separated)')
    etl_update.add_argument('--force', action='store_true', help='Force reprocess existing data')
    etl_update.set_defaults(func=cmd_etl_update)

    # -------------------------------------------------------------------------
    # Analysis Commands
    # -------------------------------------------------------------------------
    analyze_parser = subparsers.add_parser('analyze', help='Stock analysis')
    analyze_subparsers = analyze_parser.add_subparsers(dest='analyze_command', help='Analysis commands')

    # Common analysis arguments
    def add_analysis_args(p, combined=False):
        p.add_argument('-s', '--start-date', type=validate_date, required=True,
                       help='Start date (YYYYMMDD)')
        p.add_argument('-e', '--end-date', type=validate_date, required=True,
                       help='End date (YYYYMMDD)')
        p.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                       help='Markets (comma-separated)')
        if combined:
            p.add_argument('--price-percentile', type=validate_percentile, default=5,
                           help='Top percentile for price (1-100)')
            p.add_argument('--value-percentile', type=validate_percentile, default=5,
                           help='Top percentile for value (1-100)')
        else:
            p.add_argument('-p', '--percentile', type=validate_percentile, default=1,
                           help='Top percentile (1-100)')
        p.add_argument('--export', action='store_true', help='Export to Excel')

    # analyze price
    analyze_price = analyze_subparsers.add_parser('price', help='Screen by price increase')
    add_analysis_args(analyze_price)
    analyze_price.set_defaults(func=cmd_analyze_price)

    # analyze value
    analyze_value = analyze_subparsers.add_parser('value', help='Screen by trading value')
    add_analysis_args(analyze_value)
    analyze_value.set_defaults(func=cmd_analyze_value)

    # analyze combined
    analyze_combined = analyze_subparsers.add_parser('combined', help='Combined screening')
    add_analysis_args(analyze_combined, combined=True)
    analyze_combined.set_defaults(func=cmd_analyze_combined)

    # -------------------------------------------------------------------------
    # Quick Command
    # -------------------------------------------------------------------------
    quick_parser = subparsers.add_parser('quick', help='Quick screening for current year')
    quick_parser.add_argument('-p', '--percentile', type=validate_percentile, default=5,
                              help='Top percentile (1-100)')
    quick_parser.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                              help='Markets (comma-separated)')
    quick_parser.add_argument('--export', action='store_true', help='Export to Excel')
    quick_parser.set_defaults(func=cmd_quick)

    # -------------------------------------------------------------------------
    # Backtest Command
    # -------------------------------------------------------------------------
    backtest_parser = subparsers.add_parser('backtest', help='Backtest screening strategy')
    backtest_parser.add_argument('-s', '--start-date', type=validate_date, required=True,
                                 help='Backtest start date (YYYYMMDD) - first holding period starts here')
    backtest_parser.add_argument('-e', '--end-date', type=validate_date, required=True,
                                 help='Backtest end date (YYYYMMDD)')
    backtest_parser.add_argument('-n', '--n-stocks', type=int, default=5,
                                 help='Number of top stocks to select (default: 5)')
    backtest_parser.add_argument('--hold-months', type=int, default=12,
                                 help='Holding period in months (default: 12)')
    backtest_parser.add_argument('--rebalance-months', type=int, default=12,
                                 help='Rebalancing interval in months (default: 12)')
    backtest_parser.add_argument('-w', '--weighting', default='equal',
                                 choices=['equal', 'value', 'inverse_value', 'mcap', 'inverse_mcap'],
                                 help='Portfolio weighting method (default: equal)')
    backtest_parser.add_argument('--sort-by', default='price',
                                 choices=['price', 'value', 'combined'],
                                 help='How to rank top N: price (return), value (trading value), combined (default: price)')
    backtest_parser.add_argument('--price-percentile', type=validate_percentile, default=5,
                                 help='Top percentile for price screening (default: 5)')
    backtest_parser.add_argument('--value-percentile', type=validate_percentile, default=5,
                                 help='Top percentile for value screening (default: 5)')
    backtest_parser.add_argument('-m', '--markets', type=validate_markets, default='kospi,kosdaq',
                                 help='Markets (comma-separated)')
    backtest_parser.add_argument('--export', action='store_true', help='Export results to CSV')
    backtest_parser.set_defaults(func=cmd_backtest)

    # -------------------------------------------------------------------------
    # Parse and Execute
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(args.log_file, log_level)

    # Handle missing commands
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'etl' and not getattr(args, 'etl_command', None):
        etl_parser.print_help()
        sys.exit(1)

    if args.command == 'analyze' and not getattr(args, 'analyze_command', None):
        analyze_parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
