#!/usr/bin/env python3
"""
Command Line Interface for Stock Analysis

Provides CLI commands for running stock screening and analysis operations.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from analyzer.screeners import StockScreener

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        else:
            raise argparse.ArgumentTypeError(f"Percentile must be between 1 and 100, got {value}")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid percentile: {percentile}. Must be an integer")

def validate_markets(markets_str: str) -> list:
    """Validate markets list."""
    valid_markets = ['kospi', 'kosdaq', 'kodex']
    markets = [m.strip().lower() for m in markets_str.split(',')]
    
    for market in markets:
        if market not in valid_markets:
            raise argparse.ArgumentTypeError(f"Invalid market: {market}. Valid markets: {', '.join(valid_markets)}")
    
    return markets

def cmd_price_increase(args):
    """Handle price increase screening command."""
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
        filename = screener.export_results(results, "price_increase_screening")
        print(f"\nResults exported to: {filename}")

def cmd_volume(args):
    """Handle volume screening command."""
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
        filename = screener.export_results(results, "volume_screening")
        print(f"\nResults exported to: {filename}")

def cmd_combined(args):
    """Handle combined screening command."""
    screener = StockScreener(args.db_path)
    
    print(f"Combined screening: top {args.price_percentile}% by price AND top {args.volume_percentile}% by volume")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Markets: {', '.join(args.markets)}")
    
    results = screener.combined_screen(
        args.start_date, args.end_date, args.markets, 
        args.price_percentile, args.volume_percentile
    )
    
    if len(results) == 0:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} stocks meeting both criteria:")
    print(results.to_string())
    
    if args.export:
        filename = screener.export_results(results, "combined_screening")
        print(f"\nResults exported to: {filename}")

def cmd_quick_screen(args):
    """Handle quick screening for current year."""
    screener = StockScreener(args.db_path)
    
    # Get current year
    current_year = datetime.now().year
    start_date = f"{current_year}0101"
    end_date = f"{current_year}1231"
    
    print(f"Quick screening for {current_year}...")
    print(f"Top {args.percentile}% by price increase and volume")
    
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
        print(f"\nResults exported to: {filename}")

def main():
    """Main CLI entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description='AlgoStock Analysis Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Screen for top 1% by price increase in 2025
  python analyzer_cli.py price-increase --start-date 20250101 --end-date 20251231 --percentile 1
  
  # Screen for top 5% by volume in 2025
  python analyzer_cli.py volume --start-date 20250101 --end-date 20251231 --percentile 5
  
  # Combined screening for top 1% by both metrics
  python analyzer_cli.py combined --start-date 20250101 --end-date 20251231 --price-percentile 1 --volume-percentile 1
  
  # Quick screening for current year
  python analyzer_cli.py quick --percentile 5
        """
    )
    
    # Global arguments
    parser.add_argument('--db-path', default='krx_stock_data.db', help='Path to database file')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Price increase screening
    price_parser = subparsers.add_parser('price-increase', help='Screen by price increase')
    price_parser.add_argument('--start-date', type=validate_date, required=True, help='Start date (YYYYMMDD)')
    price_parser.add_argument('--end-date', type=validate_date, required=True, help='End date (YYYYMMDD)')
    price_parser.add_argument('--markets', type=validate_markets, default='kospi,kosdaq', help='Markets to include (comma-separated)')
    price_parser.add_argument('--percentile', type=validate_percentile, default=1, help='Top percentile (1-100)')
    price_parser.add_argument('--export', action='store_true', help='Export results to Excel')
    price_parser.set_defaults(func=cmd_price_increase)
    
    # Volume screening
    volume_parser = subparsers.add_parser('volume', help='Screen by trading volume')
    volume_parser.add_argument('--start-date', type=validate_date, required=True, help='Start date (YYYYMMDD)')
    volume_parser.add_argument('--end-date', type=validate_date, required=True, help='End date (YYYYMMDD)')
    volume_parser.add_argument('--markets', type=validate_markets, default='kospi,kosdaq', help='Markets to include (comma-separated)')
    volume_parser.add_argument('--percentile', type=validate_percentile, default=1, help='Top percentile (1-100)')
    volume_parser.add_argument('--export', action='store_true', help='Export results to Excel')
    volume_parser.set_defaults(func=cmd_volume)
    
    # Combined screening
    combined_parser = subparsers.add_parser('combined', help='Combined screening by price and volume')
    combined_parser.add_argument('--start-date', type=validate_date, required=True, help='Start date (YYYYMMDD)')
    combined_parser.add_argument('--end-date', type=validate_date, required=True, help='End date (YYYYMMDD)')
    combined_parser.add_argument('--markets', type=validate_markets, default='kospi,kosdaq', help='Markets to include (comma-separated)')
    combined_parser.add_argument('--price-percentile', type=validate_percentile, default=5, help='Top percentile for price increase (1-100)')
    combined_parser.add_argument('--volume-percentile', type=validate_percentile, default=5, help='Top percentile for volume (1-100)')
    combined_parser.add_argument('--export', action='store_true', help='Export results to Excel')
    combined_parser.set_defaults(func=cmd_combined)
    
    # Quick screening
    quick_parser = subparsers.add_parser('quick', help='Quick screening for current year')
    quick_parser.add_argument('--markets', type=validate_markets, default='kospi,kosdaq', help='Markets to include (comma-separated)')
    quick_parser.add_argument('--percentile', type=validate_percentile, default=5, help='Top percentile for both metrics (1-100)')
    quick_parser.add_argument('--export', action='store_true', help='Export results to Excel')
    quick_parser.set_defaults(func=cmd_quick_screen)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run the selected command
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()