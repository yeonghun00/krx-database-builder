#!/usr/bin/env python3
"""
AlgoStock Analyzer - Main entry point for stock analysis.

This script provides a simple interface to the analyzer module.
"""

import sys
import os

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer.screeners import StockScreener
import argparse
from datetime import datetime

def main():
    """Main entry point for the analyzer."""
    
    parser = argparse.ArgumentParser(
        description='AlgoStock Analyzer - Fast stock screening for KRX data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick screening for current year (top 5%)
  python3 algostock_analyzer.py
  
  # Top 1% by price increase in 2025
  python3 algostock_analyzer.py --start-date 20250101 --end-date 20251231 --percentile 1 --type price
  
  # Top 1% by volume in 2025
  python3 algostock_analyzer.py --start-date 20250101 --end-date 20251231 --percentile 1 --type volume
  
  # Combined screening (top 5% by both)
  python3 algostock_analyzer.py --start-date 20250101 --end-date 20251231 --percentile 5 --type combined
        """
    )
    
    parser.add_argument('--db-path', default='krx_stock_data.db', help='Path to database file')
    parser.add_argument('--start-date', help='Start date (YYYYMMDD format)')
    parser.add_argument('--end-date', help='End date (YYYYMMDD format)')
    parser.add_argument('--percentile', type=int, default=5, help='Top percentile (1-100)')
    parser.add_argument('--type', choices=['price', 'value', 'combined'], default='combined', 
                       help='Screening type: price (price increase), value (거래대금/trading value), combined (both)')
    parser.add_argument('--markets', default='kospi,kosdaq', help='Markets to include (comma-separated)')
    parser.add_argument('--export', action='store_true', help='Export results to Excel')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.percentile < 1 or args.percentile > 100:
        print("Error: Percentile must be between 1 and 100")
        sys.exit(1)
    
    # Parse markets
    markets = [m.strip().lower() for m in args.markets.split(',')]
    valid_markets = ['kospi', 'kosdaq', 'kodex']
    for market in markets:
        if market not in valid_markets:
            print(f"Error: Invalid market: {market}. Valid markets: {', '.join(valid_markets)}")
            sys.exit(1)
    
    # Initialize screener
    print(f"Initializing AlgoStock Analyzer...")
    print(f"Database: {args.db_path}")
    print(f"Markets: {', '.join(markets)}")
    
    try:
        screener = StockScreener(args.db_path)
    except Exception as e:
        print(f"Error initializing screener: {e}")
        sys.exit(1)
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
        print(f"Date range: {start_date} to {end_date}")
    else:
        # Use current year
        current_year = datetime.now().year
        start_date = f"{current_year}0101"
        end_date = f"{current_year}1231"
        print(f"Using current year: {current_year}")
    
    # Run screening
    print(f"\nRunning {args.type} screening for top {args.percentile}%...")
    
    try:
        if args.type == 'price':
            results = screener.top_price_increase(start_date, end_date, markets, args.percentile)
            title = f"Top {args.percentile}% by Price Increase"
            
        elif args.type == 'value':
            results = screener.top_value(start_date, end_date, markets, args.percentile)
            title = f"Top {args.percentile}% by Trading Value (거래대금)"
            
        else:  # combined
            results = screener.combined_screen(start_date, end_date, markets, args.percentile, args.percentile)
            title = f"Top {args.percentile}% by Both Price Increase and Trading Value (거래대금)"
        
        # Display results
        print(f"\n=== {title} ===")
        
        if len(results) == 0:
            print("No results found.")
            return
        
        print(f"Found {len(results)} stocks:")
        print(results.to_string())
        
        # Export if requested
        if args.export:
            filename = screener.export_results(results, f"{args.type}_screening_{start_date}_{end_date}")
            print(f"\nResults exported to: {filename}")
        
        print(f"\nScreening completed successfully!")
        
    except Exception as e:
        print(f"Error during screening: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()