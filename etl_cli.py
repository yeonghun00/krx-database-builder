#!/usr/bin/env python3
"""
Simple CLI for Clean ETL Pipeline
"""

import argparse
import logging
from clean_etl import CleanETLPipeline

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('clean_etl.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Clean ETL Pipeline for KRX Stock Data')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show ETL status')
    status_parser.add_argument('--db-path', default='krx_stock_data.db', help='Database path')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data integrity')
    validate_parser.add_argument('--db-path', default='krx_stock_data.db', help='Database path')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument('--db-path', default='krx_stock_data.db', help='Database path')
    cleanup_parser.add_argument('--days', type=int, default=365, help='Days to keep')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        pipeline = CleanETLPipeline(args.db_path)
        
        if args.command == 'status':
            status = pipeline.get_status()
            print("\n=== Clean ETL Status ===")
            print(f"Stocks: {status['stocks']:,}")
            print(f"Daily Prices: {status['daily_prices']:,}")
            print(f"Date Range: {status['date_range'][0]} to {status['date_range'][1]}")
            
        elif args.command == 'validate':
            validation = pipeline.validate_data()
            print("\n=== Validation Results ===")
            print(f"Orphaned Prices: {validation['orphaned_prices']}")
            print(f"Duplicate Prices: {validation['duplicate_prices']}")
            print(f"Validation Passed: {'Yes' if validation['validation_passed'] else 'No'}")
            
            if not validation['validation_passed']:
                logger.error("Validation failed!")
                exit(1)
                
        elif args.command == 'cleanup':
            logger.info(f"Cleaning up data older than {args.days} days...")
            pipeline.cleanup_old_data(days_to_keep=args.days)
            logger.info("Cleanup completed!")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    main()
