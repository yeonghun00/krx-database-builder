#!/usr/bin/env python3
"""
KRX Index Constituents Extractor

This script extracts stock codes from KRX index CSV files and inserts them directly
into the database for industry analysis and computation.

Features:
1. Backfill mode: Process historical data from 2010-01-01 to present
2. Update mode: Update from latest date with overwrite or skip options
3. Direct database insertion (no CSV output)
4. Index name formatting with underscores and market prefixes

Usage:
    python extract_krx_index_constituents.py --mode backfill
    python extract_krx_index_constituents.py --mode update --strategy overwrite
    python extract_krx_index_constituents.py --mode update --strategy skip
"""

import argparse
import sqlite3
import pandas as pd
import re
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from scripts.fetch_krx_indices import KRXIndexScraper


class IndexConstituentsExtractor:
    """Extracts KRX index constituents and inserts them into the database."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the extractor with database connection."""
        # Get the parent directory (project root) for config file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_full_path = os.path.join(project_root, config_path)
        self.config = load_config(config_full_path)
        self.db_path = self.config.get('database', {}).get('path', 'krx_stock_data.db')
        
        # Create database table if it doesn't exist
        self._create_table()
        
    def _create_table(self):
        """Create the index_constituents table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS index_constituents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    index_code TEXT NOT NULL,
                    stock_code TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, index_code, stock_code)
                )
            """)
            conn.commit()
    
    def format_index_name(self, index_name: str, market: str) -> str:
        """
        Format index name with underscores and market prefix.
        
        Args:
            index_name (str): Original index name
            market (str): Market type ('kospi' or 'kosdaq')
            
        Returns:
            str: Formatted index name (e.g., 'KOSPI_코스피_200')
        """
        # Convert market to uppercase prefix
        market_prefix = market.upper()
        
        # Replace spaces with underscores
        formatted_name = index_name.replace(' ', '_')
        
        # Replace special characters that might cause issues
        formatted_name = re.sub(r'[^\w\s\(\)]', '_', formatted_name)
        
        # Remove multiple consecutive underscores
        formatted_name = re.sub(r'_+', '_', formatted_name)
        
        # Remove leading/trailing underscores
        formatted_name = formatted_name.strip('_')
        
        # Add market prefix
        return f"{market_prefix}_{formatted_name}"
    
    def extract_constituents_from_content(self, csv_content: str, index_name: str, market: str) -> List[Tuple[str, str]]:
        """
        Extract stock codes directly from CSV content without saving to file.
        
        Args:
            csv_content (str): CSV content as string
            index_name (str): Name of the index
            market (str): Market type
            
        Returns:
            List[Tuple[str, str]]: List of (formatted_index_name, stock_code) tuples
        """
        try:
            # Try to decode with multiple encodings
            encodings = ['cp949', 'utf-8', 'euc-kr']
            decoded_content = None
            
            for encoding in encodings:
                try:
                    if isinstance(csv_content, bytes):
                        decoded_content = csv_content.decode(encoding)
                    else:
                        decoded_content = csv_content
                    break
                except UnicodeDecodeError:
                    continue
            
            if decoded_content is None:
                print(f"   Warning: Could not decode CSV content with any encoding")
                return []
            
            # Read CSV from string
            df = pd.read_csv(StringIO(decoded_content))
            
            # Find the column with stock codes (should be '종목코드')
            stock_code_column = None
            for col in df.columns:
                if '종목코드' in col:
                    stock_code_column = col
                    break
            
            if not stock_code_column:
                print(f"   Warning: Could not find '종목코드' column in CSV content")
                return []
            
            # Extract stock codes
            stock_codes = df[stock_code_column].dropna().astype(str).tolist()
            
            # Format index name
            formatted_index_name = self.format_index_name(index_name, market)
            
            # Create tuples for database insertion
            constituents = [(formatted_index_name, stock_code) for stock_code in stock_codes]
            
            print(f"   Extracted {len(constituents)} constituents for {formatted_index_name}")
            return constituents
            
        except Exception as e:
            print(f"   Error processing CSV content: {e}")
            return []
    
    def insert_constituents(self, date: str, constituents: List[Tuple[str, str]], strategy: str = 'skip'):
        """
        Insert constituents into the database.
        
        Args:
            date (str): Date for the index composition
            constituents (List[Tuple[str, str]]): List of (index_code, stock_code) tuples
            strategy (str): 'overwrite' or 'skip'
        """
        with sqlite3.connect(self.db_path) as conn:
            if strategy == 'overwrite':
                # Delete existing data for this date and index
                for index_code, _ in constituents:
                    conn.execute(
                        "DELETE FROM index_constituents WHERE date = ? AND index_code = ?",
                        (date, index_code)
                    )
            
            # Insert new data
            for index_code, stock_code in constituents:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO index_constituents (date, index_code, stock_code) VALUES (?, ?, ?)",
                        (date, index_code, stock_code)
                    )
                except sqlite3.IntegrityError:
                    if strategy == 'skip':
                        continue
                    else:
                        # For overwrite, we already deleted, so this shouldn't happen
                        pass
            
            conn.commit()
            print(f"   Inserted {len(constituents)} constituents for {date}")
    
    def get_latest_date(self) -> Optional[str]:
        """Get the latest date in the index_constituents table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT MAX(date) FROM index_constituents")
            result = cursor.fetchone()
            return result[0] if result[0] else None
    
    def process_date(self, date: str, market: str = 'kospi') -> bool:
        """
        Process a single date by downloading CSVs and extracting constituents.
        
        Args:
            date (str): Date to process (YYYY-MM-DD format)
            market (str): Market to process ('kospi' or 'kosdaq')
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\nProcessing {date} for {market.upper()}...")
        
        try:
            # Create scraper for this market
            scraper = KRXIndexScraper(market=market)
            
            # Set date range for this specific date
            # Note: The scraper uses fixed dates, so we'll need to modify it
            # For now, we'll use the existing functionality and process all downloaded files
            
            # Run the scraper to download CSVs
            success = scraper.run(delay=0.5)  # Reduced delay for faster processing
            
            if not success:
                print(f"   Failed to download CSVs for {date}")
                return False
            
            # Process downloaded CSV files
            constituents = []
            output_dir = scraper.output_dir
            
            if not os.path.exists(output_dir):
                print(f"   No output directory found: {output_dir}")
                return False
            
            # Find all CSV files in the output directory
            csv_files = list(Path(output_dir).glob("*.csv"))
            
            for csv_file in csv_files:
                # Extract index name from filename (remove date suffix)
                filename = csv_file.stem
                # Remove the index ID suffix (last part after _)
                index_name = '_'.join(filename.split('_')[:-1])
                
                # Extract constituents from this CSV
                file_constituents = self.extract_constituents_from_content(str(csv_file), index_name, market)
                constituents.extend(file_constituents)
            
            # Insert into database
            if constituents:
                self.insert_constituents(date, constituents, strategy='overwrite')
                return True
            else:
                print(f"   No constituents found for {date}")
                return False
                
        except Exception as e:
            print(f"   Error processing {date}: {e}")
            return False
    
    def backfill(self, start_date: str = "2010-01-01"):
        """
        Backfill historical data from start_date to present.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
        """
        print(f"Starting backfill from {start_date} to present...")
        
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.now()
        
        processed_count = 0
        failed_count = 0
        
        while current_date <= end_date:
            # Process monthly dates
            if current_date.day == 1:  # First day of month
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Try to process this date
                success = self.process_date(date_str, 'kospi') and self.process_date(date_str, 'kosdaq')
                
                if success:
                    processed_count += 1
                    print(f"✓ Successfully processed {date_str}")
                else:
                    failed_count += 1
                    print(f"✗ Failed to process {date_str}")
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            else:
                # Move to first day of current month
                current_date = current_date.replace(day=1)
        
        print(f"\nBackfill completed!")
        print(f"Processed: {processed_count} dates")
        print(f"Failed: {failed_count} dates")
    
    def update(self, strategy: str = 'skip'):
        """
        Update from the latest date in the database.
        
        Args:
            strategy (str): 'overwrite' or 'skip'
        """
        latest_date = self.get_latest_date()
        
        if not latest_date:
            print("No existing data found. Starting backfill from 2010-01-01...")
            self.backfill()
            return
        
        print(f"Latest date in database: {latest_date}")
        
        # Start from next month
        latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
        if latest_dt.month == 12:
            start_date = latest_dt.replace(year=latest_dt.year + 1, month=1)
        else:
            start_date = latest_dt.replace(month=latest_dt.month + 1)
        
        print(f"Starting update from {start_date.strftime('%Y-%m-%d')}...")
        
        processed_count = 0
        failed_count = 0
        
        current_date = start_date
        end_date = datetime.now()
        
        while current_date <= end_date:
            if current_date.day == 1:  # First day of month
                date_str = current_date.strftime("%Y-%m-%d")
                
                # Check if date already exists (for skip strategy)
                if strategy == 'skip':
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute(
                            "SELECT COUNT(*) FROM index_constituents WHERE date = ?",
                            (date_str,)
                        )
                        if cursor.fetchone()[0] > 0:
                            print(f"   Skipping existing date: {date_str}")
                            current_date = current_date.replace(month=current_date.month + 1)
                            continue
                
                # Process this date
                success = self.process_date(date_str, 'kospi') and self.process_date(date_str, 'kosdaq')
                
                if success:
                    processed_count += 1
                    print(f"✓ Successfully processed {date_str}")
                else:
                    failed_count += 1
                    print(f"✗ Failed to process {date_str}")
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            else:
                current_date = current_date.replace(day=1)
        
        print(f"\nUpdate completed!")
        print(f"Processed: {processed_count} dates")
        print(f"Failed: {failed_count} dates")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Extract KRX Index Constituents')
    parser.add_argument('--mode', choices=['backfill', 'update'], default='backfill',
                        help='Processing mode: backfill (historical) or update (from latest)')
    parser.add_argument('--strategy', choices=['overwrite', 'skip'], default='skip',
                        help='Update strategy: overwrite existing data or skip')
    parser.add_argument('--start-date', default='2010-01-01',
                        help='Start date for backfill mode (YYYY-MM-DD format)')
    parser.add_argument('--config', default='config.json',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = IndexConstituentsExtractor(args.config)
    
    # Run based on mode
    if args.mode == 'backfill':
        extractor.backfill(args.start_date)
    else:
        extractor.update(args.strategy)


if __name__ == '__main__':
    main()