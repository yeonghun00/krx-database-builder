#!/usr/bin/env python3
"""
Simple KRX Delisted Stocks Data Fetcher

Downloads delisted stock data from KRX and converts to pandas DataFrame.
Uses pd.read_html() with converters for proper stock code handling.
Saves data to CSV file - no database complexity.

Usage:
    python fetch_delisted_stocks_simple.py
"""

import pandas as pd
import requests
import sqlite3
import logging
import sys
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_database_table(db_path="krx_stock_data.db"):
    """Create the delisted_stocks table if it doesn't exist."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS delisted_stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stock_code TEXT NOT NULL,
                    company_name TEXT,
                    delisting_date DATE,
                    delisting_reason TEXT,
                    notes TEXT,
                    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(stock_code)
                )
            """)
            
            # Create index for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_delisted_stocks_code ON delisted_stocks(stock_code)")
            
            conn.commit()
            logger.info("Delisted stocks table created/verified in database")
            return True
    except Exception as e:
        logger.error(f"Error creating database table: {e}")
        return False


def parse_delisting_date(date_str):
    """Parse delisting date string to proper DATE format."""
    try:
        if not date_str or pd.isna(date_str):
            return None
        
        # Parse date string in format YYYY-MM-DD
        date_obj = datetime.strptime(str(date_str).strip(), '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError as e:
        logger.warning(f"Invalid date format: {date_str}, error: {e}")
        return None


def insert_delisted_stocks_to_db(df, db_path="krx_stock_data.db"):
    """Insert delisted stocks data into the database."""
    if df.empty:
        logger.warning("No data to insert into database")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Prepare data for insertion
            data_to_insert = []
            for _, row in df.iterrows():
                # Map Korean column names to English
                stock_code = str(row.get('종목코드', '')).strip()
                company_name = str(row.get('회사명', '')).strip()
                delisting_date_str = str(row.get('폐지일자', '')).strip()
                delisting_reason = str(row.get('폐지사유', '')).strip()
                notes = str(row.get('비고', '')).strip()
                
                # Parse date to proper DATE format
                delisting_date = parse_delisting_date(delisting_date_str)
                
                # Only insert if we have a valid stock code
                if stock_code and len(stock_code) == 6 and stock_code.isdigit():
                    data_to_insert.append((
                        stock_code,
                        company_name,
                        delisting_date,
                        delisting_reason,
                        notes
                    ))
            
            if data_to_insert:
                # Clear existing data and insert new data
                conn.execute("DELETE FROM delisted_stocks")
                conn.executemany("""
                    INSERT OR REPLACE INTO delisted_stocks 
                    (stock_code, company_name, delisting_date, delisting_reason, notes)
                    VALUES (?, ?, ?, ?, ?)
                """, data_to_insert)
                
                conn.commit()
                logger.info(f"Inserted {len(data_to_insert)} records into database")
                return True
            else:
                logger.warning("No valid data to insert into database")
                return False
                
    except Exception as e:
        logger.error(f"Error inserting data into database: {e}")
        return False


def get_database_stats(db_path="krx_stock_data.db"):
    """Get statistics for the delisted_stocks table."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM delisted_stocks")
            total_records = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(downloaded_at), MAX(downloaded_at) FROM delisted_stocks")
            date_range = cursor.fetchone()
            
            return {
                'total_records': total_records,
                'min_date': date_range[0],
                'max_date': date_range[1]
            }
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return None


def download_delisted_stocks():
    """
    Download delisted stocks from KRX and return as pandas DataFrame.
    
    Returns:
        pd.DataFrame: Delisted stocks data
        dict: Name dictionary for stock codes
    """
    # KRX API endpoint and parameters
    url = 'https://kind.krx.co.kr/investwarn/delcompany.do'
    
    # Parameters from your reference
    payload = {
        'method': 'searchDelCompanySub',
        'currentPageSize': '3000',
        'pageIndex': '1',
        'orderMode': '2',
        'orderStat': 'D',
        'tabType': '1',
        'searchMode': '1',
        'searchCodeType': 'searchCorpName',
        'repIsuSrtCd': '',
        'forward': 'delcompany_down',
        'searchType': 'marketType',
        'searchCorpNameTmp': '',
        'fromDate': '1999-01-01',  # Start date as requested
        'toDate': datetime.now().strftime('%Y-%m-%d')  # Today's date
    }
    
    # Headers from your reference
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'https://kind.krx.co.kr',
        'Referer': 'https://kind.krx.co.kr/investwarn/delcompany.do?method=searchDelCompanyMain',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'Priority': 'u=0, i'
    }
    
    logger.info(f"Downloading delisted stocks from {payload['fromDate']} to {payload['toDate']}")
    
    try:
        # Make the request
        response = requests.post(url, data=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if response is Excel file
        content_type = response.headers.get('content-type', '')
        if 'application/vnd.ms-excel' not in content_type:
            logger.error(f"Unexpected content type: {content_type}")
            return None, None
        
        logger.info("Successfully downloaded Excel file")
        
        # Read Excel content with converters (as you specified)
        # pd.read_html() reads all tables in HTML, returns list of DataFrames
        # We use converters to ensure 종목코드 (stock code) is read as string
        df_list = pd.read_html(response.content, converters={'종목코드': str})
        
        if not df_list:
            logger.error("No tables found in Excel content")
            return None, None
        
        # Get the first table (usually the main data)
        df = df_list[0]
        
        logger.info(f"Successfully processed Excel content: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Clean the data
        df = df.dropna(how='all')  # Remove empty rows
        
        # Create name dictionary as you showed
        if '종목코드' in df.columns and '회사명' in df.columns:
            name_d = df.set_index('종목코드')['회사명'].to_dict()
            logger.info(f"Created name dictionary with {len(name_d)} entries")
        else:
            logger.warning("Could not find 종목코드 or 회사명 columns for name dictionary")
            name_d = {}
        
        return df, name_d
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None, None


def main():
    """Main function."""
    logger.info("Starting delisted stocks download...")
    
    # Download data
    df, name_d = download_delisted_stocks()
    
    if df is None:
        logger.error("Failed to download delisted stocks data")
        sys.exit(1)
    
    # Create database table
    logger.info("Setting up database...")
    if not create_database_table():
        logger.error("Failed to create database table")
        sys.exit(1)
    
    # Insert data into database
    logger.info("Inserting data into database...")
    if not insert_delisted_stocks_to_db(df):
        logger.error("Failed to insert data into database")
        sys.exit(1)
    
    # Get database statistics
    db_stats = get_database_stats()
    
    # Print summary
    logger.info(f"Downloaded {len(df)} delisted stocks")
    logger.info(f"Name dictionary contains {len(name_d)} entries")
    
    if db_stats:
        logger.info(f"Database contains {db_stats['total_records']} records")
        logger.info(f"Data downloaded at: {db_stats['max_date']}")
    
    # Show sample data
    if len(df) > 0:
        logger.info("Sample data:")
        print(df.head().to_string(index=False))
    
    return df, name_d


if __name__ == '__main__':
    main()