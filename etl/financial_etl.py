"""
Financial Statement Data Loader

Parses and loads BS (재무상태표), PL (손익계산서), CF (현금흐름표)
from raw zip files into SQLite database.

Implements 45/90 rule to prevent look-ahead bias.
"""

import sqlite3
import pandas as pd
import numpy as np
import zipfile
import io
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Item Code Mapping (ifrs_ → ifrs-full_)
# ============================================
ITEM_CODE_MAPPING = {
    # Balance Sheet
    'ifrs_Assets': 'ifrs-full_Assets',
    'ifrs_Equity': 'ifrs-full_Equity',
    'ifrs_Liabilities': 'ifrs-full_Liabilities',
    'ifrs_CurrentAssets': 'ifrs-full_CurrentAssets',
    'ifrs_CurrentLiabilities': 'ifrs-full_CurrentLiabilities',
    'ifrs_NoncurrentAssets': 'ifrs-full_NoncurrentAssets',
    'ifrs_NoncurrentLiabilities': 'ifrs-full_NoncurrentLiabilities',
    'ifrs_CashAndCashEquivalents': 'ifrs-full_CashAndCashEquivalents',
    'ifrs_Inventories': 'ifrs-full_Inventories',
    'ifrs_PropertyPlantAndEquipment': 'ifrs-full_PropertyPlantAndEquipment',
    'ifrs_RetainedEarnings': 'ifrs-full_RetainedEarnings',
    'ifrs_IssuedCapital': 'ifrs-full_IssuedCapital',

    # Income Statement
    'ifrs_Revenue': 'ifrs-full_Revenue',
    'ifrs_CostOfSales': 'ifrs-full_CostOfSales',
    'ifrs_GrossProfit': 'ifrs-full_GrossProfit',
    'ifrs_ProfitLoss': 'ifrs-full_ProfitLoss',
    'ifrs_ProfitLossBeforeTax': 'ifrs-full_ProfitLossBeforeTax',
    'ifrs_IncomeTaxExpenseContinuingOperations': 'ifrs-full_IncomeTaxExpenseContinuingOperations',
    'ifrs_FinanceIncome': 'ifrs-full_FinanceIncome',
    'ifrs_FinanceCosts': 'ifrs-full_FinanceCosts',

    # Cash Flow
    'ifrs_CashFlowsFromUsedInOperatingActivities': 'ifrs-full_CashFlowsFromUsedInOperatingActivities',
    'ifrs_CashFlowsFromUsedInInvestingActivities': 'ifrs-full_CashFlowsFromUsedInInvestingActivities',
    'ifrs_CashFlowsFromUsedInFinancingActivities': 'ifrs-full_CashFlowsFromUsedInFinancingActivities',
}


def normalize_item_code(code: str) -> str:
    """Normalize item code to ifrs-full_ format."""
    if code in ITEM_CODE_MAPPING:
        return ITEM_CODE_MAPPING[code]
    return code


def parse_number(value) -> Optional[float]:
    """Parse Korean number format (with commas) to float."""
    if pd.isna(value) or value == '' or value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    # Remove commas and whitespace
    cleaned = str(value).replace(',', '').strip()

    if cleaned == '' or cleaned == '-':
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def get_available_date(fiscal_date: str, fiscal_month: int) -> str:
    """
    Calculate when financial data becomes publicly available.
    Implements 45/90 rule for look-ahead bias prevention.

    Args:
        fiscal_date: Period end date (YYYY-MM-DD)
        fiscal_month: Company's fiscal year end month (결산월)

    Returns:
        First date the data can be used in trading (YYYYMMDD)
    """
    year, month, day = fiscal_date.split('-')
    year, month = int(year), int(month)

    # Standard December fiscal year
    if fiscal_month == 12:
        if month == 3:    # Q1 -> Available May 16
            return f"{year}0516"
        elif month == 6:  # Q2 -> Available Aug 16
            return f"{year}0816"
        elif month == 9:  # Q3 -> Available Nov 15
            return f"{year}1115"
        elif month == 12: # Q4 -> Available Apr 1 next year
            return f"{year + 1}0401"

    # Non-standard fiscal year (March, June, etc.)
    # Calculate based on quarter offset from fiscal month
    quarters_from_fy = ((month - fiscal_month) % 12) // 3

    if quarters_from_fy == 0:  # Q4 (annual) -> 90 days
        # Add ~90 days
        if month <= 9:
            return f"{year}{month + 3:02d}01"
        else:
            return f"{year + 1}{(month + 3 - 12):02d}01"
    else:  # Q1/Q2/Q3 -> 45 days
        # Add ~45 days
        if month <= 10:
            return f"{year}{month + 2:02d}16"
        else:
            return f"{year + 1}{(month + 2 - 12):02d}16"

    # Fallback
    return f"{year}{month:02d}16"


def extract_stock_code(raw_code: str) -> str:
    """Extract stock code from [XXXXXX] format."""
    match = re.search(r'\[(\d{6})\]', str(raw_code))
    if match:
        return match.group(1)
    return str(raw_code).strip()


def extract_consolidation_type(statement_type: str) -> str:
    """Extract 별도/연결 from statement type string."""
    if '연결' in str(statement_type):
        return '연결'
    elif '별도' in str(statement_type):
        return '별도'
    return '별도'  # Default


class FinancialDataLoader:
    """Loads financial statement data from zip files into database."""

    def __init__(self, db_path: str, raw_data_dir: str):
        """
        Initialize loader.

        Args:
            db_path: Path to SQLite database
            raw_data_dir: Path to raw_financial/ directory with zip files
        """
        self.db_path = Path(db_path)
        self.raw_data_dir = Path(raw_data_dir)
        self.conn = None

    def connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(self.db_path)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def create_tables(self):
        """Create financial data tables if not exist."""
        cursor = self.conn.cursor()

        # Financial periods (metadata)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_periods (
                id INTEGER PRIMARY KEY,
                stock_code TEXT NOT NULL,
                company_name TEXT,
                market_type TEXT,
                industry_code TEXT,
                industry_name TEXT,
                fiscal_month INTEGER,
                fiscal_date TEXT NOT NULL,
                available_date TEXT NOT NULL,
                report_type TEXT,
                consolidation_type TEXT,
                currency TEXT DEFAULT 'KRW',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(stock_code, fiscal_date, consolidation_type)
            )
        """)

        # BS/CF items (simple structure)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_items_bs_cf (
                id INTEGER PRIMARY KEY,
                period_id INTEGER NOT NULL REFERENCES financial_periods(id),
                statement_type TEXT NOT NULL,
                item_code TEXT NOT NULL,
                item_code_normalized TEXT,
                item_name TEXT NOT NULL,
                amount_current REAL,
                amount_prev REAL,
                amount_prev2 REAL,
                UNIQUE(period_id, statement_type, item_code)
            )
        """)

        # PL items (complex structure with quarterly breakdown)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_items_pl (
                id INTEGER PRIMARY KEY,
                period_id INTEGER NOT NULL REFERENCES financial_periods(id),
                item_code TEXT NOT NULL,
                item_code_normalized TEXT,
                item_name TEXT NOT NULL,
                amount_current_qtr REAL,
                amount_current_ytd REAL,
                amount_prev_qtr REAL,
                amount_prev_ytd REAL,
                amount_prev_year REAL,
                amount_prev2_year REAL,
                UNIQUE(period_id, item_code)
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fin_periods_stock_date ON financial_periods(stock_code, fiscal_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fin_periods_available ON financial_periods(available_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fin_bs_cf_period ON financial_items_bs_cf(period_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fin_pl_period ON financial_items_pl(period_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fin_bs_cf_code ON financial_items_bs_cf(item_code_normalized)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fin_pl_code ON financial_items_pl(item_code_normalized)")

        self.conn.commit()
        logger.info("Created financial data tables")

    def read_zip_file(self, zip_path: Path) -> pd.DataFrame:
        """
        Read financial data from zip file.

        Args:
            zip_path: Path to zip file

        Returns:
            DataFrame with financial data
        """
        dfs = []

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for file_name in zf.namelist():
                if file_name.endswith('.txt'):
                    with zf.open(file_name) as f:
                        # Read with CP949 encoding
                        content = f.read().decode('cp949', errors='replace')
                        df = pd.read_csv(
                            io.StringIO(content),
                            sep='\t',
                            dtype=str,
                            on_bad_lines='skip'
                        )
                        dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def get_or_create_period(self, row: pd.Series, cursor) -> Optional[int]:
        """Get or create financial_periods record, return period_id."""
        stock_code = extract_stock_code(row.iloc[1])
        company_name = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else None
        market_type = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else None
        industry_code = str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else None
        industry_name = str(row.iloc[5]).strip() if pd.notna(row.iloc[5]) else None
        fiscal_month = int(row.iloc[6]) if pd.notna(row.iloc[6]) else 12
        fiscal_date = str(row.iloc[7]).strip() if pd.notna(row.iloc[7]) else None
        report_type = str(row.iloc[8]).strip() if pd.notna(row.iloc[8]) else None
        currency = str(row.iloc[9]).strip() if pd.notna(row.iloc[9]) else 'KRW'
        consolidation_type = extract_consolidation_type(row.iloc[0])

        if not fiscal_date or not stock_code:
            return None

        available_date = get_available_date(fiscal_date, fiscal_month)

        # Try to get existing period
        cursor.execute("""
            SELECT id FROM financial_periods
            WHERE stock_code = ? AND fiscal_date = ? AND consolidation_type = ?
        """, (stock_code, fiscal_date, consolidation_type))

        result = cursor.fetchone()
        if result:
            return result[0]

        # Create new period
        cursor.execute("""
            INSERT INTO financial_periods
            (stock_code, company_name, market_type, industry_code, industry_name,
             fiscal_month, fiscal_date, available_date, report_type, consolidation_type, currency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (stock_code, company_name, market_type, industry_code, industry_name,
              fiscal_month, fiscal_date, available_date, report_type, consolidation_type, currency))

        return cursor.lastrowid

    def parse_bs_cf(self, df: pd.DataFrame, statement_type: str) -> List[dict]:
        """Parse Balance Sheet or Cash Flow data."""
        results = []
        cursor = self.conn.cursor()

        # Group by unique period (stock + date + consolidation)
        period_cache = {}

        for idx, row in df.iterrows():
            # Skip header row
            if idx == 0 or '항목코드' in str(row.iloc[10]):
                continue

            item_code = str(row.iloc[10]).strip() if pd.notna(row.iloc[10]) else ''
            item_name = str(row.iloc[11]).strip() if pd.notna(row.iloc[11]) else ''

            # Skip abstract/header items
            if not item_code or '[abstract]' in item_name.lower():
                continue

            # Get or create period
            period_key = (
                extract_stock_code(row.iloc[1]),
                str(row.iloc[7]).strip(),
                extract_consolidation_type(row.iloc[0])
            )

            if period_key not in period_cache:
                period_id = self.get_or_create_period(row, cursor)
                period_cache[period_key] = period_id
            else:
                period_id = period_cache[period_key]

            if not period_id:
                continue

            # Parse values (columns 12, 13, 14 for BS; 12, 14, 15 for CF)
            if statement_type == 'CF':
                amount_current = parse_number(row.iloc[12]) if len(row) > 12 else None
                amount_prev = parse_number(row.iloc[14]) if len(row) > 14 else None
                amount_prev2 = parse_number(row.iloc[15]) if len(row) > 15 else None
            else:  # BS
                amount_current = parse_number(row.iloc[12]) if len(row) > 12 else None
                amount_prev = parse_number(row.iloc[13]) if len(row) > 13 else None
                amount_prev2 = parse_number(row.iloc[14]) if len(row) > 14 else None

            results.append({
                'period_id': period_id,
                'statement_type': statement_type,
                'item_code': item_code,
                'item_code_normalized': normalize_item_code(item_code),
                'item_name': item_name,
                'amount_current': amount_current,
                'amount_prev': amount_prev,
                'amount_prev2': amount_prev2,
            })

        self.conn.commit()
        return results

    def parse_pl(self, df: pd.DataFrame) -> List[dict]:
        """Parse Income Statement data."""
        results = []
        cursor = self.conn.cursor()
        period_cache = {}

        # Detect if quarterly or annual based on report type
        is_quarterly = False
        if len(df) > 0:
            report_type = str(df.iloc[1, 8]) if len(df) > 1 else ''
            is_quarterly = '분기' in report_type or '반기' in report_type

        for idx, row in df.iterrows():
            # Skip header row
            if idx == 0 or '항목코드' in str(row.iloc[10]):
                continue

            item_code = str(row.iloc[10]).strip() if pd.notna(row.iloc[10]) else ''
            item_name = str(row.iloc[11]).strip() if pd.notna(row.iloc[11]) else ''

            if not item_code or '[abstract]' in item_name.lower():
                continue

            # Get or create period
            period_key = (
                extract_stock_code(row.iloc[1]),
                str(row.iloc[7]).strip(),
                extract_consolidation_type(row.iloc[0])
            )

            if period_key not in period_cache:
                period_id = self.get_or_create_period(row, cursor)
                period_cache[period_key] = period_id
            else:
                period_id = period_cache[period_key]

            if not period_id:
                continue

            # Parse values based on report type
            if is_quarterly:
                # Quarterly: cols 12-17 (당기3개월, 당기누적, 전기3개월, 전기누적, 전기연간, 전전기연간)
                amount_current_qtr = parse_number(row.iloc[12]) if len(row) > 12 else None
                amount_current_ytd = parse_number(row.iloc[13]) if len(row) > 13 else None
                amount_prev_qtr = parse_number(row.iloc[14]) if len(row) > 14 else None
                amount_prev_ytd = parse_number(row.iloc[15]) if len(row) > 15 else None
                amount_prev_year = parse_number(row.iloc[16]) if len(row) > 16 else None
                amount_prev2_year = parse_number(row.iloc[17]) if len(row) > 17 else None
            else:
                # Annual: cols 13 (당기), 16 (전기), 17 (전전기) - with empty cols
                amount_current_qtr = None
                amount_current_ytd = parse_number(row.iloc[13]) if len(row) > 13 else None
                amount_prev_qtr = None
                amount_prev_ytd = None
                amount_prev_year = parse_number(row.iloc[16]) if len(row) > 16 else None
                amount_prev2_year = parse_number(row.iloc[17]) if len(row) > 17 else None

            results.append({
                'period_id': period_id,
                'item_code': item_code,
                'item_code_normalized': normalize_item_code(item_code),
                'item_name': item_name,
                'amount_current_qtr': amount_current_qtr,
                'amount_current_ytd': amount_current_ytd,
                'amount_prev_qtr': amount_prev_qtr,
                'amount_prev_ytd': amount_prev_ytd,
                'amount_prev_year': amount_prev_year,
                'amount_prev2_year': amount_prev2_year,
            })

        self.conn.commit()
        return results

    def insert_bs_cf_items(self, items: List[dict]):
        """Insert BS/CF items into database."""
        cursor = self.conn.cursor()

        for item in items:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO financial_items_bs_cf
                    (period_id, statement_type, item_code, item_code_normalized,
                     item_name, amount_current, amount_prev, amount_prev2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item['period_id'],
                    item['statement_type'],
                    item['item_code'],
                    item['item_code_normalized'],
                    item['item_name'],
                    item['amount_current'],
                    item['amount_prev'],
                    item['amount_prev2'],
                ))
            except Exception as e:
                logger.warning(f"Error inserting BS/CF item: {e}")

        self.conn.commit()

    def insert_pl_items(self, items: List[dict]):
        """Insert PL items into database."""
        cursor = self.conn.cursor()

        for item in items:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO financial_items_pl
                    (period_id, item_code, item_code_normalized, item_name,
                     amount_current_qtr, amount_current_ytd, amount_prev_qtr,
                     amount_prev_ytd, amount_prev_year, amount_prev2_year)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item['period_id'],
                    item['item_code'],
                    item['item_code_normalized'],
                    item['item_name'],
                    item['amount_current_qtr'],
                    item['amount_current_ytd'],
                    item['amount_prev_qtr'],
                    item['amount_prev_ytd'],
                    item['amount_prev_year'],
                    item['amount_prev2_year'],
                ))
            except Exception as e:
                logger.warning(f"Error inserting PL item: {e}")

        self.conn.commit()

    def process_file(self, zip_path: Path) -> Tuple[int, int]:
        """
        Process a single zip file.

        Returns:
            (periods_count, items_count)
        """
        # Determine statement type from filename
        filename = zip_path.name

        if '_BS_' in filename:
            statement_type = 'BS'
        elif '_PL_' in filename:
            statement_type = 'PL'
        elif '_CF_' in filename:
            statement_type = 'CF'
        elif '_CE_' in filename:
            logger.info(f"Skipping CE file: {filename}")
            return (0, 0)
        else:
            logger.warning(f"Unknown statement type: {filename}")
            return (0, 0)

        # Read and parse
        df = self.read_zip_file(zip_path)

        if df.empty:
            logger.warning(f"Empty dataframe for {filename}")
            return (0, 0)

        if statement_type == 'PL':
            items = self.parse_pl(df)
            self.insert_pl_items(items)
        else:  # BS or CF
            items = self.parse_bs_cf(df, statement_type)
            self.insert_bs_cf_items(items)

        # Count periods created
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM financial_periods")
        periods_count = cursor.fetchone()[0]

        return (periods_count, len(items))

    def process_all(self, pattern: str = "*.zip") -> Dict[str, int]:
        """
        Process all zip files in raw_data_dir.

        Returns:
            Statistics dict
        """
        zip_files = sorted(self.raw_data_dir.glob(pattern))

        # Filter out CE files
        zip_files = [f for f in zip_files if '_CE_' not in f.name]

        logger.info(f"Found {len(zip_files)} files to process (excluding CE)")

        stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'total_items': 0,
            'errors': 0,
        }

        for i, zip_path in enumerate(zip_files):
            try:
                periods, items = self.process_file(zip_path)
                stats['files_processed'] += 1
                stats['total_items'] += items

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(zip_files)} files...")

            except Exception as e:
                logger.error(f"Error processing {zip_path.name}: {e}")
                stats['errors'] += 1

        # Final counts
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM financial_periods")
        stats['total_periods'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM financial_items_bs_cf")
        stats['bs_cf_items'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM financial_items_pl")
        stats['pl_items'] = cursor.fetchone()[0]

        return stats


def load_financial_data(db_path: str, raw_data_dir: str) -> Dict[str, int]:
    """
    Main entry point to load financial data.

    Args:
        db_path: Path to SQLite database
        raw_data_dir: Path to raw_financial/ directory

    Returns:
        Statistics dict
    """
    loader = FinancialDataLoader(db_path, raw_data_dir)
    loader.connect()

    try:
        loader.create_tables()
        stats = loader.process_all()

        logger.info("=" * 50)
        logger.info("Financial Data Loading Complete!")
        logger.info(f"  Files processed: {stats['files_processed']}")
        logger.info(f"  Total periods: {stats['total_periods']}")
        logger.info(f"  BS/CF items: {stats['bs_cf_items']}")
        logger.info(f"  PL items: {stats['pl_items']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info("=" * 50)

        return stats

    finally:
        loader.close()


if __name__ == "__main__":
    import sys
    import os

    # Get script directory for relative paths
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Default paths (relative to script location)
    db_path = script_dir / "krx_stock_data.db"
    raw_data_dir = script_dir / "raw_financial"

    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        raw_data_dir = Path(sys.argv[2])

    load_financial_data(str(db_path), str(raw_data_dir))
