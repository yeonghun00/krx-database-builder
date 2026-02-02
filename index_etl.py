#!/usr/bin/env python3
"""
ETL Pipeline for Market Index Data

Handles:
- KOSPI/KOSDAQ indices (kospi_dd_trd, kosdaq_dd_trd)
- Bond indices (bon_dd_trd)
- Government bond market data (kts_bydd_trd)
- Derivatives indices (drvprod_dd_trd)
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class IndexETLPipeline:
    """ETL pipeline for market index data."""

    def __init__(self, db_path: str = "krx_stock_data.db"):
        """
        Initialize index ETL pipeline.

        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def init_tables(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Market indices (KOSPI/KOSDAQ)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_indices (
                date TEXT NOT NULL,
                index_class TEXT NOT NULL,
                index_name TEXT NOT NULL,
                closing_index REAL,
                change REAL,
                change_rate REAL,
                opening_index REAL,
                high_index REAL,
                low_index REAL,
                trading_volume INTEGER,
                trading_value INTEGER,
                market_cap INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, index_class, index_name)
            )
        """)

        # Bond indices
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bond_indices (
                date TEXT NOT NULL,
                index_group_name TEXT NOT NULL,
                total_return_index REAL,
                total_return_change REAL,
                net_price_index REAL,
                net_price_change REAL,
                zero_reinvest_index REAL,
                zero_reinvest_change REAL,
                call_reinvest_index REAL,
                call_reinvest_change REAL,
                market_price_index REAL,
                market_price_change REAL,
                avg_duration REAL,
                avg_convexity REAL,
                avg_yield REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, index_group_name)
            )
        """)

        # Government bonds
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS government_bonds (
                date TEXT NOT NULL,
                market_name TEXT NOT NULL,
                issue_code TEXT NOT NULL,
                issue_name TEXT NOT NULL,
                maturity_type TEXT,
                issue_type TEXT,
                closing_price REAL,
                price_change REAL,
                closing_yield REAL,
                opening_price REAL,
                opening_yield REAL,
                high_price REAL,
                high_yield REAL,
                low_price REAL,
                low_yield REAL,
                trading_volume INTEGER,
                trading_value INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, issue_code)
            )
        """)

        # Derivatives indices
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS derivatives_indices (
                date TEXT NOT NULL,
                index_class TEXT NOT NULL,
                index_name TEXT NOT NULL,
                closing_index REAL,
                change REAL,
                change_rate REAL,
                opening_index REAL,
                high_index REAL,
                low_index REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, index_class, index_name)
            )
        """)

        conn.commit()
        self._create_indexes()
        self.logger.info("Index tables created successfully")

    def _create_indexes(self):
        """Create performance indexes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_indices_date ON market_indices(date)",
            "CREATE INDEX IF NOT EXISTS idx_market_indices_class ON market_indices(index_class)",
            "CREATE INDEX IF NOT EXISTS idx_market_indices_name ON market_indices(index_name)",
            "CREATE INDEX IF NOT EXISTS idx_bond_indices_date ON bond_indices(date)",
            "CREATE INDEX IF NOT EXISTS idx_bond_indices_group ON bond_indices(index_group_name)",
            "CREATE INDEX IF NOT EXISTS idx_govt_bonds_date ON government_bonds(date)",
            "CREATE INDEX IF NOT EXISTS idx_govt_bonds_maturity ON government_bonds(maturity_type)",
            "CREATE INDEX IF NOT EXISTS idx_govt_bonds_issue_type ON government_bonds(issue_type)",
            "CREATE INDEX IF NOT EXISTS idx_deriv_indices_date ON derivatives_indices(date)",
            "CREATE INDEX IF NOT EXISTS idx_deriv_indices_class ON derivatives_indices(index_class)",
            "CREATE INDEX IF NOT EXISTS idx_deriv_indices_name ON derivatives_indices(index_name)"
        ]

        for idx_sql in indexes:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError as e:
                self.logger.warning(f"Index creation warning: {e}")

        conn.commit()

    @staticmethod
    def parse_number(value: str) -> Optional[float]:
        """
        Parse numeric value from API response.

        Args:
            value: String value from API (may contain commas, dashes, etc.)

        Returns:
            Float value or None if unparseable
        """
        if value is None or value == '' or value == '-':
            return None
        try:
            # Remove commas and convert
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def parse_int(value: str) -> Optional[int]:
        """
        Parse integer value from API response.

        Args:
            value: String value from API

        Returns:
            Integer value or None if unparseable
        """
        parsed = IndexETLPipeline.parse_number(value)
        return int(parsed) if parsed is not None else None

    def process_market_indices(self, records: List[Dict], index_class: str) -> int:
        """
        Process KOSPI or KOSDAQ index data.

        Args:
            records: List of API response records
            index_class: 'KOSPI' or 'KOSDAQ'

        Returns:
            Number of records processed
        """
        if not records:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        rows = []
        for r in records:
            rows.append((
                r.get('BAS_DD'),
                index_class,
                r.get('IDX_NM'),
                self.parse_number(r.get('CLSPRC_IDX')),
                self.parse_number(r.get('CMPPREVDD_IDX')),
                self.parse_number(r.get('FLUC_RT')),
                self.parse_number(r.get('OPNPRC_IDX')),
                self.parse_number(r.get('HGPRC_IDX')),
                self.parse_number(r.get('LWPRC_IDX')),
                self.parse_int(r.get('ACC_TRDVOL')),
                self.parse_int(r.get('ACC_TRDVAL')),
                self.parse_int(r.get('MKTCAP'))
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO market_indices
            (date, index_class, index_name, closing_index, change, change_rate,
             opening_index, high_index, low_index, trading_volume, trading_value, market_cap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

        conn.commit()
        return len(rows)

    def process_bond_indices(self, records: List[Dict]) -> int:
        """
        Process bond index data.

        Args:
            records: List of API response records

        Returns:
            Number of records processed
        """
        if not records:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        rows = []
        for r in records:
            rows.append((
                r.get('BAS_DD'),
                r.get('BND_IDX_GRP_NM'),
                self.parse_number(r.get('TOT_EARNG_IDX')),
                self.parse_number(r.get('TOT_EARNG_IDX_CMPPREVDD')),
                self.parse_number(r.get('NETPRC_IDX')),
                self.parse_number(r.get('NETPRC_IDX_CMPPREVDD')),
                self.parse_number(r.get('ZERO_REINVST_IDX')),
                self.parse_number(r.get('ZERO_REINVST_IDX_CMPPREVDD')),
                self.parse_number(r.get('CALL_REINVST_IDX')),
                self.parse_number(r.get('CALL_REINVST_IDX_CMPPREVDD')),
                self.parse_number(r.get('MKT_PRC_IDX')),
                self.parse_number(r.get('MKT_PRC_IDX_CMPPREVDD')),
                self.parse_number(r.get('AVG_DURATION')),
                self.parse_number(r.get('AVG_CONVEXITY_PRC')),
                self.parse_number(r.get('BND_IDX_AVG_YD'))
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO bond_indices
            (date, index_group_name, total_return_index, total_return_change,
             net_price_index, net_price_change, zero_reinvest_index, zero_reinvest_change,
             call_reinvest_index, call_reinvest_change, market_price_index, market_price_change,
             avg_duration, avg_convexity, avg_yield)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

        conn.commit()
        return len(rows)

    def process_government_bonds(self, records: List[Dict]) -> int:
        """
        Process government bond market data.

        Args:
            records: List of API response records

        Returns:
            Number of records processed
        """
        if not records:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        rows = []
        for r in records:
            rows.append((
                r.get('BAS_DD'),
                r.get('MKT_NM'),
                r.get('ISU_CD'),
                r.get('ISU_NM'),
                r.get('BND_EXP_TP_NM'),
                r.get('GOVBND_ISU_TP_NM'),
                self.parse_number(r.get('CLSPRC')),
                self.parse_number(r.get('CMPPREVDD_PRC')),
                self.parse_number(r.get('CLSPRC_YD')),
                self.parse_number(r.get('OPNPRC')),
                self.parse_number(r.get('OPNPRC_YD')),
                self.parse_number(r.get('HGPRC')),
                self.parse_number(r.get('HGPRC_YD')),
                self.parse_number(r.get('LWPRC')),
                self.parse_number(r.get('LWPRC_YD')),
                self.parse_int(r.get('ACC_TRDVOL')),
                self.parse_int(r.get('ACC_TRDVAL'))
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO government_bonds
            (date, market_name, issue_code, issue_name, maturity_type, issue_type,
             closing_price, price_change, closing_yield, opening_price, opening_yield,
             high_price, high_yield, low_price, low_yield, trading_volume, trading_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

        conn.commit()
        return len(rows)

    def process_derivatives(self, records: List[Dict]) -> int:
        """
        Process derivatives index data.

        Args:
            records: List of API response records

        Returns:
            Number of records processed
        """
        if not records:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        rows = []
        for r in records:
            rows.append((
                r.get('BAS_DD'),
                r.get('IDX_CLSS'),
                r.get('IDX_NM'),
                self.parse_number(r.get('CLSPRC_IDX')),
                self.parse_number(r.get('CMPPREVDD_IDX')),
                self.parse_number(r.get('FLUC_RT')),
                self.parse_number(r.get('OPNPRC_IDX')),
                self.parse_number(r.get('HGPRC_IDX')),
                self.parse_number(r.get('LWPRC_IDX'))
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO derivatives_indices
            (date, index_class, index_name, closing_index, change, change_rate,
             opening_index, high_index, low_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

        conn.commit()
        return len(rows)

    def process_all_index_data(self, index_data: Dict[str, List[Dict]]) -> Dict[str, int]:
        """
        Process all index data types from a single API fetch.

        Args:
            index_data: Dictionary with index_type as key and records as value

        Returns:
            Dictionary with counts for each index type processed
        """
        results = {}

        # Process KOSPI indices
        if 'kospi_index' in index_data:
            results['kospi_index'] = self.process_market_indices(
                index_data['kospi_index'], 'KOSPI'
            )

        # Process KOSDAQ indices
        if 'kosdaq_index' in index_data:
            results['kosdaq_index'] = self.process_market_indices(
                index_data['kosdaq_index'], 'KOSDAQ'
            )

        # Process bond indices
        if 'bond_index' in index_data:
            results['bond_index'] = self.process_bond_indices(
                index_data['bond_index']
            )

        # Process government bonds
        if 'govt_bond' in index_data:
            results['govt_bond'] = self.process_government_bonds(
                index_data['govt_bond']
            )

        # Process derivatives
        if 'derivatives' in index_data:
            results['derivatives'] = self.process_derivatives(
                index_data['derivatives']
            )

        return results

    def check_date_exists(self, date: str, table: str = 'market_indices') -> bool:
        """
        Check if data exists for a specific date in a table.

        Args:
            date: Date in YYYYMMDD format
            table: Table name to check

        Returns:
            True if data exists, False otherwise
        """
        valid_tables = ['market_indices', 'bond_indices', 'government_bonds', 'derivatives_indices']
        if table not in valid_tables:
            raise ValueError(f"Invalid table: {table}")

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM {table} WHERE date = ? LIMIT 1", (date,))
        return cursor.fetchone() is not None

    def get_existing_dates(self, start_date: str, end_date: str, table: str = 'market_indices') -> Set[str]:
        """
        Get all dates that already have data in the specified range.

        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            table: Table name to check

        Returns:
            Set of dates that already have data
        """
        valid_tables = ['market_indices', 'bond_indices', 'government_bonds', 'derivatives_indices']
        if table not in valid_tables:
            raise ValueError(f"Invalid table: {table}")

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT DISTINCT date FROM {table}
            WHERE date >= ? AND date <= ?
            ORDER BY date
        """, (start_date, end_date))
        return {row[0] for row in cursor.fetchall()}

    def get_stats(self) -> Dict:
        """
        Get statistics for all index tables.

        Returns:
            Dictionary with stats for each table
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}
        tables = ['market_indices', 'bond_indices', 'government_bonds', 'derivatives_indices']

        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]

                cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table}")
                row = cursor.fetchone()

                # Get unique dates count
                cursor.execute(f"SELECT COUNT(DISTINCT date) FROM {table}")
                unique_dates = cursor.fetchone()[0]

                stats[table] = {
                    'count': count,
                    'unique_dates': unique_dates,
                    'min_date': row[0],
                    'max_date': row[1]
                }
            except sqlite3.OperationalError:
                stats[table] = {
                    'count': 0,
                    'unique_dates': 0,
                    'min_date': None,
                    'max_date': None
                }

        return stats

    def get_market_index_names(self, index_class: str = None) -> List[str]:
        """
        Get list of unique index names.

        Args:
            index_class: Optional filter by 'KOSPI' or 'KOSDAQ'

        Returns:
            List of index names
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if index_class:
            cursor.execute("""
                SELECT DISTINCT index_name FROM market_indices
                WHERE index_class = ?
                ORDER BY index_name
            """, (index_class,))
        else:
            cursor.execute("""
                SELECT DISTINCT index_name FROM market_indices
                ORDER BY index_name
            """)

        return [row[0] for row in cursor.fetchall()]

    def get_bond_index_names(self) -> List[str]:
        """Get list of unique bond index group names."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT index_group_name FROM bond_indices
            ORDER BY index_group_name
        """)
        return [row[0] for row in cursor.fetchall()]

    def get_derivative_index_names(self, index_class: str = None) -> List[str]:
        """
        Get list of unique derivative index names.

        Args:
            index_class: Optional filter by index class (선물지수, 옵션지수, 전략지수)

        Returns:
            List of index names
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if index_class:
            cursor.execute("""
                SELECT DISTINCT index_name FROM derivatives_indices
                WHERE index_class = ?
                ORDER BY index_name
            """, (index_class,))
        else:
            cursor.execute("""
                SELECT DISTINCT index_name FROM derivatives_indices
                ORDER BY index_name
            """)

        return [row[0] for row in cursor.fetchall()]

    def validate_data(self) -> Dict:
        """
        Validate data integrity across all tables.

        Returns:
            Dictionary with validation results
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        results = {}

        # Check for NULL closing values in market indices
        cursor.execute("""
            SELECT COUNT(*) FROM market_indices WHERE closing_index IS NULL
        """)
        results['market_indices_null_closing'] = cursor.fetchone()[0]

        # Check for NULL yields in government bonds (for benchmark issues)
        cursor.execute("""
            SELECT COUNT(*) FROM government_bonds
            WHERE closing_yield IS NULL AND issue_type = '지표'
        """)
        results['govt_bonds_null_yield'] = cursor.fetchone()[0]

        # Check for duplicate entries
        for table in ['market_indices', 'bond_indices', 'derivatives_indices']:
            cursor.execute(f"""
                SELECT COUNT(*) FROM (
                    SELECT date, COUNT(*) as cnt
                    FROM {table}
                    GROUP BY date
                    HAVING cnt > 100
                )
            """)
            results[f'{table}_date_anomalies'] = cursor.fetchone()[0]

        results['validation_passed'] = all(v == 0 for v in results.values())
        return results

    def optimize_database(self):
        """Optimize database performance by rebuilding indexes and vacuuming."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Rebuild all indexes
        tables = ['market_indices', 'bond_indices', 'government_bonds', 'derivatives_indices']
        for table in tables:
            try:
                cursor.execute(f"REINDEX {table}")
            except sqlite3.OperationalError:
                pass

        # Vacuum database
        conn.execute('VACUUM')
        conn.commit()
        self.logger.info("Database optimization completed")
