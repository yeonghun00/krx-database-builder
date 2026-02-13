#!/usr/bin/env python3
"""
ETL Pipeline for Market Index Data
Direct Normalization: API → Transform → Load (No Raw Storage)

Handles:
- KOSPI/KOSDAQ indices (kospi_dd_trd, kosdaq_dd_trd)
- Bond indices (bon_dd_trd)
- Government bond market data (kts_bydd_trd)
- Derivatives indices (drvprod_dd_trd)
"""

import sqlite3
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class IndexETLPipeline:
    """ETL pipeline for market index data with direct normalization."""

    def __init__(self, db_path: str = "krx_stock_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._conn = None
        self._create_normalized_tables()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_normalized_tables(self):
        """Create normalized tables only."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # ===========================================
        # MARKET INDICES (KOSPI/KOSDAQ)
        # ===========================================

        # Master table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indices (
                index_code TEXT PRIMARY KEY,
                current_name TEXT NOT NULL,
                index_class TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # History table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                index_code TEXT NOT NULL REFERENCES indices(index_code),
                effective_date TEXT NOT NULL,
                name TEXT,
                index_class TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(index_code, effective_date)
            )
        """)

        # Daily prices
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_daily_prices (
                index_code TEXT NOT NULL REFERENCES indices(index_code),
                date TEXT NOT NULL,
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
                PRIMARY KEY (index_code, date)
            )
        """)

        # ===========================================
        # BOND INDICES
        # ===========================================

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bond_indices (
                index_code TEXT PRIMARY KEY,
                current_name TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bond_index_daily (
                index_code TEXT NOT NULL REFERENCES bond_indices(index_code),
                date TEXT NOT NULL,
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
                PRIMARY KEY (index_code, date)
            )
        """)

        # ===========================================
        # GOVERNMENT BONDS
        # ===========================================

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS govt_bonds (
                issue_code TEXT PRIMARY KEY,
                current_name TEXT NOT NULL,
                market_name TEXT,
                maturity_type TEXT,
                issue_type TEXT,
                is_active BOOLEAN DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS govt_bond_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_code TEXT NOT NULL REFERENCES govt_bonds(issue_code),
                effective_date TEXT NOT NULL,
                name TEXT,
                market_name TEXT,
                maturity_type TEXT,
                issue_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(issue_code, effective_date)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS govt_bond_daily (
                issue_code TEXT NOT NULL REFERENCES govt_bonds(issue_code),
                date TEXT NOT NULL,
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
                PRIMARY KEY (issue_code, date)
            )
        """)

        # ===========================================
        # DERIVATIVES INDICES
        # ===========================================

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deriv_indices (
                index_code TEXT PRIMARY KEY,
                current_name TEXT NOT NULL,
                index_class TEXT,
                is_active BOOLEAN DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deriv_index_daily (
                index_code TEXT NOT NULL REFERENCES deriv_indices(index_code),
                date TEXT NOT NULL,
                closing_index REAL,
                change REAL,
                change_rate REAL,
                opening_index REAL,
                high_index REAL,
                low_index REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (index_code, date)
            )
        """)

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_indices_class ON indices(index_class)",
            "CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily_prices(date)",
            "CREATE INDEX IF NOT EXISTS idx_index_history_date ON index_history(effective_date)",
            "CREATE INDEX IF NOT EXISTS idx_bond_index_daily_date ON bond_index_daily(date)",
            "CREATE INDEX IF NOT EXISTS idx_govt_bond_daily_date ON govt_bond_daily(date)",
            "CREATE INDEX IF NOT EXISTS idx_govt_bond_history_date ON govt_bond_history(effective_date)",
            "CREATE INDEX IF NOT EXISTS idx_deriv_indices_class ON deriv_indices(index_class)",
            "CREATE INDEX IF NOT EXISTS idx_deriv_index_daily_date ON deriv_index_daily(date)",
        ]
        for idx_sql in indexes:
            try:
                cursor.execute(idx_sql)
            except sqlite3.OperationalError:
                pass

        conn.commit()
        self.logger.info("Normalized index tables created")

    def init_tables(self):
        """Alias for backward compatibility."""
        pass  # Tables created in __init__

    @staticmethod
    def _generate_index_code(index_class: str, index_name: str) -> str:
        """Generate unique index code with proper formatting."""
        # Apply formatting rules:
        # 1. Replace · (middle dot) with _
        # 2. Remove % if it exists at the end
        formatted_name = index_name.replace("·", "_").rstrip("%")
        return f"{index_class}_{formatted_name}".replace(" ", "_").replace("/", "_")

    @staticmethod
    def parse_number(value: str) -> Optional[float]:
        if value is None or value == '' or value == '-':
            return None
        try:
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def parse_int(value: str) -> Optional[int]:
        parsed = IndexETLPipeline.parse_number(value)
        return int(parsed) if parsed is not None else None

    def _get_indices_metadata_batch(self, index_codes: Set[str]) -> Dict[str, Dict]:
        """Batch fetch existing index metadata."""
        if not index_codes:
            return {}
        conn = self._get_connection()
        cursor = conn.cursor()
        placeholders = ','.join('?' * len(index_codes))
        cursor.execute(f'''
            SELECT index_code, current_name, index_class
            FROM indices WHERE index_code IN ({placeholders})
        ''', list(index_codes))
        return {row[0]: {'name': row[1], 'index_class': row[2]} for row in cursor.fetchall()}

    def process_market_indices(self, records: List[Dict], index_class: str) -> int:
        """
        Process KOSPI or KOSDAQ index data.
        API → Transform → Load (direct normalization)
        """
        if not records:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        # Extract metadata
        index_codes = set()
        index_updates = {}
        for r in records:
            name = r.get('IDX_NM')
            if not name:
                continue
            code = self._generate_index_code(index_class, name)
            index_codes.add(code)
            index_updates[code] = {
                'name': name,
                'index_class': index_class,
                'effective_date': r.get('BAS_DD')
            }

        # Check existing metadata
        existing = self._get_indices_metadata_batch(index_codes)

        # Upsert master + history
        for code, meta in index_updates.items():
            old = existing.get(code, {})
            if old.get('name') != meta['name'] or old.get('index_class') != meta['index_class']:
                cursor.execute('''
                    INSERT OR REPLACE INTO indices (index_code, current_name, index_class, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (code, meta['name'], meta['index_class']))
                cursor.execute('''
                    INSERT OR IGNORE INTO index_history (index_code, effective_date, name, index_class)
                    VALUES (?, ?, ?, ?)
                ''', (code, meta['effective_date'], meta['name'], meta['index_class']))

        # Insert daily prices
        prices = []
        for r in records:
            name = r.get('IDX_NM')
            if not name:
                continue
            code = self._generate_index_code(index_class, name)
            prices.append((
                code,
                r.get('BAS_DD'),
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
            INSERT OR REPLACE INTO index_daily_prices
            (index_code, date, closing_index, change, change_rate,
             opening_index, high_index, low_index, trading_volume, trading_value, market_cap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, prices)

        conn.commit()
        return len(prices)

    def process_bond_indices(self, records: List[Dict]) -> int:
        """Process bond index data with direct normalization."""
        if not records:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        # Upsert master
        for r in records:
            name = r.get('BND_IDX_GRP_NM')
            if not name:
                continue
            code = f"BOND_{name}".replace(" ", "_")
            cursor.execute('''
                INSERT OR REPLACE INTO bond_indices (index_code, current_name, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (code, name))

        # Insert daily prices
        prices = []
        for r in records:
            name = r.get('BND_IDX_GRP_NM')
            if not name:
                continue
            code = f"BOND_{name}".replace(" ", "_")
            prices.append((
                code,
                r.get('BAS_DD'),
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
            INSERT OR REPLACE INTO bond_index_daily
            (index_code, date, total_return_index, total_return_change,
             net_price_index, net_price_change, zero_reinvest_index, zero_reinvest_change,
             call_reinvest_index, call_reinvest_change, market_price_index, market_price_change,
             avg_duration, avg_convexity, avg_yield)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, prices)

        conn.commit()
        return len(prices)

    def process_government_bonds(self, records: List[Dict]) -> int:
        """Process government bond data with direct normalization."""
        if not records:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        # Extract and check metadata
        bond_updates = {}
        for r in records:
            code = r.get('ISU_CD')
            if not code:
                continue
            bond_updates[code] = {
                'name': r.get('ISU_NM'),
                'market_name': r.get('MKT_NM'),
                'maturity_type': r.get('BND_EXP_TP_NM'),
                'issue_type': r.get('GOVBND_ISU_TP_NM'),
                'effective_date': r.get('BAS_DD')
            }

        # Upsert master + history
        for code, meta in bond_updates.items():
            cursor.execute('''
                INSERT OR REPLACE INTO govt_bonds
                (issue_code, current_name, market_name, maturity_type, issue_type, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (code, meta['name'], meta['market_name'], meta['maturity_type'], meta['issue_type']))
            cursor.execute('''
                INSERT OR IGNORE INTO govt_bond_history
                (issue_code, effective_date, name, market_name, maturity_type, issue_type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (code, meta['effective_date'], meta['name'], meta['market_name'],
                  meta['maturity_type'], meta['issue_type']))

        # Insert daily prices
        prices = []
        for r in records:
            code = r.get('ISU_CD')
            if not code:
                continue
            prices.append((
                code,
                r.get('BAS_DD'),
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
            INSERT OR REPLACE INTO govt_bond_daily
            (issue_code, date, closing_price, price_change, closing_yield,
             opening_price, opening_yield, high_price, high_yield, low_price, low_yield,
             trading_volume, trading_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, prices)

        conn.commit()
        return len(prices)

    def process_derivatives(self, records: List[Dict]) -> int:
        """Process derivatives index data with direct normalization."""
        if not records:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        # Upsert master
        for r in records:
            name = r.get('IDX_NM')
            idx_class = r.get('IDX_CLSS')
            if not name:
                continue
            code = f"DERIV_{idx_class}_{name}".replace(" ", "_").replace("/", "_")
            cursor.execute('''
                INSERT OR REPLACE INTO deriv_indices (index_code, current_name, index_class, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (code, name, idx_class))

        # Insert daily prices
        prices = []
        for r in records:
            name = r.get('IDX_NM')
            idx_class = r.get('IDX_CLSS')
            if not name:
                continue
            code = f"DERIV_{idx_class}_{name}".replace(" ", "_").replace("/", "_")
            prices.append((
                code,
                r.get('BAS_DD'),
                self.parse_number(r.get('CLSPRC_IDX')),
                self.parse_number(r.get('CMPPREVDD_IDX')),
                self.parse_number(r.get('FLUC_RT')),
                self.parse_number(r.get('OPNPRC_IDX')),
                self.parse_number(r.get('HGPRC_IDX')),
                self.parse_number(r.get('LWPRC_IDX'))
            ))

        cursor.executemany("""
            INSERT OR REPLACE INTO deriv_index_daily
            (index_code, date, closing_index, change, change_rate,
             opening_index, high_index, low_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, prices)

        conn.commit()
        return len(prices)

    def process_all_index_data(self, index_data: Dict[str, List[Dict]]) -> Dict[str, int]:
        """Process all index data from API fetch."""
        results = {}

        if 'kospi_index' in index_data:
            results['kospi_index'] = self.process_market_indices(index_data['kospi_index'], 'KOSPI')

        if 'kosdaq_index' in index_data:
            results['kosdaq_index'] = self.process_market_indices(index_data['kosdaq_index'], 'KOSDAQ')

        if 'bond_index' in index_data:
            results['bond_index'] = self.process_bond_indices(index_data['bond_index'])

        if 'govt_bond' in index_data:
            results['govt_bond'] = self.process_government_bonds(index_data['govt_bond'])

        if 'derivatives' in index_data:
            results['derivatives'] = self.process_derivatives(index_data['derivatives'])

        return results

    def check_date_exists(self, date: str, table: str = 'index_daily_prices') -> bool:
        """Check if data exists for a date."""
        table_map = {
            'market_indices': 'index_daily_prices',
            'bond_indices': 'bond_index_daily',
            'government_bonds': 'govt_bond_daily',
            'derivatives_indices': 'deriv_index_daily'
        }
        check_table = table_map.get(table, table)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM {check_table} WHERE date = ? LIMIT 1", (date,))
        return cursor.fetchone() is not None

    def get_existing_dates(self, start_date: str, end_date: str, table: str = 'index_daily_prices') -> Set[str]:
        """Get dates with existing data."""
        table_map = {
            'market_indices': 'index_daily_prices',
            'bond_indices': 'bond_index_daily',
            'government_bonds': 'govt_bond_daily',
            'derivatives_indices': 'deriv_index_daily'
        }
        check_table = table_map.get(table, table)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT DISTINCT date FROM {check_table}
            WHERE date >= ? AND date <= ?
        """, (start_date, end_date))
        return {row[0] for row in cursor.fetchall()}

    def get_stats(self) -> Dict:
        """Get statistics for all tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        stats = {}

        tables = [
            ('indices', 'index_daily_prices'),
            ('bond_indices', 'bond_index_daily'),
            ('govt_bonds', 'govt_bond_daily'),
            ('deriv_indices', 'deriv_index_daily')
        ]

        for master, daily in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {master}")
                master_count = cursor.fetchone()[0]

                cursor.execute(f"SELECT COUNT(*) FROM {daily}")
                daily_count = cursor.fetchone()[0]

                cursor.execute(f"SELECT MIN(date), MAX(date) FROM {daily}")
                date_range = cursor.fetchone()

                cursor.execute(f"SELECT COUNT(DISTINCT date) FROM {daily}")
                unique_dates = cursor.fetchone()[0]

                stats[master] = {
                    'master_count': master_count,
                    'daily_count': daily_count,
                    'unique_dates': unique_dates,
                    'min_date': date_range[0],
                    'max_date': date_range[1]
                }
            except sqlite3.OperationalError:
                stats[master] = {'master_count': 0, 'daily_count': 0, 'unique_dates': 0, 'min_date': None, 'max_date': None}

        return stats

    def get_market_index_names(self, index_class: str = None) -> List[str]:
        """Get list of index names."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if index_class:
            cursor.execute("SELECT DISTINCT current_name FROM indices WHERE index_class = ? ORDER BY current_name", (index_class,))
        else:
            cursor.execute("SELECT DISTINCT current_name FROM indices ORDER BY current_name")
        return [row[0] for row in cursor.fetchall()]

    def get_bond_index_names(self) -> List[str]:
        """Get list of bond index names."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT current_name FROM bond_indices ORDER BY current_name")
        return [row[0] for row in cursor.fetchall()]

    def get_derivative_index_names(self, index_class: str = None) -> List[str]:
        """Get list of derivative index names."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if index_class:
            cursor.execute("SELECT DISTINCT current_name FROM deriv_indices WHERE index_class = ? ORDER BY current_name", (index_class,))
        else:
            cursor.execute("SELECT DISTINCT current_name FROM deriv_indices ORDER BY current_name")
        return [row[0] for row in cursor.fetchall()]

    def validate_data(self) -> Dict:
        """Validate data integrity."""
        conn = self._get_connection()
        cursor = conn.cursor()
        results = {}

        # Check orphaned prices
        cursor.execute("""
            SELECT COUNT(*) FROM index_daily_prices dp
            LEFT JOIN indices i ON dp.index_code = i.index_code
            WHERE i.index_code IS NULL
        """)
        results['orphaned_index_prices'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM index_daily_prices WHERE closing_index IS NULL")
        results['null_closing_index'] = cursor.fetchone()[0]

        results['validation_passed'] = all(v == 0 for v in results.values())
        return results

    def optimize_database(self):
        """Optimize database."""
        conn = self._get_connection()
        conn.execute('VACUUM')
        conn.commit()
        self.logger.info("Database optimized")
