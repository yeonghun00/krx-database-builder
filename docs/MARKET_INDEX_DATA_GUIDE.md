# Market Index Data Integration Guide

This guide explains how to add KOSPI/KOSDAQ indices, bond indices, government bond market data, and derivatives data to the AlgoStock database.

## Table of Contents
1. [Overview](#overview)
2. [New Data Sources](#new-data-sources)
3. [Database Schema Design](#database-schema-design)
4. [API Integration](#api-integration)
5. [ETL Implementation](#etl-implementation)
6. [Backfill & Update Procedures](#backfill--update-procedures)
7. [Usage Examples](#usage-examples)

---

## Overview

### Current Architecture
- **Database**: SQLite (`krx_stock_data.db`)
- **Pattern**: Direct normalization (API → Transform → Load)
- **Upsert Strategy**: `INSERT OR REPLACE` with composite primary keys
- **Rate Limiting**: 0.5s (backfill) / 1.0s (daily updates)

### New Data Categories

| Category | API Endpoint | Description |
|----------|-------------|-------------|
| KOSPI Index | `kospi_dd_trd` | KOSPI index family daily data |
| KOSDAQ Index | `kosdaq_dd_trd` | KOSDAQ index family daily data |
| Bond Index | `bon_dd_trd` | KRX bond indices |
| Government Bond | `kts_bydd_trd` | Government bond market data |
| Derivatives | `drvprod_dd_trd` | Futures/options indices |

---

## New Data Sources

### 1. KOSPI Index Data (`kospi_dd_trd`)

**API Response Fields:**
| Field | Korean | Description | Type |
|-------|--------|-------------|------|
| `BAS_DD` | 기준일자 | Base date (YYYYMMDD) | string |
| `IDX_CLSS` | 지수분류 | Index classification (KOSPI) | string |
| `IDX_NM` | 지수명 | Index name | string |
| `CLSPRC_IDX` | 종가 | Closing price index | string |
| `CMPPREVDD_IDX` | 대비 | Change from previous day | string |
| `FLUC_RT` | 등락률 | Fluctuation rate (%) | string |
| `OPNPRC_IDX` | 시가 | Opening price index | string |
| `HGPRC_IDX` | 고가 | High price index | string |
| `LWPRC_IDX` | 저가 | Low price index | string |
| `ACC_TRDVOL` | 거래량 | Accumulated trading volume | string |
| `ACC_TRDVAL` | 거래대금 | Accumulated trading value | string |
| `MKTCAP` | 시가총액 | Market capitalization | string |

**Sample Indices:**
- 코스피 100
- 코스피 200 건설
- 코스피 200 철강/소재
- 코스피 200 헬스케어
- 금융, 증권, 종이·목재, 비금속, 의료·정밀기기

### 2. KOSDAQ Index Data (`kosdaq_dd_trd`)

**Same fields as KOSPI** with `IDX_CLSS` = "KOSDAQ"

**Sample Indices:**
- 코스닥 (외국주포함)
- 코스닥 150
- 코스닥 150 정보기술
- 코스닥 150 헬스케어
- 코스닥 150 커뮤니케이션서비스
- 코스닥 150 소재/산업재/소비재

### 3. Bond Index Data (`bon_dd_trd`)

**API Response Fields:**
| Field | Korean | Description | Type |
|-------|--------|-------------|------|
| `BAS_DD` | 기준일자 | Base date | string |
| `BND_IDX_GRP_NM` | 채권지수그룹명 | Bond index group name | string |
| `TOT_EARNG_IDX` | 총수익지수 | Total return index | string |
| `TOT_EARNG_IDX_CMPPREVDD` | 총수익지수대비 | Total return change | string |
| `NETPRC_IDX` | 순가격지수 | Net price index | string |
| `NETPRC_IDX_CMPPREVDD` | 순가격지수대비 | Net price change | string |
| `ZERO_REINVST_IDX` | 제로재투자지수 | Zero reinvestment index | string |
| `ZERO_REINVST_IDX_CMPPREVDD` | 제로재투자지수대비 | Zero reinvest change | string |
| `CALL_REINVST_IDX` | 콜재투자지수 | Call reinvestment index | string |
| `CALL_REINVST_IDX_CMPPREVDD` | 콜재투자지수대비 | Call reinvest change | string |
| `MKT_PRC_IDX` | 시장가격지수 | Market price index | string |
| `MKT_PRC_IDX_CMPPREVDD` | 시장가격지수대비 | Market price change | string |
| `AVG_DURATION` | 평균듀레이션 | Average duration | string |
| `AVG_CONVEXITY_PRC` | 평균볼록성 | Average convexity | string |
| `BND_IDX_AVG_YD` | 평균수익률 | Average yield | string |

**Sample Indices:**
- KRX 채권지수
- KTB 지수
- 국고채프라임지수

### 4. Government Bond Market (`kts_bydd_trd`)

**API Response Fields:**
| Field | Korean | Description | Type |
|-------|--------|-------------|------|
| `BAS_DD` | 기준일자 | Base date | string |
| `MKT_NM` | 시장명 | Market name | string |
| `ISU_CD` | 종목코드 | Issue code | string |
| `ISU_NM` | 종목명 | Issue name | string |
| `BND_EXP_TP_NM` | 만기구분 | Maturity type (3/5/10/20/30Y) | string |
| `GOVBND_ISU_TP_NM` | 발행구분 | Issue type (지표/경과) | string |
| `CLSPRC` | 종가 | Closing price | string |
| `CMPPREVDD_PRC` | 대비 | Price change | string |
| `CLSPRC_YD` | 종가수익률 | Closing yield | string |
| `OPNPRC` | 시가 | Opening price | string |
| `OPNPRC_YD` | 시가수익률 | Opening yield | string |
| `HGPRC` | 고가 | High price | string |
| `HGPRC_YD` | 고가수익률 | High yield | string |
| `LWPRC` | 저가 | Low price | string |
| `LWPRC_YD` | 저가수익률 | Low yield | string |
| `ACC_TRDVOL` | 거래량 | Trading volume | string |
| `ACC_TRDVAL` | 거래대금 | Trading value | string |

**Maturity Types:**
- 3년물 (3-year)
- 5년물 (5-year)
- 10년물 (10-year)
- 20년물 (20-year)
- 30년물 (30-year)

### 5. Derivatives Data (`drvprod_dd_trd`)

**API Response Fields:**
| Field | Korean | Description | Type |
|-------|--------|-------------|------|
| `BAS_DD` | 기준일자 | Base date | string |
| `IDX_CLSS` | 지수분류 | Index class (선물/옵션/전략) | string |
| `IDX_NM` | 지수명 | Index name | string |
| `CLSPRC_IDX` | 종가 | Closing index | string |
| `CMPPREVDD_IDX` | 대비 | Change | string |
| `FLUC_RT` | 등락률 | Fluctuation rate | string |
| `OPNPRC_IDX` | 시가 | Opening index | string |
| `HGPRC_IDX` | 고가 | High index | string |
| `LWPRC_IDX` | 저가 | Low index | string |

**Index Classes:**
- 선물지수 (Futures Index): 코스피 200 선물지수
- 옵션지수 (Options Index): 코스피 200 변동성지수 (VIX)
- 전략지수 (Strategy Index): 인버스지수, TR지수, 배당지수

---

## Database Schema Design

### Option 1: Normalized Tables (Recommended)

```sql
-- KOSPI/KOSDAQ Index Daily Data
CREATE TABLE IF NOT EXISTS market_indices (
    date TEXT NOT NULL,
    index_class TEXT NOT NULL,      -- 'KOSPI', 'KOSDAQ'
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
);

CREATE INDEX idx_market_indices_date ON market_indices(date);
CREATE INDEX idx_market_indices_class ON market_indices(index_class);
CREATE INDEX idx_market_indices_name ON market_indices(index_name);

-- Bond Index Daily Data
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
);

CREATE INDEX idx_bond_indices_date ON bond_indices(date);

-- Government Bond Market Daily Data
CREATE TABLE IF NOT EXISTS government_bonds (
    date TEXT NOT NULL,
    market_name TEXT NOT NULL,
    issue_code TEXT NOT NULL,
    issue_name TEXT NOT NULL,
    maturity_type TEXT,             -- '3', '5', '10', '20', '30'
    issue_type TEXT,                -- '지표', '경과'
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
);

CREATE INDEX idx_govt_bonds_date ON government_bonds(date);
CREATE INDEX idx_govt_bonds_maturity ON government_bonds(maturity_type);
CREATE INDEX idx_govt_bonds_issue_type ON government_bonds(issue_type);

-- Derivatives Index Daily Data
CREATE TABLE IF NOT EXISTS derivatives_indices (
    date TEXT NOT NULL,
    index_class TEXT NOT NULL,      -- '선물지수', '옵션지수', '전략지수'
    index_name TEXT NOT NULL,
    closing_index REAL,
    change REAL,
    change_rate REAL,
    opening_index REAL,
    high_index REAL,
    low_index REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, index_class, index_name)
);

CREATE INDEX idx_deriv_indices_date ON derivatives_indices(date);
CREATE INDEX idx_deriv_indices_class ON derivatives_indices(index_class);
```

### Option 2: Single Unified Table

```sql
-- All market data in one table with type column
CREATE TABLE IF NOT EXISTS market_data_indices (
    date TEXT NOT NULL,
    data_type TEXT NOT NULL,        -- 'KOSPI', 'KOSDAQ', 'BOND', 'GOVT_BOND', 'DERIVATIVES'
    category TEXT NOT NULL,         -- Index class or group name
    name TEXT NOT NULL,             -- Index or issue name
    code TEXT,                      -- Issue code (for govt bonds)
    closing_value REAL,
    change_value REAL,
    change_rate REAL,
    opening_value REAL,
    high_value REAL,
    low_value REAL,
    volume INTEGER,
    value INTEGER,
    market_cap INTEGER,
    -- Bond-specific fields
    duration REAL,
    convexity REAL,
    yield REAL,
    -- Govt bond specific
    maturity_type TEXT,
    issue_type TEXT,
    metadata JSON,                  -- Additional fields as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, data_type, category, name)
);
```

---

## API Integration

### Add New Endpoints to `krx_api.py`

```python
# Add to market_endpoints dictionary
index_endpoints = {
    'kospi_index': 'https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd',
    'kosdaq_index': 'https://data-dbg.krx.co.kr/svc/apis/idx/kosdaq_dd_trd',
    'bond_index': 'https://data-dbg.krx.co.kr/svc/apis/bon/bon_dd_trd',
    'govt_bond': 'https://data-dbg.krx.co.kr/svc/apis/bon/kts_bydd_trd',
    'derivatives': 'https://data-dbg.krx.co.kr/svc/apis/idx/drvprod_dd_trd'
}

class KRXAPI:
    def __init__(self, auth_key: str, config: dict = None):
        self.auth_key = auth_key
        self.config = config or {}
        self.base_url = "https://data-dbg.krx.co.kr/svc/apis"

        # Existing endpoints
        self.market_endpoints = {
            'kospi': f'{self.base_url}/sto/stk_bydd_trd',
            'kosdaq': f'{self.base_url}/sto/ksq_bydd_trd',
            'kodex': f'{self.base_url}/sto/knx_bydd_trd'
        }

        # NEW: Index endpoints
        self.index_endpoints = {
            'kospi_index': f'{self.base_url}/idx/kospi_dd_trd',
            'kosdaq_index': f'{self.base_url}/idx/kosdaq_dd_trd',
            'bond_index': f'{self.base_url}/bon/bon_dd_trd',
            'govt_bond': f'{self.base_url}/bon/kts_bydd_trd',
            'derivatives': f'{self.base_url}/idx/drvprod_dd_trd'
        }

    def fetch_index_data(self, date: str, index_type: str) -> list:
        """
        Fetch index data for a specific date.

        Args:
            date: Date in YYYYMMDD format
            index_type: One of 'kospi_index', 'kosdaq_index', 'bond_index',
                       'govt_bond', 'derivatives'

        Returns:
            List of records from OutBlock_1
        """
        if index_type not in self.index_endpoints:
            raise ValueError(f"Unknown index type: {index_type}")

        url = self.index_endpoints[index_type]
        params = {
            'AUTH_KEY': self.auth_key,
            'basDd': date
        }

        self._rate_limit()

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get('OutBlock_1', [])

    def fetch_all_index_data(self, date: str) -> dict:
        """
        Fetch all index data types for a specific date.

        Returns:
            Dict with keys for each index type
        """
        result = {}
        for index_type in self.index_endpoints.keys():
            try:
                result[index_type] = self.fetch_index_data(date, index_type)
            except Exception as e:
                logging.error(f"Failed to fetch {index_type} for {date}: {e}")
                result[index_type] = []
        return result
```

---

## ETL Implementation

### Create `index_etl.py`

```python
"""
ETL Pipeline for Market Index Data

Handles:
- KOSPI/KOSDAQ indices
- Bond indices
- Government bond market data
- Derivatives indices
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class IndexETLPipeline:
    """ETL pipeline for market index data."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None

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

    def _create_indexes(self):
        """Create performance indexes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_indices_date ON market_indices(date)",
            "CREATE INDEX IF NOT EXISTS idx_market_indices_class ON market_indices(index_class)",
            "CREATE INDEX IF NOT EXISTS idx_bond_indices_date ON bond_indices(date)",
            "CREATE INDEX IF NOT EXISTS idx_govt_bonds_date ON government_bonds(date)",
            "CREATE INDEX IF NOT EXISTS idx_govt_bonds_maturity ON government_bonds(maturity_type)",
            "CREATE INDEX IF NOT EXISTS idx_deriv_indices_date ON derivatives_indices(date)",
            "CREATE INDEX IF NOT EXISTS idx_deriv_indices_class ON derivatives_indices(index_class)"
        ]

        for idx_sql in indexes:
            cursor.execute(idx_sql)

        conn.commit()

    @staticmethod
    def parse_number(value: str) -> Optional[float]:
        """Parse numeric value from API response."""
        if value is None or value == '' or value == '-':
            return None
        try:
            # Remove commas and convert
            return float(str(value).replace(',', ''))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def parse_int(value: str) -> Optional[int]:
        """Parse integer value from API response."""
        parsed = IndexETLPipeline.parse_number(value)
        return int(parsed) if parsed is not None else None

    def process_market_indices(self, records: list, index_class: str) -> int:
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

    def process_bond_indices(self, records: list) -> int:
        """Process bond index data."""
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

    def process_government_bonds(self, records: list) -> int:
        """Process government bond market data."""
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

    def process_derivatives(self, records: list) -> int:
        """Process derivatives index data."""
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

    def check_date_exists(self, date: str, table: str) -> bool:
        """Check if data exists for a specific date in a table."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM {table} WHERE date = ? LIMIT 1", (date,))
        return cursor.fetchone() is not None

    def get_stats(self) -> dict:
        """Get statistics for all tables."""
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

                stats[table] = {
                    'count': count,
                    'min_date': row[0],
                    'max_date': row[1]
                }
            except sqlite3.OperationalError:
                stats[table] = {'count': 0, 'min_date': None, 'max_date': None}

        return stats
```

### Main ETL Runner

```python
"""
run_index_etl.py - Main runner for index data ETL
"""

import argparse
import logging
from datetime import datetime, timedelta

from krx_api import KRXAPI
from index_etl import IndexETLPipeline
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_trading_dates(start_date: str, end_date: str) -> list:
    """Generate list of trading dates (weekdays only)."""
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


def backfill_index_data(
    api: KRXAPI,
    pipeline: IndexETLPipeline,
    start_date: str,
    end_date: str,
    force: bool = False
):
    """Backfill historical index data."""

    dates = generate_trading_dates(start_date, end_date)
    logger.info(f"Backfilling {len(dates)} trading dates from {start_date} to {end_date}")

    for i, date in enumerate(dates):
        # Skip if already exists (unless force)
        if not force and pipeline.check_date_exists(date, 'market_indices'):
            logger.debug(f"Skipping {date} - already exists")
            continue

        try:
            # Fetch all index data for this date
            logger.info(f"[{i+1}/{len(dates)}] Processing {date}")

            # KOSPI indices
            kospi_data = api.fetch_index_data(date, 'kospi_index')
            kospi_count = pipeline.process_market_indices(kospi_data, 'KOSPI')

            # KOSDAQ indices
            kosdaq_data = api.fetch_index_data(date, 'kosdaq_index')
            kosdaq_count = pipeline.process_market_indices(kosdaq_data, 'KOSDAQ')

            # Bond indices
            bond_data = api.fetch_index_data(date, 'bond_index')
            bond_count = pipeline.process_bond_indices(bond_data)

            # Government bonds
            govt_bond_data = api.fetch_index_data(date, 'govt_bond')
            govt_count = pipeline.process_government_bonds(govt_bond_data)

            # Derivatives
            deriv_data = api.fetch_index_data(date, 'derivatives')
            deriv_count = pipeline.process_derivatives(deriv_data)

            logger.info(
                f"  KOSPI: {kospi_count}, KOSDAQ: {kosdaq_count}, "
                f"Bond: {bond_count}, Govt: {govt_count}, Deriv: {deriv_count}"
            )

        except Exception as e:
            logger.error(f"Error processing {date}: {e}")
            continue


def daily_update(api: KRXAPI, pipeline: IndexETLPipeline):
    """Update with yesterday's data."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    # Check if weekday
    dt = datetime.strptime(yesterday, '%Y%m%d')
    if dt.weekday() >= 5:
        logger.info(f"{yesterday} is a weekend, skipping")
        return

    backfill_index_data(api, pipeline, yesterday, yesterday, force=False)


def main():
    parser = argparse.ArgumentParser(description='Index Data ETL Pipeline')
    parser.add_argument('--db-path', default='krx_stock_data.db', help='Database path')
    parser.add_argument('--backfill', action='store_true', help='Run backfill')
    parser.add_argument('--daily-update', action='store_true', help='Run daily update')
    parser.add_argument('--start-date', help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', help='End date (YYYYMMDD)')
    parser.add_argument('--force', action='store_true', help='Force reprocess existing dates')
    parser.add_argument('--status', action='store_true', help='Show status')

    args = parser.parse_args()

    config = load_config()
    api = KRXAPI(config['api']['auth_key'], config)

    with IndexETLPipeline(args.db_path) as pipeline:
        pipeline.init_tables()

        if args.status:
            stats = pipeline.get_stats()
            print("\n=== Index Data Status ===")
            for table, info in stats.items():
                print(f"\n{table}:")
                print(f"  Records: {info['count']:,}")
                print(f"  Date range: {info['min_date']} to {info['max_date']}")
            return

        if args.backfill:
            if not args.start_date or not args.end_date:
                parser.error("--backfill requires --start-date and --end-date")
            backfill_index_data(api, pipeline, args.start_date, args.end_date, args.force)

        elif args.daily_update:
            daily_update(api, pipeline)


if __name__ == '__main__':
    main()
```

---

## Backfill & Update Procedures

### Initial Setup

```bash
# 1. Create tables
python run_index_etl.py --db-path krx_stock_data.db --status

# 2. Backfill historical data (example: 2020-2024)
python run_index_etl.py --backfill \
    --start-date 20200101 \
    --end-date 20241231 \
    --db-path krx_stock_data.db

# 3. Verify data
python run_index_etl.py --status --db-path krx_stock_data.db
```

### Daily Updates

```bash
# Run after market close (e.g., via cron at 6 PM KST)
python run_index_etl.py --daily-update --db-path krx_stock_data.db
```

### Cron Schedule Example

```cron
# Run daily at 6:30 PM KST (9:30 AM UTC)
30 9 * * 1-5 cd /path/to/algostock && python run_index_etl.py --daily-update >> /var/log/index_etl.log 2>&1
```

---

## Usage Examples

### Query KOSPI 200 Index History

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('krx_stock_data.db')

# Get KOSPI 200 history
df = pd.read_sql_query("""
    SELECT date, closing_index, change_rate, trading_volume
    FROM market_indices
    WHERE index_name = '코스피 200'
    ORDER BY date DESC
    LIMIT 30
""", conn)

print(df)
```

### Get Bond Yield Trends

```python
# Get KRX bond index with yield
df = pd.read_sql_query("""
    SELECT date, index_group_name, total_return_index, avg_yield, avg_duration
    FROM bond_indices
    WHERE index_group_name = 'KRX 채권지수'
    ORDER BY date DESC
""", conn)
```

### Get 10-Year Treasury Yield

```python
# Get 10-year benchmark government bond
df = pd.read_sql_query("""
    SELECT date, issue_name, closing_yield, closing_price
    FROM government_bonds
    WHERE maturity_type = '10'
      AND issue_type = '지표'
    ORDER BY date DESC
""", conn)
```

### Get VIX (Volatility Index)

```python
# Get KOSPI 200 volatility index
df = pd.read_sql_query("""
    SELECT date, closing_index as vix, change_rate
    FROM derivatives_indices
    WHERE index_name = '코스피 200 변동성지수'
    ORDER BY date DESC
""", conn)
```

### Feature Engineering Integration

```python
# Example: Add market regime features based on indices
def add_market_features(stock_df, conn):
    """Add market-level features to stock data."""

    # Get KOSPI index data
    kospi_df = pd.read_sql_query("""
        SELECT date, closing_index as kospi_index, change_rate as kospi_return
        FROM market_indices
        WHERE index_name = '코스피'
    """, conn)

    # Get VIX
    vix_df = pd.read_sql_query("""
        SELECT date, closing_index as vix
        FROM derivatives_indices
        WHERE index_name = '코스피 200 변동성지수'
    """, conn)

    # Get 10-year yield
    yield_df = pd.read_sql_query("""
        SELECT date, closing_yield as treasury_10y
        FROM government_bonds
        WHERE maturity_type = '10' AND issue_type = '지표'
    """, conn)

    # Merge features
    stock_df = stock_df.merge(kospi_df, on='date', how='left')
    stock_df = stock_df.merge(vix_df, on='date', how='left')
    stock_df = stock_df.merge(yield_df, on='date', how='left')

    return stock_df
```

---

## Data Quality Checks

### Validation Queries

```sql
-- Check for missing dates (gaps)
SELECT date,
       LAG(date) OVER (ORDER BY date) as prev_date,
       JULIANDAY(date) - JULIANDAY(LAG(date) OVER (ORDER BY date)) as gap_days
FROM (SELECT DISTINCT date FROM market_indices)
WHERE gap_days > 3;  -- Flag gaps > 3 days (includes weekends)

-- Check for NULL values in key fields
SELECT date, COUNT(*) as null_count
FROM market_indices
WHERE closing_index IS NULL
GROUP BY date
ORDER BY date;

-- Verify index count per date
SELECT date, COUNT(*) as index_count
FROM market_indices
GROUP BY date
HAVING index_count < 10  -- Flag dates with fewer than expected indices
ORDER BY date;
```

### Data Validation Script

```python
def validate_index_data(conn):
    """Run data quality checks."""
    cursor = conn.cursor()

    checks = {
        'market_indices': [
            ("Missing closing prices",
             "SELECT COUNT(*) FROM market_indices WHERE closing_index IS NULL"),
            ("Duplicate entries",
             "SELECT date, index_name, COUNT(*) FROM market_indices GROUP BY date, index_name HAVING COUNT(*) > 1"),
        ],
        'government_bonds': [
            ("Missing yields",
             "SELECT COUNT(*) FROM government_bonds WHERE closing_yield IS NULL AND issue_type = '지표'"),
        ]
    }

    for table, table_checks in checks.items():
        print(f"\n{table}:")
        for name, query in table_checks:
            cursor.execute(query)
            result = cursor.fetchone()[0]
            status = "✓" if result == 0 else f"⚠ {result}"
            print(f"  {name}: {status}")
```

---

## Summary

| Step | Command |
|------|---------|
| **Initialize tables** | `python run_index_etl.py --status` |
| **Backfill 2020-2024** | `python run_index_etl.py --backfill --start-date 20200101 --end-date 20241231` |
| **Daily update** | `python run_index_etl.py --daily-update` |
| **Check status** | `python run_index_etl.py --status` |
| **Force reprocess** | `python run_index_etl.py --backfill --start-date 20240101 --end-date 20240131 --force` |

This guide provides a complete framework for integrating Korean market index data into the AlgoStock database following the existing ETL patterns and best practices.
