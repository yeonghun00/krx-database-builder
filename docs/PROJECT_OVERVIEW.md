# AlgoStock - Korean Stock Market Data System

A comprehensive data engineering platform for collecting, processing, and analyzing Korean stock market data from the KRX (Korea Exchange).

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Data Flow](#data-flow)
5. [Database Schema](#database-schema)
6. [Usage Guide](#usage-guide)
7. [Improvement Recommendations](#improvement-recommendations)

---

## Project Overview

**AlgoStock** is a production-grade ETL (Extract-Transform-Load) system that:

- **Collects** daily stock data from KRX official API (KOSPI, KOSDAQ, KODEX markets)
- **Processes** and normalizes raw API data with validation and cleaning
- **Stores** data in an optimized SQLite database (~984MB for 10+ years)
- **Analyzes** stocks using fast screening algorithms (price increase, trading value)

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Market Support** | KOSPI (main), KOSDAQ (growth), KODEX (derivatives) |
| **Parallel Processing** | ThreadPoolExecutor with 3 concurrent requests |
| **Rate Limiting** | Thread-safe with 0.5s backfill / 5.0s normal delay |
| **Resume Capability** | JSON-based progress tracking for interrupted operations |
| **Data Validation** | Comprehensive cleaning and integrity checks |
| **Fast Screening** | Pandas/NumPy optimized stock analysis |

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         KRX Official API            │
                    │   (3 market endpoints)              │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │           krx_api.py                │
                    │   • Rate limiting (thread-safe)     │
                    │   • Parallel fetching               │
                    │   • Data validation                 │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │          clean_etl.py               │
                    │   • Direct normalization            │
                    │   • Batch upserts                   │
                    │   • Progress tracking               │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │     SQLite Database (984MB)         │
                    │   • stocks (metadata)               │
                    │   • stock_history (changes)         │
                    │   • daily_prices (normalized)       │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │      analyzer/screeners.py          │
                    │   • Price increase screening        │
                    │   • Trading value screening         │
                    │   • Combined analysis               │
                    │   • Excel export                    │
                    └─────────────────────────────────────┘
```

---

## Components

### 1. KRX API Client (`krx_api.py` - 637 lines)

Handles all communication with KRX data API.

**Key Classes/Methods:**

| Method | Purpose |
|--------|---------|
| `KRXAPI.__init__(auth_key)` | Initialize with API credentials |
| `fetch_data_for_date(date)` | Fetch single date, single market |
| `fetch_data_for_date_parallel(date, markets)` | Fetch single date, multiple markets |
| `fetch_data_range_parallel(start, end, markets)` | Fetch date range in parallel |
| `_validate_and_clean_record(record, date)` | Clean numeric fields, validate data |

**Market Endpoints:**
```python
market_endpoints = {
    'kospi': 'https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd',
    'kosdaq': 'https://data-dbg.krx.co.kr/svc/apis/sto/ksq_bydd_trd',
    'kodex': 'https://data-dbg.krx.co.kr/svc/apis/sto/knx_bydd_trd'
}
```

### 2. ETL Pipeline (`clean_etl.py` - 720 lines)

Clean Extract-Transform-Load with direct normalization (no raw storage).

**Key Classes/Methods:**

| Method | Purpose |
|--------|---------|
| `CleanETLPipeline.__init__(db_path)` | Initialize and create schema |
| `process_data(raw_data)` | Transform and load API data |
| `get_backfill_progress(start, end)` | Check processing status |
| `save_progress(data)` / `load_progress()` | Resume capability |
| `validate_data()` | Check referential integrity |
| `optimize_database()` | VACUUM and REINDEX |

### 3. Stock Screener (`analyzer/screeners.py` - 303 lines)

Fast stock screening with pandas/numpy operations.

**Screening Methods:**

| Method | Description |
|--------|-------------|
| `top_price_increase(start, end, markets, percentile)` | Top X% by price change |
| `top_value(start, end, markets, percentile)` | Top X% by trading value (거래대금) |
| `combined_screen(start, end, markets, price_pct, value_pct)` | Intersection of both |
| `export_results(df, prefix)` | Export to Excel with summary |

### 4. CLI Tools

| File | Purpose |
|------|---------|
| `algostock_analyzer.py` | Main entry point for analysis |
| `cli/analyzer_cli.py` | Subcommand-based analysis CLI |
| `etl_cli.py` | ETL status, validation, cleanup |
| `clean_etl.py` (main) | Backfill and daily update operations |

### 5. Configuration (`config.py` + `config.json`)

```json
{
  "database": { "path": "krx_stock_data.db", "backup_enabled": true },
  "api": {
    "auth_key": "YOUR_API_KEY",
    "request_delay": 1.0,
    "backfill_request_delay": 0.5,
    "max_concurrent_requests": 3
  },
  "backfill": { "start_year": 2011 }
}
```

---

## Data Flow

### Daily Update Flow

```
1. Trigger daily update (manual or scheduled)
2. Check if data exists for target date
3. Fetch data from all configured markets
4. Validate and clean records
5. Upsert stock metadata (track changes in history)
6. Insert/replace daily prices
7. Log completion
```

### Backfill Flow

```
1. Calculate date range and trading days
2. Check existing data (skip already processed)
3. Load previous progress (if resuming)
4. For each date:
   a. Fetch all markets in parallel
   b. Process and store data
   c. Save progress every 10 dates
5. Clean up progress file on completion
```

---

## Database Schema

### Tables

**`stocks`** - Current stock metadata
```sql
CREATE TABLE stocks (
    stock_code TEXT PRIMARY KEY,
    current_name TEXT NOT NULL,
    current_market_type TEXT,
    current_sector_type TEXT,
    shares_outstanding INTEGER,
    is_active BOOLEAN DEFAULT 1,
    updated_at TIMESTAMP
);
```

**`stock_history`** - Historical metadata changes
```sql
CREATE TABLE stock_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    effective_date TEXT NOT NULL,
    name TEXT,
    market_type TEXT,
    sector_type TEXT,
    shares_outstanding INTEGER,
    UNIQUE(stock_code, effective_date)
);
```

**`daily_prices`** - Normalized price data
```sql
CREATE TABLE daily_prices (
    stock_code TEXT NOT NULL,
    date TEXT NOT NULL,
    closing_price INTEGER,
    change INTEGER,
    change_rate REAL,
    opening_price INTEGER,
    high_price INTEGER,
    low_price INTEGER,
    volume INTEGER,
    value INTEGER,
    market_cap INTEGER,
    market_type TEXT,
    PRIMARY KEY (stock_code, date)
);
```

### Indexes

```sql
idx_daily_prices_date           -- (date)
idx_daily_prices_stock_date     -- (stock_code, date)
idx_daily_prices_market         -- (market_type)
idx_daily_prices_date_market    -- (date, market_type)
idx_stock_history_date          -- (effective_date)
```

---

## Usage Guide

**Note:** All commands are now available through the unified `algostock_cli.py` CLI.

### Running a Backfill

```bash
# Full backfill for all markets
python3 algostock_cli.py etl backfill \
    -s 20200101 \
    -e 20251231 \
    -m kospi,kosdaq,kodex

# Resume interrupted backfill (will prompt to resume)
python3 algostock_cli.py etl backfill -s 20200101 -e 20251231
```

### Daily Updates

```bash
# Update yesterday's data
python3 algostock_cli.py etl update -m kospi,kosdaq,kodex

# Update specific date
python3 algostock_cli.py etl update --date 20250128
```

### Stock Screening

```bash
# Quick screening (current year, top 5%)
python3 algostock_cli.py quick

# Top 1% by price increase
python3 algostock_cli.py analyze price \
    -s 20250101 \
    -e 20251231 \
    -p 1 \
    --export

# Top 5% by trading value
python3 algostock_cli.py analyze value \
    -s 20250101 \
    -e 20251231 \
    -p 5 \
    --export

# Combined screening
python3 algostock_cli.py analyze combined \
    -s 20250101 \
    -e 20251231 \
    --price-percentile 5 \
    --value-percentile 5 \
    --export
```

### ETL Management

```bash
# Check status
python3 algostock_cli.py etl status

# Validate data integrity
python3 algostock_cli.py etl validate

# Cleanup old data (keep 365 days)
python3 algostock_cli.py etl cleanup --days 365
```

---

## Implemented Improvements

The following improvements have been implemented:

### 1. Code Organization - Merged Duplicate Methods (DONE)

Merged `_validate_and_clean_record()` and `_validate_and_clean_record_multi_market()` into a single method with optional market parameter in `krx_api.py`.

### 2. Removed Unused Imports (DONE)

- Removed unused `asyncio` and `aiohttp` imports from `krx_api.py`
- Added missing `logging` import to `cli/analyzer_cli.py`
- Fixed `top_volume()` to `top_value()` in `cli/analyzer_cli.py`

### 3. Configuration Integration (DONE)

`KRXAPI` now accepts a `config` dict in its constructor and uses values from `config.json` instead of hardcoded defaults:
```python
api = KRXAPI(config_dict['api']['auth_key'], config_dict.get('api', {}))
```

### 4. Connection Pooling (DONE)

`CleanETLPipeline` now implements connection pooling with `_get_connection()` method. The connection is reused across operations and can be closed with `close()` or used as a context manager.

### 5. Batch Processing Optimization (DONE)

`_extract_stocks()` now uses `_get_stocks_metadata_batch()` to fetch all stock metadata in a single query instead of N individual queries.

### 6. Unified CLI (DONE)

All three CLI tools (`algostock_analyzer.py`, `cli/analyzer_cli.py`, `etl_cli.py`) have been consolidated into a single `algostock_cli.py` with subcommands:
- `algostock_cli.py etl {backfill|update|status|validate|cleanup}`
- `algostock_cli.py analyze {price|value|combined}`
- `algostock_cli.py quick`

---

## Future Improvement Recommendations

### 1. SQL Injection Prevention

**Issue:** `cleanup_old_data()` uses string formatting for SQL:
```python
cursor.execute('''
    DELETE FROM daily_prices
    WHERE date < date('now', '-{} days')
'''.format(days_to_keep))  # Potential SQL injection
```

**Solution:** Use parameterized query:
```python
cursor.execute('''
    DELETE FROM daily_prices
    WHERE date < date('now', '-' || ? || ' days')
''', (days_to_keep,))
```

### 2. Type Safety with Dataclasses

**Issue:** Data passed around as dictionaries without type validation.

**Solution:** Use dataclasses for structured data:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class StockRecord:
    stock_code: str
    date: str
    closing_price: Optional[int] = None
    volume: Optional[int] = None
```

### 3. Async API Requests

**Issue:** The code uses synchronous `requests` with threading.

**Solution:** Consider implementing true async with `aiohttp` for better performance:
```python
import aiohttp
import asyncio

async def fetch_data_async(self, dates: List[str], markets: List[str]):
    async with aiohttp.ClientSession(headers=self.headers) as session:
        tasks = [
            self._fetch_single(session, date, market)
            for date in dates for market in markets
        ]
        return await asyncio.gather(*tasks)
```

### 4. Centralized Error Handling

**Issue:** Error handling scattered with varied patterns.

**Solution:** Create custom exceptions:
```python
class AlgoStockError(Exception):
    """Base exception for AlgoStock."""

class APIError(AlgoStockError):
    """KRX API related errors."""
```

### 5. Add Retry Logic

**Issue:** API requests fail immediately without retry on transient errors.

**Solution:** Use `tenacity` library for automatic retries:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _make_request(self, date: str, market: str) -> Optional[Dict]:
    # ... existing code ...
```

### 6. Improve Screener Query Performance

**Issue:** `top_price_increase()` loads all data into pandas, then filters.

**Solution:** Push more computation to SQL using window functions for better performance with large datasets.

---

## Summary

AlgoStock is a well-structured data engineering project for Korean stock market analysis.

### Completed Improvements

1. **Code deduplication** - Merged duplicate validation methods in `krx_api.py`
2. **Bug fixes** - Fixed missing import in CLI, corrected method name `top_volume` -> `top_value`
3. **Removed unused code** - Removed unused `asyncio`/`aiohttp` imports
4. **Configuration** - `KRXAPI` now accepts config dict instead of hardcoded values
5. **Performance** - Implemented connection pooling and batch queries in `CleanETLPipeline`
6. **UX** - Consolidated three CLI tools into unified `algostock_cli.py`

### Remaining Future Improvements

1. **Security** - Parameterize SQL in `cleanup_old_data()`
2. **Type Safety** - Add dataclasses for structured data
3. **Async I/O** - Implement true async API requests
4. **Error Handling** - Centralized custom exceptions
5. **Retry Logic** - Add automatic retry for transient API failures

The core architecture is sound, with good separation of concerns between API, ETL, and analysis layers.
