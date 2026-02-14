# ETL Pipelines

Four independent ETL pipelines populate `krx_stock_data.db`. Run them in order since later pipelines don't depend on earlier ones for ingestion, but the ML pipeline expects all tables to be populated.

## Unified Runner (`scripts/run_etl.py`)

The recommended way to run ETL. A single command that manages all 4 pipelines, auto-detects gaps, and skips data that already exists.

### Daily update (most common usage)

```bash
python3 scripts/run_etl.py update --markets kospi,kosdaq --workers 4
```

This auto-detects what's stale for each pipeline:
- **Prices**: fetches from `MAX(date)+1` to today, skips existing dates
- **Index constituents**: processes months from latest stored month+1 to now
- **Delisted stocks**: full refresh (single HTTP call, idempotent)
- **Financials**: only processes new ZIP files not yet in `.processed_files` marker

### Historical backfill

```bash
python3 scripts/run_etl.py backfill --start-date 20200101 --end-date 20251231
```

### Skip specific pipelines

```bash
# Only run prices and delisted
python3 scripts/run_etl.py update --skip index financial

# Only run financials
python3 scripts/run_etl.py update --skip prices index delisted
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--markets` | `kospi,kosdaq` | Comma-separated markets for price fetching |
| `--workers` | `4` | Parallel workers for index scraping |
| `--skip` | none | Pipelines to skip: `prices`, `index`, `delisted`, `financial` |

Before running, prints a status table showing each pipeline's latest data and estimated gap. After running, prints a results summary.

---

## Individual Pipelines

You can still run each pipeline independently if needed. The unified runner imports and calls the same classes described below.

## Database Schema

### `stocks` (stock master)

| Column | Type | Description |
|--------|------|-------------|
| `stock_code` | TEXT PK | 6-digit KRX code (e.g. `005930`) |
| `current_name` | TEXT | Latest company name |
| `current_market_type` | TEXT | `kospi` or `kosdaq` |
| `current_sector_type` | TEXT | KRX sector classification |
| `shares_outstanding` | INTEGER | Current shares outstanding |
| `is_active` | BOOLEAN | Whether the stock is still listed |

### `stock_history` (name/market changes over time)

| Column | Type | Description |
|--------|------|-------------|
| `stock_code` | TEXT FK | References `stocks` |
| `effective_date` | TEXT | Date this snapshot is from (YYYYMMDD) |
| `name` | TEXT | Company name at that date |
| `market_type` | TEXT | Market at that date |

### `daily_prices` (OHLCV + market cap)

| Column | Type | Description |
|--------|------|-------------|
| `stock_code` | TEXT FK | References `stocks` |
| `date` | TEXT | Trading date (YYYYMMDD) |
| `closing_price` | INTEGER | Close price (KRW) |
| `opening_price` | INTEGER | Open price |
| `high_price` | INTEGER | Day high |
| `low_price` | INTEGER | Day low |
| `volume` | INTEGER | Shares traded |
| `value` | INTEGER | Value traded (KRW) |
| `market_cap` | INTEGER | Market capitalization (KRW) |
| `change` | INTEGER | Price change from prev close |
| `change_rate` | REAL | % change |

Primary key: `(stock_code, date)`

### `index_constituents` (monthly index membership snapshots)

| Column | Type | Description |
|--------|------|-------------|
| `date` | TEXT | Snapshot date (YYYY-MM-DD format) |
| `stock_code` | TEXT | Member stock code |
| `index_code` | TEXT | Index identifier (e.g. `KOSPI_코스피_200`, `KOSDAQ_코스닥_IT`) |

This is the source for both index membership counts and sector assignment. Each stock's "sector" is determined by the most specific (smallest) non-broad index it belongs to on a given month.

### `delisted_stocks`

| Column | Type | Description |
|--------|------|-------------|
| `stock_code` | TEXT UNIQUE | Delisted stock code |
| `company_name` | TEXT | Company name |
| `delisting_date` | DATE | When it was delisted (YYYY-MM-DD) |
| `delisting_reason` | TEXT | Reason for delisting |

### `financial_periods` (financial statement metadata)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment ID |
| `stock_code` | TEXT | Company code |
| `fiscal_date` | TEXT | Period end date (YYYY-MM-DD) |
| `available_date` | TEXT | First date this data can be used (PIT safe) |
| `consolidation_type` | TEXT | `연결` (consolidated) or `별도` (separate) |
| `fiscal_month` | INTEGER | Fiscal year-end month |
| `report_type` | TEXT | Annual/quarterly indicator |

The `available_date` is calculated using the **45/90-day rule**:
- Q1/Q2/Q3 reports: available ~45 days after fiscal period end
- Q4 (annual) reports: available ~90 days after fiscal period end

Example for a December fiscal year company:
- Q1 ends Mar 31 -> available May 16
- Q2 ends Jun 30 -> available Aug 16
- Q3 ends Sep 30 -> available Nov 15
- Q4 ends Dec 31 -> available Apr 1 next year

### `financial_items_bs_cf` (balance sheet + cash flow line items)

| Column | Type | Description |
|--------|------|-------------|
| `period_id` | INTEGER FK | References `financial_periods.id` |
| `item_code_normalized` | TEXT | Normalized IFRS code (e.g. `ifrs-full_Equity`) |
| `item_name` | TEXT | Human-readable item name |
| `amount_current` | REAL | Current period amount |
| `amount_prev` | REAL | Previous period amount |

Key items used by the model: `ifrs-full_Equity`, `ifrs-full_Assets`, `ifrs-full_CashFlowsFromUsedInOperatingActivities`

### `financial_items_pl` (income statement line items)

| Column | Type | Description |
|--------|------|-------------|
| `period_id` | INTEGER FK | References `financial_periods.id` |
| `item_code_normalized` | TEXT | Normalized IFRS code |
| `item_name` | TEXT | Human-readable item name |
| `amount_current_ytd` | REAL | Year-to-date amount |
| `amount_current_qtr` | REAL | Current quarter amount |

Key items used by the model: `ifrs-full_ProfitLoss` (net income), `ifrs-full_GrossProfit`

---

## Pipeline 1: Prices + Stock Master (`clean_etl.py`)

**Source**: KRX market data API
**Tables updated**: `stocks`, `daily_prices`, `stock_history`

### Daily update (run each trading day)

```bash
python3 etl/clean_etl.py \
  --daily-update --date 20260214 \
  --markets kospi,kosdaq --db-path krx_stock_data.db
```

### Historical backfill

```bash
python3 etl/clean_etl.py \
  --backfill --start-date 20100101 --end-date 20251231 \
  --markets kospi,kosdaq --db-path krx_stock_data.db
```

**Notes**:
- `--markets` accepts `kospi`, `kosdaq`, `kodex` (comma-separated)
- If a progress file exists from a previous run, it may prompt to resume
- Use `--force` to reprocess dates that already have data

---

## Pipeline 2: Index Constituents (`index_constituents_etl.py`)

**Source**: KRX index website (via Selenium + Chrome)
**Table updated**: `index_constituents`

### Update (latest month)

```bash
python3 etl/index_constituents_etl.py \
  --mode update --strategy skip --workers 4 --config config.json
```

### Backfill (full history)

```bash
python3 etl/index_constituents_etl.py \
  --mode backfill --start-date 2010-01-01 --workers 4 --config config.json
```

**Notes**:
- Requires Chrome + matching ChromeDriver installed
- `--strategy overwrite` replaces existing rows; `skip` skips dates that already have data
- Each snapshot captures which stocks belong to which KRX indices on that date
- Used for two purposes: (1) counting how many indices a stock belongs to, (2) assigning each stock a sector based on its most specific index

---

## Pipeline 3: Delisted Stocks (`delisted_stocks_etl.py`)

**Source**: KRX KIND endpoint
**Table updated**: `delisted_stocks`

```bash
python3 etl/delisted_stocks_etl.py
```

Rebuilds the entire table each run. Simple and idempotent. The model uses this to exclude stocks that traded before their delisting date (survivorship bias control).

---

## Pipeline 4: Financial Statements (`financial_etl.py`)

**Source**: Raw ZIP files in `data/raw_financial/` (downloaded separately from DART/KRX)
**Tables updated**: `financial_periods`, `financial_items_bs_cf`, `financial_items_pl`

```bash
python3 etl/financial_etl.py krx_stock_data.db data/raw_financial
```

**Notes**:
- ZIP files contain BS (balance sheet), PL (income statement), and CF (cash flow) data
- Item codes are normalized from `ifrs_X` to `ifrs-full_X` format during loading
- Only consolidated (`연결`) statements are used by the model
- The `available_date` field enforces PIT safety -- the model never uses financial data before it would have been publicly available

---

## Recommended Run Order

```
1. clean_etl.py          (prices must exist first)
2. index_constituents_etl.py
3. delisted_stocks_etl.py
4. financial_etl.py
```

## Validation After ETL

```bash
# Row counts
sqlite3 krx_stock_data.db "SELECT COUNT(*) AS daily_prices_rows FROM daily_prices;"
sqlite3 krx_stock_data.db "SELECT COUNT(*) AS index_rows FROM index_constituents;"
sqlite3 krx_stock_data.db "SELECT COUNT(*) AS delisted_rows FROM delisted_stocks;"
sqlite3 krx_stock_data.db "SELECT COUNT(*) AS financial_rows FROM financial_periods;"

# Data freshness
sqlite3 krx_stock_data.db "SELECT MAX(date) FROM daily_prices;"
sqlite3 krx_stock_data.db "SELECT MAX(date) FROM index_constituents;"
```

## Common Issues

| Problem | Fix |
|---------|-----|
| Selenium/Chrome errors in constituents ETL | Install/update Chrome and matching ChromeDriver |
| Financial ETL loads 0 rows | Check that ZIP files exist in `data/raw_financial/` |
| Very slow backfill | Use `--workers 4` for constituents; split date ranges for prices |
| `market_type` column missing | Run `clean_etl.py` first -- it creates the `daily_prices` table with all columns |
