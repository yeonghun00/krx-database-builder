# Database Schema Reference

**File:** `krx_stock_data.db` (11GB, SQLite WAL mode, page_size=4096)

## Table Sizes

| Table | Rows | Data Size | Notes |
|-------|------|-----------|-------|
| `daily_prices` | 8,722,511 | 748 MB (+2.2GB indexes) | Main bottleneck |
| `stocks` | 4,750 | tiny | KOSPI: 2210, KOSDAQ: 2315, KONEX: 225 |
| `financial_periods` | 158,094 | small | Links stocks → financial items |
| `financial_items_bs_cf` | 13,491,922 | 1,984 MB | Balance Sheet + Cash Flow |
| `financial_items_pl` | 3,871,357 | 564 MB | Profit & Loss |
| `index_daily_prices` | ~50K | small | KOSPI, KOSDAQ indices |

## Core Tables

### `daily_prices` (PK: stock_code, date)
```
stock_code    TEXT     -- e.g. '005930' (삼성전자)
date          TEXT     -- e.g. '20260205' (YYYYMMDD format)
closing_price INTEGER  -- 종가 (원)
change        INTEGER  -- 전일 대비 변동 (원)
change_rate   REAL     -- 전일 대비 변동률 (%)
opening_price INTEGER  -- 시가
high_price    INTEGER  -- 고가
low_price     INTEGER  -- 저가
volume        INTEGER  -- 거래량 (주)
value         INTEGER  -- 거래대금 (원)
market_cap    INTEGER  -- 시가총액 (원) — 0 for ~342K rows (거래정지 등)
market_type   TEXT     -- 'kospi', 'kosdaq', 'kodex' (소문자)
created_at    TIMESTAMP
```

**NULL/Zero patterns:**
- `closing_price = 0`: 0 rows (filtered in ETL)
- `volume = 0`: 342,330 rows (거래정지일)
- `market_cap = 0`: 0 rows (but NULL없음)
- `opening_price = 0`: 342,467 rows (same days as volume=0)
- `value = 0`: 342,330 rows

**Market type distribution:**
- kospi: 3,416,197 rows (2,210 stocks)
- kosdaq: 4,935,221 rows (2,334 stocks)
- kodex: 371,093 rows (302 ETFs) — NOT used in backtest

**Date range:** 20110104 ~ 20260205

### `stocks` (PK: stock_code)
```
stock_code           TEXT  -- e.g. '005930'
current_name         TEXT  -- e.g. '삼성전자'
current_market_type  TEXT  -- 'KOSPI', 'KOSDAQ', 'KONEX' (대문자! daily_prices는 소문자)
current_sector_type  TEXT  -- e.g. '전자부품, 컴퓨터, ...'
shares_outstanding   INTEGER
is_active            BOOLEAN
updated_at           TIMESTAMP
```

**IMPORTANT:** `stocks.current_market_type` is UPPERCASE ('KOSPI'), but `daily_prices.market_type` is lowercase ('kospi').

### `financial_periods` (PK: id, UNIQUE: stock_code + fiscal_date + consolidation_type)
```
id                  INTEGER  -- PK, referenced by financial_items as period_id
stock_code          TEXT
company_name        TEXT
market_type         TEXT     -- '유가증권시장상장법인', '코스닥시장상장법인'
industry_code       TEXT     -- e.g. '291'
industry_name       TEXT     -- e.g. '일반 목적용 기계 제조업'
fiscal_month        INTEGER  -- 3, 6, 9, 12
fiscal_date         TEXT     -- e.g. '2015-03-31'
available_date      TEXT     -- e.g. '20150601' (YYYYMMDD — when data becomes public)
report_type         TEXT     -- '사업보고서', '분기보고서', etc.
consolidation_type  TEXT     -- '연결' (consolidated) or '별도' (separate)
currency            TEXT     -- 'KRW'
```

**Key filter:** `consolidation_type = '연결'` for consolidated financials.
**70,526 rows** match `연결 AND available_date >= 20100101`.

### `financial_items_bs_cf` (PK: id, UNIQUE: period_id + statement_type + item_code)
```
id                    INTEGER
period_id             INTEGER  -- FK → financial_periods.id
statement_type        TEXT     -- 'BS' or 'CF'
item_code             TEXT     -- raw XBRL code
item_code_normalized  TEXT     -- standardized code (e.g. 'ifrs-full_Equity')
item_name             TEXT     -- Korean name (e.g. '자본총계')
amount_current        REAL     -- 당기 금액
amount_prev           REAL     -- 전기
amount_prev2          REAL     -- 전전기
```

**Key item_code_normalized values used:**
- `ifrs-full_Equity` — 자본총계
- `ifrs-full_Assets` — 자산총계
- `ifrs-full_Liabilities` — 부채총계
- `ifrs-full_CashAndCashEquivalents` — 현금성자산
- `ifrs-full_CashFlowsFromUsedInOperatingActivities` — 영업활동CF
- `ifrs-full_CashFlowsFromUsedInInvestingActivities` — 투자활동CF

### `financial_items_pl` (PK: id, UNIQUE: period_id + item_code)
```
id                    INTEGER
period_id             INTEGER  -- FK → financial_periods.id
item_code             TEXT
item_code_normalized  TEXT
item_name             TEXT
amount_current_qtr    REAL     -- 당분기
amount_current_ytd    REAL     -- 당기누적 (used for features)
amount_prev_qtr       REAL
amount_prev_ytd       REAL
amount_prev_year      REAL
amount_prev2_year     REAL
```

**Key item_code_normalized values used:**
- `ifrs-full_Revenue` — 매출액
- `dart_OperatingIncomeLoss` — 영업이익
- `ifrs-full_ProfitLoss` — 당기순이익
- `ifrs-full_GrossProfit` — 매출총이익
- `ifrs-full_IncomeTaxExpenseContinuingOperations` — 법인세
- `ifrs-full_FinanceCosts` — 금융비용

### `index_daily_prices` (PK: index_code, date)
```
index_code      TEXT  -- e.g. 'KOSPI_코스피_200'
date            TEXT  -- YYYYMMDD
closing_index   REAL
opening_index   REAL
high_index      REAL
low_index       REAL
trading_volume  INTEGER
trading_value   INTEGER
market_cap      INTEGER
```

## Performance-Critical Indexes

| Index | Columns | Used By |
|-------|---------|---------|
| `idx_dp_stock_date` | (stock_code, date) | Main query (covering PK) |
| `idx_dp_mcap_date_stock` | (market_cap, date, stock_code) | Pre-filter eligible stocks |
| `idx_bs_period_item` | (period_id, item_code_normalized) | Chunked BS/CF load |
| `idx_pl_period_item` | (period_id, item_code_normalized) | Chunked PL load |
| `idx_fp_consol_avail` | (consolidation_type, available_date) | Financial periods filter |

## Query Performance Notes

**11GB DB on MacBook Air — key findings:**

1. **SQL JOIN is catastrophic.** `daily_prices JOIN stocks` turns 115s → 1000s+ (10x penalty). Always load tables separately and merge in pandas.

2. **IN-subquery on large tables is catastrophic.** `WHERE stock_code IN (SELECT DISTINCT stock_code FROM daily_prices WHERE ...)` scans 8.7M rows = 480s. Use EXISTS on small table instead.

3. **Chunked period_id queries are 5x faster** than SQL JOINs for financial tables. Load `financial_periods` first (instant), then query items in batches of 500 period_ids.

4. **Default SQLite cache (8MB) is too small** for 11GB DB. Use `PRAGMA cache_size = -256000` (256MB).

5. **Baseline read speeds (Python sqlite3 module):**
   - 8M rows, 10 cols, no JOIN: ~115s
   - With stock_code pre-filter (~5M rows): ~60-80s
   - Single stock query: instant (<10ms)
