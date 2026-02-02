# AlgoStock Database Structure

## Overview

Our stock analysis system uses a SQLite database (`krx_stock_data.db`) to store Korean stock market data. The database contains historical price data and company information for all KOSPI and KOSDAQ listed companies since 2011.

---

## Data at a Glance

| What We Have | Count | Period |
|-------------|-------|--------|
| Total Companies | 4,749 | - |
| Active Companies | ~2,500 | Current |
| Daily Price Records | 8.7 million | 2011 ~ Present |
| Markets Covered | KOSPI, KOSDAQ | - |

---

## Three Main Tables

### 1. Stocks (Company Master List)

**Purpose**: Contains basic information about all listed companies.

| Field | Description | Example |
|-------|-------------|---------|
| stock_code | 6-digit stock code (unique ID) | 005930 |
| current_name | Company name | 삼성전자 |
| current_market_type | Which market | kospi / kosdaq |
| current_sector_type | Industry category | 반도체 |
| shares_outstanding | Total shares issued | 5,969,782,550 |
| is_active | Still listed? | Yes / No |

**Key Points**:
- Each company has one unique 6-digit code
- Name changes are tracked (e.g., when a company rebrands)
- Includes both active and delisted companies

---

### 2. Daily Prices (Market Data)

**Purpose**: Daily trading data for every stock.

| Field | Description | Example |
|-------|-------------|---------|
| stock_code | Which company | 005930 |
| date | Trading date | 20240115 |
| closing_price | End-of-day price | 78,000 |
| opening_price | Start-of-day price | 77,500 |
| high_price | Highest price of day | 78,500 |
| low_price | Lowest price of day | 77,200 |
| volume | Shares traded | 15,234,567 |
| value | Trading value (KRW) | 1,185,276,226,000 |
| market_cap | Total company value | 465,243,078,780,000 |
| change | Price change (won) | +500 |
| change_rate | Price change (%) | +0.65 |

**Key Points**:
- One row per company per trading day
- Prices in Korean Won (원)
- Market cap = Current price x Shares outstanding
- 약 15년치 일일 데이터 보유 (2011~현재)

---

### 3. Stock History (Change Log)

**Purpose**: Tracks when company information changed over time.

| Field | Description | Example |
|-------|-------------|---------|
| stock_code | Which company | 035720 |
| effective_date | When change happened | 20230801 |
| name | Name at that time | 카카오 (was 다음) |
| market_type | Market at that time | kospi |
| sector_type | Industry at that time | 인터넷서비스 |

**Key Points**:
- Useful for historical analysis
- Tracks mergers, name changes, market transfers
- Helps avoid "survivorship bias" in backtesting

---

## How Tables Connect

```
┌─────────────┐
│   stocks    │ ←── Master list of all companies
│ (4,749 rows)│
└─────┬───────┘
      │
      │ stock_code (links)
      │
      ├──────────────────────────────────────┐
      │                                      │
      ▼                                      ▼
┌─────────────────┐              ┌──────────────────┐
│  daily_prices   │              │  stock_history   │
│ (8.7M rows)     │              │ (change tracking)│
│                 │              │                  │
│ - OHLCV data    │              │ - Name changes   │
│ - Market cap    │              │ - Market moves   │
│ - Trading value │              │ - Sector changes │
└─────────────────┘              └──────────────────┘
```

---

## Common Use Cases

### 1. Get Today's Large-Cap Stocks
Find stocks with market cap over 5000억 원:
- Filter `daily_prices` where `market_cap > 500,000,000,000`
- Join with `stocks` to get company names

### 2. Historical Price Analysis
Analyze price trends for a specific company:
- Query `daily_prices` by `stock_code` and date range
- Calculate returns, moving averages, etc.

### 3. Track Corporate Events
See when a company changed markets or names:
- Query `stock_history` by `stock_code`
- Useful for understanding stock's journey

---

## Data Quality Notes

| Aspect | Status |
|--------|--------|
| Price Adjustments | Raw prices (not adjusted for splits) |
| Data Completeness | Complete for trading days |
| Update Frequency | Daily after market close |
| Currency | Korean Won (KRW) |
| Time Zone | KST (Korea Standard Time) |

---

## What's NOT in the Database (Yet)

The following data types are available as raw files but not yet integrated:

- **Financial Statements** (재무제표)
  - Balance Sheet (재무상태표)
  - Income Statement (손익계산서)
  - Cash Flow Statement (현금흐름표)
  - Statement of Changes in Equity (자본변동표)

- **ETF Data**
  - KODEX, TIGER 등 ETF 가격 데이터

---

## Quick Reference: Top 10 Large-Cap Stocks (Recent)

| Stock Code | Name | Market Cap (조 원) |
|------------|------|-------------------|
| 005930 | 삼성전자 | 961.3 |
| 000660 | SK하이닉스 | 612.2 |
| 005380 | 현대차 | 112.4 |
| 005935 | 삼성전자우 | 96.2 |
| 207940 | 삼성바이오로직스 | 91.0 |
| 035720 | 카카오 (다음) | 75.2 |
| 035420 | NAVER | 74.6 |
| 051910 | LG화학 | 72.6 |
| 012450 | 한화에어로스페이스 | 68.6 |
| 000270 | 기아 | 67.2 |

---

*Last Updated: 2026-01-30*
