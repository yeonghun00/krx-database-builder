# Financial Statement Data Integration Plan

## Overview

This document outlines how to integrate the raw financial statement data (`raw_financial/`) into our database system. The plan considers data normalization, efficiency, and the critical **45/90 rule** to prevent look-ahead bias.

---

## Raw Data Inventory

### Available Files
- **Period**: 2015 Q4 ~ 2024 Q4 (약 10년치)
- **Total Files**: 160 zip files
- **File Types**: 4 types per quarter

### File Naming Convention
```
{YEAR}_{QUARTER}_{TYPE}_{TIMESTAMP}.zip

Examples:
- 2015_4Q_BS_20230503040109.zip
- 2024_1Q_PL_20250221181911.zip
```

### Statement Types

| Code | Korean Name | English Name | Complexity | Priority |
|------|-------------|--------------|------------|----------|
| BS | 재무상태표 | Balance Sheet | Simple (15 cols) | **Phase 1** |
| PL | 손익계산서 | Income Statement | Medium (18 cols) | **Phase 1** |
| CF | 현금흐름표 | Cash Flow Statement | Medium (16 cols) | **Phase 1** |
| CE | 자본변동표 | Statement of Changes in Equity | Complex (Matrix) | Phase 2 (Optional) |

> **Note**: CE (자본변동표) is skipped in Phase 1. Most CE data (배당금, 자본변동) can be derived from BS + CF. The matrix format requires 10x parsing effort for ~10% extra information.

---

## Raw Data Structure (Per Statement Type)

### Common Columns (1-12) - All Statement Types

| Col | Name | Description | Example |
|-----|------|-------------|---------|
| 1 | 재무제표종류 | Statement type + consolidation | 재무상태표, 유동/비유동법 - 별도 |
| 2 | 종목코드 | Stock code (with brackets) | [005930] |
| 3 | 회사명 | Company name | 삼성전자 |
| 4 | 시장구분 | Market type | 유가증권시장상장법인, 코스닥시장상장법인 |
| 5 | 업종 | Industry code | 264, 291, 715 |
| 6 | 업종명 | Industry name | 반도체 제조업 |
| 7 | 결산월 | Fiscal month (1-12) | 12, 03, 06 |
| 8 | 결산기준일 | Period end date | 2024-03-31 |
| 9 | 보고서종류 | Report type | 사업보고서, 1분기보고서, 반기보고서 |
| 10 | 통화 | Currency | KRW |
| 11 | 항목코드 | IFRS item code | ifrs-full_Assets, dart_OperatingIncomeLoss |
| 12 | 항목명 | Item name in Korean | 자산총계, 영업이익 |

---

### BS (재무상태표) - Balance Sheet

**Structure**: 15 columns (Simple)

| Col | 연간보고서 (4Q) | 분기보고서 (1Q/2Q/3Q) |
|-----|----------------|---------------------|
| 13 | 당기 | 당기 N분기말 |
| 14 | 전기 | 전기말 |
| 15 | 전전기 | 전전기말 |

**Example Data**:
```
재무상태표, 유동/비유동법 - 별도 | [005930] | 삼성전자 | ... | ifrs-full_Assets | 자산총계 | 455,000,000,000,000 | 426,000,000,000,000 | 398,000,000,000,000
```

**Key Point**: BS shows **point-in-time** values (snapshot at period end).

---

### PL (손익계산서) - Income Statement

**Structure**: 18 columns (Complex - varies by report type)

#### For 연간보고서 (4Q Annual Report):

| Col | Header | Description |
|-----|--------|-------------|
| 13 | (empty) | - |
| 14 | 당기 | Current year total |
| 15-16 | (empty) | - |
| 17 | 전기 | Previous year total |
| 18 | 전전기 | 2 years ago total |

#### For 분기보고서 (Quarterly Report):

| Col | Header | Description |
|-----|--------|-------------|
| 13 | 당기 N분기 3개월 | Current quarter only (3 months) |
| 14 | 당기 N분기 누적 | Year-to-date cumulative |
| 15 | 전기 N분기 3개월 | Same quarter last year (3 months) |
| 16 | 전기 N분기 누적 | Last year YTD cumulative |
| 17 | 전기 | Previous full year |
| 18 | 전전기 | 2 years ago full year |

**Example Data (Quarterly)**:
```
손익계산서, 기능별 분류 - 별도 | [001040] | CJ | ... | dart_OperatingIncomeLoss | 영업이익 | 67,501,188,000 | 67,501,188,000 | 83,405,860,000 | 83,405,860,000 | |
```

**Key Point**: PL shows **flow** values (activity during period). Use **누적** for YTD, **3개월** for single quarter.

---

### CF (현금흐름표) - Cash Flow Statement

**Structure**: 16 columns

#### For 연간보고서 (4Q):

| Col | Header | Description |
|-----|--------|-------------|
| 13 | 당기 | Current year total |
| 14 | (empty) | - |
| 15 | 전기 | Previous year |
| 16 | 전전기 | 2 years ago |

#### For 분기보고서 (Quarterly):

| Col | Header | Description |
|-----|--------|-------------|
| 13 | 당기N분기 | Current YTD |
| 14 | 전기N분기 | Last year same period YTD |
| 15 | 전기 | Previous full year |
| 16 | 전전기 | 2 years ago |

**Example Data**:
```
현금흐름표, 간접법 - 별도 | [060310] | 3S | ... | ifrs_CashFlowsFromUsedInOperatingActivities | 영업활동현금흐름 | 292,361,418 | | 2,296,140,969 | 2,962,155,040
```

**Key Point**: CF shows **flow** values like PL. Cumulative for the period.

---

### CE (자본변동표) - Statement of Changes in Equity

> **⚠️ PHASE 2 - OPTIONAL**: CE is skipped in initial implementation. See "Why Skip CE?" below.

**Structure**: MATRIX FORMAT (Variable columns, 30+ cols typical)

This is fundamentally different from BS/PL/CF. It's a **pivot table** structure.

#### Header Rows (First 5-6 rows per company):

| Row | Purpose | Example Content |
|-----|---------|-----------------|
| 1 | Column group codes | ifrs-full_EquityMember repeated |
| 2 | Sub-column codes | ifrs-full_IssuedCapitalMember, dart_CapitalSurplusMember |
| 3 | Column group names | 자본 [구성요소] repeated |
| 4 | Sub-column names | 자본금, 주식발행초과금, 감자차익, 이익잉여금, 자본합계 |
| 5+ | Data rows | 기초자본, 유상증자, 당기순이익, 기말자본 등 |

#### Typical Column Categories:

```
자본금 | 주식발행초과금 | 감자차익 | 전환권대가 | 신주인수권대가 |
자기주식처분손실 | 기타자본구성요소 | 이익잉여금(결손금) | 자본 합계
```

#### Example Data Structure:
```
Col 11-12: 항목코드, 항목명
Col 13+: Dynamic columns for each equity component

Row: dart_EquityAtBeginningOfPeriod | 기초자본 | 19,530,576,500 | 18,111,436,602 | 200,000,000 | ... | 29,392,981,131
Row: ifrs-full_Equity | 기말자본 | 20,446,134,000 | 35,657,511,841 | ... | 37,669,538,708
```

**Key Point**: CE requires **special parsing logic**. Cannot use same parser as BS/PL/CF.

#### Why Skip CE in Phase 1?

| CE Data | Alternative Source | Notes |
|---------|-------------------|-------|
| 배당금 | CF (재무활동현금흐름) | `dart_PaymentOfDividends` |
| 기말자본 | BS (자본총계) | `ifrs-full_Equity` - identical |
| 유상증자 | CF (재무활동현금흐름) | `ProceedsFromIssuingShares` |
| 자사주매입 | CF (재무활동현금흐름) | `PurchaseOfTreasuryShares` |

**Conclusion**: 95% of CE information is available from BS + CF. Matrix parsing adds significant complexity for minimal ML value.

---

## Item Code Evolution

**Important**: IFRS codes changed over time!

| Period | Code Format | Example |
|--------|-------------|---------|
| 2015-2017 | `ifrs_` prefix | `ifrs_Assets`, `ifrs_Revenue` |
| 2018+ | `ifrs-full_` prefix | `ifrs-full_Assets`, `ifrs-full_Revenue` |
| All periods | `dart_` prefix | `dart_OperatingIncomeLoss` (DART-specific) |
| All periods | `entity00xxx_udf_` | Company-specific custom items |

**Solution**: Create mapping table to normalize codes:
```python
ITEM_CODE_MAPPING = {
    'ifrs_Assets': 'ifrs-full_Assets',
    'ifrs_Revenue': 'ifrs-full_Revenue',
    'ifrs_Equity': 'ifrs-full_Equity',
    # ... more mappings
}
```

---

## The 45/90 Rule (Look-Ahead Bias Prevention)

### Why This Matters

Financial data becomes **publicly available** AFTER the filing deadline, NOT on the fiscal period end date. Using data before it's actually available creates **look-ahead bias** - a fatal flaw in backtesting.

### Legal Disclosure Deadlines (공시 기한)

| Quarter | Fiscal End | Filing Deadline | Safe Available Date |
|---------|------------|-----------------|---------------------|
| Q1 | 3월 31일 | 5월 15일 (45일) | **5월 16일** |
| Q2 | 6월 30일 | 8월 15일 (45일) | **8월 16일** |
| Q3 | 9월 30일 | 11월 14일 (45일) | **11월 15일** |
| Q4 (연간) | 12월 31일 | 3월 31일 (90일) | **4월 1일** |

### Implementation Strategy

```python
def get_available_date(fiscal_date: str, fiscal_month: int) -> str:
    """
    Calculate when financial data becomes publicly available.

    Args:
        fiscal_date: Period end date (YYYY-MM-DD)
        fiscal_month: Company's fiscal year end month (결산월)

    Returns:
        First date the data can be used in trading (YYYYMMDD)
    """
    year, month, day = fiscal_date.split('-')
    year, month = int(year), int(month)

    # Determine if this is Q1, Q2, Q3, or Q4 based on fiscal_month
    if fiscal_month == 12:  # Standard Dec fiscal year
        if month == 3:   return f"{year}0516"      # Q1
        elif month == 6: return f"{year}0816"      # Q2
        elif month == 9: return f"{year}1115"      # Q3
        elif month == 12: return f"{year + 1}0401" # Q4
    else:
        # Non-standard fiscal year (e.g., March fiscal year)
        # Calculate based on quarters from fiscal_month
        # ... custom logic needed
        pass
```

---

## Proposed Database Schema

### Phase 1 Schema (BS/PL/CF - All Columns Preserved)

```sql
-- Financial statement metadata (period-level)
CREATE TABLE financial_periods (
    id INTEGER PRIMARY KEY,
    stock_code TEXT NOT NULL,
    company_name TEXT,                    -- 회사명
    market_type TEXT,                     -- 시장구분
    industry_code TEXT,                   -- 업종
    industry_name TEXT,                   -- 업종명
    fiscal_month INTEGER,                 -- 결산월
    fiscal_date TEXT NOT NULL,            -- 결산기준일 (YYYY-MM-DD)
    available_date TEXT NOT NULL,         -- 45/90 rule applied (YYYYMMDD)
    report_type TEXT,                     -- 보고서종류 (사업보고서, 분기보고서)
    consolidation_type TEXT,              -- 별도/연결
    currency TEXT DEFAULT 'KRW',          -- 통화
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(stock_code, fiscal_date, consolidation_type)
);

-- BS/CF items (simple structure)
CREATE TABLE financial_items_bs_cf (
    id INTEGER PRIMARY KEY,
    period_id INTEGER NOT NULL REFERENCES financial_periods(id),
    statement_type TEXT NOT NULL,         -- BS or CF
    item_code TEXT NOT NULL,              -- 항목코드
    item_name TEXT NOT NULL,              -- 항목명
    amount_current REAL,                  -- 당기 (or 당기N분기말/당기N분기)
    amount_prev REAL,                     -- 전기 (or 전기말/전기N분기)
    amount_prev2 REAL,                    -- 전전기 (or 전전기말)
    UNIQUE(period_id, statement_type, item_code)
);

-- PL items (complex structure with quarterly breakdown)
CREATE TABLE financial_items_pl (
    id INTEGER PRIMARY KEY,
    period_id INTEGER NOT NULL REFERENCES financial_periods(id),
    item_code TEXT NOT NULL,              -- 항목코드
    item_name TEXT NOT NULL,              -- 항목명
    -- For quarterly reports:
    amount_current_qtr REAL,              -- 당기 N분기 3개월
    amount_current_ytd REAL,              -- 당기 N분기 누적
    amount_prev_qtr REAL,                 -- 전기 N분기 3개월
    amount_prev_ytd REAL,                 -- 전기 N분기 누적
    -- For annual reports (or reference):
    amount_prev_year REAL,                -- 전기 (연간)
    amount_prev2_year REAL,               -- 전전기 (연간)
    UNIQUE(period_id, item_code)
);

-- Indexes
CREATE INDEX idx_periods_stock_date ON financial_periods(stock_code, fiscal_date);
CREATE INDEX idx_periods_available ON financial_periods(available_date);
CREATE INDEX idx_bs_cf_period ON financial_items_bs_cf(period_id);
CREATE INDEX idx_pl_period ON financial_items_pl(period_id);

-- ============================================
-- PHASE 2 (Optional): CE table
-- ============================================
-- CREATE TABLE financial_items_ce (
--     id INTEGER PRIMARY KEY,
--     period_id INTEGER NOT NULL REFERENCES financial_periods(id),
--     row_code TEXT NOT NULL,               -- 행 항목코드 (기초자본, 유상증자 등)
--     row_name TEXT NOT NULL,               -- 행 항목명
--     column_code TEXT NOT NULL,            -- 열 항목코드 (자본금, 이익잉여금 등)
--     column_name TEXT NOT NULL,            -- 열 항목명
--     amount REAL,                          -- 값
--     UNIQUE(period_id, row_code, column_code)
-- );
-- CREATE INDEX idx_ce_period ON financial_items_ce(period_id);
```

---

## Parsing Strategy by Statement Type

### BS/CF Parser (Simple)
```python
def parse_bs_cf(df: pd.DataFrame, statement_type: str) -> pd.DataFrame:
    """
    Parse Balance Sheet or Cash Flow data.
    Columns 13-15 (or 13-16 for CF) are value columns.
    """
    cols = df.columns.tolist()

    # Map columns based on report type
    if '분기' in df.iloc[0]['보고서종류']:
        value_cols = {
            'amount_current': cols[12],  # 당기 N분기말/당기N분기
            'amount_prev': cols[13] if len(cols) > 13 else None,
            'amount_prev2': cols[14] if len(cols) > 14 else None,
        }
    else:  # 사업보고서 (Annual)
        value_cols = {
            'amount_current': cols[12],  # 당기
            'amount_prev': cols[13],     # 전기
            'amount_prev2': cols[14],    # 전전기
        }

    return df, value_cols
```

### PL Parser (Medium Complexity)
```python
def parse_pl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Income Statement data.
    Handle both annual (당기/전기/전전기) and quarterly
    (3개월/누적 for current and previous year).
    """
    cols = df.columns.tolist()

    if '분기' in df.iloc[0]['보고서종류']:
        # Quarterly: cols 13-18
        value_cols = {
            'amount_current_qtr': cols[12],   # 당기 N분기 3개월
            'amount_current_ytd': cols[13],   # 당기 N분기 누적
            'amount_prev_qtr': cols[14],      # 전기 N분기 3개월
            'amount_prev_ytd': cols[15],      # 전기 N분기 누적
            'amount_prev_year': cols[16],     # 전기
            'amount_prev2_year': cols[17],    # 전전기
        }
    else:  # Annual
        # Annual: cols 13 (empty), 14 (당기), 15-16 (empty), 17 (전기), 18 (전전기)
        value_cols = {
            'amount_current_ytd': cols[13],   # 당기
            'amount_prev_year': cols[16],     # 전기
            'amount_prev2_year': cols[17],    # 전전기
        }

    return df, value_cols
```

### CE Parser (Phase 2 - Optional)

> **Note**: CE parsing is deferred to Phase 2. The code below is for future reference.

<details>
<summary>Click to expand CE parser code (Phase 2)</summary>

```python
def parse_ce(df: pd.DataFrame) -> List[dict]:
    """
    Parse Statement of Changes in Equity.
    This requires special handling due to matrix format.

    Strategy:
    1. Read first 4-5 rows to build column mapping
    2. Extract column names from row 4 (자본금, 주식발행초과금, etc.)
    3. Parse data rows (row 5+) with row_code/row_name and values
    """
    # Step 1: Find header rows (contain column definitions)
    header_rows = []
    data_start_idx = 0

    for idx, row in df.iterrows():
        if pd.notna(row['항목코드']) and row['항목코드'] != '':
            data_start_idx = idx
            break
        header_rows.append(row)

    # Step 2: Build column mapping from header rows
    # Row with Korean names (자본금, 주식발행초과금, etc.) is typically row 4
    column_names = header_rows[-1].iloc[12:].tolist()
    column_codes = header_rows[-2].iloc[12:].tolist() if len(header_rows) > 1 else column_names

    # Step 3: Parse data rows
    results = []
    for idx in range(data_start_idx, len(df)):
        row = df.iloc[idx]
        row_code = row['항목코드']
        row_name = row['항목명']

        for col_idx, (col_code, col_name) in enumerate(zip(column_codes, column_names)):
            value = row.iloc[12 + col_idx]
            if pd.notna(value) and value != '':
                results.append({
                    'row_code': row_code,
                    'row_name': row_name,
                    'column_code': col_code,
                    'column_name': col_name,
                    'amount': parse_number(value)
                })

    return results
```

</details>

---

## Key Financial Items to Extract

### Balance Sheet (BS) - Priority Items

| IFRS Code | Korean | Use Case |
|-----------|--------|----------|
| ifrs-full_Assets | 자산총계 | Size, ROA |
| ifrs-full_Equity | 자본총계 | ROE, Book Value |
| ifrs-full_Liabilities | 부채총계 | Debt Ratio |
| ifrs-full_CurrentAssets | 유동자산 | Liquidity |
| ifrs-full_CurrentLiabilities | 유동부채 | Current Ratio |
| ifrs-full_CashAndCashEquivalents | 현금및현금성자산 | Cash Position |
| ifrs-full_Inventories | 재고자산 | Inventory Turnover |

### Income Statement (PL) - Priority Items

| IFRS Code | Korean | Use Case |
|-----------|--------|----------|
| ifrs-full_Revenue | 매출액 | Growth Rate |
| dart_OperatingIncomeLoss | 영업이익 | Profitability |
| ifrs-full_ProfitLoss | 당기순이익 | EPS, PE |
| ifrs-full_GrossProfit | 매출총이익 | Margin Analysis |

### Cash Flow (CF) - Priority Items

| IFRS Code | Korean | Use Case |
|-----------|--------|----------|
| ifrs-full_CashFlowsFromUsedInOperatingActivities | 영업활동현금흐름 | Cash Generation |
| ifrs-full_CashFlowsFromUsedInInvestingActivities | 투자활동현금흐름 | CapEx |
| ifrs-full_CashFlowsFromUsedInFinancingActivities | 재무활동현금흐름 | Debt/Equity Changes |

### Changes in Equity (CE) - Phase 2 Optional

> **Skipped in Phase 1** - Use CF for dividend data instead.

| Row Item | Korean | Alternative Source |
|----------|--------|-------------------|
| dart_EquityAtBeginningOfPeriod | 기초자본 | BS 전기 자본총계 |
| ifrs-full_Equity | 기말자본 | BS 당기 자본총계 |
| ifrs-full_ProfitLoss | 당기순이익 | PL 당기순이익 |
| dart_Dividends | 배당금 | CF 재무활동현금흐름 |

---

## Data Quality Considerations

### 1. Empty Values
- Many cells contain empty strings or just whitespace
- Parse as NULL, not 0

### 2. Number Format
- Numbers contain commas: `1,234,567,890`
- Some negative numbers: `-1,234,567,890`
- Need to strip commas and convert to float

### 3. Duplicate Entries
- Same company may appear multiple times (별도 + 연결)
- **Prefer 연결 (consolidated)** for analysis
- Store both but flag consolidation type

### 4. Company-Specific Codes
- Many items use `entity00xxxxx_udf_` prefix
- These are company-specific custom items
- Map to standard IFRS codes where possible, otherwise store as-is

---

## Summary Table for ML

After loading raw data, create a summary table for ML features:

```sql
CREATE TABLE financial_summary (
    stock_code TEXT NOT NULL,
    fiscal_date TEXT NOT NULL,
    available_date TEXT NOT NULL,
    consolidation_type TEXT,

    -- Balance Sheet (point-in-time)
    total_assets REAL,
    total_liabilities REAL,
    total_equity REAL,
    current_assets REAL,
    current_liabilities REAL,
    cash REAL,
    inventory REAL,

    -- Income Statement (period flow)
    revenue REAL,
    revenue_qtr REAL,              -- Single quarter
    operating_income REAL,
    operating_income_qtr REAL,
    net_income REAL,
    net_income_qtr REAL,

    -- Cash Flow (period flow)
    operating_cf REAL,
    investing_cf REAL,
    financing_cf REAL,

    -- Derived Ratios
    roe REAL,
    roa REAL,
    debt_ratio REAL,
    current_ratio REAL,
    operating_margin REAL,

    PRIMARY KEY(stock_code, fiscal_date, consolidation_type)
);
```

---

## Implementation Phases

### Phase 1A: Schema & Infrastructure (BS/PL/CF only)
- [ ] Create 3 tables: financial_periods, financial_items_bs_cf, financial_items_pl
- [ ] Write parsing utilities for BS, PL, CF
- [ ] Implement 45/90 rule logic
- [ ] Create item code mapping table (ifrs_ → ifrs-full_)

### Phase 1B: Data Loading (BS/PL/CF)
- [ ] Process all 160 zip files (skip CE files)
- [ ] Handle CP949 encoding
- [ ] Validate data integrity
- [ ] Log parsing errors

### Phase 1C: Summary & Feature Engineering
- [ ] Create financial_summary table
- [ ] Calculate derived ratios (ROE, ROA, etc.)
- [ ] Implement TTM calculations
- [ ] Add to ML feature pipeline

### Phase 1D: Integration & Testing
- [ ] Update `features.py` to use financial data
- [ ] Backtest with new features
- [ ] Validate no look-ahead bias

### Phase 2 (Optional): CE Integration
- [ ] Create financial_items_ce table
- [ ] Write matrix parser for CE
- [ ] Extract 배당금, 자본변동 details
- [ ] Add dividend-related ML features

---

## Estimated Data Volume

### Phase 1 (BS/PL/CF)

| Table | Estimated Rows |
|-------|---------------|
| financial_periods | ~100,000 (2,500 companies x 40 quarters) |
| financial_items_bs_cf | ~2,000,000 |
| financial_items_pl | ~1,500,000 |
| financial_summary | ~100,000 |

**Phase 1 Storage**: ~500MB additional SQLite space

### Phase 2 (Optional - CE)

| Table | Estimated Rows |
|-------|---------------|
| financial_items_ce | ~500,000 |

**Phase 2 Storage**: ~100MB additional

---

## Key Takeaways

1. **Phase 1 focuses on BS/PL/CF** - Skip CE (자본변동표) for now
2. **Different parsers needed** for BS/CF (simple) vs PL (medium)
3. **Item codes changed** from `ifrs_` to `ifrs-full_` around 2018
4. **Quarterly vs Annual** reports have different column structures
5. **45/90 rule is critical** - always use `available_date`, never `fiscal_date`
6. **Prefer 연결 (consolidated)** over 별도 (separate) for analysis
7. **CE is redundant** - 배당금/자본변동 available from CF (현금흐름표)

---

*Document Version: 2.1*
*Updated: 2026-01-30*
