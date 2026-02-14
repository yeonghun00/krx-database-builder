# AlgoStock Overview

A quantitative stock-picking system for Korean equities (KOSPI/KOSDAQ). It pulls market data from KRX, engineers ~30 features, trains a LightGBM ranking model via walk-forward validation, and outputs ranked stock picks.

## Architecture

```
KRX APIs / Raw Files
        |
   ETL Pipelines  ──►  krx_stock_data.db (SQLite)
        |
  Feature Engineering  (ml/features.py)
        |
  LightGBM Ranker      (ml/model.py)
        |
   ┌────┴────┐
Backtest     Picks
(scripts/    (scripts/
run_backtest get_picks
.py)         .py)
```

## Directory Structure

```
algostock/
├── etl/                  # Data ingestion pipelines
│   ├── clean_etl.py              # Prices + stock master
│   ├── index_constituents_etl.py # Index membership snapshots
│   ├── delisted_stocks_etl.py    # Delisted stock list
│   └── financial_etl.py          # Financial statements (BS/PL/CF)
├── ml/                   # ML pipeline
│   ├── features.py               # Feature engineering (FeatureEngineer)
│   ├── model.py                  # LightGBM wrapper (MLRanker)
│   └── backtest.py               # Legacy backtester (unused by main scripts)
├── scripts/              # Entry points
│   ├── run_backtest.py           # Train + backtest + save model
│   └── get_picks.py              # Generate today's picks from trained model
├── models/               # Saved model artifacts
│   └── lgbm_unified.pkl
├── data/
│   └── raw_financial/            # Raw financial statement ZIPs
├── docs/                 # You are here
└── krx_stock_data.db     # SQLite database (all market data)
```

## End-to-End Workflow

### Step 1: Refresh Data (ETL)

Run in this order. Each pipeline is independent but later ones assume prices exist.

```bash
# 1. Prices + stock master (takes longest)
python3 etl/clean_etl.py --daily-update --date 20260214 --markets kospi,kosdaq

# 2. Index constituents
python3 etl/index_constituents_etl.py --mode update --strategy skip --workers 4 --config config.json

# 3. Delisted stocks
python3 etl/delisted_stocks_etl.py

# 4. Financial statements
python3 etl/financial_etl.py krx_stock_data.db data/raw_financial
```

See [ETL.md](ETL.md) for full details and backfill commands.

### Step 2: Train Model + Backtest

```bash
python3 scripts/run_backtest.py \
  --start 20100101 --end 20251231 \
  --workers 4 --no-cache
```

This does everything: builds features, runs walk-forward train/test splits, simulates portfolio rebalancing with transaction costs, saves the trained model to `models/lgbm_unified.pkl`, and writes result CSVs.

See [MODEL.md](MODEL.md) for backtest mechanics, features, and all CLI flags.

### Step 3: Generate Picks

```bash
python3 scripts/get_picks.py \
  --start 20220101 --end 20260214 \
  --top 20 --bottom 10
```

Outputs a ranked list of top picks (buy) and bottom picks (avoid) for the latest date in the data. Also saves a full ranking CSV.

## Key Design Principles

**Point-in-time (PIT) safety** -- Financial data is only used after its `available_date` (45/90-day rule). Index membership is matched by month. Delisted stocks are excluded before their delist date. No future information leaks into training or evaluation.

**Walk-forward validation** -- The model is never tested on data it trained on. Training uses a rolling N-year window, and testing is always the next calendar year.

**Transaction-cost-aware** -- Backtest applies buy/sell fees on every rebalance. Hysteresis rules reduce unnecessary turnover. Stress mode doubles fees for conservative estimates.

**Sector-aware** -- Features include sector-relative momentum, sector z-scores, sector breadth, and rotation signals. Scoring can optionally be sector-neutralized.

## Output Files

| File | Description |
|------|-------------|
| `backtest_unified_results.csv` | Per-rebalance returns, alpha, IC, quintile stats |
| `*_rolling_sharpe.csv` | Rolling 12-period Sharpe ratio |
| `*_quintiles.csv` | Average return by model-score quintile (Q1-Q5) |
| `*_sector_attribution.csv` | Sector weight and contribution per rebalance |
| `models/lgbm_unified.pkl` | Trained LightGBM model (pickle) |
| `picks_unified_YYYYMMDD.csv` | Full stock ranking for the latest date |

## Key Metrics to Watch

- **Mean IC (Spearman)**: Rank correlation between model scores and actual forward returns. > 0.03 is decent.
- **Quintile monotonicity**: Q5 mean return should be > Q4 > Q3 > Q2 > Q1. If not, the model's ranking power is broken.
- **Average turnover**: Lower is better after fees. Hysteresis helps here.
- **Max drawdown / underwater duration**: How bad it gets and how long recovery takes.
- **Sharpe / Calmar**: Risk-adjusted performance ratios.

## Quick Validation Commands

```bash
# Check DB tables exist
sqlite3 krx_stock_data.db "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"

# Check data freshness
sqlite3 krx_stock_data.db "SELECT MAX(date) FROM daily_prices;"
sqlite3 krx_stock_data.db "SELECT MAX(date) FROM index_constituents;"
sqlite3 krx_stock_data.db "SELECT COUNT(*) FROM financial_periods;"
```
