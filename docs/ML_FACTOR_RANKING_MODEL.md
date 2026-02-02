# ML Factor-Ranking Model for Korean Stock Alpha

A comprehensive guide to building a Machine Learning-based factor ranking system using 10+ years of KRX market data.

---

## Table of Contents

1. [Overview](#overview)
2. [Why This Approach Works](#why-this-approach-works)
3. [Phase 1: Feature Engineering](#phase-1-feature-engineering)
4. [Phase 2: Target Labeling](#phase-2-target-labeling)
5. [Phase 3: Model Architecture](#phase-3-model-architecture)
6. [Phase 4: Walk-Forward Validation](#phase-4-walk-forward-validation)
7. [Phase 5: Implementation](#phase-5-implementation)
8. [Phase 6: Analysis & Monitoring](#phase-6-analysis--monitoring)
9. [Database Schema Requirements](#database-schema-requirements)
10. [Code Implementation](#code-implementation)

---

## Overview

### The Problem with Traditional Approaches

| Approach | Problem |
|----------|---------|
| Predict if stock goes up/down | Too noisy, ~51% accuracy at best |
| Predict exact price | Impossible, markets are efficient |
| Simple momentum/value rules | Linear, misses complex patterns |

### The Solution: Cross-Sectional Ranking

Instead of predicting absolute returns, **rank all stocks relative to each other** and buy the top performers.

```
Traditional: "Will Samsung go up 5%?" → Very hard
Ranking:     "Will Samsung outperform LG?" → Much easier
```

### Why LightGBM?

| Model | Pros | Cons |
|-------|------|------|
| **LightGBM** | Fast, handles tabular data well, feature importance | Requires feature engineering |
| XGBoost | Similar to LightGBM | Slower training |
| LSTM/Deep Learning | Can learn features automatically | Needs massive data, overfits on finance |
| Linear Regression | Simple, interpretable | Misses non-linear patterns |

**Verdict**: LightGBM is the industry standard for tabular financial data.

---

## Why This Approach Works

### Korean Market Characteristics

1. **High Retail Participation**
   - KOSDAQ has significant retail trading
   - Creates predictable volume/momentum patterns
   - ML excels at detecting these patterns

2. **Small-Cap Effect**
   - Small caps in Korea show persistent alpha
   - But high volatility requires proper risk management
   - ML can identify "good" small caps vs "bad" ones

3. **Transaction Costs**
   - Securities Transaction Tax: ~0.18-0.20%
   - Kills high-frequency strategies
   - Monthly/6-month rebalancing is optimal

4. **Non-Linear Relationships**
   ```
   Linear Rule:  Score = 0.5×Momentum + 0.5×Size

   ML Rule:      IF (MarketCap < 500B)
                 AND (Volume > 3× Average)
                 AND (Price near 52-week high)
                 THEN alpha_probability = 80%
   ```

---

## Phase 1: Feature Engineering

Feature engineering is **the most important step**. Raw OHLCV data is useless—you must transform it into meaningful "alpha factors."

### 1.1 Size Factors

```python
# Log Market Cap (reduces skewness)
log_market_cap = np.log(market_cap)

# Market Cap Rank (cross-sectional)
market_cap_rank = market_cap.rank(pct=True)  # 0 to 1
```

**Why it matters**: In Korea, small-cap stocks often outperform, but with higher risk.

### 1.2 Liquidity Factors

```python
# Turnover Ratio
turnover_ratio = traded_value / market_cap

# Relative Volume (RVOL)
avg_volume_20d = volume.rolling(20).mean()
rvol = volume / avg_volume_20d

# Amihud Illiquidity (lower = more liquid)
daily_return = close.pct_change()
amihud = abs(daily_return) / traded_value
amihud_20d = amihud.rolling(20).mean()
```

**Why it matters**: Volume spikes often precede price moves. Illiquid stocks have higher expected returns but higher risk.

### 1.3 Momentum Factors

```python
# Simple Momentum
mom_1m = close / close.shift(21) - 1      # 1-month
mom_3m = close / close.shift(63) - 1      # 3-month
mom_6m = close / close.shift(126) - 1     # 6-month
mom_12m = close / close.shift(252) - 1    # 12-month

# Momentum with Skip (avoid short-term reversal)
# This is a VERY strong signal in Korea
mom_12m_skip_1m = close.shift(21) / close.shift(252) - 1

# Weighted Momentum (recent matters more)
weighted_mom = (0.5 * mom_1m) + (0.3 * mom_3m) + (0.2 * mom_6m)
```

**Key Insight**: In Korea, `12-month minus 1-month` momentum is one of the strongest alpha signals.

### 1.4 Volatility Factors

```python
# Historical Volatility
returns = close.pct_change()
volatility_20d = returns.rolling(20).std() * np.sqrt(252)  # Annualized
volatility_60d = returns.rolling(60).std() * np.sqrt(252)

# Volatility Ratio (regime detection)
vol_ratio = volatility_20d / volatility_60d

# Downside Volatility (only negative returns)
downside_returns = returns.where(returns < 0, 0)
downside_vol = downside_returns.rolling(20).std() * np.sqrt(252)
```

**Why it matters**: Lower volatility often leads to better risk-adjusted returns (Low-Vol Anomaly).

### 1.5 Price Location Factors

```python
# Price relative to 52-week range
high_52w = high.rolling(252).max()
low_52w = low.rolling(252).min()
price_location = (close - low_52w) / (high_52w - low_52w)

# Distance from Moving Averages
ma_20 = close.rolling(20).mean()
ma_60 = close.rolling(60).mean()
ma_120 = close.rolling(120).mean()

dist_ma_20 = (close - ma_20) / ma_20
dist_ma_60 = (close - ma_60) / ma_60
dist_ma_120 = (close - ma_120) / ma_120

# Golden/Death Cross Signal
ma_cross = (ma_20 > ma_60).astype(int)
```

**Why it matters**: Stocks near 52-week highs often continue (breakout momentum).

### 1.6 Value Factors (if available)

```python
# If you have fundamental data:
# earnings_yield = earnings / price
# book_to_market = book_value / market_cap

# Proxy with price data only:
# Price-to-Sales approximation using market cap and volume
implied_revenue_proxy = traded_value.rolling(252).sum()
price_to_volume = market_cap / implied_revenue_proxy
```

### 1.7 Technical Factors

```python
# RSI (Relative Strength Index)
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

rsi_14 = calculate_rsi(close, 14)

# MACD
ema_12 = close.ewm(span=12).mean()
ema_26 = close.ewm(span=26).mean()
macd = ema_12 - ema_26
macd_signal = macd.ewm(span=9).mean()
macd_histogram = macd - macd_signal

# Bollinger Band Position
bb_middle = close.rolling(20).mean()
bb_std = close.rolling(20).std()
bb_upper = bb_middle + 2 * bb_std
bb_lower = bb_middle - 2 * bb_std
bb_position = (close - bb_lower) / (bb_upper - bb_lower)
```

### Feature Summary Table

| Category | Features | Count |
|----------|----------|-------|
| Size | log_market_cap, market_cap_rank | 2 |
| Liquidity | turnover_ratio, rvol, amihud_20d | 3 |
| Momentum | mom_1m, mom_3m, mom_6m, mom_12m, mom_12m_skip_1m | 5 |
| Volatility | vol_20d, vol_60d, vol_ratio, downside_vol | 4 |
| Price Location | price_location, dist_ma_20/60/120, ma_cross | 5 |
| Technical | rsi_14, macd_histogram, bb_position | 3 |
| **Total** | | **22** |

---

## Phase 2: Target Labeling

### The Wrong Way: Absolute Returns

```python
# DON'T DO THIS
target = future_return  # e.g., +5%, -3%
```

**Problem**: If the market drops 10%, a stock that drops 5% is actually a "winner" but would be labeled as a loser.

### The Right Way: Cross-Sectional Rank

```python
# DO THIS
def create_ranking_target(df, forward_days=21):
    """
    Create cross-sectional ranking target.

    For each date, rank all stocks by their future return.
    Convert to percentile (0 to 1).
    """
    # Calculate forward return
    df['forward_return'] = df.groupby('stock_code')['close'].pct_change(forward_days).shift(-forward_days)

    # Rank within each date (cross-sectional)
    df['target'] = df.groupby('date')['forward_return'].rank(pct=True)

    return df
```

### Target Horizons

| Horizon | Forward Days | Use Case |
|---------|--------------|----------|
| 1 Month | 21 | Higher turnover, more signals |
| 3 Month | 63 | Balanced |
| 6 Month | 126 | Lower turnover, tax efficient |

### Example

```
Date: 2024-01-01

Stock          Future Return    Rank (Target)
────────────────────────────────────────────
Samsung        +8%              0.95 (top 5%)
SK Hynix       +5%              0.85
Kakao          +2%              0.60
Naver          -1%              0.40
LG Energy      -5%              0.15
Celltrion      -10%             0.05 (bottom 5%)
```

Even if the whole market dropped, Samsung (+8%) is still labeled as a winner (0.95).

---

## Phase 3: Model Architecture

### LightGBM Configuration

```python
import lightgbm as lgb

# For Ranking (Recommended)
params_ranker = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [10, 20, 50],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 500,
    'early_stopping_rounds': 50
}

# For Regression (Alternative)
params_regressor = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'verbose': -1,
    'n_estimators': 500
}
```

### Why LambdaRank?

| Objective | Goal | Best For |
|-----------|------|----------|
| `regression` | Minimize prediction error | Predicting exact values |
| `binary` | Classify up/down | Binary signals |
| **`lambdarank`** | Get ranking order correct | Stock selection |

LambdaRank focuses on **ranking the best stocks at the top**, not predicting exact returns.

### Training with Groups

For ranking, LightGBM needs to know which samples belong together (same date):

```python
def train_ranker(X_train, y_train, dates_train, X_val, y_val, dates_val):
    """
    Train LightGBM ranker with proper grouping.
    """
    # Calculate group sizes (stocks per date)
    train_groups = dates_train.value_counts().sort_index().values
    val_groups = dates_val.value_counts().sort_index().values

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=train_groups
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=val_groups,
        reference=train_data
    )

    model = lgb.train(
        params_ranker,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50)]
    )

    return model
```

---

## Phase 4: Walk-Forward Validation

### Why Not Random Split?

```
❌ Random Split (WRONG for time series):
   Train: 2015, 2018, 2021 (random mix)
   Test:  2016, 2019, 2022

   Problem: Model "sees" future data during training → Look-ahead bias
```

```
✅ Walk-Forward (CORRECT):
   Fold 1: Train 2011-2015, Test 2016
   Fold 2: Train 2011-2016, Test 2017
   Fold 3: Train 2011-2017, Test 2018
   ...

   Model NEVER sees future data
```

### Implementation

```python
def walk_forward_validation(df, feature_cols, target_col,
                            train_years=5, test_years=1):
    """
    Walk-forward validation for time series.

    Args:
        df: DataFrame with 'date' column
        feature_cols: List of feature column names
        target_col: Target column name
        train_years: Number of years for training
        test_years: Number of years for testing

    Yields:
        (train_data, test_data, fold_info)
    """
    df['year'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.year
    years = sorted(df['year'].unique())

    results = []

    for i in range(train_years, len(years)):
        train_end_year = years[i - 1]
        test_year = years[i]
        train_start_year = years[max(0, i - train_years)]

        # Split data
        train_mask = (df['year'] >= train_start_year) & (df['year'] <= train_end_year)
        test_mask = df['year'] == test_year

        train_data = df[train_mask]
        test_data = df[test_mask]

        fold_info = {
            'train_years': f"{train_start_year}-{train_end_year}",
            'test_year': test_year,
            'train_samples': len(train_data),
            'test_samples': len(test_data)
        }

        yield train_data, test_data, fold_info


# Usage
for train_df, test_df, info in walk_forward_validation(df, features, 'target'):
    print(f"Training: {info['train_years']}, Testing: {info['test_year']}")

    X_train = train_df[features]
    y_train = train_df['target']

    X_test = test_df[features]
    y_test = test_df['target']

    # Train and evaluate
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate ranking performance
    evaluate_ranking(test_df, predictions)
```

### Validation Metrics

```python
def evaluate_ranking(df, predictions, top_n=20):
    """
    Evaluate ranking model performance.
    """
    df = df.copy()
    df['prediction'] = predictions

    results = []

    for date, group in df.groupby('date'):
        # Sort by prediction (descending)
        group = group.sort_values('prediction', ascending=False)

        # Get top N stocks
        top_stocks = group.head(top_n)

        # Calculate metrics
        results.append({
            'date': date,
            'avg_forward_return': top_stocks['forward_return'].mean(),
            'avg_target_rank': top_stocks['target'].mean(),
            'hit_rate': (top_stocks['target'] > 0.5).mean()  # % in top half
        })

    results_df = pd.DataFrame(results)

    print(f"Average Forward Return: {results_df['avg_forward_return'].mean():.2%}")
    print(f"Average Target Rank: {results_df['avg_target_rank'].mean():.2f}")
    print(f"Hit Rate (top half): {results_df['hit_rate'].mean():.2%}")

    return results_df
```

---

## Phase 5: Implementation

### Monthly Trading Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    MONTHLY WORKFLOW                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Day 1 of Month                                              │
│  ┌──────────────┐                                           │
│  │   FILTER     │  Remove illiquid & penny stocks           │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │   COMPUTE    │  Calculate all features for today         │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │   PREDICT    │  Run LightGBM model                       │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │   SELECT     │  Pick top 20 by model score               │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │   EXECUTE    │  Rebalance portfolio                      │
│  └──────────────┘                                           │
│                                                              │
│  Days 2-30: HOLD                                            │
│                                                              │
│  Repeat next month...                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Filtering Criteria

```python
def filter_universe(df, date):
    """
    Filter stock universe for trading.

    Removes:
    - Bottom 15% by liquidity
    - Penny stocks (< 1,000 KRW)
    - Stocks with insufficient history
    """
    daily_df = df[df['date'] == date].copy()

    # Remove penny stocks
    daily_df = daily_df[daily_df['close'] >= 1000]

    # Remove illiquid stocks (bottom 15% by value)
    value_threshold = daily_df['traded_value'].quantile(0.15)
    daily_df = daily_df[daily_df['traded_value'] > value_threshold]

    # Remove stocks with NaN features (insufficient history)
    daily_df = daily_df.dropna()

    return daily_df
```

### Portfolio Construction

```python
def construct_portfolio(predictions_df, n_stocks=20, weighting='equal'):
    """
    Construct portfolio from model predictions.

    Args:
        predictions_df: DataFrame with 'stock_code', 'prediction', 'market_cap'
        n_stocks: Number of stocks to hold
        weighting: 'equal' or 'score_weighted' or 'cap_weighted'

    Returns:
        DataFrame with stock weights
    """
    # Sort by prediction score
    df = predictions_df.sort_values('prediction', ascending=False).head(n_stocks)

    if weighting == 'equal':
        df['weight'] = 1.0 / n_stocks

    elif weighting == 'score_weighted':
        # Weight by prediction score (normalized)
        df['weight'] = df['prediction'] / df['prediction'].sum()

    elif weighting == 'cap_weighted':
        # Weight by market cap (within selected stocks)
        df['weight'] = df['market_cap'] / df['market_cap'].sum()

    elif weighting == 'inverse_vol':
        # Risk parity - weight inversely by volatility
        df['weight'] = (1 / df['volatility_20d'])
        df['weight'] = df['weight'] / df['weight'].sum()

    return df[['stock_code', 'prediction', 'weight']]
```

---

## Phase 6: Analysis & Monitoring

### Feature Importance Analysis

```python
def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance.
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print("Top 10 Features by Importance:")
    print(importance.head(10).to_string(index=False))

    # Interpretation guide
    print("\n=== Interpretation Guide ===")
    top_feature = importance.iloc[0]['feature']

    if 'turnover' in top_feature.lower() or 'volume' in top_feature.lower():
        print("→ Liquidity-driven alpha (volume predicts moves)")
    elif 'momentum' in top_feature.lower() or 'mom' in top_feature.lower():
        print("→ Momentum-driven alpha (trend following)")
    elif 'market_cap' in top_feature.lower() or 'size' in top_feature.lower():
        print("→ Size-driven alpha (small-cap premium)")
    elif 'volatility' in top_feature.lower() or 'vol' in top_feature.lower():
        print("→ Low-volatility alpha (risk-adjusted)")

    return importance
```

### Performance Attribution

```python
def performance_attribution(backtest_results):
    """
    Attribute performance to different factors.
    """
    # Calculate factor exposures of winning vs losing months
    winning_months = backtest_results[backtest_results['return'] > 0]
    losing_months = backtest_results[backtest_results['return'] <= 0]

    print("=== Performance Attribution ===")
    print(f"Winning Months: {len(winning_months)}")
    print(f"Losing Months: {len(losing_months)}")
    print(f"Win Rate: {len(winning_months) / len(backtest_results):.1%}")

    # Analyze characteristics of winning portfolios
    print("\nWinning Portfolio Characteristics:")
    print(f"  Avg Market Cap Rank: {winning_months['avg_mcap_rank'].mean():.2f}")
    print(f"  Avg Momentum: {winning_months['avg_momentum'].mean():.2%}")
    print(f"  Avg Turnover: {winning_months['avg_turnover'].mean():.4f}")
```

### Risk Monitoring

```python
def calculate_risk_metrics(returns):
    """
    Calculate key risk metrics.
    """
    # Annualized metrics
    annual_return = returns.mean() * 12
    annual_vol = returns.std() * np.sqrt(12)
    sharpe_ratio = annual_return / annual_vol

    # Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown)

    print("=== Risk Metrics ===")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Annual Volatility: {annual_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")

    return {
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }
```

---

## Database Schema Requirements

### Current Schema (You Have)

```sql
-- stocks table
CREATE TABLE stocks (
    stock_code TEXT PRIMARY KEY,
    current_name TEXT,
    market_type TEXT
);

-- daily_prices table
CREATE TABLE daily_prices (
    stock_code TEXT,
    date TEXT,
    market_type TEXT,
    opening_price INTEGER,
    high_price INTEGER,
    low_price INTEGER,
    closing_price INTEGER,
    volume INTEGER,
    value INTEGER,          -- traded value
    market_cap INTEGER,
    PRIMARY KEY (stock_code, date)
);
```

### New Tables Needed

```sql
-- Pre-computed features (for speed)
CREATE TABLE ml_features (
    stock_code TEXT,
    date TEXT,
    -- Size
    log_market_cap REAL,
    market_cap_rank REAL,
    -- Liquidity
    turnover_ratio REAL,
    rvol REAL,
    amihud_20d REAL,
    -- Momentum
    mom_1m REAL,
    mom_3m REAL,
    mom_6m REAL,
    mom_12m REAL,
    mom_12m_skip_1m REAL,
    -- Volatility
    volatility_20d REAL,
    volatility_60d REAL,
    vol_ratio REAL,
    -- Price Location
    price_location REAL,
    dist_ma_20 REAL,
    dist_ma_60 REAL,
    -- Technical
    rsi_14 REAL,
    macd_histogram REAL,
    -- Target
    forward_return_21d REAL,
    target_rank_21d REAL,
    PRIMARY KEY (stock_code, date)
);

-- Model predictions
CREATE TABLE ml_predictions (
    date TEXT,
    stock_code TEXT,
    model_version TEXT,
    prediction_score REAL,
    prediction_rank INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, stock_code, model_version)
);

-- Backtest results
CREATE TABLE ml_backtest_results (
    backtest_id TEXT,
    period_start TEXT,
    period_end TEXT,
    n_stocks INTEGER,
    portfolio_return REAL,
    benchmark_return REAL,
    alpha REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (backtest_id, period_start)
);
```

---

## Code Implementation

### Directory Structure

```
algostock/
├── ml/
│   ├── __init__.py
│   ├── features.py          # Feature engineering
│   ├── labeling.py          # Target creation
│   ├── model.py             # LightGBM training
│   ├── validation.py        # Walk-forward validation
│   ├── portfolio.py         # Portfolio construction
│   └── backtest.py          # ML-based backtesting
├── algostock_cli.py         # Add ML commands
└── docs/
    └── ML_FACTOR_RANKING_MODEL.md
```

### Implementation Checklist

- [ ] **Phase 1: Feature Engineering**
  - [ ] Create `ml/features.py`
  - [ ] Implement all feature calculations
  - [ ] Add feature storage to database
  - [ ] CLI command: `python3 algostock_cli.py ml compute-features`

- [ ] **Phase 2: Target Labeling**
  - [ ] Create `ml/labeling.py`
  - [ ] Implement cross-sectional ranking
  - [ ] Support multiple horizons (21d, 63d, 126d)
  - [ ] CLI command: `python3 algostock_cli.py ml compute-targets`

- [ ] **Phase 3: Model Training**
  - [ ] Create `ml/model.py`
  - [ ] Implement LightGBM ranker
  - [ ] Support hyperparameter tuning
  - [ ] CLI command: `python3 algostock_cli.py ml train`

- [ ] **Phase 4: Validation**
  - [ ] Create `ml/validation.py`
  - [ ] Implement walk-forward validation
  - [ ] Generate validation reports
  - [ ] CLI command: `python3 algostock_cli.py ml validate`

- [ ] **Phase 5: Prediction & Portfolio**
  - [ ] Create `ml/portfolio.py`
  - [ ] Implement monthly stock selection
  - [ ] Support different weighting schemes
  - [ ] CLI command: `python3 algostock_cli.py ml predict`

- [ ] **Phase 6: Full Backtest**
  - [ ] Create `ml/backtest.py`
  - [ ] End-to-end ML backtest
  - [ ] Generate performance reports
  - [ ] CLI command: `python3 algostock_cli.py ml backtest`

### CLI Commands (Planned)

```bash
# Compute features for all historical data
python3 algostock_cli.py ml compute-features -s 20110101 -e 20251231

# Compute targets (forward returns and ranks)
python3 algostock_cli.py ml compute-targets --horizon 21

# Train model with walk-forward validation
python3 algostock_cli.py ml train --train-years 5 --test-years 1

# Generate predictions for current date
python3 algostock_cli.py ml predict --top 20

# Run full ML backtest
python3 algostock_cli.py ml backtest -s 20160101 -e 20251231 \
    --n-stocks 20 \
    --rebalance monthly \
    --weighting equal
```

---

## Expected Results

Based on academic research and industry practice:

| Metric | Naive Momentum | ML Factor Model |
|--------|----------------|-----------------|
| Annual Return | 8-12% | 15-25% |
| Sharpe Ratio | 0.3-0.5 | 0.7-1.2 |
| Max Drawdown | -40% | -25% |
| Win Rate | 52% | 58-62% |

**Important Caveats**:
- Past performance does not guarantee future results
- Transaction costs and slippage reduce real returns
- Model performance degrades as more people use similar strategies
- Requires regular retraining as market conditions change

---

## References

1. Korean Market Anomalies Research
2. LightGBM Documentation
3. Advances in Financial Machine Learning (Marcos López de Prado)
4. Machine Learning for Asset Managers (Marcos López de Prado)

---

## Next Steps

1. **Start Simple**: Implement basic momentum + size factors first
2. **Validate**: Run walk-forward validation before adding complexity
3. **Iterate**: Add features one by one, measure improvement
4. **Monitor**: Track feature importance and model drift over time

Ready to implement? Start with Phase 1 (Feature Engineering).
