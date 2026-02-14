# Model & Backtest

## Architecture

```
  DB (krx_stock_data.db)
      │
      ▼
  ml/features/_pipeline.py     ← data loading, merging, orchestration
      │
      ▼
  ml/features/registry.py      ← FeatureGroup base + @register + topological sort
      │
      ├── momentum.py           (4 features)
      ├── volume.py             (3 features)
      ├── volatility.py         (4 features)
      ├── fundamental.py        (4 features)
      ├── market.py             (2 features)
      ├── sector.py             (7 features)
      ├── sector_neutral.py     (4 features)
      ├── distress.py           (5 features)
      └── sector_rotation.py    (3 features)
      │
      ▼
  ml/models/                    ← multi-model support
      ├── lgbm.py               LGBMRanker (default)
      ├── xgboost.py            XGBRanker
      └── catboost.py           CatBoostRanker
      │
      ▼
  scripts/run_backtest.py       ← walk-forward backtest
  scripts/get_picks.py          ← live stock picks
```

---

## How It Works

1. `FeatureEngineer` builds a panel of ~36 features per stock per trading day from the DB
2. Forward returns are computed as targets (default: 21 trading days ahead)
3. Data is split into yearly walk-forward folds (train on N years, test on the next year)
4. A model (LightGBM/XGBoost/CatBoost) is trained per fold to predict outperformance
5. On each rebalance date, the model scores all eligible stocks, picks the top N, and simulates a portfolio
6. Transaction costs, turnover, and Spearman IC are tracked at every rebalance

---

## How to Add a New Feature

1. Create a new file in `ml/features/`, e.g. `ml/features/my_feature.py`
2. Copy this template:

```python
"""My custom features."""
from __future__ import annotations
import pandas as pd
from .registry import FeatureGroup, register

@register
class MyFeatures(FeatureGroup):
    name = "my_feature"
    columns = ["my_col_1", "my_col_2"]       # Feature names this group produces
    dependencies = ["closing_price", "ret_1d"] # Columns that must exist first

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("stock_code")
        df["my_col_1"] = g["closing_price"].pct_change(10)
        df["my_col_2"] = g["ret_1d"].rolling(10).std().droplevel(0)
        return df
```

3. Import it in `ml/features/__init__.py`:

```python
from ml.features import my_feature  # noqa: F401
```

4. Done. The `@register` decorator adds it to the registry automatically. `FEATURE_COLUMNS` updates, and the model will include it on the next run.

**Key rules:**
- `columns` must list every column your `compute()` adds
- `dependencies` must list columns produced by other groups (the registry topologically sorts groups so yours runs after its dependencies)
- `compute()` receives the full DataFrame and should return it with new columns added

---

## How to Add a New Model

1. Create `ml/models/my_model.py`:

```python
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
from .base import BaseRanker

class MyRanker(BaseRanker):
    BEST_PARAMS = {"learning_rate": 0.03, "n_estimators": 800}

    def train(self, train_df, val_df=None, params=None, sample_weight=None):
        params = params or self.BEST_PARAMS.copy()
        X = train_df[self.feature_cols].to_numpy()
        y = train_df[self.target_col].to_numpy()
        weight = self._calculate_time_weights(train_df)
        # ... train your model here ...
        self.model = trained_model
        return self

    def predict(self, df):
        return self.model.predict(df[self.feature_cols].to_numpy())
```

2. Register it in `ml/models/__init__.py`:

```python
from ml.models.my_model import MyRanker

def get_model_class(name):
    models = {
        "lgbm": LGBMRanker,
        "xgboost": XGBRanker,
        "catboost": CatBoostRanker,
        "my_model": MyRanker,     # ← add here
    }
    ...
```

3. Use it: `python scripts/run_backtest.py --model my_model`

---

## Feature Group Reference (36 features)

### Momentum (4)

| Feature | What it measures |
|---------|-----------------|
| `mom_5d` | 5-day price return |
| `mom_21d` | 1-month price return |
| `mom_63d` | 3-month price return |
| `mom_126d` | 6-month price return |

### Volume & Liquidity (3)

| Feature | What it measures |
|---------|-----------------|
| `volume_ratio_21d` | Today's volume / 21-day average volume |
| `turnover_21d` | 21-day average traded value / market cap |
| `amihud_21d` | 21-day average of |return| / value (price impact) |

### Volatility & Risk (4)

| Feature | What it measures |
|---------|-----------------|
| `volatility_21d` | Std dev of daily returns over 21 days |
| `volatility_63d` | Std dev of daily returns over 63 days |
| `drawdown_252d` | Current price / 252-day rolling high - 1 |
| `rolling_beta_60d` | 60-day rolling beta vs KOSPI 200 |

### Fundamental (4, PIT-safe)

| Feature | What it measures |
|---------|-----------------|
| `roe` | Net income / equity |
| `gpa` | Gross profit / assets (Novy-Marx factor) |
| `sector_zscore_roe` | ROE z-scored within sector |
| `sector_zscore_gpa` | GPA z-scored within sector |

### Market Context (2)

| Feature | What it measures |
|---------|-----------------|
| `market_regime_120d` | KOSPI 200 current / 120-day MA - 1 |
| `constituent_index_count` | Number of KRX indices the stock belongs to |

### Sector (7)

| Feature | What it measures |
|---------|-----------------|
| `sector_momentum_21d` | 21-day return of the stock's sector index |
| `sector_momentum_63d` | 63-day return of the stock's sector index |
| `sector_relative_momentum_20d` | Sector 20d return minus KOSPI 20d return |
| `sector_relative_momentum_21d` | Sector 21d return minus KOSPI 21d return |
| `sector_relative_momentum_63d` | Sector 63d return minus KOSPI 63d return |
| `sector_breadth_21d` | % of sector constituents with positive 21d momentum |
| `sector_constituent_share` | Relative size of sector |

### Sector-Neutralized (4)

| Feature | Underlying raw feature |
|---------|----------------------|
| `sector_zscore_mom_21d` | `mom_21d` |
| `sector_zscore_turnover_21d` | `turnover_21d` |
| `sector_zscore_volatility_21d` | `volatility_21d` |
| `sector_zscore_drawdown_252d` | `drawdown_252d` |

### Distress Detection (5)

| Feature | What it measures |
|---------|-----------------|
| `liquidity_decay_score` | 20-day avg value / 252-day avg value |
| `low_price_trap` | log(price / sector avg price) |
| `is_liquidity_distressed` | Binary: liquidity_decay_score <= 0.2 |
| `is_low_price_trap` | Binary: price < 1000 or low_price_trap < -1.0 |
| `distress_composite_score` | Weighted combo (0-1 scale) |

### Sector Rotation (3)

| Feature | What it measures |
|---------|-----------------|
| `sector_dispersion` | Cross-sectional std dev within sector |
| `sector_dispersion_21d` | 21-day smoothed sector dispersion |
| `sector_rotation_signal` | Positive when sector has good momentum AND low dispersion |

---

## Model Reference

### LightGBM (default, `--model lgbm`)

```python
{
    "objective": "huber", "huber_delta": 1.0,
    "num_leaves": 63, "learning_rate": 0.03,
    "feature_fraction": 0.75, "bagging_fraction": 0.8,
    "min_data_in_leaf": 80, "n_estimators": 800,
}
```

Robust to outliers via Huber loss. Best general-purpose choice.

### XGBoost (`--model xgboost`)

```python
{
    "objective": "reg:pseudohubererror",
    "max_depth": 6, "learning_rate": 0.03,
    "subsample": 0.8, "colsample_bytree": 0.75,
    "min_child_weight": 80, "n_estimators": 800,
}
```

Tree-based alternative. Requires `xgboost` package.

### CatBoost (`--model catboost`)

```python
{
    "loss_function": "Huber:delta=1.0",
    "depth": 6, "learning_rate": 0.03,
    "subsample": 0.8, "colsample_bylevel": 0.75,
    "min_data_in_leaf": 80, "iterations": 800,
}
```

Good for ordered features. Requires `catboost` package.

---

## Universe Filters

1. **Penny stock exclusion**: `closing_price >= 2000` KRW
2. **Low liquidity exclusion**: Drop bottom 20% by 20-day average traded value
3. **Accrual quality filter**: Exclude positive net income + negative operating CF
4. **Market cap floor**: Default `500B KRW`
5. **Delisted stock exclusion**: Remove after delisting date

---

## Target Variable

| Priority | Target column | What it is |
|----------|--------------|------------|
| 1st | `target_riskadj_rank_{H}d` | Rank of (forward return / volatility_21d) |
| 2nd | `target_residual_rank_{H}d` | Rank of (forward return - beta * market return) |
| 3rd | `target_rank_{H}d` | Rank of raw forward return |

---

## Walk-Forward Validation

```
Fold 1:  Train [2010-2012]  Test [2013]
Fold 2:  Train [2011-2013]  Test [2014]
...
```

- Last training year held out for early stopping validation
- Purged embargo (`--embargo-days 21`) prevents leakage

---

## CLI Reference

### `run_backtest.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `lgbm` | Model type: lgbm, xgboost, catboost |
| `--start` | `20120101` | Backtest start date |
| `--end` | `20260213` | Backtest end date |
| `--horizon` | `21` | Forward return horizon (trading days) |
| `--top-n` | `30` | Portfolio size |
| `--rebalance-days` | `63` | Days between rebalances |
| `--train-years` | `3` | Rolling training window size |
| `--min-market-cap` | `500000000000` | Min market cap (500B KRW) |
| `--time-decay` | `0.4` | Training sample recency weighting |
| `--learning-rate` | `0.01` | Learning rate |
| `--n-estimators` | `2000` | Max boosting rounds |
| `--patience` | `200` | Early stopping patience |
| `--buy-fee` | `0.5` | Buy fee % |
| `--sell-fee` | `0.5` | Sell fee % |
| `--buy-rank` | `5` | Max rank to buy new stocks |
| `--hold-rank` | `50` | Max rank to keep existing stocks |
| `--embargo-days` | `21` | Purged embargo gap |
| `--workers` | `4` | Parallel fold workers |
| `--stress-mode` | off | Enable stress testing |
| `--no-cache` | off | Skip feature cache |

### `get_picks.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `lgbm` | Model type: lgbm, xgboost, catboost |
| `--end` | today | Feature data end |
| `--top` | `20` | Number of buy picks |
| `--bottom` | `10` | Number of avoid picks |
| `--horizon` | `21` | Forward return horizon |
| `--min-market-cap` | `500000000000` | Min market cap |
| `--model-path` | `models/lgbm_unified.pkl` | Pre-trained model path |
| `--retrain` | off | Retrain instead of loading |
| `--no-cache` | off | Skip feature cache |
