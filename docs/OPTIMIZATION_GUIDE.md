# Performance Optimization Guide

Practical optimizations for the existing pipeline. Each item targets a specific
bottleneck in the current code with a concrete fix.

Baseline (from timing instrumentation already in `ml/features.py`):

| Stage | File | Estimated Time |
|-------|------|---------------|
| Raw SQL load | `load_raw_data` | ~30s |
| Technical features | `compute_features` | ~60s |
| Financial features | `_load_financial_features_fast` | ~15s |
| Financial merge | `merge_financial_features` | ~30s |
| **Rolling beta (252d + 60d)** | `add_forward_returns` | **~250s** |
| Forward returns + ranks | `add_forward_returns` | ~10s |
| Filter universe | `filter_universe` | ~2s |
| Macro features | `_add_macro_features` | ~5s |
| Walk-forward loop (per year) | `run_backtest.py` | ~3s x N |

---

## Priority 1: Rolling Beta (biggest bottleneck)

**File:** `ml/features.py`, `add_forward_returns()` lines 806-837

**Problem:** Two `groupby('stock_code').apply()` calls — one for 252d beta,
one for 60d beta. Each invokes a Python function per stock (~1000+ stocks),
each doing its own `rolling().cov()`. The per-group overhead dominates.

**Fix: Pre-align market returns, then use a single vectorized rolling cov/var**

```python
# BEFORE (slow — Python function per stock)
df['rolling_beta'] = grouped.apply(
    lambda g: _calc_stock_beta(g)
).reset_index(level=0, drop=True)

# AFTER (fast — one vectorized operation on the whole DataFrame)
market_ret = df.groupby('date')['return'].transform('median')
rolling_cov = (
    grouped['return']
    .rolling(252, min_periods=60)
    .cov(market_ret.groupby(df['stock_code']))
    .reset_index(level=0, drop=True)
)
rolling_var = (
    market_ret
    .groupby(df['stock_code'])
    .rolling(252, min_periods=60)
    .var()
    .reset_index(level=0, drop=True)
)
df['rolling_beta'] = (rolling_cov / rolling_var.clip(lower=1e-8)).clip(-3, 3)
df['rolling_beta'] = df['rolling_beta'].fillna(1.0)
```

If the above still hits pandas overhead, the nuclear option is **numba**:

```python
from numba import njit

@njit
def rolling_beta_numba(stock_ret, market_ret, window, min_periods):
    n = len(stock_ret)
    out = np.empty(n)
    out[:] = np.nan
    for i in range(window - 1, n):
        s = stock_ret[i - window + 1 : i + 1]
        m = market_ret[i - window + 1 : i + 1]
        valid = ~(np.isnan(s) | np.isnan(m))
        if valid.sum() >= min_periods:
            sv = s[valid]
            mv = m[valid]
            cov = np.mean(sv * mv) - np.mean(sv) * np.mean(mv)
            var = np.mean(mv * mv) - np.mean(mv) ** 2
            if var > 1e-8:
                out[i] = max(-3, min(3, cov / var))
    return out
```

Then apply per-stock using `groupby().transform()` with the numba function on
raw numpy arrays. This turns Python overhead into machine-code speed.

**Expected improvement:** 250s → 20-40s

---

## Priority 2: Redundant 52-week High/Low Computation

**File:** `ml/features.py`

**Problem:** `high_52w` and `low_52w` are computed in `_compute_volatility_features()`
(line 350-351), then dropped in `_cleanup_intermediate_cols()` (line 505),
then **recomputed** in `_compute_intuition_features()` (lines 319-324 of optimized /
implicit via `df['high_52w']`). Rolling max/min over 252 days on 8M rows is expensive.

**Fix:** Compute once, keep until after intuition features, then drop.
Remove `high_52w` and `low_52w` from `_cleanup_intermediate_cols()`. Instead,
drop them at the end of `compute_features()` after all groups have used them.

**Expected improvement:** ~10-15s saved

---

## Priority 3: Grouped Rolling Operations in Feature Methods

**File:** `ml/features.py`, all `_compute_*` methods

**Problem:** Each method calls `grouped = df.groupby('stock_code')` or uses the
passed `grouped` object repeatedly. Each `.rolling().mean().reset_index(level=0, drop=True)`
creates intermediate Series objects.

**Fix: Batch rolling windows of the same column**

```python
# BEFORE: separate operations
df['vol_5d'] = grouped['volume'].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
df['vol_20d'] = grouped['volume'].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)

# AFTER: use .agg or compute on sorted contiguous arrays
# The key insight: if df is already sorted by (stock_code, date),
# you can use pandas EWM/rolling on the whole column and just mask
# the stock boundaries. But the simplest real win is reducing
# the number of groupby calls:

vol_group = grouped['volume']
df['vol_5d'] = vol_group.rolling(5, min_periods=3).mean().droplevel(0)
df['vol_20d'] = vol_group.rolling(20, min_periods=10).mean().droplevel(0)
df['vol_60d_max'] = vol_group.rolling(60, min_periods=30).max().droplevel(0)
```

Using `.droplevel(0)` instead of `.reset_index(level=0, drop=True)` is
slightly faster (avoids full index reconstruction).

**Expected improvement:** ~10-15s across all feature methods

---

## Priority 4: SQL Index Optimization

**File:** Database `krx_stock_data.db`

**Problem:** Queries in `load_raw_data()` and `_load_financial_features_fast()` filter on
`date`, `market_type`, `stock_code`, and join on `stock_code`. Without proper indexes
the DB does full table scans.

**Fix: Run once to create covering indexes**

```sql
-- Main query index (covers the WHERE + JOIN + ORDER BY)
CREATE INDEX IF NOT EXISTS idx_dp_market_date_stock
ON daily_prices(market_type, date, stock_code);

-- Financial period lookup
CREATE INDEX IF NOT EXISTS idx_fp_consol_avail
ON financial_periods(consolidation_type, available_date);

-- Financial items lookup by period_id
CREATE INDEX IF NOT EXISTS idx_bs_period_item
ON financial_items_bs_cf(period_id, item_code_normalized);

CREATE INDEX IF NOT EXISTS idx_pl_period_item
ON financial_items_pl(period_id, item_code_normalized);

ANALYZE;
```

Put this in a one-time script or add to the ETL pipeline.

**Expected improvement:** SQL load 30s → 10-15s

---

## Priority 5: dtype Optimization for Memory + Speed

**File:** `ml/features.py`, after `load_raw_data()`

**Problem:** Pandas defaults to int64/float64. The DataFrame has 8M+ rows with
columns like `volume`, `market_cap` that could be float32 with no loss of
meaningful precision for ML features. Halving memory means faster cache
utilization and faster rolling operations.

**Fix:**

```python
def _optimize_dtypes(self, df):
    float_cols = df.select_dtypes('float64').columns
    df[float_cols] = df[float_cols].astype('float32')

    # stock_code and sector as categorical
    for col in ['stock_code', 'sector', 'market_type']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df
```

Call after `load_raw_data()` and after each merge that introduces new float columns.

**Expected improvement:** ~20-30% memory reduction, ~10% speed gain on rolling ops

---

## Priority 6: Cache Granularity

**File:** `ml/features.py`, `prepare_ml_data()`

**Problem:** The current cache is all-or-nothing — one parquet file for the
entire pipeline output. If the DB hasn't changed, it loads from cache. But if
anything changes, the entire pipeline re-runs.

**Fix: Stage-level caching**

```python
# After raw load + tech features:
tech_cache = f".cache/tech_{cache_hash}.parquet"
if os.path.exists(tech_cache) and os.path.getmtime(tech_cache) > db_mtime:
    df = pd.read_parquet(tech_cache)
else:
    df = self.load_raw_data(...)
    df = self.compute_features(df)
    df.to_parquet(tech_cache)

# After financial merge:
fin_cache = f".cache/fin_{cache_hash}.parquet"
# ... same pattern
```

This way, if you're iterating on beta calculation or forward returns, you skip
the 90s of raw load + feature computation every time.

**Expected improvement:** Development iteration time from ~700s to ~30s for beta-only changes

---

## Priority 7: Walk-Forward Training Parallelization

**File:** `run_backtest.py`, the `for test_year in years:` loop

**Problem:** Each year's train/predict is independent but runs sequentially.
With ~12 years of walk-forward, this is ~36s of serial LightGBM training.

**Fix:** Use `joblib` or `concurrent.futures`:

```python
from joblib import Parallel, delayed

def train_and_predict_year(test_year, df, all_features, ...):
    # ... existing per-year logic ...
    return result_dict

all_results = Parallel(n_jobs=-1)(
    delayed(train_and_predict_year)(year, df, all_features, ...)
    for year in years
)
```

LightGBM already uses `n_jobs=-1` internally, so set LightGBM to `n_jobs=1`
when parallelizing at the year level to avoid thread contention.

**Expected improvement:** Walk-forward loop 36s → 10-15s

---

## Priority 8: Sector Z-Score in Financial Features

**File:** `ml/features.py`, `_load_financial_features_fast()` line 640-651

**Problem:** `groupby(['date', sector_col])['pb_ratio'].transform(_sector_zscore)`
uses a Python function callback per group. With ~3000 dates x ~30 sectors =
~90,000 group calls.

**Fix:** Use vectorized zscore:

```python
group = merged.groupby(['date', sector_col])['pb_ratio']
mean = group.transform('mean')
std = group.transform('std').clip(lower=0.01)
merged['pb_sector_zscore'] = ((merged['pb_ratio'] - mean) / std).clip(-3, 3)
```

Apply the same pattern to `roe_sector_zscore` in `load_financial_features()`.

**Expected improvement:** ~5s saved

---

## Summary: Expected Total Impact

| Optimization | Time Saved | Effort |
|-------------|-----------|--------|
| Vectorized beta (P1) | ~200s | Medium |
| Remove duplicate 52w (P2) | ~12s | Easy |
| Batch rolling ops (P3) | ~12s | Easy |
| SQL indexes (P4) | ~15s | Easy (one-time) |
| dtype optimization (P5) | ~30s | Easy |
| Stage caching (P6) | dev-time only | Easy |
| Parallel walk-forward (P7) | ~20s | Medium |
| Vectorized zscore (P8) | ~5s | Easy |
| **Total** | **~300s** | |

Realistic target: **700s → 350-400s** with P1-P5 (the easy wins).
With numba for beta (P1 nuclear option): **700s → 200-250s**.
