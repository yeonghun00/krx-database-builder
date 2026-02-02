# ML Factor Ranking Model - Problem Analysis Report

**Date**: 2026-01-30
**Author**: Analysis Team
**Status**: Critical Issues Found - Model NOT Tradeable

---

## Executive Summary

Our ML Factor Ranking Model shows **unrealistically high returns** (60-85% annually) and **suspiciously high Sharpe ratios** (2.7-3.6). After thorough analysis and stress testing, we identified the root cause:

**CRITICAL FINDING**: The model's alpha comes **entirely from small-cap stocks**. When restricted to large-cap stocks only, the strategy produces **ZERO alpha** (0.9% annual return, Sharpe 0.03, p-value 0.72).

This means the model is exploiting the small-cap anomaly, not generating genuine stock selection alpha. Small-cap Korean stocks are difficult to trade at scale due to low liquidity.

---

## Table of Contents

1. [Observed Results](#observed-results)
2. [Issues Identified](#issues-identified)
3. [Detailed Analysis](#detailed-analysis)
4. [Root Cause Analysis](#root-cause-analysis)
5. [Recommendations](#recommendations)

---

## Observed Results

### Backtest Performance (Claimed)

| Metric | 10 Stocks | 20 Stocks |
|--------|-----------|-----------|
| Annual Return (net) | 84.7% | 60.9% |
| Sharpe Ratio | 3.61 | 2.78 |
| Max Drawdown | -12.9% | -13.4% |
| Win Rate | 78.0% | 78.0% |

### Why These Results Are Suspicious

| Benchmark | Value | Our Model |
|-----------|-------|-----------|
| S&P 500 historical avg | 10% | 60-85% |
| Top hedge fund (Renaissance) | ~30-40% | 60-85% |
| Sharpe > 2 | Extremely rare | 2.78-3.61 |
| Warren Buffett lifetime | ~20% | 60-85% |

**Conclusion**: Results are 3-4x better than the best investors in history. This is statistically implausible.

---

## Issues Identified

### Issue #1: High Prediction-Return Correlation

```
ML Score vs Forward Return Correlation: 0.164 (full universe)
ML Score vs Forward Return Correlation: 0.068 (large-cap only)
Expected for legitimate model: 0.05-0.15
```

**Update (2026-01-30)**: After stress testing, the correlation is 0.164 for the full universe (high but not extreme). However, for large-cap stocks only, the correlation drops to 0.068 (normal range).

This suggests:
- The model's predictive power is concentrated in small-cap stocks
- Small-caps have more extreme and predictable patterns
- The correlation is NOT due to data leakage

### Issue #2: Extreme Return Stocks Dominating Results

From our period analysis:

```
Period       Avg Return   Max Return   >30% Stocks
20210104     8.16%        31.40%       1/10
20210308     15.34%       43.36%       2/10
20210406     7.69%        64.98%       1/10
20211008     16.09%       93.56%       1/10
```

**9% of selected stocks hit the +30% cap**, meaning the model consistently picks extreme winners. This is either:
- Legitimate alpha (unlikely at this magnitude)
- Survivorship bias
- Look-ahead bias

### Issue #3: Feature Importance Reveals Selection Bias

Top features by importance:
```
1. amihud_20d        (illiquidity measure)
2. log_market_cap    (company size)
3. volatility_60d    (price volatility)
4. market_cap_rank   (size ranking)
```

The model is essentially selecting:
- **Small-cap stocks** (higher volatility, more extreme moves)
- **Illiquid stocks** (harder to trade in practice)
- **Volatile stocks** (prone to extreme returns)

This creates a systematic bias toward stocks that:
- Have extreme historical returns (survivorship bias)
- Are difficult to trade at backtest prices (execution gap)
- May no longer exist (delisted companies excluded)

### Issue #4: Survivorship Bias

Our dataset only contains stocks that:
- Currently exist in the database
- Have continuous price history
- Meet minimum trading requirements

Stocks that went bankrupt, were delisted, or had trading halts are **excluded**, artificially inflating returns.

### Issue #5: Execution Assumptions

The backtest assumes:
- We can buy at the exact closing price (unrealistic)
- No market impact from our trades
- Unlimited liquidity
- No slippage beyond our 0.5% estimate

For small-cap Korean stocks, actual execution costs can be 2-5%.

### Issue #6: Forward Return Calculation Verified But...

We verified the forward return calculation is mathematically correct:

```python
# Method 1 and Method 2 produce identical results
Forward return = (Price_T+21 - Price_T) / Price_T
```

However, the issue is not in the calculation but in **what stocks are available** to calculate returns for.

---

## Detailed Analysis

### Period-by-Period Breakdown (First 10 Periods of 2021)

| Period | Avg Return | Capped Return | Extreme+ | Extreme- |
|--------|------------|---------------|----------|----------|
| 20210104 | 8.16% | 8.03% | 1 | 0 |
| 20210202 | 0.74% | 0.07% | 1 | 0 |
| 20210308 | 15.34% | 13.47% | 2 | 0 |
| 20210406 | 7.69% | 4.91% | 1 | 1 |
| 20210506 | -0.83% | -0.83% | 0 | 0 |
| 20210607 | 11.36% | 10.85% | 1 | 0 |
| 20210706 | 0.73% | 0.73% | 0 | 0 |
| 20210804 | 2.92% | 1.32% | 1 | 0 |
| 20210903 | -5.01% | -5.88% | 1 | 1 |
| 20211008 | 16.09% | 9.73% | 1 | 0 |
| **Average** | **5.72%** | **4.24%** | 0.9 | 0.2 |

**Observation**: Even with ±30% capping, average period return is 4.24%, which compounds to:
- 10 periods: +49.1%
- 59 periods (5 years): **~1,000%+**

This is still unrealistically high.

### Stock Selection Analysis

Most frequently selected stocks:
```
Stock          Times Selected   Avg Return
신풍제약         5               +15.2%
씨젠            4               +8.3%
셀리버리         4               -5.1%
LG전자          3               +10.5%
```

Many of these are:
- COVID-19 beneficiary stocks (씨젠, 셀리버리)
- High-volatility theme stocks (신풍제약)
- Stocks with extreme moves during the test period

### The 2021-2025 Period Was Exceptional

The test period (2021-2025) included:
- COVID-19 recovery rally (2021)
- Battery/EV stock boom (2021-2022)
- AI stock surge (2023-2024)

These extreme market conditions may not repeat, making backtest results unreliable for future performance.

---

## Root Cause Analysis

### Primary Cause: Selection Bias Toward Extreme Movers

The model learns that certain characteristics (small cap, high volatility, low liquidity) are associated with extreme returns. It then selects stocks with these characteristics.

```
Training Target = rank(forward_return)
Model learns: small + volatile + illiquid → high rank
At test time: selects small + volatile + illiquid stocks
Result: captures extreme moves (but also extreme losses in practice)
```

### Secondary Cause: Survivorship Bias

The database excludes:
- Delisted companies
- Bankrupt companies
- Stocks with trading halts

This creates an upward bias because we only see stocks that "survived."

### Tertiary Cause: Unrealistic Execution Assumptions

Small-cap Korean stocks have:
- Wide bid-ask spreads (1-5%)
- Low daily volume
- Price impact from large orders

Our 0.7% cost assumption is too optimistic.

---

## Recommendations

### Short-Term Fixes

1. **Stricter Liquidity Filter**
   ```python
   # Current: market_cap >= 100B, value >= 5B
   # Recommended: market_cap >= 500B, value >= 20B
   ```

2. **Stricter Return Cap**
   ```python
   # Current: ±30%
   # Recommended: ±15% (more realistic for tradeable stocks)
   ```

3. **Higher Transaction Costs**
   ```python
   # Current: 0.7%
   # Recommended: 2.0% (including market impact)
   ```

### Medium-Term Improvements

1. **Out-of-Sample Testing**
   - Train on 2011-2018
   - Validate on 2019-2020
   - Test on 2021-2025
   - Never touch test data during development

2. **Walk-Forward with Gap**
   - 6-month gap between training and testing
   - Prevents pattern overfitting

3. **Include Delisted Stocks**
   - Add historical data for stocks that were delisted
   - Reduces survivorship bias

### Long-Term Solutions

1. **Live Paper Trading**
   - Run strategy with simulated money for 6+ months
   - Compare to backtest predictions

2. **Realistic Execution Simulation**
   - Use order book data if available
   - Simulate market impact
   - Include failed trades (no liquidity)

3. **Multiple Market Regime Testing**
   - Test on bull markets, bear markets, sideways markets
   - Ensure strategy works in different conditions

---

## Stress Test Results (2026-01-30)

We ran four critical stress tests to identify the root cause of unrealistic returns.

### Test 1: Slippage Floor (High Transaction Costs)

| Cost | Annual Return | Sharpe | Significant? |
|------|---------------|--------|--------------|
| 0.7% | 84.7% | 3.61 | Yes |
| 1.5% | 68.5% | 2.92 | Yes |
| 2.0% | 59.1% | 2.52 | Yes |
| 3.0% | 41.6% | 1.77 | Yes |

**Result**: Strategy survives high costs. This is NOT the problem.

### Test 2: Exclude Micro-Caps (Large-Cap Only)

| Universe | Annual Return | Sharpe | p-value | Significant? |
|----------|---------------|--------|---------|--------------|
| All stocks (baseline) | 84.7% | 3.61 | 0.0000 | Yes |
| Market cap >= 1T KRW | **0.9%** | **0.03** | **0.7238** | **NO** |
| Top 100 by market cap | **0.2%** | **0.01** | **0.7818** | **NO** |

**Result**: **SMOKING GUN FOUND**. The strategy produces ZERO alpha with large-caps. All returns come from small-cap stocks that are difficult to trade.

### Test 3: Purged Cross-Validation (Gap Between Train/Test)

| Gap | Annual Return | Sharpe | ML-Return Correlation |
|-----|---------------|--------|----------------------|
| 0 days | 84.7% | 3.61 | 0.164 |
| 21 days | 89.5% | 3.79 | 0.158 |
| 42 days | 87.8% | 3.64 | 0.159 |
| 63 days | 95.1% | 3.99 | 0.152 |

**Result**: Adding gaps doesn't reduce returns. Data leakage between train/test is NOT the issue.

### Test 4: Random Shuffle Test

| Condition | Annual Return | Sharpe | ML-Return Correlation |
|-----------|---------------|--------|----------------------|
| Original | 84.7% | 3.61 | 0.164 |
| Shuffled (avg of 5 trials) | 9.1% | 0.41 | -0.008 |

**Result**: Shuffling destroys performance. The features DO have predictive power, but only for small-caps.

### Stress Test Conclusion

| Test | Result | Implication |
|------|--------|-------------|
| High Costs | PASSED | Not a cost issue |
| Large-Cap Only | **FAILED** | Alpha is small-cap only |
| Purged CV | PASSED | Not data leakage |
| Random Shuffle | PASSED | Features have signal |

**The model has genuine predictive power, but ONLY for small-cap stocks that are NOT tradeable at scale.**

---

## Conclusion

The ML Factor Ranking Model produces results that are **too good to be true**. The primary issues are:

1. ~~High prediction-return correlation (0.391) suggests data leakage~~ **PARTIALLY CLARIFIED**: Correlation is 0.164 (not 0.391) - still high but not extreme
2. **Selection bias** toward small-cap stocks - **CONFIRMED as root cause**
3. **Survivorship bias** from excluding failed companies
4. **Small-cap premium is not tradeable** - this is the key finding

**Verdict**: The model works, but only on stocks you cannot trade. This is a common trap in quantitative finance - strategies that look great in backtests but fail in practice due to liquidity constraints.

**Recommended Action**: Do not deploy this model for real trading. If continuing development:
- Restrict universe to large-cap stocks only (KOSPI 200)
- Accept that realistic returns will be much lower (10-15% if lucky)
- Consider the model as a "research tool" rather than a trading strategy

---

## Appendix: Realistic Expectations

Based on academic research and industry practice:

| Strategy Type | Expected Annual Return | Expected Sharpe |
|---------------|------------------------|-----------------|
| Passive Index | 8-10% | 0.3-0.5 |
| Factor Investing | 10-15% | 0.5-0.8 |
| Quantitative Hedge Fund | 15-25% | 0.8-1.5 |
| Top Quant Fund (rare) | 25-40% | 1.5-2.0 |
| Our Model (claimed) | 60-85% | 2.7-3.6 |

Our model's claimed performance is 2-3x better than the best quant funds, which is statistically implausible.

---

*End of Report*
