# Model Versions Guide (V3 ~ V11)

A plain-language walkthrough of every model version in AlgoStock, what data each one looks at, how it decides which stocks to buy, and why each piece of data matters.

> **Reading tip** -- Each version inherits everything from the version before it unless stated otherwise. Think of them as layers on a cake: V4 = V3 + macro data, V5 = V4 + smarter feature selection, and so on.

---

## How the System Works (60-Second Overview)

1. **Collect data** -- price history, trading volume, financial statements, market-wide indicators.
2. **Compute features** -- turn raw data into numbers that describe "what is happening with this stock right now" (e.g., "price went up 12% in the last 3 months").
3. **Train a model** -- feed historical features + the actual future returns to a machine-learning model (LightGBM) so it learns "when feature X looked like this in the past, the stock usually went up."
4. **Score & rank** -- for today's stocks, compute the same features, let the model score them, pick the top-ranked ones.
5. **Manage the portfolio** -- decide how many stocks to hold, how much money to put in each, and when to step aside into cash.

Every version below changes *something* in steps 2-5. The table shows what changed.

---

## Quick Comparison

| Version | Features | What It Predicts | Portfolio | Key Idea |
|---------|----------|-----------------|-----------|----------|
| V3 | 28 (price + volume + financials) | Future return rank | 20 stocks, rebalance quarterly | Baseline |
| V4 | 38 (+ macro indicators) | Same | Same | "Read the room" -- add economy-wide signals |
| V4.2 | 38 | Same | Same + cash rules | "Run away" -- go 100% cash in downturns |
| V4.3 | 38 | Beta-adjusted residual rank | Same + liquidity filter | "Pure alpha" -- remove market noise |
| V5 | 10 (curated subset) | Same as V4.3 | Same | "Less is more" -- only research-proven features |
| V6 | 22 (V5 + quality metrics) | Same | Same | "Quality matters" -- add company health checks |
| V7 | 5 (composite scores) | Same | Same | "Big picture only" -- 5 pillar scores |
| V8 | 25 (V6 + 3 new) | Hybrid rank (LambdaRank) | 50 stocks, overlapping cohorts | "Smooth sailing" -- staggered rebalancing |
| V9 | 23 (V6 - hvup + reversal) | Log return (Huber regression) | 50 stocks, buffer zone | "Don't overtrade" -- buffer zone reduces churn |
| V10 | 23 | Z-score (cross-sectional) | 50 stocks, sector caps | "Relative winner" -- compare within same day |
| V11 | 23 | Sector-neutral Z-score | 50 stocks, risk-adjusted weights | "Best in class" -- compare within same sector |

---

## V3 -- The Baseline

### What data it uses (28 features)

**Price Momentum (6 features)** -- "Is the stock trending up or down?"

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `mom_5d` | Price change over the last 5 trading days (~1 week) | Short bursts of buying often continue for a few more days |
| `mom_60d` | Price change over 3 months | Medium-term trends tend to persist ("momentum effect") |
| `mom_126d` | Price change over 6 months | The classic academic momentum factor |
| `rs_vs_market_20d` | How much this stock beat (or lagged) the overall market over 20 days | A stock rising while the market is flat is more meaningful than one rising in a broad rally |
| `rs_vs_sector_20d` | Same idea, but compared to stocks in the same industry | The leader within its sector tends to keep leading |
| `rs_acceleration` | Is relative strength speeding up or slowing down? (20d RS minus 60d RS) | Acceleration means fresh money is coming in |

**Volume & Money Flow (7 features)** -- "Is smart money buying?"

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `volume_surprise` | Today's trading volume divided by the 20-day average | A sudden spike in volume often precedes a big price move |
| `volume_trend` | 5-day average volume / 20-day average volume | Rising volume trend = growing interest from traders |
| `value_surprise` | Same as volume surprise but measured in KRW traded (price x volume) | Catches cases where a higher share price drives up dollar volume without more shares traded |
| `accumulation_index` | Running total of up-volume minus down-volume | Positive = more shares changing hands on up-days = buying pressure |
| `smart_money_flow` | Price-weighted volume flow that gives more weight to the close | Institutional investors tend to trade near the close |
| `volume_breakout` | Is volume breaking above a recent ceiling? | Volume breakouts often validate price breakouts |
| `price_volume_synergy` | Price going up AND volume going up at the same time? | A price rise on thin volume is suspicious; a rise on heavy volume is real |

**Volatility & Risk (5 features)** -- "How bumpy is the ride?"

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `volatility_20d` | Standard deviation of daily returns over 20 days | More volatile stocks need a bigger edge to be worth the risk |
| `volatility_ratio` | Short-term vol / long-term vol | A falling ratio (volatility contraction) often comes right before a breakout |
| `drawdown_from_high` | How far the stock has fallen from its recent peak | Deeply discounted stocks can snap back; also warns us about falling knives |
| `recovery_from_low` | How far the stock has bounced from its recent trough | A stock that has started recovering may have more room to run |
| `rolling_beta` | How much the stock moves relative to the market (1.0 = same, 2.0 = twice as much) | Used later (V4.3+) to strip out market-driven returns |

**"Gut Feel" Pattern Features (7 features)** -- "Classic setups that traders watch for"

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `past_glory_1y` | The biggest single-period run-up in the past year | Stocks that *can* make big moves are more likely to do it again |
| `fallen_angel_score` | High past glory + deep current drawdown | A stock that was once loved and is now cheap -- potential comeback story |
| `vcp_score` | Volatility Contraction Pattern -- price swings getting smaller and smaller | Textbook setup before a breakout (Mark Minervini's pattern) |
| `glory_correction_volume` | Combines past glory, drawdown, and volume spike | The trifecta: a stock people loved, now cheap, with sudden buying interest |
| `fear_greed_signal` | Stock near its lows but volume is surging | "Be greedy when others are fearful" -- contrarian buy signal |
| `smart_accumulation` | Quiet, persistent buying without big price moves | Institutional accumulation -- the "smart money" is building a position slowly |
| `vcp_breakout_potential` | VCP score multiplied by rising volume trend | The pattern is set AND the fuel (volume) is arriving |

**Traditional Technical Indicators (2 features)**

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `rsi_14` | Relative Strength Index -- 0 to 100 oscillator | Below 30 = oversold (potential bounce), above 70 = overbought |
| `bb_squeeze` | Bollinger Band width is unusually narrow | Narrow bands = low volatility = often precedes a sharp move |

**Financial Health (1 composite feature)**

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `financial_quality_index` | Single score combining ROE improvement, revenue acceleration, and margin improvement | Ensures we're buying stocks of companies that are actually getting better, not just riding hype |

### How it picks stocks

- **Model**: LightGBM regression -- predicts a rank score for each stock each day.
- **Target**: `target_rank_{horizon}d` -- the actual return-based ranking over the next *horizon* trading days.
- **Portfolio**: Top 20 stocks by predicted rank. Equal weight. Rebalance every quarter (63 trading days).
- **QEPM mode** (optional): Limits to max 3 stocks per sector, weights inversely by volatility (riskier stocks get less money).

---

## V4 -- Macro Regime Detection

**Key idea**: V3 only looks at individual stocks. V4 adds "what is the overall economy doing?" so the model can learn that the same stock pattern behaves differently in bull vs bear markets.

### New data (10 macro features added, total 38)

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `market_regime_score` | KOSPI index price vs its 120-day moving average | Above average = bull market, below = bear. The single most important market timing signal |
| `kosdaq_regime_score` | Same for KOSDAQ (small-cap index) | Small caps often lead market turns |
| `size_spread` | Large-cap return minus small-cap return | When big stocks outperform small ones, it often signals risk-off behavior |
| `market_breadth` | Percentage of sectors above their moving average | Healthy markets have broad participation; narrow rallies are fragile |
| `fear_index_delta` | 5-day change in VKOSPI (Korea's "fear gauge") | A sudden spike = panic; the model learns to be cautious |
| `fear_index_level` | Absolute level of VKOSPI | Persistently high fear = bad environment for stock picking |
| `dollar_impact` | USD/KRW futures 20-day momentum | A surging dollar often means foreign investors are pulling out of Korea |
| `bond_stock_spread` | Bond return minus stock return | When bonds beat stocks, money is fleeing to safety |
| `macro_risk_score` | Composite of all the above | One-number summary of macro danger |
| `regime_momentum_interaction` | Regime score multiplied by momentum | A momentum signal is more trustworthy in a healthy market |

### Portfolio changes

Optional cash timing: when `market_regime_score` falls below a threshold, the model reduces stock exposure.

---

## V4.2 -- Market Exit Strategy

**Key idea**: V4's cash timing was too gentle. V4.2 goes all-in on capital preservation: if the market is below its moving average, go to 100% cash. No half measures.

### Same features as V4, but new portfolio rules

- **100% cash rule**: When `market_regime_score < 0` (market below 120-day MA), sell everything and sit in cash.
- **Stop-loss**: If any holding drops more than 7% within 21 days, sell it.
- **Score filter**: Optionally reject stocks below a minimum ML confidence score.

### Why this matters

In the 2022 bear market, a simple "step aside" rule would have avoided most of the ~25% KOSPI drawdown. The cost is occasionally missing the first few days of a recovery.

---

## V4.3 -- Residual Alpha (Beta-Neutral)

**Key idea**: Instead of predicting raw returns, predict *alpha* -- the part of the return that is NOT explained by the market. If the market goes up 10% and a stock goes up 15%, the alpha is ~5%.

### What changed

- **Target**: `target_residual_rank` -- ranks stocks by their beta-adjusted residual return instead of raw return.
- **Purged cross-validation**: When training, remove data that overlaps in time with the test period (prevents data leakage).
- **Liquidity filter**: Remove the top 10% most illiquid stocks (measured by Amihud ratio). Illiquid stocks have unreliable prices.
- **Sample weighting**: Training samples from high-risk macro periods get 2x weight, so the model pays extra attention to how stocks behave in crises.

### Why this matters

Raw returns mix signal (stock-specific alpha) with noise (market-wide moves). By stripping out beta, the model focuses on what it can actually predict.

---

## V5 -- Research-Backed 10-Feature Mode

**Key idea**: V4.3 used 38 features. Many were redundant or noisy. V5 keeps only 10 features that are backed by published academic research or clear economic logic.

### The 10 chosen features

| Feature | Category | Why It Survived the Cut |
|---------|----------|------------------------|
| `gp_over_assets` | Profitability | Gross profit / total assets. Robert Novy-Marx (2013) showed this is the best single profitability predictor |
| `roe_delta_yoy` | Improvement | Year-over-year change in ROE. Companies getting better tend to keep getting better |
| `pb_sector_zscore` | Value | Price-to-book relative to sector peers. Classic value factor, but sector-adjusted to avoid comparing tech to utilities |
| `intermediate_momentum` | Momentum | Price change from 6 months ago to 1 month ago (skips last month). Jegadeesh & Titman's momentum with short-term reversal removed |
| `drawdown_from_high` | Risk | How far from the peak. Both a risk indicator and a contrarian opportunity signal |
| `fallen_angel_score` | Pattern | Once-great stock now beaten down. Deep value meets behavioral finance |
| `volume_surprise` | Demand | Sudden volume spike. The clearest signal that something is happening |
| `mom_5d` | Short-term | 1-week momentum. Captures earnings surprises and news reactions |
| `market_regime_score` | Macro | Bull vs bear market. Critical context for all other signals |
| `rolling_beta_60d` | Risk | 60-day market sensitivity. Used to adjust for systematic risk |

### Why fewer features?

More features = more ways for the model to overfit (memorize noise instead of learning real patterns). With only 10 high-quality features, each one has to carry its weight.

---

## V6 -- Quality Features (5-Pillar Framework)

**Key idea**: V5's 10 features were mostly price-based. V6 adds 12 fundamental quality metrics organized into 5 "pillars" of company health. This is the Warren Buffett layer.

### New data (12 quality features, total 22)

**Pillar 1: Economic Moat** -- "Can the company defend its profits?"

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `roic` | Return on Invested Capital -- how much profit per dollar invested in the business | High ROIC = the company has pricing power or a cost advantage |
| `gross_margin_cv` | How stable is the gross margin over time? (coefficient of variation) | Stable margins = durable competitive advantage |
| `oper_margin_cv` | Same for operating margin | Volatile margins = the company is in a commodity business |

**Pillar 2: Capital Efficiency** -- "Is management using money wisely?"

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `net_debt_to_equity` | Total debt minus cash, divided by shareholder equity | Low debt = financial flexibility, less bankruptcy risk |
| `fcf_to_ni` | Free cash flow / net income | If earnings are real, they should show up as cash. If this ratio is low, earnings might be accounting tricks |
| `ocf_to_ni` | Operating cash flow / net income | Same idea -- cash doesn't lie |

**Pillar 3: Earnings Quality** -- "Are the reported numbers trustworthy?"

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `accruals_ratio` | Non-cash portion of earnings | High accruals = earnings are mostly accounting entries, not real cash. Sloan (1996) showed high-accrual stocks underperform |

**Pillar 4: Growth & Reinvestment** -- "Is the company investing in its future?"

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `revenue_cagr_3y` | 3-year compound annual revenue growth | Consistent growers tend to keep growing |
| `growth_efficiency` | How much profit growth per unit of capital reinvested | Some companies burn cash to grow; others grow efficiently |

**Pillar 5: Value** -- "Is the price right?"

| Feature | Plain English | Why It Helps |
|---------|--------------|--------------|
| `fcf_yield` | Free cash flow / market cap | The "real" earnings yield. High FCF yield = cheap |
| `ev_to_ebit` | Enterprise value / EBIT | Like P/E but accounts for debt. Lower = cheaper |
| `value_score` | Composite value score | Combines multiple value metrics into one |

---

## V7 -- 5-Pillar Only Model

**Key idea**: What if we don't use any raw features at all, and just use 5 high-level composite scores? This is the extreme "simplify everything" experiment.

### The 5 features

| Feature | What It Aggregates |
|---------|--------------------|
| `moat_score` | ROIC + margin stability |
| `capital_efficiency_score` | Debt levels + cash flow quality |
| `earnings_quality_score` | Accruals + cash-flow-to-earnings ratio |
| `reinvestment_score` | Revenue growth + growth efficiency |
| `margin_of_safety_score` | FCF yield + EV/EBIT + value composite |

### Tradeoff

Extreme simplicity reduces overfitting risk, but may lose the nuance that raw features provide. This is an experimental version.

---

## V8 -- LambdaRank + Overlapping Portfolios

**Key idea**: Two big changes. (1) Switch from regression to a ranking-specific algorithm. (2) Smooth out portfolio turnover by overlapping three sub-portfolios.

### Features (25 = V6's 22 + 3 new)

| New Feature | Plain English | Why It Helps |
|-------------|--------------|--------------|
| `inventory_sales_gap` | Inventory growing faster than sales | Manufacturing red flag -- goods piling up = demand is weakening |
| `current_ratio` | Current assets / current liabilities | Basic liquidity health check. Below 1.0 = trouble paying bills |
| `hvup_ratio` | Fraction of high-volume days that were up-days | If volume spikes happen on up-days, institutions are accumulating |

### Model change: LambdaRank

Previous versions used regression (predict a number). V8 uses **LambdaRank**, a learning-to-rank algorithm. Instead of predicting exact returns, it directly optimizes "is stock A ranked above stock B?" -- which is what we actually care about.

- **Target**: Hybrid rank (70% residual return rank + 30% absolute return rank), converted to relevance grades 0-4.
- **Metric**: NDCG (Normalized Discounted Cumulative Gain) -- a ranking quality metric.

### Portfolio change: Overlapping cohorts

Instead of replacing the entire portfolio every 63 days:
- Every 21 days, select a new "cohort" of 50 stocks.
- Hold 3 cohorts at once (current + two previous).
- Active portfolio = union of all 3 cohorts.

**Why**: If one rebalance date happens to be unlucky (bad picks), the other two cohorts dilute the damage. Also, only ~1/3 of the portfolio changes at each rebalance, cutting transaction costs.

---

## V9 -- Huber Regression + Buffer Zone

**Key idea**: LambdaRank (V8) was complex and didn't improve results enough. V9 goes back to regression but uses **Huber loss** (robust to outliers) and introduces a **buffer zone** to prevent unnecessary trading.

### Features (23 = V6's 22 - hvup + reversal)

- Removed: `hvup_ratio` (too noisy)
- Added: `short_term_reversal` (= negative of `mom_5d`). The idea: stocks that dropped sharply in the last week tend to bounce back within the next few weeks.

### Model change: Huber regression

- **Target**: `target_log_return_21d` -- the signed log of the 21-day forward return. Log transform reduces the impact of extreme outliers (stocks that go up 200% or crash 50%).
- **Huber delta = 0.5**: Errors below 0.5 are treated as normal (squared loss); errors above 0.5 are treated as outliers (absolute loss). This prevents a single crazy stock from dominating the model's learning.
- **21-day horizon**: Shorter than V8's 63 days. Korean stocks are heavily driven by themes and news, which play out over weeks, not months.

### Portfolio change: Buffer zone

The biggest hidden cost in stock picking is **turnover** -- buying and selling too often eats into returns via commissions and slippage.

- **Buy threshold**: A stock must rank in the top 50 to enter the portfolio.
- **Hold threshold**: An existing holding stays as long as it ranks in the top 120.
- This means a stock at rank 60 won't be bought, but if it was already held and drops to rank 60, it stays. This dramatically reduces unnecessary trades.

**Daily risk management** (not annual):
- If *both* regime is bad AND VKOSPI is spiking: 0% exposure (100% cash).
- If either is bad: 50% exposure.
- Otherwise: fully invested.

---

## V10 -- Z-Score Target + Sector Constraints

**Key idea**: V9 predicted absolute log returns, which vary wildly across different market conditions (a "good" return in a crash is different from a "good" return in a boom). V10 predicts **cross-sectional Z-scores** -- how a stock compares to other stocks *on the same day*.

### What changed

**Target**: `target_zscore_42d`
- For each date, take all stocks' 42-day forward returns, compute the mean and standard deviation, then convert each stock's return to a Z-score: `(return - mean) / std`.
- A Z-score of +2 means "this stock's return was 2 standard deviations above average that day."
- This makes the target stationary -- the model doesn't need to learn "what is a good return in absolute terms" because it's always relative.

**Model tuning**:
- Learning rate: 0.01 (10x slower than V9). Slower learning = less overfitting.
- Max trees: 1000 (2x V9's 500). More trees offset the slower learning rate.
- Huber delta: 1.0 (wider than V9's 0.5). The Z-score target already handles outliers, so the loss function can be less aggressive.
- **Shuffled validation**: 10% of training data is randomly shuffled out for validation (instead of using the last few months). This prevents the model from under-fitting when recent months look different from older data.
- Patience: 100 rounds (vs V9's 50). More patience before stopping.

**Portfolio**:
- **42-day horizon** (up from 21 in V9). Reduces rebalancing frequency and transaction costs.
- **Sector cap: max 7 stocks per sector**. Prevents the model from loading up on one hot sector.
- Tighter buffer zone: hold threshold = 2.0x (was 2.4x in V9).

---

## V11 -- Sector-Neutral + Risk-Adjusted (Current Best)

**Key idea**: V10 compared stocks across the whole market. But tech stocks and utility stocks have very different return distributions. V11 compares each stock **only to other stocks in the same sector**, making the signal even purer. It also adjusts stock scores by their volatility and assigns unequal weights.

### What changed

**Target**: `target_sector_zscore_42d`
- Same Z-score idea as V10, but computed *within each sector*. A tech stock is compared to other tech stocks; a bank is compared to other banks.
- This forces the model to find alpha within every sector, not just overweight the best sector.

**Risk-adjusted ranking**:
- Raw model score is divided by `volatility_20d`: `risk_adj_score = pred_score / volatility`.
- A stock with a score of 10 and volatility of 5% is ranked the same as a stock with score 20 and volatility of 10%. This penalizes risky bets.

**Rank-based weighting** (not equal weight):
- Stocks ranked 1-10: get **3x** weight.
- Stocks ranked 11-30: get **2x** weight.
- Stocks ranked 31-50: get **1x** weight.
- The model's highest-conviction picks get the most capital.

**Dynamic cash allocation** (graduated, not binary):
- Regime bad AND VKOSPI > 25: **30% invested** (70% cash).
- Regime bad OR VKOSPI > 25: **50-70% invested**.
- Normal conditions: **100% invested**.

**Lower effective slippage**: Rank weighting means top picks rarely change (they stay on top), so actual turnover in the high-weight positions is very low. Slippage multiplier is 0.3x.

---

## Data Pipeline Summary

All models consume the same underlying data, just different subsets:

```
Raw Data Sources
├── Stock prices (daily OHLCV from KRX)          → Momentum, Volume, Volatility features
├── Financial statements (quarterly from DART)     → Fundamental, Quality features
├── Market indices (KOSPI, KOSDAQ, sector indices) → Macro, Regime features
├── VKOSPI (volatility index)                      → Fear/risk features
├── USD/KRW futures                                → Dollar impact feature
└── Bond yields                                    → Bond-stock spread feature
```

---

## Version Inheritance Chain

```
V3 (baseline)
 └─ V4 (+macro)
     └─ V4.2 (+market exit)
         └─ V4.3 (+residual alpha, purged CV, liquidity filter)
             └─ V5 (10 curated features)
                 └─ V6 (+12 quality features = 22 total)
                     ├─ V7 (5 composite scores only -- experimental)
                     ├─ V8 (+3 features, LambdaRank, overlapping portfolios)
                     └─ V9 (-hvup +reversal = 23 features, Huber, buffer zone)
                         └─ V10 (Z-score target, sector caps, 42d horizon)
                             └─ V11 (sector-neutral Z-score, risk-adjusted, rank weights)
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **Alpha** | Return above what the market delivered. If the market returned 10% and your stock returned 15%, your alpha is 5%. |
| **Beta** | How sensitive a stock is to market moves. Beta of 1.5 means if the market moves 1%, the stock moves ~1.5%. |
| **Drawdown** | The drop from a peak to a trough. A stock at 80 that was once at 100 has a 20% drawdown. |
| **ICIR** | Information Coefficient / Information Ratio. Measures how consistently the model's predictions correlate with actual outcomes. Higher is better. |
| **Huber loss** | A loss function that behaves like squared error for small errors but like absolute error for large ones. Makes the model robust to outliers. |
| **LambdaRank** | A machine learning algorithm designed specifically for ranking tasks (like search engines or stock selection). |
| **LightGBM** | A fast gradient boosting framework that builds many small decision trees, each correcting the previous one's mistakes. |
| **Momentum** | The tendency of stocks that went up recently to continue going up (and vice versa). One of the strongest market anomalies. |
| **NDCG** | Normalized Discounted Cumulative Gain. A metric that measures ranking quality, giving more credit for getting the top positions right. |
| **Rebalance** | Updating the portfolio -- selling stocks that no longer qualify and buying new ones. |
| **Slippage** | The difference between the expected price and the actual execution price. In practice, you always buy slightly high and sell slightly low. |
| **Turnover** | How much of the portfolio changes at each rebalance. 50% turnover means half the stocks are replaced. |
| **VKOSPI** | Korea's volatility index, similar to the US VIX. High values = high fear in the market. |
| **Z-score** | How many standard deviations a value is from the mean. Z=0 is average, Z=+2 is very high, Z=-2 is very low. |
