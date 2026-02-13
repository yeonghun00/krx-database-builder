"""
Feature Engineering V4 - 매크로 Regime Detection 통합

V2 → V3 → V4 진화:
1. 모멘텀/수급 피처 대폭 강화 (재무 30-40% 목표)
2. 섹터 중립화 (Sector Neutralization)
3. Delta 피처 추가 (QoQ, YoY 변화)
4. 본능 전략 피처 (낙폭과대, 거래량 폭발, 과거 영광)
5. [V4 신규] 매크로 Regime Detection - 2021~2022 폭락장 회피용
   - market_regime_score: 시장 이격도
   - fear_index_delta: VKOSPI 변화
   - dollar_impact: 달러선물 모멘텀
   - bond_stock_spread: 채권 vs 주식
   - sector_relative_momentum: 섹터 대비 종목 알파
"""

import gc
import sys
import multiprocessing
import pandas as pd
import numpy as np
import sqlite3
import logging

# Force unbuffered output so timing prints appear instantly
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
from typing import List, Optional
from pathlib import Path

# Optional numba for accelerated rolling beta
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _rolling_beta_numba(ret, mkt, window, min_periods):
        """Single-pass rolling beta: cov(ret, mkt) / var(mkt), per element."""
        n = len(ret)
        beta = np.full(n, np.nan, dtype=np.float64)
        for i in range(min_periods - 1, n):
            start = max(0, i - window + 1)
            cnt = i - start + 1
            if cnt < min_periods:
                continue
            r = ret[start:i + 1]
            m = mkt[start:i + 1]
            mean_r = 0.0
            mean_m = 0.0
            for j in range(cnt):
                mean_r += r[j]
                mean_m += m[j]
            mean_r /= cnt
            mean_m /= cnt
            cov = 0.0
            var = 0.0
            for j in range(cnt):
                dr = r[j] - mean_r
                dm = m[j] - mean_m
                cov += dr * dm
                var += dm * dm
            cov /= cnt
            var /= cnt
            if var > 1e-8:
                b = cov / var
                beta[i] = max(-3.0, min(3.0, b))
            else:
                beta[i] = 1.0
        return beta

# Import financial feature generator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.financial_features import FinancialFeatureGenerator
from ml.macro_features import MacroFeatureEngineer


_SHARED_FIN_PIVOT = None  # Set by main process, inherited by fork workers (copy-on-write)


class FeatureEngineer:
    """V3: 퀀트 팀장 피드백 반영 - 피처 다이어트 + 본능 강화"""

    CACHE_VERSION = "v7_5pillar_opt_20260207"

    # =========================================================================
    # 피처 그룹 정의 (V3: 45개 → 25개로 압축)
    # =========================================================================

    # [그룹 1] 모멘텀 피처 - V4.1 터보 엔진 🔥
    MOMENTUM_FEATURES = [
        'mom_5d',                # 단기 모멘텀 (1주)
        'mom_60d',               # 중기 모멘텀 (3개월)
        'mom_126d',              # 장기 모멘텀 (6개월)
        'rs_vs_market_20d',      # 시장 대비 상대강도
        # === V4.1 신규 ===
        'rs_vs_sector_20d',      # 🔥 섹터 대비 상대강도 (섹터 내 대장주)
        'rs_acceleration',       # 🔥 상대강도 가속도 (20d - 60d)
    ]

    # [그룹 2] 수급/거래량 피처 - V4.1 터보 엔진 🔥
    VOLUME_FEATURES = [
        'volume_surprise',       # 거래량 폭발 (20일 평균 대비)
        'volume_trend',          # 거래량 추세 (5일 vs 20일)
        'value_surprise',        # 거래대금 폭발
        'accumulation_index',    # 누적/배분 지표
        'smart_money_flow',      # 스마트머니 흐름
        'volume_breakout',       # 거래량 돌파 신호
        # === V4.1 신규: "돈 냄새" 피처 ===
        'price_volume_synergy',  # 🔥 가격×거래량 시너지 (진짜 상승)
    ]

    # [그룹 3] 변동성/리스크 피처 - V4.3 Beta 추가
    VOLATILITY_FEATURES = [
        'volatility_20d',
        'volatility_ratio',      # 단기/장기 변동성 비율 (VCP)
        'drawdown_from_high',    # 고점 대비 낙폭 🔥
        'recovery_from_low',     # 저점 대비 반등
        'rolling_beta',          # 🔥 V4.3: 시장 Beta (Residual 계산용)
    ]

    # [그룹 4] 본능 전략 피처 - V4.1 터보 엔진 🔥
    INTUITION_FEATURES = [
        'past_glory_1y',         # 1년간 최대 상승률
        'fallen_angel_score',    # 추락한 천사 점수
        'vcp_score',             # Volatility Contraction Pattern
        'glory_correction_volume',  # 영광 * 낙폭 * 거래량폭발
        'fear_greed_signal',        # 공포 속 탐욕 신호
        'smart_accumulation',       # 스마트머니 매집 신호
        # === V4.1 신규: 폭발 직전 포착 ===
        'vcp_breakout_potential',   # 🔥 VCP × 거래량추세 (폭발 임박)
    ]

    # [그룹 5] 전통적 기술 지표 - 축소
    TRADITIONAL_FEATURES = [
        'rsi_14',
        'bb_squeeze',            # 볼린저밴드 수축
    ]

    # [그룹 6] 재무 피처 - V4.1: 단일 피처로 압축! 🔥
    # "재무는 입장권, 우승자는 모멘텀에서 나온다"
    FUNDAMENTAL_FEATURES = [
        'financial_quality_index',  # 🔥 재무 4개 통합 (ROE + 매출가속 + 마진개선)
    ]

    # 내부 계산용 (피처로 직접 사용 안 함)
    _FUNDAMENTAL_RAW = [
        'roe_delta_qoq',
        'roe_sector_zscore',
        'revenue_growth_accel',
        'margin_improvement',
    ]

    # [그룹 7] V4 신규: 매크로 Regime Detection 피처 🔥🔥🔥
    # "2021~2022년 폭락장을 피하기 위한 시장 온도계"
    MACRO_FEATURES = [
        # 1단계: 시장의 온도계 (Regime Detection)
        'market_regime_score',      # KOSPI 200 이격도 (120일 MA 대비)
        'kosdaq_regime_score',      # KOSDAQ 150 이격도
        'size_spread',              # 대형주 - 소형주 수익률 (음수 = 불장)
        'market_breadth',           # MA 위 섹터 비율 (0~1)

        # 3단계: 매크로 공포 레이더 (Inter-market Analysis)
        'fear_index_delta',         # VKOSPI 5일 변화 (급등 = 위험)
        'fear_index_level',         # VKOSPI 절대 레벨
        'dollar_impact',            # 달러선물 20일 모멘텀 (급등 = 외인 이탈)
        'bond_stock_spread',        # 채권 - 주식 수익률 (양수 = Risk-off)

        # 복합 피처
        'macro_risk_score',         # 종합 매크로 리스크 점수
        'regime_momentum_interaction',  # regime * momentum 상호작용
    ]

    # [V5] Research-backed 10-Feature Mode
    V5_FEATURES = [
        'gp_over_assets', 'roe_delta_yoy', 'pb_sector_zscore',
        'intermediate_momentum', 'drawdown_from_high', 'fallen_angel_score',
        'volume_surprise', 'mom_5d', 'market_regime_score', 'rolling_beta_60d',
    ]

    # [V6] Fundamental Quality Features (5-Pillar Framework)
    QUALITY_FEATURES = [
        'roic', 'gross_margin_cv', 'oper_margin_cv',
        'net_debt_to_equity', 'fcf_to_ni', 'ocf_to_ni', 'accruals_ratio',
        'revenue_cagr_3y', 'growth_efficiency',
        'fcf_yield', 'ev_to_ebit', 'value_score',
    ]

    V6_FEATURES = V5_FEATURES + QUALITY_FEATURES

    # [V8] V6 + Manufacturing/Liquidity/Accumulation features (25 total)
    V8_FEATURES = V6_FEATURES + [
        'inventory_sales_gap',   # Manufacturing red flag
        'current_ratio',         # Liquidity health
        'hvup_ratio',            # Institutional accumulation proxy
    ]

    # [V9] V6 + short-term reversal, no hvup (23 features)
    V9_FEATURES = [f for f in V6_FEATURES if f != 'hvup_ratio'] + [
        'short_term_reversal',
    ]

    # [V7] 5-Pillar Only Model (Moat / Capital / Earnings / Reinvest / Value)
    MODEL7_FEATURES = [
        'moat_score',
        'capital_efficiency_score',
        'earnings_quality_score',
        'reinvestment_score',
        'margin_of_safety_score',
    ]

    # 전체 피처 리스트
    FEATURE_COLUMNS = (
        MOMENTUM_FEATURES +
        VOLUME_FEATURES +
        VOLATILITY_FEATURES +
        INTUITION_FEATURES +
        TRADITIONAL_FEATURES +
        FUNDAMENTAL_FEATURES +
        MACRO_FEATURES  # V4 추가
    )

    def __init__(self, db_path: str = "krx_stock_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._indexes_ensured = False

    def _ensure_indexes(self):
        """Create covering indexes for faster SQL queries (one-time)."""
        import time as _t
        if self._indexes_ensured:
            return
        print('    🔧 Creating SQL indexes (one-time)...')
        t0 = _t.time()
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            # Primary performance index: covers ORDER BY stock_code, date
            cur.execute("CREATE INDEX IF NOT EXISTS idx_dp_stock_date ON daily_prices(stock_code, date)")
            print(f'      idx_dp_stock_date: {_t.time()-t0:.1f}s')
            t1 = _t.time()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_dp_market_date_stock ON daily_prices(market_type, date, stock_code)")
            print(f'      idx_dp_market_date_stock: {_t.time()-t1:.1f}s')
            t1 = _t.time()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_fp_consol_avail ON financial_periods(consolidation_type, available_date)")
            print(f'      idx_fp_consol_avail: {_t.time()-t1:.1f}s')
            t1 = _t.time()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_bs_period_item ON financial_items_bs_cf(period_id, item_code_normalized)")
            print(f'      idx_bs_period_item: {_t.time()-t1:.1f}s')
            t1 = _t.time()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pl_period_item ON financial_items_pl(period_id, item_code_normalized)")
            print(f'      idx_pl_period_item: {_t.time()-t1:.1f}s')
            conn.commit()
            self._indexes_ensured = True
            print(f'    🔧 Indexes done: {_t.time()-t0:.1f}s total')
        except Exception as e:
            self.logger.warning(f"Index creation failed (non-fatal): {e}")
            print(f'    ⚠️ Index creation failed: {e}')
        finally:
            conn.close()

    def _optimize_dtypes(self, df):
        """Downcast float64→float32 and use categoricals for memory + speed."""
        float_cols = df.select_dtypes('float64').columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype('float32')
        for col in ['stock_code', 'sector', 'market_type']:
            if col in df.columns:
                df[col] = df[col].astype('category')
        return df

    def load_raw_data(self, start_date: str, end_date: str,
                      markets: List[str] = None,
                      min_market_cap: int = 0,
                      use_cache: bool = True) -> pd.DataFrame:
        """Load raw OHLCV data from database.

        Strategy (11GB DB, 8.7M rows):
        1. Parquet cache → 1.7s (best case, warm)
        2. Helper table + temp table JOIN → 69s for 5M rows (cold)
        3. Never use SQL JOIN with stocks or IN-clause (catastrophic)
        """
        import time as _t
        import os
        import hashlib

        markets = markets or ['kospi', 'kosdaq']

        # Raw data parquet cache (SQL: 69s → parquet: 1.7s)
        cache_key = f"raw_{start_date}_{end_date}_{'_'.join(sorted(markets))}_{min_market_cap}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        raw_cache = f".cache/raw_{cache_hash}.parquet"
        os.makedirs(".cache", exist_ok=True)

        t0 = _t.time()
        if use_cache and os.path.exists(raw_cache):
            db_mtime = os.path.getmtime(self.db_path)
            cache_mtime = os.path.getmtime(raw_cache)
            if cache_mtime > db_mtime:
                df = pd.read_parquet(raw_cache)
                print(f'      raw cache hit: {_t.time()-t0:.1f}s ({len(df):,} rows)')
                self.logger.info(f"Loaded {len(df):,} rows from raw cache")
                return df

        # Cold path: filtered load from SQLite
        self._ensure_indexes()
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA cache_size = -256000")

        # Pre-filter eligible stocks via latest-date market_cap filter.
        # Checks last 5 trading days (~22K rows) — always <1s, no helper table needed.
        if min_market_cap > 0:
            mcap_buffer = int(min_market_cap * 0.5)
            eligible = pd.read_sql_query(
                """SELECT DISTINCT stock_code FROM daily_prices
                   WHERE date IN (
                       SELECT DISTINCT date FROM daily_prices ORDER BY date DESC LIMIT 5
                   ) AND market_cap >= ?
                     AND market_type IN ('kospi','kosdaq')""",
                conn, params=[mcap_buffer]
            )
            codes = eligible['stock_code'].tolist()
            print(f'      eligible stocks: {len(codes)} ({_t.time()-t0:.1f}s)')
            t0 = _t.time()

            # Load into temp table for fast JOIN (avoids slow IN-clause)
            conn.execute("CREATE TEMP TABLE _elig (stock_code TEXT PRIMARY KEY)")
            conn.executemany("INSERT INTO _elig VALUES (?)", [(c,) for c in codes])

            query = """
            SELECT dp.stock_code, dp.date, dp.market_type, dp.opening_price,
                   dp.high_price, dp.low_price, dp.closing_price,
                   dp.volume, dp.value, dp.market_cap
            FROM daily_prices dp
            INNER JOIN _elig e ON dp.stock_code = e.stock_code
            WHERE dp.date >= ? AND dp.date <= ?
              AND dp.closing_price > 0
              AND dp.volume > 0
            """
            params = [start_date, end_date]
        else:
            market_placeholders = ','.join(['?' for _ in markets])
            query = f"""
            SELECT stock_code, date, market_type, opening_price, high_price,
                   low_price, closing_price, volume, value, market_cap
            FROM daily_prices
            WHERE date >= ? AND date <= ?
              AND market_type IN ({market_placeholders})
              AND closing_price > 0
              AND volume > 0
            """
            params = [start_date, end_date] + markets

        df = pd.read_sql_query(query, conn, params=params)
        t1 = _t.time()
        print(f'      SQL query: {t1-t0:.1f}s ({len(df):,} rows)')

        # Merge stock name/sector from stocks table (instant — 4750 rows)
        stocks_df = pd.read_sql_query(
            "SELECT stock_code, current_name as name, current_sector_type as sector FROM stocks",
            conn
        )
        conn.close()
        df = df.merge(stocks_df, on='stock_code', how='left')
        print(f'      stocks merge: {_t.time()-t1:.1f}s')

        # Cache raw data as parquet for next run
        if use_cache:
            t2 = _t.time()
            df.to_parquet(raw_cache, index=False)
            print(f'      raw cache saved: {_t.time()-t2:.1f}s')

        self.logger.info(f"Loaded {len(df):,} rows from {start_date} to {end_date}")
        return df

    def _load_raw_chunk(self, start_date: str, end_date: str,
                        eligible_codes: List[str]) -> pd.DataFrame:
        """Load raw OHLCV data for a specific date range and pre-computed eligible codes.

        Lightweight SQL loader for year-chunk processing — no parquet caching
        (chunks are small ~400K rows). Uses temp table JOIN for fast filtering.
        """
        import time as _t
        t0 = _t.time()

        self._ensure_indexes()
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA cache_size = -256000")

        # Load into temp table for fast JOIN
        conn.execute("CREATE TEMP TABLE _elig (stock_code TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO _elig VALUES (?)", [(c,) for c in eligible_codes])

        query = """
        SELECT dp.stock_code, dp.date, dp.market_type, dp.opening_price,
               dp.high_price, dp.low_price, dp.closing_price,
               dp.volume, dp.value, dp.market_cap
        FROM daily_prices dp
        INNER JOIN _elig e ON dp.stock_code = e.stock_code
        WHERE dp.date >= ? AND dp.date <= ?
          AND dp.closing_price > 0
          AND dp.volume > 0
        """
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])

        # Merge stock name/sector from stocks table
        stocks_df = pd.read_sql_query(
            "SELECT stock_code, current_name as name, current_sector_type as sector FROM stocks",
            conn
        )
        conn.close()
        df = df.merge(stocks_df, on='stock_code', how='left')

        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all V2 features."""
        import time as _t
        self.logger.info("Computing V2 features (momentum-heavy)...")

        t0 = _t.time()
        df = df.sort_values(['stock_code', 'date']).copy()
        grouped = df.groupby('stock_code')

        # Calculate returns first (needed for forward returns calculation)
        df['return'] = grouped['closing_price'].pct_change()
        df['log_return'] = np.log1p(df['return'])
        print(f'      sort+returns: {_t.time()-t0:.1f}s')

        t0 = _t.time()
        self._compute_momentum_features(df, grouped)
        print(f'      momentum features: {_t.time()-t0:.1f}s')

        t0 = _t.time()
        self._compute_volume_features(df, grouped)
        print(f'      volume features: {_t.time()-t0:.1f}s')

        t0 = _t.time()
        self._compute_volatility_features(df, grouped)
        print(f'      volatility features: {_t.time()-t0:.1f}s')

        t0 = _t.time()
        self._compute_intuition_features(df, grouped)
        print(f'      intuition features: {_t.time()-t0:.1f}s')

        t0 = _t.time()
        self._compute_traditional_features(df, grouped)
        print(f'      traditional features: {_t.time()-t0:.1f}s')

        t0 = _t.time()
        self._apply_sector_neutralization(df)
        print(f'      sector neutralization: {_t.time()-t0:.1f}s')

        # Rolling beta (horizon-independent, cached with tech features)
        self._compute_rolling_beta(df, grouped)

        # Cleanup
        self._cleanup_intermediate_cols(df)
        # Drop 52w high/low after intuition features have used them
        df.drop(columns=['high_52w', 'low_52w'], errors='ignore', inplace=True)

        self.logger.info(f"Computed {len(self.FEATURE_COLUMNS)} V2 features")
        return df

    def _compute_momentum_features(self, df: pd.DataFrame, grouped) -> None:
        """모멘텀 피처 계산 (V3: 압축됨 - 5d, 60d, 126d, RS만) - 최적화 버전"""

        # Multi-timeframe momentum - 벡터화 (pct_change는 이미 빠름)
        for period in [5, 10, 20, 60, 120, 126]:
            df[f'mom_{period}d'] = grouped['closing_price'].pct_change(period)

        # V5: Intermediate Momentum (skip last month) - price_t-21 / price_t-126 - 1
        df['intermediate_momentum'] = grouped['closing_price'].shift(21) / grouped['closing_price'].shift(126) - 1

        # Moving averages - 벡터화된 rolling
        for period in [5, 20, 60, 120]:
            df[f'ma_{period}'] = grouped['closing_price'].rolling(
                period, min_periods=period//2
            ).mean().droplevel(0)

        # dist_ma 한 번에 계산
        for period in [5, 20, 60, 120]:
            df[f'dist_ma_{period}'] = (
                df['closing_price'] / df[f'ma_{period}'].clip(lower=1) - 1
            )

        # MA Trend (정배열 여부): 5 > 20 > 60
        df['ma_trend'] = (
            (df['ma_5'] > df['ma_20']).astype(float) * 0.5 +
            (df['ma_20'] > df['ma_60']).astype(float) * 0.5
        )

        # Relative Strength vs Market - 벡터화
        for period in [20, 60]:
            market_ret = df.groupby('date')[f'mom_{period}d'].transform('median')
            df[f'rs_vs_market_{period}d'] = df[f'mom_{period}d'] - market_ret

        # Momentum Consistency (상승일 비율)
        df['up_day'] = (df['return'] > 0).astype(float)
        df['mom_consistency'] = grouped['up_day'].rolling(
            20, min_periods=10
        ).mean().droplevel(0)

        # Momentum Acceleration (최근 모멘텀 / 과거 모멘텀)
        df['mom_acceleration'] = df['mom_20d'] / df['mom_60d'].clip(lower=0.01).abs()
        df['mom_acceleration'] = df['mom_acceleration'].clip(-5, 5)

        # === V4.1 신규: 터보 모멘텀 피처 ===
        # RS Acceleration (상대강도 가속도) - "최근 더 강해지는 놈"
        df['rs_acceleration'] = df['rs_vs_market_20d'] - df.get('rs_vs_market_60d', 0)
        df['rs_acceleration'] = df['rs_acceleration'].clip(-0.5, 0.5)

    def _compute_volume_features(self, df: pd.DataFrame, grouped) -> None:
        """수급/거래량 피처 계산 - 최적화 버전"""

        # Rolling 연산을 한 번에 묶어서 처리
        vol_rolling_5 = grouped['volume'].rolling(5, min_periods=3)
        vol_rolling_20 = grouped['volume'].rolling(20, min_periods=10)
        vol_rolling_60 = grouped['volume'].rolling(60, min_periods=30)

        df['vol_5d'] = vol_rolling_5.mean().droplevel(0)
        df['vol_20d'] = vol_rolling_20.mean().droplevel(0)
        df['vol_60d_max'] = vol_rolling_60.max().droplevel(0)

        # Volume Surprise / Trend / Breakout - 벡터 연산
        df['volume_surprise'] = df['volume'] / df['vol_20d'].clip(lower=1)
        df['volume_trend'] = df['vol_5d'] / df['vol_20d'].clip(lower=1)
        df['volume_breakout'] = df['volume'] / df['vol_60d_max'].clip(lower=1)

        # Value Surprise (거래대금 폭발)
        df['value_20d'] = grouped['value'].rolling(20, min_periods=10).mean().droplevel(0)
        df['value_surprise'] = df['value'] / df['value_20d'].clip(lower=1)

        # Smart Money Flow (종가 위치 * 거래량)
        df['close_location'] = (
            (df['closing_price'] - df['low_price']) /
            (df['high_price'] - df['low_price']).clip(lower=1)
        )
        df['daily_mf'] = (df['close_location'] * 2 - 1) * df['volume']

        # Rolling sums - 한 번에 계산
        mf_rolling_sum = grouped['daily_mf'].rolling(20, min_periods=10).sum().droplevel(0)
        vol_rolling_sum = grouped['volume'].rolling(20, min_periods=10).sum().droplevel(0)
        df['smart_money_flow'] = mf_rolling_sum / vol_rolling_sum.clip(lower=1)

        # Accumulation Index
        df['accumulation_index'] = grouped['smart_money_flow'].rolling(
            10, min_periods=5
        ).mean().droplevel(0)

        # === V4.1 신규: "돈 냄새" 피처 ===
        # Price-Volume Synergy (가격×거래량 시너지) - "진짜 상승"만 포착
        df['price_volume_synergy'] = (
            df['mom_5d'].clip(-0.3, 0.3) *
            (df['volume_surprise'] - 1).clip(0, 5)
        )
        df['price_volume_synergy'] = df['price_volume_synergy'].clip(-1, 1)

        # === V8: HVUP (High-Volume Up-Days) — institutional accumulation proxy ===
        up_big = (df['return'] > 0.02).astype(float)
        df['_vol_on_up'] = df['volume'] * up_big
        hvup_vol = grouped['_vol_on_up'].rolling(20, min_periods=10).sum().droplevel(0)
        total_vol = grouped['volume'].rolling(20, min_periods=10).sum().droplevel(0)
        df['hvup_ratio'] = (hvup_vol / total_vol.clip(lower=1)).clip(0, 1)
        df.drop(columns=['_vol_on_up'], errors='ignore', inplace=True)

        # === V9: Short-term reversal (mean reversion signal) ===
        df['short_term_reversal'] = -df['mom_5d'].clip(-0.3, 0.3)

        # === V4.3 신규: Amihud Illiquidity 🔥 ===
        # "거래량 대비 가격 변동이 큰 종목 = 슬리피지 지옥"
        # 높을수록 비유동적 → 필터링 대상
        df['amihud_illiquidity'] = (
            df['return'].abs() / (df['value'].clip(lower=1e6) / 1e9)  # 10억 단위
        )
        df['amihud_illiquidity'] = grouped['amihud_illiquidity'].rolling(
            20, min_periods=10
        ).mean().droplevel(0)
        # Percentile rank (높을수록 비유동적)
        df['amihud_rank'] = df.groupby('date')['amihud_illiquidity'].rank(pct=True)

    def _compute_volatility_features(self, df: pd.DataFrame, grouped) -> None:
        """변동성/리스크 피처 계산 - 최적화 버전"""

        # Historical Volatility - 벡터화
        sqrt_252 = np.sqrt(252)
        df['volatility_20d'] = grouped['return'].rolling(20, min_periods=10).std().droplevel(0) * sqrt_252
        df['volatility_60d'] = grouped['return'].rolling(60, min_periods=30).std().droplevel(0) * sqrt_252

        # Volatility Ratio (VCP 패턴 감지)
        df['volatility_ratio'] = df['volatility_20d'] / df['volatility_60d'].clip(lower=0.01)

        # ATR - 벡터화
        prev_close = grouped['closing_price'].shift(1)
        df['tr'] = np.maximum(
            df['high_price'] - df['low_price'],
            np.maximum(
                (df['high_price'] - prev_close).abs(),
                (df['low_price'] - prev_close).abs()
            )
        )
        df['atr_20'] = grouped['tr'].rolling(20, min_periods=10).mean().droplevel(0)
        df['atr_ratio'] = df['tr'] / df['atr_20'].clip(lower=1)

        # Drawdown from High / Recovery from Low - 벡터화
        df['high_52w'] = grouped['high_price'].rolling(252, min_periods=126).max().droplevel(0)
        df['low_52w'] = grouped['low_price'].rolling(252, min_periods=126).min().droplevel(0)

        df['drawdown_from_high'] = df['closing_price'] / df['high_52w'].clip(lower=1) - 1
        df['recovery_from_low'] = df['closing_price'] / df['low_52w'].clip(lower=1) - 1

    def _compute_intuition_features(self, df: pd.DataFrame, grouped) -> None:
        """본능 전략 피처: 존나 센 놈 + 조정 -30~50%"""

        # 과거의 영광 (1년간 최대 상승률)
        df['past_glory_1y'] = df['high_52w'] / df['low_52w'].clip(lower=1) - 1

        # 영광 대비 현재 낙폭
        df['max_drawdown_from_glory'] = df['drawdown_from_high']

        # Fallen Angel Score (추락한 천사)
        # 과거에 잘 나갔는데 (glory > 100%) 지금 많이 빠진 (-30% ~ -50%)
        glory_condition = (df['past_glory_1y'] > 1.0).astype(float)
        drawdown_condition = (
            (df['drawdown_from_high'] < -0.30) &
            (df['drawdown_from_high'] > -0.50)
        ).astype(float)
        df['fallen_angel_score'] = glory_condition * drawdown_condition * (
            df['past_glory_1y'] * (-df['drawdown_from_high'])
        )

        # Bounce Potential (반등 잠재력)
        # 낙폭 + 거래량 축소 + 변동성 수축
        volume_dryup = (df['volume_trend'] < 0.8).astype(float)
        vol_contraction = (df['volatility_ratio'] < 0.7).astype(float)
        df['bounce_potential'] = (
            df['fallen_angel_score'] *
            (1 + volume_dryup * 0.3) *
            (1 + vol_contraction * 0.3)
        )

        # VCP Score (Volatility Contraction Pattern)
        # 변동성 수축 + 거래량 축소 + 가격 횡보
        price_stable = ((df['dist_ma_20'].abs() < 0.05)).astype(float)
        df['vcp_score'] = (
            vol_contraction *
            volume_dryup *
            price_stable *
            df['past_glory_1y'].clip(0, 2)
        )

        # ================================================================
        # V3 신규: 결합 피처 (Interaction Features) 🔥
        # "피처를 나열만 하지 말고, 본능 전략용 결합 피처를 직접 만들어라"
        # ================================================================

        # Glory_Correction_Volume: 영광 * 낙폭 * 거래량폭발
        # "과거에 화려했고 + 지금 충분히 빠졌는데 + 거래량이 터지기 시작한 놈"
        df['glory_correction_volume'] = (
            df['past_glory_1y'].clip(0, 5) *
            (-df['drawdown_from_high']).clip(0, 1) *
            (df['volume_surprise'] - 1).clip(0, 10)
        )

        # Fear_Greed_Signal: 공포 속 탐욕 신호
        # "남들이 공포(Volatility)를 느낄 때 탐욕(Volume)을 발견"
        high_volatility = (df['volatility_20d'] > df['volatility_60d']).astype(float)
        volume_spike = (df['volume_surprise'] > 2.0).astype(float)
        price_down = (df['drawdown_from_high'] < -0.20).astype(float)
        df['fear_greed_signal'] = high_volatility * volume_spike * price_down * df['past_glory_1y'].clip(0, 3)

        # Smart_Accumulation: 스마트머니 매집 신호
        # "조용히 매집 중 - 거래량 증가 + 스마트머니 유입 + 낙폭과대"
        smart_inflow = (df['smart_money_flow'] > 0.3).astype(float)
        df['smart_accumulation'] = (
            smart_inflow *
            (df['accumulation_index'] + 1).clip(0, 2) *
            (-df['drawdown_from_high']).clip(0, 0.5) *
            df['volume_trend'].clip(0.5, 2)
        )

        # === V4.1 신규: 폭발 직전 포착 ===
        # VCP Breakout Potential (VCP × 거래량 추세)
        # 변동성이 죽어가다가 + 거래량이 고개를 드는 = 폭발 임박
        df['vcp_breakout_potential'] = (
            df['vcp_score'].clip(0, 2) *
            (df['volume_trend'] - 0.8).clip(0, 2)  # 거래량이 평균 이상으로 늘어날 때
        )

    def _compute_traditional_features(self, df: pd.DataFrame, grouped) -> None:
        """전통적 기술 지표 - 최적화 버전"""

        # RSI - 벡터화
        delta = grouped['closing_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = grouped.apply(lambda x: gain.loc[x.index].rolling(14).mean()).droplevel(0) if False else \
                   gain.groupby(df['stock_code']).rolling(14).mean().droplevel(0)
        avg_loss = loss.groupby(df['stock_code']).rolling(14).mean().droplevel(0)

        rs = avg_gain / avg_loss.clip(lower=0.001)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # RSI Divergence (가격은 신저가인데 RSI는 아닌 경우 = 반등 신호)
        price_new_low = (df['closing_price'] <= df['low_52w'] * 1.05).astype(float)
        rsi_not_low = (df['rsi_14'] > 30).astype(float)
        df['rsi_divergence'] = price_new_low * rsi_not_low

        # Bollinger Bands - 벡터화
        df['bb_mid'] = df['ma_20']
        df['bb_std'] = grouped['closing_price'].rolling(20, min_periods=10).std().droplevel(0)
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (
            (df['closing_price'] - df['bb_lower']) /
            (df['bb_upper'] - df['bb_lower']).clip(lower=1)
        )

        # BB Squeeze (볼린저밴드 수축) - 벡터화
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].clip(lower=1)
        df['bb_width_avg'] = grouped['bb_width'].rolling(60, min_periods=30).mean().droplevel(0)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width_avg'] * 0.8).astype(float)

    def _apply_sector_neutralization(self, df: pd.DataFrame) -> None:
        """섹터 중립화: 모든 피처를 섹터 내 순위로 변환"""

        # Sector-relative momentum (핵심!) - vectorized zscore
        _grp = df.groupby(['date', 'sector'])['mom_20d']
        _mean = _grp.transform('mean')
        _std = _grp.transform('std').clip(lower=0.01)
        df['rs_vs_sector_20d'] = (df['mom_20d'] - _mean) / _std

        # 모멘텀 피처들을 섹터 내 랭크로 변환
        momentum_cols = ['mom_5d', 'mom_10d', 'mom_20d', 'mom_60d']
        for col in momentum_cols:
            if col in df.columns:
                df[f'{col}_sector_rank'] = df.groupby(['date', 'sector'])[col].rank(pct=True)

    def _compute_rolling_beta(self, df: pd.DataFrame, grouped) -> None:
        """Compute rolling beta (252d + 60d) — horizon-independent, cacheable.

        Uses numba JIT if available (~5x faster), otherwise vectorized pandas.
        """
        import time as _t
        t0 = _t.time()

        # Market return (daily median)
        market_ret_series = df.groupby('date')['return'].transform('median')
        df['_market_return_daily'] = market_ret_series

        if HAS_NUMBA:
            # ---- Numba path: single-pass per group ----
            mkt_arr = df['_market_return_daily'].values.astype(np.float64)
            ret_arr = df['return'].values.astype(np.float64)

            # Replace NaN with 0 for numba (NaN arithmetic is slower)
            mkt_arr = np.where(np.isnan(mkt_arr), 0.0, mkt_arr)
            ret_arr = np.where(np.isnan(ret_arr), 0.0, ret_arr)

            # Pre-compute group indices once (data already sorted by stock_code, date)
            stock_groups = df.groupby('stock_code').groups

            for col_name, window, min_p in [('rolling_beta', 252, 60), ('rolling_beta_60d', 60, 30)]:
                result = np.full(len(df), np.nan, dtype=np.float64)
                for _, grp_idx in stock_groups.items():
                    idx = grp_idx.values if hasattr(grp_idx, 'values') else np.array(grp_idx)
                    r = ret_arr[idx]
                    m = mkt_arr[idx]
                    beta_vals = _rolling_beta_numba(r, m, window, min_p)
                    result[idx] = beta_vals
                df[col_name] = result
                df[col_name] = df[col_name].fillna(1.0).astype('float32')
            print(f'      rolling beta (numba 252d+60d): {_t.time()-t0:.1f}s')
        else:
            # ---- Pandas vectorized path (fallback) ----
            mkt = df['_market_return_daily']
            df['_ret_x_mkt'] = df['return'] * mkt
            df['_mkt_sq'] = mkt ** 2

            g_ret = df.groupby('stock_code')['return']
            g_mkt = df.groupby('stock_code')['_market_return_daily']
            g_prod = df.groupby('stock_code')['_ret_x_mkt']
            g_msq = df.groupby('stock_code')['_mkt_sq']

            for window, min_p, col_name in [(252, 60, 'rolling_beta'), (60, 30, 'rolling_beta_60d')]:
                mean_xy = g_prod.rolling(window, min_periods=min_p).mean().droplevel(0)
                mean_x = g_ret.rolling(window, min_periods=min_p).mean().droplevel(0)
                mean_y = g_mkt.rolling(window, min_periods=min_p).mean().droplevel(0)
                mean_ysq = g_msq.rolling(window, min_periods=min_p).mean().droplevel(0)
                cov = mean_xy - mean_x * mean_y
                var = (mean_ysq - mean_y ** 2).clip(lower=1e-8)
                df[col_name] = (cov / var).clip(-3, 3).fillna(1.0)

            df.drop(columns=['_ret_x_mkt', '_mkt_sq'], inplace=True)
            print(f'      rolling beta (pandas 252d+60d): {_t.time()-t0:.1f}s')

        df.drop(columns=['_market_return_daily'], errors='ignore', inplace=True)
        gc.collect()

    def _cleanup_intermediate_cols(self, df: pd.DataFrame) -> None:
        """중간 계산 컬럼 제거"""
        intermediate = [
            'log_return', 'up_day',
            'ma_5', 'ma_20', 'ma_60', 'ma_120',
            'vol_5d', 'vol_20d', 'value_20d',
            'close_location', 'daily_mf',
            'tr', 'atr_20',
            'bb_mid', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width', 'bb_width_avg',
        ]
        df.drop(columns=intermediate, errors='ignore', inplace=True)

    def _load_financial_features_fast(self, start_date: str, end_date: str,
                                       min_market_cap: int,
                                       price_df: pd.DataFrame = None) -> pd.DataFrame:
        """V4.1: 경량화된 재무 피처 로딩 (필요한 컬럼만 직접 쿼리)

        DB Schema:
        - financial_periods: stock_code, fiscal_date, available_date (158K rows)
        - financial_items_bs_cf: period_id (FK), item_code_normalized (13.5M rows)
        - financial_items_pl: period_id (FK), item_code_normalized (3.9M rows)

        Performance: Two-phase load (periods first, then items by period_id chunks)
        avoids slow SQL JOINs on large tables. BS: 107s→21s, PL: 52s→7s.

        Args:
            price_df: Optional pre-loaded price data to avoid redundant DB query
        """
        import sqlite3

        BS_ITEMS = (
            'ifrs-full_Equity', 'ifrs-full_Assets',
            'ifrs-full_Liabilities', 'ifrs-full_CashAndCashEquivalents',
            'ifrs-full_CashFlowsFromUsedInOperatingActivities',
            'ifrs-full_CashFlowsFromUsedInInvestingActivities',
            'ifrs-full_Inventories',                    # V8: 재고자산
            'ifrs-full_CurrentAssets',                  # V8: 유동자산
            'ifrs-full_CurrentLiabilities',             # V8: 유동부채
        )
        PL_ITEMS = (
            'ifrs-full_Revenue', 'dart_OperatingIncomeLoss',
            'ifrs-full_ProfitLoss', 'ifrs-full_GrossProfit',
            'ifrs-full_IncomeTaxExpenseContinuingOperations',
            'ifrs-full_FinanceCosts',
        )
        CHUNK_SIZE = 500  # SQLite param limit safe batch size

        try:
            import time as _t
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA cache_size = -256000")

            # 1. Load matching financial_periods (small table — instant)
            available_start = str(int(start_date[:4]) - 1) + start_date[4:]
            t0 = _t.time()
            fp_df = pd.read_sql_query(
                """SELECT id, stock_code, available_date, industry_name as sector
                   FROM financial_periods
                   WHERE consolidation_type = ? AND available_date >= ?""",
                conn, params=['연결', available_start]
            )
            period_ids = fp_df['id'].tolist()
            print(f'      financial_periods: {_t.time()-t0:.1f}s ({len(fp_df):,} rows)')

            # 2. Load BS/CF items in chunks (avoids slow SQL JOIN on 13.5M row table)
            t0 = _t.time()
            item_ph = ','.join(['?' for _ in BS_ITEMS])
            bs_chunks = []
            for i in range(0, len(period_ids), CHUNK_SIZE):
                chunk = period_ids[i:i + CHUNK_SIZE]
                pid_ph = ','.join(['?' for _ in chunk])
                q = f"""SELECT period_id, item_code_normalized as item_code,
                               amount_current as amount
                        FROM financial_items_bs_cf
                        WHERE period_id IN ({pid_ph})
                          AND item_code_normalized IN ({item_ph})"""
                bs_chunks.append(pd.read_sql_query(q, conn, params=chunk + list(BS_ITEMS)))
            bs_items = pd.concat(bs_chunks, ignore_index=True) if bs_chunks else pd.DataFrame()
            bs_df = fp_df.merge(bs_items, left_on='id', right_on='period_id', how='inner')
            print(f'      BS query: {_t.time()-t0:.1f}s ({len(bs_df):,} rows)')

            # 3. Load PL items in chunks
            t0 = _t.time()
            item_ph = ','.join(['?' for _ in PL_ITEMS])
            pl_chunks = []
            for i in range(0, len(period_ids), CHUNK_SIZE):
                chunk = period_ids[i:i + CHUNK_SIZE]
                pid_ph = ','.join(['?' for _ in chunk])
                q = f"""SELECT period_id, item_code_normalized as item_code,
                               amount_current_ytd as amount
                        FROM financial_items_pl
                        WHERE period_id IN ({pid_ph})
                          AND item_code_normalized IN ({item_ph})"""
                pl_chunks.append(pd.read_sql_query(q, conn, params=chunk + list(PL_ITEMS)))
            pl_items = pd.concat(pl_chunks, ignore_index=True) if pl_chunks else pd.DataFrame()
            pl_df = fp_df.merge(pl_items, left_on='id', right_on='period_id', how='inner')
            conn.close()
            print(f'      PL query: {_t.time()-t0:.1f}s ({len(pl_df):,} rows)')

            # 합치기
            fin_df = pd.concat([bs_df, pl_df], ignore_index=True)

            if len(fin_df) == 0:
                return None

            # Pivot: item_code → columns
            t0 = _t.time()
            fin_pivot = fin_df.pivot_table(
                index=['stock_code', 'available_date', 'sector'],
                columns='item_code',
                values='amount',
                aggfunc='first'
            ).reset_index()
            print(f'      Pivot: {_t.time()-t0:.1f}s')

            # Rename columns
            rename_map = {
                'ifrs-full_Equity': 'equity',
                'ifrs-full_Assets': 'assets',
                'ifrs-full_Revenue': 'revenue',
                'dart_OperatingIncomeLoss': 'operating_income',
                'ifrs-full_ProfitLoss': 'net_income',
                'ifrs-full_GrossProfit': 'gross_profit',
                'ifrs-full_Liabilities': 'liabilities',
                'ifrs-full_CashAndCashEquivalents': 'cash',
                'ifrs-full_CashFlowsFromUsedInOperatingActivities': 'ocf',
                'ifrs-full_CashFlowsFromUsedInInvestingActivities': 'invest_cf',
                'ifrs-full_IncomeTaxExpenseContinuingOperations': 'income_tax',
                'ifrs-full_FinanceCosts': 'finance_costs',
                'ifrs-full_Inventories': 'inventories',
                'ifrs-full_CurrentAssets': 'current_assets',
                'ifrs-full_CurrentLiabilities': 'current_liabilities',
            }
            fin_pivot = fin_pivot.rename(columns=rename_map)
            fin_pivot['available_date'] = pd.to_datetime(fin_pivot['available_date'], format='%Y%m%d')

            # === V6: Quarterly stability metrics (before merge/ffill) ===
            if 'gross_profit' in fin_pivot.columns and 'revenue' in fin_pivot.columns:
                fin_pivot['_gm'] = fin_pivot['gross_profit'] / fin_pivot['revenue'].clip(lower=1)
                gp_grp = fin_pivot.groupby('stock_code')['_gm']
                _mean = gp_grp.rolling(4, min_periods=2).mean().droplevel(0)
                _std = gp_grp.rolling(4, min_periods=2).std().droplevel(0)
                fin_pivot['gross_margin_cv'] = (_std / _mean.clip(lower=0.01)).clip(0, 5)
                fin_pivot.drop(columns=['_gm'], inplace=True)

            if 'operating_income' in fin_pivot.columns and 'revenue' in fin_pivot.columns:
                fin_pivot['_om'] = fin_pivot['operating_income'] / fin_pivot['revenue'].clip(lower=1)
                om_grp = fin_pivot.groupby('stock_code')['_om']
                _mean = om_grp.rolling(4, min_periods=2).mean().droplevel(0)
                _std = om_grp.rolling(4, min_periods=2).std().droplevel(0)
                fin_pivot['oper_margin_cv'] = (_std / _mean.clip(lower=0.01)).clip(0, 5)
                fin_pivot.drop(columns=['_om'], inplace=True)

            # 2. Use pre-loaded price data if available (avoids redundant 800s+ query)
            t0 = _t.time()
            if price_df is not None:
                # Filter and prepare from pre-loaded data
                price_df = price_df[['stock_code', 'date', 'market_cap', 'sector']].copy()
                price_df = price_df.rename(columns={'sector': 'sector_price'})
                price_df = price_df[
                    (price_df['date'] >= start_date) &
                    (price_df['date'] <= end_date) &
                    (price_df['market_cap'] >= min_market_cap)
                ]
                price_df = price_df.sort_values(['stock_code', 'date']).reset_index(drop=True)
                print(f'      Price from cache: {_t.time()-t0:.1f}s ({len(price_df):,} rows)')
            else:
                # Fallback: query from DB (no JOIN — merge stocks in pandas)
                conn = sqlite3.connect(self.db_path)
                conn.execute("PRAGMA cache_size = -256000")
                price_query = """
                SELECT stock_code, date, market_cap
                FROM daily_prices
                WHERE date >= ? AND date <= ? AND market_cap >= ?
                ORDER BY stock_code, date
                """
                price_df = pd.read_sql_query(price_query, conn, params=[start_date, end_date, min_market_cap])
                stocks_df = pd.read_sql_query(
                    "SELECT stock_code, current_sector_type as sector_price FROM stocks", conn)
                conn.close()
                price_df = price_df.merge(stocks_df, on='stock_code', how='left')
                print(f'      Price query: {_t.time()-t0:.1f}s ({len(price_df):,} rows)')

            if len(price_df) == 0:
                return None

            price_df['date'] = pd.to_datetime(price_df['date'], format='%Y%m%d')

            # 3. Fast forward-fill approach (merge_asof 대신)
            # 각 stock의 마지막 재무 데이터를 daily로 forward-fill
            price_df = price_df.sort_values(['stock_code', 'date']).reset_index(drop=True)
            fin_pivot = fin_pivot.sort_values(['stock_code', 'available_date']).reset_index(drop=True)

            # 재무 데이터를 일별로 확장 (available_date를 date 컬럼으로)
            fin_pivot = fin_pivot.rename(columns={'available_date': 'date'})
            fin_cols = ['equity', 'assets', 'revenue', 'operating_income', 'net_income',
                        'gross_profit', 'liabilities', 'cash', 'ocf', 'invest_cf',
                        'income_tax', 'finance_costs',
                        'gross_margin_cv', 'oper_margin_cv',
                        'inventories', 'current_assets', 'current_liabilities']
            fin_cols = [c for c in fin_cols if c in fin_pivot.columns]

            # Merge and forward fill within each stock
            t0 = _t.time()
            merged = price_df.merge(
                fin_pivot[['stock_code', 'date'] + fin_cols],
                on=['stock_code', 'date'],
                how='left'
            )
            merged = merged.sort_values(['stock_code', 'date'])

            # Forward fill financial data within each stock (data leakage fix)
            # Financial data should only be available after the available_date
            for col in fin_cols:
                # Only forward fill if the financial data is available (not NaN)
                merged[col] = merged.groupby('stock_code')[col].ffill()
            print(f'      Merge+ffill: {_t.time()-t0:.1f}s')

            # 4. 비율 계산
            merged['roe'] = merged['net_income'] / merged['equity'].clip(lower=1)
            merged['operating_margin'] = merged['operating_income'] / merged['revenue'].clip(lower=1)

            # Valuation basics
            if 'market_cap' in merged.columns and 'net_income' in merged.columns:
                merged['pe'] = merged['market_cap'] / merged['net_income'].clip(lower=1)
                merged.loc[merged['net_income'] <= 0, 'pe'] = np.nan

            # 5. YoY 계산 (252 거래일 ≈ 1년)
            merged = merged.sort_values(['stock_code', 'date'])
            grouped = merged.groupby('stock_code')
            merged['revenue_prev'] = grouped['revenue'].shift(252)
            merged['revenue_yoy'] = (merged['revenue'] - merged['revenue_prev']) / merged['revenue_prev'].abs().clip(lower=1)

            # V5: GP/A (Gross Profit / Assets)
            if 'gross_profit' in merged.columns and 'assets' in merged.columns:
                merged['gp_over_assets'] = merged['gross_profit'] / merged['assets'].clip(lower=1)

            # V5: ROE Delta YoY (shift 252 = 1 year)
            if 'roe' in merged.columns:
                merged['roe_delta_yoy'] = merged['roe'] - grouped['roe'].shift(252)

            # V5: P/B Sector Z-Score (market_cap / equity, sector z-score)
            if 'market_cap' in merged.columns and 'equity' in merged.columns:
                merged['pb_ratio'] = merged['market_cap'] / merged['equity'].clip(lower=1)

                if 'sector' in merged.columns or 'sector_price' in merged.columns:
                    sector_col = 'sector_price' if 'sector_price' in merged.columns else 'sector'
                    _grp = merged.groupby(['date', sector_col])['pb_ratio']
                else:
                    _grp = merged.groupby('date')['pb_ratio']
                _mean = _grp.transform('mean')
                _std = _grp.transform('std').clip(lower=0.01)
                merged['pb_sector_zscore'] = (merged['pb_ratio'] - _mean) / _std
                merged['pb_sector_zscore'] = merged['pb_sector_zscore'].clip(-3, 3)
                merged.drop(columns=['pb_ratio'], errors='ignore', inplace=True)

            # === V6: Quality Features ===
            # 1. ROIC = NOPAT / Invested Capital
            if all(c in merged.columns for c in ['operating_income', 'income_tax', 'net_income', 'equity', 'liabilities', 'cash']):
                pretax = merged['net_income'] + merged['income_tax'].fillna(0)
                eff_tax = (merged['income_tax'].fillna(0) / pretax.clip(lower=1)).clip(0, 0.5)
                nopat = merged['operating_income'] * (1 - eff_tax)
                invested_capital = (merged['equity'] + merged['liabilities'] - merged['cash'].fillna(0)).clip(lower=1)
                merged['roic'] = (nopat / invested_capital).clip(-1, 1)

            # 4. Net Debt / Equity
            if all(c in merged.columns for c in ['liabilities', 'cash', 'equity']):
                net_debt = merged['liabilities'] - merged['cash'].fillna(0)
                merged['net_debt_to_equity'] = (net_debt / merged['equity'].clip(lower=1)).clip(-5, 10)

            # 5-7. Earnings Quality (OCF-based)
            if 'ocf' in merged.columns:
                if 'net_income' in merged.columns:
                    fcf = merged['ocf'] + merged.get('invest_cf', pd.Series(0, index=merged.index))
                    merged['fcf_to_ni'] = (fcf / merged['net_income'].clip(lower=1)).clip(-5, 5)
                    merged['ocf_to_ni'] = (merged['ocf'] / merged['net_income'].clip(lower=1)).clip(-5, 5)
                    if 'market_cap' in merged.columns:
                        merged['p_fcf'] = merged['market_cap'] / fcf.clip(lower=1)
                        merged.loc[fcf <= 0, 'p_fcf'] = np.nan
                if 'assets' in merged.columns:
                    merged['accruals_ratio'] = ((merged['net_income'] - merged['ocf']) / merged['assets'].clip(lower=1)).clip(-1, 1)

            # 8. Revenue CAGR 3Y (shift 756 trading days ~ 3 years)
            if 'revenue' in merged.columns:
                rev_3y_ago = grouped['revenue'].shift(756)
                ratio = (merged['revenue'] / rev_3y_ago.clip(lower=1))
                merged['revenue_cagr_3y'] = (ratio.clip(lower=0.01) ** (1/3) - 1).clip(-0.5, 2.0)

            # 9. Growth Efficiency = Revenue Growth / (|InvestCF| / Assets)
            if all(c in merged.columns for c in ['revenue_yoy', 'invest_cf', 'assets']):
                capex_intensity = merged['invest_cf'].abs() / merged['assets'].clip(lower=1)
                merged['growth_efficiency'] = (merged['revenue_yoy'] / capex_intensity.clip(lower=0.01)).clip(-10, 10)

            # 10. FCF Yield = FCF / Market Cap
            if 'ocf' in merged.columns and 'market_cap' in merged.columns:
                fcf = merged['ocf'] + merged.get('invest_cf', pd.Series(0, index=merged.index))
                merged['fcf_yield'] = (fcf / merged['market_cap'].clip(lower=1)).clip(-0.5, 0.5)

            # 11. EV/EBIT
            if all(c in merged.columns for c in ['market_cap', 'liabilities', 'cash', 'operating_income']):
                ev = merged['market_cap'] + merged['liabilities'] - merged['cash'].fillna(0)
                merged['ev_to_ebit'] = (ev / merged['operating_income'].clip(lower=1)).clip(-100, 200)

            # 12. Value Score — computed in merge_financial_features() where date-groupby is available
            # 13. PEG (using 3Y revenue CAGR as growth proxy)
            if 'pe' in merged.columns and 'revenue_cagr_3y' in merged.columns:
                growth_pct = (merged['revenue_cagr_3y'] * 100).clip(lower=1)
                merged['peg'] = (merged['pe'] / growth_pct).clip(0, 50)

            # === V8: Manufacturing red flag features ===
            if 'inventories' in merged.columns and 'revenue' in merged.columns:
                inv_prev = grouped['inventories'].shift(252)
                rev_prev = grouped['revenue'].shift(252)
                inv_growth = (merged['inventories'] - inv_prev) / inv_prev.abs().clip(lower=1)
                rev_growth = (merged['revenue'] - rev_prev) / rev_prev.abs().clip(lower=1)
                merged['inventory_sales_gap'] = (inv_growth - rev_growth).clip(-2, 2)

            if 'current_assets' in merged.columns and 'current_liabilities' in merged.columns:
                merged['current_ratio'] = (
                    merged['current_assets'] / merged['current_liabilities'].clip(lower=1)
                ).clip(0, 10)

            # inf 처리
            for col in ['roe', 'operating_margin', 'revenue_yoy', 'pe', 'p_fcf', 'peg',
                         'gp_over_assets', 'roe_delta_yoy', 'pb_sector_zscore',
                         'roic', 'net_debt_to_equity', 'fcf_to_ni', 'ocf_to_ni',
                         'accruals_ratio', 'revenue_cagr_3y', 'growth_efficiency',
                         'fcf_yield', 'ev_to_ebit',
                         'inventory_sales_gap', 'current_ratio']:
                if col in merged.columns:
                    merged[col] = merged[col].replace([np.inf, -np.inf], np.nan)

            # sector 정리
            merged['sector'] = merged.get('sector_price', merged.get('sector'))
            merged = merged.drop(columns=['sector_price', 'available_date'], errors='ignore')

            # date를 문자열 포맷으로 변환 (merge 호환성)
            merged['date'] = merged['date'].dt.strftime('%Y%m%d')

            self.logger.info(f"Fast loaded {len(merged):,} financial records")
            return merged

        except Exception as e:
            self.logger.warning(f"Fast financial loading failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_financial_pivot(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load and pivot financial data ONCE (SQL-heavy part).

        Extracts the DB query + pivot from _load_financial_features_fast (lines 822-924).
        Returns pivoted DataFrame with columns: stock_code, available_date, sector,
        equity, assets, revenue, ..., gross_margin_cv, oper_margin_cv.

        This is meant to be called once in the main process; workers inherit
        the result via fork copy-on-write and only do the cheap merge step.
        """
        import time as _t

        BS_ITEMS = (
            'ifrs-full_Equity', 'ifrs-full_Assets',
            'ifrs-full_Liabilities', 'ifrs-full_CashAndCashEquivalents',
            'ifrs-full_CashFlowsFromUsedInOperatingActivities',
            'ifrs-full_CashFlowsFromUsedInInvestingActivities',
            'ifrs-full_Inventories',                    # V8: 재고자산
            'ifrs-full_CurrentAssets',                  # V8: 유동자산
            'ifrs-full_CurrentLiabilities',             # V8: 유동부채
        )
        PL_ITEMS = (
            'ifrs-full_Revenue', 'dart_OperatingIncomeLoss',
            'ifrs-full_ProfitLoss', 'ifrs-full_GrossProfit',
            'ifrs-full_IncomeTaxExpenseContinuingOperations',
            'ifrs-full_FinanceCosts',
        )
        CHUNK_SIZE = 500

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA cache_size = -256000")

        # 1. Load matching financial_periods
        available_start = str(int(start_date[:4]) - 1) + start_date[4:]
        t0 = _t.time()
        fp_df = pd.read_sql_query(
            """SELECT id, stock_code, available_date, industry_name as sector
               FROM financial_periods
               WHERE consolidation_type = ? AND available_date >= ?""",
            conn, params=['연결', available_start]
        )
        period_ids = fp_df['id'].tolist()
        print(f'      financial_periods: {_t.time()-t0:.1f}s ({len(fp_df):,} rows)')

        # 2. Load BS/CF items in chunks
        t0 = _t.time()
        item_ph = ','.join(['?' for _ in BS_ITEMS])
        bs_chunks = []
        for i in range(0, len(period_ids), CHUNK_SIZE):
            chunk = period_ids[i:i + CHUNK_SIZE]
            pid_ph = ','.join(['?' for _ in chunk])
            q = f"""SELECT period_id, item_code_normalized as item_code,
                           amount_current as amount
                    FROM financial_items_bs_cf
                    WHERE period_id IN ({pid_ph})
                      AND item_code_normalized IN ({item_ph})"""
            bs_chunks.append(pd.read_sql_query(q, conn, params=chunk + list(BS_ITEMS)))
        bs_items = pd.concat(bs_chunks, ignore_index=True) if bs_chunks else pd.DataFrame()
        bs_df = fp_df.merge(bs_items, left_on='id', right_on='period_id', how='inner')
        print(f'      BS query: {_t.time()-t0:.1f}s ({len(bs_df):,} rows)')

        # 3. Load PL items in chunks
        t0 = _t.time()
        item_ph = ','.join(['?' for _ in PL_ITEMS])
        pl_chunks = []
        for i in range(0, len(period_ids), CHUNK_SIZE):
            chunk = period_ids[i:i + CHUNK_SIZE]
            pid_ph = ','.join(['?' for _ in chunk])
            q = f"""SELECT period_id, item_code_normalized as item_code,
                           amount_current_ytd as amount
                    FROM financial_items_pl
                    WHERE period_id IN ({pid_ph})
                      AND item_code_normalized IN ({item_ph})"""
            pl_chunks.append(pd.read_sql_query(q, conn, params=chunk + list(PL_ITEMS)))
        pl_items = pd.concat(pl_chunks, ignore_index=True) if pl_chunks else pd.DataFrame()
        pl_df = fp_df.merge(pl_items, left_on='id', right_on='period_id', how='inner')
        conn.close()
        print(f'      PL query: {_t.time()-t0:.1f}s ({len(pl_df):,} rows)')

        # Combine and pivot
        fin_df = pd.concat([bs_df, pl_df], ignore_index=True)
        if len(fin_df) == 0:
            return pd.DataFrame()

        t0 = _t.time()
        fin_pivot = fin_df.pivot_table(
            index=['stock_code', 'available_date', 'sector'],
            columns='item_code',
            values='amount',
            aggfunc='first'
        ).reset_index()
        print(f'      Pivot: {_t.time()-t0:.1f}s')

        # Rename columns
        rename_map = {
            'ifrs-full_Equity': 'equity',
            'ifrs-full_Assets': 'assets',
            'ifrs-full_Revenue': 'revenue',
            'dart_OperatingIncomeLoss': 'operating_income',
            'ifrs-full_ProfitLoss': 'net_income',
            'ifrs-full_GrossProfit': 'gross_profit',
            'ifrs-full_Liabilities': 'liabilities',
            'ifrs-full_CashAndCashEquivalents': 'cash',
            'ifrs-full_CashFlowsFromUsedInOperatingActivities': 'ocf',
            'ifrs-full_CashFlowsFromUsedInInvestingActivities': 'invest_cf',
            'ifrs-full_IncomeTaxExpenseContinuingOperations': 'income_tax',
            'ifrs-full_FinanceCosts': 'finance_costs',
            'ifrs-full_Inventories': 'inventories',
            'ifrs-full_CurrentAssets': 'current_assets',
            'ifrs-full_CurrentLiabilities': 'current_liabilities',
        }
        fin_pivot = fin_pivot.rename(columns=rename_map)
        fin_pivot['available_date'] = pd.to_datetime(fin_pivot['available_date'], format='%Y%m%d')

        # Quarterly stability metrics
        if 'gross_profit' in fin_pivot.columns and 'revenue' in fin_pivot.columns:
            fin_pivot['_gm'] = fin_pivot['gross_profit'] / fin_pivot['revenue'].clip(lower=1)
            gp_grp = fin_pivot.groupby('stock_code')['_gm']
            _mean = gp_grp.rolling(4, min_periods=2).mean().droplevel(0)
            _std = gp_grp.rolling(4, min_periods=2).std().droplevel(0)
            fin_pivot['gross_margin_cv'] = (_std / _mean.clip(lower=0.01)).clip(0, 5)
            fin_pivot.drop(columns=['_gm'], inplace=True)

        if 'operating_income' in fin_pivot.columns and 'revenue' in fin_pivot.columns:
            fin_pivot['_om'] = fin_pivot['operating_income'] / fin_pivot['revenue'].clip(lower=1)
            om_grp = fin_pivot.groupby('stock_code')['_om']
            _mean = om_grp.rolling(4, min_periods=2).mean().droplevel(0)
            _std = om_grp.rolling(4, min_periods=2).std().droplevel(0)
            fin_pivot['oper_margin_cv'] = (_std / _mean.clip(lower=0.01)).clip(0, 5)
            fin_pivot.drop(columns=['_om'], inplace=True)

        fin_pivot = fin_pivot.sort_values(['stock_code', 'available_date']).reset_index(drop=True)
        return fin_pivot

    def _merge_financial_with_prices(self, fin_pivot: pd.DataFrame,
                                      price_df: pd.DataFrame,
                                      start_date: str, end_date: str,
                                      min_market_cap: int) -> pd.DataFrame:
        """Cheap merge of pre-loaded financial pivot with price data (~1-2s).

        Extracts the price-merge + ffill + ratio computation part from
        _load_financial_features_fast (lines 925-1097). Workers call this
        instead of the full SQL-heavy method.
        """
        import time as _t

        # Prepare price data
        t0 = _t.time()
        price_df = price_df[['stock_code', 'date', 'market_cap', 'sector']].copy()
        price_df = price_df.rename(columns={'sector': 'sector_price'})
        price_df = price_df[
            (price_df['date'] >= start_date) &
            (price_df['date'] <= end_date) &
            (price_df['market_cap'] >= min_market_cap)
        ]
        price_df = price_df.sort_values(['stock_code', 'date']).reset_index(drop=True)

        if len(price_df) == 0:
            return None

        price_df['date'] = pd.to_datetime(price_df['date'], format='%Y%m%d')

        # Work on a copy of fin_pivot to avoid mutating the shared data
        fp = fin_pivot.copy()
        fp = fp.rename(columns={'available_date': 'date'})
        fin_cols = ['equity', 'assets', 'revenue', 'operating_income', 'net_income',
                    'gross_profit', 'liabilities', 'cash', 'ocf', 'invest_cf',
                    'income_tax', 'finance_costs',
                    'gross_margin_cv', 'oper_margin_cv',
                    'inventories', 'current_assets', 'current_liabilities']
        fin_cols = [c for c in fin_cols if c in fp.columns]

        # Merge and forward fill
        t0 = _t.time()
        merged = price_df.merge(
            fp[['stock_code', 'date'] + fin_cols],
            on=['stock_code', 'date'],
            how='left'
        )
        merged = merged.sort_values(['stock_code', 'date'])

        for col in fin_cols:
            merged[col] = merged.groupby('stock_code')[col].ffill()
        print(f'      Merge+ffill: {_t.time()-t0:.1f}s')

        # Ratio computation (same as _load_financial_features_fast)
        merged['roe'] = merged['net_income'] / merged['equity'].clip(lower=1)
        merged['operating_margin'] = merged['operating_income'] / merged['revenue'].clip(lower=1)

        if 'market_cap' in merged.columns and 'net_income' in merged.columns:
            merged['pe'] = merged['market_cap'] / merged['net_income'].clip(lower=1)
            merged.loc[merged['net_income'] <= 0, 'pe'] = np.nan

        merged = merged.sort_values(['stock_code', 'date'])
        grouped = merged.groupby('stock_code')
        merged['revenue_prev'] = grouped['revenue'].shift(252)
        merged['revenue_yoy'] = (merged['revenue'] - merged['revenue_prev']) / merged['revenue_prev'].abs().clip(lower=1)

        if 'gross_profit' in merged.columns and 'assets' in merged.columns:
            merged['gp_over_assets'] = merged['gross_profit'] / merged['assets'].clip(lower=1)

        if 'roe' in merged.columns:
            merged['roe_delta_yoy'] = merged['roe'] - grouped['roe'].shift(252)

        if 'market_cap' in merged.columns and 'equity' in merged.columns:
            merged['pb_ratio'] = merged['market_cap'] / merged['equity'].clip(lower=1)
            if 'sector' in merged.columns or 'sector_price' in merged.columns:
                sector_col = 'sector_price' if 'sector_price' in merged.columns else 'sector'
                _grp = merged.groupby(['date', sector_col])['pb_ratio']
            else:
                _grp = merged.groupby('date')['pb_ratio']
            _mean = _grp.transform('mean')
            _std = _grp.transform('std').clip(lower=0.01)
            merged['pb_sector_zscore'] = (merged['pb_ratio'] - _mean) / _std
            merged['pb_sector_zscore'] = merged['pb_sector_zscore'].clip(-3, 3)
            merged.drop(columns=['pb_ratio'], errors='ignore', inplace=True)

        # V6: Quality Features
        if all(c in merged.columns for c in ['operating_income', 'income_tax', 'net_income', 'equity', 'liabilities', 'cash']):
            pretax = merged['net_income'] + merged['income_tax'].fillna(0)
            eff_tax = (merged['income_tax'].fillna(0) / pretax.clip(lower=1)).clip(0, 0.5)
            nopat = merged['operating_income'] * (1 - eff_tax)
            invested_capital = (merged['equity'] + merged['liabilities'] - merged['cash'].fillna(0)).clip(lower=1)
            merged['roic'] = (nopat / invested_capital).clip(-1, 1)

        if all(c in merged.columns for c in ['liabilities', 'cash', 'equity']):
            net_debt = merged['liabilities'] - merged['cash'].fillna(0)
            merged['net_debt_to_equity'] = (net_debt / merged['equity'].clip(lower=1)).clip(-5, 10)

        if 'ocf' in merged.columns:
            if 'net_income' in merged.columns:
                fcf = merged['ocf'] + merged.get('invest_cf', pd.Series(0, index=merged.index))
                merged['fcf_to_ni'] = (fcf / merged['net_income'].clip(lower=1)).clip(-5, 5)
                merged['ocf_to_ni'] = (merged['ocf'] / merged['net_income'].clip(lower=1)).clip(-5, 5)
                if 'market_cap' in merged.columns:
                    merged['p_fcf'] = merged['market_cap'] / fcf.clip(lower=1)
                    merged.loc[fcf <= 0, 'p_fcf'] = np.nan
            if 'assets' in merged.columns:
                merged['accruals_ratio'] = ((merged['net_income'] - merged['ocf']) / merged['assets'].clip(lower=1)).clip(-1, 1)

        if 'revenue' in merged.columns:
            rev_3y_ago = grouped['revenue'].shift(756)
            ratio = (merged['revenue'] / rev_3y_ago.clip(lower=1))
            merged['revenue_cagr_3y'] = (ratio.clip(lower=0.01) ** (1/3) - 1).clip(-0.5, 2.0)

        if all(c in merged.columns for c in ['revenue_yoy', 'invest_cf', 'assets']):
            capex_intensity = merged['invest_cf'].abs() / merged['assets'].clip(lower=1)
            merged['growth_efficiency'] = (merged['revenue_yoy'] / capex_intensity.clip(lower=0.01)).clip(-10, 10)

        if 'ocf' in merged.columns and 'market_cap' in merged.columns:
            fcf = merged['ocf'] + merged.get('invest_cf', pd.Series(0, index=merged.index))
            merged['fcf_yield'] = (fcf / merged['market_cap'].clip(lower=1)).clip(-0.5, 0.5)

        if all(c in merged.columns for c in ['market_cap', 'liabilities', 'cash', 'operating_income']):
            ev = merged['market_cap'] + merged['liabilities'] - merged['cash'].fillna(0)
            merged['ev_to_ebit'] = (ev / merged['operating_income'].clip(lower=1)).clip(-100, 200)

        if 'pe' in merged.columns and 'revenue_cagr_3y' in merged.columns:
            growth_pct = (merged['revenue_cagr_3y'] * 100).clip(lower=1)
            merged['peg'] = (merged['pe'] / growth_pct).clip(0, 50)

        # === V8: Manufacturing red flag features ===
        if 'inventories' in merged.columns and 'revenue' in merged.columns:
            inv_prev = grouped['inventories'].shift(252)
            rev_prev = grouped['revenue'].shift(252)
            inv_growth = (merged['inventories'] - inv_prev) / inv_prev.abs().clip(lower=1)
            rev_growth = (merged['revenue'] - rev_prev) / rev_prev.abs().clip(lower=1)
            merged['inventory_sales_gap'] = (inv_growth - rev_growth).clip(-2, 2)

        if 'current_assets' in merged.columns and 'current_liabilities' in merged.columns:
            merged['current_ratio'] = (
                merged['current_assets'] / merged['current_liabilities'].clip(lower=1)
            ).clip(0, 10)

        # inf cleanup
        for col in ['roe', 'operating_margin', 'revenue_yoy', 'pe', 'p_fcf', 'peg',
                     'gp_over_assets', 'roe_delta_yoy', 'pb_sector_zscore',
                     'roic', 'net_debt_to_equity', 'fcf_to_ni', 'ocf_to_ni',
                     'accruals_ratio', 'revenue_cagr_3y', 'growth_efficiency',
                     'fcf_yield', 'ev_to_ebit',
                     'inventory_sales_gap', 'current_ratio']:
            if col in merged.columns:
                merged[col] = merged[col].replace([np.inf, -np.inf], np.nan)

        # sector cleanup
        merged['sector'] = merged.get('sector_price', merged.get('sector'))
        merged = merged.drop(columns=['sector_price', 'available_date'], errors='ignore')

        # date format
        merged['date'] = merged['date'].dt.strftime('%Y%m%d')

        return merged

    def load_financial_features(self, start_date: str, end_date: str,
                                min_market_cap: int = 500000000000,
                                price_df: pd.DataFrame = None) -> pd.DataFrame:
        """Load financial features with Delta calculations - V4.1 경량화 버전.

        Args:
            price_df: Optional pre-loaded price data to avoid redundant DB query
        """
        self.logger.info("Loading financial features (lightweight V4.1)...")

        # V4.1: 직접 DB에서 필요한 컬럼만 로드 (143초 → 5초)
        # Pass price_df to avoid redundant 800s+ query
        fin_df = self._load_financial_features_fast(start_date, end_date, min_market_cap, price_df=price_df)
        if fin_df is not None and len(fin_df) > 0:
            return fin_df

        # Fallback: 기존 방식
        self.logger.info("Falling back to full FinancialFeatureGenerator...")
        with FinancialFeatureGenerator(self.db_path) as fin_gen:
            fin_df = fin_gen.generate_features(
                start_date=start_date,
                end_date=end_date,
                min_market_cap=min_market_cap,
                include_ranks=True,
                include_missing_indicators=True
            )

        # Add Delta features (QoQ changes)
        fin_df = fin_df.sort_values(['stock_code', 'date'])
        grouped = fin_df.groupby('stock_code')

        # ROE Delta (분기 변화)
        if 'roe' in fin_df.columns:
            fin_df['roe_prev'] = grouped['roe'].shift(63)  # ~1분기 전
            fin_df['roe_delta_qoq'] = fin_df['roe'] - fin_df['roe_prev']

            # V3 신규: ROE 섹터 대비 Z-score (vectorized)
            if 'sector' in fin_df.columns:
                _grp = fin_df.groupby(['date', 'sector'])['roe']
            else:
                _grp = fin_df.groupby('date')['roe']
            _mean = _grp.transform('mean')
            _std = _grp.transform('std').clip(lower=0.01)
            fin_df['roe_sector_zscore'] = ((fin_df['roe'] - _mean) / _std).clip(-3, 3)

        # Revenue Growth Acceleration
        if 'revenue_yoy' in fin_df.columns:
            fin_df['revenue_yoy_prev'] = grouped['revenue_yoy'].shift(63)
            fin_df['revenue_growth_accel'] = fin_df['revenue_yoy'] - fin_df['revenue_yoy_prev']

        # Margin Improvement
        if 'operating_margin' in fin_df.columns:
            fin_df['margin_prev'] = grouped['operating_margin'].shift(63)
            fin_df['margin_improvement'] = fin_df['operating_margin'] - fin_df['margin_prev']

        # Convert date
        fin_df['date'] = fin_df['date'].dt.strftime('%Y%m%d')

        self.logger.info(f"Loaded {len(fin_df):,} financial records with deltas")
        return fin_df

    def merge_financial_features(self, tech_df: pd.DataFrame,
                                 fin_df: pd.DataFrame) -> pd.DataFrame:
        """Merge technical and financial features."""
        # V4.1: 재무 피처 4개를 단일 financial_quality_index로 압축
        # V5: 추가 재무 피처 포함
        # V6: Quality features pass-through
        _v5_fin_cols = ['gp_over_assets', 'roe_delta_yoy', 'pb_sector_zscore']
        _v6_quality_cols = [
            'roic', 'gross_margin_cv', 'oper_margin_cv',
            'net_debt_to_equity', 'fcf_to_ni', 'ocf_to_ni', 'accruals_ratio',
            'revenue_cagr_3y', 'growth_efficiency',
            'fcf_yield', 'ev_to_ebit', 'p_fcf', 'peg',
        ]
        _extra_fin_cols = _v5_fin_cols + _v6_quality_cols
        raw_cols = ['stock_code', 'date'] + [
            col for col in self._FUNDAMENTAL_RAW + _extra_fin_cols
            if col in fin_df.columns
        ]
        fin_subset = fin_df[raw_cols].drop_duplicates(subset=['stock_code', 'date'])

        merged = pd.merge(
            tech_df,
            fin_subset,
            on=['stock_code', 'date'],
            how='left'
        )

        # Fill missing raw features
        for col in self._FUNDAMENTAL_RAW + _extra_fin_cols:
            if col in merged.columns:
                merged[col] = merged.groupby('date')[col].transform(
                    lambda x: x.fillna(x.median())
                )

        # === V4.1: Financial Quality Index 계산 ===
        # 각 재무 피처를 0~1 랭크로 변환 후 평균
        rank_cols = []
        for col in self._FUNDAMENTAL_RAW:
            if col in merged.columns:
                rank_col = f'{col}_rank'
                merged[rank_col] = merged.groupby('date')[col].rank(pct=True)
                rank_cols.append(rank_col)

        if rank_cols:
            merged['financial_quality_index'] = merged[rank_cols].mean(axis=1)
            # 임시 랭크 컬럼 제거
            merged.drop(columns=rank_cols, inplace=True)
        else:
            merged['financial_quality_index'] = 0.5  # fallback

        # === V6: Value Score (composite rank — needs date-groupby) ===
        if 'fcf_yield' in merged.columns and 'ev_to_ebit' in merged.columns:
            r1 = merged.groupby('date')['fcf_yield'].rank(pct=True)
            r2 = merged.groupby('date')['ev_to_ebit'].rank(pct=True, ascending=False)
            merged['value_score'] = (r1 + r2) / 2

        # === V7: 5-Pillar Composite Scores ===
        def _rank_pct(series: pd.Series, dates: pd.Series, higher_is_better: bool = True) -> pd.Series:
            s = series
            if not higher_is_better:
                s = -s
            return s.groupby(dates).rank(pct=True, ascending=False)

        date_key = merged['date']

        # 1) Moat: ROIC ↑, margin volatility ↓
        moat_components = []
        if 'roic' in merged.columns:
            moat_components.append(_rank_pct(merged['roic'], date_key, True))
        if 'gross_margin_cv' in merged.columns:
            moat_components.append(_rank_pct(merged['gross_margin_cv'], date_key, False))
        if 'oper_margin_cv' in merged.columns:
            moat_components.append(_rank_pct(merged['oper_margin_cv'], date_key, False))
        if moat_components:
            merged['moat_score'] = sum(moat_components) / len(moat_components)

        # 2) Capital Efficiency: ROIC/ROE ↑, leverage ↓
        cap_components = []
        if 'roic' in merged.columns:
            cap_components.append(_rank_pct(merged['roic'], date_key, True))
        if 'roe' in merged.columns:
            cap_components.append(_rank_pct(merged['roe'], date_key, True))
        if 'net_debt_to_equity' in merged.columns:
            cap_components.append(_rank_pct(merged['net_debt_to_equity'], date_key, False))
        if cap_components:
            merged['capital_efficiency_score'] = sum(cap_components) / len(cap_components)

        # 3) Earnings Quality: FCF/NI ↑, OCF/NI ↑, accruals ↓
        eq_components = []
        if 'fcf_to_ni' in merged.columns:
            eq_components.append(_rank_pct(merged['fcf_to_ni'], date_key, True))
        if 'ocf_to_ni' in merged.columns:
            eq_components.append(_rank_pct(merged['ocf_to_ni'], date_key, True))
        if 'accruals_ratio' in merged.columns:
            eq_components.append(_rank_pct(merged['accruals_ratio'], date_key, False))
        if eq_components:
            merged['earnings_quality_score'] = sum(eq_components) / len(eq_components)

        # 4) Reinvestment: 3Y revenue CAGR ↑, growth efficiency ↑
        reinvest_components = []
        if 'revenue_cagr_3y' in merged.columns:
            reinvest_components.append(_rank_pct(merged['revenue_cagr_3y'], date_key, True))
        if 'growth_efficiency' in merged.columns:
            reinvest_components.append(_rank_pct(merged['growth_efficiency'], date_key, True))
        if reinvest_components:
            merged['reinvestment_score'] = sum(reinvest_components) / len(reinvest_components)

        # 5) Margin of Safety: FCF yield ↑, EV/EBIT ↓, P/FCF ↓, PEG ↓
        mos_components = []
        if 'fcf_yield' in merged.columns:
            mos_components.append(_rank_pct(merged['fcf_yield'], date_key, True))
        if 'ev_to_ebit' in merged.columns:
            mos_components.append(_rank_pct(merged['ev_to_ebit'], date_key, False))
        if 'p_fcf' in merged.columns:
            mos_components.append(_rank_pct(merged['p_fcf'], date_key, False))
        if 'peg' in merged.columns:
            mos_components.append(_rank_pct(merged['peg'], date_key, False))
        if mos_components:
            merged['margin_of_safety_score'] = sum(mos_components) / len(mos_components)

        return merged

    def add_forward_returns(self, df: pd.DataFrame,
                            horizons: List[int] = None) -> pd.DataFrame:
        """
        Add forward returns using Open-to-Open pricing (정석).

        Signal: T일 종가 데이터까지 보고 생성
        Buy:  T+1일 시가 (Open)  ← T일 종가로는 매수 불가
        Sell: T+1+h일 시가 (Open)

        V4.3: Residual Return 추가 (Beta-adjusted)
        - De Prado (2018): "Alpha를 찾으려면 Beta를 제거하라"
        """
        horizons = horizons or [21]

        df = df.sort_values(['stock_code', 'date']).copy()
        grouped = df.groupby('stock_code')

        # ================================================================
        # Rolling Beta — skip if already computed in compute_features()
        # ================================================================
        import time as _t
        if 'rolling_beta' not in df.columns or 'rolling_beta_60d' not in df.columns:
            t0 = _t.time()
            self._compute_rolling_beta(df, grouped)
            print(f'      rolling beta (computed in add_forward_returns): {_t.time()-t0:.1f}s')
        else:
            print(f'      rolling beta: skipped (already computed)')

        # ================================================================
        # Horizon별 forward return + alpha + residual + ranks
        # ================================================================
        t0 = _t.time()
        for h in horizons:
            col_name = f'forward_return_{h}d'

            # Open-to-Open: T+1 시가 → T+1+h 시가 (look-ahead bias 방지)
            open_t1 = grouped['opening_price'].shift(-1)
            open_t1_h = grouped['opening_price'].shift(-1 - h)
            df[col_name] = (open_t1_h - open_t1) / open_t1
            df[col_name] = df[col_name].clip(-0.50, 0.50)

            # Alpha (시장 대비 초과수익률)
            forward_market = df.groupby('date')[col_name].transform('median')
            alpha_col = f'forward_alpha_{h}d'
            df[alpha_col] = df[col_name] - forward_market

            # Residual = Actual - (Beta × Market_Return)
            residual_col = f'forward_residual_{h}d'
            df[residual_col] = df[col_name] - (df['rolling_beta'] * forward_market)

            # Target Ranks - Batched computation (single groupby for all ranks)
            # This avoids 4 separate groupby operations
            rank_cols = [col_name, alpha_col, residual_col]
            rank_names = [f'target_rank_{h}d', f'target_alpha_rank_{h}d', f'target_residual_rank_{h}d']

            # Compute all date-based ranks in one pass
            date_ranks = df.groupby('date')[rank_cols].rank(pct=True)
            for i, name in enumerate(rank_names):
                df[name] = date_ranks.iloc[:, i].values

            # Sector rank (separate groupby required due to different grouping)
            df[f'target_sector_rank_{h}d'] = df.groupby(['date', 'sector'])[col_name].rank(pct=True)

            # V8 Hybrid Target: 70% residual rank + 30% absolute return rank
            df[f'target_hybrid_rank_{h}d'] = (
                0.7 * df[f'target_residual_rank_{h}d'] +
                0.3 * df[f'target_rank_{h}d']
            )

            # V9: Signed log return (robust continuous target)
            df[f'target_log_return_{h}d'] = np.sign(df[col_name]) * np.log1p(df[col_name].abs())

            # V10: Cross-sectional Z-score of forward returns
            # Model learns "who ranks 1st today?" not "will the market go up?"
            _grp = df.groupby('date')[col_name]
            _cs_mean = _grp.transform('mean')
            _cs_std = _grp.transform('std').clip(lower=0.001)
            df[f'target_zscore_{h}d'] = (df[col_name] - _cs_mean) / _cs_std
            df[f'target_zscore_{h}d'] = df[f'target_zscore_{h}d'].clip(-3, 3)

            # V10: Sector-neutralized Z-score (stock return - sector mean, then Z-score)
            if 'sector' in df.columns:
                _sec_mean = df.groupby(['date', 'sector'])[col_name].transform('mean')
                df['_sec_resid_tmp'] = df[col_name] - _sec_mean
                _grp2 = df.groupby('date')['_sec_resid_tmp']
                _cs_mean2 = _grp2.transform('mean')
                _cs_std2 = _grp2.transform('std').clip(lower=0.001)
                df[f'target_sector_zscore_{h}d'] = (df['_sec_resid_tmp'] - _cs_mean2) / _cs_std2
                df[f'target_sector_zscore_{h}d'] = df[f'target_sector_zscore_{h}d'].clip(-3, 3)
                df.drop(columns=['_sec_resid_tmp'], inplace=True)

        print(f'      forward returns + ranks: {_t.time()-t0:.1f}s')

        # 임시 컬럼 정리
        df.drop(columns=['_market_return_daily'], errors='ignore', inplace=True)

        return df

    def filter_universe(self, df: pd.DataFrame,
                        min_price: int = 1000,
                        min_market_cap: int = 500000000000,
                        min_value: int = 10000000000) -> pd.DataFrame:
        """Filter stock universe."""
        original_len = len(df)

        df = df[df['closing_price'] >= min_price]
        df = df[df['market_cap'] >= min_market_cap]
        df = df[df['value'] >= min_value]

        # Drop rows with NaN in key features
        key_features = ['mom_20d', 'volume_surprise', 'volatility_20d']
        key_features = [f for f in key_features if f in df.columns]
        df = df.dropna(subset=key_features)

        self.logger.info(f"Filtered: {original_len:,} -> {len(df):,} rows")
        return df

    def prepare_ml_data(self, start_date: str, end_date: str,
                        target_horizon: int = 21,  # V2: 기본 21일
                        min_market_cap: int = 500000000000,
                        include_fundamental: bool = True,
                        include_macro: bool = False,
                        use_cache: bool = True,
                        n_workers: int = None) -> pd.DataFrame:
        """
        Full pipeline for ML data preparation — year-by-year multiprocessing.

        Each year is processed independently in parallel workers:
        load → features → financial → forward returns → trim to year.
        Then concat + filter + macro.

        Args:
            include_macro: V4 매크로 피처 포함 여부 (Regime Detection)
            use_cache: 캐시 사용 여부 (기본 True - 속도 향상)
            n_workers: Number of parallel workers (default: min(8, cpu_count))
        """
        import hashlib
        import os
        import time

        if n_workers is None:
            n_workers = min(4, multiprocessing.cpu_count())

        # 캐시 파일명 생성
        cache_key = f"{start_date}_{end_date}_{target_horizon}_{min_market_cap}_{include_fundamental}_{include_macro}_{self.CACHE_VERSION}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_file = f".cache/features_{cache_hash}.parquet"

        # 캐시 디렉토리 생성
        os.makedirs(".cache", exist_ok=True)

        # 캐시 확인 (warm path — ~2s)
        if use_cache and os.path.exists(cache_file):
            cache_mtime = os.path.getmtime(cache_file)
            db_mtime = os.path.getmtime(self.db_path)
            if cache_mtime > db_mtime:
                self.logger.info(f"Loading cached features from {cache_file}")
                return pd.read_parquet(cache_file)

        pipeline_t0 = time.time()

        # Step 1: Get eligible stocks via latest-date market_cap filter (<1s)
        t0 = time.time()
        self.logger.info("[1] Getting eligible stocks...")
        conn = sqlite3.connect(self.db_path)
        mcap_buffer = int(min_market_cap * 0.5)
        eligible = pd.read_sql_query(
            """SELECT DISTINCT stock_code FROM daily_prices
               WHERE date IN (
                   SELECT DISTINCT date FROM daily_prices ORDER BY date DESC LIMIT 5
               ) AND market_cap >= ?
                 AND market_type IN ('kospi','kosdaq')""",
            conn, params=[mcap_buffer]
        )
        conn.close()
        eligible_codes = eligible['stock_code'].tolist()
        print(f'  [1] Eligible stocks: {len(eligible_codes)} ({time.time()-t0:.1f}s)')

        # Step 1.5: Pre-load financial pivot once (workers inherit via fork COW)
        global _SHARED_FIN_PIVOT
        if include_fundamental:
            t0 = time.time()
            self.logger.info("[1.5] Pre-loading financial pivot (one-time)...")
            print(f'  [1.5] Loading financial pivot (one-time)...')
            self._ensure_indexes()
            _SHARED_FIN_PIVOT = self._load_financial_pivot(start_date, end_date)
            print(f'  [1.5] Financial pivot: {time.time()-t0:.1f}s ({len(_SHARED_FIN_PIVOT):,} rows)')
        else:
            _SHARED_FIN_PIVOT = None

        # Step 2: Build year range and dispatch to multiprocessing pool
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        years = list(range(start_year, end_year + 1))

        args_list = [
            (year, eligible_codes, self.db_path, target_horizon, min_market_cap, include_fundamental)
            for year in years
        ]

        t0 = time.time()
        if len(years) <= 2:
            # Small range: process sequentially (avoid multiprocessing overhead)
            self.logger.info(f"[2] Processing {len(years)} year(s) sequentially...")
            print(f'  [2] Processing {len(years)} year-chunk(s) sequentially...')
            results = [_process_year_chunk(a) for a in args_list]
        else:
            effective_workers = min(n_workers, len(years))
            self.logger.info(f"[2] Processing {len(years)} years with {effective_workers} workers...")
            print(f'  [2] Dispatching {len(years)} year-chunks to {effective_workers} workers...')
            # Use 'fork' context to avoid re-importing __main__ on macOS (default is 'spawn')
            # maxtasksperchild=1: recycle workers after each year to free memory
            ctx = multiprocessing.get_context('fork')
            with ctx.Pool(effective_workers, maxtasksperchild=1) as pool:
                results = list(pool.imap_unordered(_process_year_chunk, args_list))

        # Filter out empty results
        results = [r for r in results if len(r) > 0]
        print(f'  [2] Parallel processing done: {time.time()-t0:.1f}s ({len(results)} chunks)')

        # Free shared financial pivot (no longer needed after workers finish)
        _SHARED_FIN_PIVOT = None

        # Step 3: Concat all year results
        t0 = time.time()
        df = pd.concat(results, ignore_index=True)
        del results
        gc.collect()
        print(f'  [3] Concat: {time.time()-t0:.1f}s ({len(df):,} rows)')

        # Step 4: Filter universe
        t0 = time.time()
        df = self.filter_universe(df, min_market_cap=min_market_cap)

        # Filter to requested date range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        print(f'  [4] Filter universe: {time.time()-t0:.1f}s ({len(df):,} rows)')

        # Step 5: V4 - Add macro features (Regime Detection)
        if include_macro:
            t0 = time.time()
            self.logger.info("[5] Adding macro features...")
            df = self._add_macro_features(df, start_date, end_date)
            print(f'  [5] Macro features: {time.time()-t0:.1f}s')

        # Final cache (all stages complete)
        if use_cache:
            t0 = time.time()
            df.to_parquet(cache_file, index=False)
            print(f'  Cache save: {time.time()-t0:.1f}s')
            self.logger.info(f"Cached features to {cache_file}")

        print(f'  Total pipeline: {time.time()-pipeline_t0:.1f}s')

        # Feature count
        tech_count = len([c for c in df.columns if c in
                         self.MOMENTUM_FEATURES + self.VOLUME_FEATURES +
                         self.VOLATILITY_FEATURES + self.INTUITION_FEATURES +
                         self.TRADITIONAL_FEATURES])
        fund_count = len([c for c in self.FUNDAMENTAL_FEATURES if c in df.columns])
        macro_count = len([c for c in self.MACRO_FEATURES if c in df.columns])

        version = "V4" if include_macro else "V3"
        self.logger.info(f"{version} ML data: {len(df):,} samples")
        self.logger.info(f"  Technical: {tech_count}, Fundamental: {fund_count}, Macro: {macro_count}")
        if tech_count + fund_count > 0:
            self.logger.info(f"  Fundamental ratio: {fund_count/(tech_count+fund_count)*100:.0f}%")

        return df

    def _add_macro_features(self, df: pd.DataFrame,
                            start_date: str, end_date: str) -> pd.DataFrame:
        """
        V4: 매크로 Regime Detection 피처 추가

        "2021~2022년 폭락장을 피하기 위한 시장 온도계"
        """
        self.logger.info("Adding V4 macro features (Regime Detection)...")

        # MacroFeatureEngineer 초기화
        macro_eng = MacroFeatureEngineer(self.db_path)

        # 날짜별 매크로 피처 준비
        macro_df = macro_eng.prepare_macro_features(start_date, end_date)

        # 종목 데이터에 병합
        df = pd.merge(df, macro_df, on='date', how='left')


        # 복합 피처 계산
        df = self._compute_macro_composite_features(df)

        # Forward fill (주말/휴일 데이터)
        macro_cols = [c for c in self.MACRO_FEATURES if c in df.columns]
        df[macro_cols] = df.groupby('stock_code')[macro_cols].ffill()

        self.logger.info(f"Added {len(macro_cols)} macro features")
        return df

    def _compute_macro_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        V4 복합 매크로 피처 계산

        - macro_risk_score: 종합 리스크 점수
        - regime_momentum_interaction: regime과 모멘텀 상호작용
        """
        df = df.copy()

        # === Macro Risk Score ===
        # 높을수록 위험 (음수 regime + 높은 fear + 양수 dollar + 양수 bond spread)
        risk_components = []

        if 'market_regime_score' in df.columns:
            # Regime이 음수면 리스크 높음 (정규화)
            risk_components.append(-df['market_regime_score'].clip(-0.3, 0.3) / 0.3)

        if 'fear_index_delta' in df.columns:
            # VKOSPI 급등하면 리스크 높음
            risk_components.append(df['fear_index_delta'].clip(-10, 10) / 10)

        if 'dollar_impact' in df.columns:
            # 달러 급등하면 리스크 높음
            risk_components.append(df['dollar_impact'].clip(-0.1, 0.1) / 0.1)

        if 'bond_stock_spread' in df.columns:
            # 채권 선호 높으면 리스크 높음
            risk_components.append(df['bond_stock_spread'].clip(-0.1, 0.1) / 0.1)

        if risk_components:
            df['macro_risk_score'] = sum(risk_components) / len(risk_components)
        else:
            df['macro_risk_score'] = 0

        # === Regime-Momentum Interaction ===
        # "좋은 장에서 좋은 모멘텀을 가진 놈" vs "나쁜 장에서 모멘텀만 좋은 놈"
        if 'market_regime_score' in df.columns and 'mom_20d' in df.columns:
            df['regime_momentum_interaction'] = (
                df['market_regime_score'].clip(-0.2, 0.2) *
                df['mom_20d'].clip(-0.5, 0.5)
            )
        else:
            df['regime_momentum_interaction'] = 0

        return df


# =============================================================================
# Top-level worker function for multiprocessing (must be picklable)
# =============================================================================

def _process_year_chunk(args):
    """Process a single year-chunk: load data, compute features, return trimmed DataFrame.

    Must be a top-level function (not a method) for multiprocessing pickling.
    Each worker creates its own FeatureEngineer + DB connection.
    Suppresses verbose prints from sub-methods — only prints the summary line.
    """
    import time as _t
    import io
    import contextlib
    year, eligible_codes, db_path, target_horizon, min_market_cap, include_fundamental = args

    t0 = _t.time()
    fe = FeatureEngineer(db_path)
    fe._indexes_ensured = True  # Skip index creation prints (indexes already exist)

    # Buffer: 1 year before for rolling features (252-day lookback)
    buffer_start = f"{year - 1}0101"
    # Buffer after: covers forward returns shift(-1 - horizon)
    buffer_months = max(3, (target_horizon // 20) + 2)
    buffer_end_year = year + 1 if buffer_months > 0 else year
    buffer_end_month = min(12, buffer_months)
    buffer_end = f"{buffer_end_year}{buffer_end_month:02d}28"

    # Suppress verbose prints from sub-methods
    with contextlib.redirect_stdout(io.StringIO()):
        # Step 1: Load raw data for this chunk
        df = fe._load_raw_chunk(buffer_start, buffer_end, eligible_codes)
        if len(df) == 0:
            return pd.DataFrame()
        df = fe._optimize_dtypes(df)
        t1 = _t.time()

        # Step 2: Compute technical features
        df = fe.compute_features(df)
        df = fe._optimize_dtypes(df)
        t2 = _t.time()

        # Step 3: Financial features (use pre-loaded pivot if available)
        if include_fundamental:
            if _SHARED_FIN_PIVOT is not None and len(_SHARED_FIN_PIVOT) > 0:
                # Fast path: merge pre-loaded pivot with prices (~1-2s)
                fin_df = fe._merge_financial_with_prices(
                    _SHARED_FIN_PIVOT, df,
                    f"{year}0101", f"{year}1231", min_market_cap
                )
            else:
                # Fallback: full SQL load (original path)
                fin_df = fe.load_financial_features(
                    f"{year}0101", f"{year}1231", min_market_cap, price_df=df
                )
            if fin_df is not None and len(fin_df) > 0:
                df = fe.merge_financial_features(df, fin_df)
                df = fe._optimize_dtypes(df)
        t3 = _t.time()

        # Step 4: Forward returns
        df = fe.add_forward_returns(df, [target_horizon])
        t4 = _t.time()

    # Step 5: Trim to target year only
    year_start = f"{year}0101"
    year_end = f"{year}1231"
    df = df[(df['date'] >= year_start) & (df['date'] <= year_end)]

    total = _t.time() - t0
    print(f'    [Year {year}] sql={t1-t0:.1f}s feat={t2-t1:.1f}s fin={t3-t2:.1f}s fwd={t4-t3:.1f}s total={total:.1f}s rows={len(df):,}', flush=True)

    gc.collect()
    return df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Feature Engineering V4')
    parser.add_argument('--version', choices=['v3', 'v4'], default='v4',
                        help='Feature version (v3: no macro, v4: with macro)')
    parser.add_argument('--start-date', default='20200101', help='Start date')
    parser.add_argument('--end-date', default='20260128', help='End date')
    parser.add_argument('--output', default=None, help='Output parquet file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    fe = FeatureEngineer('krx_stock_data.db')
    include_macro = (args.version == 'v4')

    print(f"\n{'='*60}")
    print(f"Feature Engineering {args.version.upper()}")
    print(f"{'='*60}")

    df = fe.prepare_ml_data(
        start_date=args.start_date,
        end_date=args.end_date,
        target_horizon=21,
        include_fundamental=True,
        include_macro=include_macro
    )

    # 피처 통계
    print(f"\n{'='*60}")
    print(f"Feature Summary")
    print(f"{'='*60}")

    tech_features = [c for c in df.columns if c in
                     fe.MOMENTUM_FEATURES + fe.VOLUME_FEATURES +
                     fe.VOLATILITY_FEATURES + fe.INTUITION_FEATURES +
                     fe.TRADITIONAL_FEATURES]
    fund_features = [c for c in fe.FUNDAMENTAL_FEATURES if c in df.columns]
    macro_features = [c for c in fe.MACRO_FEATURES if c in df.columns]

    print(f"\nTechnical features ({len(tech_features)}):")
    for f in tech_features[:10]:
        print(f"  - {f}")
    if len(tech_features) > 10:
        print(f"  ... and {len(tech_features)-10} more")

    print(f"\nFundamental features ({len(fund_features)}):")
    for f in fund_features:
        print(f"  - {f}")

    if include_macro:
        print(f"\nMacro features ({len(macro_features)}):")
        for f in macro_features:
            print(f"  - {f}")

        # 매크로 피처 통계
        print(f"\nMacro Feature Statistics:")
        for col in macro_features:
            if col in df.columns:
                print(f"  {col}: mean={df[col].mean():.4f}, "
                      f"std={df[col].std():.4f}, "
                      f"null%={df[col].isna().mean()*100:.1f}%")

    print(f"\n총 피처 수: {len(tech_features) + len(fund_features) + len(macro_features)}")
    print(f"데이터 크기: {len(df):,} rows")
    print(f"종목 수: {df['stock_code'].nunique()}")
    print(f"날짜 범위: {df['date'].min()} ~ {df['date'].max()}")

    # 저장
    if args.output:
        df.to_parquet(args.output, index=False)
        print(f"\nSaved to {args.output}")
