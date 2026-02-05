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
import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import List, Optional
from pathlib import Path

# Import financial feature generator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.financial_features import FinancialFeatureGenerator
from ml.macro_features import MacroFeatureEngineer


class FeatureEngineer:
    """V3: 퀀트 팀장 피드백 반영 - 피처 다이어트 + 본능 강화"""

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

    def load_raw_data(self, start_date: str, end_date: str,
                      markets: List[str] = None) -> pd.DataFrame:
        """Load raw OHLCV data from database."""
        markets = markets or ['kospi', 'kosdaq']
        market_placeholders = ','.join(['?' for _ in markets])

        query = f"""
        SELECT
            dp.stock_code,
            dp.date,
            dp.market_type,
            dp.opening_price,
            dp.high_price,
            dp.low_price,
            dp.closing_price,
            dp.volume,
            dp.value,
            dp.market_cap,
            s.current_name as name,
            s.current_sector_type as sector
        FROM daily_prices dp
        JOIN stocks s ON dp.stock_code = s.stock_code
        WHERE dp.date >= ? AND dp.date <= ?
          AND dp.market_type IN ({market_placeholders})
          AND dp.closing_price > 0
          AND dp.volume > 0
        ORDER BY dp.stock_code, dp.date
        """

        params = [start_date, end_date] + markets

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        self.logger.info(f"Loaded {len(df):,} rows from {start_date} to {end_date}")
        return df

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all V2 features."""
        self.logger.info("Computing V2 features (momentum-heavy)...")

        df = df.sort_values(['stock_code', 'date']).copy()
        grouped = df.groupby('stock_code')
        
        # Calculate returns first (needed for forward returns calculation)
        df['return'] = grouped['closing_price'].pct_change()
        df['log_return'] = np.log1p(df['return'])

        # ================================================================
        # [그룹 1] 모멘텀 피처
        # ================================================================
        self._compute_momentum_features(df, grouped)

        # ================================================================
        # [그룹 2] 수급/거래량 피처
        # ================================================================
        self._compute_volume_features(df, grouped)

        # ================================================================
        # [그룹 3] 변동성/리스크 피처
        # ================================================================
        self._compute_volatility_features(df, grouped)

        # ================================================================
        # [그룹 4] 본능 전략 피처
        # ================================================================
        self._compute_intuition_features(df, grouped)

        # ================================================================
        # [그룹 5] 전통적 기술 지표
        # ================================================================
        self._compute_traditional_features(df, grouped)

        # ================================================================
        # 섹터 중립화 (Sector Neutralization)
        # ================================================================
        self._apply_sector_neutralization(df)

        # Cleanup
        self._cleanup_intermediate_cols(df)

        self.logger.info(f"Computed {len(self.FEATURE_COLUMNS)} V2 features")
        return df

    def _compute_momentum_features(self, df: pd.DataFrame, grouped) -> None:
        """모멘텀 피처 계산 (V3: 압축됨 - 5d, 60d, 126d, RS만) - 최적화 버전"""

        # Multi-timeframe momentum - 벡터화 (pct_change는 이미 빠름)
        for period in [5, 20, 60, 120, 126]:
            df[f'mom_{period}d'] = grouped['closing_price'].pct_change(period)

        # V5: Intermediate Momentum (skip last month) - price_t-21 / price_t-126 - 1
        df['intermediate_momentum'] = grouped['closing_price'].shift(21) / grouped['closing_price'].shift(126) - 1

        # Moving averages - 벡터화된 rolling
        for period in [5, 20, 60, 120]:
            df[f'ma_{period}'] = grouped['closing_price'].rolling(
                period, min_periods=period//2
            ).mean().reset_index(level=0, drop=True)

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
        ).mean().reset_index(level=0, drop=True)

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

        df['vol_5d'] = vol_rolling_5.mean().reset_index(level=0, drop=True)
        df['vol_20d'] = vol_rolling_20.mean().reset_index(level=0, drop=True)
        df['vol_60d_max'] = vol_rolling_60.max().reset_index(level=0, drop=True)

        # Volume Surprise / Trend / Breakout - 벡터 연산
        df['volume_surprise'] = df['volume'] / df['vol_20d'].clip(lower=1)
        df['volume_trend'] = df['vol_5d'] / df['vol_20d'].clip(lower=1)
        df['volume_breakout'] = df['volume'] / df['vol_60d_max'].clip(lower=1)

        # Value Surprise (거래대금 폭발)
        df['value_20d'] = grouped['value'].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
        df['value_surprise'] = df['value'] / df['value_20d'].clip(lower=1)

        # Smart Money Flow (종가 위치 * 거래량)
        df['close_location'] = (
            (df['closing_price'] - df['low_price']) /
            (df['high_price'] - df['low_price']).clip(lower=1)
        )
        df['daily_mf'] = (df['close_location'] * 2 - 1) * df['volume']

        # Rolling sums - 한 번에 계산
        mf_rolling_sum = grouped['daily_mf'].rolling(20, min_periods=10).sum().reset_index(level=0, drop=True)
        vol_rolling_sum = grouped['volume'].rolling(20, min_periods=10).sum().reset_index(level=0, drop=True)
        df['smart_money_flow'] = mf_rolling_sum / vol_rolling_sum.clip(lower=1)

        # Accumulation Index
        df['accumulation_index'] = grouped['smart_money_flow'].rolling(
            10, min_periods=5
        ).mean().reset_index(level=0, drop=True)

        # === V4.1 신규: "돈 냄새" 피처 ===
        # Price-Volume Synergy (가격×거래량 시너지) - "진짜 상승"만 포착
        df['price_volume_synergy'] = (
            df['mom_5d'].clip(-0.3, 0.3) *
            (df['volume_surprise'] - 1).clip(0, 5)
        )
        df['price_volume_synergy'] = df['price_volume_synergy'].clip(-1, 1)

        # === V4.3 신규: Amihud Illiquidity 🔥 ===
        # "거래량 대비 가격 변동이 큰 종목 = 슬리피지 지옥"
        # 높을수록 비유동적 → 필터링 대상
        df['amihud_illiquidity'] = (
            df['return'].abs() / (df['value'].clip(lower=1e6) / 1e9)  # 10억 단위
        )
        df['amihud_illiquidity'] = grouped['amihud_illiquidity'].rolling(
            20, min_periods=10
        ).mean().reset_index(level=0, drop=True)
        # Percentile rank (높을수록 비유동적)
        df['amihud_rank'] = df.groupby('date')['amihud_illiquidity'].rank(pct=True)

    def _compute_volatility_features(self, df: pd.DataFrame, grouped) -> None:
        """변동성/리스크 피처 계산 - 최적화 버전"""

        # Historical Volatility - 벡터화
        sqrt_252 = np.sqrt(252)
        df['volatility_20d'] = grouped['return'].rolling(20, min_periods=10).std().reset_index(level=0, drop=True) * sqrt_252
        df['volatility_60d'] = grouped['return'].rolling(60, min_periods=30).std().reset_index(level=0, drop=True) * sqrt_252

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
        df['atr_20'] = grouped['tr'].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
        df['atr_ratio'] = df['tr'] / df['atr_20'].clip(lower=1)

        # Drawdown from High / Recovery from Low - 벡터화
        df['high_52w'] = grouped['high_price'].rolling(252, min_periods=126).max().reset_index(level=0, drop=True)
        df['low_52w'] = grouped['low_price'].rolling(252, min_periods=126).min().reset_index(level=0, drop=True)

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

        avg_gain = grouped.apply(lambda x: gain.loc[x.index].rolling(14).mean()).reset_index(level=0, drop=True) if False else \
                   gain.groupby(df['stock_code']).rolling(14).mean().reset_index(level=0, drop=True)
        avg_loss = loss.groupby(df['stock_code']).rolling(14).mean().reset_index(level=0, drop=True)

        rs = avg_gain / avg_loss.clip(lower=0.001)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # RSI Divergence (가격은 신저가인데 RSI는 아닌 경우 = 반등 신호)
        price_new_low = (df['closing_price'] <= df['low_52w'] * 1.05).astype(float)
        rsi_not_low = (df['rsi_14'] > 30).astype(float)
        df['rsi_divergence'] = price_new_low * rsi_not_low

        # Bollinger Bands - 벡터화
        df['bb_mid'] = df['ma_20']
        df['bb_std'] = grouped['closing_price'].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (
            (df['closing_price'] - df['bb_lower']) /
            (df['bb_upper'] - df['bb_lower']).clip(lower=1)
        )

        # BB Squeeze (볼린저밴드 수축) - 벡터화
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].clip(lower=1)
        df['bb_width_avg'] = grouped['bb_width'].rolling(60, min_periods=30).mean().reset_index(level=0, drop=True)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width_avg'] * 0.8).astype(float)

    def _apply_sector_neutralization(self, df: pd.DataFrame) -> None:
        """섹터 중립화: 모든 피처를 섹터 내 순위로 변환"""

        # Sector-relative momentum (핵심!)
        def sector_zscore(x):
            std = x.std()
            if std < 0.01:
                std = 0.01
            return (x - x.mean()) / std

        df['rs_vs_sector_20d'] = df.groupby(['date', 'sector'])['mom_20d'].transform(sector_zscore)

        # 모멘텀 피처들을 섹터 내 랭크로 변환
        momentum_cols = ['mom_5d', 'mom_10d', 'mom_20d', 'mom_60d']
        for col in momentum_cols:
            if col in df.columns:
                df[f'{col}_sector_rank'] = df.groupby(['date', 'sector'])[col].rank(pct=True)

    def _cleanup_intermediate_cols(self, df: pd.DataFrame) -> None:
        """중간 계산 컬럼 제거"""
        intermediate = [
            'log_return', 'up_day',
            'ma_5', 'ma_20', 'ma_60', 'ma_120',
            'vol_5d', 'vol_20d', 'value_20d',
            'close_location', 'daily_mf',
            'tr', 'atr_20',
            'high_52w', 'low_52w',
            'bb_mid', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width', 'bb_width_avg',
        ]
        df.drop(columns=intermediate, errors='ignore', inplace=True)

    def _load_financial_features_fast(self, start_date: str, end_date: str,
                                       min_market_cap: int) -> pd.DataFrame:
        """V4.1: 경량화된 재무 피처 로딩 (필요한 컬럼만 직접 쿼리)

        DB Schema:
        - financial_periods: stock_code, fiscal_date, available_date
        - financial_items_bs_cf: period_id (FK), item_code_normalized, amount_current
        - financial_items_pl: period_id (FK), item_code_normalized, amount_current_ytd
        """
        import sqlite3

        try:
            conn = sqlite3.connect(self.db_path)

            # 1. 필요한 재무 항목만 로드 (분리 쿼리 - 인덱스 활용)
            available_start = str(int(start_date[:4]) - 1) + start_date[4:]

            # BS/CF 쿼리 (자본, 자산)
            bs_query = """
            SELECT fp.stock_code, fp.available_date, fp.industry_name as sector,
                   bs.item_code_normalized as item_code, bs.amount_current as amount
            FROM financial_periods fp
            JOIN financial_items_bs_cf bs ON bs.period_id = fp.id
            WHERE fp.consolidation_type = '연결'
              AND fp.available_date >= ?
              AND bs.item_code_normalized IN ('ifrs-full_Equity', 'ifrs-full_Assets')
            """
            bs_df = pd.read_sql_query(bs_query, conn, params=[available_start])

            # PL 쿼리 (매출, 영업이익, 순이익)
            pl_query = """
            SELECT fp.stock_code, fp.available_date, fp.industry_name as sector,
                   pl.item_code_normalized as item_code, pl.amount_current_ytd as amount
            FROM financial_periods fp
            JOIN financial_items_pl pl ON pl.period_id = fp.id
            WHERE fp.consolidation_type = '연결'
              AND fp.available_date >= ?
              AND pl.item_code_normalized IN ('ifrs-full_Revenue', 'dart_OperatingIncomeLoss', 'ifrs-full_ProfitLoss', 'ifrs-full_GrossProfit')
            """
            pl_df = pd.read_sql_query(pl_query, conn, params=[available_start])

            # 합치기
            fin_df = pd.concat([bs_df, pl_df], ignore_index=True)

            if len(fin_df) == 0:
                conn.close()
                return None

            # Pivot: item_code → columns
            fin_pivot = fin_df.pivot_table(
                index=['stock_code', 'available_date', 'sector'],
                columns='item_code',
                values='amount',
                aggfunc='first'
            ).reset_index()

            # Rename columns
            rename_map = {
                'ifrs-full_Equity': 'equity',
                'ifrs-full_Assets': 'assets',
                'ifrs-full_Revenue': 'revenue',
                'dart_OperatingIncomeLoss': 'operating_income',
                'ifrs-full_ProfitLoss': 'net_income',
                'ifrs-full_GrossProfit': 'gross_profit',
            }
            fin_pivot = fin_pivot.rename(columns=rename_map)
            fin_pivot['available_date'] = pd.to_datetime(fin_pivot['available_date'], format='%Y%m%d')

            # 2. Daily prices 로드 (market cap filter)
            price_query = """
            SELECT dp.stock_code, dp.date, dp.market_cap, s.current_sector_type as sector_price
            FROM daily_prices dp
            JOIN stocks s ON dp.stock_code = s.stock_code
            WHERE dp.date >= ? AND dp.date <= ? AND dp.market_cap >= ?
            ORDER BY dp.stock_code, dp.date
            """
            price_df = pd.read_sql_query(price_query, conn, params=[start_date, end_date, min_market_cap])
            conn.close()

            if len(price_df) == 0:
                return None

            price_df['date'] = pd.to_datetime(price_df['date'], format='%Y%m%d')

            # 3. Fast forward-fill approach (merge_asof 대신)
            # 각 stock의 마지막 재무 데이터를 daily로 forward-fill
            price_df = price_df.sort_values(['stock_code', 'date']).reset_index(drop=True)
            fin_pivot = fin_pivot.sort_values(['stock_code', 'available_date']).reset_index(drop=True)

            # 재무 데이터를 일별로 확장 (available_date를 date 컬럼으로)
            fin_pivot = fin_pivot.rename(columns={'available_date': 'date'})
            fin_cols = ['equity', 'assets', 'revenue', 'operating_income', 'net_income']
            fin_cols = [c for c in fin_cols if c in fin_pivot.columns]

            # Merge and forward fill within each stock
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

            # 4. 비율 계산
            merged['roe'] = merged['net_income'] / merged['equity'].clip(lower=1)
            merged['operating_margin'] = merged['operating_income'] / merged['revenue'].clip(lower=1)

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

                def _sector_zscore(x):
                    std = x.std()
                    if std < 0.01:
                        std = 0.01
                    return (x - x.mean()) / std

                if 'sector' in merged.columns or 'sector_price' in merged.columns:
                    sector_col = 'sector_price' if 'sector_price' in merged.columns else 'sector'
                    merged['pb_sector_zscore'] = merged.groupby(['date', sector_col])['pb_ratio'].transform(_sector_zscore)
                else:
                    merged['pb_sector_zscore'] = merged.groupby('date')['pb_ratio'].transform(_sector_zscore)
                merged['pb_sector_zscore'] = merged['pb_sector_zscore'].clip(-3, 3)
                merged.drop(columns=['pb_ratio'], errors='ignore', inplace=True)

            # inf 처리
            for col in ['roe', 'operating_margin', 'revenue_yoy',
                         'gp_over_assets', 'roe_delta_yoy', 'pb_sector_zscore']:
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

    def load_financial_features(self, start_date: str, end_date: str,
                                min_market_cap: int = 500000000000) -> pd.DataFrame:
        """Load financial features with Delta calculations - V4.1 경량화 버전."""
        self.logger.info("Loading financial features (lightweight V4.1)...")

        # V4.1: 직접 DB에서 필요한 컬럼만 로드 (143초 → 5초)
        fin_df = self._load_financial_features_fast(start_date, end_date, min_market_cap)
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

            # V3 신규: ROE 섹터 대비 Z-score
            # "절대값 Rank가 아니라, 섹터 내에서 얼마나 튀어나왔는지"
            def sector_zscore(x):
                mean = x.mean()
                std = x.std()
                if std == 0 or pd.isna(std):
                    return 0
                return (x - mean) / std

            if 'sector' in fin_df.columns:
                fin_df['roe_sector_zscore'] = fin_df.groupby(['date', 'sector'])['roe'].transform(sector_zscore)
                fin_df['roe_sector_zscore'] = fin_df['roe_sector_zscore'].clip(-3, 3)
            else:
                fin_df['roe_sector_zscore'] = fin_df.groupby('date')['roe'].transform(sector_zscore)
                fin_df['roe_sector_zscore'] = fin_df['roe_sector_zscore'].clip(-3, 3)

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
        _v5_fin_cols = ['gp_over_assets', 'roe_delta_yoy', 'pb_sector_zscore']
        raw_cols = ['stock_code', 'date'] + [
            col for col in self._FUNDAMENTAL_RAW + _v5_fin_cols
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
        for col in self._FUNDAMENTAL_RAW + _v5_fin_cols:
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

        # V4.3: 시장 수익률 (Beta 추정용)
        market_ret_series = df.groupby('date')['return'].transform('median')
        df['_market_return_daily'] = market_ret_series

        # ================================================================
        # Rolling Beta 계산 (한 번만, 모든 horizon에서 공유)
        # ================================================================
        def _calc_stock_beta(stock_df):
            """단일 종목의 rolling beta 계산"""
            if len(stock_df) < 60:
                return pd.Series(np.nan, index=stock_df.index)
            market_rets = market_ret_series.loc[stock_df.index]
            stock_rets = stock_df['return']
            rolling_cov = stock_rets.rolling(252, min_periods=60).cov(market_rets)
            rolling_var = market_rets.rolling(252, min_periods=60).var()
            return (rolling_cov / rolling_var.clip(lower=1e-8)).clip(-3, 3)

        # Stock-by-stock beta (groupby.apply)
        df['rolling_beta'] = grouped.apply(
            lambda g: _calc_stock_beta(g)
        ).reset_index(level=0, drop=True)
        df['rolling_beta'] = df['rolling_beta'].fillna(1.0)

        # V5: Rolling Beta 60d (short-window beta)
        def _calc_stock_beta_60d(stock_df):
            """단일 종목의 60일 rolling beta 계산"""
            if len(stock_df) < 30:
                return pd.Series(np.nan, index=stock_df.index)
            market_rets = market_ret_series.loc[stock_df.index]
            stock_rets = stock_df['return']
            rolling_cov = stock_rets.rolling(60, min_periods=30).cov(market_rets)
            rolling_var = market_rets.rolling(60, min_periods=30).var()
            return (rolling_cov / rolling_var.clip(lower=1e-8)).clip(-3, 3)

        df['rolling_beta_60d'] = grouped.apply(
            lambda g: _calc_stock_beta_60d(g)
        ).reset_index(level=0, drop=True)
        df['rolling_beta_60d'] = df['rolling_beta_60d'].fillna(1.0)
        gc.collect()

        # ================================================================
        # Horizon별 forward return + alpha + residual + ranks
        # ================================================================
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

            # Target Ranks (모두 loop 안에서 계산)
            df[f'target_rank_{h}d'] = df.groupby('date')[col_name].rank(pct=True)
            df[f'target_alpha_rank_{h}d'] = df.groupby('date')[alpha_col].rank(pct=True)
            df[f'target_residual_rank_{h}d'] = df.groupby('date')[residual_col].rank(pct=True)
            df[f'target_sector_rank_{h}d'] = df.groupby(['date', 'sector'])[col_name].rank(pct=True)

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
                        use_cache: bool = True) -> pd.DataFrame:
        """
        Full pipeline for ML data preparation.

        Args:
            include_macro: V4 매크로 피처 포함 여부 (Regime Detection)
            use_cache: 캐시 사용 여부 (기본 True - 속도 향상)
        """
        import hashlib
        import os
        import time

        # 캐시 파일명 생성
        cache_key = f"{start_date}_{end_date}_{target_horizon}_{min_market_cap}_{include_fundamental}_{include_macro}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_file = f".cache/features_{cache_hash}.parquet"

        # 캐시 디렉토리 생성
        os.makedirs(".cache", exist_ok=True)

        # 캐시 확인
        if use_cache and os.path.exists(cache_file):
            cache_mtime = os.path.getmtime(cache_file)
            db_mtime = os.path.getmtime(self.db_path)
            if cache_mtime > db_mtime:
                self.logger.info(f"Loading cached features from {cache_file}")
                return pd.read_parquet(cache_file)

        pipeline_t0 = time.time()

        # 1년 버퍼
        buffer_start = str(int(start_date[:4]) - 1) + start_date[4:]

        # Step 1: Load raw data (SQL)
        t0 = time.time()
        self.logger.info("[1/5] Loading raw data...")
        df = self.load_raw_data(buffer_start, end_date)
        print(f'  ⏱ [1/5] Raw data SQL load: {time.time()-t0:.1f}s ({len(df):,} rows)')

        # Step 2: Compute technical features
        t0 = time.time()
        self.logger.info("[2/5] Computing technical features...")
        df = self.compute_features(df)
        print(f'  ⏱ [2/5] Technical features: {time.time()-t0:.1f}s')

        # Step 3: Load and merge financial features
        if include_fundamental:
            t0 = time.time()
            self.logger.info("[3/5] Loading financial features...")
            fin_df = self.load_financial_features(start_date, end_date, min_market_cap)
            t_load = time.time() - t0
            t0 = time.time()
            df = self.merge_financial_features(df, fin_df)
            t_merge = time.time() - t0
            print(f'  ⏱ [3/5] Financial features: load={t_load:.1f}s, merge={t_merge:.1f}s')

        # Step 4: Add forward returns (+ rolling beta)
        t0 = time.time()
        self.logger.info("[4/5] Computing forward returns...")
        df = self.add_forward_returns(df, [target_horizon])
        print(f'  ⏱ [4/5] Forward returns + beta: {time.time()-t0:.1f}s')

        # Step 5: Filter universe
        t0 = time.time()
        df = self.filter_universe(df, min_market_cap=min_market_cap)

        # Filter to requested date range
        df = df[df['date'] >= start_date]
        print(f'  ⏱ [5/5] Filter universe: {time.time()-t0:.1f}s ({len(df):,} rows)')

        # Step 6: V4 - Add macro features (Regime Detection)
        if include_macro:
            t0 = time.time()
            self.logger.info("[6] Adding macro features...")
            df = self._add_macro_features(df, start_date, end_date)
            print(f'  ⏱ [6] Macro features: {time.time()-t0:.1f}s')

        # 캐시 저장
        if use_cache:
            t0 = time.time()
            df.to_parquet(cache_file, index=False)
            print(f'  ⏱ Cache save: {time.time()-t0:.1f}s')
            self.logger.info(f"Cached features to {cache_file}")

        print(f'  ⏱ Total pipeline: {time.time()-pipeline_t0:.1f}s')

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
