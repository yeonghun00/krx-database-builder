"""
Feature Engineering V2 - 모델 심폐소생술

변경사항:
1. 모멘텀/수급 피처 대폭 강화 (재무 30-40% 목표)
2. 섹터 중립화 (Sector Neutralization)
3. Delta 피처 추가 (QoQ, YoY 변화)
4. 본능 전략 피처 (낙폭과대, 거래량 폭발, 과거 영광)
"""

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


class FeatureEngineer:
    """V3: 퀀트 팀장 피드백 반영 - 피처 다이어트 + 본능 강화"""

    # =========================================================================
    # 피처 그룹 정의 (V3: 45개 → 25개로 압축)
    # =========================================================================

    # [그룹 1] 모멘텀 피처 - 15개 → 4개로 압축!
    # "단기(5d), 중기(60d), 장기(126d), 시장대비(RS)" 만 남김
    MOMENTUM_FEATURES = [
        'mom_5d',                # 단기 모멘텀 (1주)
        'mom_60d',               # 중기 모멘텀 (3개월)
        'mom_126d',              # 장기 모멘텀 (6개월)
        'rs_vs_market_20d',      # 시장 대비 상대강도 (핵심!)
    ]

    # [그룹 2] 수급/거래량 피처 - 강화! (세력 감지 = 핵심)
    VOLUME_FEATURES = [
        'volume_surprise',       # 거래량 폭발 (20일 평균 대비) 🔥
        'volume_trend',          # 거래량 추세 (5일 vs 20일)
        'value_surprise',        # 거래대금 폭발
        'accumulation_index',    # 누적/배분 지표
        'smart_money_flow',      # 스마트머니 흐름
        'volume_breakout',       # 거래량 돌파 신호 (신규)
    ]

    # [그룹 3] 변동성/리스크 피처 - 유지
    VOLATILITY_FEATURES = [
        'volatility_20d',
        'volatility_ratio',      # 단기/장기 변동성 비율 (VCP)
        'drawdown_from_high',    # 고점 대비 낙폭 🔥
        'recovery_from_low',     # 저점 대비 반등
    ]

    # [그룹 4] 본능 전략 피처 - 결합 피처 추가! 🔥
    INTUITION_FEATURES = [
        'past_glory_1y',         # 1년간 최대 상승률
        'fallen_angel_score',    # 추락한 천사 점수
        'vcp_score',             # Volatility Contraction Pattern
        # === 신규: 결합 피처 (Interaction Features) ===
        'glory_correction_volume',  # 영광 * 낙폭 * 거래량폭발 🔥🔥🔥
        'fear_greed_signal',        # 공포 속 탐욕 신호
        'smart_accumulation',       # 스마트머니 매집 신호
    ]

    # [그룹 5] 전통적 기술 지표 - 축소
    TRADITIONAL_FEATURES = [
        'rsi_14',
        'bb_squeeze',            # 볼린저밴드 수축
    ]

    # [그룹 6] 재무 피처 - Rank 제거, Delta/Z-score만!
    FUNDAMENTAL_FEATURES = [
        # Delta 피처만 (변화가 중요!)
        'roe_delta_qoq',         # ROE 분기 변화
        'roe_sector_zscore',     # ROE 섹터 대비 Z-score (신규)
        'revenue_growth_accel',  # 매출 성장 가속도
        'margin_improvement',    # 마진 개선
    ]

    # 전체 피처 리스트
    FEATURE_COLUMNS = (
        MOMENTUM_FEATURES +
        VOLUME_FEATURES +
        VOLATILITY_FEATURES +
        INTUITION_FEATURES +
        TRADITIONAL_FEATURES +
        FUNDAMENTAL_FEATURES
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

        # 기본 계산
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
        """모멘텀 피처 계산 (V3: 압축됨 - 5d, 60d, 126d, RS만)"""

        # Multi-timeframe momentum (V3: 5, 60, 126일만 사용)
        for period in [5, 20, 60, 120, 126]:
            df[f'mom_{period}d'] = grouped['closing_price'].transform(
                lambda x: x.pct_change(period)
            )

        # Moving averages
        for period in [5, 20, 60, 120]:
            df[f'ma_{period}'] = grouped['closing_price'].transform(
                lambda x: x.rolling(period, min_periods=period//2).mean()
            )
            df[f'dist_ma_{period}'] = (
                df['closing_price'] / df[f'ma_{period}'].clip(lower=1) - 1
            )

        # MA Trend (정배열 여부): 5 > 20 > 60
        df['ma_trend'] = (
            (df['ma_5'] > df['ma_20']).astype(float) * 0.5 +
            (df['ma_20'] > df['ma_60']).astype(float) * 0.5
        )

        # Relative Strength vs Market
        for period in [20, 60]:
            stock_ret = df[f'mom_{period}d']
            market_ret = df.groupby('date')[f'mom_{period}d'].transform('median')
            df[f'rs_vs_market_{period}d'] = stock_ret - market_ret

        # Momentum Consistency (상승일 비율)
        df['up_day'] = (df['return'] > 0).astype(float)
        df['mom_consistency'] = grouped['up_day'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )

        # Momentum Acceleration (최근 모멘텀 / 과거 모멘텀)
        df['mom_acceleration'] = df['mom_20d'] / df['mom_60d'].clip(lower=0.01).abs()
        df['mom_acceleration'] = df['mom_acceleration'].clip(-5, 5)

    def _compute_volume_features(self, df: pd.DataFrame, grouped) -> None:
        """수급/거래량 피처 계산"""

        # Volume averages
        df['vol_5d'] = grouped['volume'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        df['vol_20d'] = grouped['volume'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )

        # Volume Surprise (거래량 폭발)
        df['volume_surprise'] = df['volume'] / df['vol_20d'].clip(lower=1)

        # Volume Trend
        df['volume_trend'] = df['vol_5d'] / df['vol_20d'].clip(lower=1)

        # Value Surprise (거래대금 폭발)
        df['value_20d'] = grouped['value'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        df['value_surprise'] = df['value'] / df['value_20d'].clip(lower=1)

        # Smart Money Flow (종가 위치 * 거래량)
        # 종가가 고가 근처면 매집, 저가 근처면 투매
        df['close_location'] = (
            (df['closing_price'] - df['low_price']) /
            (df['high_price'] - df['low_price']).clip(lower=1)
        )
        df['daily_mf'] = (df['close_location'] * 2 - 1) * df['volume']
        df['smart_money_flow'] = grouped['daily_mf'].transform(
            lambda x: x.rolling(20, min_periods=10).sum()
        ) / grouped['volume'].transform(
            lambda x: x.rolling(20, min_periods=10).sum()
        ).clip(lower=1)

        # Accumulation Index
        df['accumulation_index'] = grouped['smart_money_flow'].transform(
            lambda x: x.rolling(10, min_periods=5).mean()
        )

        # Volume Breakout (V3 신규) - 거래량이 최근 60일 최고 대비 얼마나 높은지
        df['vol_60d_max'] = grouped['volume'].transform(
            lambda x: x.rolling(60, min_periods=30).max()
        )
        df['volume_breakout'] = df['volume'] / df['vol_60d_max'].clip(lower=1)

    def _compute_volatility_features(self, df: pd.DataFrame, grouped) -> None:
        """변동성/리스크 피처 계산"""

        # Historical Volatility
        for period in [20, 60]:
            df[f'volatility_{period}d'] = grouped['return'].transform(
                lambda x: x.rolling(period, min_periods=period//2).std() * np.sqrt(252)
            )

        # Volatility Ratio (VCP 패턴 감지)
        df['volatility_ratio'] = (
            df['volatility_20d'] / df['volatility_60d'].clip(lower=0.01)
        )

        # ATR
        df['tr'] = np.maximum(
            df['high_price'] - df['low_price'],
            np.maximum(
                (df['high_price'] - df.groupby('stock_code')['closing_price'].shift(1)).abs(),
                (df['low_price'] - df.groupby('stock_code')['closing_price'].shift(1)).abs()
            )
        )
        df['atr_20'] = grouped['tr'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        df['atr_ratio'] = df['tr'] / df['atr_20'].clip(lower=1)

        # Drawdown from High (고점 대비 낙폭)
        df['high_52w'] = grouped['high_price'].transform(
            lambda x: x.rolling(252, min_periods=126).max()
        )
        df['drawdown_from_high'] = df['closing_price'] / df['high_52w'].clip(lower=1) - 1

        # Recovery from Low (저점 대비 반등)
        df['low_52w'] = grouped['low_price'].transform(
            lambda x: x.rolling(252, min_periods=126).min()
        )
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

    def _compute_traditional_features(self, df: pd.DataFrame, grouped) -> None:
        """전통적 기술 지표"""

        # RSI
        delta = grouped['closing_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))

        avg_gain = grouped['closing_price'].transform(
            lambda x: x.diff().where(x.diff() > 0, 0).rolling(14).mean()
        )
        avg_loss = grouped['closing_price'].transform(
            lambda x: (-x.diff().where(x.diff() < 0, 0)).rolling(14).mean()
        )

        rs = avg_gain / avg_loss.clip(lower=0.001)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # RSI Divergence (가격은 신저가인데 RSI는 아닌 경우 = 반등 신호)
        price_new_low = (df['closing_price'] <= df['low_52w'] * 1.05).astype(float)
        rsi_not_low = (df['rsi_14'] > 30).astype(float)
        df['rsi_divergence'] = price_new_low * rsi_not_low

        # Bollinger Bands
        df['bb_mid'] = df['ma_20']
        df['bb_std'] = grouped['closing_price'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (
            (df['closing_price'] - df['bb_lower']) /
            (df['bb_upper'] - df['bb_lower']).clip(lower=1)
        )

        # BB Squeeze (볼린저밴드 수축)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].clip(lower=1)
        df['bb_width_avg'] = grouped['bb_width'].transform(
            lambda x: x.rolling(60, min_periods=30).mean()
        )
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
            'return', 'log_return', 'up_day',
            'ma_5', 'ma_20', 'ma_60', 'ma_120',
            'vol_5d', 'vol_20d', 'value_20d',
            'close_location', 'daily_mf',
            'tr', 'atr_20',
            'high_52w', 'low_52w',
            'bb_mid', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width', 'bb_width_avg',
        ]
        df.drop(columns=intermediate, errors='ignore', inplace=True)

    def load_financial_features(self, start_date: str, end_date: str,
                                min_market_cap: int = 500000000000) -> pd.DataFrame:
        """Load financial features with Delta calculations."""
        self.logger.info("Loading financial features (with deltas)...")

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
        fin_cols = ['stock_code', 'date'] + [
            col for col in self.FUNDAMENTAL_FEATURES
            if col in fin_df.columns
        ]
        fin_subset = fin_df[fin_cols].drop_duplicates(subset=['stock_code', 'date'])

        merged = pd.merge(
            tech_df,
            fin_subset,
            on=['stock_code', 'date'],
            how='left'
        )

        # Fill missing fundamental features
        for col in self.FUNDAMENTAL_FEATURES:
            if col in merged.columns:
                if col.startswith('is_null_'):
                    merged[col] = merged[col].fillna(1)
                elif col.endswith('_rank'):
                    merged[col] = merged[col].fillna(0.5)
                else:
                    merged[col] = merged.groupby('date')[col].transform(
                        lambda x: x.fillna(x.median())
                    )

        return merged

    def add_forward_returns(self, df: pd.DataFrame,
                            horizons: List[int] = None) -> pd.DataFrame:
        """
        Add forward returns using Open-to-Open pricing (정석)

        Signal: T일 종가 데이터까지 보고 생성
        Buy: T+1일 시가 (Open)
        Sell: T+1+h일 시가 (Open)

        Example (h=21):
          - T일 저녁에 신호 생성
          - T+1일 시가에 매수
          - T+22일 시가에 매도 (21 거래일 보유)
        """
        horizons = horizons or [21]  # V2 기본값: 21일

        df = df.sort_values(['stock_code', 'date']).copy()
        grouped = df.groupby('stock_code')

        for h in horizons:
            col_name = f'forward_return_{h}d'

            # Open-to-Open: T+1 시가 → T+1+h 시가
            # (T+1+h일 시가 - T+1일 시가) / T+1일 시가
            open_t1 = grouped['opening_price'].shift(-1)        # T+1 시가
            open_t1_h = grouped['opening_price'].shift(-1 - h)  # T+1+h 시가
            df[col_name] = (open_t1_h - open_t1) / open_t1

            df[col_name] = df[col_name].clip(-0.50, 0.50)

            # V3 신규: Alpha (시장 대비 초과수익률) 🔥
            # "코스피가 5% 오를 때 내 종목이 5% 오른 건 실력이 아니다"
            # "시장이 -2% 빠질 때 +3% 오른 놈을 찾아라"
            market_return = df.groupby('date')[col_name].transform('median')
            alpha_col = f'forward_alpha_{h}d'
            df[alpha_col] = df[col_name] - market_return

            # Sector-neutralized target (핵심!)
            rank_col = f'target_rank_{h}d'
            df[rank_col] = df.groupby('date')[col_name].rank(pct=True)

            # Alpha rank (V3: 이걸 타겟으로!)
            alpha_rank_col = f'target_alpha_rank_{h}d'
            df[alpha_rank_col] = df.groupby('date')[alpha_col].rank(pct=True)

            # Sector-relative rank
            sector_rank_col = f'target_sector_rank_{h}d'
            df[sector_rank_col] = df.groupby(['date', 'sector'])[col_name].rank(pct=True)

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
                        include_fundamental: bool = True) -> pd.DataFrame:
        """Full pipeline for ML data preparation."""

        # 1년 버퍼
        buffer_start = str(int(start_date[:4]) - 1) + start_date[4:]

        # Step 1: Load and compute technical features
        df = self.load_raw_data(buffer_start, end_date)
        df = self.compute_features(df)

        # Step 2: Load and merge financial features
        if include_fundamental:
            fin_df = self.load_financial_features(start_date, end_date, min_market_cap)
            df = self.merge_financial_features(df, fin_df)

        # Step 3: Add forward returns
        df = self.add_forward_returns(df, [target_horizon])

        # Step 4: Filter universe
        df = self.filter_universe(df, min_market_cap=min_market_cap)

        # Filter to requested date range
        df = df[df['date'] >= start_date]

        # Feature count
        tech_count = len([c for c in df.columns if c in
                         self.MOMENTUM_FEATURES + self.VOLUME_FEATURES +
                         self.VOLATILITY_FEATURES + self.INTUITION_FEATURES +
                         self.TRADITIONAL_FEATURES])
        fund_count = len([c for c in self.FUNDAMENTAL_FEATURES if c in df.columns])

        self.logger.info(f"V2 ML data: {len(df):,} samples")
        self.logger.info(f"  Technical: {tech_count}, Fundamental: {fund_count}")
        self.logger.info(f"  Target ratio: {fund_count/(tech_count+fund_count)*100:.0f}% fundamental (목표: 30-40%)")

        return df


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    fe = FeatureEngineer('krx_stock_data.db')
    df = fe.prepare_ml_data(
        start_date='20240101',
        end_date='20260128',
        target_horizon=21,
        include_fundamental=True
    )

    print(f"\n총 피처 수: {len([c for c in df.columns if c in fe.FEATURE_COLUMNS])}")
    print(f"데이터 크기: {len(df):,} rows")
