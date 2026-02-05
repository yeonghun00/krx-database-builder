"""
V4 Macro Feature Engineering - 시장 Regime Detection & Inter-market Analysis

2021~2022년 같은 폭락장을 피하기 위한 매크로 피처:

1단계: 시장의 온도계 (Regime Detection)
- market_regime_score: KOSPI 200 이격도 (120일 MA 대비)
- size_spread: 대형주 vs 소형주 수익률 차이
- market_breadth: MA 위 섹터 개수

2단계: 종목의 '진짜 실력' (Sector Alpha)
- sector_relative_momentum: 종목 수익률 - 섹터 수익률
- sector_momentum_20d/60d: 섹터 모멘텀

3단계: 매크로 공포 레이더 (Inter-market Analysis)
- fear_index_delta: VKOSPI 5일 변화량
- dollar_impact: 달러선물 20일 모멘텀
- bond_stock_spread: 채권 vs 주식 수익률 스프레드
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class MacroFeatureEngineer:
    """V4: 매크로 Regime Detection + Inter-market Analysis"""

    # 핵심 지수 코드 매핑
    INDEX_CODES = {
        # Market indices
        'kospi_200': 'KOSPI_코스피_200',
        'kosdaq_150': 'KOSDAQ_코스닥_150',
        'kospi_large': 'KOSPI_코스피_대형주',
        'kospi_small': 'KOSPI_코스피_소형주',
        'kosdaq_large': 'KOSDAQ_코스닥_대형주',
        'kosdaq_small': 'KOSDAQ_코스닥_소형주',

        # KOSPI 200 섹터 지수들
        'kospi_200_it': 'KOSPI_코스피_200_정보기술',
        'kospi_200_finance': 'KOSPI_코스피_200_금융',
        'kospi_200_industrial': 'KOSPI_코스피_200_산업재',
        'kospi_200_consumer': 'KOSPI_코스피_200_경기소비재',
        'kospi_200_healthcare': 'KOSPI_코스피_200_헬스케어',
        'kospi_200_material': 'KOSPI_코스피_200_철강_소재',
        'kospi_200_energy': 'KOSPI_코스피_200_에너지_화학',
        'kospi_200_construct': 'KOSPI_코스피_200_건설',
        'kospi_200_heavy': 'KOSPI_코스피_200_중공업',
        'kospi_200_consumer_def': 'KOSPI_코스피_200_생활소비재',
        'kospi_200_comm': 'KOSPI_코스피_200_커뮤니케이션서비스',
    }

    DERIV_INDEX_CODES = {
        # Derivatives indices
        'vkospi': 'DERIV_옵션지수_코스피_200_변동성지수',
        'dollar_futures': 'DERIV_선물지수_미국달러선물지수',
        'kospi_200_futures': 'DERIV_선물지수_코스피_200_선물지수',
    }

    BOND_INDEX_CODES = {
        # Bond indices
        'govt_bond_prime': 'BOND_국고채프라임지수',
        'ktb_index': 'BOND_KTB_지수',
    }

    # V4 매크로 피처 리스트
    MACRO_FEATURES = [
        # 1단계: Regime Detection
        'market_regime_score',          # KOSPI 200 120일 MA 대비 이격도
        'kosdaq_regime_score',          # KOSDAQ 150 120일 MA 대비 이격도
        'size_spread',                  # 대형주 - 소형주 수익률
        'market_breadth',               # MA 위 섹터 비율 (0~1)
        'market_breadth_change',        # 시장 폭 변화

        # 2단계: Sector Alpha
        'sector_relative_momentum',     # 종목 21일 수익률 - 섹터 21일 수익률
        'sector_momentum_20d',          # 섹터 20일 모멘텀
        'sector_momentum_60d',          # 섹터 60일 모멘텀

        # 3단계: Inter-market Analysis
        'fear_index_delta',             # VKOSPI 5일 변화
        'fear_index_level',             # VKOSPI 절대 레벨
        'dollar_impact',                # 달러선물 20일 모멘텀
        'bond_stock_spread',            # 채권 - 주식 수익률 스프레드
        'yield_gap',                    # 주식 기대수익률 - 채권 수익률

        # 복합 피처
        'macro_risk_score',             # 종합 매크로 리스크 점수
        'regime_momentum_interaction',  # regime * momentum 상호작용
    ]

    def __init__(self, db_path: str = "krx_stock_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._market_data_cache = {}

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # =========================================================================
    # 데이터 로딩
    # =========================================================================

    def load_market_index_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """시장 지수 데이터 로드 (KOSPI/KOSDAQ)"""
        conn = self._get_connection()

        # 필요한 모든 지수 코드
        index_codes = list(self.INDEX_CODES.values())
        placeholders = ','.join(['?' for _ in index_codes])

        query = f"""
        SELECT
            index_code,
            date,
            closing_index as close,
            change_rate
        FROM index_daily_prices
        WHERE index_code IN ({placeholders})
          AND date >= ? AND date <= ?
        ORDER BY index_code, date
        """

        params = index_codes + [start_date, end_date]
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        self.logger.info(f"Loaded {len(df):,} market index records")
        return df

    def load_deriv_index_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """파생상품 지수 데이터 로드 (VKOSPI, 달러선물 등)"""
        conn = self._get_connection()

        index_codes = list(self.DERIV_INDEX_CODES.values())
        placeholders = ','.join(['?' for _ in index_codes])

        query = f"""
        SELECT
            index_code,
            date,
            closing_index as close,
            change_rate
        FROM deriv_index_daily
        WHERE index_code IN ({placeholders})
          AND date >= ? AND date <= ?
        ORDER BY index_code, date
        """

        params = index_codes + [start_date, end_date]
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        self.logger.info(f"Loaded {len(df):,} derivatives index records")
        return df

    def load_bond_index_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """채권 지수 데이터 로드"""
        conn = self._get_connection()

        index_codes = list(self.BOND_INDEX_CODES.values())
        placeholders = ','.join(['?' for _ in index_codes])

        query = f"""
        SELECT
            index_code,
            date,
            total_return_index as close,
            total_return_change as change_rate
        FROM bond_index_daily
        WHERE index_code IN ({placeholders})
          AND date >= ? AND date <= ?
        ORDER BY index_code, date
        """

        params = index_codes + [start_date, end_date]
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        self.logger.info(f"Loaded {len(df):,} bond index records")
        return df

    def load_sector_indices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """모든 KOSPI/KOSDAQ 섹터 지수 로드"""
        conn = self._get_connection()

        query = """
        SELECT
            i.index_code,
            i.current_name as name,
            i.index_class,
            p.date,
            p.closing_index as close,
            p.change_rate
        FROM indices i
        JOIN index_daily_prices p ON i.index_code = p.index_code
        WHERE p.date >= ? AND p.date <= ?
          AND (i.index_class = 'KOSPI' OR i.index_class = 'KOSDAQ')
        ORDER BY i.index_code, p.date
        """

        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()

        self.logger.info(f"Loaded {len(df):,} sector index records "
                         f"({df['index_code'].nunique()} sectors)")
        return df

    # =========================================================================
    # 1단계: 시장의 온도계 (Regime Detection)
    # =========================================================================

    def compute_regime_features(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        시장 Regime 피처 계산

        - market_regime_score: KOSPI 200 이격도 (120일 MA 대비)
        - size_spread: 대형주 vs 소형주 수익률 차이
        """
        self.logger.info("Computing regime features...")

        # Pivot to wide format (각 지수별 컬럼)
        pivot = market_df.pivot(index='date', columns='index_code', values='close')

        result = pd.DataFrame(index=pivot.index)

        # === Market Regime Score ===
        # (현재가 / 120일 이평선) - 1
        kospi_200 = self.INDEX_CODES['kospi_200']
        kosdaq_150 = self.INDEX_CODES['kosdaq_150']

        if kospi_200 in pivot.columns:
            ma_120 = pivot[kospi_200].rolling(120, min_periods=60).mean()
            result['market_regime_score'] = (pivot[kospi_200] / ma_120) - 1

        if kosdaq_150 in pivot.columns:
            ma_120_kq = pivot[kosdaq_150].rolling(120, min_periods=60).mean()
            result['kosdaq_regime_score'] = (pivot[kosdaq_150] / ma_120_kq) - 1

        # === Size Spread ===
        # 대형주 수익률 - 소형주 수익률 (20일 기준)
        large_code = self.INDEX_CODES['kospi_large']
        small_code = self.INDEX_CODES['kospi_small']

        if large_code in pivot.columns and small_code in pivot.columns:
            large_ret = pivot[large_code].pct_change(20)
            small_ret = pivot[small_code].pct_change(20)
            result['size_spread'] = large_ret - small_ret
            # 양수면 대형주 강세 (불안한 장), 음수면 소형주 강세 (불장)

        result = result.reset_index()
        return result

    def compute_market_breadth(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        시장 폭(Market Breadth) 계산

        - 20일 이동평균선 위에 있는 섹터의 비율
        - 에너지 측정: 0.5 미만이면 약세장, 0.7 이상이면 강세장
        """
        self.logger.info("Computing market breadth...")

        # 각 섹터별로 20일 MA 계산
        sector_df = sector_df.sort_values(['index_code', 'date']).copy()
        sector_df['ma_20'] = sector_df.groupby('index_code')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        sector_df['above_ma'] = (sector_df['close'] > sector_df['ma_20']).astype(int)

        # 날짜별 MA 위 섹터 비율
        breadth = sector_df.groupby('date').agg(
            sectors_above_ma=('above_ma', 'sum'),
            total_sectors=('above_ma', 'count')
        ).reset_index()

        breadth['market_breadth'] = breadth['sectors_above_ma'] / breadth['total_sectors']

        # 시장 폭 변화 (5일)
        breadth['market_breadth_change'] = breadth['market_breadth'].diff(5)

        return breadth[['date', 'market_breadth', 'market_breadth_change']]

    # =========================================================================
    # 2단계: 종목의 '진짜 실력' (Sector Alpha)
    # =========================================================================

    def compute_sector_momentum(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """
        섹터별 모멘텀 계산

        종목 수준에서 사용할 섹터 모멘텀 피처를 미리 계산
        """
        self.logger.info("Computing sector momentum...")

        sector_df = sector_df.sort_values(['index_code', 'date']).copy()

        # 섹터별 모멘텀
        for period in [20, 60]:
            sector_df[f'sector_mom_{period}d'] = sector_df.groupby('index_code')['close'].transform(
                lambda x: x.pct_change(period)
            )

        # 21일 수익률 (종목 비교용)
        sector_df['sector_return_21d'] = sector_df.groupby('index_code')['close'].transform(
            lambda x: x.pct_change(21)
        )

        return sector_df

    def create_sector_mapping(self) -> Dict[str, str]:
        """
        종목 섹터 → 섹터 지수 매핑 생성

        종목의 sector_type을 해당 섹터 지수 코드로 매핑
        """
        # 기본 섹터 매핑 (stocks.current_sector_type → index_code)
        mapping = {
            # KOSPI 섹터
            '전기전자': 'KOSPI_코스피_200_정보기술',
            '금융': 'KOSPI_코스피_200_금융',
            '서비스업': 'KOSPI_코스피_200_경기소비재',
            '화학': 'KOSPI_코스피_200_에너지_화학',
            '철강금속': 'KOSPI_코스피_200_철강_소재',
            '기계': 'KOSPI_코스피_200_산업재',
            '운수장비': 'KOSPI_코스피_200_경기소비재',
            '의약품': 'KOSPI_코스피_200_헬스케어',
            '건설업': 'KOSPI_코스피_200_건설',
            '음식료품': 'KOSPI_코스피_200_생활소비재',
            '유통업': 'KOSPI_코스피_200_경기소비재',
            '통신업': 'KOSPI_코스피_200_커뮤니케이션서비스',
            '전기가스업': 'KOSPI_코스피_200_에너지_화학',
            '운수창고업': 'KOSPI_코스피_200_산업재',
            '섬유의복': 'KOSPI_코스피_200_경기소비재',
            '종이목재': 'KOSPI_코스피_200_철강_소재',
            '비금속광물': 'KOSPI_코스피_200_철강_소재',
            '의료정밀': 'KOSPI_코스피_200_헬스케어',
            '보험': 'KOSPI_코스피_200_금융',
            '증권': 'KOSPI_코스피_200_금융',
            '은행': 'KOSPI_코스피_200_금융',

            # KOSDAQ 섹터 (기본 KOSDAQ 150으로 매핑)
            'IT 서비스': 'KOSDAQ_코스닥_150_정보기술',
            '제약': 'KOSDAQ_코스닥_150_헬스케어',
            '소프트웨어': 'KOSDAQ_코스닥_150_정보기술',
            '반도체': 'KOSDAQ_코스닥_150_정보기술',
            '디스플레이': 'KOSDAQ_코스닥_150_정보기술',
            '바이오': 'KOSDAQ_코스닥_150_헬스케어',
        }
        return mapping

    # =========================================================================
    # 3단계: 매크로 공포 레이더 (Inter-market Analysis)
    # =========================================================================

    def compute_fear_features(self, deriv_df: pd.DataFrame) -> pd.DataFrame:
        """
        공포 지수 피처 계산

        - fear_index_delta: VKOSPI 5일 변화량
        - fear_index_level: VKOSPI 절대 레벨 (높을수록 공포)
        """
        self.logger.info("Computing fear index features...")

        vkospi_code = self.DERIV_INDEX_CODES['vkospi']
        vkospi_df = deriv_df[deriv_df['index_code'] == vkospi_code].copy()

        if len(vkospi_df) == 0:
            self.logger.warning("No VKOSPI data found")
            return pd.DataFrame(columns=['date', 'fear_index_delta', 'fear_index_level'])

        vkospi_df = vkospi_df.sort_values('date')
        vkospi_df['fear_index_level'] = vkospi_df['close']
        vkospi_df['fear_index_delta'] = vkospi_df['close'].diff(5)

        return vkospi_df[['date', 'fear_index_delta', 'fear_index_level']]

    def compute_dollar_impact(self, deriv_df: pd.DataFrame) -> pd.DataFrame:
        """
        달러 영향 피처 계산

        - dollar_impact: 미국달러선물지수 20일 모멘텀
        - 달러 급등 = 외인 자금 이탈 신호
        """
        self.logger.info("Computing dollar impact features...")

        dollar_code = self.DERIV_INDEX_CODES['dollar_futures']
        dollar_df = deriv_df[deriv_df['index_code'] == dollar_code].copy()

        if len(dollar_df) == 0:
            self.logger.warning("No dollar futures data found")
            return pd.DataFrame(columns=['date', 'dollar_impact'])

        dollar_df = dollar_df.sort_values('date')
        dollar_df['dollar_impact'] = dollar_df['close'].pct_change(20)

        return dollar_df[['date', 'dollar_impact']]

    def compute_bond_stock_spread(self, bond_df: pd.DataFrame,
                                    market_df: pd.DataFrame) -> pd.DataFrame:
        """
        채권-주식 스프레드 계산

        - bond_stock_spread: 국고채 수익률 - KOSPI 200 수익률
        - 양수면 안전자산 선호 (Risk-off)
        """
        self.logger.info("Computing bond-stock spread...")

        # 채권 수익률 (20일)
        bond_code = self.BOND_INDEX_CODES['govt_bond_prime']
        bond_only = bond_df[bond_df['index_code'] == bond_code].copy()

        if len(bond_only) == 0:
            self.logger.warning("No bond index data found")
            return pd.DataFrame(columns=['date', 'bond_stock_spread'])

        bond_only = bond_only.sort_values('date')
        bond_only['bond_return_20d'] = bond_only['close'].pct_change(20)

        # KOSPI 200 수익률 (20일)
        kospi_code = self.INDEX_CODES['kospi_200']
        kospi_only = market_df[market_df['index_code'] == kospi_code].copy()
        kospi_only = kospi_only.sort_values('date')
        kospi_only['kospi_return_20d'] = kospi_only['close'].pct_change(20)

        # Merge
        merged = pd.merge(
            bond_only[['date', 'bond_return_20d']],
            kospi_only[['date', 'kospi_return_20d']],
            on='date',
            how='inner'
        )

        merged['bond_stock_spread'] = merged['bond_return_20d'] - merged['kospi_return_20d']

        return merged[['date', 'bond_stock_spread']]

    # =========================================================================
    # 종목 수준 피처 결합
    # =========================================================================

    def merge_macro_to_stocks(self, stock_df: pd.DataFrame,
                               macro_features: pd.DataFrame,
                               sector_features: pd.DataFrame) -> pd.DataFrame:
        """
        매크로 피처를 종목 데이터에 결합

        Args:
            stock_df: 종목별 일별 데이터 (stock_code, date, sector 포함)
            macro_features: 날짜별 매크로 피처
            sector_features: 섹터별 모멘텀 피처

        Returns:
            매크로 피처가 결합된 종목 데이터
        """
        self.logger.info("Merging macro features to stock data...")

        # 1. 날짜 기준 매크로 피처 조인
        merged = pd.merge(stock_df, macro_features, on='date', how='left')

        # 2. 섹터 매핑 생성
        sector_mapping = self.create_sector_mapping()

        # 3. 종목별 섹터 지수 매칭 및 피처 추가
        if 'sector' in merged.columns:
            # 섹터 지수 코드 매핑
            merged['sector_index_code'] = merged['sector'].map(sector_mapping)

            # 섹터 피처 준비
            sector_features_wide = sector_features.pivot(
                index='date',
                columns='index_code',
                values=['sector_mom_20d', 'sector_mom_60d', 'sector_return_21d']
            )

            # 종목별로 해당 섹터의 피처 매칭
            for idx, row in merged.iterrows():
                sector_idx = row.get('sector_index_code')
                date_val = row['date']

                if pd.notna(sector_idx) and date_val in sector_features_wide.index:
                    try:
                        merged.loc[idx, 'sector_momentum_20d'] = \
                            sector_features_wide.loc[date_val, ('sector_mom_20d', sector_idx)]
                        merged.loc[idx, 'sector_momentum_60d'] = \
                            sector_features_wide.loc[date_val, ('sector_mom_60d', sector_idx)]

                        # Sector Relative Momentum 계산
                        if 'mom_21d' in merged.columns:
                            sector_ret = sector_features_wide.loc[
                                date_val, ('sector_return_21d', sector_idx)]
                            if pd.notna(sector_ret):
                                merged.loc[idx, 'sector_relative_momentum'] = \
                                    row.get('mom_21d', 0) - sector_ret
                    except KeyError:
                        pass

        return merged

    def compute_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        복합 매크로 피처 계산

        - macro_risk_score: 종합 리스크 점수
        - regime_momentum_interaction: regime과 모멘텀 상호작용
        """
        self.logger.info("Computing composite macro features...")

        df = df.copy()

        # === Macro Risk Score ===
        # 높을수록 위험 (음수 regime + 높은 fear + 양수 dollar + 양수 bond spread)
        risk_components = []

        if 'market_regime_score' in df.columns:
            # Regime이 음수면 리스크 높음
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
            # Regime이 양수일 때 모멘텀 효과 증폭
            df['regime_momentum_interaction'] = (
                df['market_regime_score'].clip(-0.2, 0.2) *
                df['mom_20d'].clip(-0.5, 0.5)
            )
        else:
            df['regime_momentum_interaction'] = 0

        return df

    # =========================================================================
    # 메인 파이프라인
    # =========================================================================

    def prepare_macro_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        전체 매크로 피처 파이프라인 (날짜 수준)

        Returns:
            날짜별 매크로 피처 DataFrame
        """
        # 1년 버퍼 (MA 계산용)
        buffer_start = str(int(start_date[:4]) - 1) + start_date[4:]

        self.logger.info(f"Preparing macro features from {buffer_start} to {end_date}")

        # 데이터 로드
        market_df = self.load_market_index_data(buffer_start, end_date)
        deriv_df = self.load_deriv_index_data(buffer_start, end_date)
        bond_df = self.load_bond_index_data(buffer_start, end_date)
        sector_df = self.load_sector_indices(buffer_start, end_date)

        # 1단계: Regime Detection
        regime_df = self.compute_regime_features(market_df)
        breadth_df = self.compute_market_breadth(sector_df)

        # 3단계: Inter-market Analysis
        fear_df = self.compute_fear_features(deriv_df)
        dollar_df = self.compute_dollar_impact(deriv_df)
        spread_df = self.compute_bond_stock_spread(bond_df, market_df)

        # 모든 피처 병합
        result = regime_df.copy()

        for df in [breadth_df, fear_df, dollar_df, spread_df]:
            if len(df) > 0:
                result = pd.merge(result, df, on='date', how='outer')

        # 날짜 필터링
        result = result[result['date'] >= start_date].copy()
        result = result.sort_values('date')

        # Forward fill (주말/휴일)
        feature_cols = [c for c in result.columns if c != 'date']
        result[feature_cols] = result[feature_cols].ffill()

        self.logger.info(f"Prepared {len(result)} days of macro features")
        return result

    def add_macro_features_to_stocks(self, stock_df: pd.DataFrame,
                                      start_date: str, end_date: str) -> pd.DataFrame:
        """
        종목 데이터에 매크로 피처 추가

        Args:
            stock_df: 종목별 일별 데이터 (date, stock_code, sector 필수)

        Returns:
            매크로 피처가 추가된 종목 데이터
        """
        # 매크로 피처 준비
        macro_df = self.prepare_macro_features(start_date, end_date)

        # 섹터 피처 준비
        buffer_start = str(int(start_date[:4]) - 1) + start_date[4:]
        sector_df = self.load_sector_indices(buffer_start, end_date)
        sector_features = self.compute_sector_momentum(sector_df)

        # 종목에 결합
        merged = self.merge_macro_to_stocks(stock_df, macro_df, sector_features)

        # 복합 피처 계산
        merged = self.compute_composite_features(merged)

        return merged


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 테스트 실행
    macro_eng = MacroFeatureEngineer('krx_stock_data.db')
    macro_df = macro_eng.prepare_macro_features(
        start_date='20240101',
        end_date='20260128'
    )

    print("\n=== Macro Features Summary ===")
    print(f"Date range: {macro_df['date'].min()} to {macro_df['date'].max()}")
    print(f"Total days: {len(macro_df)}")
    print("\nFeature statistics:")
    print(macro_df.describe())
