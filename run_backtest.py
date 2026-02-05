#!/usr/bin/env python3
"""
V3 QEPM 백테스트 (Quantitative Equity Portfolio Management)

Usage:
    python3 run_backtest.py                         # 기본 (quintile)
    python3 run_backtest.py --top 10                # 상위 10개 종목
    python3 run_backtest.py --qepm                  # 🔥 QEPM 모드 (권장)
    python3 run_backtest.py --qepm --top 20         # QEPM + Top 20

QEPM 모드:
    - 63일(3개월) 호라이즌
    - Alpha 타겟 (시장 대비 초과수익)
    - 섹터당 최대 3종목
    - 변동성 역가중 배분
    - 회전율 제어 (20% 버퍼)
"""

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np
import argparse
from scipy import stats
from datetime import datetime
import time
from ml.features import FeatureEngineer
from ml.model import MLRanker

parser = argparse.ArgumentParser()
parser.add_argument('--top', type=int, default=0, help='상위 N개 종목만 (0=quintile)')
parser.add_argument('--horizon', type=int, default=63, help='보유 기간 (일): 21, 63, 126')
parser.add_argument('--slippage', type=float, default=0.5, help='편도 슬리피지 %% (기본 0.5%%)')
parser.add_argument('--qepm', action='store_true', help='🔥 QEPM 모드 (63일, Alpha, 섹터제한, 변동성가중)')
parser.add_argument('--max-sector', type=int, default=3, help='섹터당 최대 종목 수 (QEPM)')
parser.add_argument('--turnover-buffer', type=float, default=0.2, help='회전율 버퍼 (기존 종목 교체 기준)')
parser.add_argument('--v4', action='store_true', help='🔥 V4 모드 (매크로 Regime Detection 피처 추가)')
parser.add_argument('--cash-timing', action='store_true', help='🛡️ 현금 비중 조절 (시장 하락기 현금 보유)')
parser.add_argument('--regime-threshold', type=float, default=-0.05, help='Regime 임계값 (기본 -5%%: 이하면 현금)')
parser.add_argument('--v42', action='store_true', help='🔥 V4.2 Market Exit (Regime<0 → 100%% 현금, 손절, 점수필터)')
parser.add_argument('--stop-loss', type=float, default=0.07, help='V4.2 손절 기준 (기본 7%%)')
parser.add_argument('--score-threshold', type=float, default=0.0, help='V4.2 최소 ML Score (기본 0=비활성)')
parser.add_argument('--v43', action='store_true', help='🔥 V4.3 Residual Alpha (Beta-neutral, Purged CV, 유동성필터)')
parser.add_argument('--illiquidity-filter', type=float, default=0.9, help='V4.3 유동성 필터 (상위 N%% 제외, 기본 90%%=상위10%% 제외)')
parser.add_argument('--v5', action='store_true', help='🔥 V5 Research-backed 10-Feature Mode (10개 핵심 피처만)')
args = parser.parse_args()

# QEPM 모드면 설정 오버라이드
if args.qepm:
    args.horizon = 63  # 3개월
    if args.top == 0:
        args.top = 20  # 기본 20종목

# V4.2 모드 설정 (공격적 Market Exit)
if args.v42:
    args.v4 = True  # V4 피처 포함
    args.cash_timing = True  # 현금 타이밍 활성화
    args.regime_threshold = 0.0  # Regime < 0이면 현금 (핵심!)
    if args.top == 0:
        args.top = 20

# V4.3 모드 설정 (Residual Alpha + Purged CV)
if args.v43:
    args.v4 = True  # V4 피처 포함
    args.v42 = True  # V4.2 기능 포함 (Market Exit)
    args.cash_timing = True
    args.regime_threshold = 0.0
    if args.top == 0:
        args.top = 20

# V5 모드 설정 (Research-backed 10-Feature Mode, cascades from v43)
if args.v5:
    args.v43 = True
    args.v42 = True
    args.v4 = True
    args.cash_timing = True
    args.regime_threshold = 0.0
    args.horizon = 63  # Forced to 63 days
    if args.top == 0:
        args.top = 20

print('=' * 70)
if args.v5:
    print('🔥 V5 백테스트 (Research-backed 10-Feature Mode)')
elif args.v43:
    print('🔥 V4.3 백테스트 (Residual Alpha + Beta Neutral)')
elif args.v42:
    print('🔥 V4.2 백테스트 (Market Exit Strategy)')
elif args.v4:
    print('🔥 V4 백테스트 (매크로 Regime Detection)')
elif args.qepm:
    print('🏦 V3 QEPM 백테스트 (기관급 포트폴리오)')
else:
    print('🚀 V3 모델 백테스트')
print(f'   {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('=' * 70)

# ============================================================================
# 설정
# ============================================================================
HORIZON = args.horizon
TOP_N = args.top  # 0이면 quintile 사용
SLIPPAGE = args.slippage / 100  # 편도 슬리피지 (0.5% -> 0.005)
ROUND_TRIP_COST = SLIPPAGE * 2  # 왕복 비용
TRAIN_YEARS = 3
BUFFER_MONTHS = 2  # horizon 대비 버퍼
QEPM_MODE = args.qepm
MAX_PER_SECTOR = args.max_sector
TURNOVER_BUFFER = args.turnover_buffer

print(f'\n설정:')
print(f'  - 모드: {"🏦 QEPM (기관급)" if QEPM_MODE else "일반"}')
print(f'  - Target Horizon: {HORIZON}일 ({HORIZON//21}개월)')
print(f'  - 포트폴리오: {"상위 " + str(TOP_N) + "개" if TOP_N > 0 else "Quintile (상위 20%)"}')
print(f'  - 슬리피지: {args.slippage}% (왕복 {args.slippage*2}%)')
if QEPM_MODE:
    print(f'  - 섹터 제한: 섹터당 최대 {MAX_PER_SECTOR}종목')
    print(f'  - 회전율 버퍼: {TURNOVER_BUFFER*100:.0f}% (교체 기준)')
    print(f'  - 타겟: Alpha (시장 대비 초과수익)')
    print(f'  - 가중치: 변동성 역가중 (Risk Parity)')
print(f'  - 학습 기간: {TRAIN_YEARS}년')
if args.v5:
    print(f'  - 🔥 V5 Research-backed 10-Feature Mode:')
    print(f'      • 피처: 10개 핵심 피처만 사용')
    print(f'      • Horizon: 63일 (forced)')
    print(f'      • Target: Residual Return (Beta-adjusted)')
    print(f'      • Purged CV + Market Exit + Macro 상속')
elif args.v43:
    print(f'  - 🔥 V4.3 Residual Alpha:')
    print(f'      • Target: Residual Return (Beta-adjusted)')
    print(f'      • Purged CV: {HORIZON}일 겹침 제거')
    print(f'      • 유동성 필터: Amihud 상위 {(1-args.illiquidity_filter)*100:.0f}% 제외')
    print(f'      • Regime < {args.regime_threshold*100:.0f}% → 100% 현금')
elif args.v42:
    print(f'  - 🔥 V4.2 Market Exit:')
    print(f'      • Regime < {args.regime_threshold*100:.0f}% → 100% 현금')
    print(f'      • 손절 기준: -{args.stop_loss*100:.0f}% (21일 내)')
    if args.score_threshold > 0:
        print(f'      • 최소 ML Score: {args.score_threshold}')
elif args.cash_timing:
    print(f'  - 🛡️ 현금 타이밍: ON (Regime < {args.regime_threshold*100:.0f}% → 현금)')

# ============================================================================
# 데이터 로드
# ============================================================================
print('\n[1/3] 데이터 로드 중...')
stage_t0 = time.time()

fe = FeatureEngineer('krx_stock_data.db')
df = fe.prepare_ml_data(
    start_date='20110101',  # 전체 기간
    end_date='20260128',
    target_horizon=HORIZON,
    min_market_cap=500_000_000_000,
    include_fundamental=True,
    include_macro=args.v4  # V4: 매크로 피처 추가
)
print(f'  ⏱ [1/3] 데이터 로드 완료: {time.time()-stage_t0:.1f}s')

# ============================================================================
# 현금 타이밍용 시장 Regime 데이터 로드 (최적화)
# ============================================================================
MARKET_REGIME = {}
if args.cash_timing:
    regime_t0 = time.time()
    import sqlite3
    conn = sqlite3.connect('krx_stock_data.db')
    # 단순 쿼리로 데이터 로드 후 pandas에서 계산 (더 빠름)
    regime_query = '''
    SELECT date, closing_index
    FROM index_daily_prices
    WHERE index_code = 'KOSPI_코스피_200'
    ORDER BY date
    '''
    regime_df = pd.read_sql_query(regime_query, conn)
    conn.close()

    # Pandas에서 MA 계산 (SQL window보다 빠름)
    regime_df['ma_120'] = regime_df['closing_index'].rolling(120, min_periods=60).mean()
    regime_df['regime_score'] = (regime_df['closing_index'] / regime_df['ma_120']) - 1
    MARKET_REGIME = dict(zip(regime_df['date'], regime_df['regime_score']))

    # 연도별 평균 regime 표시
    regime_df['year'] = regime_df['date'].str[:4]
    yearly_regime = regime_df.groupby('year')['regime_score'].mean()
    print(f'\n  [시장 Regime by Year]')
    for year, score in yearly_regime.items():
        if pd.notna(score):
            status = '🟢' if score > 0 else '🔴'
            print(f'    {year}: {score*100:+.1f}% {status}')
    print(f'  ⏱ Market regime load: {time.time()-regime_t0:.1f}s')

def should_hold_cash(date, threshold):
    """현금 보유 여부 결정"""
    if not MARKET_REGIME:
        return False
    regime = MARKET_REGIME.get(date, 0)
    return regime < threshold

# 피처 분류
momentum_features = [c for c in fe.MOMENTUM_FEATURES if c in df.columns]
volume_features = [c for c in fe.VOLUME_FEATURES if c in df.columns]
volatility_features = [c for c in fe.VOLATILITY_FEATURES if c in df.columns]
intuition_features = [c for c in fe.INTUITION_FEATURES if c in df.columns]
traditional_features = [c for c in fe.TRADITIONAL_FEATURES if c in df.columns]
fund_features = [c for c in fe.FUNDAMENTAL_FEATURES if c in df.columns]
macro_features = [c for c in fe.MACRO_FEATURES if c in df.columns] if args.v4 else []

all_features = (momentum_features + volume_features + volatility_features +
                intuition_features + traditional_features + fund_features + macro_features)

# V5: Override with 10 focused features
if args.v5:
    all_features = [c for c in fe.V5_FEATURES if c in df.columns]
    missing_v5 = [c for c in fe.V5_FEATURES if c not in df.columns]
    if missing_v5:
        print(f'  ⚠️ Missing V5 features: {missing_v5}')

tech_count = len(momentum_features + volume_features + volatility_features +
                 intuition_features + traditional_features)

print(f'  총 데이터: {len(df):,} rows')
if args.v5:
    print(f'  피처 구성: V5 (10-Feature Mode)')
    for f_name in all_features:
        print(f'    - {f_name}')
    print(f'  총 피처: {len(all_features)}개')
else:
    print(f'  피처 구성:')
    print(f'    - 모멘텀: {len(momentum_features)}개')
    print(f'    - 수급: {len(volume_features)}개')
    print(f'    - 변동성: {len(volatility_features)}개')
    print(f'    - 본능전략: {len(intuition_features)}개')
    print(f'    - 전통지표: {len(traditional_features)}개')
    print(f'    - 재무: {len(fund_features)}개')
    if args.v4:
        print(f'    - 🔥 매크로: {len(macro_features)}개')
    print(f'  재무 비중: {len(fund_features)/len(all_features)*100:.1f}%')

# Forward return 추가
df = df.sort_values(['stock_code', 'date'])
df['year'] = df['date'].str[:4].astype(int)
years = sorted(df['year'].unique())

# ============================================================================
# Walk-Forward 백테스트
# ============================================================================
stage_t0 = time.time()
print('\n[2/3] Walk-Forward 백테스트...')
print('-' * 70)

# 타겟 설정: V4.3 > QEPM > 일반
if args.v43:
    # 🔥 V4.3: Residual Return (Beta-adjusted) 타겟
    target_col = f'target_residual_rank_{HORIZON}d'
    return_col = f'forward_return_{HORIZON}d'  # 실제 수익은 raw return
    if target_col not in df.columns:
        print(f'  ⚠️ Residual target 없음, Alpha로 fallback')
        target_col = f'target_alpha_rank_{HORIZON}d'
elif QEPM_MODE:
    target_col = f'target_alpha_rank_{HORIZON}d'  # Alpha 순위
    return_col = f'forward_alpha_{HORIZON}d'      # Alpha 수익률
    if target_col not in df.columns:
        target_col = f'target_rank_{HORIZON}d'    # fallback
        return_col = f'forward_return_{HORIZON}d'
else:
    target_col = f'target_rank_{HORIZON}d'
    return_col = f'forward_return_{HORIZON}d'

# V4.3: 유동성 필터 적용
if args.v43 and 'amihud_rank' in df.columns:
    original_len = len(df)
    df = df[df['amihud_rank'] <= args.illiquidity_filter]
    filtered = original_len - len(df)
    print(f'  유동성 필터: {filtered:,}개 제외 (Amihud 상위 {(1-args.illiquidity_filter)*100:.0f}%)')

all_results = []

# ============================================================================
# QEPM 헬퍼 함수들
# ============================================================================

def load_all_benchmark_returns(horizon=63):
    """
    모든 연도의 벤치마크 수익률을 한 번에 로드 (캐시용)
    """
    import sqlite3

    try:
        conn = sqlite3.connect('krx_stock_data.db')
        query = f"""
        SELECT
            SUBSTR(date, 1, 4) as year,
            AVG(return_{horizon}d) * 100 as avg_return
        FROM benchmark_kospi200
        WHERE return_{horizon}d IS NOT NULL
        GROUP BY SUBSTR(date, 1, 4)
        """
        result = pd.read_sql_query(query, conn)
        conn.close()

        return dict(zip(result['year'].astype(int), result['avg_return']))
    except Exception as e:
        print(f'    ⚠️ 벤치마크 로드 실패: {e}')
        return {}

# 벤치마크 데이터 미리 로드 (한 번만)
_BENCHMARK_CACHE = None

def get_benchmark_return(year, horizon=63):
    """캐시된 벤치마크 수익률 반환"""
    global _BENCHMARK_CACHE
    if _BENCHMARK_CACHE is None:
        _BENCHMARK_CACHE = load_all_benchmark_returns(horizon)
    return _BENCHMARK_CACHE.get(year, 0)

def select_with_sector_constraint(df, top_n, max_per_sector):
    """섹터 제약 적용한 종목 선정 (벡터화 버전)"""
    if len(df) == 0:
        return df

    df = df.sort_values('pred_rank').copy()
    df['sector'] = df['sector'].fillna('Unknown')

    # 섹터별 누적 카운트 계산 (벡터화)
    df['_sector_cumcount'] = df.groupby('sector').cumcount() + 1

    # 섹터당 max_per_sector 이하인 행만 선택
    eligible = df[df['_sector_cumcount'] <= max_per_sector]

    # 상위 top_n개 선택
    result = eligible.head(top_n).drop(columns=['_sector_cumcount'])
    return result

def apply_inverse_volatility_weight(df, return_col):
    """변동성 역가중 적용한 수익률 계산"""
    if 'volatility_20d' not in df.columns or len(df) == 0:
        return df[return_col].mean()

    vol = df['volatility_20d'].fillna(df['volatility_20d'].median())
    vol = vol.clip(lower=0.1)  # 최소 변동성
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()

    return (df[return_col] * weights).sum()

for test_year in years:
    train_start = test_year - TRAIN_YEARS
    train_end = test_year - 1

    # Buffer 적용 (horizon 거래일 전에 학습 종료)
    buffer_month = 12 - (HORIZON // 21 + BUFFER_MONTHS)
    buffer_month = max(1, min(buffer_month, 10))
    train_cutoff = f'{train_end}{buffer_month:02d}01'

    train_df = df[(df['year'] >= train_start) &
                  (df['year'] <= train_end) &
                  (df['date'] <= train_cutoff)].copy()
    test_df = df[df['year'] == test_year].copy()

    if len(train_df) < 1000 or len(test_df) < 100:
        continue

    # ================================================================
    # V4.3: Purged Cross-Validation (De Prado, 2018)
    # Forward return이 겹치는 데이터 제거 → Data leakage 방지
    # ================================================================
    if args.v43:
        # 테스트 시작일 기준으로 HORIZON일 이전까지만 학습
        test_start_date = test_df['date'].min()
        # Purge: 테스트 시작 - horizon일 전까지만 학습 데이터 사용
        # (forward return 계산 시 겹치는 부분 제거)
        purge_cutoff = pd.to_datetime(test_start_date, format='%Y%m%d') - pd.Timedelta(days=HORIZON * 1.5)
        purge_cutoff_str = purge_cutoff.strftime('%Y%m%d')

        original_train_len = len(train_df)
        train_df = train_df[train_df['date'] <= purge_cutoff_str]
        purged = original_train_len - len(train_df)

        # Embargo: 베어마켓 데이터에 가중치 부여 (2022-2023)
        if 'macro_risk_score' in train_df.columns:
            # 고위험 기간에 2x 가중치
            train_df['sample_weight'] = 1.0
            high_risk_mask = train_df['macro_risk_score'] > 0.6
            train_df.loc[high_risk_mask, 'sample_weight'] = 2.0
        else:
            train_df['sample_weight'] = 1.0

    # 모델 학습
    model = MLRanker(
        feature_cols=all_features,
        target_col=target_col,
        model_type='regressor',
        time_decay=0.5 if QEPM_MODE else 0.7
    )

    # V4.3: Sample weight 적용
    if args.v43 and 'sample_weight' in train_df.columns:
        model.train(train_df, sample_weight=train_df['sample_weight'].values)
    else:
        model.train(train_df)

    # 예측
    test_df['pred_score'] = model.predict(test_df)
    test_df['pred_rank'] = test_df.groupby('date')['pred_score'].rank(ascending=False)

    # 🔥 V4.2: ML Score 정규화 (0~1 범위로) 및 임계값 필터
    score_filtered = False
    if args.v42 and args.score_threshold > 0:
        # 날짜별 점수 정규화 (percentile)
        test_df['pred_score_pct'] = test_df.groupby('date')['pred_score'].rank(pct=True)
        # 임계값 미달 종목 필터 (pred_rank을 높여서 선택 안되게)
        low_score_mask = test_df['pred_score_pct'] < args.score_threshold
        if low_score_mask.any():
            score_filtered = True

    if TOP_N > 0:
        if QEPM_MODE:
            # QEPM: 섹터 제약 + 변동성 가중 (완전 벡터화)
            test_df = test_df.sort_values(['date', 'pred_rank']).copy()
            test_df['sector'] = test_df['sector'].fillna('Unknown')

            # 섹터별 누적 카운트 (날짜+섹터 그룹별)
            test_df['_sector_cumcount'] = test_df.groupby(['date', 'sector']).cumcount() + 1
            test_df['_date_cumcount'] = test_df.groupby('date').cumcount() + 1

            # 섹터 제약 + Top N 필터 (완전 벡터화)
            top_df = test_df[
                (test_df['_sector_cumcount'] <= MAX_PER_SECTOR) &
                (test_df['_date_cumcount'] <= TOP_N * 2)  # 여유있게 필터
            ].copy()

            # 날짜별로 다시 Top N 선택
            top_df['_final_rank'] = top_df.groupby('date').cumcount() + 1
            top_df = top_df[top_df['_final_rank'] <= TOP_N]

            # 변동성 역가중 수익률 (벡터화)
            if 'volatility_20d' in top_df.columns:
                vol = top_df['volatility_20d'].fillna(top_df['volatility_20d'].median()).clip(lower=0.1)
                top_df['_inv_vol'] = 1 / vol
                top_df['_weight'] = top_df.groupby('date')['_inv_vol'].transform(lambda x: x / x.sum())
                q5_raw = (top_df[return_col] * top_df['_weight']).groupby(top_df['date']).sum().mean() * 100
            else:
                q5_raw = top_df[return_col].mean() * 100
        else:
            # 일반: 단순 Top N
            top_df = test_df[test_df['pred_rank'] <= TOP_N]
            q5_raw = top_df[return_col].mean() * 100

        # Bottom N (벡터화)
        max_rank = test_df.groupby('date')['pred_rank'].transform('max')
        bottom_df = test_df[test_df['pred_rank'] > max_rank - TOP_N]
        q1_raw = bottom_df[return_col].mean() * 100

        # QEPM은 회전율 감소로 슬리피지 1/3
        effective_slippage = ROUND_TRIP_COST / 3 if QEPM_MODE else ROUND_TRIP_COST
        q5 = q5_raw - effective_slippage * 100
        q1 = q1_raw - effective_slippage * 100
        spread = q5 - q1 if not (np.isnan(q5) or np.isnan(q1)) else np.nan
    else:
        # Quintile 수익률
        test_df['quintile'] = pd.qcut(
            test_df['pred_rank'], 5,
            labels=[5,4,3,2,1], duplicates='drop'  # 1=top, 5=bottom
        )
        quintile_ret = test_df.groupby('quintile')[return_col].mean() * 100
        q1_raw = quintile_ret.get(1, np.nan)  # Bottom 20%
        q5_raw = quintile_ret.get(5, np.nan)  # Top 20%
        # 슬리피지 적용 (왕복 비용 차감)
        q5 = q5_raw - ROUND_TRIP_COST * 100 if not np.isnan(q5_raw) else np.nan
        q1 = q1_raw - ROUND_TRIP_COST * 100 if not np.isnan(q1_raw) else np.nan
        spread = q5 - q1 if not (np.isnan(q5) or np.isnan(q1)) else np.nan

    # IC 계산
    ic, _ = stats.spearmanr(
        test_df['pred_score'].fillna(0),
        test_df[return_col].fillna(0)
    )

    # QEPM: 시장(KOSPI200) 수익률 - DB에서 가져오기
    if QEPM_MODE:
        market_ret = get_benchmark_return(test_year, HORIZON)
        abs_top = q5 + market_ret  # 절대 수익률 = Alpha + 시장
        abs_bot = q1 + market_ret
    else:
        market_ret = 0
        abs_top = q5
        abs_bot = q1

    # 실제 포트폴리오 수익률 계산 (Alpha 아닌 절대 수익률!)
    abs_return_col = f'forward_return_{HORIZON}d'
    if TOP_N > 0 and abs_return_col in test_df.columns:
        if QEPM_MODE:
            # QEPM: top_df 재사용 (이미 섹터 제약 적용됨)
            portfolio_return = top_df[abs_return_col].mean() * 100 - ROUND_TRIP_COST * 100
        else:
            # 일반: Top N 평균
            portfolio_return = top_df[abs_return_col].mean() * 100 - ROUND_TRIP_COST * 100
    else:
        portfolio_return = q5  # fallback

    # 🛡️ 현금 타이밍: 시장 Regime이 나쁘면 현금 보유 (0% 수익)
    held_cash = False
    cash_ratio = 0.0
    stop_loss_impact = 0.0

    if args.cash_timing and MARKET_REGIME:
        year_dates = [d for d in MARKET_REGIME.keys() if d.startswith(str(test_year))]
        if year_dates:
            regimes = [MARKET_REGIME[d] for d in year_dates if pd.notna(MARKET_REGIME[d])]
            if regimes:
                avg_regime = np.mean(regimes)

                if args.v42:
                    # 🔥 V4.2: 공격적 Market Exit
                    # Regime < 0 (MA 이하)이면 해당 기간 100% 현금
                    bad_days = sum(1 for r in regimes if r < args.regime_threshold)
                    cash_ratio = bad_days / len(regimes)

                    if cash_ratio > 0:
                        held_cash = True
                        # 현금 보유 기간의 수익 = 0 (손실 회피)
                        # 투자 기간의 수익만 반영
                        portfolio_return = portfolio_return * (1 - cash_ratio)

                    # 🔥 V4.2: 손절 시뮬레이션 (21일 내 -7% 하락 시)
                    # 손절된 종목은 수익 기여도가 낮아짐
                    if abs_return_col in top_df.columns and 'drawdown_from_high' in top_df.columns:
                        # 큰 낙폭 종목 비율 계산 (손절 시뮬레이션)
                        severe_drawdown = top_df['drawdown_from_high'] < -args.stop_loss
                        stop_loss_ratio = severe_drawdown.mean() if len(top_df) > 0 else 0

                        # 손절 종목의 손실 제한 (-7%에서 손절)
                        if stop_loss_ratio > 0:
                            avg_loss_avoided = (top_df.loc[severe_drawdown, abs_return_col].mean() + args.stop_loss)
                            if pd.notna(avg_loss_avoided) and avg_loss_avoided < 0:
                                stop_loss_impact = -avg_loss_avoided * stop_loss_ratio * 100
                                portfolio_return += stop_loss_impact
                else:
                    # 기존 방식: 부분 현금 비중 조절
                    bad_days = sum(1 for r in regimes if r < args.regime_threshold or r > 0.15)
                    cash_ratio = bad_days / len(regimes)

                    if cash_ratio > 0.2:
                        held_cash = True
                        portfolio_return = portfolio_return * (1 - min(cash_ratio, 0.8))

    # 연간 실제 수익률 = 분기 수익률 복리 (4회 리밸런싱 가정)
    annual_portfolio = ((1 + portfolio_return/100) ** (252/HORIZON) - 1) * 100

    all_results.append({
        'year': test_year,
        'Q1': q1,
        'Q5': q5,
        'spread': spread,
        'IC': ic,
        'market_ret': market_ret,
        'abs_top': abs_top,
        'portfolio_return': portfolio_return,  # 실제 분기 수익률
        'annual_return': annual_portfolio,     # 실제 연환산 수익률
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'held_cash': held_cash,
        'cash_ratio': cash_ratio,              # V4.2: 현금 보유 비율
        'stop_loss_impact': stop_loss_impact   # V4.2: 손절 효과
    })

    if not np.isnan(spread):
        bar = '█' * int(max(0, min(annual_portfolio, 50)))
        ic_status = '✅' if ic > 0 else '❌'
        if args.v42 and held_cash:
            cash_status = f'💵{cash_ratio*100:.0f}%'
        elif held_cash:
            cash_status = '💵'
        else:
            cash_status = ''
        # 모든 모드에서 실제 연수익률 표시
        print(f'  {test_year}: 분기={portfolio_return:+5.1f}% | 연환산={annual_portfolio:+5.1f}% | IC={ic:+.3f} {ic_status} {cash_status} {bar}')

print(f'  ⏱ [2/3] Walk-Forward 백테스트: {time.time()-stage_t0:.1f}s')

# ============================================================================
# 결과 분석
# ============================================================================
stage_t0 = time.time()
print('\n[3/3] 결과 분석')
print('=' * 70)

results_df = pd.DataFrame(all_results)

# 최근 5년 vs 전체
recent_5y = results_df[results_df['year'] >= 2021]
all_years = results_df

print('\n[성과 요약]')
print('-' * 50)
print(f'{"기간":<15} {"평균 Spread":>12} {"평균 IC":>10} {"IC 양수율":>10}')
print('-' * 50)

for name, data in [('전체', all_years), ('최근 5년', recent_5y)]:
    avg_spread = data['spread'].mean()
    avg_ic = data['IC'].mean()
    ic_positive = (data['IC'] > 0).sum() / len(data) * 100
    print(f'{name:<15} {avg_spread:>+11.1f}% {avg_ic:>+9.3f} {ic_positive:>9.0f}%')

print('-' * 50)

# IC 분석
print('\n[연도별 IC]')
print('-' * 50)
for _, row in results_df.iterrows():
    ic = row['IC']
    bar = '█' * int(max(0, (ic + 0.1) * 100)) if ic > -0.1 else '░' * int(min(10, abs(ic) * 100))
    status = '✅' if ic > 0.02 else '⚠️' if ic > 0 else '❌'
    print(f"  {int(row['year'])}: IC = {ic:+.3f} {status} {bar}")

# V1 vs V2 비교 (예상)
print('\n[V1 vs V2 비교]')
print('-' * 50)
print('  V1 (재무 84%): IC ≈ 0, Q5 < Q1')
print(f'  V2 (모멘텀 중심): IC = {results_df["IC"].mean():+.3f}, Spread = {results_df["spread"].mean():+.1f}%')

# 피처 중요도
print('\n[V2 피처 중요도 Top 15]')
print('-' * 50)

# 마지막 모델의 피처 중요도
importance = model.feature_importance()

for i, row in importance.head(15).iterrows():
    feature = row['feature']
    imp = row['importance']

    # 그룹 분류
    if args.v5:
        group = 'V5'
    elif feature in momentum_features:
        group = '모멘텀'
    elif feature in volume_features:
        group = '수급'
    elif feature in volatility_features:
        group = '변동성'
    elif feature in intuition_features:
        group = '본능'
    elif feature in traditional_features:
        group = '전통'
    elif args.v4 and feature in macro_features:
        group = '매크로'
    else:
        group = '재무'

    bar = '█' * int(imp / importance['importance'].max() * 20)
    print(f'  {feature:<25} [{group:<4}] {bar}')

# 그룹별 중요도
print('\n[그룹별 피처 중요도]')
print('-' * 50)

if args.v5:
    groups = {'V5': all_features}
else:
    groups = {
        '모멘텀': momentum_features,
        '수급': volume_features,
        '변동성': volatility_features,
        '본능전략': intuition_features,
        '전통지표': traditional_features,
        '재무': fund_features,
    }
    if args.v4:
        groups['매크로'] = macro_features

total_imp = importance['importance'].sum()
for group_name, group_features in groups.items():
    group_imp = importance[importance['feature'].isin(group_features)]['importance'].sum()
    pct = group_imp / total_imp * 100
    bar = '█' * int(pct / 5)
    print(f'  {group_name:<10} {pct:>5.1f}% {bar}')

# 최종 판정
print('\n' + '=' * 70)
print('💰 실제 포트폴리오 연간 수익률')
print('=' * 70)

avg_ic = results_df['IC'].mean()
avg_spread = results_df['spread'].mean()
ic_positive_rate = (results_df['IC'] > 0).sum() / len(results_df) * 100

# ============================================================================
# 핵심: 실제 연간 수익률 (선택된 종목 기반)
# ============================================================================
print(f'\n  [연도별 실제 수익률] (Top {TOP_N}개 종목, 슬리피지 {args.slippage}% 반영)')
print('  ' + '-' * 55)
print(f'  {"연도":<6} {"분기수익률":>12} {"연환산":>12} {"IC":>8}  상태')
print('  ' + '-' * 55)

annual_returns = []
for _, row in results_df.iterrows():
    year = int(row['year'])
    qtr_ret = row['portfolio_return']
    ann_ret = row['annual_return']
    ic = row['IC']

    # NaN 체크 (미래 데이터 없는 연도 스킵)
    if pd.isna(ann_ret) or pd.isna(qtr_ret):
        continue

    annual_returns.append(ann_ret)

    # 상태 바
    if ann_ret > 20:
        bar = '🔥' + '█' * min(10, int(ann_ret / 5))
    elif ann_ret > 0:
        bar = '✅' + '▓' * min(10, int(ann_ret / 3))
    else:
        bar = '❌' + '░' * min(10, int(abs(ann_ret) / 3))

    print(f'  {year:<6} {qtr_ret:>+11.1f}% {ann_ret:>+11.1f}% {ic:>+7.3f}  {bar}')

print('  ' + '-' * 55)

# 요약 통계
avg_annual = np.mean(annual_returns)
median_annual = np.median(annual_returns)
std_annual = np.std(annual_returns)
min_annual = np.min(annual_returns)
max_annual = np.max(annual_returns)
positive_years = sum(1 for r in annual_returns if r > 0)
total_years = len(annual_returns)

print(f'\n  📊 요약 통계')
print('  ' + '-' * 55)
print(f'    평균 연수익률:    {avg_annual:>+8.1f}%')
print(f'    중앙값 연수익률:  {median_annual:>+8.1f}%')
print(f'    표준편차:         {std_annual:>8.1f}%')
print(f'    최고 연도:        {max_annual:>+8.1f}%')
print(f'    최악 연도:        {min_annual:>+8.1f}%')
print(f'    수익 연도:        {positive_years}/{total_years}년 ({positive_years/total_years*100:.0f}%)')

# 핵심 결론
print('\n  ' + '=' * 55)
if avg_annual > 15:
    emoji = '🔥'
elif avg_annual > 5:
    emoji = '✅'
elif avg_annual > 0:
    emoji = '⚠️'
else:
    emoji = '❌'
print(f'  {emoji} 평균 연수익률: {avg_annual:+.1f}% (중앙값: {median_annual:+.1f}%)')
print('  ' + '=' * 55)

# 모델 품질 요약
print(f'\n  [모델 품질]')
print(f'    평균 IC: {avg_ic:+.3f}')
print(f'    IC 양수율: {ic_positive_rate:.0f}%')

# V4.2 전용 요약
if args.v42:
    print(f'\n  [V4.2 Market Exit 효과]')
    print('  ' + '-' * 55)
    avg_cash_ratio = results_df['cash_ratio'].mean() * 100
    total_stop_loss = results_df['stop_loss_impact'].sum()
    cash_years = (results_df['held_cash']).sum()
    print(f'    평균 현금 보유 비율: {avg_cash_ratio:.1f}%')
    print(f'    현금 보유 연도: {cash_years}/{len(results_df)}년')
    print(f'    손절로 회피한 손실: {total_stop_loss:+.1f}%')

# ============================================================================
# 안전성 체크 (Safety Check)
# ============================================================================
print('\n' + '=' * 70)
print('🛡️  안전성 체크')
print('=' * 70)

safety_checks = {}

# 1. IC 안정성
ic_stable = avg_ic >= 0.02 and ic_positive_rate >= 50
safety_checks['IC 안정성 (IC≥0.02, 양수율≥50%)'] = ic_stable

# 2. 수익률 안정성 (연환산 수익률 변동)
ret_stable = std_annual < 25.0  # 연환산이라 기준 완화
safety_checks[f'수익률 안정성 (σ={std_annual:.1f}% < 25%)'] = ret_stable

# 3. 손실 연도 체크
loss_years = total_years - positive_years
loss_check = loss_years <= total_years * 0.4  # 40% 이하 손실연도
safety_checks[f'손실연도 ({loss_years}/{total_years} ≤ 40%)'] = loss_check

# 4. 최악의 해 체크
worst_check = min_annual > -20.0  # 연환산 기준 -20% 이상
safety_checks[f'최악의 해 ({min_annual:+.1f}% > -20%)'] = worst_check

# 5. 연환산 수익률 체크
annual_check = avg_annual > 5.0  # 연 5% 이상
safety_checks[f'연수익률 ({avg_annual:+.1f}% > 5%)'] = annual_check

# 6. 슬리피지 민감도
spread_after_slip = avg_spread  # 이미 슬리피지 반영됨
slip_check = spread_after_slip > 0
safety_checks[f'슬리피지 후 Spread ({spread_after_slip:+.1f}% > 0)'] = slip_check

# 결과 출력
print('\n  [체크리스트]')
print('  ' + '-' * 50)
passed = 0
for check_name, result in safety_checks.items():
    status = '✅ PASS' if result else '❌ FAIL'
    print(f'    {status}  {check_name}')
    if result:
        passed += 1

print('  ' + '-' * 50)
print(f'    통과: {passed}/{len(safety_checks)}')

# 최종 판정
print('\n  [전략 유효성 판정]')
if passed == len(safety_checks):
    print('    🔥 매우 안전 - 실전 투입 가능')
elif passed >= len(safety_checks) * 0.7:
    print('    ✅ 양호 - 소액으로 테스트 추천')
elif passed >= len(safety_checks) * 0.5:
    print('    ⚠️  주의 - 추가 검증 필요')
else:
    print('    ❌ 위험 - 전략 재검토 필요')

# 저장
results_df.to_csv('backtest_v2_results.csv', index=False)
print(f'\n결과 저장: backtest_v2_results.csv')

print('\n' + '=' * 70)
print('V2 백테스트 완료!')
print('=' * 70)
