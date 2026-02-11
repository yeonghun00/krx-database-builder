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
import sys
from pathlib import Path
from scipy import stats
from datetime import datetime
import time

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
parser.add_argument('--v6', action='store_true', help='🔥 V6 Quality Features (V5 + 12 fundamental quality)')
parser.add_argument('--v7', action='store_true', help='🔥 V7 5-Pillar Only (Moat/Capital/Quality/Reinvest/Value)')
parser.add_argument('--v8', action='store_true',
    help='V8: 50 stocks, overlapping portfolios, lambdarank, hybrid target')
parser.add_argument('--v9', action='store_true',
    help='V9: Huber regression, 21d horizon, buffer zone portfolio, no hvup')
parser.add_argument('--v10', action='store_true',
    help='V10: Z-score target, LR=0.01, sector constraints, 42d horizon')
parser.add_argument('--v11', action='store_true',
    help='V11: Sector-neutral target, risk-adjusted ranking, dynamic cash, rank weighting')
parser.add_argument('--no-cache', action='store_true', help='캐시 미사용 (cold run 강제)')
parser.add_argument('--workers', type=int, default=None, help='병렬 워커 수 (기본: min(8, cpu_count))')
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

# V11 모드 설정 (Sector-neutral target, risk-adjusted ranking, dynamic cash, rank weighting)
if args.v11:
    args.v10 = True  # Cascades: V10 → V9 → V6 → V5 → V4.3 → V4.2 → V4
    args.top = 50
    args.horizon = 42
    args.slippage = 0.2

# V10 모드 설정 (Z-score target, LR=0.01, sector constraints, 42d horizon)
if args.v10 and not args.v11:
    args.v9 = True   # Inherit V9 cascade (V6 → V5 → V4.3 → V4.2 → V4)
    args.top = 50
    args.horizon = 42  # 21 → 42 days (2 months, reduces turnover cost)
    args.slippage = 0.2  # tighter slippage assumption

if args.v11:
    args.v9 = True   # Inherit V9 cascade

# V9 모드 설정 (Huber regression, 21d horizon, buffer zone, no hvup)
if args.v9:
    args.v6 = True   # Inherit V6 (NOT V8)
    if not args.v10:
        args.top = 50
        args.horizon = 21  # 21 days, not 63

# V8 모드 설정 (V6 + overlapping portfolios + lambdarank + hybrid target)
if args.v8:
    args.v6 = True  # Inherit V6 cascade (V5→V4.3→V4.2→V4)
    args.top = 50   # 50 stocks instead of 20
    args.horizon = 63

# V6 모드 설정 (V5 + 12 Quality Features)
if args.v6:
    args.v5 = True
    args.v43 = True
    args.v42 = True
    args.v4 = True
    args.cash_timing = True
    args.regime_threshold = 0.0
    if not args.v9:  # V9 overrides horizon to 21d
        args.horizon = 63
    if args.top == 0:
        args.top = 20

# V5 모드 설정 (Research-backed 10-Feature Mode, cascades from v43)
if args.v5:
    args.v43 = True
    args.v42 = True
    args.v4 = True
    args.cash_timing = True
    args.regime_threshold = 0.0
    if not args.v9:  # V9 overrides horizon to 21d
        args.horizon = 63  # Forced to 63 days
    if args.top == 0:
        args.top = 20

print('=' * 70)
if args.v11:
    print('🔥 V11 백테스트 (Sector-Neutral + Risk-Adjusted + Dynamic Cash + Rank Weighting)')
elif args.v10:
    print('🔥 V10 백테스트 (Z-score Target + LR=0.01 + Sector-Constrained + 42d Horizon)')
elif args.v9:
    print('🔥 V9 백테스트 (Huber Regression + 21d Horizon + Buffer Zone Portfolio)')
elif args.v8:
    print('🔥 V8 백테스트 (LambdaRank + Overlapping Portfolios + Hybrid Target)')
elif args.v7:
    print('🔥 V7 백테스트 (5-Pillar Only Model)')
elif args.v6:
    print('🔥 V6 백테스트 (V5 + 12 Fundamental Quality Features)')
elif args.v5:
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
if args.v11:
    print(f'  - 🔥 V11 Final Alpha Build:')
    print(f'      • Target: Sector-neutral Z-score')
    print(f'      • Ranking: pred_score / volatility_20d (risk-adjusted)')
    print(f'      • Weighting: Rank-based (top10=3x, 11-30=2x, 31-50=1x)')
    print(f'      • Cash: VKOSPI absolute + regime graduated')
    print(f'      • Slippage ×: 0.3 (rank weighting reduces tail churn)')
    print(f'      • Everything else: inherited from V10')
elif args.v10:
    print(f'  - 🔥 V10 Underfitting/Slippage/Sector Fix:')
    print(f'      • 피처: V9 23개 (inherited)')
    print(f'      • Target: Cross-sectional Z-score (relative ranking)')
    print(f'      • Model: Huber (delta=1.0), LR=0.01, 1000 trees, patience=100')
    print(f'      • Portfolio: 50 stocks, sector max 7, buffer zone (hold<=100)')
    print(f'      • Horizon: 42d, rebalance every 42d')
    print(f'      • Validation: Shuffled 10% (not time-based)')
    print(f'      • Risk: VKOSPI spike de-grossing (inherited)')
elif args.v9:
    print(f'  - 🔥 V9 Recovery Build:')
    print(f'      • 피처: V6 22개 + reversal 1개 = 23개 (no hvup)')
    print(f'      • Target: Signed Log Return (continuous, robust)')
    print(f'      • Model: Huber Regression + Early Stopping')
    print(f'      • Portfolio: 50 stocks, buffer zone (buy<=50, hold<=120)')
    print(f'      • Horizon: 21d (Korean theme-driven market)')
    print(f'      • Risk: VKOSPI spike de-grossing')
elif args.v8:
    print(f'  - 🔥 V8 IC-to-Return Gap Fix:')
    print(f'      • 피처: V6 22개 + 3개 = 25개')
    print(f'      • Target: Hybrid Rank (70% residual + 30% absolute)')
    print(f'      • Model: LambdaRank + Early Stopping')
    print(f'      • Portfolio: 50 stocks, 3 overlapping cohorts (21d stagger)')
    print(f'      • Risk: VKOSPI spike de-grossing')
elif args.v6:
    print(f'  - 🔥 V6 Quality Features (5-Pillar Framework):')
    print(f'      • 피처: V5 10개 + Quality 12개 = 22개')
    print(f'      • Pillars: Moat, Capital, Quality, Growth, Value')
    print(f'      • V5 기능 모두 상속')
elif args.v7:
    print(f'  - 🔥 V7 5-Pillar Only:')
    print(f'      • 피처: 5개 합성 스코어만 사용')
    print(f'      • Pillars: Moat, Capital, Earnings, Reinvestment, Value')
elif args.v5:
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
    include_macro=args.v4,  # V4: 매크로 피처 추가
    use_cache=not args.no_cache,
    n_workers=args.workers
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

# V9: Override with 23 features (V6 22 + reversal, no hvup)
if args.v9:
    all_features = [c for c in fe.V9_FEATURES if c in df.columns]
    missing_v9 = [c for c in fe.V9_FEATURES if c not in df.columns]
    if missing_v9:
        print(f'  ⚠️ Missing V9 features: {missing_v9}')
# V8: Override with 25 features (V6 22 + 3 new)
elif args.v8:
    all_features = [c for c in fe.V8_FEATURES if c in df.columns]
    missing_v8 = [c for c in fe.V8_FEATURES if c not in df.columns]
    if missing_v8:
        print(f'  ⚠️ Missing V8 features: {missing_v8}')
# V6: Override with 22 features (V5 10 + Quality 12)
elif args.v6:
    all_features = [c for c in fe.V6_FEATURES if c in df.columns]
    missing_v6 = [c for c in fe.V6_FEATURES if c not in df.columns]
    if missing_v6:
        print(f'  ⚠️ Missing V6 features: {missing_v6}')
# V5: Override with 10 focused features
elif args.v5:
    all_features = [c for c in fe.V5_FEATURES if c in df.columns]
    missing_v5 = [c for c in fe.V5_FEATURES if c not in df.columns]
    if missing_v5:
        print(f'  ⚠️ Missing V5 features: {missing_v5}')
elif args.v7:
    all_features = [c for c in fe.MODEL7_FEATURES if c in df.columns]
    missing_v7 = [c for c in fe.MODEL7_FEATURES if c not in df.columns]
    if missing_v7:
        print(f'  ⚠️ Missing V7 features: {missing_v7}')

tech_count = len(momentum_features + volume_features + volatility_features +
                 intuition_features + traditional_features)

print(f'  총 데이터: {len(df):,} rows')
if args.v9:
    v6_in_use = [c for c in fe.V6_FEATURES if c in df.columns]
    v9_new = [c for c in ['short_term_reversal'] if c in df.columns]
    print(f'  피처 구성: V9 (V6 {len(v6_in_use)} + reversal {len(v9_new)} = {len(all_features)}, no hvup)')
    print(f'  총 피처: {len(all_features)}개')
elif args.v8:
    v8_new = ['inventory_sales_gap', 'current_ratio', 'hvup_ratio']
    v8_in_use = [c for c in v8_new if c in df.columns]
    v6_in_use = [c for c in fe.V6_FEATURES if c in df.columns]
    print(f'  피처 구성: V8 (V6 {len(v6_in_use)} + New {len(v8_in_use)} = {len(all_features)})')
    print(f'    [V8 New]:')
    for f_name in v8_in_use:
        print(f'      - {f_name}')
    print(f'  총 피처: {len(all_features)}개')
elif args.v6:
    v5_in_use = [c for c in fe.V5_FEATURES if c in df.columns]
    quality_in_use = [c for c in fe.QUALITY_FEATURES if c in df.columns]
    print(f'  피처 구성: V6 (V5 {len(v5_in_use)} + Quality {len(quality_in_use)} = {len(all_features)})')
    print(f'    [V5 Base]:')
    for f_name in v5_in_use:
        print(f'      - {f_name}')
    print(f'    [Quality]:')
    for f_name in quality_in_use:
        print(f'      - {f_name}')
    print(f'  총 피처: {len(all_features)}개')
elif args.v5:
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

# 타겟 설정: V11 > V10 > V9 > V8 > V4.3 > QEPM > 일반
if args.v11:
    target_col = f'target_sector_zscore_{HORIZON}d'
    return_col = f'forward_return_{HORIZON}d'
elif args.v10:
    target_col = f'target_zscore_{HORIZON}d'
    return_col = f'forward_return_{HORIZON}d'
elif args.v9:
    target_col = f'target_log_return_{HORIZON}d'
    return_col = f'forward_return_{HORIZON}d'
elif args.v8:
    target_col = f'target_hybrid_rank_{HORIZON}d'
    return_col = f'forward_return_{HORIZON}d'
    if target_col not in df.columns:
        print(f'  ⚠️ Hybrid target 없음, residual rank로 fallback')
        target_col = f'target_residual_rank_{HORIZON}d'
elif args.v43:
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
all_test_dfs = {}  # year -> test_df for decile/sector analysis

# ============================================================================
# QEPM 헬퍼 함수들
# ============================================================================

def load_all_benchmark_returns(horizon=63):
    """
    모든 연도의 벤치마크 수익률을 한 번에 로드 (캐시용)
    Uses index_daily_prices table with KOSPI index.
    Computes forward returns from closing_index via pandas shift.
    """
    import sqlite3

    try:
        conn = sqlite3.connect('krx_stock_data.db')
        query = """
        SELECT date, closing_index
        FROM index_daily_prices
        WHERE index_code = 'KOSPI_코스피'
          AND closing_index IS NOT NULL
        ORDER BY date
        """
        bm_df = pd.read_sql_query(query, conn)
        conn.close()

        if len(bm_df) == 0:
            print('    ⚠️ 벤치마크 데이터 없음 (KOSPI_코스피)')
            return {}

        # Compute forward return over horizon trading days
        bm_df['forward_price'] = bm_df['closing_index'].shift(-horizon)
        bm_df['forward_return'] = (bm_df['forward_price'] / bm_df['closing_index'] - 1) * 100
        bm_df['year'] = bm_df['date'].str[:4].astype(int)

        # Average forward return per year
        yearly = bm_df.dropna(subset=['forward_return']).groupby('year')['forward_return'].mean()
        return yearly.to_dict()
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

# Parallel walk-forward: train all years concurrently
from concurrent.futures import ThreadPoolExecutor

def _train_year(test_year):
    """Train and predict for one test year (thread-safe)."""
    train_start = test_year - TRAIN_YEARS
    train_end = test_year - 1
    buffer_month = 12 - (HORIZON // 21 + BUFFER_MONTHS)
    buffer_month = max(1, min(buffer_month, 10))
    train_cutoff = f'{train_end}{buffer_month:02d}01'

    train_df = df[(df['year'] >= train_start) &
                  (df['year'] <= train_end) &
                  (df['date'] <= train_cutoff)].copy()
    test_df = df[df['year'] == test_year].copy()

    if len(train_df) < 1000 or len(test_df) < 100:
        return None

    if args.v43:
        test_start_date = test_df['date'].min()
        purge_cutoff = pd.to_datetime(test_start_date, format='%Y%m%d') - pd.Timedelta(days=HORIZON * 1.5)
        purge_cutoff_str = purge_cutoff.strftime('%Y%m%d')
        train_df = train_df[train_df['date'] <= purge_cutoff_str]
        if 'macro_risk_score' in train_df.columns:
            train_df['sample_weight'] = 1.0
            high_risk_mask = train_df['macro_risk_score'] > 0.6
            train_df.loc[high_risk_mask, 'sample_weight'] = 2.0
        else:
            train_df['sample_weight'] = 1.0

    if args.v11:
        # V11: Sector-neutral target, same training as V10 but with V11_PARAMS
        _model = MLRanker(
            feature_cols=all_features,
            target_col=target_col,
            model_type='regressor',
            time_decay=0.7,
            patience=100
        )
        lgb_params = {**MLRanker.V11_PARAMS, 'n_jobs': 1}

        _shuffled = train_df.sample(frac=1.0, random_state=42)
        val_size = max(int(len(_shuffled) * 0.1), 100)
        val_df = _shuffled.iloc[:val_size]
        train_only = _shuffled.iloc[val_size:]

        if len(val_df) > 100:
            _model.train(train_only, val_df=val_df, params=lgb_params)
        else:
            _model.train(train_only, params=lgb_params)
    elif args.v10:
        # V10: Huber regression on Z-score target, shuffled validation, patience=100
        _model = MLRanker(
            feature_cols=all_features,
            target_col=target_col,
            model_type='regressor',
            time_decay=0.7,
            patience=100
        )
        lgb_params = {**MLRanker.V10_PARAMS, 'n_jobs': 1}

        # Validation: shuffled 10% of train (not time-based split)
        # Fixes underfitting caused by temporal distribution shift
        _shuffled = train_df.sample(frac=1.0, random_state=42)
        val_size = max(int(len(_shuffled) * 0.1), 100)
        val_df = _shuffled.iloc[:val_size]
        train_only = _shuffled.iloc[val_size:]

        if len(val_df) > 100:
            _model.train(train_only, val_df=val_df, params=lgb_params)
        else:
            _model.train(train_only, params=lgb_params)
    elif args.v9:
        # V9: Huber regression with early stopping
        _model = MLRanker(
            feature_cols=all_features,
            target_col=target_col,
            model_type='regressor',
            time_decay=0.7
        )
        lgb_params = {**MLRanker.V9_HUBER_PARAMS, 'n_jobs': 1}
        val_cutoff = f'{train_end}0801'
        val_df = train_df[train_df['date'] > val_cutoff]
        train_only = train_df[train_df['date'] <= val_cutoff]
        if len(val_df) > 100:
            _model.train(train_only, val_df=val_df, params=lgb_params)
        else:
            _model.train(train_only, params=lgb_params)
    elif args.v8:
        # V8: LambdaRank with validation + early stopping
        # LambdaRank requires integer relevance labels (0-4)
        # Convert continuous percentile rank (0-1) → 5 relevance grades
        # Drop rows with NaN target first (no forward return data)
        _v8_target = f'_v8_relevance'
        train_df = train_df.dropna(subset=[target_col]).copy()
        train_df[_v8_target] = (train_df[target_col] * 4).round().astype(int).clip(0, 4)

        _model = MLRanker(
            feature_cols=all_features,
            target_col=_v8_target,
            model_type='ranker',
            time_decay=0.0  # lambdarank doesn't use sample weights
        )

        # Split: last 4 months of training → validation
        val_cutoff = f'{train_end}0801'
        val_df = train_df[train_df['date'] > val_cutoff].copy()
        train_only = train_df[train_df['date'] <= val_cutoff].copy()

        lgb_params = {**MLRanker.V8_RANK_PARAMS, 'n_jobs': 1}
        if len(val_df) > 100:
            _model.train(train_only, val_df=val_df, params=lgb_params)
        else:
            _model.train(train_only, params=lgb_params)
    else:
        # n_jobs=1 per model to avoid thread contention
        lgb_params = {**MLRanker.REGRESSION_PARAMS, 'n_jobs': 1}
        _model = MLRanker(
            feature_cols=all_features,
            target_col=target_col,
            model_type='regressor',
            time_decay=0.5 if QEPM_MODE else 0.7
        )

        if args.v43 and 'sample_weight' in train_df.columns:
            _model.train(train_df, params=lgb_params, sample_weight=train_df['sample_weight'].values)
        else:
            _model.train(train_df, params=lgb_params)

    test_df['pred_score'] = _model.predict(test_df)
    test_df['pred_rank'] = test_df.groupby('date')['pred_score'].rank(ascending=False)

    return (test_year, _model, test_df, len(train_df), len(test_df))

# Pre-populate benchmark cache before parallel execution
if QEPM_MODE and years:
    get_benchmark_return(years[0], HORIZON)

with ThreadPoolExecutor() as pool:
    _trained = [r for r in pool.map(_train_year, years) if r is not None]

for test_year, model, test_df, _n_train, _n_test in _trained:
    avg_turnover = 0.0  # default, overridden in V9/V10/V11

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
        if args.v11:
            # V11: Sector-neutral + risk-adjusted ranking + rank weighting + dynamic cash
            test_df = test_df.sort_values(['date', 'pred_rank']).copy()
            test_df['sector'] = test_df['sector'].fillna('Unknown')
            dates = sorted(test_df['date'].unique())

            hold_threshold = int(TOP_N * 2.0)  # 100
            MAX_SECTOR = 7
            rebal_interval = 42

            rebal_dates = dates[::rebal_interval]

            # Pre-compute daily VKOSPI absolute level for dynamic cash
            _v11_vkospi = {}
            if 'fear_index_level' in test_df.columns:
                fear_by_date = test_df.groupby('date')['fear_index_level'].first().sort_index()
                for d in fear_by_date.index:
                    if pd.notna(fear_by_date[d]):
                        _v11_vkospi[d] = fear_by_date[d]

            prev_portfolio = set()
            cohort_picks = {}
            cohort_weights = {}  # stock_code -> weight
            turnover_list = []

            for rd in rebal_dates:
                day_df = test_df[test_df['date'] == rd].copy()

                # Risk-adjusted ranking: pred_score / volatility_20d
                if 'volatility_20d' in day_df.columns:
                    vol = day_df['volatility_20d'].fillna(day_df['volatility_20d'].median()).clip(lower=0.05)
                    day_df['risk_adj_score'] = day_df['pred_score'] / vol
                else:
                    day_df['risk_adj_score'] = day_df['pred_score']
                day_df['risk_adj_rank'] = day_df['risk_adj_score'].rank(ascending=False)
                day_df = day_df.sort_values('risk_adj_rank')

                # Survivors: existing holdings still within hold threshold
                if prev_portfolio:
                    hold_df = day_df[day_df['stock_code'].isin(prev_portfolio)]
                    survivors = set(hold_df[hold_df['risk_adj_rank'] <= hold_threshold]['stock_code'].tolist())
                else:
                    survivors = set()

                # New buys: top-ranked, sector-constrained
                day_df['_sc'] = day_df.groupby('sector').cumcount() + 1
                eligible = day_df[day_df['_sc'] <= MAX_SECTOR]
                new_buys = set(eligible.head(TOP_N)['stock_code'].tolist())

                portfolio = new_buys | survivors
                cohort_picks[rd] = portfolio

                # Turnover tracking
                if prev_portfolio:
                    buys = portfolio - prev_portfolio
                    sells = prev_portfolio - portfolio
                    denom = max(len(portfolio), 1)
                    turnover_list.append((len(buys) + len(sells)) / (2 * denom))

                # Rank-based weighting: top10=3x, 11-30=2x, 31-50=1x
                weights = {}
                for _, row in eligible.head(TOP_N).iterrows():
                    r = row['risk_adj_rank']
                    if r <= 10:
                        w = 3.0
                    elif r <= 30:
                        w = 2.0
                    else:
                        w = 1.0
                    weights[row['stock_code']] = w
                # Survivors get weight 1.0 if not already in new_buys
                for sc in survivors:
                    if sc not in weights:
                        weights[sc] = 1.0
                # Normalize weights
                total_w = sum(weights.values())
                if total_w > 0:
                    weights = {k: v / total_w for k, v in weights.items()}
                cohort_weights[rd] = weights

                prev_portfolio = portfolio
            avg_turnover = np.mean(turnover_list) if turnover_list else 0

            # Compute daily returns with rank weighting + dynamic cash
            daily_returns = []
            v11_cash_days = 0
            v11_degross_days = 0
            for d in dates:
                active_rebals = [rd for rd in rebal_dates if rd <= d]
                if not active_rebals:
                    continue
                latest_rebal = active_rebals[-1]
                weights = cohort_weights.get(latest_rebal, {})

                day_df = test_df[test_df['date'] == d]
                port_df = day_df[day_df['stock_code'].isin(weights.keys())]

                if len(port_df) == 0:
                    continue

                # Weighted return
                port_df = port_df.copy()
                port_df['_w'] = port_df['stock_code'].map(weights).fillna(0)
                w_sum = port_df['_w'].sum()
                if w_sum > 0:
                    port_df['_w'] = port_df['_w'] / w_sum
                raw_ret = (port_df[return_col] * port_df['_w']).sum()

                # Dynamic cash: VKOSPI absolute level + regime graduated
                exposure = 1.0
                regime_bad = MARKET_REGIME.get(d, 0) < args.regime_threshold if MARKET_REGIME else False
                vkospi_level = _v11_vkospi.get(d, 0)
                vkospi_high = vkospi_level > 25

                if regime_bad and vkospi_high:
                    exposure = 0.3   # 70% cash
                    v11_cash_days += 1
                elif regime_bad:
                    exposure = 0.5   # 50% cash
                    v11_degross_days += 1
                elif vkospi_high:
                    exposure = 0.7   # 30% cash
                    v11_degross_days += 1

                daily_returns.append(raw_ret * exposure)

            q5_raw = np.mean(daily_returns) * 100 if daily_returns else 0
            top_df = test_df[test_df['pred_rank'] <= TOP_N]

            # Track cash/degross for reporting
            total_days = len(daily_returns) if daily_returns else 1
            cash_ratio = v11_cash_days / total_days
            held_cash = (v11_cash_days + v11_degross_days) > 0

            # Bottom N
            max_rank = test_df.groupby('date')['pred_rank'].transform('max')
            bottom_df = test_df[test_df['pred_rank'] > max_rank - TOP_N]
            q1_raw = bottom_df[return_col].mean() * 100

            # Lower effective slippage: rank weighting reduces tail churn
            effective_slippage = ROUND_TRIP_COST * 0.3
            q5 = q5_raw - effective_slippage * 100
            q1 = q1_raw - effective_slippage * 100
            spread = q5 - q1 if not (np.isnan(q5) or np.isnan(q1)) else np.nan

        elif args.v10:
            # V10: Sector-constrained buffer zone portfolio
            # Max 7 stocks per sector, 42-day rebalance, tighter hold threshold
            test_df = test_df.sort_values(['date', 'pred_rank']).copy()
            test_df['sector'] = test_df['sector'].fillna('Unknown')
            dates = sorted(test_df['date'].unique())

            hold_threshold = int(TOP_N * 2.0)  # 100 (tighter than V9's 120)
            MAX_SECTOR = 7
            rebal_interval = 42  # match horizon

            rebal_dates = dates[::rebal_interval]

            # Pre-compute daily VKOSPI spike signal for this year
            _v10_fear_spike = {}
            if 'fear_index_level' in test_df.columns:
                fear_by_date = test_df.groupby('date')['fear_index_level'].first().sort_index()
                fear_ma20 = fear_by_date.rolling(20, min_periods=10).mean()
                for d in fear_by_date.index:
                    if pd.notna(fear_by_date[d]) and pd.notna(fear_ma20.get(d)):
                        _v10_fear_spike[d] = fear_by_date[d] > (fear_ma20[d] * 1.2)
                    else:
                        _v10_fear_spike[d] = False

            prev_portfolio = set()
            cohort_picks = {}
            turnover_list = []

            for rd in rebal_dates:
                day_df = test_df[test_df['date'] == rd].copy()
                day_df = day_df.sort_values('pred_rank')

                # Survivors: existing holdings still within hold threshold
                if prev_portfolio:
                    hold_df = day_df[day_df['stock_code'].isin(prev_portfolio)]
                    survivors = set(hold_df[hold_df['pred_rank'] <= hold_threshold]['stock_code'].tolist())
                else:
                    survivors = set()

                # New buys: top-ranked, sector-constrained
                day_df['_sc'] = day_df.groupby('sector').cumcount() + 1
                eligible = day_df[day_df['_sc'] <= MAX_SECTOR]
                new_buys = set(eligible.head(TOP_N)['stock_code'].tolist())

                portfolio = new_buys | survivors
                cohort_picks[rd] = portfolio

                # Turnover tracking
                if prev_portfolio:
                    buys = portfolio - prev_portfolio
                    sells = prev_portfolio - portfolio
                    denom = max(len(portfolio), 1)
                    turnover_list.append((len(buys) + len(sells)) / (2 * denom))

                prev_portfolio = portfolio
            avg_turnover = np.mean(turnover_list) if turnover_list else 0

            # Compute daily returns with per-day regime/VKOSPI gross exposure
            daily_returns = []
            v10_cash_days = 0
            v10_degross_days = 0
            for d in dates:
                active_rebals = [rd for rd in rebal_dates if rd <= d]
                if not active_rebals:
                    continue
                latest_rebal = active_rebals[-1]
                active_stocks = cohort_picks.get(latest_rebal, set())

                day_df = test_df[test_df['date'] == d]
                port_df = day_df[day_df['stock_code'].isin(active_stocks)]

                if len(port_df) == 0:
                    continue

                raw_ret = port_df[return_col].mean()

                # Daily gross exposure based on regime + VKOSPI
                exposure = 1.0
                regime_bad = MARKET_REGIME.get(d, 0) < args.regime_threshold if MARKET_REGIME else False
                vkospi_spike = _v10_fear_spike.get(d, False)

                if regime_bad and vkospi_spike:
                    exposure = 0.0
                    v10_cash_days += 1
                elif regime_bad:
                    exposure = 0.5
                    v10_degross_days += 1
                elif vkospi_spike:
                    exposure = 0.5
                    v10_degross_days += 1

                daily_returns.append(raw_ret * exposure)

            q5_raw = np.mean(daily_returns) * 100 if daily_returns else 0
            top_df = test_df[test_df['pred_rank'] <= TOP_N]

            # Track cash/degross for reporting
            total_days = len(daily_returns) if daily_returns else 1
            cash_ratio = v10_cash_days / total_days
            held_cash = (v10_cash_days + v10_degross_days) > 0

            # Bottom N
            max_rank = test_df.groupby('date')['pred_rank'].transform('max')
            bottom_df = test_df[test_df['pred_rank'] > max_rank - TOP_N]
            q1_raw = bottom_df[return_col].mean() * 100

            # 42d rebalance + buffer zone reduces turnover even more
            effective_slippage = ROUND_TRIP_COST * 0.4
            q5 = q5_raw - effective_slippage * 100
            q1 = q1_raw - effective_slippage * 100
            spread = q5 - q1 if not (np.isnan(q5) or np.isnan(q1)) else np.nan

        elif args.v9:
            # V9: Buffer zone portfolio — reduce turnover via hold threshold
            # Buy threshold: pred_rank <= TOP_N (50)
            # Hold threshold: pred_rank <= TOP_N * 2.4 (120) — existing holdings stay
            # Rebalance every 21 trading days
            # Regime/VKOSPI checks integrated at daily level (not annual post-hoc)
            test_df = test_df.sort_values(['date', 'pred_rank']).copy()
            test_df['sector'] = test_df['sector'].fillna('Unknown')
            dates = sorted(test_df['date'].unique())

            rebal_dates = dates[::21]
            hold_threshold = int(TOP_N * 2.4)

            # Pre-compute daily VKOSPI spike signal for this year
            _v9_fear_spike = {}
            if 'fear_index_level' in test_df.columns:
                fear_by_date = test_df.groupby('date')['fear_index_level'].first().sort_index()
                fear_ma20 = fear_by_date.rolling(20, min_periods=10).mean()
                for d in fear_by_date.index:
                    if pd.notna(fear_by_date[d]) and pd.notna(fear_ma20.get(d)):
                        _v9_fear_spike[d] = fear_by_date[d] > (fear_ma20[d] * 1.2)
                    else:
                        _v9_fear_spike[d] = False

            prev_portfolio = set()
            cohort_picks = {}
            turnover_list = []
            for rd in rebal_dates:
                day_df = test_df[test_df['date'] == rd].copy()
                day_df = day_df.sort_values('pred_rank')

                # New buys: rank <= TOP_N
                new_buys = set(day_df[day_df['pred_rank'] <= TOP_N]['stock_code'].tolist())

                # Existing holdings that survive hold threshold
                if prev_portfolio:
                    hold_df = day_df[day_df['stock_code'].isin(prev_portfolio)]
                    survivors = set(hold_df[hold_df['pred_rank'] <= hold_threshold]['stock_code'].tolist())
                else:
                    survivors = set()

                # Portfolio = new buys + survivors
                portfolio = new_buys | survivors
                cohort_picks[rd] = portfolio

                # Turnover tracking
                if prev_portfolio:
                    buys = portfolio - prev_portfolio
                    sells = prev_portfolio - portfolio
                    denom = max(len(portfolio), 1)
                    turnover_list.append((len(buys) + len(sells)) / (2 * denom))

                prev_portfolio = portfolio
            avg_turnover = np.mean(turnover_list) if turnover_list else 0

            # Compute daily returns with per-day regime/VKOSPI gross exposure
            daily_returns = []
            v9_cash_days = 0
            v9_degross_days = 0
            for d in dates:
                active_rebals = [rd for rd in rebal_dates if rd <= d]
                if not active_rebals:
                    continue
                latest_rebal = active_rebals[-1]
                active_stocks = cohort_picks.get(latest_rebal, set())

                day_df = test_df[test_df['date'] == d]
                port_df = day_df[day_df['stock_code'].isin(active_stocks)]

                if len(port_df) == 0:
                    continue

                raw_ret = port_df[return_col].mean()

                # Daily gross exposure based on regime + VKOSPI
                exposure = 1.0
                regime_bad = MARKET_REGIME.get(d, 0) < args.regime_threshold if MARKET_REGIME else False
                vkospi_spike = _v9_fear_spike.get(d, False)

                if regime_bad and vkospi_spike:
                    exposure = 0.0   # Both signals → 100% cash
                    v9_cash_days += 1
                elif regime_bad:
                    exposure = 0.5   # Regime only → 50% de-gross
                    v9_degross_days += 1
                elif vkospi_spike:
                    exposure = 0.5   # VKOSPI spike only → 50% de-gross
                    v9_degross_days += 1

                daily_returns.append(raw_ret * exposure)

            q5_raw = np.mean(daily_returns) * 100 if daily_returns else 0
            top_df = test_df[test_df['pred_rank'] <= TOP_N]

            # Track cash/degross for reporting
            total_days = len(daily_returns) if daily_returns else 1
            cash_ratio = v9_cash_days / total_days
            held_cash = (v9_cash_days + v9_degross_days) > 0

            # Bottom N
            max_rank = test_df.groupby('date')['pred_rank'].transform('max')
            bottom_df = test_df[test_df['pred_rank'] > max_rank - TOP_N]
            q1_raw = bottom_df[return_col].mean() * 100

            # Buffer zone reduces turnover ~50%
            effective_slippage = ROUND_TRIP_COST * 0.5
            q5 = q5_raw - effective_slippage * 100
            q1 = q1_raw - effective_slippage * 100
            spread = q5 - q1 if not (np.isnan(q5) or np.isnan(q1)) else np.nan

        elif args.v8:
            # V8: Overlapping portfolios — 3 cohorts, each held 63 days, staggered 21 days
            test_df = test_df.sort_values(['date', 'pred_rank']).copy()
            test_df['sector'] = test_df['sector'].fillna('Unknown')
            dates = sorted(test_df['date'].unique())

            # Rebalance dates: every 21 trading days
            rebal_dates = dates[::21]

            # For each rebalance date, select top N stocks with sector constraint
            cohort_picks = {}
            for rd in rebal_dates:
                day_df = test_df[test_df['date'] == rd].copy()
                day_df = day_df.sort_values('pred_rank')
                day_df['_sc'] = day_df.groupby('sector').cumcount() + 1
                picks = day_df[day_df['_sc'] <= MAX_PER_SECTOR].head(TOP_N)
                cohort_picks[rd] = set(picks['stock_code'].tolist())

            # For each date, active portfolio = union of 3 most recent cohorts
            daily_returns = []
            for d in dates:
                active_rebals = [rd for rd in rebal_dates if rd <= d][-3:]
                active_stocks = set()
                for rd in active_rebals:
                    active_stocks |= cohort_picks.get(rd, set())

                day_df = test_df[test_df['date'] == d]
                port_df = day_df[day_df['stock_code'].isin(active_stocks)]

                if len(port_df) > 0:
                    daily_returns.append(port_df[return_col].mean())

            q5_raw = np.mean(daily_returns) * 100 if daily_returns else 0
            # Also compute top_df for downstream compatibility (last rebalance cohort)
            top_df = test_df[test_df['pred_rank'] <= TOP_N]

            # Bottom N
            max_rank = test_df.groupby('date')['pred_rank'].transform('max')
            bottom_df = test_df[test_df['pred_rank'] > max_rank - TOP_N]
            q1_raw = bottom_df[return_col].mean() * 100

            # Reduced slippage: only 1/3 portfolio turns over each rebalance
            effective_slippage = ROUND_TRIP_COST / 3
            q5 = q5_raw - effective_slippage * 100
            q1 = q1_raw - effective_slippage * 100
            spread = q5 - q1 if not (np.isnan(q5) or np.isnan(q1)) else np.nan

        elif QEPM_MODE:
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

        if not args.v8 and not args.v9 and not args.v10:
            # Bottom N (벡터화) — V8/V9/V10 compute this above
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
    if (args.v8 or args.v9 or args.v10) and TOP_N > 0:
        # V8/V9/V10: portfolio_return comes from cohort-based daily returns
        portfolio_return = q5  # already computed from overlapping/buffer cohorts
    elif TOP_N > 0 and abs_return_col in test_df.columns:
        if QEPM_MODE:
            # QEPM: top_df 재사용 (이미 섹터 제약 적용됨)
            portfolio_return = top_df[abs_return_col].mean() * 100 - ROUND_TRIP_COST * 100
        else:
            # 일반: Top N 평균
            portfolio_return = top_df[abs_return_col].mean() * 100 - ROUND_TRIP_COST * 100
    else:
        portfolio_return = q5  # fallback

    # 🛡️ 현금 타이밍: 시장 Regime이 나쁘면 현금 보유 (0% 수익)
    # V9/V10 handles regime/VKOSPI at daily level inside buffer zone portfolio loop
    if not args.v9 and not args.v10:
        held_cash = False
        cash_ratio = 0.0
    stop_loss_impact = 0.0

    if not args.v9 and not args.v10 and args.cash_timing and MARKET_REGIME:
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

    # V8: VKOSPI spike de-grossing (replaces blunt 120d MA cash timing)
    # V9 handles VKOSPI de-grossing daily inside buffer zone loop; V8 uses post-hoc
    if args.v8 and 'fear_index_level' in df.columns:
        year_df = df[df['year'] == test_year]
        if len(year_df) > 0:
            fear_by_date = year_df.groupby('date')['fear_index_level'].first()
            fear_ma20 = fear_by_date.rolling(20, min_periods=10).mean()
            # Spike = current > 1.2x 20-day mean
            spike_mask = fear_by_date > (fear_ma20 * 1.2)
            spike_ratio = spike_mask.mean() if len(spike_mask) > 0 else 0
            if spike_ratio > 0:
                # Reduce position by 50% during spike days
                degross_factor = 1 - (spike_ratio * 0.5)
                portfolio_return = portfolio_return * degross_factor
                held_cash = True
                cash_ratio = max(cash_ratio, spike_ratio * 0.5)

    # 연간 실제 수익률 = 분기 수익률 복리 (4회 리밸런싱 가정)
    annual_portfolio = ((1 + portfolio_return/100) ** (252/HORIZON) - 1) * 100

    # Store test_df for decile/sector analysis
    all_test_dfs[test_year] = test_df

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
        'train_samples': _n_train,
        'test_samples': _n_test,
        'held_cash': held_cash,
        'cash_ratio': cash_ratio,              # V4.2: 현금 보유 비율
        'stop_loss_impact': stop_loss_impact,  # V4.2: 손절 효과
        'avg_turnover': avg_turnover           # Per-rebalance turnover
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
# [3/3] Quant Tearsheet Report
# ============================================================================
stage_t0 = time.time()
print('\n[3/3] Quant Tearsheet')
print('=' * 70)

results_df = pd.DataFrame(all_results)

# --- Load KOSPI200 benchmark returns (annualized, same compounding) --------
benchmark_by_year = load_all_benchmark_returns(HORIZON)
# Annualize benchmark the same way as portfolio
bm_annual_by_year = {}
for yr, pct_ret in benchmark_by_year.items():
    bm_annual_by_year[yr] = ((1 + pct_ret / 100) ** (252 / HORIZON) - 1) * 100

# Add benchmark columns to results_df
results_df['benchmark_return'] = results_df['year'].map(bm_annual_by_year)
results_df['alpha'] = results_df['annual_return'] - results_df['benchmark_return']

# Filter to rows with valid data
valid = results_df.dropna(subset=['annual_return', 'benchmark_return'])
port_rets = valid['annual_return'].values
bm_rets = valid['benchmark_return'].values
alphas = valid['alpha'].values
ic_series = results_df['IC'].dropna().values

# ────────────────────────────────────────────────────────────────────────────
# Block 1: IC Analysis
# ────────────────────────────────────────────────────────────────────────────
mean_ic = np.mean(ic_series)
std_ic = np.std(ic_series, ddof=1) if len(ic_series) > 1 else 1e-9
icir = mean_ic / std_ic if std_ic > 1e-9 else 0.0
ic_hit_rate = np.mean(ic_series > 0) * 100
ic_strong_rate = np.mean(ic_series > 0.02) * 100
best_ic_idx = np.argmax(ic_series)
worst_ic_idx = np.argmin(ic_series)

print('\nIC Summary')
print('-' * 45)
print(f'  Mean IC          : {mean_ic:+.3f}')
print(f'  IC Std           :  {std_ic:.3f}')
print(f'  ICIR (IC/Std)    :  {icir:.2f}')
print(f'  IC Hit Rate      : {ic_hit_rate:.0f}% (>0)')
print(f'  IC > 0.02 Rate   : {ic_strong_rate:.0f}%')
print()
print(f'  {"Year":<6} {"IC":>8}  Note')
print(f'  {"----":<6} {"------":>8}  ----')
for i, (_, row) in enumerate(results_df.iterrows()):
    ic_val = row['IC']
    if pd.isna(ic_val):
        continue
    note = ''
    if i == best_ic_idx:
        note = '<- best'
    elif i == worst_ic_idx:
        note = '<- worst'
    print(f'  {int(row["year"]):<6} {ic_val:>+8.3f}  {note}')

# ────────────────────────────────────────────────────────────────────────────
# Block 1.5: Decile Analysis (pred_score → forward return monotonicity)
# ────────────────────────────────────────────────────────────────────────────
if all_test_dfs:
    print('\nDecile Analysis (pred_score → forward return)')
    print('-' * 70)
    decile_returns_all = {}  # decile -> list of returns across years
    for yr in sorted(all_test_dfs.keys()):
        tdf = all_test_dfs[yr]
        valid_mask = tdf['pred_score'].notna() & tdf[return_col].notna()
        tdf_valid = tdf[valid_mask]
        if len(tdf_valid) < 100:
            continue
        try:
            tdf_valid = tdf_valid.copy()
            tdf_valid['decile'] = pd.qcut(tdf_valid['pred_score'], 10, labels=False, duplicates='drop') + 1
            dec_ret = tdf_valid.groupby('decile')[return_col].mean() * 100
            for d in dec_ret.index:
                decile_returns_all.setdefault(d, []).append(dec_ret[d])
        except Exception:
            pass

    if decile_returns_all:
        n_deciles = max(decile_returns_all.keys())
        print(f'  {"Decile":<8}', end='')
        for d in range(1, n_deciles + 1):
            print(f' {"D"+str(d):>7}', end='')
        print()
        print(f'  {"------":<8}', end='')
        for _ in range(n_deciles):
            print(f' {"------":>7}', end='')
        print()
        avg_dec = {}
        for d in range(1, n_deciles + 1):
            vals = decile_returns_all.get(d, [0])
            avg_dec[d] = np.mean(vals)
        print(f'  {"Avg":>7}', end='')
        for d in range(1, n_deciles + 1):
            print(f' {avg_dec[d]:>+6.1f}%', end='')
        print()
        # Monotonicity check: D10 > D9 > ... > D1 (higher decile = higher pred_score = higher return)
        mono_ok = all(avg_dec.get(d+1, 0) >= avg_dec.get(d, 0) for d in range(1, n_deciles))
        mono_str = 'YES' if mono_ok else 'NO'
        spread_d = avg_dec.get(n_deciles, 0) - avg_dec.get(1, 0)
        print(f'\n  D{n_deciles}-D1 Spread: {spread_d:+.1f}%   Monotonic: {mono_str}')

# ────────────────────────────────────────────────────────────────────────────
# Block 1.6: IC Decay Analysis (signal strength at multiple horizons)
# ────────────────────────────────────────────────────────────────────────────
if all_test_dfs:
    print('\nIC Decay (Signal Strength by Horizon)')
    print('-' * 55)
    decay_horizons = [5, 10, 21, 42]
    horizon_ics = {h: [] for h in decay_horizons}

    for yr in sorted(all_test_dfs.keys()):
        tdf = all_test_dfs[yr].copy()
        if 'pred_score' not in tdf.columns:
            continue
        tdf = tdf.sort_values(['stock_code', 'date'])
        grouped = tdf.groupby('stock_code')
        for h in decay_horizons:
            col = f'_fwd_{h}d'
            if f'forward_return_{h}d' in tdf.columns:
                fwd = tdf[f'forward_return_{h}d']
            else:
                # Compute on-the-fly from closing_price
                if 'closing_price' in tdf.columns:
                    fwd = grouped['closing_price'].shift(-h) / tdf['closing_price'] - 1
                else:
                    continue
            valid_mask = tdf['pred_score'].notna() & fwd.notna()
            if valid_mask.sum() > 50:
                ic_h, _ = stats.spearmanr(tdf.loc[valid_mask, 'pred_score'], fwd[valid_mask])
                horizon_ics[h].append(ic_h)

    print(f'  {"Horizon":<10} {"Mean IC":>8} {"Std":>7} {"ICIR":>7} {"Decay%":>8}')
    print(f'  {"-------":<10} {"------":>8} {"---":>7} {"----":>7} {"------":>8}')
    base_ic = None
    for h in decay_horizons:
        ics = horizon_ics[h]
        if ics:
            m = np.mean(ics)
            s = np.std(ics, ddof=1) if len(ics) > 1 else 1e-9
            ir = m / s if s > 1e-9 else 0
            if base_ic is None:
                base_ic = m
                decay_pct = 0
            else:
                decay_pct = ((m - base_ic) / abs(base_ic) * 100) if abs(base_ic) > 1e-9 else 0
            print(f'  {h:>4}d     {m:>+7.3f}  {s:>6.3f}  {ir:>6.2f}  {decay_pct:>+6.0f}%')

# ────────────────────────────────────────────────────────────────────────────
# Block 2: Return Attribution (Portfolio vs Benchmark)
# ────────────────────────────────────────────────────────────────────────────
print(f'\nAnnual Returns: Portfolio vs KOSPI200  (Top {TOP_N}, slip {args.slippage}%)')
print('-' * 68)
print(f'  {"Year":<6} {"Portfolio":>10} {"KOSPI200":>10} {"Alpha":>10} {"Cash%":>7} {"Holdings":>9}')
print('-' * 68)
for _, row in results_df.iterrows():
    yr = int(row['year'])
    ann = row['annual_return']
    bm = row.get('benchmark_return', np.nan)
    alp = row.get('alpha', np.nan)
    cr = row['cash_ratio'] * 100
    if pd.isna(ann):
        continue
    bm_str = f'{bm:>+9.1f}%' if pd.notna(bm) else f'{"n/a":>10}'
    alp_str = f'{alp:>+9.1f}%' if pd.notna(alp) else f'{"n/a":>10}'
    print(f'  {yr:<6} {ann:>+9.1f}% {bm_str} {alp_str} {cr:>5.0f}%  {TOP_N:>7}')
print('-' * 68)
if len(valid) > 0:
    print(f'  {"Mean":<6} {np.mean(port_rets):>+9.1f}% {np.mean(bm_rets):>+9.1f}% {np.mean(alphas):>+9.1f}%')
    print(f'  {"Median":<6} {np.median(port_rets):>+9.1f}% {np.median(bm_rets):>+9.1f}% {np.median(alphas):>+9.1f}%')

# ────────────────────────────────────────────────────────────────────────────
# Block 3: Risk Metrics
# ────────────────────────────────────────────────────────────────────────────
RF = 3.0  # Korea risk-free rate approx

if len(port_rets) > 1:
    port_mean = np.mean(port_rets)
    port_std = np.std(port_rets, ddof=1)
    bm_mean = np.mean(bm_rets) if len(bm_rets) > 0 else 0
    bm_std = np.std(bm_rets, ddof=1) if len(bm_rets) > 1 else 1e-9

    # Sharpe
    port_sharpe = (port_mean - RF) / port_std if port_std > 1e-9 else 0
    bm_sharpe = (bm_mean - RF) / bm_std if bm_std > 1e-9 else 0

    # Sortino (downside deviation: years below Rf)
    port_downside = port_rets[port_rets < RF] - RF
    port_dd_std = np.std(port_downside, ddof=1) if len(port_downside) > 1 else (abs(port_downside[0]) if len(port_downside) == 1 else 1e-9)
    port_sortino = (port_mean - RF) / port_dd_std if port_dd_std > 1e-9 else 0

    bm_downside = bm_rets[bm_rets < RF] - RF
    bm_dd_std = np.std(bm_downside, ddof=1) if len(bm_downside) > 1 else (abs(bm_downside[0]) if len(bm_downside) == 1 else 1e-9)
    bm_sortino = (bm_mean - RF) / bm_dd_std if bm_dd_std > 1e-9 else 0

    # Max Drawdown (worst single-year, proxy)
    port_maxdd = np.min(port_rets)
    bm_maxdd = np.min(bm_rets) if len(bm_rets) > 0 else 0

    # Calmar
    port_calmar = port_mean / abs(port_maxdd) if abs(port_maxdd) > 1e-9 else 0
    bm_calmar = bm_mean / abs(bm_maxdd) if abs(bm_maxdd) > 1e-9 else 0

    # Win rate
    port_winrate = np.mean(port_rets > 0) * 100
    bm_winrate = np.mean(bm_rets > 0) * 100 if len(bm_rets) > 0 else 0

    # Tracking error, Info Ratio, Beta, Alpha
    alpha_std = np.std(alphas, ddof=1) if len(alphas) > 1 else 1e-9
    tracking_error = alpha_std
    info_ratio = np.mean(alphas) / alpha_std if alpha_std > 1e-9 else 0

    # Beta = cov(port, bm) / var(bm)
    if len(port_rets) > 1 and len(bm_rets) > 1:
        cov_matrix = np.cov(port_rets, bm_rets)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 1e-9 else 0
    else:
        beta = 0

    ann_alpha = np.mean(alphas)

    # MDD Duration: max consecutive losing years
    losing_streak = 0
    max_losing_streak = 0
    for r in port_rets:
        if r < 0:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0

    # VaR 95%
    port_var95 = np.percentile(port_rets, 5)
    bm_var95 = np.percentile(bm_rets, 5) if len(bm_rets) > 0 else 0

    # Annualized turnover from tracked data
    turnover_vals = results_df['avg_turnover'].values
    ann_turnover = np.mean(turnover_vals) * (252 / HORIZON) if np.any(turnover_vals > 0) else 0

    print('\nRisk-Adjusted Performance')
    print('-' * 50)
    print(f'  {"":22} {"Portfolio":>10} {"KOSPI200":>10}')
    print(f'  {"CAGR":<22} {port_mean:>+9.1f}% {bm_mean:>+9.1f}%')
    print(f'  {"Volatility":<22} {port_std:>9.1f}% {bm_std:>9.1f}%')
    print(f'  {"Sharpe (Rf=3%)":<22} {port_sharpe:>+9.2f}  {bm_sharpe:>+9.2f}')
    print(f'  {"Sortino":<22} {port_sortino:>+9.2f}  {bm_sortino:>+9.2f}')
    print(f'  {"Max Drawdown":<22} {port_maxdd:>+9.1f}% {bm_maxdd:>+9.1f}%')
    print(f'  {"MDD Duration (yrs)":<22} {max_losing_streak:>9}')
    print(f'  {"VaR 95%":<22} {port_var95:>+9.1f}% {bm_var95:>+9.1f}%')
    print(f'  {"Calmar Ratio":<22} {port_calmar:>+9.2f}  {bm_calmar:>+9.2f}')
    print(f'  {"Win Rate":<22} {port_winrate:>8.0f}%  {bm_winrate:>8.0f}%')
    print(f'  {"Best Year":<22} {np.max(port_rets):>+9.1f}% {np.max(bm_rets):>+9.1f}%')
    print(f'  {"Worst Year":<22} {port_maxdd:>+9.1f}% {bm_maxdd:>+9.1f}%')
    if ann_turnover > 0:
        print(f'  {"Ann. Turnover":<22} {ann_turnover*100:>8.0f}%')
    print('-' * 50)
    print(f'  {"Tracking Error":<22} {tracking_error:>9.1f}%')
    print(f'  {"Information Ratio":<22} {info_ratio:>+9.2f}')
    print(f'  {"Alpha (ann.)":<22} {ann_alpha:>+9.1f}%')
    print(f'  {"Beta vs KOSPI200":<22} {beta:>9.2f}')

    # Cumulative return tracking
    print('\nCumulative Returns')
    print('-' * 55)
    cum_port = 1.0
    cum_bm = 1.0
    print(f'  {"Year":<6} {"Port Ann":>10} {"Cum Port":>10} {"BM Ann":>10} {"Cum BM":>10}')
    print(f'  {"----":<6} {"-------":>10} {"-------":>10} {"------":>10} {"------":>10}')
    for _, row in valid.iterrows():
        yr = int(row['year'])
        p_ret = row['annual_return']
        b_ret = row['benchmark_return']
        cum_port *= (1 + p_ret / 100)
        cum_bm *= (1 + b_ret / 100)
        print(f'  {yr:<6} {p_ret:>+9.1f}% {(cum_port-1)*100:>+9.1f}% {b_ret:>+9.1f}% {(cum_bm-1)*100:>+9.1f}%')
    print(f'  {"Total":<6} {"":>10} {(cum_port-1)*100:>+9.1f}% {"":>10} {(cum_bm-1)*100:>+9.1f}%')
else:
    port_mean = np.mean(port_rets) if len(port_rets) > 0 else 0
    port_std = 0
    port_sharpe = 0
    port_maxdd = np.min(port_rets) if len(port_rets) > 0 else 0
    port_winrate = 0
    ann_alpha = np.mean(alphas) if len(alphas) > 0 else 0
    print('\nRisk-Adjusted Performance')
    print('-' * 50)
    print('  (Insufficient data for full risk analysis)')

# ────────────────────────────────────────────────────────────────────────────
# Block 4: Regime Analysis
# ────────────────────────────────────────────────────────────────────────────
print('\nRegime Effectiveness')
print('-' * 55)
print(f'  {"Year":<6} {"Regime":>8} {"Cash%":>7} {"StopLoss":>10} {"Return":>10}')
print(f'  {"----":<6} {"------":>8} {"-----":>7} {"--------":>10} {"------":>10}')
for _, row in results_df.iterrows():
    yr = int(row['year'])
    ann = row['annual_return']
    if pd.isna(ann):
        continue
    cr = row['cash_ratio'] * 100
    sl = row.get('stop_loss_impact', 0)
    regime_str = f'{cr:>6.0f}%' if cr > 0 else f'{"0%":>7}'
    sl_str = f'{sl:>+9.1f}%' if sl != 0 else f'{"--":>10}'
    print(f'  {yr:<6} {"active" if cr > 0 else "off":>8} {regime_str} {sl_str} {ann:>+9.1f}%')
avg_cash = results_df['cash_ratio'].mean() * 100
cash_active_years = (results_df['cash_ratio'] > 0).sum()
print(f'\n  Avg cash ratio: {avg_cash:.1f}%   Active years: {cash_active_years}/{len(results_df)}')

# ────────────────────────────────────────────────────────────────────────────
# Block 5: Feature Importance (last model)
# ────────────────────────────────────────────────────────────────────────────
importance = model.feature_importance()

print('\nFeature Importance (Top 15)')
print('-' * 55)

for i, row in importance.head(15).iterrows():
    feature = row['feature']
    imp = row['importance']

    # Group classification
    v8_new = ['inventory_sales_gap', 'current_ratio', 'hvup_ratio']
    if args.v9 and feature == 'short_term_reversal':
        group = 'V9'
    elif args.v8 and feature in v8_new:
        group = 'V8'
    elif (args.v8 or args.v6) and feature in fe.QUALITY_FEATURES:
        group = 'Qual'
    elif (args.v8 or args.v6) and feature in fe.V5_FEATURES:
        group = 'V5'
    elif args.v5:
        group = 'V5'
    elif feature in momentum_features:
        group = 'Mom'
    elif feature in volume_features:
        group = 'Vol'
    elif feature in volatility_features:
        group = 'Risk'
    elif feature in intuition_features:
        group = 'Intu'
    elif feature in traditional_features:
        group = 'Trad'
    elif args.v4 and feature in macro_features:
        group = 'Macr'
    else:
        group = 'Fund'

    bar = '|' * int(imp / importance['importance'].max() * 20)
    print(f'  {feature:<25} [{group:<4}] {bar}')

# Group importance
print('\nFeature Group Importance')
print('-' * 45)

if args.v9:
    v6_no_hvup = [c for c in fe.V6_FEATURES if c != 'hvup_ratio' and c in df.columns]
    v5_in_use = [c for c in fe.V5_FEATURES if c in df.columns]
    quality_in_use = [c for c in fe.QUALITY_FEATURES if c in df.columns]
    v9_new = [c for c in ['short_term_reversal'] if c in df.columns]
    groups = {'V5 Base': v5_in_use, 'Quality': quality_in_use, 'V9 New': v9_new}
elif args.v8:
    v5_in_use = [c for c in fe.V5_FEATURES if c in df.columns]
    quality_in_use = [c for c in fe.QUALITY_FEATURES if c in df.columns]
    v8_new_in_use = [c for c in ['inventory_sales_gap', 'current_ratio', 'hvup_ratio'] if c in df.columns]
    groups = {'V5 Base': v5_in_use, 'Quality': quality_in_use, 'V8 New': v8_new_in_use}
elif args.v6:
    v5_in_use = [c for c in fe.V5_FEATURES if c in df.columns]
    quality_in_use = [c for c in fe.QUALITY_FEATURES if c in df.columns]
    groups = {'V5 Base': v5_in_use, 'Quality': quality_in_use}
elif args.v5:
    groups = {'V5': all_features}
else:
    groups = {
        'Momentum': momentum_features,
        'Volume': volume_features,
        'Volatility': volatility_features,
        'Intuition': intuition_features,
        'Traditional': traditional_features,
        'Fundament.': fund_features,
    }
    if args.v4:
        groups['Macro'] = macro_features

total_imp = importance['importance'].sum()
for group_name, group_features in groups.items():
    group_imp = importance[importance['feature'].isin(group_features)]['importance'].sum()
    pct = group_imp / total_imp * 100
    bar = '|' * int(pct / 5)
    print(f'  {group_name:<12} {pct:>5.1f}% {bar}')

# ────────────────────────────────────────────────────────────────────────────
# Block 5.5: Sector Concentration Analysis
# ────────────────────────────────────────────────────────────────────────────
if all_test_dfs and TOP_N > 0:
    print('\nSector Concentration (Top-N Portfolio)')
    print('-' * 55)
    sector_counts_all = {}  # sector -> total count across years
    sector_warnings = []
    for yr in sorted(all_test_dfs.keys()):
        tdf = all_test_dfs[yr]
        if 'sector' not in tdf.columns:
            continue
        top = tdf[tdf['pred_rank'] <= TOP_N].copy()
        if len(top) == 0:
            continue
        # Per-date sector counts, then average across dates
        dates_in_yr = top['date'].unique()
        yr_sector = {}
        for d in dates_in_yr:
            day_top = top[top['date'] == d]
            for s in day_top['sector'].fillna('Unknown'):
                yr_sector[s] = yr_sector.get(s, 0) + 1
        # Normalize by number of dates
        for s in yr_sector:
            avg_count = yr_sector[s] / len(dates_in_yr)
            sector_counts_all[s] = sector_counts_all.get(s, 0) + avg_count
        # Check concentration per year
        total_holdings = sum(yr_sector.values())
        for s, c in yr_sector.items():
            pct = c / total_holdings * 100
            if pct > 30:
                sector_warnings.append((yr, s, pct))

    # Show top 5 sectors by frequency
    top_sectors = sorted(sector_counts_all.items(), key=lambda x: -x[1])[:5]
    n_years = len(all_test_dfs)
    print(f'  {"Sector":<25} {"Avg Holdings/Year":>18}')
    print(f'  {"------":<25} {"----------------":>18}')
    for s, c in top_sectors:
        print(f'  {s:<25} {c/n_years:>17.1f}')

    if sector_warnings:
        print(f'\n  ⚠️ Concentration Warnings (sector > 30% of portfolio):')
        for yr, s, pct in sector_warnings:
            print(f'    {yr}: {s} = {pct:.0f}%')
    else:
        print(f'\n  ✅ No sector exceeds 30% concentration')

# ────────────────────────────────────────────────────────────────────────────
# Block 6: Diagnostics / Verdict
# ────────────────────────────────────────────────────────────────────────────
print('\nStrategy Diagnostics')
print('-' * 50)

checks = []

# 1. ICIR > 0.5
icir_pass = icir > 0.5
checks.append(('ICIR > 0.5', icir_pass, f'{icir:.2f}'))

# 2. IC hit rate > 60%
ic_hit_pass = ic_hit_rate > 60
checks.append(('IC hit rate > 60%', ic_hit_pass, f'{ic_hit_rate:.0f}%'))

# 3. Alpha > 0
alpha_pass = ann_alpha > 0
checks.append(('Alpha > 0', alpha_pass, f'{ann_alpha:+.1f}%'))

# 4. Sharpe > 0.3
sharpe_val = port_sharpe if len(port_rets) > 1 else 0
sharpe_pass = sharpe_val > 0.3
checks.append(('Sharpe > 0.3', sharpe_pass, f'{sharpe_val:+.2f}'))

# 5. MaxDD > -25%
maxdd_val = port_maxdd if len(port_rets) > 0 else 0
maxdd_pass = maxdd_val > -25
checks.append(('MaxDD > -25%', maxdd_pass, f'{maxdd_val:+.1f}%'))

# 6. Win rate > 50%
winrate_val = port_winrate if len(port_rets) > 1 else 0
winrate_pass = winrate_val > 50
checks.append(('Win rate > 50%', winrate_pass, f'{winrate_val:.0f}%'))

passed = 0
for label, ok, val in checks:
    tag = '[PASS]' if ok else '[FAIL]'
    print(f'  {tag} {label:<22}: {val}')
    if ok:
        passed += 1

total_checks = len(checks)
print('-' * 50)

# Verdict
ic_ok = icir_pass and ic_hit_pass
alpha_ok = alpha_pass and sharpe_pass
if passed == total_checks:
    verdict = 'All checks passed. Strategy is robust.'
elif ic_ok and not alpha_ok:
    verdict = f'Score: {passed}/{total_checks} -- Model has signal (IC), but portfolio construction destroys alpha.'
elif alpha_ok and not ic_ok:
    verdict = f'Score: {passed}/{total_checks} -- Returns OK, but IC quality is weak (may be lucky).'
elif passed >= total_checks * 0.5:
    verdict = f'Score: {passed}/{total_checks} -- Mixed results. Further investigation needed.'
else:
    verdict = f'Score: {passed}/{total_checks} -- Strategy needs significant rework.'
print(f'  {verdict}')

# Save results
results_df.to_csv('backtest_v2_results.csv', index=False)
print(f'\nResults saved: backtest_v2_results.csv')

print(f'\n  [3/3] Report: {time.time()-stage_t0:.1f}s')
print('=' * 70)
print('Backtest complete.')
print('=' * 70)
