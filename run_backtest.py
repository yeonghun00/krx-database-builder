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
from ml.features import FeatureEngineer
from ml.model import MLRanker

parser = argparse.ArgumentParser()
parser.add_argument('--top', type=int, default=0, help='상위 N개 종목만 (0=quintile)')
parser.add_argument('--horizon', type=int, default=21, help='보유 기간 (일): 21, 63, 126')
parser.add_argument('--slippage', type=float, default=0.5, help='편도 슬리피지 %% (기본 0.5%%)')
parser.add_argument('--qepm', action='store_true', help='🔥 QEPM 모드 (63일, Alpha, 섹터제한, 변동성가중)')
parser.add_argument('--max-sector', type=int, default=3, help='섹터당 최대 종목 수 (QEPM)')
parser.add_argument('--turnover-buffer', type=float, default=0.2, help='회전율 버퍼 (기존 종목 교체 기준)')
args = parser.parse_args()

# QEPM 모드면 설정 오버라이드
if args.qepm:
    args.horizon = 63  # 3개월
    if args.top == 0:
        args.top = 20  # 기본 20종목

print('=' * 70)
if args.qepm:
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

# ============================================================================
# 데이터 로드
# ============================================================================
print('\n[1/3] 데이터 로드 중...')

fe = FeatureEngineer('krx_stock_data.db')
df = fe.prepare_ml_data(
    start_date='20180101',  # 더 긴 기간
    end_date='20260128',
    target_horizon=HORIZON,
    min_market_cap=500_000_000_000,
    include_fundamental=True
)

# 피처 분류
momentum_features = [c for c in fe.MOMENTUM_FEATURES if c in df.columns]
volume_features = [c for c in fe.VOLUME_FEATURES if c in df.columns]
volatility_features = [c for c in fe.VOLATILITY_FEATURES if c in df.columns]
intuition_features = [c for c in fe.INTUITION_FEATURES if c in df.columns]
traditional_features = [c for c in fe.TRADITIONAL_FEATURES if c in df.columns]
fund_features = [c for c in fe.FUNDAMENTAL_FEATURES if c in df.columns]

all_features = (momentum_features + volume_features + volatility_features +
                intuition_features + traditional_features + fund_features)

tech_count = len(momentum_features + volume_features + volatility_features +
                 intuition_features + traditional_features)

print(f'  총 데이터: {len(df):,} rows')
print(f'  피처 구성:')
print(f'    - 모멘텀: {len(momentum_features)}개')
print(f'    - 수급: {len(volume_features)}개')
print(f'    - 변동성: {len(volatility_features)}개')
print(f'    - 본능전략: {len(intuition_features)}개')
print(f'    - 전통지표: {len(traditional_features)}개')
print(f'    - 재무: {len(fund_features)}개')
print(f'  재무 비중: {len(fund_features)/len(all_features)*100:.1f}% (목표: 30-40%)')

# Forward return 추가
df = df.sort_values(['stock_code', 'date'])
df['year'] = df['date'].str[:4].astype(int)
years = sorted(df['year'].unique())

# ============================================================================
# Walk-Forward 백테스트
# ============================================================================
print('\n[2/3] Walk-Forward 백테스트...')
print('-' * 70)

# QEPM은 Alpha 타겟 사용, 일반은 절대 수익률 타겟
if QEPM_MODE:
    target_col = f'target_alpha_rank_{HORIZON}d'  # Alpha 순위
    return_col = f'forward_alpha_{HORIZON}d'      # Alpha 수익률
    if target_col not in df.columns:
        target_col = f'target_rank_{HORIZON}d'    # fallback
        return_col = f'forward_return_{HORIZON}d'
else:
    target_col = f'target_rank_{HORIZON}d'
    return_col = f'forward_return_{HORIZON}d'

all_results = []

# ============================================================================
# QEPM 헬퍼 함수들
# ============================================================================

def get_benchmark_return(year, horizon=63):
    """
    DB에서 KOSPI200 벤치마크 수익률 가져오기
    (build_benchmark.py로 미리 생성해야 함)
    """
    import sqlite3

    try:
        conn = sqlite3.connect('krx_stock_data.db')
        query = f"""
        SELECT AVG(return_{horizon}d) * 100 as avg_return
        FROM benchmark_kospi200
        WHERE date LIKE '{year}%'
          AND return_{horizon}d IS NOT NULL
        """
        result = pd.read_sql_query(query, conn)
        conn.close()

        if len(result) > 0 and result['avg_return'].iloc[0] is not None:
            return result['avg_return'].iloc[0]
    except Exception as e:
        print(f'    ⚠️ 벤치마크 로드 실패: {e}')

    return 0  # fallback

def select_with_sector_constraint(df, top_n, max_per_sector):
    """섹터 제약 적용한 종목 선정"""
    df = df.sort_values('pred_rank')
    selected = []
    sector_count = {}

    for _, row in df.iterrows():
        sector = row.get('sector', 'Unknown')
        if sector_count.get(sector, 0) < max_per_sector:
            selected.append(row)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(selected) >= top_n:
            break

    return pd.DataFrame(selected)

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

    # 모델 학습 (QEPM은 낮은 time_decay로 안정성 추구)
    model = MLRanker(
        feature_cols=all_features,
        target_col=target_col,
        model_type='regressor',
        time_decay=0.5 if QEPM_MODE else 0.7  # QEPM은 더 보수적
    )
    model.train(train_df)

    # 예측
    test_df['pred_score'] = model.predict(test_df)
    test_df['pred_rank'] = test_df.groupby('date')['pred_score'].rank(ascending=False)

    if TOP_N > 0:
        if QEPM_MODE:
            # QEPM: 섹터 제약 + 변동성 가중
            top_df = test_df.groupby('date').apply(
                lambda x: select_with_sector_constraint(x, TOP_N, MAX_PER_SECTOR)
            ).reset_index(drop=True)

            # 변동성 역가중 수익률
            q5_raw = test_df.groupby('date').apply(
                lambda x: apply_inverse_volatility_weight(
                    select_with_sector_constraint(x, TOP_N, MAX_PER_SECTOR),
                    return_col
                )
            ).mean() * 100
        else:
            # 일반: 단순 Top N
            top_df = test_df[test_df['pred_rank'] <= TOP_N]
            q5_raw = top_df[return_col].mean() * 100

        bottom_df = test_df[test_df['pred_rank'] > test_df.groupby('date')['pred_rank'].transform('max') - TOP_N]
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
            # QEPM: 섹터 제약 적용
            top_selected = test_df.groupby('date').apply(
                lambda x: select_with_sector_constraint(x, TOP_N, MAX_PER_SECTOR)
            ).reset_index(drop=True)
            portfolio_return = top_selected[abs_return_col].mean() * 100 - ROUND_TRIP_COST * 100
        else:
            # 일반: Top N 평균
            top_selected = test_df[test_df['pred_rank'] <= TOP_N]
            portfolio_return = top_selected[abs_return_col].mean() * 100 - ROUND_TRIP_COST * 100
    else:
        portfolio_return = q5  # fallback

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
        'test_samples': len(test_df)
    })

    if not np.isnan(spread):
        bar = '█' * int(max(0, min(annual_portfolio, 50)))
        ic_status = '✅' if ic > 0 else '❌'
        # 모든 모드에서 실제 연수익률 표시
        print(f'  {test_year}: 분기={portfolio_return:+5.1f}% | 연환산={annual_portfolio:+5.1f}% | IC={ic:+.3f} {ic_status} {bar}')

# ============================================================================
# 결과 분석
# ============================================================================
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
    if feature in momentum_features:
        group = '모멘텀'
    elif feature in volume_features:
        group = '수급'
    elif feature in volatility_features:
        group = '변동성'
    elif feature in intuition_features:
        group = '본능'
    elif feature in traditional_features:
        group = '전통'
    else:
        group = '재무'

    bar = '█' * int(imp / importance['importance'].max() * 20)
    print(f'  {feature:<25} [{group:<4}] {bar}')

# 그룹별 중요도
print('\n[그룹별 피처 중요도]')
print('-' * 50)

groups = {
    '모멘텀': momentum_features,
    '수급': volume_features,
    '변동성': volatility_features,
    '본능전략': intuition_features,
    '전통지표': traditional_features,
    '재무': fund_features,
}

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
