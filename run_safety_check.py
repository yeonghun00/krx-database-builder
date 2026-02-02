#!/usr/bin/env python3
"""
V2 모델 안전성 점검 리포트 (전체 기간, 모든 Horizon)

5가지 위험 신호 체크 x 3개 Horizon (1/3/6개월)

Usage:
    python3 run_safety_check_v2.py
"""

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from ml.features import FeatureEngineer
from ml.model import MLRanker

print('=' * 70)
print('🛡️  V2 모델 안전성 종합 점검')
print(f'    {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('=' * 70)

# ============================================================================
# 설정
# ============================================================================
HORIZONS = [21, 63, 126]
HORIZON_NAMES = {21: '1개월', 63: '3개월', 126: '6개월'}
TRAIN_YEARS = 3
TEST_YEAR = 2025

# ============================================================================
# 데이터 로드
# ============================================================================
print('\n[준비] 데이터 로드 중...')

fe = FeatureEngineer('krx_stock_data.db')
df = fe.prepare_ml_data(
    start_date='20200101',
    end_date='20260128',
    target_horizon=21,
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

# Forward returns 계산
df = df.sort_values(['stock_code', 'date'])
grouped = df.groupby('stock_code')

for horizon in HORIZONS + [1, 5]:
    col = f'forward_return_{horizon}d'
    if col not in df.columns:
        df[col] = grouped['closing_price'].transform(
            lambda x: x.pct_change(horizon).shift(-horizon)
        )

    target_col = f'target_rank_{horizon}d'
    if target_col not in df.columns:
        df[target_col] = df.groupby('date')[col].rank(pct=True)

df['year'] = df['date'].str[:4].astype(int)
df['month'] = df['date'].str[:6]

print(f'    총 데이터: {len(df):,} rows')
print(f'    피처: {len(all_features)}개')

# ============================================================================
# 각 Horizon별 분석
# ============================================================================

all_checks = {}

for horizon in HORIZONS:
    horizon_name = HORIZON_NAMES[horizon]
    target_col = f'target_rank_{horizon}d'
    return_col = f'forward_return_{horizon}d'

    print('\n' + '=' * 70)
    print(f'📊 {horizon_name} ({horizon}일) Horizon 분석')
    print('=' * 70)

    # Buffer 계산
    buffer_month = 12 - (horizon // 21 + 1)
    buffer_month = max(1, min(buffer_month, 9))
    buffer_date = f'2024{buffer_month:02d}01'

    # Train/Test 분리
    train_df = df[(df['year'] <= 2024) & (df['date'] <= buffer_date)].copy()
    test_df = df[df['year'] == TEST_YEAR].copy()

    print(f'\n    학습: {len(train_df):,} rows (~{buffer_date})')
    print(f'    테스트: {len(test_df):,} rows (2025)')

    # 모델 학습
    model = MLRanker(
        feature_cols=all_features,
        target_col=target_col,
        model_type='regressor',
        time_decay=0.7
    )
    model.train(train_df)

    test_df['pred_score'] = model.predict(test_df)
    test_df['pred_rank'] = test_df.groupby('date')['pred_score'].rank(pct=True)
    test_df['quintile'] = pd.qcut(test_df['pred_rank'], 5, labels=[1,2,3,4,5], duplicates='drop')

    # =========================================================================
    # 1. IC 분석
    # =========================================================================
    print(f'\n  [1] IC (Information Coefficient) 분석')
    print('  ' + '-' * 50)

    monthly_ic = []
    for month, group in test_df.groupby('month'):
        if len(group) < 50:
            continue
        ic, _ = stats.spearmanr(
            group['pred_score'].fillna(0),
            group[return_col].fillna(0)
        )
        monthly_ic.append({'month': month, 'ic': ic})

        bar = '█' * int(max(0, ic * 50)) if ic > 0 else '░' * int(max(0, -ic * 50))
        status = '✅' if ic > 0.02 else '⚠️' if ic > 0 else '❌'
        print(f"    {month}: IC = {ic:+.3f} {status} {bar}")

    ic_df = pd.DataFrame(monthly_ic)
    avg_ic = ic_df['ic'].mean() if len(ic_df) > 0 else 0
    ic_positive_rate = (ic_df['ic'] > 0).mean() * 100 if len(ic_df) > 0 else 0

    print(f'\n    평균 IC: {avg_ic:+.3f}')
    print(f'    IC 양수율: {ic_positive_rate:.0f}%')

    # IC Decay
    print(f'\n  [IC Decay]')
    decay_results = []
    for days in [1, 5, 21, 63, 126]:
        col = f'forward_return_{days}d'
        if col in test_df.columns:
            ic, _ = stats.spearmanr(
                test_df['pred_score'].fillna(0),
                test_df[col].fillna(0)
            )
            decay_results.append({'days': days, 'ic': ic})
            marker = '◀' if days == horizon else ''
            print(f'    {days:>3}일: IC = {ic:+.3f} {marker}')

    # =========================================================================
    # 2. 종목/섹터 쏠림
    # =========================================================================
    print(f'\n  [2] 종목/섹터 쏠림 분석')
    print('  ' + '-' * 50)

    q5_df = test_df[test_df['quintile'] == 5].copy()
    q5_df['contribution'] = q5_df[return_col].fillna(0)

    stock_contrib = q5_df.groupby('stock_code').agg({
        'contribution': 'sum',
        'name': 'first'
    }).sort_values('contribution', ascending=False)

    total_return = stock_contrib['contribution'].sum()
    if total_return != 0:
        stock_contrib['pct'] = stock_contrib['contribution'] / total_return * 100
    else:
        stock_contrib['pct'] = 0

    print('    [Top 5 종목 기여도]')
    for i, (code, row) in enumerate(stock_contrib.head(5).iterrows(), 1):
        name = str(row['name'])[:8] if pd.notna(row['name']) else 'N/A'
        print(f'      {i}. {code} {name}: {row["pct"]:+.1f}%')

    top1_pct = stock_contrib['pct'].iloc[0] if len(stock_contrib) > 0 else 0
    top3_pct = stock_contrib['pct'].head(3).sum() if len(stock_contrib) >= 3 else 0

    # 섹터
    def classify_sector(name):
        if pd.isna(name):
            return '기타'
        name = str(name)
        if any(k in name for k in ['조선', '중공업', '해양', 'HD현대']):
            return '조선'
        elif any(k in name for k in ['금융', '은행', '증권', '보험', '지주']):
            return '금융'
        elif any(k in name for k in ['반도체', '하이닉스', '삼성전자']):
            return '반도체'
        elif any(k in name for k in ['바이오', '제약', '셀']):
            return '바이오'
        elif any(k in name for k in ['배터리', '이차전지', '에코프로']):
            return '2차전지'
        else:
            return '기타'

    q5_df['sector_inferred'] = q5_df['name'].apply(classify_sector)
    sector_pct = q5_df.groupby('sector_inferred').size() / len(q5_df) * 100
    max_sector_pct = sector_pct.max() if len(sector_pct) > 0 else 0

    print(f'\n    Top 1 종목 집중도: {top1_pct:.1f}%')
    print(f'    최대 섹터 집중도: {max_sector_pct:.1f}%')

    # =========================================================================
    # 3. MDD 분석
    # =========================================================================
    print(f'\n  [3] MDD (Maximum Drawdown) 분석')
    print('  ' + '-' * 50)

    test_df_sorted = test_df.sort_values(['stock_code', 'date'])
    test_df_sorted['daily_return'] = test_df_sorted.groupby('stock_code')['closing_price'].pct_change()

    q5_daily = test_df_sorted[test_df_sorted['quintile'] == 5].groupby('date')['daily_return'].mean()
    q5_daily = q5_daily.sort_index().fillna(0)

    if len(q5_daily) > 1:
        cumulative = (1 + q5_daily).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        mdd = drawdown.min()
        q5_total_return = (cumulative.iloc[-1] / cumulative.iloc[0] - 1) * 100
    else:
        mdd = 0
        q5_total_return = 0

    print(f'    MDD: {mdd*100:.1f}%')
    print(f'    Q5 총 수익률: {q5_total_return:+.1f}%')

    # =========================================================================
    # 4. 슬리피지 민감도
    # =========================================================================
    print(f'\n  [4] 슬리피지 민감도')
    print('  ' + '-' * 50)

    base_q5 = test_df[test_df['quintile'] == 5][return_col].mean() * 100
    base_q1 = test_df[test_df['quintile'] == 1][return_col].mean() * 100
    base_spread = base_q5 - base_q1

    # 보유기간에 따른 매매 횟수 조정
    trades_per_year = 252 / horizon

    for slippage in [0, 0.3, 0.5, 1.0]:
        cost = slippage * 2 * (trades_per_year / 4)  # 분기당 비용
        adjusted_spread = base_spread - cost
        status = '✅' if adjusted_spread > base_spread * 0.5 else '⚠️'
        print(f'    {slippage:.1f}%: Spread {adjusted_spread:+.1f}% {status}')

    spread_05 = base_spread - (0.5 * 2 * trades_per_year / 4)

    # =========================================================================
    # 5. 데이터 누수 체크
    # =========================================================================
    print(f'\n  [5] 데이터 누수 체크')
    print('  ' + '-' * 50)

    test_df_sorted['yesterday_return'] = test_df_sorted.groupby('stock_code')['closing_price'].pct_change()
    leakage_corr, _ = stats.spearmanr(
        test_df_sorted['pred_score'].fillna(0),
        test_df_sorted['yesterday_return'].fillna(0)
    )
    print(f'    어제 수익률 상관: {leakage_corr:+.3f}')

    # =========================================================================
    # 체크리스트
    # =========================================================================
    checks = {
        'IC 안정성': avg_ic >= 0.02 and ic_positive_rate >= 50,
        '종목 집중도': top1_pct < 30,
        '섹터 집중도': max_sector_pct < 50,
        'MDD': mdd > -0.30,
        '슬리피지': spread_05 > 0,
        '데이터 누수': abs(leakage_corr) < 0.2,
    }

    print(f'\n  [체크리스트]')
    print('  ' + '-' * 50)
    passed = 0
    for check, result in checks.items():
        status = '✅ PASS' if result else '❌ FAIL'
        print(f'    {check:<15} {status}')
        if result:
            passed += 1

    print('  ' + '-' * 50)
    print(f'    통과: {passed}/{len(checks)}')

    all_checks[horizon_name] = {
        'avg_ic': avg_ic,
        'ic_positive_rate': ic_positive_rate,
        'mdd': mdd,
        'spread': base_spread,
        'passed': passed,
        'total': len(checks)
    }

# ============================================================================
# 종합 리포트
# ============================================================================
print('\n' + '=' * 70)
print('📋 V2 모델 종합 안전성 리포트')
print('=' * 70)

print('\n[Horizon별 요약]')
print('-' * 70)
print(f'{"Horizon":<10} {"평균 IC":>10} {"IC양수율":>10} {"MDD":>10} {"Spread":>10} {"통과":>8}')
print('-' * 70)

for horizon_name, data in all_checks.items():
    print(f'{horizon_name:<10} {data["avg_ic"]:>+9.3f} {data["ic_positive_rate"]:>9.0f}% {data["mdd"]*100:>9.1f}% {data["spread"]:>+9.1f}% {data["passed"]}/{data["total"]}')

print('-' * 70)

# 최종 판정
total_passed = sum(d['passed'] for d in all_checks.values())
total_checks = sum(d['total'] for d in all_checks.values())
avg_ic_all = sum(d['avg_ic'] for d in all_checks.values()) / len(all_checks)

print('\n[최종 판정]')
if total_passed >= total_checks * 0.8:
    print(f'  🔥 V2 모델 안전! ({total_passed}/{total_checks} 통과)')
elif total_passed >= total_checks * 0.6:
    print(f'  ✅ V2 모델 양호 ({total_passed}/{total_checks} 통과)')
else:
    print(f'  ⚠️ V2 모델 주의 필요 ({total_passed}/{total_checks} 통과)')

print(f'\n  전체 평균 IC: {avg_ic_all:+.3f}')

print('\n' + '=' * 70)
print('V2 안전성 점검 완료!')
print('=' * 70)
