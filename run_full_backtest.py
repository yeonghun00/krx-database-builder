#!/usr/bin/env python3
"""
V2 모델 전체 백테스트 (1개월, 3개월, 6개월)

Usage:
    python3 run_full_backtest_v2.py
"""

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from ml.features import FeatureEngineer
from ml.model import MLRanker

# 설정
HORIZONS = [21, 63, 126]  # 1개월, 3개월, 6개월
HORIZON_NAMES = {21: '1개월', 63: '3개월', 126: '6개월'}
TRAIN_YEARS = 3

print('=' * 70)
print('🚀 V2 모델 종합 백테스트 (1/3/6개월)')
print(f'   {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('=' * 70)

# ============================================================================
# 데이터 로드
# ============================================================================
print('\n[1/4] 데이터 로드 중...')

fe = FeatureEngineer('krx_stock_data.db')
df_base = fe.prepare_ml_data(
    start_date='20150101',
    end_date='20260128',
    target_horizon=21,  # 일단 21일로 로드
    min_market_cap=500_000_000_000,
    include_fundamental=True
)

# 피처 분류
momentum_features = [c for c in fe.MOMENTUM_FEATURES if c in df_base.columns]
volume_features = [c for c in fe.VOLUME_FEATURES if c in df_base.columns]
volatility_features = [c for c in fe.VOLATILITY_FEATURES if c in df_base.columns]
intuition_features = [c for c in fe.INTUITION_FEATURES if c in df_base.columns]
traditional_features = [c for c in fe.TRADITIONAL_FEATURES if c in df_base.columns]
fund_features = [c for c in fe.FUNDAMENTAL_FEATURES if c in df_base.columns]

all_features = (momentum_features + volume_features + volatility_features +
                intuition_features + traditional_features + fund_features)

print(f'   총 데이터: {len(df_base):,} rows')
print(f'   피처: {len(all_features)}개')

# ============================================================================
# Forward Return 계산
# ============================================================================
print('\n[2/4] Forward Return 계산 중...')

df_base = df_base.sort_values(['stock_code', 'date'])
grouped = df_base.groupby('stock_code')

for horizon in HORIZONS:
    col_name = f'forward_return_{horizon}d'
    if col_name not in df_base.columns:
        df_base[col_name] = grouped['closing_price'].transform(
            lambda x: x.pct_change(horizon).shift(-horizon)
        )
        df_base[col_name] = df_base[col_name].clip(-0.5, 0.5)

    target_col = f'target_rank_{horizon}d'
    df_base[target_col] = df_base.groupby('date')[col_name].rank(pct=True)

print(f'   Forward returns 계산 완료: {HORIZONS}')

df_base['year'] = df_base['date'].str[:4].astype(int)
years = sorted(df_base['year'].unique())

# ============================================================================
# Walk-Forward 백테스트
# ============================================================================
print('\n[3/4] Walk-Forward 백테스트...')
print('-' * 70)

all_results = []

for horizon in HORIZONS:
    horizon_name = HORIZON_NAMES[horizon]
    target_col = f'target_rank_{horizon}d'
    return_col = f'forward_return_{horizon}d'

    # Buffer 계산 (horizon 거래일 + 여유)
    buffer_month = 12 - (horizon // 21 + 1)
    buffer_month = max(1, min(buffer_month, 9))

    print(f'\n  [{horizon_name} ({horizon}일)]')

    for test_year in years:
        train_start = test_year - TRAIN_YEARS
        train_end = test_year - 1

        train_cutoff = f'{train_end}{buffer_month:02d}01'

        train_df = df_base[(df_base['year'] >= train_start) &
                           (df_base['year'] <= train_end) &
                           (df_base['date'] <= train_cutoff)].copy()
        test_df = df_base[df_base['year'] == test_year].copy()

        if len(train_df) < 1000 or len(test_df) < 100:
            continue

        # 모델 학습
        model = MLRanker(
            feature_cols=all_features,
            target_col=target_col,
            model_type='regressor',
            time_decay=0.7
        )
        model.train(train_df)

        # 예측
        test_df['pred_score'] = model.predict(test_df)
        test_df['pred_rank'] = test_df.groupby('date')['pred_score'].rank(pct=True)

        # Quintile 수익률
        test_df['quintile'] = pd.qcut(
            test_df['pred_rank'], 5,
            labels=[1,2,3,4,5], duplicates='drop'
        )

        quintile_ret = test_df.groupby('quintile')[return_col].mean() * 100

        q1 = quintile_ret.get(1, np.nan)
        q5 = quintile_ret.get(5, np.nan)
        spread = q5 - q1 if not (np.isnan(q5) or np.isnan(q1)) else np.nan

        # IC 계산
        ic, _ = stats.spearmanr(
            test_df['pred_score'].fillna(0),
            test_df[return_col].fillna(0)
        )

        all_results.append({
            'horizon': horizon_name,
            'horizon_days': horizon,
            'year': test_year,
            'Q1': q1,
            'Q5': q5,
            'spread': spread,
            'IC': ic
        })

        if not np.isnan(spread):
            bar = '█' * int(max(0, min(spread + 5, 15)))
            print(f'    {test_year}: Q1={q1:+5.1f}% Q5={q5:+5.1f}% Spread={spread:+5.1f}% IC={ic:+.3f} {bar}')

# ============================================================================
# 결과 분석
# ============================================================================
results_df = pd.DataFrame(all_results)

print('\n' + '=' * 70)
print('📊 V2 종합 백테스트 결과')
print('=' * 70)

# Pivot table
print('\n[연도별 Spread 요약]')
print('-' * 70)

pivot = results_df.pivot_table(
    index='year',
    columns='horizon',
    values='spread',
    aggfunc='first'
)
pivot = pivot[[HORIZON_NAMES[h] for h in HORIZONS]]
print(pivot.round(1).to_string())

# IC Pivot
print('\n[연도별 IC 요약]')
print('-' * 70)

ic_pivot = results_df.pivot_table(
    index='year',
    columns='horizon',
    values='IC',
    aggfunc='first'
)
ic_pivot = ic_pivot[[HORIZON_NAMES[h] for h in HORIZONS]]
print(ic_pivot.round(3).to_string())

# 보유 기간별 성과 요약
print('\n' + '-' * 70)
print('[보유 기간별 성과 요약]')
print('-' * 70)
print(f'{"보유기간":<10} {"평균 Q5":>10} {"평균 Q1":>10} {"평균 Spread":>12} {"연환산":>10} {"평균 IC":>10} {"IC양수율":>8}')
print('-' * 70)

summary_data = []
for horizon in HORIZONS:
    horizon_name = HORIZON_NAMES[horizon]
    h_results = results_df[results_df['horizon_days'] == horizon].dropna(subset=['spread'])

    if len(h_results) == 0:
        continue

    avg_q5 = h_results['Q5'].mean()
    avg_q1 = h_results['Q1'].mean()
    avg_spread = h_results['spread'].mean()
    avg_ic = h_results['IC'].mean()
    ic_positive = (h_results['IC'] > 0).sum() / len(h_results) * 100

    # 연환산
    periods_per_year = 252 / horizon
    annual_spread = avg_spread * periods_per_year

    print(f'{horizon_name:<10} {avg_q5:>+9.1f}% {avg_q1:>+9.1f}% {avg_spread:>+11.1f}% {annual_spread:>+9.1f}% {avg_ic:>+9.3f} {ic_positive:>7.0f}%')

    summary_data.append({
        '보유기간': horizon_name,
        'horizon_days': horizon,
        '평균 Q5': avg_q5,
        '평균 Q1': avg_q1,
        '평균 Spread': avg_spread,
        '연환산 Spread': annual_spread,
        '평균 IC': avg_ic,
        'IC 양수율': ic_positive
    })

print('-' * 70)

# 최종 판정
print('\n[최종 판정]')
print('-' * 70)

best_horizon = max(summary_data, key=lambda x: x['평균 IC'])
print(f'✅ 최적 보유기간: {best_horizon["보유기간"]} (IC {best_horizon["평균 IC"]:+.3f}, Spread {best_horizon["평균 Spread"]:+.1f}%)')

avg_ic_all = sum(d['평균 IC'] for d in summary_data) / len(summary_data)
avg_ic_positive = sum(d['IC 양수율'] for d in summary_data) / len(summary_data)

if avg_ic_all >= 0.05 and avg_ic_positive >= 70:
    print('🔥 모델 성능: 우수 (평균 IC 0.05+, IC 양수율 70%+)')
elif avg_ic_all >= 0.02 and avg_ic_positive >= 50:
    print('✅ 모델 성능: 양호 (평균 IC 0.02+)')
else:
    print('⚠️ 모델 성능: 개선 필요')

avg_annual = sum(d['연환산 Spread'] for d in summary_data) / len(summary_data)
if avg_annual >= 20:
    print(f'🔥 수익성: 우수 (평균 연환산 Spread {avg_annual:+.1f}%)')
elif avg_annual >= 10:
    print(f'✅ 수익성: 양호 (평균 연환산 Spread {avg_annual:+.1f}%)')
else:
    print(f'⚠️ 수익성: 보통 (평균 연환산 Spread {avg_annual:+.1f}%)')

# 저장
results_df.to_csv('backtest_v2_full_results.csv', index=False)
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('backtest_v2_summary.csv', index=False)

print('\n[파일 저장]')
print('  - 상세 결과: backtest_v2_full_results.csv')
print('  - 요약: backtest_v2_summary.csv')

print('\n' + '=' * 70)
print('V2 종합 백테스트 완료!')
print('=' * 70)
