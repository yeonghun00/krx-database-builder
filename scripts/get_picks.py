#!/usr/bin/env python3
"""
V3 QEPM 오늘의 추천 종목

Usage:
    python3 get_picks.py                    # 일반 모드 (상위 20개)
    python3 get_picks.py --qepm             # 🔥 QEPM 모드 (Top 10, 섹터제한)
    python3 get_picks.py --qepm --top 10    # QEPM + Top 10
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

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.features import FeatureEngineer
from ml.model import MLRanker

parser = argparse.ArgumentParser()
parser.add_argument('--top', type=int, default=20, help='상위 N개 종목')
parser.add_argument('--bottom', type=int, default=10, help='하위 N개 종목')
parser.add_argument('--horizon', type=int, default=21, help='보유 기간 (일)')
parser.add_argument('--qepm', action='store_true', help='🔥 QEPM 모드 (63일, Alpha, 섹터제한)')
parser.add_argument('--max-sector', type=int, default=3, help='섹터당 최대 종목 수')
parser.add_argument('--v7', action='store_true', help='🔥 V7 5-Pillar Only 모델')
parser.add_argument('--no-cache', action='store_true', help='캐시 미사용 (cold run 강제)')
args = parser.parse_args()

# QEPM 모드 설정
if args.qepm:
    args.horizon = 63
    if args.top == 20:  # 기본값이면 10으로
        args.top = 10

HORIZON = args.horizon
QEPM_MODE = args.qepm
MAX_PER_SECTOR = args.max_sector

# Alpha 타겟 vs 절대 수익률 타겟
TARGET_COL = f'target_alpha_rank_{HORIZON}d' if QEPM_MODE else f'target_rank_{HORIZON}d'

print('=' * 70)
if QEPM_MODE:
    print('🏦 V3 QEPM 오늘의 추천 종목 (기관급)')
elif args.v7:
    print('🔥 V7 5-Pillar Only 오늘의 추천 종목')
else:
    print('🎯 V3 모델 오늘의 추천 종목')
print('=' * 70)

# 데이터 로드
print('\n[1/3] 데이터 로드 중...')
fe = FeatureEngineer('krx_stock_data.db')

# 학습 데이터 (최근 3년)
train_df = fe.prepare_ml_data(
    start_date='20220101',
    end_date='20260128',
    target_horizon=HORIZON,
    min_market_cap=500_000_000_000,
    include_fundamental=True,
    use_cache=not args.no_cache
)

# 피처
momentum_features = [c for c in fe.MOMENTUM_FEATURES if c in train_df.columns]
volume_features = [c for c in fe.VOLUME_FEATURES if c in train_df.columns]
volatility_features = [c for c in fe.VOLATILITY_FEATURES if c in train_df.columns]
intuition_features = [c for c in fe.INTUITION_FEATURES if c in train_df.columns]
traditional_features = [c for c in fe.TRADITIONAL_FEATURES if c in train_df.columns]
fund_features = [c for c in fe.FUNDAMENTAL_FEATURES if c in train_df.columns]

all_features = (momentum_features + volume_features + volatility_features +
                intuition_features + traditional_features + fund_features)

if args.v7:
    all_features = [c for c in fe.MODEL7_FEATURES if c in train_df.columns]
    missing_v7 = [c for c in fe.MODEL7_FEATURES if c not in train_df.columns]
    if missing_v7:
        print(f'  ⚠️ Missing V7 features: {missing_v7}')

# 모델 학습 (최근 데이터까지)
print('[2/3] 모델 학습 중...')

# Target fallback (Alpha 타겟이 없으면 일반 타겟 사용)
if TARGET_COL not in train_df.columns:
    TARGET_COL = f'target_rank_{HORIZON}d'

model = MLRanker(
    feature_cols=all_features,
    target_col=TARGET_COL,
    model_type='regressor',
    time_decay=0.5 if QEPM_MODE else 0.7  # QEPM은 보수적
)

# 학습 (가장 최근 날짜 제외)
latest_date = train_df['date'].max()
model_train_df = train_df[train_df['date'] < latest_date].copy()
model.train(model_train_df)

# 최신 날짜 예측
print('[3/3] 종목 스코어링 중...')
latest_df = train_df[train_df['date'] == latest_date].copy()
latest_df['score'] = model.predict(latest_df)
latest_df['rank'] = latest_df['score'].rank(ascending=False).astype(int)

print(f'\n기준일: {latest_date}')
print(f'유니버스: {len(latest_df)}개 종목 (시총 5000억+)')
print(f'보유 기간: {HORIZON}일 ({HORIZON//21}개월)')
if QEPM_MODE:
    print(f'모드: 🏦 QEPM (Alpha 타겟, 섹터당 최대 {MAX_PER_SECTOR}종목)')

# ============================================================================
# QEPM: 섹터 제약 적용 함수
# ============================================================================
def apply_sector_constraint(df, top_n, max_per_sector):
    """섹터당 최대 종목 수 제한"""
    df_sorted = df.sort_values('rank')
    selected = []
    sector_count = {}

    for _, row in df_sorted.iterrows():
        sector = row.get('sector', 'Unknown')
        if pd.isna(sector):
            sector = 'Unknown'
        if sector_count.get(sector, 0) < max_per_sector:
            selected.append(row)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(selected) >= top_n:
            break

    return pd.DataFrame(selected)

# ============================================================================
# Top N 출력
# ============================================================================
print(f'\n{"="*70}')
if QEPM_MODE:
    print(f'🏦 QEPM 매수 추천 (상위 {args.top}개, 섹터당 최대 {MAX_PER_SECTOR}개)')
else:
    print(f'📈 매수 추천 (상위 {args.top}개)')
print('─' * 70)

if QEPM_MODE:
    # QEPM: 섹터 제약 + 변동성 표시
    print(f'{"순위":>4} {"종목코드":<8} {"종목명":<12} {"섹터":<8} {"현재가":>10} {"변동성":>8} {"모멘텀":>8}')
    print('─' * 70)
    top_n = apply_sector_constraint(latest_df, args.top, MAX_PER_SECTOR)
else:
    print(f'{"순위":>4} {"종목코드":<8} {"종목명":<12} {"현재가":>10} {"시총(억)":>10} {"모멘텀":>8} {"낙폭":>8}')
    print('─' * 70)
    top_n = latest_df.nsmallest(args.top, 'rank')

for _, row in top_n.iterrows():
    name = str(row['name'])[:10] if pd.notna(row['name']) else 'N/A'
    price = f"{int(row['closing_price']):,}"
    mcap = f"{int(row['market_cap']/100000000):,}"

    # 핵심 지표
    mom = row.get('mom_60d', row.get('mom_20d', 0)) * 100 if QEPM_MODE else row.get('mom_20d', 0) * 100
    mom = mom if pd.notna(mom) else 0
    dd = row.get('drawdown_from_high', 0) * 100 if pd.notna(row.get('drawdown_from_high')) else 0
    vol = row.get('volatility_20d', 0) * 100 if pd.notna(row.get('volatility_20d')) else 0
    sector = str(row.get('sector', 'N/A'))[:6] if pd.notna(row.get('sector')) else 'N/A'

    if QEPM_MODE:
        print(f'{int(row["rank"]):>4} {row["stock_code"]:<8} {name:<12} {sector:<8} {price:>10} {vol:>7.1f}% {mom:>+7.1f}%')
    else:
        print(f'{int(row["rank"]):>4} {row["stock_code"]:<8} {name:<12} {price:>10} {mcap:>10} {mom:>+7.1f}% {dd:>+7.1f}%')

print('─' * 70)

# QEPM: 변동성 역가중 비중 계산
if QEPM_MODE and len(top_n) > 0:
    print(f'\n💰 변동성 역가중 비중 (Risk Parity)')
    print('─' * 70)

    vol_col = 'volatility_20d'
    if vol_col in top_n.columns:
        top_n_copy = top_n.copy()
        top_n_copy['vol'] = top_n_copy[vol_col].fillna(top_n_copy[vol_col].median()).clip(lower=0.1)
        top_n_copy['inv_vol'] = 1 / top_n_copy['vol']
        top_n_copy['weight'] = top_n_copy['inv_vol'] / top_n_copy['inv_vol'].sum() * 100

        for _, row in top_n_copy.iterrows():
            name = str(row['name'])[:10] if pd.notna(row['name']) else 'N/A'
            weight = row['weight']
            vol = row['vol'] * 100
            print(f'  {row["stock_code"]} {name:<12} 비중: {weight:>5.1f}%  (변동성 {vol:.1f}%)')

        print('─' * 70)
        print(f'  합계: 100.0%')
    print('─' * 70)

# ============================================================================
# Bottom N 출력
# ============================================================================
print(f'\n🚫 매도/회피 추천 (하위 {args.bottom}개)')
print('─' * 70)

bottom_n = latest_df.nlargest(args.bottom, 'rank')
print(f'{"순위":>4} {"종목코드":<8} {"종목명":<12} {"현재가":>10} {"시총(억)":>10}')
print('─' * 70)

for _, row in bottom_n.iterrows():
    name = str(row['name'])[:10] if pd.notna(row['name']) else 'N/A'
    price = f"{int(row['closing_price']):,}"
    mcap = f"{int(row['market_cap']/100000000):,}"
    print(f'{row["rank"]:>4} {row["stock_code"]:<8} {name:<12} {price:>10} {mcap:>10}')

print('─' * 70)

# ============================================================================
# 핵심 지표별 Top 5
# ============================================================================
print(f'\n{"="*70}')
print('📊 핵심 지표별 Top 5')
print('─' * 70)

# 과거의 영광 + 낙폭 (본능 전략)
if 'fallen_angel_score' in latest_df.columns:
    print('\n[추락한 천사 - 과거 영광 + 현재 낙폭]')
    fallen = latest_df.nlargest(5, 'fallen_angel_score')
    for _, row in fallen.iterrows():
        name = str(row['name'])[:10] if pd.notna(row['name']) else 'N/A'
        glory = row.get('past_glory_1y', 0) * 100
        dd = row.get('drawdown_from_high', 0) * 100
        print(f'  {row["stock_code"]} {name}: 영광 {glory:+.0f}%, 낙폭 {dd:+.0f}%')

# 거래량 폭발
if 'volume_surprise' in latest_df.columns:
    print('\n[거래량 폭발 - 평소 대비 급증]')
    vol_surge = latest_df.nlargest(5, 'volume_surprise')
    for _, row in vol_surge.iterrows():
        name = str(row['name'])[:10] if pd.notna(row['name']) else 'N/A'
        vs = row.get('volume_surprise', 0)
        print(f'  {row["stock_code"]} {name}: 거래량 {vs:.1f}배')

# VCP 패턴 (변동성 수축)
if 'vcp_score' in latest_df.columns:
    print('\n[VCP 패턴 - 변동성 수축 후 돌파 대기]')
    vcp = latest_df.nlargest(5, 'vcp_score')
    for _, row in vcp.iterrows():
        name = str(row['name'])[:10] if pd.notna(row['name']) else 'N/A'
        score = row.get('vcp_score', 0)
        print(f'  {row["stock_code"]} {name}: VCP {score:.2f}')

# ============================================================================
# CSV 저장
# ============================================================================
output_cols = ['rank', 'stock_code', 'name', 'closing_price', 'market_cap', 'score']

# 추가 지표들
extra_cols = ['mom_20d', 'drawdown_from_high', 'volume_surprise',
              'past_glory_1y', 'fallen_angel_score', 'vcp_score']
for col in extra_cols:
    if col in latest_df.columns:
        output_cols.append(col)

output_file = f'picks_v2_{latest_date}.csv'
latest_df[output_cols].sort_values('rank').to_csv(output_file, index=False)

print(f'\n{"="*70}')
print(f'전체 순위 저장: {output_file}')
print('\n⚠️  주의: 모델 예측일 뿐, 투자 결정은 본인 책임!')
print('=' * 70)
