#!/usr/bin/env python3
"""
KOSPI200 유사 벤치마크 지수 생성 및 DB 저장

방식: 분기 시작일 기준 시총 상위 200개 고정
      → 일별 시총가중 수익률 계산 → 누적해서 지수화

Usage:
    python3 build_benchmark.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

print('=' * 70)
print('📊 KOSPI200 유사 벤치마크 생성')
print(f'   {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('=' * 70)

DB_PATH = 'krx_stock_data.db'
TOP_N = 200  # 시총 상위 N개

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print('\n[1/5] 데이터 로드 중...')

conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    date,
    stock_code,
    opening_price,
    closing_price,
    market_cap
FROM daily_prices
WHERE market_cap > 0
  AND closing_price > 0
  AND opening_price > 0
ORDER BY date, stock_code
"""

df = pd.read_sql_query(query, conn)
print(f'  로드 완료: {len(df):,} rows')

# ============================================================================
# 2. 일별 수익률 계산
# ============================================================================
print('\n[2/5] 일별 수익률 계산 중...')

df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_values(['stock_code', 'date'])

# 종목별 일간 수익률
df['daily_return'] = df.groupby('stock_code')['closing_price'].pct_change()
df['daily_return'] = df['daily_return'].clip(-0.3, 0.3)  # ±30% 상한제

print(f'  일별 수익률 계산 완료')

# ============================================================================
# 3. 분기별 구성종목 결정
# ============================================================================
print('\n[3/5] 분기별 구성종목 결정 중...')

df['year'] = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter
df['year_quarter'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)

# 각 분기 첫 거래일 찾기
quarter_first_dates = df.groupby('year_quarter')['date'].min().to_dict()

# 각 분기 시작일의 시총 상위 200개 선정
quarter_compositions = {}

for yq, start_date in quarter_first_dates.items():
    day_data = df[df['date'] == start_date]
    top_stocks = set(day_data.nlargest(TOP_N, 'market_cap')['stock_code'].tolist())
    quarter_compositions[yq] = top_stocks

print(f'  분기 수: {len(quarter_compositions)}')

# ============================================================================
# 4. 일별 벤치마크 수익률 계산 (시총가중)
# ============================================================================
print('\n[4/5] 일별 벤치마크 수익률 계산 중...')

benchmark_daily = []

for date in sorted(df['date'].unique()):
    day_data = df[df['date'] == date].copy()

    if len(day_data) == 0:
        continue

    # 해당 분기의 구성종목
    yq = day_data['year_quarter'].iloc[0]
    if yq not in quarter_compositions:
        continue

    composition = quarter_compositions[yq]

    # 구성종목만 필터 + 수익률 있는 것만
    comp_data = day_data[
        (day_data['stock_code'].isin(composition)) &
        (day_data['daily_return'].notna())
    ]

    if len(comp_data) < 50:  # 최소 50개
        continue

    # 시총 가중 일간 수익률
    total_mcap = comp_data['market_cap'].sum()
    weighted_return = (comp_data['daily_return'] * comp_data['market_cap']).sum() / total_mcap

    benchmark_daily.append({
        'date': date,
        'year_quarter': yq,
        'daily_return': weighted_return,
        'num_stocks': len(comp_data),
        'total_mcap': total_mcap
    })

bench_df = pd.DataFrame(benchmark_daily)
bench_df = bench_df.sort_values('date').reset_index(drop=True)

print(f'  계산 완료: {len(bench_df):,} 거래일')

# ============================================================================
# 5. 지수화 및 기간 수익률 계산
# ============================================================================
print('\n[5/5] 지수화 및 기간 수익률 계산 중...')

# 누적 지수 (시작 = 1000)
bench_df['index_value'] = (1 + bench_df['daily_return']).cumprod() * 1000

# 기간별 Forward Return 계산
for horizon in [1, 5, 21, 63, 126]:
    col_name = f'return_{horizon}d'
    bench_df[col_name] = bench_df['index_value'].shift(-horizon) / bench_df['index_value'] - 1

# 날짜 포맷 변환
bench_df['date_str'] = bench_df['date'].dt.strftime('%Y%m%d')
bench_df['year'] = bench_df['date'].dt.year

print(f'  지수화 완료')

# ============================================================================
# DB 저장
# ============================================================================
print('\n[저장] DB에 저장 중...')

# 저장용 컬럼 선택
save_df = bench_df[['date_str', 'year_quarter', 'index_value', 'daily_return',
                     'num_stocks', 'return_1d', 'return_5d', 'return_21d',
                     'return_63d', 'return_126d']].copy()
save_df.columns = ['date', 'year_quarter', 'index_value', 'daily_return',
                   'num_stocks', 'return_1d', 'return_5d', 'return_21d',
                   'return_63d', 'return_126d']

# 테이블 생성
conn.execute("DROP TABLE IF EXISTS benchmark_kospi200")
save_df.to_sql('benchmark_kospi200', conn, if_exists='replace', index=False)
conn.commit()

print(f'  저장 완료: benchmark_kospi200 테이블 ({len(save_df):,} rows)')

# ============================================================================
# 검증
# ============================================================================
print('\n' + '=' * 70)
print('📈 벤치마크 검증')
print('=' * 70)

print('\n[연도별 KOSPI200 수익률]')
print('-' * 60)
print(f'{"연도":<6} {"시작지수":>10} {"종료지수":>10} {"연수익률":>10} {"63일평균":>10}')
print('-' * 60)

for year in sorted(bench_df['year'].unique()):
    year_data = bench_df[bench_df['year'] == year]
    if len(year_data) < 10:
        continue

    start_idx = year_data['index_value'].iloc[0]
    end_idx = year_data['index_value'].iloc[-1]
    annual_ret = (end_idx / start_idx - 1) * 100
    avg_63d = year_data['return_63d'].mean() * 100

    print(f'{year:<6} {start_idx:>10.1f} {end_idx:>10.1f} {annual_ret:>+9.1f}% {avg_63d:>+9.1f}%')

print('-' * 60)

# 전체 CAGR
total_years = (bench_df['date'].max() - bench_df['date'].min()).days / 365
start_val = bench_df['index_value'].iloc[0]
end_val = bench_df['index_value'].iloc[-1]
cagr = ((end_val / start_val) ** (1/total_years) - 1) * 100

print(f'\n전체 기간: {bench_df["date"].min().strftime("%Y-%m-%d")} ~ {bench_df["date"].max().strftime("%Y-%m-%d")}')
print(f'시작 지수: {start_val:.1f} → 종료 지수: {end_val:.1f}')
print(f'CAGR (연평균 복리수익률): {cagr:+.1f}%')

conn.close()

print('\n' + '=' * 70)
print('✅ 벤치마크 생성 완료!')
print('=' * 70)
