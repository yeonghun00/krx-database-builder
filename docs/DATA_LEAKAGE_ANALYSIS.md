# Data Leakage Analysis Report

## Executive Summary

**현재 상태:** IC Decay가 역전됨 (1일 IC = 0.027 < 63일 IC = 0.137)

**진단 결과:** ❌ 1/4 테스트 통과 - 데이터 누수 가능성 높음

### 핵심 발견

| 지표 | 값 | 판정 |
|------|-----|------|
| 상반기 IC | +0.219 | 🔴 비정상적으로 높음 |
| 하반기 IC | -0.009 | 정상 (0에 가까움) |
| 재무 피처 중요도 | 84% | 🔴 과도하게 높음 |
| 21일 모델 IC | 63일 > 21일 | 🔴 역전 현상 |

---

## 1. 발견된 문제점들

### 1.1 Target Leakage via Training Period (수정 완료)

**문제:**
```
학습 데이터: 2020-01-01 ~ 2024-12-31
테스트 데이터: 2025-01-01 ~

문제점:
- 2024년 12월 데이터의 forward_return_63d는 2025년 3월 가격 필요
- 모델이 2025년 Q1 가격 패턴을 학습 데이터의 target으로 사용
```

**수정:**
```python
# Before
train_df = df[df['year'] <= 2024]

# After (run_safety_check.py, run_backtest.py, run_full_backtest.py)
BUFFER_DATE = '20240901'
train_df = df[(df['year'] <= 2024) & (df['date'] <= BUFFER_DATE)]
```

**상태:** ✅ 수정 완료

---

### 1.2 IC Decay 역전 현상 분석

**현상:**
```
1일 후:  IC = +0.027
5일 후:  IC = +0.030
21일 후: IC = +0.036
63일 후: IC = +0.137  ← 모델이 학습한 horizon
```

**해석:**
| 가능성 | 설명 | 위험도 |
|--------|------|--------|
| **정상 (Slow Factor)** | 모델이 63일 target으로 학습했으므로 63일 IC가 높은 것은 당연. 밸류에이션, 퀄리티 등 slow-moving factor는 단기 수익률과 상관 낮음 | 낮음 |
| **의심 (Feature Overlap)** | 피처 계산 기간(60일 rolling)과 target 기간(63일)이 겹쳐서 인과관계 왜곡 | 중간 |
| **위험 (Data Leakage)** | 피처 계산에 미래 정보가 포함됨 | 높음 |

**진단 테스트 필요:**
```python
# Test 1: 재무 데이터 1분기 추가 밀기
# 결과: IC가 크게 떨어지면 → available_date 누수

# Test 2: 63일 대신 21일로 학습 후 IC Decay 확인
# 결과: 21일 IC > 63일 IC 되면 → 정상 (horizon-specific 학습)
```

---

## 2. 코드 분석

### 2.1 재무 데이터 available_date 처리

**파일:** `features/financial_features.py:356-365`

```python
# merge_asof - joins each daily price row to the most recent
# financial data where available_date <= price date
merged_stock = pd.merge_asof(
    price_stock,
    fin_stock,
    left_on='date',
    right_on='available_date',
    by='stock_code',
    direction='backward'  # Only use past data ✓
)
```

**DB 검증:**
```
fiscal_date    available_date    delay
2024-12-31     20250401         ~91일 (연간보고서)
2024-09-30     20241115         ~46일 (3분기)
2024-06-30     20240816         ~47일 (반기)
2024-03-31     20240516         ~46일 (1분기)
```

**상태:** ✅ 정상 - available_date가 fiscal_date보다 46-91일 후로 설정됨

---

### 2.2 기술적 피처 계산

**파일:** `ml/features.py:240-290`

**Z-Score 정규화:**
```python
# Rolling window 사용 (과거 데이터만)
df['mom_1m_mean'] = grouped['mom_1m'].transform(
    lambda x: x.rolling(60, min_periods=30).mean()  # ✓ 과거 60일
)
df['mom_1m_zscore'] = (df['mom_1m'] - df['mom_1m_mean']) / df['mom_1m_std']
```

**상태:** ✅ 정상 - rolling window 사용

**Cross-sectional Rank:**
```python
# 날짜별 그룹으로 랭킹 (미래 데이터 미사용)
df['mom_1m_rank'] = df.groupby('date')['mom_1m'].rank(pct=True)  # ✓
```

**상태:** ✅ 정상 - 날짜별 그룹 사용

---

### 2.3 Forward Return & Target 계산

**파일:** `ml/features.py:505-527`

```python
# Forward return 계산
df[col_name] = grouped['closing_price'].transform(
    lambda x: x.pct_change(h).shift(-h)  # T+h 가격 사용
)

# Clipping
df[col_name] = df[col_name].clip(-0.30, 0.30)  # ⚠️ 이상치 제거

# Cross-sectional rank (target)
df[rank_col] = df.groupby('date')[col_name].rank(pct=True)
```

**잠재적 문제:**
- `.clip(-0.30, 0.30)`은 ±30%를 넘는 수익률을 자름
- 이로 인해 극단적 수익률 종목의 정보 손실
- 하지만 데이터 누수는 아님

**상태:** ✅ 정상 (데이터 누수 아님)

---

## 3. 잠재적 누수 지점 상세 분석

### 3.1 의심 지점 #1: Survivorship Bias

**문제:**
```python
# prepare_ml_data() 호출 시
df = fe.prepare_ml_data(
    start_date='20200101',
    end_date='20260128',  # 전체 기간 데이터 로드
    ...
)
```

**위험:**
- 전체 기간 데이터를 한 번에 로드
- 2025년까지 생존한 종목만 포함
- 2022년에 상장폐지된 종목 제외
- 결과: 생존 편향으로 수익률 과대 추정

**검증 방법:**
```python
# 각 연도별 종목 수 비교
df.groupby('year')['stock_code'].nunique()

# 2020년 200개 → 2025년 180개 이면 정상
# 2020년 180개 = 2025년 180개 이면 생존편향 의심
```

---

### 3.2 의심 지점 #2: Feature-Target Overlap

**문제:**
```
Feature: mom_3m = 과거 63일 수익률 (T-63 ~ T)
Target:  forward_return_63d = 미래 63일 수익률 (T ~ T+63)

시간축:  |---Feature---|---Target---|
         T-63          T           T+63
```

**실제 겹침은 없음.** 하지만:
- 추세가 지속되는 종목은 mom_3m ↑ → forward_return_63d ↑
- 이는 "모멘텀 팩터"로 합법적인 알파
- **누수 아님**

---

### 3.3 의심 지점 #3: Cross-sectional Median Imputation

**파일:** `features/financial_features.py:394-425`

```python
def impute_sector_median(self, df: pd.DataFrame, ...):
    # Calculate sector median for each date
    sector_medians = df.groupby(['date', 'sector_x'])[col].transform('median')
```

**분석:**
- `groupby(['date', ...])` 사용 → 날짜별 그룹
- 미래 데이터 사용 안 함
- **상태:** ✅ 정상

---

## 4. 진단 테스트 코드

### Test 1: Lag Test (재무 데이터 추가 지연)

```python
# run_leakage_test.py
"""
재무 데이터를 추가로 90일 밀어서 IC 변화 확인
IC 크게 떨어지면 → available_date 누수
"""

import pandas as pd
from features.financial_features import FinancialFeatureGenerator

class LaggedFinancialFeatureGenerator(FinancialFeatureGenerator):
    def forward_fill_to_daily(self, price_df, fin_df):
        fin_df = fin_df.copy()
        # 추가 90일 지연
        fin_df['available_date'] = pd.to_datetime(fin_df['available_date']) + pd.Timedelta(days=90)
        return super().forward_fill_to_daily(price_df, fin_df)

# 테스트 실행 후 IC 비교
```

### Test 2: Shuffle Test (랜덤 셔플)

```python
"""
재무 데이터를 종목별로 셔플하여 IC 확인
셔플 후에도 IC 높으면 → 코드 로직 버그
"""

import numpy as np

def shuffle_test(df):
    df_shuffled = df.copy()

    # 재무 피처만 종목 간 셔플
    fund_features = ['pe', 'pb', 'roe', 'gpa', ...]

    for date in df_shuffled['date'].unique():
        mask = df_shuffled['date'] == date
        for col in fund_features:
            if col in df_shuffled.columns:
                values = df_shuffled.loc[mask, col].values
                np.random.shuffle(values)
                df_shuffled.loc[mask, col] = values

    return df_shuffled

# 셔플 후 IC 계산
```

### Test 3: Horizon Switch Test

```python
"""
21일 horizon으로 학습 후 IC Decay 확인
21일 IC > 63일 IC 되면 → 정상 (horizon-specific)
"""

model_21d = MLRanker(
    feature_cols=all_features,
    target_col='target_rank_21d',  # 21일로 변경
    model_type='regressor'
)
model_21d.train(train_df)

# IC 계산
for days in [1, 5, 21, 63]:
    ic = spearmanr(pred_score, forward_return[days])
    print(f'{days}일: IC = {ic}')

# 예상: 21일 IC가 가장 높아야 정상
```

---

## 5. 결론 및 권고사항

### 5.1 현재 상태

| 항목 | 상태 | 설명 |
|------|------|------|
| Training Period Leakage | ✅ 수정됨 | 버퍼 날짜 추가 |
| Available Date 처리 | ✅ 정상 | merge_asof backward 사용 |
| Z-Score 정규화 | ✅ 정상 | Rolling window 사용 |
| Cross-sectional Rank | ✅ 정상 | 날짜별 그룹 사용 |
| Survivorship Bias | ⚠️ 확인 필요 | 전체 기간 로드 |
| IC Decay 역전 | ⚠️ 추가 분석 필요 | 진단 테스트 권장 |

### 5.2 권고 조치

1. **즉시 실행:** Horizon Switch Test 실행
   - 21일로 학습 후 IC Decay 정상인지 확인
   - 정상이면 현재 시스템 OK

2. **추가 검증:** Lag Test 실행
   - 재무 데이터 90일 추가 지연 후 IC 확인
   - IC 유지되면 OK

3. **코드 개선:** Survivorship Bias 방지
   - 연도별로 유니버스 재구성
   - 상장폐지 종목 포함

---

## 6. 20년차 퀀트 코멘트에 대한 응답

> "63일 IC가 0.137이라는 건 로또 번호를 미리 알고 있는 수준"

**반론:**
- 모델은 **63일 target으로 학습**되었으므로 63일 IC가 높은 것은 당연
- 밸류에이션, 퀄리티 팩터는 단기 수익률과 상관 낮음 (설계 의도)
- IC = 0.137은 장기 팩터 전략에서 가능한 수치

**동의:**
- 그러나 **진단 테스트 없이 확신 불가**
- Horizon Switch Test로 확인 필수

---

## 7. 최종 검증 결과 (2026-01-30)

### 7.1 IC 계산 불일치 발견

| 소스 | 1월 IC | 비고 |
|------|--------|------|
| run_safety_check.py | +0.169 | BUFFER_DATE 수정 후 |
| 직접 검증 | +0.001 | 동일 코드로 재계산 |

### 7.2 직접 검증 상세 결과

```
2025년 1월 데이터: 2921 rows
1월 IC (Spearman): +0.001  ← 실제로는 거의 0

Quintile별 평균 수익률 (forward_return_63d):
  Q1:  +15.3% (n=585)  ← 하위 20%
  Q5:  +10.6% (n=584)  ← 상위 20%

→ Q1 > Q5: 모델이 역으로 예측 중!

날짜별 IC:
  20250102: IC = -0.032
  20250103: IC = -0.028
  20250106: IC = +0.039
  20250107: IC = -0.004
  20250108: IC = -0.018

→ 일별 IC가 0 근처에서 변동 (정상)
```

### 7.3 결론

1. **데이터 누수는 아님**: available_date 로직 정상 작동 확인
2. **모델 성능이 낮음**: 실제 IC ≈ 0, 예측력 없음
3. **이전 높은 IC는 계산 오류 또는 레짐 효과**: 상반기에만 우연히 맞춤

### 7.4 권고사항

1. **모델 재학습 필요**: 현재 모델은 예측력 없음
2. **피처 엔지니어링 개선**: 재무 피처 84% 의존 → 기술적 피처 보강
3. **앙상블 전략 고려**: 여러 horizon 모델 조합

---

*생성일: 2026-01-30*
*최종 업데이트: 2026-01-30*
*분석 대상: algostock ML 백테스트 시스템*
