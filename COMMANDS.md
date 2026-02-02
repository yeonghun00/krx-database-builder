# AlgoStock 명령어 가이드

## 🚀 Quick Start

```bash
# 오늘의 추천 종목 (V2 모델)
python3 get_picks.py

# 백테스트 실행
python3 run_full_backtest.py
```

---

## 📊 백테스트

### V2 모델 (권장 - 모멘텀 중심)

```bash
# 전체 백테스트 (1/3/6개월 한번에)
python3 run_full_backtest.py

# 21일(1개월) 단일 백테스트
python3 run_backtest.py
```

### V1 모델 (구버전 - 재무 중심)

```bash
# 전체 백테스트
python3 run_full_backtest.py

# Horizon별 백테스트
python3 run_backtest.py --horizon 21   # 1개월
python3 run_backtest.py --horizon 63   # 3개월
python3 run_backtest.py --horizon 126  # 6개월
```

---

## 🛡️ 안전성 점검

```bash
# V2 안전성 점검 (모든 Horizon)
python3 run_safety_check.py

# V1 안전성 점검
python3 run_safety_check.py

# 데이터 누수 진단
python3 run_leakage_tests.py
```

---

## 🎯 종목 추천

### V2 모델 (권장)

```bash
# 오늘의 추천 종목 (상위 20개)
python3 get_picks.py

# 상위 50개
python3 get_picks.py --top 50

# 하위 종목도 보기
python3 get_picks.py --top 20 --bottom 10
```

### V1 모델 (구버전)

```bash
python3 get_picks.py
python3 get_picks.py --top 50
```

---

## 📁 출력 파일

| 파일명 | 설명 |
|--------|------|
| `backtest_v2_full_results.csv` | V2 전체 백테스트 상세 결과 |
| `backtest_v2_summary.csv` | V2 백테스트 요약 |
| `picks_YYYYMMDD.csv` | 해당 날짜 전체 종목 순위 |

---

## 🔧 모델 비교

| 항목 | V1 (구버전) | V2 (신버전) |
|------|-------------|-------------|
| 핵심 피처 | 재무 84% | 재무 50%, 모멘텀 50% |
| Target Horizon | 63일 | 21일 |
| 평균 IC | ~0 | +0.07 |
| Spread | 음수 | +2.2% |

---

## 💡 권장 워크플로우

```bash
# 1. 백테스트로 모델 검증
python3 run_full_backtest.py

# 2. 안전성 점검
python3 run_safety_check.py

# 3. 오늘의 추천 종목 확인
python3 get_picks.py

# 4. CSV로 상세 분석
cat picks_*.csv
```

---

*Last Updated: 2026-01-30*
