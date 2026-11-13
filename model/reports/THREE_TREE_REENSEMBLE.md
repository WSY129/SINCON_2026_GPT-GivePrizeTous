# LightGBM 중심 3모델 재앙상블

> 이 문서는 이전 실험 기록입니다. 현재 모델과 점수는 [LightGBM 재학습 보고서](LIGHTGBM_TUNING.md)를 참고하세요.

작성일: 2026-11-11. 현재 기본 설정은 `configs/default.json`, 현재 산출물은 `outputs/three_tree`다.

## 결과

기존 앙상블 RMSLE **0.2938525365 → 0.2584334086**, 상대 오차 **12.05% 감소**.
같은 2026-02-28~2026-06-30의 123일·157,194행에서 측정했다. 예측 기준일은 2026-02-27이다.
LightGBM Large 단독 **0.2539725321**보다 앙상블의 RMSLE가 높다. 세 모델 결합 자체가 단독 모델보다 우수하다고 결론내리지 않는다.
Kaggle에 제출하지 않았으며 이 수치는 리더보드 점수가 아니다.

| 모델 | 개발 1 | 개발 2 | 재검증 3 | 가중치 |
|---|---:|---:|---:|---:|
| lightgbm_large | 0.657957785 | 0.224127678 | 0.253972532 | 50.000000000% |
| xgboost | 0.758377827 | 0.239893624 | 0.264857792 | 38.911594198% |
| catboost | 0.447682437 | 0.220653990 | 0.279055773 | 11.088405802% |
| ensemble | 0.429915072 | 0.223838663 | 0.258433409 | 100.000000000% |


## 변경 내용과 가중치 선정

- 활성 학습·앙상블·저장 모델은 LightGBM Large, XGBoost, CatBoost 세 개다.
- `seasonal7`, `weekday_mean`, LightGBM Small, MLP를 활성 모델 목록에서 제거했다. 계절성·과거 요일 통계는 입력 특징으로 계속 사용한다.
- LightGBM Large를 메인으로 고정하기 위해 가중치 최소 50%, 모든 가중치 비음수, 합 1의 제약을 사용한다. 다른 두 모델에는 양수 하한을 강제하지 않았으며 최적화 결과 자연스럽게 양의 비중을 얻었다.
- 개발 fold 1·2 전체 OOF의 log-MSE를 SLSQP로 최소화했다. 로그 타겟과 예측의 잔차 Gram 행렬을 사용한 볼록 이차 최적화이며, 가중치는 마지막 fold 예측 전에 저장했다.
- 결합 산식: `expm1(sum(weight[m] * log1p(pred[m])))`. 상수 편향 보정은 0, 잔차 상관성에 의한 자동 제거는 비활성화했다.
- 특징 60개, 모델 하이퍼파라미터, 학습 표본·시간 분할은 기존과 같다. 세 모델을 각 fold 및 최종 전체 학습 데이터에서 다시 학습했다.

## 검증 범위와 해석

| 구간 | 예측 기준일 | 검증 날짜 | 용도 |
|---|---|---|---|
| 개발 1 | 2025-06-26 | 2025-06-27~2025-10-27 | 가중치 최적화 |
| 개발 2 | 2025-10-27 | 2025-10-28~2026-02-27 | 가중치 최적화 |
| 재검증 3 | 2026-02-27 | 2026-02-28~2026-06-30 | 고정 가중치 비교 |

가중치 계산에는 fold 3의 타겟을 사용하지 않았다. 다만 모델 구성 변경과 편향 보정 제거는 앞서 같은 fold 3을 분석한 뒤 결정했다.
따라서 이번 결과는 **기존 검증 구간의 재평가**이며, 새로운 독립 holdout 성능으로 해석하지 않는다.
개발 1에는 신규 역의 초기 관측이 포함되어 실제 테스트와 개체 구성이 다르다. 이번에는 기존 결과와 비교하기 위해 개발 행·기간을 바꾸지 않았다.
운영진 공식 베이스라인은 2026-03-01부터 122일을 평가하므로 그 점수 0.30770과 완전히 동일한 구간 비교는 아니다.

## 실행 및 검증

- 전체 학습·진단·최종 예측: 66.1초, 프로세스 peak RSS 2.31 GiB. 환경 초기 설치와 검증 재추론 시간 제외.
- 새 제약 최적화·기존 누수·달력·지표·제출 테스트 32개 통과.
- 기존 저장 OOF와 세 모델의 재학습 예측이 일치함을 확인했다. 특징이나 모델 변경 효과를 섞지 않고 앙상블 변경 효과를 비교했다.
- 저장 OOF에서 공식 RMSLE 및 가중 결합 산식을 독립 계산해 요약 결과와 일치함을 확인했다.
- 저장 bundle에는 요청한 세 모델만 존재하며, 재로딩 후 생성한 CSV의 SHA-256이 일치한다.
- 제출 행 157,194개, ID와 순서, NaN·무한대·음수 검수를 통과했다.

## 산출물

상세 원인 분석: [앙상블이 LightGBM 단독보다 나쁜 이유](../analysis/three_tree/WHY_ENSEMBLE_LOSES.md).

- [현재 앙상블 제출 파일](../outputs/three_tree/submission.csv)
- [LightGBM Large 단독 제출 파일](../outputs/three_tree/submission_primary.csv)
- [가중치와 선정 설정](../outputs/three_tree/ensemble_plan.json)
- [독립 재계산 점수](../outputs/three_tree/verified_scores.csv)
- [재학습·재추론 검증](../outputs/three_tree/verification.json)
- [현재 재현 노트북](../notebooks/SINCON_reproduce.ipynb)

`submission_best_single.csv`는 개발 1·2를 합친 점수가 가장 좋은 단일 모델(CatBoost)의 파일이다. LightGBM Large 단독은 명시적으로 `submission_primary.csv`를 사용한다.
기존 `outputs/baseline`은 이전 실험 기록으로 보존했다. 이전 7모델 재현 설정은 `configs/legacy_seven.json`이다.

```bash
cd /Users/hongtaeyang/Desktop/SINCON
.venv/bin/python -m src.pipeline train --output outputs/three_tree
.venv/bin/python -m src.pipeline infer --model outputs/three_tree/model_bundle.joblib --output outputs/three_tree_reloaded
.venv/bin/python scripts/build_three_tree_report.py
```
