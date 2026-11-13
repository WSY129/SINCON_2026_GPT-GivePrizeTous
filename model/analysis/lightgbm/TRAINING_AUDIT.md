# LightGBM Large / Small 학습 설정 점검

2026-11-11, `src/models.py`와 실제 저장 estimator를 확인했다. 모델을 재학습하거나 변경하지 않았다.

## 결론

Large는 220 boosting rounds, Small은 150 rounds를 실제 실행했다. 둘 다 early stopping이 없다.
학습 과정의 검증 점수 기록도 없으므로 220회가 충분한지, 최적 반복 수인지 확인하지 않았다.
현재 모델은 고정된 초기 설정으로 학습·검증한 베이스라인이다. 수렴·하이퍼파라미터 최적화를 완료한 모델로 볼 수 없다.

## 차이

| 항목 | Small | Large |
|---|---:|---:|
| n_estimators | 150 | 220 |
| num_leaves | 31 | 63 |
| 실제 트리 수 | 150 | 220 |
| 트리당 실제 리프 수 | 모두 31 | 모두 63 |
| 전체 리프 수 | 4,650 | 13,860 |
| 개발 1 RMSLE | 0.632489 | 0.657958 |
| 개발 2 RMSLE | 0.227001 | 0.224128 |
| 기존 마지막 검증 RMSLE | 0.262701 | 0.253973 |

Large/Small은 프로젝트 내부 명칭이다. LightGBM 공식 별도 아키텍처나 사전학습 모델을 뜻하지 않는다.
둘 다 같은 GBDT이며 설정 차이는 트리 수와 트리당 최대 리프 수 두 개뿐이다.
Large의 총 리프 수는 약 2.98배지만 이것을 정확한 파라미터 수나 모델 성능 배율로 해석하지 않는다.
두 설정을 동시에 바꿨으므로 성능 차이에서 트리 수 효과와 리프 수 효과를 분리할 수 없다.

## 공통 설정과 과적합 억제

| 설정 | 값 | 역할 |
|---|---|---|
| boosting_type | gbdt | 리프 기준 분할의 gradient boosting |
| objective / target | regression / log1p(boardings) | RMSLE에 맞춘 로그 공간 L2 학습 |
| learning_rate | 0.055 | 트리별 기여를 축소 |
| min_child_samples | 80 | 관측 수가 작은 리프 억제 |
| reg_lambda | 3 | 리프 출력값 L2 규제 |
| colsample_bytree | 0.9 | 트리별 특징 일부 샘플링 |
| subsample / subsample_freq | 0.9 / 1 | 매 라운드 학습 행 일부 샘플링 |
| max_depth | -1 | 별도 깊이 상한 없음; 리프 수 제약은 적용 |
| reg_alpha | 0 | 추가 L1 규제 없음 |
| min_split_gain | 0 | 추가 분할 이득 하한 기본값 |
| min_child_weight | 0.001 | 최소 Hessian 합 기본값 |
| linear_tree | false | 리프 내 선형 회귀 사용 안 함 |
| random_state / n_jobs | 2026 / 4 | 고정 시드, CPU 스레드 수 |

이 규제들은 과적합을 억제하는 장치이며, 현재 데이터에서 최적으로 조정됐거나 과적합을 제거했다는 증거는 아니다.
고정 시드와 CPU 스레드 설정 자체는 과적합 방지 기법이 아니다.

입력은 60개 특징이며 `station_id, canonical_line, hour, dow`를 LightGBM 범주형 특징으로 지정한다.
스케일링 없이 트리에 입력하고 결측값은 유지한다. 최종 학습은 2026-06-30까지 관측으로 만든
여러 과거 예측 기준일의 학습 사례 220,000행을 샘플링한다. 원본 688,176행을 그대로 모두 입력하는 방식이 아니다.
예측 기준일별 학습 타겟 구간과 특징 관측 시점을 제한한다. 이는 누수 방지와 평가 타당성 조치이며 트리 규제와 구별한다.

## Epoch와 early stopping

신경망의 데이터 전체 반복(epoch) 대신 boosting iteration 또는 round를 사용한다.
이번 단일 타겟 GBDT 회귀는 한 round에 트리 하나를 추가하므로 Large 220회 = 220개 트리다.
`best_iteration_=0`, `best_score_={}`, `evals_result_={}`가 두 모델 모두에서 확인됐다.
0은 학습을 0회 했다는 의미가 아니라 검증 기반 최적 반복 수가 선택되지 않았다는 의미다.
`fit`에 `eval_set`, `eval_metric`, early stopping callback이 없다. 완료 후 CV 점수만 계산한다.
따라서 현재 값 220은 조기 종료로 발견한 값이 아니라 사전에 고정한 값이다.

## 충분한 학습·전역 최적점 판단

- 설정된 학습을 정상 완료했는가: **예**, 실제 220개 트리 확인.
- 220회까지 학습이 부족했거나 과도했는가: **미확인**, 반복별 train/validation 곡선 없음.
- 220이 검증 최적 반복 수인가: **미확인**, 반복 수 탐색·early stopping 미실시.
- Large가 항상 Small보다 좋은가: **아니오**, 개발 1에서는 더 나쁘다. 그 차이만으로 과적합을 확정할 수 없다.
- 전체 트리 구조와 하이퍼파라미터의 전역 최적점을 찾았는가: **보장할 수 없음**.

LightGBM은 매 단계에서 이득이 큰 리프 분할을 고르는 탐욕적 구조 탐색과 순차 부스팅을 수행한다.
로그 타겟의 제곱오차가 예측값에 대해 볼록하더라도, 가능한 모든 트리 구조·특징 분할·설정의 전역 최적 탐색을 수행하는 것은 아니다.
early stopping을 적용해도 주어진 학습 경로에서 검증 성능이 좋은 반복 수를 고르는 것이며 전역 최적점을 증명하지 않는다.
앞서 앙상블 가중치에 대해 확인한 볼록 이차 최적화의 최적성은 고정된 모델 예측의 혼합 비율에만 해당한다.

다음 학습에서는 실제 테스트의 기존 역 조건을 반영한 개발 검증을 먼저 고정하고,
넉넉한 반복 상한·validation metric·early stopping·학습 곡선 기록을 추가해야 한다.
예를 들어 초기 실험 상한 2,000회와 patience 100을 후보로 쓸 수 있지만 이 수치 역시 아직 실험으로 정한 최적값은 아니다.
마지막 비교 구간으로 반복 수까지 선택한 후 같은 구간을 독립 테스트로 보고하면 안 된다.
최종 전체 데이터 학습은 개발 구간에서 정한 반복 수를 사용하고, 비공개 테스트 정답 없이 수행한다.

## 근거

- 실제 설정: [src/models.py](../../src/models.py)
- 저장 모델 점검: [saved_model_settings.json](saved_model_settings.json)
- 재현: `.venv/bin/python analysis/lightgbm/inspect_settings.py`
- [LightGBM LGBMRegressor 공식 API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
- [early_stopping 공식 API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html)
- [leaf-wise 트리 성장 설명](https://lightgbm.readthedocs.io/en/stable/Features.html)
