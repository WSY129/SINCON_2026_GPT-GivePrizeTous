# ensemble_v3 전체 실행 결과

KBO 편성표를 제외하고, 요청한 4개 패치와 허용된 도메인 정보를 적용한 결과다. 기존 v1/v2 산출물은 보존했다. 아래 점수는 실제 저장된 OOF에서 계산했다. v3 Public 점수는 아직 없다. 기존 v2 제출은 사용자 확인 기준 Public **0.18265**다.

## 1. 비교 기준과 결과

| model | oof_rmsle | weighted_rmsle | fold34_rmsle | fold4_rmsle | weight |
| --- | --- | --- | --- | --- | --- |
| lightgbm | 0.223453 | 0.223769 | 0.224241 | 0.234329 | 0.095299 |
| catboost | 0.226809 | 0.226800 | 0.226786 | 0.235666 | 0.000000 |
| xgboost | 0.224885 | 0.224405 | 0.223683 | 0.230935 | 0.138385 |
| tabm | 0.224599 | 0.222020 | 0.218095 | 0.224762 | 0.766316 |
| ensemble | 0.222437 | 0.220529 | 0.217637 | 0.224364 | 1.000000 |

모든 모델은 동일한 검증 행으로 비교한다. 전체 OOF는 628,776개 예측 사례이며, 동일 관측이 서로 다른 origin에 중복 등장한다. 주력 선택 지표는 Fold 3+4의 314,388개 사례를 합쳐 계산한 RMSLE다. 전체 가중 점수는 폴드별 로그 MSE에 0.15/0.15/0.35/0.35를 곱한 뒤 제곱근을 취한다.

| metric | v2 | v3 | delta_v3_minus_v2 |
| --- | --- | --- | --- |
| oof_rmsle | 0.222761 | 0.222437 | -0.000324 |
| weighted_rmsle | 0.220618 | 0.220529 | -0.000089 |
| fold34_rmsle | 0.217365 | 0.217637 | 0.000273 |
| fold4_rmsle | 0.224398 | 0.224364 | -0.000034 |

`delta_v3_minus_v2`는 음수일 때 개선이다. v1 Fold 4 RMSLE 0.247808과의 비교는 `cv_metrics.csv`에 있다. 가중치 적합에 사용한 OOF 점수는 독립적인 일반화 성능 추정치가 아니다.

**현재 판단:** v3는 주력 Fold 3+4 지표에서 v2보다 개선되지 않았다. v2 Public 0.18265 제출을 보존하고, v3는 추가 비교 후보로 취급한다. 가중 OOF의 미세한 개선만으로 v3의 Public 우위를 주장하지 않는다. 이미 확인된 v2 Public 점수에서 일정한 값을 빼서 v3 점수를 예측하지 않는다.

| model | v2_fold34_rmsle | v3_fold34_rmsle | delta_v3_minus_v2 |
| --- | --- | --- | --- |
| lightgbm | 0.228119 | 0.224241 | -0.003878 |
| catboost | 0.220714 | 0.226786 | 0.006072 |
| xgboost | 0.232881 | 0.223683 | -0.009197 |
| tabm | 0.219177 | 0.218095 | -0.001082 |

세 모델의 단일 점수가 낮아져도 결합 오차가 반드시 낮아지는 것은 아니다. 이번 비교에서는 v2에서 큰 비중을 차지했던 CatBoost의 최근 구간 점수가 악화됐으며, v3 최적 혼합에서 기여가 사실상 0이 됐다. CPU/MPS에서 학습한 v2와 CUDA에서 학습한 v3는 실행 장치와 PyTorch 버전도 다르다. 따라서 이 차이를 피처 변경만의 효과나 GPU 자체의 효과로 분리해 해석할 수 없다.

날짜 중복을 제거한 Fold 3→4 점검에서 앙상블은 0.225605, 선택된 단일 tabm은 0.224762다. 이 점검에서 단일 모델이 더 낮다면 단일 후보를 함께 유지하고, 혼합 모델의 우월성을 단정하지 않는다.

## 2. 검증과 앙상블 선택

| fold | origin | train_rows | valid_rows | origins | features | train_year364_pct | valid_year364_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2025-08-30 | 800000 | 157194 | 15 | 100 | 0.000000 | 5.450590 |
| 2 | 2025-10-30 | 800000 | 157194 | 19 | 100 | 0.000000 | 52.948586 |
| 3 | 2025-12-30 | 800000 | 157194 | 23 | 100 | 3.136625 | 95.774648 |
| 4 | 2026-02-27 | 800000 | 157194 | 28 | 100 | 29.334625 | 96.049468 |

최종 가중치: `{"lightgbm": 0.0952990745489883, "catboost": 2.290377089375628e-18, "xgboost": 0.13838451741412858, "tabm": 0.7663164080368832}`.
비교용 전체 4-Fold 가중치: `{"lightgbm": 0.25754453005082073, "catboost": 1.495182558493291e-17, "xgboost": 0.19701159612184646, "tabm": 0.5454438738273328}`.

| fit_fold | test_fold | purged_overlapping_target_dates | fit_rows | ensemble_rmsle | training_selected_single | single_rmsle | chronological |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 4 | False | 157194 | 0.225452 | tabm | 0.224762 | True |
| 3 | 4 | True | 75402 | 0.225605 | tabm | 0.224762 | True |
| 4 | 3 | False | 157194 | 0.211072 | tabm | 0.211218 | False |
| 4 | 3 | True | 75402 | 0.211031 | tabm | 0.211218 | False |

Fold 3→4는 시간순 민감도 점검이며, Fold 4→3은 역방향 점검이다. 날짜 중복 제거는 가중치 적합 행에서만 수행했다. base model early stopping과 피처 설계에는 여전히 CV 결과가 사용되므로 완전히 독립된 holdout으로 주장하지 않는다. Fold 3/4는 겨울·봄, 테스트는 여름·가을이며 year364 가용률도 완전히 같지 않다.

잔차 상관 행렬은 `residual_correlation.csv`와 `ensemble_report.json`에 있다. 상관계수 목표를 맞추기 위해 모델을 강제로 넣거나, 모델별 가장 쉬운 fold의 가중치를 조합하지 않는다. SLSQP가 0을 부여한 모델은 최종 혼합에서 기여하지 않는다. 절편·MBE 상수 보정은 없다.

## 3. 최적 반복 수와 최종 학습

| model | fold1_best | fold2_best | fold3_best | fold4_best | final_iterations |
| --- | --- | --- | --- | --- | --- |
| lightgbm | 25 | 34 | 1235 | 2312 | 2107 |
| catboost | 1 | 132 | 1694 | 3491 | 3105 |
| xgboost | 1 | 16 | 630 | 3000 | 2298 |
| tabm | 4 | 2 | 12 | 6 | 9 |

최종 반복 수는 round((0.4 × Fold3 best + 0.6 × Fold4 best) × 1.12), 최소 1이다. 각 모델을 시드 2026/2027로 전체 데이터에서 재학습하고 두 로그 예측을 평균한다. 최종 재학습에는 검증 구간이 없으므로 이 고정 횟수로 실행한다. 부스팅 횟수는 신경망의 epoch와 다르며 트리는 AdamW를 사용하지 않는다. 비선형 모델의 전역 최적점을 찾았다고 보장하지 않는다.

| model | log_rmse_between_seeds | correlation |
| --- | --- | --- |
| lightgbm | 0.026946 | 0.999643 |
| catboost | 0.031240 | 0.999584 |
| xgboost | 0.022062 | 0.999761 |
| tabm | 0.028595 | 0.999622 |


## 4. 보조 진단과 도메인 역

| model | rmsle | mae | rmse | mbe | rmse_mae_ratio | mase_macro | mase_row_weighted | acf_1d_macro | acf_7d_macro | acf_28d_macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 0.234329 | 44.081382 | 114.174630 | -31.361619 | 2.590087 | 0.938429 | 0.938429 | 0.166246 | 0.251765 | 0.165948 |
| catboost | 0.235666 | 46.170170 | 116.244712 | -33.153232 | 2.517745 | 0.970959 | 0.970959 | 0.176245 | 0.264333 | 0.183713 |
| xgboost | 0.230935 | 45.299254 | 114.950759 | -32.785748 | 2.537586 | 0.946133 | 0.946133 | 0.185167 | 0.253156 | 0.158537 |
| tabm | 0.224762 | 41.271452 | 111.019158 | -26.805159 | 2.689975 | 0.892541 | 0.892541 | 0.171478 | 0.220975 | 0.169518 |
| ensemble | 0.224364 | 41.800850 | 111.538003 | -28.164201 | 2.668319 | 0.896922 | 0.896922 | 0.171446 | 0.228664 | 0.168701 |

전체 지표 이름과 값은 `cv_metrics.csv`, 모델별 `diagnostics/summary.json`, `per_entity.csv`, `per_step.csv`, `per_month.csv`, `residual_acf.csv`를 확인한다. MASE는 과거 7일 계절 복사 오차를 분모로 사용한다. 원시 스케일 MBE를 0으로 강제하면 로그 지표가 악화될 수 있으므로 후처리에 자동 적용하지 않았다.

| subset | station_id | v2 | v3 | delta_v3_minus_v2 |
| --- | --- | --- | --- | --- |
| all | 3122 | 0.299717 | 0.293452 | -0.006265 |
| all | 3127 | 0.737389 | 0.724940 | -0.012449 |
| all | 3136 | 0.293184 | 0.267742 | -0.025442 |
| all | 3139 | 0.240986 | 0.258375 | 0.017389 |
| fold34 | 3122 | 0.268327 | 0.261138 | -0.007189 |
| fold34 | 3127 | 0.754641 | 0.750954 | -0.003687 |
| fold34 | 3136 | 0.263685 | 0.224323 | -0.039363 |
| fold34 | 3139 | 0.182390 | 0.178655 | -0.003736 |
| fold4 | 3122 | 0.257727 | 0.253858 | -0.003869 |
| fold4 | 3127 | 0.874731 | 0.863233 | -0.011498 |
| fold4 | 3136 | 0.292712 | 0.235667 | -0.057045 |
| fold4 | 3139 | 0.178069 | 0.181825 | 0.003756 |

문학경기장 3127, 송도달빛축제공원 3139, 부평삼거리 3122, 인천대입구 3136의 동일 구간 비교다. 전체 71역의 오차 기여는 `station_comparison.csv`에 있다.

## 5. 제출 파일 검수

```json
{
  "rows": 157194,
  "columns": [
    "example_id",
    "boardings"
  ],
  "ids_in_test_order": true,
  "unique_ids": true,
  "finite": true,
  "nonnegative": true,
  "mean": 341.80280903859756,
  "min": 3.45583836,
  "max": 3099.00300372,
  "mean_in_requested_315_345_range": true,
  "minimum_below_3": false,
  "quantiles": {
    "0.0": 3.45583836,
    "0.01": 11.978798293299999,
    "0.1": 54.611313530000004,
    "0.5": 247.938791285,
    "0.9": 733.1522826790001,
    "0.99": 1795.0834001461003,
    "1.0": 3099.00300372
  },
  "sha256": "f2eaac6c5f401eace2f16be733a55136157ad5f2564c95bcbaa63ac84e176bf1",
  "note": "Mean/minimum thresholds are diagnostics only. No mean forcing, floor forcing, bias calibration or domain zeroing was applied."
}
```
주 제출 후보: `submission_ensemble_v3.csv`. 단일 모델군의 2개 시드 평균과 전체 4-Fold 가중치 혼합은 `candidates/`에 별도로 저장했다. 평균 315~345와 최소 3 미만은 진단 기준이며 이를 만족시키기 위한 강제 보정은 하지 않았다. 소수 예측을 유지하며 음수 로그는 0으로 제한한다. 정수 반올림·상수 편향 보정·영업시간 강제 0 처리는 없다.

## 6. 구현·재현·제약

전체 4-Fold CV 및 최종 8개 모델 재학습은 종료 코드 0으로 완료됐으며, 소요 시간은 856.35초(약 14.27분)다. 최종 단위 테스트는 로컬 60개, RunPod 60개 모두 통과했다(`local_unit_tests.log`, `runpod_unit_tests.log`). 267개 원격 산출물의 다운로드 해시 검수와 OOF 점수 독립 재계산도 통과했다(`download_integrity.json`, `artifact_verification.json`).

구현 상세는 `IMPLEMENTATION.md`, 외부 정보 사용 결정은 `OFFICIAL_SOURCES.md`, 실행 환경은 `environment.json`, 입력·소스·설정 해시는 `run_fingerprint.json`에 있다. 새로운 출처나 소스를 추가할 때 완료 결과를 같은 실행으로 재사용하지 않는다.

```bash
python -m unittest discover -s tests -v
python -m src.pipeline_v3 run
python -m src.pipeline_v3 infer
```

`run`은 신규 실행 폴더 또는 동일 fingerprint 환경에서 사용하는 명령이다. 현재 완료 모델의 예측 재현에는 `infer`를 사용한다. 실제 원격 소스는 `training_source_snapshot.tar.gz`에도 보존했다. macOS 전송 과정에서 생성된 AppleDouble `._*` 파일 27개는 실행 코드가 아니며 원격 fingerprint에 포함되어 있다. 로컬 검수는 이 메타데이터를 별도 스냅샷에서 확인하고, 실행 소스는 현재 `src/`와 직접 대조했다. 패키징 메타데이터 차이로 기존 `run` 재개가 거부되면 해시를 지우거나 기존 결과를 덮어쓰지 말고 새 프로젝트 루트에서 재현한다. 특징 행렬 캐시는 로컬 다운로드에서 제외했으며 코드로 재생성할 수 있다.

Kaggle/Colab은 `notebooks/SINCON_reproduce_v3.ipynb`에 소스와 설정을 내장했다. 설치된 CUDA PyTorch를 유지하고 나머지 고정 버전을 설치한다. 오프라인 환경에서는 wheel을 미리 준비해야 한다. 실제 검증 장치는 RunPod RTX PRO 4500 Blackwell 32GB 1장이며 CUDA 2장·MPS의 전체 학습을 이번 실행에서 직접 수행한 것은 아니다. 장치 선택 로직과 CPU 경로는 같은 코드로 제공한다.

KBO 날짜는 입력에 없다. 학사·귀가시각은 proxy이며 미래 실측 정보가 아니다. 학습에서 상수였던 신경망 연속 변수는 추론에서도 중립화한다. 신규 축제 피처의 학습 노출이 없으면 해당 피처로 개선이 일어났다고 해석하지 않는다. v2 대비 변경 효과는 결합 실험 결과이며 각 패치의 개별 인과 효과는 ablation 없이는 분리할 수 없다.

저장된 8개 모델을 다시 불러와 수행한 독립 추론 검수:

```json
{
  "passed": true,
  "models_reloaded": 8,
  "rows": 157194,
  "ids_identical": true,
  "max_abs_log_difference": 0.0,
  "log_rms_difference": 0.0,
  "max_abs_count_difference": 0.0,
  "tolerance_log": 2e-05,
  "original_sha256": "f2eaac6c5f401eace2f16be733a55136157ad5f2564c95bcbaa63ac84e176bf1",
  "reloaded_sha256": "f2eaac6c5f401eace2f16be733a55136157ad5f2564c95bcbaa63ac84e176bf1",
  "seconds": 22.577861919999123
}
```

기존 산출물 보존 검수:

```json
{
  "checked_files": 476,
  "unchanged": true,
  "changed_files": []
}
```
