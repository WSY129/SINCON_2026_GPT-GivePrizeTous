# 앙상블 실패 원인 정량 진단

이 문서는 원본 OOF를 다시 계산한 사후 진단이다. 대안 가중치는 development fold 1/2 또는 명시된 부분집합에만 적합했다. audit 정답은 대안 적합에 넣지 않았지만, audit 결과를 본 뒤 대안을 고안·비교했으므로 대안 점수를 새로운 독립 검증으로 해석하지 않는다. 원본 모델·설정·제출 CSV는 변경하지 않았다.

## 주된 원인: 시간에 따라 부호가 바뀐 편향에 고정 보정을 적용

- 원래 가중치를 그대로 두고 로그 보정만 제거하면 audit RMSLE가 **0.293852536 → 0.262811895**로 개선된다.
- 기존 로그 보정값은 **-0.056172857**다. `(예측+1)`에 **0.945376**를 곱하는 것과 같다.
- audit의 보정 전 평균 로그 오차는 **-0.125717580**로 이미 과소예측이다. 같은 음수 보정을 추가하여 **-0.181890436**가 됐다.
- 로그 보정이 추가한 audit log-MSE는 **0.017279221**, 원래 audit log-MSE의 **20.01%**다. 보정 없는 상태보다 log-MSE가 **25.02%** 증가했다.
- 감사 구간에서 사후 관찰한 최선의 단일 모델 LightGBM large와 원래 앙상블의 log-MSE 차이 중 **79.09%**를 이 보정 하나가 설명한다. 이는 최선 모델을 감사 구간에서 선택해도 된다는 의미가 아니다.

| population | mean_log_error_before | mean_log_error_after | before_log_mse | after_log_mse | bias_mse_delta_total | clipping_rows |
| --- | --- | --- | --- | --- | --- | --- |
| development_1 | 0.117770 | 0.061597 | 0.165011 | 0.154936 | -0.010076 | 0 |
| development_2 | -0.005403 | -0.061576 | 0.052069 | 0.055831 | 0.003762 | 0 |
| development_pooled | 0.056173 | 0.000000 | 0.108530 | 0.105375 | -0.003155 | 0 |
| audit_3 | -0.125718 | -0.181890 | 0.069070 | 0.086349 | 0.017279 | 0 |

fold 1은 양의 로그 편향이 컸고 fold 2는 거의 0이어서, 합쳐 적합한 음수 상수는 fold 1을 개선하고 fold 2부터 악화시킨다. audit에서는 평균 로그 편향이 더욱 음수가 되어 악화가 커졌다. 평균값 하나를 안정적인 모집단 상수로 옮긴 것이 핵심 문제다.

## 로그 MSE 분해와 앙상블 자체의 이점

`MSE(log error) = Var(log error) + Mean(log error)^2`를 모집단 분산(ddof=0)으로 계산했다. audit의 기존 MSE는 분산 **0.053265182**와 제곱 편향 **0.033084131**의 합이다. 제곱 편향 비중은 **38.31%**다.

모든 구간에서 보정 후 0 절삭 행은 0개다. 따라서 보정의 MSE 영향은 정확히 `2*b*Mean(error_before) + b^2`이며 분산은 바뀌지 않는다. 음수 절삭이나 CSV 오염이 이번 악화 원인이 아니다.

양의 가중치 로그 앙상블에는 `weighted component log-MSE - ambiguity = uncalibrated blend log-MSE`가 성립한다. 여기서 ambiguity는 각 모델 로그 예측이 앙상블 로그 예측과 다른 정도의 가중 제곱 평균이다. 이 식은 RMSLE 자체에 직접 적용하지 않는다.

| population | weighted_component_log_mse | ambiguity_log_mse_gain | before_log_mse | bias_mse_delta_total |
| --- | --- | --- | --- | --- |
| development_1 | 0.253881 | 0.088869 | 0.165011 | -0.010076 |
| development_2 | 0.065691 | 0.013623 | 0.052069 | 0.003762 |
| development_pooled | 0.159770 | 0.051240 | 0.108530 | -0.003155 |
| audit_3 | 0.078723 | 0.009653 | 0.069070 | 0.017279 |

즉, 예측을 섞는 과정 자체는 audit에서도 분산/오차 감소 효과가 있고, 고정 보정이 그 이점 일부를 되돌린다. 정확한 동일성 검사의 최대 수치 오차는 `log_mse_bias_ambiguity_decomposition.csv`에 있다.

## 개발 구간만 적합한 대안의 사후 비교

| candidate | audit_3 | development_1 | development_2 | development_pooled |
| --- | --- | --- | --- | --- |
| original_frozen | 0.293853 | 0.393619 | 0.236286 | 0.324615 |
| frozen_weights_no_bias | 0.262812 | 0.406216 | 0.228185 | 0.329439 |
| frozen_development_best_single | 0.279056 | 0.447682 | 0.220654 | 0.352903 |
| single_lightgbm_large | 0.253973 | 0.657958 | 0.224128 | 0.491465 |
| equal_log_trees_4 | 0.262160 | 0.443040 | 0.220565 | 0.349934 |
| equal_raw_trees_4 | 0.261415 | 0.549329 | 0.220325 | 0.418487 |
| equal_log_original_positive_weight_5 | 0.260542 | 0.414821 | 0.233229 | 0.336490 |
| equal_log_all_7 | 0.262398 | 0.421605 | 0.237539 | 0.342166 |
| refit_dev2_only | 0.246523 | 0.515342 | 0.213655 | 0.394454 |
| refit_dev2_only_no_bias | 0.260720 | 0.510205 | 0.216067 | 0.391764 |
| refit_pooled_no_correlation_filter | 0.292430 | 0.393754 | 0.236082 | 0.324623 |
| refit_pooled_no_correlation_filter_no_bias | 0.261691 | 0.406237 | 0.228080 | 0.329416 |
| replace_small_with_large_no_bias | 0.261829 | 0.406270 | 0.227963 | 0.329395 |
| refit_pooled_existing_stations_only | 0.280185 | 0.491529 | 0.228410 | 0.383236 |
| refit_pooled_existing_stations_only_no_bias | 0.254833 | 0.494749 | 0.217367 | 0.382094 |

`frozen_development_best_single`은 원래 개발 OOF에서 고른 CatBoost다. `single_lightgbm_large`는 감사 결과의 참고 비교이며 실제 선택에 사용하지 않았다. `refit_dev2_only`는 최근 개발 fold 2만으로 가중치와 편향을 다시 적합했다. `no_bias`는 같은 가중치에서 상수만 제거했다. 단순 평균은 가중치를 학습하지 않았고 `raw`/`log`의 평균 공간을 이름에 표시했다.

## 잔차 상관 0.95 필터의 영향

원래 pooled 개발 점수가 더 나쁜 LightGBM large가 LightGBM small과의 로그 잔차 상관이 0.95를 넘어 제외됐다. 상관은 중복성이고, 미래 성능 우열을 보장하지 않는다.

| population | space | correlation |
| --- | --- | --- |
| development_1 | raw | 0.981575 |
| development_1 | log1p | 0.994484 |
| development_2 | raw | 0.972057 |
| development_2 | log1p | 0.972498 |
| development_pooled | raw | 0.979848 |
| development_pooled | log1p | 0.991866 |
| audit_3 | raw | 0.992517 |
| audit_3 | log1p | 0.980608 |

상관 필터를 끈 pooled 개발 재적합 및 small 가중치를 large로 그대로 옮긴 ablation 결과를 위 표에 따로 제시했다. 필터 제거가 pooled 개발 목적함수에서 large를 실제로 선택했는지는 `diagnostic_plan_weights.csv`로 확인한다. 필터가 무엇을 바꿨는지와 부호가 바뀐 편향 보정 문제를 구별해야 한다.

## 첫 개발 fold의 신규역과 구간 이동

첫 origin 이후 관측되는 역은 [3107, 3108, 3109]다. 아래는 fold 1 신규역의 행 비중과 전체 log-MSE에 대한 기여다. 이 구간은 테스트의 모든 역에 과거 이력이 존재하는 조건과 다르다.

| candidate | rows | row_fraction | rmsle | contribution_to_population_log_mse | share_of_population_squared_log_error |
| --- | --- | --- | --- | --- | --- |
| original_frozen | 6588 | 0.041924 | 1.365676 | 0.078192 | 0.504674 |
| frozen_weights_no_bias | 6588 | 0.041924 | 1.362282 | 0.077804 | 0.471506 |
| single_seasonal7 | 6588 | 0.041924 | 1.608862 | 0.108519 | 0.506077 |
| single_weekday_mean | 6588 | 0.041924 | 1.608862 | 0.108519 | 0.525376 |
| single_lightgbm_small | 6588 | 0.041924 | 2.703128 | 0.306337 | 0.765764 |
| single_lightgbm_large | 6588 | 0.041924 | 2.853409 | 0.341346 | 0.788495 |
| single_catboost | 6588 | 0.041924 | 1.587762 | 0.105691 | 0.527348 |
| single_xgboost | 6588 | 0.041924 | 3.360273 | 0.473387 | 0.823085 |
| single_mlp | 6588 | 0.041924 | 1.597873 | 0.107041 | 0.273612 |

후보의 성능 순위와 편향은 개발 fold 사이에서도 크게 달라진다. 이는 단순한 개발 표본 내 과적합만으로 단정할 수 없고, 신규역·계절·학습 이력 길이가 함께 바뀌는 검증 분포 이동의 증거다. `refit_pooled_existing_stations_only`는 신규역을 제외한 개발 자료에서만 재적합한 진단이다. 신규역 처리 정책과 새 검증 구간을 먼저 정한 뒤 수정해야 한다.

## 재현 파일

- 실행: `python analysis/ensemble/analyze_ensemble.py --project /Users/hongtaeyang/Desktop/SINCON`
- `candidate_scores.csv`: 모든 고정·진단 후보의 개발/감사 RMSLE, 로그 평균오차·분산·원단위 편향.
- `log_mse_bias_ambiguity_decomposition.csv`: 보정 영향과 ambiguity 항등식.
- `residual_correlation_pairs.csv`, `residual_correlations_*.csv`: 원단위/로그 잔차 상관.
- `station_cohort_contributions.csv`, `monthly_transport.csv`: 신규역과 월별 이동.
- `diagnostic_plans_NOT_DEPLOYED.json`, `diagnostic_plan_weights.csv`: 적합 모집단과 대안 가중치.
- `input_sha256.json`: 원본 코드·OOF·설정·메타데이터 지문.

새 모델이나 제출 파일을 생성하지 않았다. 이 진단에서 좋아 보인 대안은 다른 손대지 않은 시간 구간에서 확인하기 전까지 새 최종 모델로 부르지 않는다.
