# SINCON ensemble_v4 실행 결과

실행 상태: `{'status': 'complete', 'seconds': 4342.9076046943665, 'within_budget': True, 'start': 1794414621.842275, 'deadline': 1794421821.842275, 'inference_only_patch': 'reload_fix_audit.json'}`. 아래 수치는 저장된 실제 산출물에서만 가져왔다. 미완료 단계는 미측정으로 표기한다.

## 비교 점수

| version | oof_rmsle | weighted_rmsle | fold34_rmsle | fold4_rmsle |
| --- | --- | --- | --- | --- |
| ensemble_v2 | 0.222761 | 0.220618 | 0.217365 | 0.224398 |
| ensemble_v3 | 0.222437 | 0.220529 | 0.217637 | 0.224364 |
| v4_step_a | 미측정 | 미측정 | 0.207532 | 0.209251 |
| v4_step_b | 0.221289 | 0.214121 | 0.202893 | 0.204816 |

가중 RMSLE는 폴드 로그 MSE에 0.15/0.15/0.35/0.35를 적용한 제곱근이다. Step A에는 Fold 3·4만 있으므로 전체 OOF·4폴드 가중 점수를 부여하지 않는다. 기존 v2의 확인된 Public 점수는 **0.18265**다. v4 Public 점수는 측정하지 않았다.

## Fold 4 역별 비교

| version | station_id | rmsle | mbe | log_sse |
| --- | --- | --- | --- | --- |
| ensemble_v2 | 3107 | 0.454894 | -20.599988 | 458.138945 |
| ensemble_v3 | 3107 | 0.447252 | -19.920211 | 442.877025 |
| v4_step_a | 3107 | 0.415857 | -17.463328 | 382.882008 |
| v4_step_b | 3107 | 0.384062 | -14.164867 | 326.572291 |
| ensemble_v2 | 3111 | 0.222405 | -4.333609 | 109.513544 |
| ensemble_v3 | 3111 | 0.228523 | -4.948711 | 115.620723 |
| v4_step_a | 3111 | 0.235924 | -3.816451 | 123.232018 |
| v4_step_b | 3111 | 0.253667 | 0.261256 | 142.463675 |
| ensemble_v2 | 3122 | 0.257727 | 0.148203 | 147.061149 |
| ensemble_v3 | 3122 | 0.253858 | -6.601066 | 142.678928 |
| v4_step_a | 3122 | 0.224342 | -5.439239 | 111.428931 |
| v4_step_b | 3122 | 0.219924 | -4.467665 | 107.083665 |
| ensemble_v2 | 3125 | 0.234216 | -92.344929 | 121.453357 |
| ensemble_v3 | 3125 | 0.218538 | -76.623986 | 105.738260 |
| v4_step_a | 3125 | 0.196371 | -43.735454 | 85.375129 |
| v4_step_b | 3125 | 0.195983 | -45.179065 | 85.038360 |
| ensemble_v2 | 3127 | 0.874731 | -112.597110 | 1694.050527 |
| ensemble_v3 | 3127 | 0.863233 | -109.214094 | 1649.809408 |
| v4_step_a | 3127 | 0.830569 | -96.456240 | 1527.317508 |
| v4_step_b | 3127 | 0.812541 | -92.331806 | 1461.733852 |
| ensemble_v2 | 3136 | 0.292712 | -116.113263 | 189.695756 |
| ensemble_v3 | 3136 | 0.235667 | -85.754695 | 122.962714 |
| v4_step_a | 3136 | 0.207008 | -67.891282 | 94.874982 |
| v4_step_b | 3136 | 0.202577 | -64.899700 | 90.856871 |
| ensemble_v2 | 3137 | 0.231021 | -30.120747 | 118.163210 |
| ensemble_v3 | 3137 | 0.244663 | -33.014137 | 132.530058 |
| v4_step_a | 3137 | 0.221243 | -25.862212 | 108.371828 |
| v4_step_b | 3137 | 0.213693 | -23.842490 | 101.101691 |
| ensemble_v2 | 3201 | 0.220462 | -1.981245 | 107.608582 |
| ensemble_v3 | 3201 | 0.223833 | -0.791566 | 110.924566 |
| v4_step_a | 3201 | 0.231900 | 4.160369 | 119.063239 |
| v4_step_b | 3201 | 0.232016 | 3.736267 | 119.183027 |
| ensemble_v2 | 3208 | 0.293622 | -24.949709 | 190.878016 |
| ensemble_v3 | 3208 | 0.274342 | -21.767443 | 166.633206 |
| v4_step_a | 3208 | 0.234345 | -14.388544 | 121.587379 |
| v4_step_b | 3208 | 0.225005 | -11.383832 | 112.088982 |
| ensemble_v2 | 3209 | 0.269042 | -33.960029 | 160.256727 |
| ensemble_v3 | 3209 | 0.274737 | -35.438429 | 167.113875 |
| v4_step_a | 3209 | 0.254899 | -25.378867 | 143.851117 |
| v4_step_b | 3209 | 0.251066 | -22.606265 | 139.557219 |
| ensemble_v2 | 3210 | 0.227313 | -42.780733 | 114.399776 |
| ensemble_v3 | 3210 | 0.225003 | -38.500539 | 112.086257 |
| v4_step_a | 3210 | 0.219393 | -30.366224 | 106.566832 |
| v4_step_b | 3210 | 0.218127 | -29.148318 | 105.340288 |
| ensemble_v2 | 3217 | 0.232911 | -4.280262 | 120.103701 |
| ensemble_v3 | 3217 | 0.236272 | -5.194451 | 123.594946 |
| v4_step_a | 3217 | 0.246962 | 3.997744 | 135.032367 |
| v4_step_b | 3217 | 0.242798 | 4.577427 | 130.517492 |
| ensemble_v2 | 3226 | 0.514412 | -64.981244 | 585.867524 |
| ensemble_v3 | 3226 | 0.531787 | -65.666712 | 626.114544 |
| v4_step_a | 3226 | 0.484051 | -57.834794 | 518.751159 |
| v4_step_b | 3226 | 0.462045 | -51.501314 | 472.656181 |
| ensemble_v2 | 3227 | 0.262362 | -7.813087 | 152.397944 |
| ensemble_v3 | 3227 | 0.270439 | -8.489799 | 161.925948 |
| v4_step_a | 3227 | 0.263738 | -6.906631 | 154.000363 |
| v4_step_b | 3227 | 0.260022 | -6.391618 | 149.691539 |
| ensemble_v2 | 3754 | 0.445460 | -53.977231 | 439.333907 |
| ensemble_v3 | 3754 | 0.463389 | -55.518058 | 475.411826 |
| v4_step_a | 3754 | 0.401862 | -44.114692 | 357.545399 |
| v4_step_b | 3754 | 0.398166 | -44.579391 | 350.999599 |


## 실제 모델별 학습 반복 및 시간

| fold | model | best_iteration | seconds | matrix_rows |
| --- | --- | --- | --- | --- |
| fold_1 | catboost_cpu | 160 | 476.092973 | 1376505 |
| fold_1 | decomp_gbdt | 163 | 147.903257 | 1376505 |
| fold_1 | lightgbm | 117 | 88.868622 | 1376505 |
| fold_1 | low_demand_specialist | 126 | 80.025238 | 1376505 |
| fold_1 | st_itransformer | 1 | 61.575763 | 1376505 |
| fold_1 | tabm_v4 | 34 | 1299.730697 | 1376505 |
| fold_1 | xgboost | 138 | 67.289554 | 1376505 |
| fold_2 | catboost_cpu | 36 | 377.715183 | 1500000 |
| fold_2 | decomp_gbdt | 56 | 132.582723 | 1500000 |
| fold_2 | lightgbm | 44 | 105.406060 | 1500000 |
| fold_2 | low_demand_specialist | 1294 | 329.283795 | 1500000 |
| fold_2 | st_itransformer | 7 | 47.054282 | 1500000 |
| fold_2 | tabm_v4 | 20 | 343.363639 | 1500000 |
| fold_2 | xgboost | 51 | 25.056653 | 1500000 |
| fold_3 | catboost_cpu | 24 | 364.425992 | 1500000 |
| fold_3 | decomp_gbdt | 2234 | 607.715239 | 1500000 |
| fold_3 | lightgbm | 29 | 85.242704 | 1500000 |
| fold_3 | low_demand_specialist | 441 | 158.687070 | 1500000 |
| fold_3 | st_itransformer | 1 | 34.150910 | 1500000 |
| fold_3 | tabm_v4 | 2 | 334.712378 | 1500000 |
| fold_3 | xgboost | 31 | 66.024097 | 1500000 |
| fold_4 | catboost_cpu | 22 | 251.479745 | 1500000 |
| fold_4 | decomp_gbdt | 65 | 213.309823 | 1500000 |
| fold_4 | lightgbm | 27 | 106.957089 | 1500000 |
| fold_4 | low_demand_specialist | 376 | 112.738512 | 1500000 |
| fold_4 | st_itransformer | 7 | 52.365787 | 1500000 |
| fold_4 | tabm_v4 | 1 | 111.621993 | 1500000 |
| fold_4 | xgboost | 53 | 28.434387 | 1500000 |

`result.json`의 device 필드는 직렬화를 위해 CPU로 이동한 이후 값이다. 실제 학습 장치는 CatBoost/LightGBM CPU, XGBoost/TabM/시공간 신경망 CUDA다. matrix_rows는 공통 학습 행렬 크기이며 전문 모델의 필터링 후 표본 수나 분해·시공간 모델의 역·일 블록 수와는 다르다. OOF는 시드 2026, 최종 추론은 2026·2027 평균이다.

## step_a 모델·앙상블 비교

| model | oof_rmsle | weighted_rmsle | fold34_rmsle | fold4_rmsle |
| --- | --- | --- | --- | --- |
| lightgbm | 0.214513 | 0.214513 | 0.214513 | 0.217596 |
| catboost_cpu | 0.215952 | 0.215952 | 0.215952 | 0.217863 |
| xgboost | 0.213575 | 0.213575 | 0.213575 | 0.215862 |
| tabm_v4 | 0.210663 | 0.210663 | 0.210663 | 0.209690 |


전역 혼합: {'oof_rmsle': 0.2077049846711122, 'weighted_rmsle': 0.2077049846711122, 'fold34_rmsle': 0.2077049846711122, 'fold4_rmsle': 0.2095561776908395}

버킷 혼합: {'oof_rmsle': 0.20753198282123825, 'weighted_rmsle': 0.20753198282123822, 'fold34_rmsle': 0.20753198282123825, 'fold4_rmsle': 0.20925107762235318}

버킷 순서별 가중치(모델 순서: lightgbm, catboost_cpu, xgboost, tabm_v4)

| bucket | lightgbm | catboost_cpu | xgboost | tabm_v4 |
| --- | --- | --- | --- | --- |
| new_stations | 0.032585 | 0.138250 | 0.239060 | 0.590105 |
| high_spike | 0.032585 | 0.138250 | 0.239060 | 0.590105 |
| late_or_low | 0.032585 | 0.138250 | 0.239060 | 0.590105 |
| regular | 0.000000 | 0.306594 | 0.056886 | 0.636520 |


### 날짜 중복 제거 Fold 3 → Fold 4 점검

```json
{
  "fit_rows": 75402,
  "test_rows": 157194,
  "global_rmsle": 0.2098806712927184,
  "raw_bucket_rmsle": 0.21044097579885462,
  "training_selected_single": "tabm_v4",
  "single_rmsle": 0.20969004631854116,
  "buckets": [
    {
      "bucket": "new_stations",
      "rows": 6642,
      "bucket_rmsle": 0.288757435870885,
      "global_rmsle": 0.28167265187027235,
      "training_selected_single": "tabm_v4",
      "single_rmsle": 0.275607364962018,
      "fallback_global": true,
      "fit_weights": [
        0.7642440105840488,
        0.0,
        0.0,
        0.23575598941595133
      ]
    },
    {
      "bucket": "high_spike",
      "rows": 1365,
      "bucket_rmsle": 1.0417605551235258,
      "global_rmsle": 1.0352788455523094,
      "training_selected_single": "tabm_v4",
      "single_rmsle": 1.0417605551235258,
      "fallback_global": true,
      "fit_weights": [
        2.862293735361732e-17,
        4.163336342344337e-17,
        0.0,
        1.0
      ]
    },
    {
      "bucket": "late_or_low",
      "rows": 25296,
      "bucket_rmsle": 0.23907983556412832,
      "global_rmsle": 0.23810359956724536,
      "training_selected_single": "tabm_v4",
      "single_rmsle": 0.23686079635623603,
      "fallback_global": true,
      "fit_weights": [
        0.0,
        0.4826757573550548,
        0.05697923553722201,
        0.46034500710772314
      ]
    },
    {
      "bucket": "regular",
      "rows": 123891,
      "bucket_rmsle": 0.16760555344992012,
      "global_rmsle": 0.1680860391539262,
      "training_selected_single": "tabm_v4",
      "single_rmsle": 0.1682404300246419,
      "fallback_global": false,
      "fit_weights": [
        0.0,
        0.38652229176451514,
        0.0,
        0.613477708235485
      ]
    }
  ]
}
```

LOFO 전역 혼합 RMSLE: 0.209074

Fold 3+4 blend scores fit weights on those labels. Bucket fallback is selected using Fold 4; its selected score is not independent. Purged checks remove repeated dates but still reuse early-stopping and feature-design decisions.

### 가중치 적합 구간 비교

| model | fold34_weight | all_fold_weight |
| --- | --- | --- |
| lightgbm | 0.032585 | 0.032585 |
| catboost_cpu | 0.138250 | 0.138250 |
| xgboost | 0.239060 | 0.239060 |
| tabm_v4 | 0.590105 | 0.590105 |


### 로그 잔차 상관 행렬

| model | lightgbm | catboost_cpu | xgboost | tabm_v4 |
| --- | --- | --- | --- | --- |
| lightgbm | 1.000000 | 0.991186 | 0.994191 | 0.916666 |
| catboost_cpu | 0.991186 | 1.000000 | 0.987338 | 0.910110 |
| xgboost | 0.994191 | 0.987338 | 1.000000 | 0.923246 |
| tabm_v4 | 0.916666 | 0.910110 | 0.923246 | 1.000000 |


## step_b 모델·앙상블 비교

| model | oof_rmsle | weighted_rmsle | fold34_rmsle | fold4_rmsle |
| --- | --- | --- | --- | --- |
| lightgbm | 0.227181 | 0.222201 | 0.214513 | 0.217596 |
| catboost_cpu | 0.226696 | 0.222461 | 0.215952 | 0.217863 |
| xgboost | 0.224823 | 0.220393 | 0.213575 | 0.215862 |
| tabm_v4 | 0.222676 | 0.217950 | 0.210663 | 0.209690 |
| decomp_gbdt | 0.244055 | 0.233870 | 0.217701 | 0.209331 |
| low_demand_specialist | 0.260956 | 0.244457 | 0.217373 | 0.217968 |
| st_itransformer | 0.241324 | 0.230929 | 0.214394 | 0.205693 |


전역 혼합: {'oof_rmsle': 0.22079629990896024, 'weighted_rmsle': 0.21457651045885684, 'fold34_rmsle': 0.20489311365079727, 'fold4_rmsle': 0.20568432356707167}

버킷 혼합: {'oof_rmsle': 0.22128948991514336, 'weighted_rmsle': 0.21412065900611144, 'fold34_rmsle': 0.20289303589002797, 'fold4_rmsle': 0.20481649659373047}

버킷 순서별 가중치(모델 순서: lightgbm, catboost_cpu, xgboost, tabm_v4, decomp_gbdt, low_demand_specialist, st_itransformer)

| bucket | lightgbm | catboost_cpu | xgboost | tabm_v4 | decomp_gbdt | low_demand_specialist | st_itransformer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| new_stations | 0.000000 | 0.000000 | 0.000000 | 0.044048 | 0.000000 | 0.091205 | 0.864748 |
| high_spike | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.510721 | 0.000000 | 0.489279 |
| late_or_low | 0.000000 | 0.325703 | 0.049519 | 0.475976 | 0.000000 | 0.000000 | 0.148802 |
| regular | 0.000000 | 0.000000 | 0.000000 | 0.384889 | 0.116162 | 0.344377 | 0.154572 |


### 날짜 중복 제거 Fold 3 → Fold 4 점검

```json
{
  "fit_rows": 75402,
  "test_rows": 157194,
  "global_rmsle": 0.20940810681803573,
  "raw_bucket_rmsle": 0.2087291018698489,
  "training_selected_single": "tabm_v4",
  "single_rmsle": 0.20969004631854116,
  "buckets": [
    {
      "bucket": "new_stations",
      "rows": 6642,
      "bucket_rmsle": 0.2766325820619466,
      "global_rmsle": 0.27893019765245425,
      "training_selected_single": "tabm_v4",
      "single_rmsle": 0.275607364962018,
      "fallback_global": false,
      "fit_weights": [
        1.2034982927768e-16,
        6.613633252161577e-18,
        1.929879867024198e-17,
        5.567378155713065e-17,
        2.168404344971009e-18,
        0.6593594171234798,
        0.3406405828765201
      ]
    },
    {
      "bucket": "high_spike",
      "rows": 1365,
      "bucket_rmsle": 1.021762841978502,
      "global_rmsle": 1.026570517543125,
      "training_selected_single": "tabm_v4",
      "single_rmsle": 1.0417605551235258,
      "fallback_global": false,
      "fit_weights": [
        8.258571235729428e-19,
        9.87809823087965e-18,
        7.014279836212861e-18,
        0.5735586466758381,
        0.022264919161662875,
        2.872204020844607e-17,
        0.40417643416249904
      ]
    },
    {
      "bucket": "late_or_low",
      "rows": 25296,
      "bucket_rmsle": 0.23992866617649372,
      "global_rmsle": 0.2407940468207495,
      "training_selected_single": "tabm_v4",
      "single_rmsle": 0.23686079635623603,
      "fallback_global": false,
      "fit_weights": [
        0.0,
        0.4060919448952801,
        0.10508888663849245,
        0.3861650049170845,
        0.0,
        0.10265416354914295,
        1.5686916790066698e-18
      ]
    },
    {
      "bucket": "regular",
      "rows": 123891,
      "bucket_rmsle": 0.1670937649414598,
      "global_rmsle": 0.16738784092480935,
      "training_selected_single": "tabm_v4",
      "single_rmsle": 0.1682404300246419,
      "fallback_global": false,
      "fit_weights": [
        0.0,
        1.4075905774844104e-18,
        3.519227078344913e-18,
        0.5588949218474004,
        0.0,
        0.44110507815259964,
        7.432577832515807e-18
      ]
    }
  ]
}
```

LOFO 전역 혼합 RMSLE: 0.219989

Fold 3+4 blend scores fit weights on those labels. Bucket fallback is selected using Fold 4; its selected score is not independent. Purged checks remove repeated dates but still reuse early-stopping and feature-design decisions.

### 가중치 적합 구간 비교

| model | fold34_weight | all_fold_weight |
| --- | --- | --- |
| lightgbm | 0.000000 | 0.000000 |
| catboost_cpu | 0.000000 | 0.191862 |
| xgboost | 0.000000 | 0.207527 |
| tabm_v4 | 0.293968 | 0.467737 |
| decomp_gbdt | 0.114358 | 0.000000 |
| low_demand_specialist | 0.313054 | 0.033815 |
| st_itransformer | 0.278620 | 0.099059 |


### 로그 잔차 상관 행렬

| model | lightgbm | catboost_cpu | xgboost | tabm_v4 | decomp_gbdt | low_demand_specialist | st_itransformer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 1.000000 | 0.974188 | 0.988953 | 0.875098 | 0.947373 | 0.874342 | 0.811814 |
| catboost_cpu | 0.974188 | 1.000000 | 0.970089 | 0.874640 | 0.942980 | 0.870442 | 0.826092 |
| xgboost | 0.988953 | 0.970089 | 1.000000 | 0.889146 | 0.941754 | 0.860820 | 0.824789 |
| tabm_v4 | 0.875098 | 0.874640 | 0.889146 | 1.000000 | 0.857599 | 0.739559 | 0.864801 |
| decomp_gbdt | 0.947373 | 0.942980 | 0.941754 | 0.857599 | 1.000000 | 0.843150 | 0.788970 |
| low_demand_specialist | 0.874342 | 0.870442 | 0.860820 | 0.739559 | 0.843150 | 1.000000 | 0.681307 |
| st_itransformer | 0.811814 | 0.826092 | 0.824789 | 0.864801 | 0.788970 | 0.681307 | 1.000000 |


## 샘플링 실제 가용률

| fold | rows | candidates | rich_candidate_rows | rich_fraction | query_rich_fraction | daily_days | quota_met |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final | 2000000 | 5005396 | 2046772 | 0.850001 | 1.000000 | 278047 | True |
| fold_1 | 1376505 | 1605978 | 0 | 0.000000 | 0.105806 | 89213 | 미측정 |
| fold_2 | 1500000 | 2285874 | 0 | 0.000000 | 0.566816 | 126985 | 미측정 |
| fold_3 | 1500000 | 2961936 | 135432 | 0.090288 | 0.929577 | 164544 | False |
| fold_4 | 1500000 | 7178204 | 1505072 | 0.800002 | 0.938166 | 398728 | True |

전년 커널 가용 후보가 부족한 Fold 3에서는 목표 80%를 채우기 위해 동일 라벨을 복제하거나 미래 관측을 쓰지 않는다. 실제 가용률을 보고한다.

## 실행 해석 및 제한

- 첫 시도에서 Fourier 고정효과 전체를 0으로 수축한 문제가 확인되어 절편을 규제하지 않도록 수정했다. 최초 시도는 `attempt_a_initial/`에 보관했으며 기존 v1/v2/v3와 무관하다.
- RunPod 논리 CPU는 48개지만 cgroup 실효 할당은 약 10.2코어다. 최초 시도·수정 재실행을 합친 7,200초 마감으로 관리한다.
- 앙상블 가중치를 적합한 OOF의 점수는 독립 검증 점수가 아니다. Fold 4로 버킷 fallback을 선택하므로 선택 이후 점수를 독립적 일반화 증거로 사용하지 않는다.
- 단일 날짜 중복을 제거한 점검도 early stopping 및 피처 개발에 사용한 CV의 재사용을 완전히 없애지는 못한다.
- 시공간 모델은 직접 구현한 Graph Mixing + Temporal Transformer이며 공식 iTransformer 구현이라고 주장하지 않는다.
- 일간 총량 모델은 모든 완전한 역·일 후보를 사용한다. 시간대 점유율은 최대 60,000개 완전 일 블록, 시공간 모델은 최대 120,000개 완전 일 블록으로 학습한다.
- 점유율 타겟의 +0.5 평활화를 복원하려면 `(total+9)*share-0.5`가 필요하다. 지시서의 `total*share-0.5`는 합계에서 9명을 누락하므로 수학적으로 수정했다.
- KBO 및 미허가 민간 일정은 사용하지 않는다. 기존 허용 달력의 as_of 필터를 유지한다.
- 최소 예측값을 3명 아래로 강제로 내리거나 평균을 임의 범위에 맞추는 후처리는 적용하지 않는다.
- RMSLE 0.13 달성 및 Public 개선을 보장하지 않는다.

## ablation_step_a.json

```json
{
  "models": {
    "lightgbm": {
      "oof_rmsle": 0.2145134157081739,
      "weighted_rmsle": 0.2145134157081739,
      "fold34_rmsle": 0.2145134157081739,
      "fold4_rmsle": 0.21759605748377986
    },
    "catboost_cpu": {
      "oof_rmsle": 0.21595196269132408,
      "weighted_rmsle": 0.21595196269132405,
      "fold34_rmsle": 0.21595196269132408,
      "fold4_rmsle": 0.21786250942255714
    },
    "xgboost": {
      "oof_rmsle": 0.2135754123070717,
      "weighted_rmsle": 0.2135754123070717,
      "fold34_rmsle": 0.2135754123070717,
      "fold4_rmsle": 0.2158621648186577
    },
    "tabm_v4": {
      "oof_rmsle": 0.21066251154040924,
      "weighted_rmsle": 0.21066251154040921,
      "fold34_rmsle": 0.21066251154040924,
      "fold4_rmsle": 0.20969004631854116
    }
  },
  "global": {
    "oof_rmsle": 0.2077049846711122,
    "weighted_rmsle": 0.2077049846711122,
    "fold34_rmsle": 0.2077049846711122,
    "fold4_rmsle": 0.2095561776908395
  },
  "ensemble": {
    "oof_rmsle": null,
    "weighted_rmsle": null,
    "fold34_rmsle": 0.20753198282123825,
    "fold4_rmsle": 0.20925107762235318
  },
  "global_weights": {
    "lightgbm": 0.03258539493726605,
    "catboost_cpu": 0.13824998262248467,
    "xgboost": 0.2390600789797966,
    "tabm_v4": 0.5901045434604527
  },
  "all_fold_weights": {
    "lightgbm": 0.03258539493726605,
    "catboost_cpu": 0.13824998262248467,
    "xgboost": 0.2390600789797966,
    "tabm_v4": 0.5901045434604527
  },
  "bucket_weights": {
    "new_stations": [
      0.03258539493726605,
      0.13824998262248467,
      0.2390600789797966,
      0.5901045434604527
    ],
    "high_spike": [
      0.03258539493726605,
      0.13824998262248467,
      0.2390600789797966,
      0.5901045434604527
    ],
    "late_or_low": [
      0.03258539493726605,
      0.13824998262248467,
      0.2390600789797966,
      0.5901045434604527
    ],
    "regular": [
      0.0,
      0.3065942085192414,
      0.05688552171983756,
      0.6365202697609211
    ]
  },
  "model_order": [
    "lightgbm",
    "catboost_cpu",
    "xgboost",
    "tabm_v4"
  ],
  "purged_chronological_check": {
    "fit_rows": 75402,
    "test_rows": 157194,
    "global_rmsle": 0.2098806712927184,
    "raw_bucket_rmsle": 0.21044097579885462,
    "training_selected_single": "tabm_v4",
    "single_rmsle": 0.20969004631854116,
    "buckets": [
      {
        "bucket": "new_stations",
        "rows": 6642,
        "bucket_rmsle": 0.288757435870885,
        "global_rmsle": 0.28167265187027235,
        "training_selected_single": "tabm_v4",
        "single_rmsle": 0.275607364962018,
        "fallback_global": true,
        "fit_weights": [
          0.7642440105840488,
          0.0,
          0.0,
          0.23575598941595133
        ]
      },
      {
        "bucket": "high_spike",
        "rows": 1365,
        "bucket_rmsle": 1.0417605551235258,
        "global_rmsle": 1.0352788455523094,
        "training_selected_single": "tabm_v4",
        "single_rmsle": 1.0417605551235258,
        "fallback_global": true,
        "fit_weights": [
          2.862293735361732e-17,
          4.163336342344337e-17,
          0.0,
          1.0
        ]
      },
      {
        "bucket": "late_or_low",
        "rows": 25296,
        "bucket_rmsle": 0.23907983556412832,
        "global_rmsle": 0.23810359956724536,
        "training_selected_single": "tabm_v4",
        "single_rmsle": 0.23686079635623603,
        "fallback_global": true,
        "fit_weights": [
          0.0,
          0.4826757573550548,
          0.05697923553722201,
          0.46034500710772314
        ]
      },
      {
        "bucket": "regular",
        "rows": 123891,
        "bucket_rmsle": 0.16760555344992012,
        "global_rmsle": 0.1680860391539262,
        "training_selected_single": "tabm_v4",
        "single_rmsle": 0.1682404300246419,
        "fallback_global": false,
        "fit_weights": [
          0.0,
          0.38652229176451514,
          0.0,
          0.613477708235485
        ]
      }
    ]
  },
  "lofo_global_rmsle": 0.20907393817812275,
  "limitations": "Fold 3+4 blend scores fit weights on those labels. Bucket fallback is selected using Fold 4; its selected score is not independent. Purged checks remove repeated dates but still reuse early-stopping and feature-design decisions.",
  "gate": {
    "v3_fold4": 0.224363706,
    "fold4_improved": true,
    "stations": [
      {
        "model": "ensemble",
        "station_id": 3107,
        "rmsle": 0.4158566696615307,
        "log_sse": 382.88200812018255,
        "mbe": -17.46332809192753
      },
      {
        "model": "ensemble",
        "station_id": 3127,
        "rmsle": 0.8305692663214544,
        "log_sse": 1527.3175078332788,
        "mbe": -96.4562403169458
      },
      {
        "model": "ensemble",
        "station_id": 3226,
        "rmsle": 0.4840505654791415,
        "log_sse": 518.7511591686582,
        "mbe": -57.834793629455525
      },
      {
        "model": "ensemble",
        "station_id": 3754,
        "rmsle": 0.4018618589519923,
        "log_sse": 357.54539944829696,
        "mbe": -44.11469185740885
      }
    ]
  }
}
```

## final_iteration_plan.json

```json
{
  "lightgbm": {
    "fold3": 29,
    "fold4": 27,
    "final": 31
  },
  "catboost_cpu": {
    "fold3": 24,
    "fold4": 22,
    "final": 25
  },
  "xgboost": {
    "fold3": 31,
    "fold4": 53,
    "final": 59
  },
  "tabm_v4": {
    "fold3": 2,
    "fold4": 1,
    "final": 1
  },
  "decomp_gbdt": {
    "fold3": 2234,
    "fold4": 65,
    "final": 680
  },
  "low_demand_specialist": {
    "fold3": 441,
    "fold4": 376,
    "final": 439
  },
  "st_itransformer": {
    "fold3": 1,
    "fold4": 7,
    "final": 8
  }
}
```

## submission_validation.json

```json
{
  "rows": 157194,
  "ids_in_test_order": true,
  "finite_nonnegative": true,
  "mean": 334.35364677299833,
  "min": 2.7272169519900546,
  "max": 2798.773544923522,
  "sha256": "9b315a3ef59f3a1e847fb01468867686e6116e12913b46f9a534919d959ae998"
}
```

## reload_validation.json

```json
{
  "models": {
    "lightgbm_2026": {
      "allclose": true,
      "max_abs": 0.0
    },
    "lightgbm_2027": {
      "allclose": true,
      "max_abs": 0.0
    },
    "catboost_cpu_2026": {
      "allclose": true,
      "max_abs": 0.0
    },
    "catboost_cpu_2027": {
      "allclose": true,
      "max_abs": 0.0
    },
    "xgboost_2026": {
      "allclose": true,
      "max_abs": 0.0
    },
    "xgboost_2027": {
      "allclose": true,
      "max_abs": 0.0
    },
    "tabm_v4_2026": {
      "allclose": true,
      "max_abs": 4.76837158203125e-07
    },
    "tabm_v4_2027": {
      "allclose": true,
      "max_abs": 4.76837158203125e-07
    },
    "decomp_gbdt_2026": {
      "allclose": true,
      "max_abs": 0.0
    },
    "decomp_gbdt_2027": {
      "allclose": true,
      "max_abs": 0.0
    },
    "low_demand_specialist_2026": {
      "allclose": true,
      "max_abs": 0.0
    },
    "low_demand_specialist_2027": {
      "allclose": true,
      "max_abs": 0.0
    },
    "st_itransformer_2026": {
      "allclose": true,
      "max_abs": 4.76837158203125e-07
    },
    "st_itransformer_2027": {
      "allclose": true,
      "max_abs": 4.76837158203125e-07
    }
  },
  "submission_allclose": true,
  "original_sha256": "9b315a3ef59f3a1e847fb01468867686e6116e12913b46f9a534919d959ae998",
  "reload_sha256": "86ad9e119bd17257b2c754205a62ca6586a01c945aa819dbe4068a3382000917",
  "byte_identical": false
}
```

## preservation_check.json

```json
{
  "checked": 304,
  "changed": []
}
```

## local_preservation_check.json

```json
{
  "checked": 997,
  "changed": [
    "outputs/ensemble_v3/.DS_Store"
  ],
  "checked_unix": 1794416757.551192,
  "metadata_only_changes": [
    "outputs/ensemble_v3/.DS_Store"
  ],
  "substantive_changes": []
}
```

## correction_audit.json

```json
{
  "original_start": 1794414621.842275,
  "correction_time": 1794415338.4126785,
  "reason": "Fourier Ridge penalized entire demand level toward zero. Add unpenalized global intercept. All old-source caches/models archived and regenerated. Original 7200s deadline retained."
}
```

## anchor_only_comparison.json

```json
{
  "3": {
    "anchor_rmsle": 0.21946910253288593,
    "rolling_rmsle": 0.23312902675669875,
    "stations": {
      "3110": {
        "v4_sse": 335.66636454864556,
        "rolling_sse": 94.42903895446264
      },
      "3111": {
        "v4_sse": 478.03300261853394,
        "rolling_sse": 458.13660813181247
      },
      "3127": {
        "v4_sse": 853.550772645583,
        "rolling_sse": 1022.5674300833773
      },
      "3226": {
        "v4_sse": 370.89192188489625,
        "rolling_sse": 598.8562067890756
      },
      "3107": {
        "v4_sse": 681.2177873849992,
        "rolling_sse": 641.4046869503013
      },
      "3754": {
        "v4_sse": 347.4459400098849,
        "rolling_sse": 607.7968878036185
      }
    }
  },
  "4": {
    "anchor_rmsle": 0.2216333886680539,
    "rolling_rmsle": 0.2670231919924092,
    "stations": {
      "3110": {
        "v4_sse": 146.99069313355824,
        "rolling_sse": 150.20237763017803
      },
      "3111": {
        "v4_sse": 459.818711026991,
        "rolling_sse": 146.52606282855496
      },
      "3127": {
        "v4_sse": 1501.3869515888364,
        "rolling_sse": 2177.4059490076406
      },
      "3226": {
        "v4_sse": 456.4127252046343,
        "rolling_sse": 971.7761155340719
      },
      "3107": {
        "v4_sse": 457.7809494294059,
        "rolling_sse": 627.0910176405221
      },
      "3754": {
        "v4_sse": 341.1455649036361,
        "rolling_sse": 617.3654054976056
      }
    }
  }
}
```

## reload_fix_audit.json

```json
{
  "scope": "inference-only backend restoration; no retraining, no feature or weight change",
  "training_fingerprint_preserved": true,
  "before_sha256": "f37bc2ad11b50a6e47029ffddf42e76df6d2ae0a53013657f3d7f6decafd8c50",
  "after_sha256": "b8dedab630a11ae8689a9b0c9d94222f7f310adf81abcba5d600204990ff0d30",
  "cpu_xgb_max_log_difference": {
    "2026": 0.1948826089501381,
    "2027": 0.16235048323869705
  },
  "cuda_xgb_max_log_difference": {
    "2026": 0.0,
    "2027": 0.0
  },
  "portability_limit": "CPU inference runs but is not numerically identical to CUDA for installed XGBoost 3.4.1. Exact submission reproduction uses CUDA."
}
```

## runtime_environment.json

```json
{
  "python": "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]",
  "platform": "Linux-6.8.0-147-generic-x86_64-with-glibc2.39",
  "packages": {
    "numpy": "2.3.5",
    "pandas": "2.2.3",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.1",
    "lightgbm": "4.7.0",
    "catboost": "1.2.10",
    "xgboost": "3.4.1",
    "torch": "2.8.0+cu128",
    "joblib": "1.6.0",
    "polars": "2.1.0"
  },
  "cuda": "12.8",
  "gpu": "NVIDIA RTX PRO 4500 Blackwell",
  "cpu_quota": "1020000 100000"
}
```

## 테스트 월별 분포

| month | mean | min | max |
| --- | --- | --- | --- |
| 2026-07 | 332.924553 | 3.311565 | 2717.023169 |
| 2026-08 | 321.676243 | 2.727217 | 2687.338537 |
| 2026-09 | 337.263876 | 3.411292 | 2794.940821 |
| 2026-10 | 345.643793 | 3.256615 | 2798.773545 |

## Gemini 전달용 추가 파일

- `ensemble_report.json`: Step B 전체 보고서와 최종 제출·실행 검수 결과. 원본 `step_b_report.json`의 출처 해시 포함.
- `fold4_domain_segments.csv`: 7개 모델, 기준선, 전역·버킷 앙상블의 동일 Fold 4 구간별 MAE/RMSE/RMSLE/MBE/로그 SSE와 전체 SSE 기여율. 71개 역, 월·시간·horizon, 주요 4개 역의 월·시간 세분화 포함. 세그먼트는 서로 겹치므로 SSE 기여율을 전체 행에 걸쳐 합산하면 안 된다. 실제 승차량 기준 저수요 구간은 사후 진단에만 쓰며 추론 게이트가 아니다.
- `cv_training_budget.csv`: 4폴드×7모델의 최적 반복 수·실행 반복 수·상한·조기 종료·실제 worker 시간. 복합 모델 반복 수는 구성 모델의 최댓값이며 `history_lengths`에 구성별 기록 길이를 남겼다. worker 시간은 병렬 실행 때문에 합산해 전체 경과 시간으로 해석하면 안 된다.
- 학습·재로딩까지 약 72분 23초. 전달 묶음 430개 파일의 SHA-256 검증 완료. 기존 보호 대상의 실질 파일 변경 없음(기존 Finder `.DS_Store` 메타데이터 변경만 관측).
- 기존 테스트 70개 통과에 더해 추론 장치 복원 회귀 테스트 1개를 로컬과 RunPod에서 추가 통과했다.
