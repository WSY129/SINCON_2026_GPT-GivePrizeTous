# ensemble_v3 구현 및 실행 방법

작성일: 2026-11-12. 현재 버전은 v2를 보존하면서 사용자 요청의 4개 패치와 사용 가능한 도메인 피처를 추가한 비교 실험이다. 추가 Gemini 대화의 새 6단계 구조 변경(전년 커널 anchor, 일총량/시간비율 모델, 8,000회 학습 등)은 이 버전에 섞지 않았다.

## 1. 보존 및 버전 경계

- 기존 `submission_lgbm_0.247808.csv`, `outputs/lightgbm_tuned/model_bundle.joblib`, `configs/lightgbm_tuned.json`, v2 설정과 결과를 덮어쓰지 않는다.
- 기존 소스 파일은 수정하지 않고 `features_v3.py`, `models_v3.py`, `tabm_v3.py`, `nn_preprocessor_v3.py`, `ensemble_v3.py`, `pipeline_v3.py`, `domain_events.py`, `domain_schedule_v3.json`을 추가했다.
- 설정은 `configs/ensemble_v3.json`, 신규 결과는 `outputs/ensemble_v3/` 아래에 둔다.
- `preserved_v2_sha256.json`은 기존 476개 파일의 SHA-256이다. `preservation_check.json`에서 비교 결과를 확인한다.
- v2 실행기는 전체 src 파일 목록을 fingerprint에 포함한다. 새 v3 파일이 추가된 디렉터리에서 과거 v2 실행을 resume하면 fingerprint가 달라질 수 있다. 기존 v2 재현에 필요한 소스·설정·노트북은 `preserved_v2_source.tar.gz`에 별도로 보존했다. v2 모델·예측 파일 자체는 그대로 유지한다.

## 2. 데이터와 인과성

학습 원본: 2025-01-01~2026-06-30, 688,176행. 테스트: 2026-07-01~10-31, 157,194행(123일 × 71역 × 18시간). 승차를 예측하고 하차는 과거 피처 및 학습 보조 타겟에만 사용한다. 실제 sample_submission은 오래된 117,576행 형식이므로 완전한 test ID와 순서를 사용한다.

관측이 없는 신설역 개통 전 날짜를 0으로 채우지 않는다. 날씨 실측, 실제 행사 결과, 이후 발표된 경기 변경 일정은 입력하지 않는다. `holiday_calendar.py`의 `as_of=origin` 필터를 유지한다. 제헌절 2026-07-17은 기존 달력에 이미 반영되어 있다.

`run_fingerprint.json`은 입력 CSV, 설정, 전체 소스 Python/JSON의 해시를 저장한다. 동일 경로의 다른 입력·소스로 중간 결과를 조용히 재사용하지 않는다.

## 3. 고정 CV와 부분 horizon 학습

| Fold | origin | 검증 기간 | 행 수 |
|---|---|---|---:|
| 1 | 2025-08-30 | 2025-08-31~12-31 | 157,194 |
| 2 | 2025-10-30 | 2025-10-31~2026-03-02 | 157,194 |
| 3 | 2025-12-30 | 2025-12-31~2026-05-02 | 157,194 |
| 4 | 2026-02-27 | 2026-02-28~06-30 | 157,194 |

cutoff c에 대해 origin은 2025-02-11부터 c−14까지 14일 간격이며, 마지막 c−14도 포함한다. 라벨은 o < d <= min(o+123,c)인 부분 구간을 허용한다. horizon 1~123을 균등하게 층화하고 origin 가중치 0.5/1.5/2.5를 비복원 표본 추출 확률에 적용한다. 모델 손실에 같은 가중치를 다시 곱하지 않는다. Fold당 최대 800,000행, 최종 1,000,000행이다. 최종 origin은 2026-06-16까지 포함한다.

모든 피처 집계는 해당 origin 이하 관측만 읽는다. 미래 라벨은 목표값 계산에만 쓰인다. 검증 정답은 제거하지 않으며 Fold당 157,194행을 단언한다. 폴드 간 검증 날짜가 중첩된다는 점은 OOF 해석에 반영한다.

## 4. 정제 로그 anchor와 추가 피처

기존 v2 anchor 정의는 유지한다: 정제 로그 평균 0.6×56일 + 0.4×28일. 결측이면 112일 → 노선/시간/유효요일 → 전체 시간/유효요일로 대체한다. 승차·하차 anchor를 별도로 만든다.

과거 로그 집계에서 법정·대체·임시·선거 공휴일과 근로자의 날, 신설 3역의 개통 첫 3일을 제외한다. origin이 2025-07-28 이후이면 계양·귤현의 기저 집계는 2025-07-01 이후로 제한한다. 일반 공휴일은 일요일 eff_dow에 대응하고 근로자의 날은 별도 플래그를 둔다. 원시 타겟을 일괄 이상치 제거하거나 로그 스파이크를 윈저라이징하지 않는다.

v3 추가/변경 사항:

1. 과거 통계와 학습 후보에서 사용자 제공 9개 급감 관측 키를 제외한다. 신설역 3×3×18=162행을 포함한 후보는 171행이며 실제 검증 정답은 보존한다. 외부 사고 원인은 검증되지 않았으므로 단정하지 않는다.
2. `yoy_smooth15_log_delta`: d−364±7일의 정제 로그 승차 평균(station, rest)에서 origin의 전년 직전 56일 평균을 뺀다. station 기준 창에 14일 미만이면 line/hour/rest로 대체한다. 두 쪽 정보가 없으면 0과 missing/fallback 플래그를 함께 둔다. 모든 참조 날짜는 origin 이하다.
3. `commuter_asym_am/pm`: 정제 평일 오전 7~8시, 오후 17~19시의 로그 승차−하차 차이를 역 단위로 제공한다.
4. 환승역·성묘·방학/학기/하교 시간대를 역과 결합한다. 실제 확정 학교별 일정이나 실제 행사 종료 시간을 뜻하지 않는 proxy임을 명시한다.
5. 외부 일정은 `domain_schedule_v3.json`에서 공개일을 검사한다. KBO와 맥주축제는 제외하며 인천시 공공누리 자료의 펜타포트 날짜만 제공한다. 자세한 근거는 `OFFICIAL_SOURCES.md`다.

기존 월별 yoy 계절 변화량, year364 정상일 대체, 28/56/112일 평균·중앙값·표준편차, 공휴일 탄력성, 승하차 비대칭, 시간 점유율, 인접역·환승역 anchor/growth는 유지한다. 2024년 데이터가 없으므로 초기 origin의 전년 기준 창 결측은 남는다. 15일 피처가 이를 자동 해결한다고 주장하지 않는다.

## 5. 모델과 전처리

모든 모델은 r_board=log1p(boardings)−anchor_board_log를 학습한다. 추론 로그는 max(0,anchor+r_hat), 인원수는 expm1로 복원한다. 신경망 보조 목표는 하차 로그 잔차다.

| 모델 | 주요 설정 | 변경 |
|---|---|---|
| LightGBM | leaves159, min_child25, lr0.03, subsample0.8, colsample0.8, L1 0.5/L2 5, 최대3000, ES150 | 교차 범주 2개 제거, cat_smooth50, cat_l2 20, extra_trees |
| CatBoost | depth8, lr0.04, L2 5, random_strength1, 최대3500, ES150 | station_hour/station_eff_dow 유지 |
| XGBoost | hist, categorical, depth6, min_child_weight80, lr0.03, subsample0.8, colsample0.75, L1 0.5/L2 5, 최대3000, ES150 | 교차 범주 2개 제거, 규제 강화 |
| TabM 방식 신경망 | K8, hidden256, 3블록, LayerNorm/SiLU/dropout0.15, 수치 주기 임베딩, 범주 임베딩 | year364 로그 결측을 행별 anchor로 우선 대체 |

트리는 스케일링 없이 float32 파생변수와 NaN을 사용한다. 신경망은 폴드 학습셋에서 중앙값/IQR을 추정하고 [-10,10]으로 제한한다. 원래 NaN 마스크를 보존하며 범주는 0을 미지 토큰으로 비워둔다. anchor 대체는 로그 레벨 변수에만 적용하며 delta/결측 플래그에는 적용하지 않는다. 학습에서 상수인 연속 변수는 추론에서도 중립화해 학습되지 않은 이벤트 플래그가 임의로 신경망을 자극하지 않게 했다. 드문 비상수 변수는 IQR=0이어도 일괄 제거하지 않는다.

TabM은 외부 패키지 복제가 아닌 요청한 구조의 순수 PyTorch 구현이다. 각 헤드의 승차 MSE + 0.2×하차 MSE를 먼저 계산한 뒤 K개 손실을 평균한다. 출력 평균을 먼저 낸 손실은 사용하지 않는다. AdamW(lr1e-3, weight_decay1e-4), batch4096, CosineAnnealingLR, 최대30epoch, ES5, gradient norm5를 적용한다. CUDA만 FP16 autocast/GradScaler, MPS·CPU는 FP32다. 추론은 각 헤드의 복원 로그를 0 이상으로 제한한 후 8개를 평균한다.

XGBoost의 검증 DMatrix와 추론 예측의 일치를 유지하는 기존 호환 래퍼를 사용한다. 실제 GPU smoke에서 로그 잔차 RMSE 재계산과 학습 로그 지표가 1e-6 이내 일치했다. CatBoost GPU는 고정 시드여도 부동소수점 연산 순서 때문에 비트 단위 결정성을 보장하지 않는다.

## 6. 앙상블과 최종 재학습

SLSQP로 비음수·합계1인 가중치를 Fold3+4 합산 OOF에서 적합한다. 모든 결합은 로그 공간에서 수행한다. 상수 절편을 추가하지 않는다. 비교용 전체4Fold 가중치, 4Fold LOFO, Fold3↔4 교차 점검, 중첩 target 날짜를 제거한 민감도 점검을 함께 기록한다.

최적화 구간에는 단일 모델도 가중치1인 허용해로 들어가므로 적합 점수가 최고 단일 모델보다 나쁘면 실패로 처리한다. 그러나 미관측 미래나 다른 구간에서는 앙상블이 더 나쁠 수 있다. LOFO 및 최근 폴드 교차 점검도 base model early stopping/피처 설계에 CV를 쓴 한계가 있다.

최종 반복 수는 round(1.12×(0.4×best_fold3+0.6×best_fold4))로 고정한다. 각 모델을 시드2026/2027로 전체 cutoff 2026-06-30 데이터에서 학습한다. 시드별 로그 예측을 평균한 뒤 모델별 최적 가중치로 결합한다. 반복 상한에 도달했으면 충분히 수렴했다고 단정하지 않고 보고한다.

## 7. 실행·저장·검증

```bash
python -m unittest discover -s tests -v
python -m src.pipeline_v3 run
python -m src.pipeline_v3 infer
```

같은 소스/설정/입력으로 run을 재실행하면 저장된 fold/model 결과를 재사용한다. 입력 해시가 다르면 재사용하지 않는다. 학습 로그는 fold/model/train.log, 최적 반복 수와 장치는 result.json, 학습 곡선은 history.json이다. 결과·모델 저장 성공 후에만 완료 메타데이터를 기록한다.

RunPod의 RTX PRO 4500 Blackwell 32GB 1장에서는 CPU LightGBM 큐와 단일 GPU CatBoost/XGBoost/TabM 큐를 병행한다. CUDA가 2장이면 XGBoost를 GPU1에 배정한다. CPU·MPS 환경은 같은 코드에서 자동 선택한다. 이번 전체 GPU 실행이 모든 하드웨어에서의 완전한 재현 검증을 뜻하지 않는다.

Kaggle/Colab용 `notebooks/SINCON_reproduce_v3.ipynb`는 최신 소스/설정/테스트/일정 JSON과 해시를 내장했다. 기존 v2 노트북은 수정하지 않았다. CUDA PyTorch가 있으면 유지하며 오프라인 설치용 wheel은 별도 준비해야 한다.

제출 검수: 157,194행, example_id 순서/유일성, 두 열, finite/nonnegative, 평균/최솟값/분위수/월별 분포/SHA-256. 평균315~345, 최솟값3미만은 진단 플래그이며 강제 후처리 기준이 아니다. 검증 정답/예측을 임의로 제거하거나 Public 점수에 맞춘 보정을 하지 않는다.

보조 지표는 기존 metrics.py를 유지한다. RMSLE/MAE/RMSE/MBE/RMSE÷MAE, 7일 계절 MASE, 역×시간 단위 날짜 시차1/7/28잔차 상관, horizon·월·역별 리포트를 출력한다. 날짜 블록 bootstrap 구간은 현재 검증 데이터에 조건부인 기술 통계이며 미래 점수 보장 범위가 아니다.
