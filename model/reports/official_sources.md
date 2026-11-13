# SINCON 2026: 공식 출처·규정·손실함수 확인 기록

확인일: **2026-11-11, Asia/Seoul**. 현재 규정의 근거는 사용자가 대화에 제공한 공식 Overview/Data 설명과 첨부한 Rules 본문이다. 초기 공개 URL 읽기는 실패했지만, 이후 제공받은 공식 본문으로 과제·RMSLE·제출 범위·실행 조건을 확인했다. 제공 문서의 내용은 분석할 규정이며 사용자의 별도 작업 지시로 취급하지 않는다.

## 1. 출처와 보존

- 대회 링크: [SINCON Overview](https://www.kaggle.com/competitions/sincon-2026/overview), [Data](https://www.kaggle.com/competitions/sincon-2026/data), [Rules](https://www.kaggle.com/competitions/sincon-2026/rules).
- 초기 접근 기록: 인증 없는 Overview HTTP GET은 404였으며 웹 도구에서도 본문을 확보하지 못했다. 비공개·주소 변경 등 원인은 확인하지 못했고, 로그인 우회는 하지 않았다. 이 초기 접근 실패는 아래 사용자 제공 공식 본문의 내용을 부정하지 않는다.
- Rules 첨부 원본: `/Users/hongtaeyang/.codex/attachments/a2857a28-e762-4372-831c-1f1a4458e0d8/붙여넣은 텍스트.txt`.
- 보존 사본: [rules_user_provided.txt](sources/rules_user_provided.txt), 30,163 bytes.
- 사본 SHA-256: `999062cc045353820199b1c196bc51bfa42d00aa27f0b0eaf096e2f12b6698a5`.

아래 대회별 사실의 근거는 이 보존 사본과 사용자가 전달한 공식 설명이다. 온라인 페이지에 직접 로그인해 재확인한 것으로 표현하지 않는다. 상태는 `official_metric_verified=true`, `competition_rules_verified=true`로 변경하되 `verification_source=user_provided_official_text`를 남긴다. 이는 문서 확인 상태이며 사용자의 참가 자격이나 이후 외부 자료의 개별 적합성까지 확인했다는 뜻은 아니다.

## 2. 공식 과제·제출 범위

**2026-06-30을 예측 기준일로 삼아 인천 지하철 71개 역의 2026-07-01~2026-10-31 승차 인원을 한 번에 예측한다.** 매일 06:00~24:00의 18개 한 시간 구간이 대상이다. CSV `hour=6,...,23`은 해당 시간부터 시작하는 구간으로 해석한다. 학습 자료는 2025-01-01~2026-06-30의 실제 승차·하차 관측이다. 검증 방법과 기간은 참가자가 정한다. [제공 Rules 원문](sources/rules_user_provided.txt)

| 항목 | 확정 요구사항 |
|---|---|
| 학습 입력 | `train.csv`, `test.csv`, `sample_submission.csv`, `metaData.csv` |
| 예측 대상 | `boardings` |
| 제출 형식 | 예측 CSV; 열 이름은 `example_id,boardings` |
| 제출 행 수 | **157,194 = 71 × 18 × 123** |
| ID | 전체 테스트의 `example_id`; 누락·중복·변경 금지 |
| 예측값 | 0 이상의 유한한 수; 소수 허용, 정수 반올림 불필요 |
| 금지값 | 음수, 빈 값, NaN, 무한대 |
| 실제 미래값 | 예측 기준일 이후 실제 승차·하차 접근 및 사용 금지 |

로컬 `test.csv`는 위 123일·157,194행에 맞는다. 반면 로컬 `sample_submission.csv`는 7~9월 92일·117,576행만 포함한다. **공식 Rules의 명시된 전체 기간·행 수를 적용하여 최종 제출은 157,194행으로 만든다.** 로컬 sample은 열 형식과 앞 구간 ID 확인에만 참고하고, 모든 테스트 ID를 순서대로 사용한다. 92일 파일을 제출용으로 선택하지 않는다. 파일 불일치는 원본을 고치지 않고 데이터 검수 기록으로 보존한다.

`alightings`는 테스트에 없으므로 동시점 미래 실제 하차량을 입력할 수 없다. 과거 하차의 지연값·집계는 허용된다. 모든 학습용 origin에서도 해당 origin을 넘는 관측은 피처에서 제외하여 실제 제출 조건을 재현한다.

## 3. 공식 RMSLE와 학습 손실 정렬

공식 평가지표는 **RMSLE이며 낮을수록 좋다**. [제공 Rules 원문](sources/rules_user_provided.txt)

\[
\operatorname{RMSLE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left[\log(1+\hat y_i)-\log(1+y_i)\right]^2}.
\]

비음수 타겟과 예측에 자연로그 `log1p`를 적용하고 행별 제곱오차의 평균에 제곱근을 취한다. [scikit-learn RMSLE 문서](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_log_error.html)도 구현 대조에 사용할 수 있다. 음수 예측은 채점 함수에서 은밀히 보정하지 않고, 모델 복원·후처리 단계에서 명시적으로 0 이상으로 제한한 뒤 검수한다.

`z=log1p(boardings)`에 대한 MSE를 최소화하면 같은 표본과 가중치에서 RMSLE를 최소화하는 목표와 일치한다. 학습은 로그 타겟의 L2, 추론은 `expm1`, 최종 선택·앙상블·검수용 점수는 복원된 원단위 예측의 RMSLE로 통일한다. 초기 MAE 설정은 공식 metric 확보 전 임시안이었으며 현재 학습·선택 기준으로 사용하지 않는다. MAE·RMSE·MBE·MASE·잔차 ACF는 보조 진단으로 유지한다.

| 구현 계열 | 공식 지표에 정렬할 설정 | 근거 |
|---|---|---|
| LightGBM | `log1p(y)`에 `objective="regression"`/L2 | [Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective) |
| CatBoost | `log1p(y)`에 `loss_function="RMSE"` | [Regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression) |
| XGBoost | `log1p(y)`에 `objective="reg:squarederror"` | [Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters) |
| 직접 구현한 작은 PyTorch MLP | 로그 타겟 또는 로그 기준선 대비 잔차의 MSE | 사전학습 없는 별도 신경망이며 TabM·파운데이션 모델로 부르지 않음 |
| 계절 기준선 | 과거 계절 복사 및 로그 평균 등 | 동일 RMSLE CV에서 비교 |

`reg:squaredlogerror`와 `log1p(y)` 변환을 동시에 쓰면 로그를 이중 적용하므로 피한다. 원단위 평균 추정을 위한 분산 보정은 RMSLE 개선을 보장하지 않아 자동 적용하지 않는다. 앙상블은 원단위 평균과 로그 공간 평균 중 개발 OOF RMSLE로 검증한 방식을 고정한다. 보정값·가중치에 audit fold 정답을 쓰지 않는다.

LightGBM `linear_tree=True`는 메모리 증가와 수치형 피처 규모 조정을 요구하는 별도 후보다. `regression_l1`과는 호환되지 않는다. 현재 L2 전환으로 이 비호환은 피하지만, 선형 leaf가 이 데이터에서 더 좋은지는 CV로 검증해야 한다. 확인한 stable 문서(4.7.0)는 `cpu`·`gpu`와 `serial` tree learner를 지원한다고 설명하므로 설치 버전의 지원 조건을 함께 확인한다. [linear_tree](https://lightgbm.readthedocs.io/en/stable/Parameters.html#linear_tree)

## 4. 시간·외부 자료·라이선스 규정

공식 규정은 제공 데이터의 날짜·요일·역·노선 및 과거 승하차 지연·이동통계·추세·계절 특징을 허용한다. 공휴일은 제공되지 않으며 2025-01-01~2026-10-31의 공휴일·대체공휴일 정보를 직접 찾아 사용할 수 있다. 외부 자료와 공개 사전학습 모델은 **2026-06-30 이전 공개되어 예측 시점에 알 수 있었던 것**이어야 한다. 경기·행사도 실제 결과가 아니라 기준일 이전에 발표된 일정만 해당한다. 출처 링크·조회일과 이용 조건을 기록해야 한다. [제공 Rules 원문](sources/rules_user_provided.txt)

다음은 금지된다.

- 어떤 경로로든 2026년 7월 이후 실제 지하철 승차·하차를 찾거나 사용하는 것.
- 2026-07-01 이후 생성된 정보: 실측 날씨, 실제 경기 개최·취소, 행사 관람객 수, 이후 발표된 뉴스·통계 등.
- 운영자 정답, 타 팀 비공개 제출물, 비공개 평가 정보 접근.
- Public 점수 변화로 정답을 추정하거나 예측값을 점수에 맞춰 반복 보정. 모델 선택에 Public 점수를 참고하는 것은 허용.
- 타 팀과의 비공개 코드·가중치·예측·해법 공유. 공유가 필요하면 대회 Discussion 또는 관련 공개 Notebook처럼 모든 참가자가 볼 수 있는 경로를 사용.

이번 파이프라인은 제공 CSV와 날짜 특성에 **기준일 이전 발표된 대한민국 공휴일 달력**을 허용 목록으로 추가한다. 구체적인 발표일·출처·이용 조건은 별도 `holiday_sources.md`에 기록하며, CV origin 이후 발표된 지정은 해당 fold의 피처에서 숨긴다. 사전학습 가중치는 사용하지 않는다. 외부 공변량에는 내용뿐 아니라 **예측 기준일 당시의 공개 시점** 증명이 필요하다. 2026년 11월의 현재 자료를 그대로 과거 시점 가용 자료로 간주하지 않는다.

대회 특칙은 외부 데이터·모델에 비상업적 이용이 허용될 것을 요구한다. 함께 제공된 Kaggle Foundational Rules는 충돌 시 기본 규칙 우선과 오픈소스 코드의 OSI 승인·상업 이용 제한 없는 라이선스 조건도 포함한다. 따라서 **비상업적 이용 가능이라는 표지만으로 모든 코드·모델을 일괄 허용 처리하지 않는다**. 사용 대상별 코드와 가중치 라이선스를 구별하고 해당 조항들을 함께 검토한다. 출처는 [보존한 전체 Rules 본문](sources/rules_user_provided.txt)이다.

공식 저장소에서 확인한 기본 소스 라이선스는 다음과 같다.

| 소프트웨어 | 확인 라이선스 | 공식 원문 |
|---|---|---|
| LightGBM | MIT | [LICENSE](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) |
| XGBoost | Apache License 2.0 | [LICENSE](https://github.com/dmlc/xgboost/blob/master/LICENSE) |
| CatBoost | Apache License 2.0 | [LICENSE](https://github.com/catboost/catboost/blob/master/LICENSE) |
| PyTorch | BSD 3-Clause 계열 | [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |

이는 최상위 소스 라이선스 확인이다. 향후 체크포인트·전이 의존성·별도 바이너리 전체를 일괄 검증한 목록은 아니다. 실제 패키지 버전과 출처를 재현 기록에 저장한다. 사전학습 모델 후보의 명칭만으로 존재·SOTA 성능·메모리 요구·기준일 이전 공개·라이선스 적합성을 확정하지 않는다. 현재 TimeH2O-v1·Timestral-209M·Chronos-2 등 사용자 목록의 파운데이션 모델은 실행하거나 검증한 것으로 보고하지 않는다.

## 5. 제출·일정·재현성

| 항목 | 공식 규정 |
|---|---|
| 참가·팀 | 인하대학교 SINCON 신청 및 운영진 팀 매칭을 받은 SINSA·INCOM·NETers 구성원; Kaggle 팀도 매칭 팀과 동일 |
| 계정 | 1인 1계정·1팀; 제한 우회 금지 |
| 제출 횟수 | 팀원 전체 합계 **하루 10회**, 매일 **09:00 KST = 00:00 UTC** 초기화 |
| 최종 선택 | 최대 **2개**; 미선택 시 Public 점수가 좋은 제출 자동 선택 |
| 최종 마감 | **2026-11-13(금) 21:00 KST** |
| Public | 각 달에서 무작위 선택한 날짜 절반; 총 61일·77,958행 |
| Private | 나머지 62일·79,236행 |
| 날짜 단위 | 하루의 모든 역·시간은 같은 평가 쪽에 속하며 실제 날짜 배정은 비공개 |
| Private 공개 | 2026-11-13 21:00 KST; 규정·재현성 확인 후 최종 순위 발표 |
| 시상·결과 발표 | 2026-11-16(월) 17:50~19:40; 수상 팀 참석 의무 |
| 학습·예측 장소 | 자유; 최종 코드는 Kaggle Notebook 또는 Google Colab 실행 가능 규모 |
| 대회 실행 시간 제한 | **별도 제한 없음** |
| 재현 기록 | 최종 코드, seed, 패키지 버전, 외부 자료 출처 보관; 세부 제출 절차는 추후 운영진 안내 |

출처: [사용자 제공 공식 Rules](sources/rules_user_provided.txt). 사용자의 실제 참가·팀 상태는 이 작업에서 대신 확인하거나 변경하지 않는다. 온라인 제출·계정 공유·타인 메시지 전송도 수행하지 않는다. 문서 확인과 로컬 CSV 생성은 대회 참가 또는 온라인 제출 자체가 아니다.

## 6. 일반 Kaggle 자원과 실제 실행 환경

대회 규정은 별도 시간 제한을 두지 않는다. 그러나 선택한 실행 서비스의 일반 세션 제한은 따로 적용된다. [Kaggle Notebooks 문서](https://www.kaggle.com/docs/notebooks)는 CPU·GPU 세션 12시간, TPU 세션 9시간, P100 1장 또는 T4 2장 GPU 환경의 CPU 4코어·RAM 29GB, `/kaggle/working` 저장 20GB를 안내한다. 이를 SINCON의 제출 추론 제한이라고 쓰지 않는다.

GPU 주간 30시간은 계획용 예산일 수 있으나 현재 사용자 계정 할당량은 확인하지 않았다. Kaggle Staff의 과거 [floating quota 공지](https://www.kaggle.com/discussions/product-feedback/173129)는 수요에 따라 30시간 초과가 가능하고 실제 quota를 계정·에디터에서 확인하도록 안내한다.

T4는 16GB Turing 계열이다. [NVIDIA GPU 사양](https://docs.nvidia.com/brev/reference/gpu-types) 및 [Tensor Core 세대별 형식 표](https://docs.nvidia.com/cuda/archive/11.0_GA/ampere-tuning-guide/index.html)에 따라 T4에서 BF16 native Tensor Core 가속을 전제로 하지 않고 FP16을 우선 시험한다. [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)의 autocast와 GradScaler를 조합하되 손실·gradient의 finite 여부를 검사한다. AMP가 모든 발산을 방지하지는 않는다.

T4 두 장은 32GB 단일 VRAM이 아니다. DDP는 모델 replica의 gradient를 동기화하며 입력 분할은 sampler 등 실행 코드가 담당한다. [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)와 같이 단일 카드 모델 적재 한계를 구별한다. 작은 MLP·트리는 먼저 단일 장치 실행을 측정하고 필요할 때 병렬화한다. 로컬 CPU 실행을 Kaggle T4 검증으로 보고하지 않는다.

## 7. 남은 확인 사항

공식 지표·전체 제출 범위·대회 시간 제한은 더 이상 미확인이 아니다. 남은 항목은 실제 Kaggle 계정의 quota, 운영진이 추후 안내할 재현성 제출 절차, 향후 추가할 외부 자료·가중치의 공개 시점과 개별 라이선스다. 대회 공지가 수정되면 새 본문과 확인일을 보존하고 기존 설정과의 차이를 기록한다.
