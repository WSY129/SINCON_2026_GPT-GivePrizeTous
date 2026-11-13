# SINCON 학습 이후 처리·123일 추론·제출 검수 상세

작성 기준: 2026-11-12 로컬 파일 확인 시점. 실제 저장된 코드·설정·CSV·JSON을 기준으로 기술한다. 새 학습을 수행한 보고서가 아니며, 완료된 최신 버전은 v5다. v6는 작업 중인 별도 소스로 분리한다.

## 1. 현재 완료된 추론 경로

이 문서는 v5의 저장된 최종 모델 14개와 제출 파일을 기준으로 한다. Public 최고 제출은 v2이고, v5는 아직 Public 성적이 확인되지 않았다. 추론 결과를 만들 때에는 학습 버전의 기준선·피처·전처리기·범주 매핑·모델·가중치를 한 묶음으로 사용해야 한다. v5 모델에 구현 중인 v6 특징을 넣으면 같은 모델의 재현 추론이 아니다.

```text
입력 CSV 해시 확인
 → train 관측을 2026-06-30까지 제한
 → test를 date/station_id/hour 순서로 정렬
 → origin 시점 달력·기준선·과거 통계 생성
 → 저장된 모델별 전처리 적용
 → 7개 모델군 × seed 2026/2027 로그 예측
 → 모델군별 두 seed의 로그 평균
 → 미래 라벨 없이 5개 버킷 지정
 → 버킷별 비음수·합계1 가중 평균
 → expm1으로 인원수 복원
 → example_id로 원래 test 순서 복구
 → 157,194행 CSV 저장·검수
 → 별도 프로세스에서 모델 재로딩·동일성 검사
```

추론 중 미래 실제 승차·하차를 읽거나, 7월 예측 뒤 7월 실제값으로 8월 입력을 갱신하지 않는다. 모든 horizon 1~123일은 동일한 origin=2026-06-30의 정보로 직접 예측한다. 입력 구간 길이가 있는 공간 모델도 마지막 관측 구간은 origin까지다.

## 2. 입력과 시점 검사

원본 파일은 `data/raw/train.csv`, `test.csv`, `metaData.csv`, `sample_submission.csv`다. 번들 manifest가 기록한 원본 SHA-256과 현재 파일의 해시를 대조하며 다르면 재로딩 추론을 중단한다. source·config·input fingerprint는 결과와 코드 조합을 식별한다.

공휴일·외부 일정은 `known_since <= origin`인 정보만 노출한다. 과거 관측 집계는 `date <= origin`이다. 날짜상 미래의 달력 변수라도 origin 전에 확정된 정보라면 사용할 수 있다. 반대로 사후 경기 취소, 실제 강수량, 미래 하차량은 사용할 수 없다. 현재 v5에는 KBO 일정이 들어 있지 않다.

test에는 정답 열이 없고 example_id는 연결 키다. example_id 자체를 예측 특징으로 쓰지 않는다. 실제 로컬 sample_submission은117,576행만 있어 123일 전체를 대표하지 못한다. 제출의 완전한 ID 집합·순서는 **157,194행인 test.csv**를 기준으로 삼는다.

## 3. 추론 전 기준선 보정과 학습 이후 보정의 구분

v5의 계절 커널, 공휴일 상승·하락량, 근로자의 날 충격량, 개통 경계 제한, 신설역 감쇠 성장량은 **학습 사례를 만드는 시점부터 적용하는 anchor 구성 요소**다. 학습이 끝난 예측에 임의로 붙이는 후처리가 아니다. 모델은 이 기준선에 대한 잔차를 학습했고, 추론에서도 같은 함수를 사용한다.

| 항목 | 적용 위치 | 최종 예측에 임의 추가 여부 |
|---|---|---|
| 전년 커널·계절 성장량 | anchor/입력 피처 | 아니오 |
| 공휴일 Sunday lift·근로자의 날 delta | anchor | 아니오 |
| 신설역 탈계절화 감쇠 추세 | anchor | 아니오 |
| NN checkpoint 평균 | 저장할 모델 파라미터 | 적용 |
| 같은 모델군 두 seed 평균 | 로그 예측 | 적용 |
| 5-Bucket 가중 평균 | 로그 예측 | 적용 |
| 0 이상 로그 예측 복원 | 모델 출력 | 적용 |
| MBE 상수 보정·평균 강제 조정 | 예측 후 인원수 | 적용하지 않음 |
| 휴일 수요 0 고정·정수 반올림 | 예측 후 인원수 | 적용하지 않음 |

이 구분은 후처리의 효과를 평가할 때 중요하다. anchor를 바꾸면 잔차의 정의가 바뀌므로 해당 모델 재학습까지 포함한 별도 실험이 필요하다.

## 4. 모델별 추론 처리

### 4.1 직접 트리

LightGBM·CatBoost·XGBoost는 학습 때 저장한 특징 열 순서와 범주 매핑을 사용한다. 스케일링을 하지 않고 float32 수치와 결측치를 처리한다. CatBoost 범주는 문자열, 나머지는 저장된 category 체계로 맞춘다. 잔차 예측에 해당 행의 anchor를 더해 로그 승차량을 만든다.

$$\hat z=\max(0,a+\hat r),\qquad \hat y=\exp(\hat z)-1.$$

현재 LightGBM은 CPU, CatBoost도 CPU 정책이며 XGBoost는 사용 가능한 CUDA 장치를 활용한다. 모델 저장 시 CPU 객체로 옮긴 기록과 실제 학습·추론 장치를 구별한다. 재로딩 함수는 XGBoost의 CUDA backend를 사용 가능한 경우 복원한다. 저장된 실행에서는 XGBoost3.4.1의 CPU/CUDA predictor가 차이를 보였으므로, CPU에서 실행 가능하다는 것과 GPU 결과의 bitwise 동일성은 같은 약속이 아니다.

### 4.2 Multi-Task TabM

폴드/최종 학습에서 적합한 중앙값·IQR·범주 사전을 그대로 로드한다. 추론 데이터로 scaler를 재적합하지 않는다. 원래 NaN 마스크를 유지하면서 year364 관련 결측을 anchor로 먼저 대체하고 나머지를 학습 중앙값으로 채운다. 로버스트 스케일링 결과는[-10,10]으로 제한한다.

16개 헤드가 승차·하차 잔차를 출력한다. 각 승차 헤드의 `max(0, anchor_board_log + residual)`을 구한 뒤 16개 **로그값**을 평균한다. 하차 출력은 제출에서 버린다. 미래 하차 라벨은 학습 보조 채널에만 쓰였으며 추론 입력이 아니다.

CUDA에서는 FP16 자동 혼합정밀도를 사용하고 MPS/CPU에서는 FP32를 사용한다. GradScaler는 학습 역전파의 수치 안정화 도구이지 추론 때 예측을 보정하는 도구가 아니다.

### 4.3 일간 총량 × 시간 점유율 분해 모델

역·날짜별18시간이 하나의 그룹이다. 하루 총량의 로그 예측은 LGB/CatBoost 결과를 평균한다. 시간별 점유율 모델 출력은 softmax로 정규화해 합계1을 맞춘다. 학습 시 각 시간에 더한0.5와 하루 총18×0.5=9의 smoothing을 고려해 인원수를 복원하고 0 이상으로 제한한다.

총량 예측과 점유율의 결합에는 그룹 완전성·행 순서가 필요하다. 한 역·날짜를 임의로 잘라18개 미만으로 전달하면 같은 의미의 추론이 아니다. 내부 정렬 후 마지막에 example_id로 원래 순서를 되돌린다. 0 절삭으로 smoothing 역변환 이후 합계가 원래 총량과 미세하게 달라질 수 있으므로, softmax 합계1과 최종 절삭 후 인원수 합계 일치를 혼동하지 않는다.

### 4.4 저수요 전문 모델

전문 모델의 학습 게이트는 `anchor<3.3 또는 hour>=21`이고 메타 버킷의 저수요 기준은 `anchor_v5<3.2 또는 hour>=21`이다. 비게이트 구간에서 반환하는 anchor가 일반 구간 예측에 섞이지 않도록, 메타 단계에서는 `late_or_low` 외 모든 버킷에서 이 모델의 가중치를 강제로0으로 만든다.

실제 v5에서 late_or_low의 최적 가중치도0이었다. 따라서 이 모델은 학습·검수에는 포함되지만 현재 제출에 수치적으로 기여하지 않는다. 이전 v4의 비게이트 오염 방지와 실제 최적화 결과를 구분해 기록한다.

### 4.5 공간·시간 Transformer

origin 이전56일의 승차·하차 로그값과 결측 마스크, 노선 인접·환승 관계로 집계한 이웃 정보를 사용한다. 8개 주간 패치에 attention을 적용한 과거 표현과 미래 날짜 피처를 결합해18시간 잔차를 직접 출력한다. 논문 원형 iTransformer와 동일한 구조라고 간주하지 않는다.

같은 모델이라도 origin이 다른 입력 맥락을 쓰면 결과가 달라진다. 제출용 모델은 최종 origin의 저장된 context와 피처를 사용한다. 두 seed를 각각 추론하고 로그 공간에서 평균한다.

## 5. 파라미터 평균·헤드 평균·seed 평균·모델 평균

현재 네 단계는 서로 다른 연산이다.

1. **Checkpoint 파라미터 평균:** CV 신경망은 검증 상위3개 epoch의 파라미터를 평균했다. 최종 전체 학습에는 별도 검증 정답이 없어 예정된 마지막3개 epoch를 평균했다.
2. **TabM 헤드 평균:** 한 checkpoint가 출력하는16개 승차 로그 예측을 평균한다.
3. **Seed 평균:** 같은 모델군의2026/2027 로그 예측을 평균한다.
4. **모델군 가중 평균:** 행의 버킷에 따라7개 모델군의 로그 예측을 결합한다.

$$\bar z_{m,i}=\frac{\hat z_{m,2026,i}+\hat z_{m,2027,i}}{2},\qquad
\hat z_{i}=\sum_mw_{b(i),m}\bar z_{m,i},\qquad
\hat y_i=\operatorname{expm1}(\hat z_i).$$

인원수를 먼저 평균하는 것은 위 식과 다르다. 로그 평균은 `(1+y)`의 가중 기하평균에 해당하며 RMSLE 목적과 연결된다. 원시 인원수 평균으로 바꾸면 큰 값의 영향과 전체 평균이 달라진다. 각 seed의 인원수를 반올림한 뒤 평균하는 방식도 사용하지 않는다.

## 6. 버킷 결정과 배포 가중치

버킷은 다음 순서로 결정하며 서로 겹치지 않는다: 신설3역 → 계양·귤현 → window_spike_amp≥0.65 → 심야/저수요 → regular. 진폭은 origin 이전 전년창 관측으로 계산한다. 미래 실제값이 큰 행을 사후적으로 high_spike로 지정하지 않는다.

| 버킷 | LGB % | Cat % | XGB % | TabM % | 분해 % | 저수요 % | 공간 % |
|---|---|---|---|---|---|---|---|
| new_stations | 0.0000% | 0.0000% | 29.5769% | 23.7452% | 46.6779% | 0.0000% | 0.0000% |
| gyeyang_gyulhyeon | 0.0000% | 0.0000% | 0.0000% | 41.5672% | 40.7026% | 0.0000% | 17.7303% |
| high_spike | 16.8612% | 32.9540% | 0.0000% | 3.8926% | 46.2922% | 0.0000% | 0.0000% |
| late_or_low | 0.0000% | 36.0339% | 0.0000% | 63.9661% | 0.0000% | 0.0000% | 0.0000% |
| regular | 30.5930% | 14.6931% | 0.0000% | 46.8264% | 0.0000% | 0.0000% | 7.8875% |


실제 순서·원정밀도 값은 bundle_manifest와 ensemble_report를 사용한다. 위 표를 복사해 반올림된 비율로 다시 추론하면 완전히 같은 파일을 보장하지 않는다. 전역 가중치는 수축 prior와 비교용이며 제출에 전역 비율을 일괄 적용하지 않았다.

버킷별 비중은 검증에서 학습했고 새 test 분포에 맞춰 다시 최적화하지 않는다. Public 점수0.18265를 목표값으로 넣거나 Public 정답을 역추정하는 연산도 없다.

## 7. 실제 제출 파일의 무결성

```json
{
  "rows": 157194,
  "ids_in_test_order": true,
  "finite_nonnegative": true,
  "mean": 335.80370658704453,
  "min": 2.070506028063961,
  "max": 2822.7312287016293,
  "sha256": "500f7aca090e4157e13712b41b7a90f58aed00fffb5624c083e6f5d6a378c12b"
}
```


실제 생성 파일: [outputs/ensemble_v5/submission_ensemble_v5.csv](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/submission_ensemble_v5.csv)

| 검사 | 결과/의미 |
|---|---|
| 행 수 |157,194 |
| 열 |example_id, boardings |
| ID |test 순서와 일치, 중복 없음 |
| 값 |NaN·무한대·음수 없음 |
| 평균 |335.803707명 |
| 최소 |2.070506명 |
| 최대 |2,822.731229명 |
| CSV 숫자 저장 |소수점8자리, 정수 반올림 아님 |
| Public |미확인 |

평균315~345, 최소3 미만은 사전 제시된 **분포 진단 기준**이다. 값을 이 범위에 맞추기 위해 일괄 비례 조정하거나 최솟값을 강제로 낮추지 않았다. 범위를 통과했다고 예측 정확도가 입증되는 것도 아니다. 월별 분포는 다음과 같다.

| 월 | 평균 | 최소 | 최대 |
|---|---|---|---|
| 2026-07 | 332.831987 | 3.186643 | 2670.133824 |
| 2026-08 | 322.760284 | 3.034972 | 2696.526802 |
| 2026-09 | 338.821072 | 2.070506 | 2820.892642 |
| 2026-10 | 348.898818 | 2.889280 | 2822.731229 |


## 8. 별도 프로세스 재로딩 검증

최종 학습 프로세스의 메모리 안에 남은 모델을 그대로 재사용하는 것만으로 저장 무결성을 확인하지 않았다. 별도 프로세스가 각 model.joblib을 다시 로드하고 원본 입력으로 피처를 다시 생성해 로그 예측을 비교했다.

- 7모델군×2seed=14개 재로딩 검사: 모두 `allclose=True`.
- 비교 허용오차: `rtol=1e-6`, `atol=1e-6`.
- 저장된 실행의 최대 로그 예측 차이:14개 모두0.
- 최종 제출 수치 비교 통과, 재생성 CSV의 SHA-256도 원본과 동일.
- 따라서 해당 환경·코드·입력 조합에서는 byte-identical 재생성이 확인됐다.



근거: [outputs/ensemble_v5/reload_validation.json](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/reload_validation.json), [outputs/ensemble_v5/bundle_manifest.json](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/bundle_manifest.json), [outputs/ensemble_v5/local_backup_validation.json](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/local_backup_validation.json).


다른 라이브러리 버전·CPU/GPU backend에서도 bitwise 동일하다는 뜻은 아니다. 환경 변경 시 같은 해시·예측 비교를 다시 해야 한다. 실제 로컬 복구 기록에는507파일,452,015,883bytes, 누락0, 해시 불일치0이 저장돼 있다.

현재 CLI의 재로딩 진입점은 다음과 같다. 이 명령은 저장된 결과 경로 안에 reload 파일을 작성하므로, 원본 보존이 필요하면 **프로젝트와 산출물을 별도 재현용 복사본에 둔 후** 복사본 루트에서 실행한다. 이 문서를 작성하면서 재학습·재로딩을 다시 실행하지 않았다.

```bash
python -m src.v5.pipeline infer
```

CLI는 configs/ensemble_v5.json과 해당 버전 출력 경로를 읽는다. 사용자 임의 경로의 단일CSV만 주면 자동 추론하는 범용 서버 인터페이스는 아니다. 재현에 필요한 모델·전처리기·버킷 가중치·소스·환경·원본CSV가 모두 필요하다.

## 9. 과거 v1 후처리 실험: 채택하지 않은 이유

v1 LightGBM의 검증 RMSLE0.247808을 더 줄이기 위해 별도 후처리 실험을 수행했으나 최종 audit에서 악화되어 제출에 반영하지 않았다. 현재 원본 v1 제출과 v5 제출에 이 보정이 숨어 있는 것이 아니다.

후보54개는 잔차 설정5개×anchor 혼합10개=50개, 전역 bias2개, 진단용 상한2개였다. 개발1에서 보정값을 적합하고 개발2에서 forward 검증해 후보를 고른 다음 개발1·2로 재적합해 마지막123일 audit에 적용했다. 그 audit은 이전 분석에도 사용된 구간이라 완전히 새로운 독립 holdout은 아니다.

선택된 후보는 `hour_dow_0.5_anchor28_0.2`였다.

1. 로그 잔차 `z_true-z_pred`에서 fold별 평균을 제거한다.
2. 시간×요일 그룹 잔차를 `sum/(n+1000)`으로 수축한다.
3. [-0.1,+0.1]로 제한하고0.5를 곱한다.
4. 보정된 base 로그 예측80%와 정제28일 anchor20%를 섞는다.
5. 공휴일은 보정하지 않고 기존 base를 그대로 둔다. anchor가 없을 때도 fallback을 적용한다.

$$\delta_g=0.5\operatorname{clip}\left(\frac{\sum e_{g,centered}}{n_g+1000},-0.1,0.1\right),\qquad
z_{post}=0.8(z_{base}+\delta_g)+0.2z_{anchor28}.$$

| 같은 최종 audit 구간 | RMSLE |
|---|---:|
| 원본 LightGBM |0.247808329 |
| 잔차 보정만 |0.249572512 |
| anchor 혼합만 |0.248557340 |
| 선택된 결합 후처리 |0.249953530 |
| 악화량 |+0.002145202, 약0.87% |

공휴일 RMSLE는0.374880257로 동일했고 비공휴일은0.237979219→0.240346301로 악화됐다. 날짜 단위2,000회 bootstrap에서 악화량의 구간은 약[+0.000741,+0.003657]였다. 이는 해당 검증 날짜 표본에 조건부인 결과이며 장기 시계열 의존까지 완벽히 반영한 보장은 아니다.

전역 bias 역시 개발1에서는 좋아졌지만 개발2에서 악화됐다. 예를 들어0.5배 bias는 개발1에서0.302387→0.297429로 개선됐으나 개발2에서0.223491→0.236470으로 악화됐다. 단순 과소예측 수치만 보고 상수를 더하는 방식을 배제한 실증 근거다.



근거: [reports/POSTPROCESSING.md](/Users/hongtaeyang/Desktop/SINCON/reports/POSTPROCESSING.md).


## 10. 현재 적용하지 않는 후처리와 이유

- **MBE 0 강제:** 원시 인원수 평균 편향을 제거하는 것과 로그 제곱오차 최소화는 다르다.
- **테스트 평균 강제 조정:**315~345 같은 예상 범위는 확인용이며 정답 분포가 아니다.
- **휴일·영업종료 구간0 고정:** 이 과제의06~24시 제공 구간에는 공휴일 실제 수요가 존재한다. 일률적인0 처리는 부적절하다.
- **이벤트 급등 상한 삭제:** 반복적인 야구·행락·축제 수요는 정상 신호일 수 있다. 원본 타겟을 일괄 절삭하지 않는다.
- **최소3명 미만 강제 생성:** 낮은 수요를 자연스럽게 예측하는지 확인할 뿐 값 자체를 맞추지 않는다.
- **MAE식 중앙 분위수로 전환:** 공식 지표는RMSLE이며, 현재 학습 목표는 로그 평균에 대응한다.
- **공개 순위에 맞춘 추가 blending:** 새 가중치는 OOF 비교·누수 점검 후 별도 버전으로 관리한다.

비음수 복원·정규화·로그 앙상블은 적용하되, 성능 근거가 없는 편향 보정은 현재 제출에 추가하지 않는 것이 실제 상태다.

## 11. 검증 이후 함께 보는 보조 지표

v5 Fold4의 버킷 앙상블 결과는 다음과 같다.

| 지표 | 값 | 해석 |
|---|---:|---|
| RMSLE |0.194796131 |공식 지표와 동일한 로컬 산식 |
| MAE |31.270572 |원시 인원수 평균 절대오차 |
| RMSE |98.941009 |큰 원시 오차에 민감 |
| MBE |−9.880678 |평균적으로 과소예측 |
| RMSE/MAE |3.164029 |큰 오차가 일부 존재 |
| 7일 MASE |0.740960 |훈련 이력의7일 단순 차이 척도로 정규화 |
| 잔차 ACF1일 |0.132905 |잔차 시간 의존이 남음 |
| 잔차 ACF7일 |0.200606 |요일·주간 구조의 남은 오차 |
| 잔차 ACF28일 |0.106510 |장기 반복 구조의 남은 오차 |

MASE의 분모는 각 역×시간 훈련 관측에서 정확히7일 간격인 값들의 평균 절대차이다. 분모가 없거나0인 경우를 처리하고 유효 커버리지도 함께 기록한다. ACF는 동일 시계열의 정확한 날짜 시차에 대한 잔차 상관을 구해 집계한다. 모델의 shuffled 학습 행 순서를 시간축으로 간주하지 않는다.

RMSE/MAE>1.5 또는 ACF>0.1은 진단용 경험 기준이며 특정 원인이 확정됐다는 판정식이 아니다. 월별·역별·horizon별 오차로 어느 조건에서 집중되는지 추가 확인해야 한다. 현재 Fold4의 날짜 bootstrap RMSLE 구간은 약[0.183467,0.206985]이며 그 자체를 Public 점수 예상 구간으로 사용하지 않는다.

## 12. 전달·보관해야 할 파일

| 목적 | 경로 |
|---|---|
| 확인된 Public 최고 제출 |[outputs/ensemble_v2/submission_ensemble_v2.csv](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v2/submission_ensemble_v2.csv) |
| 최신 완료 v5 제출 |[outputs/ensemble_v5/submission_ensemble_v5.csv](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/submission_ensemble_v5.csv) |
| 재생성 제출 |[outputs/ensemble_v5/submission_reload.csv](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/submission_reload.csv) |
| 최종 모델14개 |[outputs/ensemble_v5/final](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/final) |
| 원정밀도 가중치·모델 목록 |[outputs/ensemble_v5/bundle_manifest.json](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/bundle_manifest.json) |
| 전체 앙상블 검증 |[outputs/ensemble_v5/ensemble_report.json](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/ensemble_report.json) |
| 제출 무결성 |[outputs/ensemble_v5/submission_validation.json](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/submission_validation.json) |
| 별도 프로세스 재로딩 검증 |[outputs/ensemble_v5/reload_validation.json](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/reload_validation.json) |
| 입력·코드 fingerprint |[outputs/ensemble_v5/fingerprint.json](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/fingerprint.json) |
| 로컬 백업 검수 |[outputs/ensemble_v5/local_backup_validation.json](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/local_backup_validation.json) |
| 이전 자산 보존 검수 |[outputs/ensemble_v5/desktop_preservation_check.json](/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v5/desktop_preservation_check.json) |
| 실행·추론 구현 |[src/v5/pipeline.py](/Users/hongtaeyang/Desktop/SINCON/src/v5/pipeline.py) |

Public 최고 파일을 최신 로컬 점수 파일로 덮어쓰지 않는다. v5를 제출하더라도 새로운 Public 점수는 별도 기록으로 추가해야 하며, 그 점수가 확인되기 전에는 v2의0.18265를 넘어섰다고 표현할 수 없다.

관련 문서: [대회](/Users/hongtaeyang/Desktop/SINCON/reports/current_pipeline_20261112/01_COMPETITION.md) · [데이터·EDA](/Users/hongtaeyang/Desktop/SINCON/reports/current_pipeline_20261112/02_DATA_EDA.md) · [모델·검증·Public](/Users/hongtaeyang/Desktop/SINCON/reports/current_pipeline_20261112/03_PIPELINE_AND_PUBLIC.md)
