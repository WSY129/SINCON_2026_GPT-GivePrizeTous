"""Build the four requested facts from immutable v2 results and raw inputs."""
from pathlib import Path
import json, hashlib
import pandas as pd
import numpy as np

ROOT=Path('/Users/hongtaeyang/Desktop/SINCON')
OUT=Path(__file__).resolve().parents[1]/'outputs/ensemble_v3/review_v2'
OUT.mkdir(parents=True,exist_ok=True)
V2=ROOT/'outputs/ensemble_v2'
oof=pd.read_csv(V2/'oof_predictions.csv.gz',parse_dates=['date'])
f=oof.loc[oof.fold.eq(4)].copy()
f['log_sq_error']=(f.true_log-f.ensemble)**2
f['error']=np.expm1(f.ensemble)-f.boardings
total=f.log_sq_error.sum(); n=len(f)
meta=pd.read_csv(ROOT/'data/raw/metaData.csv')

def table(frame):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'

def aggregate(keys):
    d=f.groupby(keys,observed=True).agg(rows=('log_sq_error','size'),log_sse=('log_sq_error','sum'),mbe=('error','mean')).reset_index()
    d['rmsle']=np.sqrt(d.log_sse/d.rows)
    d['sse_pct']=100*d.log_sse/total
    d['global_log_mse_contribution']=d.log_sse/n
    return d

stations=aggregate('station_id').merge(meta[['station_id','station_name','canonical_line']],on='station_id').sort_values('log_sse',ascending=False)
hours=aggregate('hour')
f['horizon_bin']=pd.cut(f.horizon,[0,30,60,90,123],labels=['1–30','31–60','61–90','91–123'])
steps=aggregate('horizon_bin')
f['month']=f.date.dt.strftime('%Y-%m');months=aggregate('month')
segments=pd.read_csv(V2/'fold4_domain_segments.csv')
segments=segments.loc[segments.model.eq('ensemble')].copy()
segments['log_sse']=segments.rows*segments.rmsle**2
segments['sse_pct']=100*segments.log_sse/total
segments['global_log_mse_contribution']=segments.log_sse/n
for name,df in [('fold4_station_sse',stations),('fold4_hour_sse',hours),('fold4_horizon_sse',steps),('fold4_month_sse',months),('fold4_segment_sse',segments)]:df.to_csv(OUT/f'{name}.csv',index=False)
report=json.loads((V2/'ensemble_report.json').read_text())
reblend=json.loads((OUT/'reblend_report.json').read_text())
metrics=pd.read_csv(V2/'cv_metrics.csv')
score=[]
for name,m in report['models'].items():score.append({'model':name,**m,'fold4_rmsle':metrics.loc[(metrics.fold==4)&(metrics.model==name),'rmsle'].iloc[0]})
score.append({'model':'ensemble_v2',**report['ensemble'],'fold4_rmsle':float(np.sqrt(total/n))})
schema=[]
for name in ['train','test','sample_submission','metaData']:
    d=pd.read_csv(ROOT/f'data/raw/{name}.csv');schema.append({'파일':name+'.csv','행':len(d),'열':len(d.columns),'정확한 컬럼':', '.join(d.columns)})
folds=[]
for i in range(1,5):
    a=json.loads((V2/f'fold_{i}/feature_audit.json').read_text());v=oof.loc[oof.fold==i]
    folds.append({'fold':i,'origin':a['origin'],'검증 시작':v.date.min().date(),'검증 끝':v.date.max().date(),'검증 행':len(v),'학습 year364 %':100*a['train_year364_available'],'검증 year364 %':100*a['valid_year364_available']})

v3root=ROOT/'outputs/ensemble_v3'
v3complete=(v3root/'ensemble_report.json').exists() and (v3root/'cv_metrics.csv').exists()
v3row='| ensemble_v3 | 실행 중 | 실행 중 | 실행 중 | 아직 제출 결과 없음 |'
v3status='v3는 구현 중이며 아래 점수는 v3 재학습 결과가 아니다.'
if v3complete:
    r3=json.loads((v3root/'ensemble_report.json').read_text())['ensemble']
    m3=pd.read_csv(v3root/'cv_metrics.csv')
    f4=float(m3.loc[(m3.fold==4)&(m3.model=='ensemble'),'rmsle'].iloc[0])
    v3row=f"| ensemble_v3 | {r3['weighted_rmsle']:.9f} | {r3['fold34_rmsle']:.9f} | {f4:.9f} | 아직 제출 결과 없음 |"
    v3status='v3의 4-Fold CV가 완료되어 4번 비교표에도 반영했다. 2·3번 상세 진단은 보존된 v2를 대상으로 한다.'

text=f'''# SINCON — Gemini 전달용 4가지 확인 자료

작성일: 2026-11-12 (한국 시간). **ensemble_v2 완료 산출물**을 다시 읽어 작성했다. {v3status} Gemini 대화의 개선 제안은 검증할 가설로 취급한다.

## 1. 사전 확정 외부 일정의 허용 범위

사용자가 제공한 공식 Rules 원문에 따르면 **2026-06-30까지 공개되어 예측 시점에 알 수 있었던 외부 데이터·공개 사전학습 모델을 사용할 수 있다.** 규정은 공휴일·학사 일정·6월 30일 이전에 발표된 경기 및 행사 일정을 직접 예로 든다. 따라서 KBO 정규시즌 원편성표, 대학 학사 일정, 송도 축제 일정은 아래 조건을 충족할 때 사용 가능하다.

- 자료별 비상업적 이용 허용 여부를 확인하고 출처 URL·조회일을 보관한다. 공개되어 있다는 사실만으로 재사용 라이선스가 확인된 것은 아니다.
- `known_since <= origin`을 학습 사례와 CV에 적용하고, 최종 추론은 `known_since <= 2026-06-30`으로 제한한다. 2026년 6월에 발표된 정보가 2025년 origin에 노출되면 안 된다.
- 사전에 발표된 경기일·홈구장·시작 시각은 사용할 수 있으나, 실제 개최·취소 여부, 경기 결과, 실제 종료 시각, 관중 수, 미래 실측 날씨는 사용할 수 없다.
- 2026년 7월 이후 실제 승하차 실적과 7월 이후 발표된 재편성 일정도 금지다. 취소된 경기를 나중에 원편성표에서 삭제하는 작업 역시 하지 않는다.
- 학사 일정은 학교별 공식 발표 근거가 있어야 한다. 현재 여름·겨울 방학 범위는 일반적인 계절 **proxy**이며 실제 학교별 확정 학사력으로 표현하지 않는다.

**최종 반영 결정(2026-11-12): 사용자가 별도 KBO 이용 허가가 없으며 사용 가능한 출처만 반영하도록 확정했다.** KBO 원편성 일정은 v3 입력에서 제외하고 `domain_schedule_v3.json`의 `kbo` 목록을 비웠다. 공식 발표일을 확인했다는 사실과 재사용 허가 확인은 구분한다. [KBO 이용약관](https://www.koreabaseball.com/Etc/Policy.aspx)의 데이터 가공·재사용 제한에 대한 별도 허가를 확인하지 못했다.

| 자료 | 보수적으로 적용한 공개일 | 사용 여부와 근거 |
|---|---|---|
| KBO 2025·2026 정규시즌 원편성 | 사용하지 않음 | 별도 허가 없음, v3 일정 입력 제외 |
| [2025 펜타포트 인천시 사전 안내](https://www.incheon.go.kr/IC010205/view?repSeq=DOM_0000000012828449) | 2025-07-30 | 공공누리 제1유형 확인. 2025-08-01~08-03 송도달빛축제공원 일정 사실만 사용 |
| [2026 펜타포트 인천시 안내](https://www.incheon.go.kr/moo/MOO030301/3074869) | 2026-05-31 | 공공누리 제1유형 확인. 2026-07-31~08-02 일정 사실만 사용 |
| 송도맥주축제 | 사용하지 않음 | 사전 발표와 재사용 조건을 함께 확인하지 못함 |
| 대학·고교 학사 일정 | 공식 일정으로 사용하지 않음 | 학교별 확정 학사력 대신 명시적인 계절 proxy 사용 |

축제 일정 출처·저작자: 인천광역시. 조회일 2026-11-12. 게시물의 참석 인원·개최 결과·사후 변경은 입력하지 않는다. 2025 축제의 더 이른 민간 공지는 재사용 조건이 확인되지 않아 공개일을 소급하지 않았다. 이 보수적 날짜 때문에 14일 간격 학습 origin에서 해당 축제 플래그가 한 번도 활성화되지 않을 수 있다. 학습에서 상수인 신경망 입력은 추론에서도 중립화하고, 이 피처의 성능 향상을 가정하지 않는다.

규정 원문 로컬 경로: `/Users/hongtaeyang/Desktop/SINCON/reports/sources/rules_user_provided.txt`.

## 2. 저장된 구간·역별 진단과 로그 제곱오차 기여

대상은 **Fold 4 ensemble_v2**, origin 2026-02-27, 검증 2026-02-28~06-30, {n:,}행이다. RMSLE **{np.sqrt(total/n):.9f}**, 로그 MSE **{total/n:.9f}**, 로그 SSE **{total:.6f}**.

각 그룹의 `SSE 비율 = 그룹 Σ(log1p(y)−예측로그)² / 전체 Σ(...)²`이다. 그룹 RMSLE만 보면 표본 수가 작은 그룹의 영향을 과대평가할 수 있으므로 SSE 비율도 함께 제시한다. 표의 로그 MSE 기여는 그룹 SSE를 전체 행 수로 나눈 값이다.

### 오차 기여 상위 15개 역

{table(stations.head(15)[['station_id','station_name','canonical_line','rows','rmsle','mbe','sse_pct','global_log_mse_contribution']])}
상위 15개 역의 전체 로그 SSE 비중은 **{stations.head(15).sse_pct.sum():.4f}%**다. 이는 현재 모델의 실제 OOF 잔차이며, 원시 데이터의 단순 프로필 잔차를 분석한 다른 수치와 혼합하지 않는다.

### 시간대별 진단

{table(hours[['hour','rows','rmsle','mbe','sse_pct']])}
### Horizon 구간별 진단

{table(steps[['horizon_bin','rows','rmsle','mbe','sse_pct']])}
### 월별 진단

{table(months[['month','rows','rmsle','mbe','sse_pct']])}
### 기존 도메인 세그먼트

{table(segments[['segment','rows','rmsle','mbe','sse_pct']])}
도메인 세그먼트는 서로 겹치므로 SSE 비중을 합산하면 안 된다. 실제 `y` 기준 저수요 그룹은 사후 진단 전용이며 추론 시 게이트나 입력 피처로 사용할 수 없다. MBE 부호는 예측−실제다.

원본 진단 파일은 `outputs/ensemble_v2/fold4_domain_segments.csv`, `per_station_metrics.csv`, `fold_4/ensemble/per_step.csv`다. 원본은 보존하고 SSE 기여가 추가된 5개 CSV를 이 문서와 같은 폴더에 별도로 생성했다.

## 3. 데이터의 정확한 컬럼 및 Local CV 기준

{table(pd.DataFrame(schema))}
학습 기간은 2025-01-01~2026-06-30, 시험 기간은 2026-07-01~10-31이다. 71개 역, 하루 18시간 구간(시작 hour 6~23), 예측 길이 123일이다. 타겟은 `boardings`; 미래 `alightings`는 입력으로 사용할 수 없고 학습 시 보조 타겟으로만 사용할 수 있다.

신설역 3107·3108·3109는 2025-06-28부터 기록이 있다. 그 이전 행은 0 수요가 아니라 관측 없음이다. 제공된 실제 sample_submission은 **117,576행인 오래된 템플릿**이므로 최종 파일은 전체 test의 **157,194개 example_id**와 순서를 기준으로 생성했다. 메타데이터 first_date/last_date는 관측 범위이며 공식 개통·폐역일이 아니다.

### 고정 검증 분할과 전년값 가용률

{table(pd.DataFrame(folds))}
4개 폴드 합산 OOF는 628,776개 **예측 사례**다. 폴드 간 동일 날짜의 관측이 중복되므로 고유 관측 628,776개가 아니다. Fold 3+4는 314,388개 예측 사례다. Fold 3+4의 year364 가용률은 100%가 아니라 약 95.8~96.0%이며, 테스트에서 100%다. 또한 검증은 겨울·봄 중심, 테스트는 여름·가을이므로 완전히 같은 분포라고 단정하지 않는다.

### 동일 기준에서 비교한 실제 점수

{table(pd.DataFrame(score)[['model','oof_rmsle','weighted_rmsle','fold34_rmsle','fold4_rmsle']])}
공식 RMSLE는 `sqrt(mean((log1p(pred)−log1p(y))²))`다. `weighted_rmsle`는 폴드별 **로그 MSE**에 0.15/0.15/0.35/0.35를 곱해 합산한 후 제곱근을 취한다. 폴드별 RMSLE를 산술 평균한 값이 아니다.

사용자가 Gemini 대화에서 제시한 **0.13은 개선 목표**이며, 아직 어떤 집계 기준인지 확정된 것은 아니다. 현재 v3의 주력 선택 기준은 요청대로 **Fold 3+4 합산 RMSLE**로 두고 전체 4-Fold 가중 점수와 Fold 4를 함께 보고한다. 따라서 잠정적으로 0.13 목표도 Fold 3+4 기준으로 해석하되, 기존 점수와 비교하려고 검증 행을 제거하거나 쉬운 기간만 고르면 안 된다. 새로운 검증기를 추가할 수는 있지만 기존 4개 폴드 점수와 병기해야 한다.

### 저장 OOF를 이용한 추가 확인

v2 예측은 그대로 두고 Fold 3+4에서 가중치만 다시 적합한 실제 결과:

- CatBoost **{reblend['weights']['catboost']:.9f}**, TabM **{reblend['weights']['tabm']:.9f}**, LightGBM/XGBoost **0**.
- Fold 3+4 RMSLE **{reblend['ensemble']['fold34_rmsle']:.9f}**. 기존 동일 구간 0.217364655 대비 개선하지만, 전체 4-Fold 가중 점수는 **{reblend['ensemble']['weighted_rmsle']:.9f}**로 기존 0.220618277보다 나빠진다.
- 이것은 가중치 적합에 사용한 OOF의 점수이며 새로운 독립 검증 점수가 아니다. Gemini가 예상한 0.216944와 구분한다.

{table(pd.DataFrame(reblend['recent_cross_checks'])[['fit_fold','test_fold','purged_overlapping_target_dates','fit_rows','ensemble_rmsle','training_selected_single','single_rmsle','chronological']])}
겹친 target 날짜를 제외한 Fold 3→4 가중치 검증에서 앙상블은 0.227717로, Fold 4 CatBoost 단일 0.226477보다 나쁘다. 따라서 최근 구간 가중치가 미래에도 항상 우월하다고 볼 수 없다. Fold 4→3은 역방향 민감도 점검이며 미래 예측 검증으로 부르지 않는다. 모든 경우 base model의 early stopping 및 피처 설계에 CV를 사용한 한계는 남아 있다.

## 4. 확인된 Public 점수 및 평가 날짜 구성

2026-11-12 사용자가 **ensemble_v2 제출 직후** 제공한 최신 Public 리더보드 캡처로 갱신했다. 아래는 캡처 시점의 순위이며 이후 변동은 반영하지 않는다.

| 캡처 당시 순위 | 팀 | Public RMSLE | 제출 횟수 |
|---|---|---:|---:|
| 1 | 칠전팔기 | 0.15850 | 9 |
| 2 | 전전컴 | 0.16102 | 5 |
| 3 | 요모조모 | 0.16791 | 3 |
| 4 | 핫식스 | 0.16910 | 4 |
| 5 | 컴온인지 | 0.17207 | 9 |
| 6 | 이산구조 | 0.17697 | 7 |
| 7 | HongTaeYang | 0.18265 | 2 |
| 8 | TeamDongTak | 0.19049 | 5 |
| 9 | 오이데 | 0.24186 | 6 |
| 10 | 삼각함수 | 0.25375 | 1 |
| 11 | Gohayoung | 0.28264 | 1 |
| 12 | 쉽죠 ㅋ | 0.28264 | 1 |
| 13 | 햄냥토 | 0.28264 | 1 |

우리 팀은 최신 캡처 기준 **7위, Public 0.18265, 제출 2회**다. 사용자가 제출한 파일은 `/Users/hongtaeyang/Desktop/SINCON/outputs/ensemble_v2/submission_ensemble_v2.csv`이며, 화면에도 직전 최고 0.21751에서 최신 제출 0.18265로 개선되었다고 명시되어 있다.

| 제출 모델/파일 | Local 4-Fold 가중 RMSLE | Local Fold 3+4 RMSLE | Local Fold 4 RMSLE | 확인된 Public RMSLE |
|---|---:|---:|---:|---:|
| 기존 LightGBM / submission_lgbm_0.247808.csv | 해당 검증 체계 아님 | 해당 검증 체계 아님 | 0.247808 | 0.21751 |
| ensemble_v2 / submission_ensemble_v2.csv | 0.220618277 | 0.217364655 | 0.224398 | **0.18265** |
{v3row}

Public 개선 폭은 **−0.03486 (약 16.03% 감소)**다. 1위 0.15850과의 차이는 **0.02415**다. v2의 CV 개선 방향이 이번 Public에서도 확인되었지만, 이 결과는 v3의 개선을 보장하지 않는다. 이번 제출은 기존 v2 최종 제출 파일이며, Fold 3+4로 가중치만 다시 적합한 별도 비교 실험(0.216882)의 Public 점수로 표기하면 안 된다. 0.13대 Public 점수는 이 캡처에서 확인되지 않는다.

캡처 출처: `codex-clipboard-10d6512d-ea51-40f6-8e9e-1e11d4626f49.png`. `Last` 상대 시간으로 절대 제출 시각을 추정하지 않는다. 제출 파일의 SHA-256과 점수 이력은 상위 폴더 `public_submission_history.json`에 기록했다.

공식 Rules에 따르면 Public은 **7·8·9·10월 각 달에서 선택한 날짜 절반, 총 61일·77,958행**, Private은 나머지 62일·79,236행이다. 하루의 모든 역·시간은 같은 쪽에 배정된다. 따라서 Public이 7월 단기 horizon에 편중되어 있다는 전제를 두면 안 된다. 어떤 날짜가 Public인지는 비공개이며 이를 역추정하는 작업도 하지 않는다.

Local CV와 Public은 기간·계절·사건 분포가 달라 모델 순위가 뒤집힐 수 있다. 기존 LightGBM과 v2의 Local–Public 차이가 서로 다르므로 새 모델 Public 점수에 일정한 보정값을 빼서 예측할 수 없다. 규정상 Public을 모델 선택에 참고할 수 있지만 Public 점수에 맞춰 예측값을 반복 보정하거나 정답을 추정하는 행위는 금지다.

---

### Gemini에게 함께 전달할 해석상 주의점

- 0.13 달성이나 “물리적 하한 0.14~0.15”는 현재 자료로 증명되지 않았다. 0.220618에서 0.13으로 가려면 로그 MSE를 약 {100*(1-.13**2/report['ensemble']['weighted_rmsle']**2):.2f}% 줄여야 하지만, 가능 여부는 추가 실험으로 판단해야 한다.
- 학습 반복 상한 도달은 더 학습할 여지가 있다는 신호이지, 반복을 늘리면 반드시 일반화가 좋아진다는 증거가 아니다.
- 최솟값 3 미만이나 평균 315~345는 진단용 가설이다. 그 수치에 강제로 맞추는 후처리는 하지 않는다.
- 신설 3역×개통 첫 3일×18시간은 **162행**이다. 별도 9개 이상 관측과 합하면 171행 후보이며, 54+9=63행이라는 이전 대화의 계산은 틀렸다. 실제 제외 여부는 원자료와 인과적 시점 조건을 검증한다.
- 2026-07-17 제헌절은 기존 holiday_calendar에 이미 반영되어 있다. “누락됐으므로 추가해야 한다”는 전제부터 확인해야 한다.
- v3는 앞서 합의한 4개 패치와 검증된 도메인 정보를 적용하는 비교 실험이다. 이번 첨부 대화의 새 6단계 재설계 제안을 이미 구현·검증한 것으로 표현하지 않는다. v3 최종 실행 결과와 제출 후보 검수는 상위 폴더 `RESULTS.md`를 확인한다.

작성 근거: 보존된 v2 `RESULTS.md`, `IMPLEMENTATION.md`, `ensemble_report.json`, `oof_predictions.csv.gz`, `cv_metrics.csv`, 폴드별 `feature_audit.json`, 원본 4개 CSV, 사용자 제공 Rules 및 리더보드 캡처. 숫자 재계산 코드는 `scripts/build_gemini_handoff.py`, 가중치 비교는 `review_v2/reblend_report.json`에 있다.
'''
path=OUT/'GEMINI_HANDOFF_4_ITEMS.md';path.write_text(text)
print(path);print('lines',len(text.splitlines()),'fold4_rows',n,'station_sse_pct_sum',stations.sse_pct.sum())
