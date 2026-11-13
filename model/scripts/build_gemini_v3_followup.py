"""Build the three requested follow-up facts from fixed v2/v3 OOF artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path('/Users/hongtaeyang/Desktop/SINCON')
OUT=Path(__file__).resolve().parents[1]/'outputs/ensemble_v3/review_v3'
OUT.mkdir(parents=True,exist_ok=True)
SIDS=[3127,3226,3107]
meta=pd.read_csv(ROOT/'data/raw/metaData.csv').set_index('station_id')
raw=pd.read_csv(ROOT/'data/raw/train.csv',parse_dates=['date'])
v3=ROOT/'outputs/ensemble_v3'
r=json.loads((v3/'ensemble_report.json').read_text())
frames={}
summary=[]
for version in ['v2','v3']:
    f=pd.read_csv(ROOT/f'outputs/ensemble_{version}/oof_predictions.csv.gz',parse_dates=['date'])
    f=f.loc[f.fold==4].copy()
    assert len(f)==157194 and not f.duplicated(['date','station_id','hour']).any()
    f['prediction']=np.expm1(f.ensemble);f['sq']=(f.true_log-f.ensemble)**2
    f['error']=f.prediction-f.boardings;f['month']=f.date.dt.strftime('%Y-%m')
    f['dow']=f.date.dt.dayofweek
    frames[version]=f
    for sid in SIDS:
        g=f.loc[f.station_id==sid]
        summary.append({'version':version,'station_id':sid,'station_name':meta.loc[sid,'station_name'],'rows':len(g),'rmsle':np.sqrt(g.sq.mean()),'mbe':g.error.mean(),'mean_actual':g.boardings.mean(),'mean_pred':g.prediction.mean(),'log_sse':g.sq.sum(),'global_sse_pct':100*g.sq.sum()/f.sq.sum()})
f=frames['v3'];g=f.loc[f.station_id.isin(SIDS)].copy()
g=g.merge(raw[['date','station_id','hour','alightings']],on=['date','station_id','hour'],how='left',validate='one_to_one')
assert not g.alightings.isna().any()
g['station_name']=g.station_id.map(meta.station_name)
station_sse=g.groupby('station_id').sq.sum()
def aggregate(keys):
    a=g.groupby(keys,observed=True).agg(rows=('sq','size'),mean_actual=('boardings','mean'),mean_pred=('prediction','mean'),mean_alightings=('alightings','mean'),mbe=('error','mean'),log_sse=('sq','sum')).reset_index()
    a['rmsle']=np.sqrt(a.log_sse/a.rows)
    a['station_sse_pct']=100*a.log_sse/a.station_id.map(station_sse)
    a['global_sse_pct']=100*a.log_sse/f.sq.sum()
    return a
monthly=aggregate(['station_id','station_name','month'])
hourly=aggregate(['station_id','station_name','hour'])
joint=aggregate(['station_id','station_name','month','hour'])
weekly=aggregate(['station_id','station_name','dow'])
for name,df in [('station_comparison',pd.DataFrame(summary)),('monthly',monthly),('hourly',hourly),('month_hour',joint),('weekday',weekly)]:df.to_csv(OUT/f'{name}.csv',index=False)
# Operational late-March boundary, not an asserted opening/scheduled-game date.
m=f.loc[f.station_id==3127].copy()
conditions={'전체 문학경기장':np.ones(len(m),bool),'03-25~06-30':m.date.ge('2026-03-25'),'화~일요일':m.dow.ge(1),'17~22시':m.hour.between(17,22),'03-25~06-30 & 화~일 & 17~22시':m.date.ge('2026-03-25')&m.dow.ge(1)&m.hour.between(17,22)}
seg=[]
for name,mask in conditions.items():
    t=m.loc[mask]
    seg.append({'segment':name,'rows':len(t),'station_row_pct':100*len(t)/len(m),'rmsle':np.sqrt(t.sq.mean()),'mbe':t.error.mean(),'mean_actual':t.boardings.mean(),'mean_pred':t.prediction.mean(),'station_sse_pct':100*t.sq.sum()/m.sq.sum(),'global_sse_pct':100*t.sq.sum()/f.sq.sum()})
segments=pd.DataFrame(seg);segments.to_csv(OUT/'munhak_focus_segments.csv',index=False)
mg=g.loc[g.station_id==3127]
daily=mg.groupby('date').agg(log_sse=('sq','sum'),rows=('sq','size'))
daily['alight_14_17']=mg.loc[mg.hour.between(14,17)].groupby('date').alightings.sum()
daily['board_20_22']=mg.loc[mg.hour.between(20,22)].groupby('date').boardings.sum()
daily['pred_20_22']=mg.loc[mg.hour.between(20,22)].groupby('date').prediction.sum()
daily['max_board_17_22']=mg.loc[mg.hour.between(17,22)].groupby('date').boardings.max()
daily['evening_surge_observed']=daily.max_board_17_22.ge(1000)
daily.to_csv(OUT/'munhak_daily_observed_profiles.csv')
profiles=daily.groupby('evening_surge_observed').agg(days=('rows','size'),mean_alight_14_17=('alight_14_17','mean'),mean_board_20_22=('board_20_22','mean'),mean_pred_20_22=('pred_20_22','mean'),log_sse=('log_sse','sum'),rows=('rows','sum')).reset_index()
profiles['rmsle']=np.sqrt(profiles.log_sse/profiles.rows)
profiles['station_sse_pct']=100*profiles.log_sse/m.sq.sum()
profiles.to_csv(OUT/'munhak_observed_surge_comparison.csv',index=False)
top=mg.nlargest(20,'sq')[['date','station_id','hour','boardings','prediction','alightings','sq','error']]
top.to_csv(OUT/'munhak_top20_errors.csv',index=False)

def table(d):
    def val(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(val(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'

metrics=pd.read_csv(v3/'cv_metrics.csv')
score=pd.read_csv(v3/'score_summary.csv')
iterations=[]
for name in ['lightgbm','catboost','xgboost','tabm']:
    vals=[json.loads((v3/f'fold_{k}/{name}/result.json').read_text())['best_iteration'] for k in range(1,5)]
    iterations.append({'model':name,**{f'fold{k+1}':v for k,v in enumerate(vals)},'final':json.loads((v3/'final_iteration_plan.json').read_text())[name]['final_iterations']})
blocks=[]
for sid in SIDS:
    blocks.append(f"### {sid} {meta.loc[sid,'station_name']}\n\n월별:\n\n"+table(monthly.loc[monthly.station_id==sid,['month','rows','rmsle','mbe','mean_actual','mean_pred','station_sse_pct','global_sse_pct']])+"\n시간대별:\n\n"+table(hourly.loc[hourly.station_id==sid,['hour','rows','rmsle','mbe','mean_actual','mean_pred','station_sse_pct']])+"\n월×시간 오차 기여 상위 8개:\n\n"+table(joint.loc[joint.station_id==sid].nlargest(8,'log_sse')[['month','hour','rows','rmsle','mbe','station_sse_pct']]))
focus=segments.iloc[-1]
three_share=sum(row['global_sse_pct'] for row in summary if row['version']=='v3')
perfect_three=np.sqrt(f.loc[~f.station_id.isin(SIDS),'sq'].sum()/len(f))
run=json.loads((v3/'run_exit.json').read_text())
text=f'''# Gemini 전달용 — v3 변경 내역·3역 오차 분포·학습 시간

작성일: 2026-11-12, 한국 시간. 이미지의 추가 질문 3개에 대한 실제 산출물 기반 답변이다. 새 모델을 추가 학습하거나 기존 검증기를 변경하지 않았다. 모든 오차는 저장된 OOF를 재계산했다.

## 1. ensemble_v3에 들어간 변경과 실제 세부 점수

### 요청받은 4가지 패치 — 모두 적용

| 패치 | 실제 구현 |
|---|---|
| 앙상블 최적화 범위 | Fold 3+4 합산 314,388행으로 비음수·합계1 SLSQP 로그 결합. 전체 4-Fold 가중치는 비교용으로 별도 저장 |
| 신경망 year364 결측 | 로그 레벨의 NaN을 해당 행의 승차/하차 anchor로 먼저 대체하고 원래 결측 마스크 유지 |
| 트리 교차 범주 규제 | LightGBM·XGBoost에서 station_hour/station_eff_dow 제거. CatBoost는 유지. LGB cat_smooth=50, cat_l2=20, extra_trees=True; XGB depth=6, min_child_weight=80 |
| 15일 계절 변화량 | d−364±7일 정제 로그 평균과 전년 origin 직전 56일 평균의 차이. date<=origin 적용, 결측·노선 fallback 플래그 제공 |

추가 적용: 통근 승하차 비대칭, 외부 환승역, 부평삼거리 명절 성묘 시간, 역별 학사 계절 proxy, 공개일 검증된 펜타포트 일정. 기존 162개 개통 첫 3일 관측에 9개 급감 키를 더해 정제/학습 후보에서 제외하되 검증 정답은 유지했다. 학습에서 상수인 신경망 입력은 추론에서도 중립화했다.

**KBO 원편성표와 맥주축제 일정은 미사용**이다. 사용자에게 별도 허가가 없고 사용 가능한 출처만 반영하라는 답을 받았다. 펜타포트만 인천광역시 공공누리 제1유형 출처의 공개일로 필터링했다. 자세한 근거는 `../OFFICIAL_SOURCES.md`. 미래 실측 날씨·실제 경기 결과·취소·관중·종료 시각을 사용하지 않았다.

**유지한 구조:** anchor=0.6×56일+0.4×28일 정제 로그 평균, 부분 horizon 표본 추출(폴드당80만/최종100만), 4개 고정 CV, 잔차 타겟, 동일 4개 모델군. 피처 수는 100개다.

**아직 적용하지 않은 제안:** ±21일 전년 커널 anchor와 성장률 수축, Fourier/Ridge 계절 기준선, 전년값 가용 표본85% 추출, 일일 총수요×시간 점유율 분리 모델, 저수요 전용 모델, 최대8,000회 학습, 시공간 Transformer. 이 제안들의 유효성을 이번 결과로 판단할 수 없다.

### 동일 구간 모델별 결과

{table(score)}
{table(pd.read_csv(v3/'v2_v3_comparison.csv'))}
{table(pd.read_csv(v3/'model_version_comparison.csv'))}
세 모델이 개선돼도 전체 결합이 크게 개선된 것은 아니다. 특히 v2에서 강했던 CatBoost가 v3 최근 구간에서 악화됐다. CPU/MPS→CUDA 및 PyTorch 버전 차이도 있으므로 피처 변경만의 효과를 분리한 실험이 아니다. **4개 패치를 한꺼번에 적용한 결과가 정체됐다는 이유로 각각을 “효과 없음”으로 확정·제외하면 안 된다.** 개별 제거/추가 실험이 필요하다.

{table(pd.DataFrame(iterations))}
Fold 4 XGBoost는 최대3,000회에 도달했다. 초기 Fold 1~2의 짧은 최적 반복 수 및 학습/검증 전년값 가용률 차이는 여전히 남는다. 연도 경계의 2024년 참조 구간 결측을 15일 피처 하나로 해결한 것은 아니다.

최종 혼합: LGB {r['weights']['lightgbm']:.6f}, CatBoost 사실상0, XGBoost {r['weights']['xgboost']:.6f}, TabM {r['weights']['tabm']:.6f}. Fold 3에서 가중치를 적합하고 겹친 target 날짜를 제외해 Fold 4를 점검하면 혼합0.225605, TabM 단일0.224762다. 가중치 적합 점수와 독립 일반화 성능을 혼동하지 않는다.

기존 확인된 Public: v1 0.21751 → v2 **0.18265**. v3는 아직 Public 결과가 없다. Local–Public 차이를 고정 보정값으로 쓰지 않는다.

## 2. 문학경기장·인천대공원·검단호수공원의 실제 OOF 오차

검증 기준: Fold 4, origin=2026-02-27, 2026-02-28~06-30, 전체157,194행. 각 역2,214행(123일×18시간). 아래 상세 표는 **v3 앙상블** 기준이며 v2 역별 총계도 병기한다. MBE=예측−실제. station_sse_pct는 해당 역의 로그 SSE 비중, global_sse_pct는 전체71역의 로그 SSE 비중이다. RMSLE가 큰 셀과 총오차 기여가 큰 셀을 구별한다. 2월은 28일 하루뿐이므로 다른 월과 표본 수가 다르다.

{table(pd.DataFrame(summary))}

세 역 합산 비중은 v3 전체 로그 SSE의 **{three_share:.4f}%**다. 이 세 역을 오차0으로 완벽히 맞추고 나머지 예측을 그대로 둔다고 가정해도 Fold 4 RMSLE는 **{perfect_three:.6f}**다. 이는 개선 여력 계산용 가정이며 모델 성능이나 물리적 하한이 아니다. 목표0.13에 도달하려면 다른 역의 공통 계절·성장·시간대 오차도 줄여야 한다.

인천대공원은 4월이 이 역 SSE의52.87%, 5월이22.31%를 차지한다. 검단호수공원은 5·6월 합산65.39%에 집중되며, 시간대별로는06시14.53%,07시11.59%가 크다. 원인을 꽃놀이·신설역 성장으로 확정한 결과가 아니라 실제 오차가 집중된 구간을 확인한 것이다.

### 문학경기장: 질문한 시간·요일·월 조건의 확인

{table(segments)}
“3월 말”은 진단상 **3월25일 이후**로 정의했다. 실제 개막일·경기일을 입력한 것이 아니다. 마지막 교집합은 문학경기장 행의 **{focus.station_row_pct:.2f}%**이면서 해당 역 로그 SSE의 **{focus.station_sse_pct:.2f}%**, 전체71역 로그 SSE의 **{focus.global_sse_pct:.2f}%**를 차지한다. 개별 행이 아닌 여러 조건의 표는 서로 겹치므로 비중을 합산하면 안 된다.

### 승차 급증과 앞선 시간대 하차량의 관측 패턴

아래는 17~22시 중 한 시간이라도 실제 승차1,000명 이상이면 “저녁 급증 관측일”로 묶은 **사후 진단**이다. 학습 피처·실제 경기일 라벨로 사용하지 않는다. alight_14_17과 board_20_22는 각 날짜의 해당 시간대 합계다.

{table(profiles)}
이 결과로 시간대별 승하차 동반 패턴을 확인할 수 있지만, 해당 날짜의 실제 야구 개최·종료 시각이나 인과관계까지 확인한 것은 아니다. 검증 구간 하차 실측치는 이 보고서의 사후 설명에만 썼으며 예측 생성에는 넣지 않았다. 실제 타겟 기반으로 급증일을 고른 만큼 이를 이용한 예측 성능 수치로도 해석하지 않는다.

문학경기장 로그 제곱오차 상위10행:

{table(top.head(10))}

{chr(10).join(blocks)}

전체 월×시간 표와 요일별 표, 날짜별 승하차 프로필은 이 문서와 같은 폴더의 CSV에 있다. 모든 그룹의 로그 SSE 합이 원래 OOF SSE와 일치하는지 생성 코드에서 확인한다.

## 3. 현재 자원과 전체 실행 시간 — 허용 예산과 실측을 구분

- 실제 최근 실행 환경: **RunPod RTX PRO 4500 Blackwell 약32GB 1장**, CUDA. 호스트48CPU스레드, 모델당8스레드. LightGBM CPU 큐와 GPU 학습 큐를 병행했다. “맥북 MPS+CPU4스레드에서만 학습”이라는 전제는 현재 실행과 다르다.
- 전체 4-Fold 피처 생성·16개 CV 모델·앙상블 계산·100만 행 최종8개 모델 재학습·제출 생성: **{run['elapsed_seconds']:.2f}초, 약14분16초**. 기존 패키지 설치·환경 준비·결과 다운로드 시간을 포함하지 않는다.
- 저장된8개 모델 재로딩 추론: **22.58초**, 원래 제출 CSV와 SHA-256까지 동일.
- 로컬/RunPod 최종 단위 테스트 각각60개 통과. 기존476개 파일 보존, OOF628,776행 및 test157,194행 검수 완료.
- **사용자가 허용한 전체 파이프라인 1회 최대 학습 시간: 2시간 이내(120분, 7,200초).** 2026-11-12 사용자 답변으로 확정했다. 모델 하나당이 아니라 전체 CV와 최종 재학습을 포함한 1회 실행의 상한으로 적용한다. 위14분16초는 이번 v3의 실측이며 새 분해 모델·8,000회 학습의 시간 보장이 아니다.
- 새 모델의 실행 시간은 먼저 작은 실험으로 측정해야 한다. 목표0.13의 가능성이나 달성 시간도 현재 결과에서 확정할 수 없다.

전달용 요약: “v3 4패치 및 일부 도메인 피처 적용 완료. 최근 OOF0.217637로 v2보다 개선되지 않았고 패치별 원인 분리 실험은 아직 없음. 첨부한3역 월·시간별 실제 SSE를 기준으로 다음 설계를 제안해 달라. GPU 환경의 기존 전체 실행은14분16초이며, 사용자가 확정한 전체 파이프라인 1회 최대 학습 시간은2시간 이내다. 이 예산 안에서 모델 수와 반복 수를 배분해 달라.”

근거: `../RESULTS.md`, `../IMPLEMENTATION.md`, `../ensemble_report.json`, `../oof_predictions.csv.gz`, v2 동일 파일, 원본 train.csv의 2026-06-30 이전 관측. 생성 코드: `scripts/build_gemini_v3_followup.py`.
'''
# Disjoint group aggregation integrity, no validation-row removal.
for df in [monthly,hourly,joint,weekly]:
    assert len(g)==df.rows.sum()
    assert np.isclose(df.log_sse.sum(),g.sq.sum(),rtol=1e-12)
(OUT/'GEMINI_V3_FOLLOWUP_3_ITEMS.md').write_text(text)
print(OUT/'GEMINI_V3_FOLLOWUP_3_ITEMS.md')
print(pd.DataFrame(summary)[['version','station_id','rmsle','global_sse_pct']].to_string(index=False))
print(segments.iloc[-1].to_dict())
print(profiles.to_string(index=False))
