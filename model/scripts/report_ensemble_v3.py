"""Summarize completed v3 artifacts without changing models or submissions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/ensemble_v3'
V2=ROOT/'outputs/ensemble_v2'

def read(name):return json.loads((OUT/name).read_text())
def table(df):
    def v(x):return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(v(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

report=read('ensemble_report.json');cfg=read('config_resolved.json');validation=read('submission_validation.json')
metrics=pd.read_csv(OUT/'cv_metrics.csv')
oof=pd.read_csv(OUT/'oof_predictions.csv.gz',parse_dates=['date'])
folds=[]
for f in range(1,5):
    audit=read(f'fold_{f}/feature_audit.json')
    row={'fold':f,'origin':audit['origin'],'train_rows':audit['rows'],'valid_rows':audit['valid_rows'],'origins':len(audit['origins']),'features':len(audit['features']),'train_year364_pct':100*audit['train_year364_available'],'valid_year364_pct':100*audit['valid_year364_available']}
    folds.append(row)
rows=[]
for name in cfg['models']+['ensemble']:
    m=report['ensemble'] if name=='ensemble' else report['models'][name]
    rows.append({'model':name,**{k:m[k] for k in ['oof_rmsle','weighted_rmsle','fold34_rmsle']},'fold4_rmsle':metrics.loc[(metrics.model==name)&(metrics.fold==4),'rmsle'].iloc[0],'weight':report['weights'].get(name,1.)})
comparison=pd.DataFrame(rows);comparison.to_csv(OUT/'score_summary.csv',index=False)
station_rows=[]
for version,folder in [('v2',V2),('v3',OUT)]:
    file=folder/'oof_predictions.csv.gz'
    if not file.exists():continue
    data=oof if version=='v3' else pd.read_csv(file)
    for subset,mask in [('all',np.ones(len(data),bool)),('fold34',data.fold>=3),('fold4',data.fold==4)]:
        sub=data.loc[mask].copy();sub['sq']=(sub.true_log-sub.ensemble)**2
        for sid,g in sub.groupby('station_id'):
            station_rows.append({'version':version,'subset':subset,'station_id':sid,'rows':len(g),'rmsle':np.sqrt(g.sq.mean()),'log_sse':g.sq.sum(),'sse_pct':100*g.sq.sum()/sub.sq.sum()})
stations=pd.DataFrame(station_rows);stations.to_csv(OUT/'station_comparison.csv',index=False)
focused=stations.loc[stations.station_id.isin([3127,3139,3122,3136])].pivot(index=['subset','station_id'],columns='version',values='rmsle').reset_index()
if 'v2' in focused:focused['delta_v3_minus_v2']=focused.v3-focused.v2
focused.to_csv(OUT/'domain_station_comparison.csv',index=False)
base=[]
model_changes=[]
if (V2/'ensemble_report.json').exists():
    old_report=json.loads((V2/'ensemble_report.json').read_text())
    old=old_report['ensemble']
    for name in cfg['models']:
        a=old_report['models'][name]['fold34_rmsle'];b=report['models'][name]['fold34_rmsle']
        model_changes.append({'model':name,'v2_fold34_rmsle':a,'v3_fold34_rmsle':b,'delta_v3_minus_v2':b-a})
    for key in ['oof_rmsle','weighted_rmsle','fold34_rmsle']:
        base.append({'metric':key,'v2':old[key],'v3':report['ensemble'][key],'delta_v3_minus_v2':report['ensemble'][key]-old[key]})
    old4=pd.read_csv(V2/'cv_metrics.csv');old4=float(old4.loc[(old4.fold==4)&(old4.model=='ensemble'),'rmsle'].iloc[0])
    new4=float(metrics.loc[(metrics.fold==4)&(metrics.model=='ensemble'),'rmsle'].iloc[0]);base.append({'metric':'fold4_rmsle','v2':old4,'v3':new4,'delta_v3_minus_v2':new4-old4})
pd.DataFrame(base).to_csv(OUT/'v2_v3_comparison.csv',index=False)
pd.DataFrame(model_changes).to_csv(OUT/'model_version_comparison.csv',index=False)
iterations=[]
for name in cfg['models']:
    row={'model':name}
    for f in range(1,5):row[f'fold{f}_best']=read(f'fold_{f}/{name}/result.json')['best_iteration']
    row['final_iterations']=read('final_iteration_plan.json')[name]['final_iterations'];iterations.append(row)
seed_rows=[]
for name in cfg['models']:
    p=[np.load(OUT/f'final/{name}/{seed}/prediction_log.npy') for seed in cfg['final_seeds']]
    seed_rows.append({'model':name,'log_rmse_between_seeds':np.sqrt(np.mean((p[0]-p[1])**2)),'correlation':np.corrcoef(p[0],p[1])[0,1]})
pd.DataFrame(seed_rows).to_csv(OUT/'seed_prediction_stability.csv',index=False)
old_recent=next((r['v2'] for r in base if r['metric']=='fold34_rmsle'),None)
recent_delta=report['ensemble']['fold34_rmsle']-old_recent if old_recent is not None else None
decision=('v3는 주력 Fold 3+4 지표에서 v2보다 개선되지 않았다. v2 Public 0.18265 제출을 보존하고, v3는 추가 비교 후보로 취급한다.' if recent_delta is not None and recent_delta>=0 else '주력 Fold 3+4 지표의 개선 여부와 별도 교차 점검을 함께 보고 제출 후보를 선택한다.')
checks=[r for r in report['recent_cross_checks'] if r['chronological'] and r['purged_overlapping_target_dates']]
check=checks[0] if checks else None
inference_check=read('inference_reproducibility.json') if (OUT/'inference_reproducibility.json').exists() else {'status':'not yet recorded'}
preservation_check=read('preservation_check.json') if (OUT/'preservation_check.json').exists() else {'status':'not yet recorded'}
run_status=read('run_exit.json')
md=f'''# ensemble_v3 전체 실행 결과

KBO 편성표를 제외하고, 요청한 4개 패치와 허용된 도메인 정보를 적용한 결과다. 기존 v1/v2 산출물은 보존했다. 아래 점수는 실제 저장된 OOF에서 계산했다. v3 Public 점수는 아직 없다. 기존 v2 제출은 사용자 확인 기준 Public **0.18265**다.

## 1. 비교 기준과 결과

{table(comparison)}
모든 모델은 동일한 검증 행으로 비교한다. 전체 OOF는 628,776개 예측 사례이며, 동일 관측이 서로 다른 origin에 중복 등장한다. 주력 선택 지표는 Fold 3+4의 314,388개 사례를 합쳐 계산한 RMSLE다. 전체 가중 점수는 폴드별 로그 MSE에 0.15/0.15/0.35/0.35를 곱한 뒤 제곱근을 취한다.

{table(pd.DataFrame(base)) if base else 'v2 결과 파일이 없어 이 환경에서 비교표를 만들지 않았다.'}
`delta_v3_minus_v2`는 음수일 때 개선이다. v1 Fold 4 RMSLE 0.247808과의 비교는 `cv_metrics.csv`에 있다. 가중치 적합에 사용한 OOF 점수는 독립적인 일반화 성능 추정치가 아니다.

**현재 판단:** {decision} 가중 OOF의 미세한 개선만으로 v3의 Public 우위를 주장하지 않는다. 이미 확인된 v2 Public 점수에서 일정한 값을 빼서 v3 점수를 예측하지 않는다.

{table(pd.DataFrame(model_changes)) if model_changes else ''}
세 모델의 단일 점수가 낮아져도 결합 오차가 반드시 낮아지는 것은 아니다. 이번 비교에서는 v2에서 큰 비중을 차지했던 CatBoost의 최근 구간 점수가 악화됐으며, v3 최적 혼합에서 기여가 사실상 0이 됐다. CPU/MPS에서 학습한 v2와 CUDA에서 학습한 v3는 실행 장치와 PyTorch 버전도 다르다. 따라서 이 차이를 피처 변경만의 효과나 GPU 자체의 효과로 분리해 해석할 수 없다.

{'날짜 중복을 제거한 Fold 3→4 점검에서 앙상블은 '+format(check['ensemble_rmsle'],'.6f')+', 선택된 단일 '+check['training_selected_single']+'은 '+format(check['single_rmsle'],'.6f')+'다. 이 점검에서 단일 모델이 더 낮다면 단일 후보를 함께 유지하고, 혼합 모델의 우월성을 단정하지 않는다.' if check else ''}

## 2. 검증과 앙상블 선택

{table(pd.DataFrame(folds))}
최종 가중치: `{json.dumps(report['weights'],ensure_ascii=False)}`.
비교용 전체 4-Fold 가중치: `{json.dumps(report['all_fold_comparison']['weights'],ensure_ascii=False)}`.

{table(pd.DataFrame(report['recent_cross_checks']).drop(columns='weights'))}
Fold 3→4는 시간순 민감도 점검이며, Fold 4→3은 역방향 점검이다. 날짜 중복 제거는 가중치 적합 행에서만 수행했다. base model early stopping과 피처 설계에는 여전히 CV 결과가 사용되므로 완전히 독립된 holdout으로 주장하지 않는다. Fold 3/4는 겨울·봄, 테스트는 여름·가을이며 year364 가용률도 완전히 같지 않다.

잔차 상관 행렬은 `residual_correlation.csv`와 `ensemble_report.json`에 있다. 상관계수 목표를 맞추기 위해 모델을 강제로 넣거나, 모델별 가장 쉬운 fold의 가중치를 조합하지 않는다. SLSQP가 0을 부여한 모델은 최종 혼합에서 기여하지 않는다. 절편·MBE 상수 보정은 없다.

## 3. 최적 반복 수와 최종 학습

{table(pd.DataFrame(iterations))}
최종 반복 수는 round((0.4 × Fold3 best + 0.6 × Fold4 best) × 1.12), 최소 1이다. 각 모델을 시드 2026/2027로 전체 데이터에서 재학습하고 두 로그 예측을 평균한다. 최종 재학습에는 검증 구간이 없으므로 이 고정 횟수로 실행한다. 부스팅 횟수는 신경망의 epoch와 다르며 트리는 AdamW를 사용하지 않는다. 비선형 모델의 전역 최적점을 찾았다고 보장하지 않는다.

{table(pd.DataFrame(seed_rows))}

## 4. 보조 진단과 도메인 역

{table(metrics.loc[metrics.fold==4,[c for c in ['model','rmsle','mae','rmse','mbe','rmse_mae_ratio','mase_macro','mase_row_weighted','acf_1d_macro','acf_7d_macro','acf_28d_macro'] if c in metrics]])}
전체 지표 이름과 값은 `cv_metrics.csv`, 모델별 `diagnostics/summary.json`, `per_entity.csv`, `per_step.csv`, `per_month.csv`, `residual_acf.csv`를 확인한다. MASE는 과거 7일 계절 복사 오차를 분모로 사용한다. 원시 스케일 MBE를 0으로 강제하면 로그 지표가 악화될 수 있으므로 후처리에 자동 적용하지 않았다.

{table(focused)}
문학경기장 3127, 송도달빛축제공원 3139, 부평삼거리 3122, 인천대입구 3136의 동일 구간 비교다. 전체 71역의 오차 기여는 `station_comparison.csv`에 있다.

## 5. 제출 파일 검수

```json
{json.dumps(validation,ensure_ascii=False,indent=2)}
```
주 제출 후보: `submission_ensemble_v3.csv`. 단일 모델군의 2개 시드 평균과 전체 4-Fold 가중치 혼합은 `candidates/`에 별도로 저장했다. 평균 315~345와 최소 3 미만은 진단 기준이며 이를 만족시키기 위한 강제 보정은 하지 않았다. 소수 예측을 유지하며 음수 로그는 0으로 제한한다. 정수 반올림·상수 편향 보정·영업시간 강제 0 처리는 없다.

## 6. 구현·재현·제약

전체 4-Fold CV 및 최종 8개 모델 재학습은 종료 코드 {run_status['exit_code']}으로 완료됐으며, 소요 시간은 {run_status['elapsed_seconds']:.2f}초(약 {run_status['elapsed_seconds']/60:.2f}분)다. 최종 단위 테스트는 로컬 60개, RunPod 60개 모두 통과했다(`local_unit_tests.log`, `runpod_unit_tests.log`). 267개 원격 산출물의 다운로드 해시 검수와 OOF 점수 독립 재계산도 통과했다(`download_integrity.json`, `artifact_verification.json`).

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
{json.dumps(inference_check,ensure_ascii=False,indent=2)}
```

기존 산출물 보존 검수:

```json
{json.dumps(preservation_check,ensure_ascii=False,indent=2)}
```
'''
(OUT/'RESULTS.md').write_text(md)
print(OUT/'RESULTS.md')
