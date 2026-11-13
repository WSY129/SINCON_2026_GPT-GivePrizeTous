"""Explain a frozen three-tree blend versus its primary model; no model writes."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ensemble import fit_ensemble,predict_ensemble
from src.pipeline import save_json

OUT=Path(__file__).resolve().parent
RUN=ROOT/'outputs/three_tree'
NAMES=['lightgbm_large','xgboost','catboost']

def markdown(frame):
    cols=list(frame.columns)
    text='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
    for row in frame.itertuples(index=False,name=None):
        text+='| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |\n'
    return text

def main():
    plan=json.loads((RUN/'ensemble_plan.json').read_text())
    folds=json.loads((RUN/'cv_folds.json').read_text())
    meta=pd.read_csv(ROOT/'data/raw/metaData.csv').set_index('station_id')
    weights=np.array([plan['weights'][n] for n in NAMES])
    frames=[];metrics=[];cohorts=[];correlations=[]
    for info in folds:
        fold=info['fold']
        f=pd.read_csv(RUN/f'fold_{fold}/oof.csv.gz')
        f['fold']=fold
        f['cold_start']=pd.to_datetime(f.station_id.map(meta.first_date))>pd.Timestamp(info['origin'])
        e=np.column_stack([np.log1p(f[n])-np.log1p(f.boardings) for n in NAMES])
        ens=e@weights
        np.testing.assert_allclose(ens,np.log1p(f.ensemble)-np.log1p(f.boardings),rtol=1e-10,atol=1e-12)
        for j,name in enumerate(NAMES+['ensemble']):
            r=e[:,j] if j<3 else ens
            metrics.append({'fold':fold,'model':name,'rmsle':float(np.sqrt(np.mean(r*r))),
                            'mean_log_error':float(np.mean(r)),'log_error_variance':float(np.var(r)),
                            'log_mse':float(np.mean(r*r))})
            for is_cold,mask in [('warm',~f.cold_start),('cold',f.cold_start)]:
                if not mask.any():continue
                cohorts.append({'fold':fold,'cohort':is_cold,'model':name,'rows':int(mask.sum()),
                                'rmsle':float(np.sqrt(np.mean(r[mask]**2))),
                                'mean_log_error':float(np.mean(r[mask])),
                                'share_of_model_log_squared_error':float(np.sum(r[mask]**2)/np.sum(r*r))})
        c=pd.DataFrame(e,columns=NAMES).corr()
        for i in range(3):
            for j in range(i+1,3):correlations.append({'fold':fold,'left':NAMES[i],'right':NAMES[j],'residual_correlation':float(c.iloc[i,j])})
        frames.append(f)
    metrics=pd.DataFrame(metrics);cohorts=pd.DataFrame(cohorts);correlations=pd.DataFrame(correlations)
    metrics.to_csv(OUT/'model_fold_metrics.csv',index=False)
    cohorts.to_csv(OUT/'warm_cold_metrics.csv',index=False)
    correlations.to_csv(OUT/'residual_correlations.csv',index=False)

    f=frames[2]
    e=np.column_stack([np.log1p(f[n])-np.log1p(f.boardings) for n in NAMES])
    ens=e@weights;delta=ens-e[:,0]
    base_mse=float(np.mean(e[:,0]**2));ens_mse=float(np.mean(ens**2))
    individual_mse=np.mean(e*e,axis=0)
    weighted_mse=float(weights@individual_mse)
    derivative=2*np.mean(e[:,[0]]*(e[:,1:]-e[:,[0]]),axis=0)
    decomposition={'primary_log_mse':base_mse,'ensemble_log_mse':ens_mse,
                   'extra_log_mse':ens_mse-base_mse,
                   'rmsle_relative_increase_percent':100*(np.sqrt(ens_mse/base_mse)-1),
                   'log_mse_relative_increase_percent':100*(ens_mse/base_mse-1),
                   'weighted_single_log_mse':weighted_mse,
                   'weighted_single_excess_over_primary':weighted_mse-base_mse,
                   'diversity_credit':weighted_mse-ens_mse,
                   'primary_error_times_adjustment_twice':float(2*np.mean(e[:,0]*delta)),
                   'adjustment_squared_mean':float(np.mean(delta*delta)),
                   'variance_difference':float(np.var(ens)-np.var(e[:,0])),
                   'squared_bias_difference':float(np.mean(ens)**2-np.mean(e[:,0])**2),
                   'audit_directional_derivatives_from_primary':dict(zip(NAMES[1:],map(float,derivative))),
                   'primary_is_audit_simplex_optimum':bool(np.all(derivative>=0)),
                   'audit_optimum_note':'Posthoc geometry diagnosis only; no audit-based weight plan is deployed.'}
    assert abs(decomposition['extra_log_mse']-(decomposition['weighted_single_excess_over_primary']-decomposition['diversity_credit']))<1e-12
    f=f.assign(primary_log_se=e[:,0]**2,ensemble_log_se=ens**2,extra_log_se=ens**2-e[:,0]**2)
    f['month']=f.date.str[:7]
    f['station_name']=f.station_id.map(meta.station_name)
    segments={}
    for group in ['month','hour','station_id']:
        records=[]
        for key,part in f.groupby(group):
            row={group:key,'rows':len(part),'primary_rmsle':float(np.sqrt(part.primary_log_se.mean())),
                 'ensemble_rmsle':float(np.sqrt(part.ensemble_log_se.mean())),
                 'extra_log_se_sum':float(part.extra_log_se.sum())}
            if group=='station_id':row['station_name']=part.station_name.iloc[0]
            records.append(row)
        segment=pd.DataFrame(records)
        segment.to_csv(OUT/f'by_{group}.csv',index=False)
        segments[group]=segment
    daily=f.groupby('date').agg(primary=('primary_log_se','sum'),ensemble=('ensemble_log_se','sum'),rows=('date','size'))
    rng=np.random.default_rng(2026)
    ix=rng.integers(0,len(daily),size=(5000,len(daily)))
    p=daily.primary.to_numpy()[ix].sum(axis=1)
    a=daily.ensemble.to_numpy()[ix].sum(axis=1)
    n=daily.rows.to_numpy()[ix].sum(axis=1)
    differences=np.sqrt(a/n)-np.sqrt(p/n)
    bootstrap={'resamples':5000,'unit':'date (paired same-date samples for both models)',
               'delta_rmsle':float(np.sqrt(ens_mse)-np.sqrt(base_mse)),
               'ci95':list(map(float,np.quantile(differences,[.025,.975]))),
               'note':'Descriptive conditional interval within this previously examined period. Adjacent dates can be dependent; not a future-performance guarantee.'}

    dev=pd.concat(frames[:2],ignore_index=True)
    candidate_metrics=[];candidate_plans={}
    for name,fit_frame in [('current_all_development',dev),('warm_development_only',dev.loc[~dev.cold_start]),('development_fold2_only',frames[1])]:
        fitted,_=fit_ensemble(fit_frame.boardings.to_numpy(),{n:fit_frame[n].to_numpy() for n in NAMES},policy=plan['policy'])
        candidate_plans[name]=fitted
        row={'diagnostic_fit':name,'fit_rows':len(fit_frame),**fitted['weights'],'fit_rmsle':fitted['development_score']}
        pred=predict_ensemble({n:f[n].to_numpy() for n in NAMES},fitted)
        row['reused_fold3_rmsle']=float(np.sqrt(np.mean((np.log1p(pred)-np.log1p(f.boardings))**2)))
        candidate_metrics.append(row)
    candidates=pd.DataFrame(candidate_metrics)
    candidates.to_csv(OUT/'diagnostic_weight_fits_NOT_DEPLOYED.csv',index=False)
    save_json(OUT/'diagnostic_plans_NOT_DEPLOYED.json',candidate_plans)
    summary={'decomposition':decomposition,'paired_date_bootstrap':bootstrap,
             'weight_fit_diagnostics_only':candidate_metrics,'active_model_changed':False,
             'evaluation_context':plan['evaluation_context']}
    save_json(OUT/'summary.json',summary)
    audit=metrics[metrics.fold==3].drop(columns='fold')
    audit_corr=correlations[correlations.fold==3].drop(columns='fold')
    new_rows=cohorts[(cohorts.fold==1)&(cohorts.cohort=='cold')]
    cold_values=new_rows.set_index('model')
    cold_rows=int(cold_values.loc['lightgbm_large','rows'])
    pooled_cold_share={name:float(cold_rows*cold_values.loc[name,'rmsle']**2/sum(
        len(frames[k])*metrics.loc[(metrics.fold==k+1)&(metrics.model==name),'log_mse'].iloc[0] for k in [0,1])) for name in NAMES}
    text=f'''# 3모델 앙상블이 LightGBM Large보다 나쁜 이유

현재 저장된 모델·가중치를 변경하지 않고 OOF를 분석했다. 공식 RMSLE는 로그 오차의 제곱평균 제곱근이다.
LightGBM Large RMSLE **{np.sqrt(base_mse):.10f}**, 앙상블 **{np.sqrt(ens_mse):.10f}**로 **{bootstrap['delta_rmsle']:.10f} / {decomposition['rmsle_relative_increase_percent']:.2f}% 악화**됐다.

## 1. 마지막 검증 구간에서는 보조 모델의 오차가 크고 방향도 비슷하다

{markdown(audit)}

{markdown(audit_corr)}

`mean_log_error < 0`는 로그 공간의 과소예측이다. 원단위 MBE나 모든 행의 일정 비율 과소예측을 뜻하지 않는다.
앙상블 후 로그 오차 분산 변화는 {decomposition['variance_difference']:.9f}, 제곱 편향 변화는 {decomposition['squared_bias_difference']:.9f}다.
모델 이름이나 학습 알고리즘이 다르더라도 동일한 특징·목표·시간 구간을 학습하면 잔차는 유사할 수 있다.
이번 결과에서 보조 모델의 추가 오차를 상쇄할 만큼 오차가 서로 상쇄되지 않았다.

## 2. 다양성으로 줄어드는 오차보다 약한 모델을 섞는 비용이 크다

가중 평균 개별 log-MSE와 앙상블 log-MSE의 차이는 예측 다양성의 이득이다.

- LightGBM 단독 log-MSE: **{base_mse:.9f}**
- 단독 모델 log-MSE 가중 평균: **{weighted_mse:.9f}**
- 다른 모델을 섞어서 늘어난 개별 오차: **+{decomposition['weighted_single_excess_over_primary']:.9f}**
- 서로 다른 예측을 평균해서 얻은 감소분: **−{decomposition['diversity_credit']:.9f}**
- 최종 증가: **+{decomposition['extra_log_mse']:.9f}**, 앙상블 log-MSE **{ens_mse:.9f}**

이는 로그 공간의 제곱오차에 대한 정확한 항등식이다. RMSLE 자체를 선형 분해한 값은 아니다.
LightGBM 잔차를 `e`, 앙상블이 가한 예측 변경을 `d`라 하면 증가분은
`2*mean(e*d) + mean(d²)` = {decomposition['primary_error_times_adjustment_twice']:.9f} + {decomposition['adjustment_squared_mean']:.9f}다.

## 3. 개발 구간에서 유리했던 가중치가 마지막 구간에는 맞지 않는다

현재 가중치는 LightGBM≥50% 제약 아래 개발 1·2의 전체 행에 대해 최적화했다.
개발 1에는 예측 기준일 이후에 처음 관측되는 역 3개가 있다. 실제 테스트에는 관측 이력이 없는 신규 역이 없다.
그 역들의 개발 1 오차는 아래와 같다.

{markdown(new_rows.drop(columns=['fold','cohort']))}

신규 역 {cold_rows:,}행은 개발 1·2 합산의 **{100*cold_rows/len(dev):.2f}%**지만,
LightGBM 개발 로그제곱오차의 **{100*pooled_cold_share['lightgbm_large']:.2f}%**, XGBoost의 **{100*pooled_cold_share['xgboost']:.2f}%**를 차지한다.
그 행들의 평균 로그 오차는 LightGBM **{cold_values.loc['lightgbm_large','mean_log_error']:.5f}**와
XGBoost **{cold_values.loc['xgboost','mean_log_error']:.5f}**로 방향이 반대다.
앙상블에서는 **{cold_values.loc['ensemble','mean_log_error']:.5f}**로 상쇄되므로 XGBoost가 38.91%라는 큰 비중을 얻었다.
마지막 검증은 모든 역에 이력이 있고 세 모델이 모두 과소예측하므로 같은 상쇄가 발생하지 않는다.

가중치 적합 구간만 달리한 원인 진단 결과다. 아래 대안은 현재 제출 파일이나 모델에 적용하지 않았다.

{markdown(candidates)}

개발 1의 큰 오차·신규 역 패턴을 상쇄하는 가중치가 더 성숙한 역의 봄·초여름 예측에 그대로 유효하다는 보장은 없다.
신규 역을 제외하거나 개발 2만 사용하면 XGBoost 비중은 0에 가까워지지만 CatBoost 비중이 커지고, 마지막 검증은 오히려 더 나빠진다.
즉 신규 역 제외만으로 해결되지 않으며 CatBoost의 구간별 성능 변화도 확인해야 한다.
이전 검증을 본 뒤 비교한 대안이므로 위 숫자를 새로운 독립 검증 점수로 해석해서는 안 된다.

## 4. 월별 영향

{markdown(segments['month'])}

`extra_log_se_sum > 0`인 그룹은 LightGBM 단독보다 앙상블이 악화됐다. 행 수가 다르면 합계 기여도와 평균 오차를 구별한다.
역별 가장 큰 악화 기여 5개:

{markdown(segments['station_id'].sort_values('extra_log_se_sum',ascending=False).head(5))}

## 5. 해석과 다음 선택

날짜 단위 paired bootstrap에서 RMSLE 차이의 조건부 95% 구간은 **[{bootstrap['ci95'][0]:.9f}, {bootstrap['ci95'][1]:.9f}]**다.
시간상 인접한 날짜의 종속성과 계절 변화까지 보장하는 구간은 아니다.
현재 검증 구간에서 LightGBM에서 XGBoost·CatBoost 방향으로 아주 조금 이동할 때 log-MSE 미분은 각각
**{derivative[0]:.9f}, {derivative[1]:.9f}**다. 둘 다 양수라면 이 세 예측의 비음수 로그 결합 중 LightGBM 단독이 해당 구간의 최적점이다.
이는 사후 설명이며 이 검증 타겟으로 새 제출 가중치를 적합하지 않았다.

따라서 앙상블 자체를 목표로 삼기보다 LightGBM 단독을 기준 후보로 유지하고,
실제 테스트처럼 모든 역에 이력이 있는 개발 구간을 더 중시하는 별도 검증 설계를 먼저 고정하는 편이 타당하다.
그 설계에서 두 보조 모델이 일관되게 이득을 내는 경우에만 비중을 부여해야 한다.
현재 결과만으로 임의의 80:10:10 비율이 미래 테스트에서 더 좋다고 주장할 근거는 없다.

재현: `.venv/bin/python analysis/three_tree/analyze.py`
원본 모델·가중치·제출 파일 변경 없음. 통계는 CSV 및 `summary.json`에 보관했다.
'''
    (OUT/'WHY_ENSEMBLE_LOSES.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
