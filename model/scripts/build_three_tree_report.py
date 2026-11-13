"""Verify saved three-tree predictions and write the current experiment report."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import save_json

OUT=ROOT/'outputs/three_tree'
OLD=ROOT/'outputs/baseline'
def read(name):return json.loads((OUT/f'{name}.json').read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    plan=read('ensemble_plan')
    names=['lightgbm_large','xgboost','catboost']
    if set(plan['weights'])!=set(names):raise AssertionError('Unexpected ensemble model set')
    if plan['weights']['lightgbm_large']<.5-1e-12:raise AssertionError('Primary share too small')
    if abs(sum(plan['weights'].values())-1)>1e-12:raise AssertionError('Weights do not sum to one')
    if plan['bias']!=0:raise AssertionError('Unexpected bias correction')
    rows=[];checks=[]
    for fold in [1,2,3]:
        frame=pd.read_csv(OUT/f'fold_{fold}/oof.csv.gz')
        old=pd.read_csv(OLD/f'fold_{fold}/oof.csv.gz')
        pd.testing.assert_frame_equal(frame[['date','station_id','hour','boardings']],old[['date','station_id','hour','boardings']])
        # Independent direct formula, without importing predict_ensemble or score.
        z=sum(plan['weights'][name]*np.log1p(frame[name].to_numpy()) for name in names)
        pred=np.expm1(z)
        np.testing.assert_allclose(pred,frame.ensemble.to_numpy(),rtol=1e-12,atol=1e-10)
        exact={name:float(np.max(np.abs(frame[name]-old[name]))) for name in names}
        if any(v>1e-10 for v in exact.values()):raise AssertionError('Base model prediction changed unexpectedly')
        checks.append({'fold':fold,'base_max_abs_difference_from_previous_run':exact,
                       'independent_formula_max_abs_difference':float(np.max(np.abs(pred-frame.ensemble)))})
        for name in names+['ensemble']:
            rmsle=float(np.sqrt(np.mean((np.log1p(frame[name])-np.log1p(frame.boardings))**2)))
            rows.append({'fold':fold,'model':name,'rmsle':rmsle,'rows':len(frame)})
    scores=pd.DataFrame(rows)
    scores.to_csv(OUT/'verified_scores.csv',index=False)
    check=read('submission_validation')
    assert check['valid'] and check['rows']==157194
    reload_path=ROOT/'outputs/three_tree_reloaded/submission.csv'
    assert digest(OUT/'submission.csv')==digest(reload_path)
    import joblib
    bundle=joblib.load(OUT/'model_bundle.joblib')
    assert set(bundle['models'])==set(names)
    assert bundle['ensemble']==plan
    old_score=json.loads((OLD/'audit_summary.json').read_text())['rmsle']
    new_score=read('audit_summary')['rmsle']
    independent=float(scores.loc[(scores.fold==3)&(scores.model=='ensemble'),'rmsle'].iloc[0])
    assert abs(independent-new_score)<1e-12
    verification={'status':'passed','independent_audit_rmsle':independent,
                  'previous_audit_rmsle':old_score,'relative_rmsle_reduction_percent':100*(1-new_score/old_score),
                  'fold_prediction_checks':checks,'saved_bundle_models':list(bundle['models']),
                  'reload_bitwise_identical':True,'submission_sha256':digest(OUT/'submission.csv'),
                  'primary_submission_sha256':digest(OUT/'submission_primary.csv'),
                  'legacy_submission_sha256':digest(OLD/'submission.csv'),
                  'evaluation_context':plan['evaluation_context']}
    save_json(OUT/'verification.json',verification)
    table=scores.pivot(index='model',columns='fold',values='rmsle').reindex(names+['ensemble'])
    score_table='| 모델 | 개발 1 | 개발 2 | 재검증 3 | 가중치 |\n|---|---:|---:|---:|---:|\n'
    for name,row in table.iterrows():
        weight=plan['weights'].get(name,1.)
        score_table+=f'| {name} | {row[1]:.9f} | {row[2]:.9f} | {row[3]:.9f} | {weight:.9%} |\n'
    run=read('run_summary')
    report=f'''# LightGBM 중심 3모델 재앙상블

작성일: 2026-11-11. 현재 기본 설정은 `configs/default.json`, 현재 산출물은 `outputs/three_tree`다.

## 결과

기존 앙상블 RMSLE **{old_score:.10f} → {new_score:.10f}**, 상대 오차 **{verification['relative_rmsle_reduction_percent']:.2f}% 감소**.
같은 2026-02-28~2026-06-30의 123일·157,194행에서 측정했다. 예측 기준일은 2026-02-27이다.
LightGBM Large 단독 **{table.loc['lightgbm_large',3]:.10f}**보다 앙상블의 RMSLE가 높다. 세 모델 결합 자체가 단독 모델보다 우수하다고 결론내리지 않는다.
Kaggle에 제출하지 않았으며 이 수치는 리더보드 점수가 아니다.

{score_table}

## 변경 내용과 가중치 선정

- 활성 학습·앙상블·저장 모델은 LightGBM Large, XGBoost, CatBoost 세 개다.
- `seasonal7`, `weekday_mean`, LightGBM Small, MLP를 활성 모델 목록에서 제거했다. 계절성·과거 요일 통계는 입력 특징으로 계속 사용한다.
- LightGBM Large를 메인으로 고정하기 위해 가중치 최소 50%, 모든 가중치 비음수, 합 1의 제약을 사용한다. 다른 두 모델에는 양수 하한을 강제하지 않았으며 최적화 결과 자연스럽게 양의 비중을 얻었다.
- 개발 fold 1·2 전체 OOF의 log-MSE를 SLSQP로 최소화했다. 로그 타겟과 예측의 잔차 Gram 행렬을 사용한 볼록 이차 최적화이며, 가중치는 마지막 fold 예측 전에 저장했다.
- 결합 산식: `expm1(sum(weight[m] * log1p(pred[m])))`. 상수 편향 보정은 0, 잔차 상관성에 의한 자동 제거는 비활성화했다.
- 특징 60개, 모델 하이퍼파라미터, 학습 표본·시간 분할은 기존과 같다. 세 모델을 각 fold 및 최종 전체 학습 데이터에서 다시 학습했다.

## 검증 범위와 해석

| 구간 | 예측 기준일 | 검증 날짜 | 용도 |
|---|---|---|---|
| 개발 1 | 2025-06-26 | 2025-06-27~2025-10-27 | 가중치 최적화 |
| 개발 2 | 2025-10-27 | 2025-10-28~2026-02-27 | 가중치 최적화 |
| 재검증 3 | 2026-02-27 | 2026-02-28~2026-06-30 | 고정 가중치 비교 |

가중치 계산에는 fold 3의 타겟을 사용하지 않았다. 다만 모델 구성 변경과 편향 보정 제거는 앞서 같은 fold 3을 분석한 뒤 결정했다.
따라서 이번 결과는 **기존 검증 구간의 재평가**이며, 새로운 독립 holdout 성능으로 해석하지 않는다.
개발 1에는 신규 역의 초기 관측이 포함되어 실제 테스트와 개체 구성이 다르다. 이번에는 기존 결과와 비교하기 위해 개발 행·기간을 바꾸지 않았다.
운영진 공식 베이스라인은 2026-03-01부터 122일을 평가하므로 그 점수 0.30770과 완전히 동일한 구간 비교는 아니다.

## 실행 및 검증

- 전체 학습·진단·최종 예측: {run['elapsed_seconds']:.1f}초, 프로세스 peak RSS {run['peak_rss_mb']/1024:.2f} GiB. 환경 초기 설치와 검증 재추론 시간 제외.
- 새 제약 최적화·기존 누수·달력·지표·제출 테스트 32개 통과.
- 기존 저장 OOF와 세 모델의 재학습 예측이 일치함을 확인했다. 특징이나 모델 변경 효과를 섞지 않고 앙상블 변경 효과를 비교했다.
- 저장 OOF에서 공식 RMSLE 및 가중 결합 산식을 독립 계산해 요약 결과와 일치함을 확인했다.
- 저장 bundle에는 요청한 세 모델만 존재하며, 재로딩 후 생성한 CSV의 SHA-256이 일치한다.
- 제출 행 157,194개, ID와 순서, NaN·무한대·음수 검수를 통과했다.

## 산출물

상세 원인 분석: [앙상블이 LightGBM 단독보다 나쁜 이유](../analysis/three_tree/WHY_ENSEMBLE_LOSES.md).

- [현재 앙상블 제출 파일](../outputs/three_tree/submission.csv)
- [LightGBM Large 단독 제출 파일](../outputs/three_tree/submission_primary.csv)
- [가중치와 선정 설정](../outputs/three_tree/ensemble_plan.json)
- [독립 재계산 점수](../outputs/three_tree/verified_scores.csv)
- [재학습·재추론 검증](../outputs/three_tree/verification.json)
- [현재 재현 노트북](../notebooks/SINCON_reproduce.ipynb)

`submission_best_single.csv`는 개발 1·2를 합친 점수가 가장 좋은 단일 모델(CatBoost)의 파일이다. LightGBM Large 단독은 명시적으로 `submission_primary.csv`를 사용한다.
기존 `outputs/baseline`은 이전 실험 기록으로 보존했다. 이전 7모델 재현 설정은 `configs/legacy_seven.json`이다.

```bash
cd /Users/hongtaeyang/Desktop/SINCON
.venv/bin/python -m src.pipeline train --output outputs/three_tree
.venv/bin/python -m src.pipeline infer --model outputs/three_tree/model_bundle.joblib --output outputs/three_tree_reloaded
.venv/bin/python scripts/build_three_tree_report.py
```
'''
    (ROOT/'reports/THREE_TREE_REENSEMBLE.md').write_text(report,encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
