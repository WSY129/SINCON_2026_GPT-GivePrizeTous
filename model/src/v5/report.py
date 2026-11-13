"""Measured artifacts only; no projected Public scores."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from ..metrics import regression_metrics
from ..holiday_calendar import calendar_features
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/ensemble_v5'

def build_report():
    r=json.loads((OUT/'step_b_report.json').read_text());cfg=json.loads((ROOT/'configs/ensemble_v5.json').read_text())
    r['submission_validation']=json.loads((OUT/'submission_validation.json').read_text());r['run_status']=json.loads((OUT/'run_status.json').read_text())
    r['provenance']={'oof_seed':2026,'final_seeds':[2026,2027],'weights_fit_folds':[1,2,3,4],'official_metric':'RMSLE','public_score':None,'source_sha256':hashlib.sha256((OUT/'step_b_report.json').read_bytes()).hexdigest()}
    (OUT/'ensemble_report.json').write_text(json.dumps(r,indent=2))
    oof=pd.read_csv(OUT/'step_b_oof.csv.gz',parse_dates=['date']);v=oof.loc[oof.fold==4].reset_index(drop=True)
    assert len(v)==157194
    cal=calendar_features(v.date,as_of='2026-02-27');masks={'all':np.ones(len(v),bool),'labor_day':cal.is_labor_day.eq(1),'holiday_known_at_origin':cal.is_holiday.eq(1),'gyeyang_gyulhyeon':v.station_id.isin([3110,3111]),'new_stations':v.station_id.isin([3107,3108,3109])}
    for sid in sorted(v.station_id.unique()):masks[f'station_{sid}']=v.station_id.eq(sid)
    for month in sorted(v.date.dt.strftime('%Y-%m').unique()):masks[f'month_{month}']=v.date.dt.strftime('%Y-%m').eq(month)
    for h in range(6,24):masks[f'hour_{h}']=v.hour.eq(h)
    for a,b in [(1,30),(31,60),(61,93),(94,123)]:masks[f'horizon_{a}_{b}']=v.horizon.between(a,b)
    for sid in [3107,3110,3111,3127,3226,3754]:
        for month in sorted(v.date.dt.strftime('%Y-%m').unique()):masks[f'station_{sid}_month_{month}']=v.station_id.eq(sid)&v.date.dt.strftime('%Y-%m').eq(month)
        for h in range(6,24):masks[f'station_{sid}_hour_{h}']=v.station_id.eq(sid)&v.hour.eq(h)
    rows=[]
    for name in cfg['models']+['ensemble','anchor_v5']:
        total=float(np.square(v[name]-v.true_log).sum())
        for segment,mask in masks.items():
            g=v.loc[mask]
            if not len(g):continue
            sse=float(np.square(g[name]-g.true_log).sum());pred=np.expm1(g[name])
            rows.append({'segment':segment,'model':name,'rows':len(g),**regression_metrics(g.boardings,pred),'log_sse':sse,'log_sse_share':sse/total,'actual_mean':float(g.boardings.mean()),'prediction_mean':float(pred.mean())})
    pd.DataFrame(rows).to_csv(OUT/'fold4_domain_segments.csv',index=False)
    def lengths(obj):
        if isinstance(obj,list):return [len(obj)]
        return [n for value in obj.values() for n in lengths(value)]
    budget=[]
    for path in sorted(OUT.glob('fold_*/*/result.json')):
        result=json.loads(path.read_text());history=json.loads((path.parent/'history.json').read_text())
        budget.append({'fold':int(path.parts[-3].split('_')[-1]),'model':path.parent.name,**result,'executed_iterations':max(lengths(history),default=0)})
    pd.DataFrame(budget).to_csv(OUT/'cv_training_budget.csv',index=False)
    assert len(budget)==28
    lines=['# SINCON ensemble_v5 실행 결과','', '## 점수 비교','', '| 버전/모델 | 전체 OOF | 가중 4-Fold | Fold 3+4 | Fold 4 |','|---|---:|---:|---:|---:|']
    for version in ['ensemble_v2','ensemble_v3','ensemble_v4']:
        candidates=[ROOT/'outputs'/version/n for n in ['ensemble_report.json','step_b_report.json']]
        for p in candidates:
            if p.exists():
                obj=json.loads(p.read_text());entry=obj.get('ensemble',{})
                if entry:lines.append('| '+version+' | '+' | '.join(f'{entry.get(k,float("nan")):.6f}' for k in ['oof_rmsle','weighted_rmsle','fold34_rmsle','fold4_rmsle'])+' |')
                break
    for name,entry in list(r['models'].items())+[('v5 ensemble',r['ensemble'])]:lines.append('| '+name+' | '+' | '.join(f'{entry[k]:.6f}' for k in ['oof_rmsle','weighted_rmsle','fold34_rmsle','fold4_rmsle'])+' |')
    lines+=['','## 구현과 검증','',
        '- 기존 소스·설정·v1~v4 산출물을 변경하지 않고 src/v5에 구현했다. 입력·소스·설정 해시를 fingerprint.json에 기록했다.',
        '- 모든 과거 집계는 기준일 이전 관측만 사용한다. 계양·귤현의 개통 전 이력은 특징 생성 전에 제거한다. 전년 커널과 성장 비교에는 대상일별 개통 경계 보호를 적용했다.',
        '- 평시 ±14일 삼각 중앙값과 고변동 ±21일 균등 로그 평균을 연속 혼합한다. 공휴일 대비 일요일·근로자의 날 과거 충격을 기준선에 더한다.',
        '- 신설역은 최근 56일의 탈계절화 주간 기울기를 0~0.015로 제한하고 0.90 감쇠를 적용한다. 노선 계절 변화는 기존 역에서 추정한다.',
        '- 정답당 단기/중기/장기 기준일 각 1개 이하, 총 3개 이하. 계절 균형 비복원 표본과 n_yoy>=2일 때 1.25배의 완만한 학습 가중치를 사용한다.',
        '- 저수요 모델은 late_or_low 버킷에서만 허용한다. 전역/regular 직접 트리 합계 최소 20%, 신설역 공간 모델 최대 45% 제약을 적용한다.',
        '- 가중치는 0.35×가중 4-Fold 로그 MSE +0.65×Fold34 로그 MSE와 버킷 수축항으로 학습했다. Fold4 결과를 보고 fallback을 선택하지 않는다.',
        '- 신경망 CV는 검증 상위 3개 체크포인트를 평균한다. 전체 자료 재학습은 검증 라벨이 없으므로 계획된 마지막 3개 에폭을 평균하며, 이를 검증 Top-3라고 부르지 않는다. 최종 예측은 모든 모델 2시드 평균이다.',
        '- 공간 신경망은 자체 graph/temporal Transformer이며 원 논문 iTransformer의 재현은 아니다.',
        '- 최적 반복 수가 작다는 사실만으로 과적합/미수렴을 단정하지 않는다. convergence_review.json에 진단과 시간 내 반 학습률 실험을 기록한다.',
        '', '## 해석 한계','',
        '- OOF 가중치 적합 점수는 독립 평가가 아니다. 별도 날짜 purge 및 forward block 메타 검증을 보고하지만, 기본 모델 조기 종료와 피처 선택을 완전히 중첩 재학습한 검증은 아니다.',
        '- v2 Public 0.18265, v4 Public 0.18678은 사용자 제출 결과다. v5 Public 점수는 아직 없다. 로컬 개선이 Public 개선을 보장하지 않는다.',
        '- 버킷 제약 유무 비교는 ensemble_report.json의 additional_checks에 있다. 세그먼트는 중복되므로 SSE 비중을 합산하면 안 된다.',
        '', '## 제출 검수','', '```json',json.dumps(r['submission_validation'],ensure_ascii=False,indent=2),'```','',f"전체 실행 경과: {r['run_status']['seconds']:.1f}초. 재로딩 일치 및 보호 파일 해시 검수는 별도 JSON에 기록했다."]
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':build_report()
