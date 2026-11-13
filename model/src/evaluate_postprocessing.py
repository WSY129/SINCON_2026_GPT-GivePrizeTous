"""Choose a postprocessor on temporal development OOF and evaluate it once later."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .features import load_data,make_features
from .ensemble import predict_ensemble
from .postprocessing import build_context,candidate_specs,fit_plan,apply_plan
from .metrics import regression_metrics
from .pipeline import save_json,export_predictions,diagnostics

ROOT=Path(__file__).resolve().parents[1]

def rmsle(actual,pred):
    return float(np.sqrt(np.mean((np.log1p(pred)-np.log1p(actual))**2)))

def run(args):
    started=time.perf_counter();out=args.output;base=args.base_output
    if (out/'model_bundle.joblib').exists():raise FileExistsError('Choose a new output directory to preserve the previous run')
    out.mkdir(parents=True,exist_ok=True)
    data=load_data(args.data_dir);train=data['train'];cfg=json.loads((base/'config_resolved.json').read_text())
    origins=cfg['fold_origins']
    frames=[];contexts=[];preds=[];truth=[]
    # Only these two OOF files are loaded until the selection plan is frozen.
    for fold in [1,2]:
        f=pd.read_csv(base/f'fold_{fold}/oof.csv.gz',parse_dates=['date'])
        f=f.loc[f.known_at_origin].reset_index(drop=True)
        context=build_context(train,f,origins[fold-1])
        frames.append(f);contexts.append(context)
        preds.append(f.lightgbm_large.to_numpy());truth.append(f.boardings.to_numpy())
    raw=[rmsle(y,p) for y,p in zip(truth,preds)]
    design={'calibration_fit_fold':1,'temporal_selection_fold':2,'fit_scope':'known stations only',
            'final_calibration_refit_folds':[1,2],'audit_selection_uses_labels':False,
            'max_historical_anchor_weight':.2,'anchor_windows_days':[28,56,112],
            'holiday_gate':'known-as-of-origin holidays remain exactly as base model predictions',
            'candidate_count':len(candidate_specs()),'decision_rule':'minimum forward fold-2 RMSLE among eligible candidates that do not worsen fold1; identity included',
            'caps_note':'95th-percentile caps are diagnostic-only: no physical capacity bound is provided',
            'context':'Development folds were already used for LightGBM tuning. Fold3 was analyzed earlier; all results are conditional reused-validation evidence.'}
    save_json(out/'selection_design.json',design)
    results=[];plans={}
    for spec in candidate_specs():
        plan=fit_plan(contexts[0],preds[0],truth[0],spec)
        p1=apply_plan(contexts[0],preds[0],plan);p2=apply_plan(contexts[1],preds[1],plan)
        a,b=rmsle(truth[0],p1),rmsle(truth[1],p2)
        allowed=spec['eligible'] and a<=raw[0]+1e-12
        results.append({'name':spec['name'],'development1_in_sample_rmsle':a,'development2_forward_rmsle':b,
                        'change_dev1':a-raw[0],'change_dev2':b-raw[1],'eligible_for_selection':bool(allowed),
                        'calibration':spec['calibration']['kind'],'calibration_scale':spec['calibration']['scale'],
                        'anchor_days':spec['anchor']['days'],'anchor_weight':spec['anchor']['weight'],'cap_days':spec['cap_days']})
        plans[spec['name']]=plan
    candidates=pd.DataFrame(results)
    candidates.to_csv(out/'candidate_development_scores.csv',index=False)
    chosen=candidates.loc[candidates.eligible_for_selection].sort_values(['development2_forward_rmsle','anchor_weight','calibration_scale']).iloc[0]
    spec=plans[chosen['name']]['spec']
    pooled=pd.concat(contexts,ignore_index=True)
    pooled_pred=np.concatenate(preds);pooled_y=np.concatenate(truth)
    fold_ids=np.concatenate([np.full(len(contexts[i]),i+1) for i in range(2)])
    plan=fit_plan(pooled,pooled_pred,pooled_y,spec,fold_ids)
    plan['selection']=design;plan['selected_forward_scores']=chosen.to_dict()
    plan['source_base_bundle_sha256']=hashlib.sha256((base/'model_bundle.joblib').read_bytes()).hexdigest()
    save_json(out/'postprocessing_plan.json',plan)
    plan_digest=hashlib.sha256((out/'postprocessing_plan.json').read_bytes()).hexdigest()
    print(f'Frozen postprocessing: {chosen["name"]}; forward dev2 {raw[1]:.9f} -> {chosen["development2_forward_rmsle"]:.9f}',flush=True)
    for i in range(2):
        f=frames[i].copy();f['postprocessed']=apply_plan(contexts[i],preds[i],plan)
        folder=out/f'fold_{i+1}';folder.mkdir(exist_ok=True)
        f.to_csv(folder/'oof.csv.gz',index=False,compression='gzip')
    # This file is read only after all candidates and parameters have been frozen.
    f=pd.read_csv(base/'fold_3/oof.csv.gz',parse_dates=['date'])
    context=build_context(train,f,origins[2]);original=f.lightgbm_large.to_numpy()
    corrected=apply_plan(context,original,plan)
    raw_score=rmsle(f.boardings,original);new_score=rmsle(f.boardings,corrected)
    folder=out/'fold_3';folder.mkdir(exist_ok=True)
    f['postprocessed']=corrected;f.to_csv(folder/'oof.csv.gz',index=False,compression='gzip')
    summary=diagnostics(train,f,corrected,pd.Timestamp(origins[2]),folder)
    save_json(out/'audit_summary.json',summary)
    assert plan_digest==hashlib.sha256((out/'postprocessing_plan.json').read_bytes()).hexdigest()
    segments=[]
    f['is_holiday']=context.is_holiday.to_numpy()
    for label,mask in [('all',np.ones(len(f),dtype=bool)),('nonholiday',~f.is_holiday),('holiday',f.is_holiday)]:
        if np.sum(mask):segments.append({'segment':label,'rows':int(np.sum(mask)),'raw_rmsle':rmsle(f.loc[mask,'boardings'],original[mask]),'postprocessed_rmsle':rmsle(f.loc[mask,'boardings'],corrected[mask])})
    for month,rows in f.groupby(f.date.dt.to_period('M')):
        ix=rows.index.to_numpy();segments.append({'segment':str(month),'rows':len(rows),'raw_rmsle':rmsle(rows.boardings,original[ix]),'postprocessed_rmsle':rmsle(rows.boardings,corrected[ix])})
    pd.DataFrame(segments).to_csv(out/'audit_segments.csv',index=False)
    # Fixed component ablations explain the outcome; no coefficients are refit
    # using the reused comparison targets and no alternative is auto-promoted.
    ablations=[]
    for name in ['raw','calibration_only','anchor_only','combined']:
        component=copy.deepcopy(plan)
        if name=='calibration_only':component['spec']['anchor']={'days':0,'weight':0.}
        if name=='anchor_only':component['spec']['calibration']={'kind':'none','scale':0.}
        values=original if name=='raw' else apply_plan(context,original,component)
        error=np.log1p(values)-np.log1p(f.boardings)
        ablations.append({'component':name,'rmsle':float(np.sqrt(np.mean(error**2))),
                          'mean_log_error':float(error.mean()),'log_error_variance':float(np.var(error))})
    pd.DataFrame(ablations).to_csv(out/'component_ablation.csv',index=False)
    daily=pd.DataFrame({'date':f.date,'raw':(np.log1p(original)-np.log1p(f.boardings))**2,'post':(np.log1p(corrected)-np.log1p(f.boardings))**2}).groupby('date').agg(raw=('raw','sum'),post=('post','sum'),n=('date','size'))
    rng=np.random.default_rng(2026);ix=rng.integers(0,len(daily),size=(2000,len(daily)))
    n=daily.n.to_numpy()[ix].sum(axis=1)
    delta=np.sqrt(daily.post.to_numpy()[ix].sum(axis=1)/n)-np.sqrt(daily.raw.to_numpy()[ix].sum(axis=1)/n)
    # Keep original weights immutable; attach the frozen postprocessor to a new bundle.
    bundle=joblib.load(base/'model_bundle.joblib')
    for name,digest in bundle['input_sha256'].items():
        if hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Input changed: {name}')
    xt=make_features(train,data['test'],data['metaData'],bundle['training_cutoff'])
    model_preds={name:model.predict(xt) for name,model in bundle['models'].items()}
    before=predict_ensemble(model_preds,bundle['ensemble'])
    test_context=build_context(train,data['test'],bundle['training_cutoff'])
    after=apply_plan(test_context,before,plan)
    bundle=copy.copy(bundle);bundle['postprocessing']=plan
    bundle['postprocessing_source']=str(base)
    joblib.dump(bundle,out/'model_bundle.joblib',compress=3)
    export_predictions(data,after,out,corrected)
    raw_output=out/'unprocessed_reference';export_predictions(data,before,raw_output,original)
    assert hashlib.sha256((raw_output/'submission.csv').read_bytes()).hexdigest()==hashlib.sha256((base/'submission.csv').read_bytes()).hexdigest()
    loaded=joblib.load(out/'model_bundle.joblib')
    reloaded=apply_plan(test_context,before,loaded['postprocessing'])
    np.testing.assert_array_equal(after,reloaded)
    assert hashlib.sha256((base/'model_bundle.joblib').read_bytes()).hexdigest()==plan['source_base_bundle_sha256']
    report={'status':'completed','base_rmsle':raw_score,'postprocessed_rmsle':new_score,
            'relative_rmsle_reduction_percent':100*(1-new_score/raw_score),'selected_candidate':spec['name'],
            'improved_reused_comparison':new_score<raw_score,'seconds':time.perf_counter()-started,
            'paired_date_bootstrap_delta_ci95':list(map(float,np.quantile(delta,[.025,.975]))),
            'bootstrap_note':'Descriptive paired date resampling; does not preserve all serial dependence or prove future performance.',
            'postprocessing_plan_frozen_before_comparison':True,'original_weights_unchanged':True,
            'test_rows':len(after),'test_holiday_predictions_unchanged':bool(np.array_equal(after[test_context.is_holiday],before[test_context.is_holiday])),
            'prediction_reloaded_identical':True,'evaluation_context':design['context']}
    save_json(out/'run_summary.json',report)
    save_json(out/'adoption_decision.json',{'adopted_as_default':False,
              'reason':'Experimental evaluation only. Keep the base model unless a separate documented adoption decision is made.',
              'base_rmsle':raw_score,'candidate_rmsle':new_score,'base_model_bundle':str(base/'model_bundle.joblib')})
    (out/'EXPERIMENT_ONLY.md').write_text('This directory contains postprocessing experiments. It is not the default submission. See adoption_decision.json and reports/POSTPROCESSING.md.\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/raw')
    p.add_argument('--base-output',type=Path,default=ROOT/'outputs/lightgbm_tuned')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/lightgbm_postprocessed')
    run(p.parse_args())

if __name__=='__main__':main()
