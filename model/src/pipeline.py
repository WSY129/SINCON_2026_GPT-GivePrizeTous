"""Reproducible EDA -> fixed temporal CV -> fit -> verified CSV pipeline."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .features import load_data,make_features,training_matrix
from .models import Model
from .ensemble import fit_ensemble,predict_ensemble
from .metrics import regression_metrics,forecast_diagnostics
from .validate_submission import validate_submission

ROOT=Path(__file__).resolve().parents[1]

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.bool_,)):return bool(value)
    return value

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def environment():
    versions={}
    for name in ['numpy','pandas','polars','scikit-learn','scipy','lightgbm','catboost','xgboost','torch','joblib']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'python':sys.version,'platform':platform.platform(),'packages':versions}

def diagnostics(history,valid,prediction,origin,path):
    result=forecast_diagnostics(history.loc[history.date<=origin],valid,prediction,origin=origin)
    for key in ['per_entity','residual_acf','per_step']:
        result[key].to_csv(path/f'{key}.csv',index=False)
    monthly=[]
    v=valid.copy();v['prediction']=prediction
    for month,rows in v.groupby(v.date.dt.to_period('M')):
        monthly.append({'month':str(month),'rows':len(rows),**regression_metrics(rows.boardings,rows.prediction)})
    pd.DataFrame(monthly).to_csv(path/'per_month.csv',index=False)
    # Day-block bootstrap reflects the competition's date-level public/private grouping.
    daily=v.assign(log_se=(np.log1p(v.prediction)-np.log1p(v.boardings))**2).groupby('date').log_se.agg(['sum','size'])
    rng=np.random.default_rng(2026)
    values=[]
    for _ in range(500):
        ix=rng.integers(0,len(daily),size=len(daily));sample=daily.iloc[ix]
        values.append(float(np.sqrt(sample['sum'].sum()/sample['size'].sum())))
    result['summary']['day_bootstrap_rmsle_ci95']=[float(np.quantile(values,.025)),float(np.quantile(values,.975))]
    result['summary']['day_bootstrap_note']='500 date-block resamples; descriptive conditional interval, not a guarantee for future months.'
    save_json(path/'summary.json',result['summary'])
    return result['summary']

def export_predictions(data,predictions,output,reference=None):
    output.mkdir(parents=True,exist_ok=True)
    full=data['test'][['example_id']].copy();full['boardings']=predictions
    full.to_csv(output/'submission.csv',index=False,float_format='%.8f')
    # Keep original stale template as evidence; this secondary file is diagnostic only.
    subset=data['sample_submission'][['example_id']].merge(full,on='example_id',how='left',validate='one_to_one')
    subset.to_csv(output/'template_diagnostic_DO_NOT_SUBMIT.csv',index=False,float_format='%.8f')
    full_report=validate_submission(output/'submission.csv',data['test'],data['sample_submission'],scope='test',reference_predictions=reference)
    full_report['official_expected_rows']=157194
    full_report['official_scope_source']='User-provided Overview/Data/Rules: 123 days and all test IDs; local sample is stale.'
    if len(full)!=157194:raise ValueError('Official SINCON row count must be 157194')
    save_json(output/'submission_validation.json',full_report)
    rich=data['test'].copy();rich['boardings']=predictions
    rich.to_csv(output/'predictions_with_keys.csv',index=False,float_format='%.8f')
    return full_report

def run(args):
    import joblib
    start=time.perf_counter()
    cfg=json.loads(args.config.read_text())
    if cfg['selection_metric']!='rmsle' or cfg['training_loss']!='log_mse':raise ValueError('Official metric/loss configuration mismatch')
    if len(cfg['fold_origins'])!=len(cfg['fold_roles']):raise ValueError('Fold role mismatch')
    if cfg['fold_roles']!=['development','development','audit']:raise ValueError('Expected two development folds followed by one evaluation fold')
    data=load_data(args.data_dir);train=data['train'];meta=data['metaData']
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    save_json(output/'config_resolved.json',cfg)
    save_json(output/'environment.json',environment())
    fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data_dir.glob('*.csv')}
    save_json(output/'input_sha256.json',fingerprints)
    fold_predictions=[];fold_truth=[];fold_frames=[];metrics=[];audits=[];run_times=[]
    ensemble_plan=None;development_pred=None
    for fold,(date,role) in enumerate(zip(cfg['fold_origins'],cfg['fold_roles'])):
        origin=pd.Timestamp(date);end=origin+pd.Timedelta(days=cfg['horizon_days'])
        if end>train.date.max():raise ValueError('Validation extends beyond observed training')
        valid=train.loc[(train.date>origin)&(train.date<=end)].copy().reset_index(drop=True)
        print(f'Fold {fold+1} ({role}) origin={date}; validation={len(valid):,}',flush=True)
        t=time.perf_counter()
        x,y,anchor_audit=training_matrix(train,meta,origin,cfg['horizon_days'],cfg['origin_stride_days'],cfg['max_training_rows'],cfg['seed'])
        xv=make_features(train,valid,meta,origin)
        assert x.columns.equals(xv.columns)
        fold_path=output/f'fold_{fold+1}';fold_path.mkdir(exist_ok=True)
        save_json(fold_path/'training_origins.json',anchor_audit)
        audits.append({'fold':fold+1,'role':role,'origin':date,'validation_start':str(valid.date.min().date()),'validation_end':str(valid.date.max().date()),'validation_rows':len(valid),'training_sample_rows':len(y),'feature_count':x.shape[1],'training_matrix_mb':float(x.memory_usage(deep=True).sum()/1e6),'training_anchor_max':anchor_audit[-1]['origin'],'training_label_end_max':max(a['label_end'] for a in anchor_audit),'full_horizon_embargo':True})
        print(f'  Features={x.shape}, created in {time.perf_counter()-t:.1f}s',flush=True)
        preds={}
        for name in cfg['models']:
            t=time.perf_counter();model=Model(name,cfg).fit(x,y)
            preds[name]=model.predict(xv)
            elapsed=time.perf_counter()-t
            if not np.isfinite(preds[name]).all():raise ValueError(f'{name}: nonfinite output')
            row={'fold':fold+1,'role':role,'model':name,'seconds':elapsed,**regression_metrics(valid.boardings,preds[name])}
            metrics.append(row);run_times.append({'phase':'cv','fold':fold+1,'model':name,'seconds':elapsed})
            print(f'  {name:18s} RMSLE={row["rmsle"]:.6f} MAE={row["mae"]:.3f} ({elapsed:.1f}s)',flush=True)
            del model
        frame=valid[['date','station_id','hour','boardings']].copy()
        frame['horizon']=(valid.date-origin).dt.days
        for name,pred in preds.items():frame[name]=pred
        fold_predictions.append(preds);fold_truth.append(valid.boardings.to_numpy());fold_frames.append(frame)
        if role=='development' and cfg['fold_roles'][fold+1]=='audit':
            dev_y=np.concatenate(fold_truth)
            dev_preds={name:np.concatenate([p[name] for p in fold_predictions]) for name in cfg['models']}
            ensemble_plan,corr=fit_ensemble(dev_y,dev_preds,cfg['selection_metric'],cfg['correlation_cutoff'],cfg.get('ensemble_policy'))
            ensemble_plan['fit_folds']=[i+1 for i in range(fold+1)]
            ensemble_plan['audit_fold_used_for_tuning']=cfg.get('evaluation_context',{}).get('policy_selection_has_seen_audit',False)
            ensemble_plan['audit_fold_used_for_weight_fit']=False
            ensemble_plan['evaluation_context']=cfg.get('evaluation_context',{'status':'original_audit'})
            save_json(output/'ensemble_plan.json',ensemble_plan)
            corr.to_csv(output/'development_log_residual_correlations.csv')
            development_pred=predict_ensemble(dev_preds,ensemble_plan)
            print('  Development-only ensemble frozen:',ensemble_plan['weights'],flush=True)
        if role=='audit':
            if ensemble_plan is None:raise ValueError('No development-only ensemble plan')
            pred=predict_ensemble(preds,ensemble_plan);frame['ensemble']=pred
            summary=diagnostics(train,valid,pred,origin,fold_path)
            metrics.append({'fold':fold+1,'role':role,'model':'ensemble',**summary})
            print(f'  AUDIT ensemble RMSLE={summary["rmsle"]:.6f}',flush=True)
            save_json(output/'audit_summary.json',summary)
        frame.to_csv(fold_path/'oof.csv.gz',index=False,compression='gzip')
        pd.DataFrame(metrics).to_csv(output/'cv_scores.csv',index=False)
        save_json(output/'cv_folds.json',audits)
        del x,y,xv
    for fold in [0,1]:
        frame=fold_frames[fold]
        pred=predict_ensemble(fold_predictions[fold],ensemble_plan)
        path=output/f'fold_{fold+1}'
        diagnostics(train,frame,pred,pd.Timestamp(cfg['fold_origins'][fold]),path)
        frame['ensemble']=pred
        frame.to_csv(path/'oof.csv.gz',index=False,compression='gzip')
    print('Fitting selected nonzero-weight models on all observed data...',flush=True)
    cutoff=train.date.max()
    x,y,final_audit=training_matrix(train,meta,cutoff,cfg['horizon_days'],cfg['origin_stride_days'],cfg['max_training_rows'],cfg['seed'])
    xt=make_features(train,data['test'],meta,cutoff)
    save_json(output/'final_training_origins.json',final_audit)
    models={};test_preds={}
    for name,weight in ensemble_plan['weights'].items():
        if weight<=0:continue
        t=time.perf_counter();model=Model(name,cfg).fit(x,y)
        test_preds[name]=model.predict(xt);models[name]=model
        elapsed=time.perf_counter()-t;run_times.append({'phase':'final_fit_predict','model':name,'seconds':elapsed})
        print(f'  Final {name}: weight={weight:.5f}, {elapsed:.1f}s',flush=True)
    predictions=predict_ensemble(test_preds,ensemble_plan)
    bundle={'config':cfg,'ensemble':ensemble_plan,'models':models,'features':x.columns.tolist(),'training_cutoff':str(cutoff.date()),'input_sha256':fingerprints}
    joblib.dump(bundle,output/'model_bundle.joblib',compress=3)
    export_predictions(data,predictions,output,fold_frames[-1]['ensemble'].to_numpy())
    # Record a strong single-model alternative without using holdout to choose it.
    best_name=min(ensemble_plan['individual_scores'],key=ensemble_plan['individual_scores'].get)
    if best_name in test_preds:
        alternative=data['test'][['example_id']].copy();alternative['boardings']=test_preds[best_name]
        alternative.to_csv(output/'submission_best_single.csv',index=False,float_format='%.8f')
        validate_submission(output/'submission_best_single.csv',data['test'],data['sample_submission'],scope='test')
    primary_name=cfg.get('ensemble_policy',{}).get('primary_model')
    if primary_name in test_preds:
        alternative=data['test'][['example_id']].copy();alternative['boardings']=test_preds[primary_name]
        alternative.to_csv(output/'submission_primary.csv',index=False,float_format='%.8f')
        validate_submission(output/'submission_primary.csv',data['test'],data['sample_submission'],scope='test')
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb=peak/(1024*1024) if sys.platform=='darwin' else peak/1024
    save_json(output/'run_summary.json',{'status':'completed','official_metric':'rmsle','elapsed_seconds':time.perf_counter()-start,'peak_rss_mb':peak_mb,'feature_count':x.shape[1],'models_evaluated':cfg['models'],'models_deployed':list(models),'timings':run_times,'external_data':['preannounced Korean calendar facts'],'pretrained_weights':False,'submission_rows':len(predictions),'submission_source':'all test IDs','audit_rmsle':metrics[-1]['rmsle']})
    print(f'Completed: {output / "submission.csv"}; {len(predictions):,} rows',flush=True)

def infer(args):
    import joblib
    data=load_data(args.data_dir)
    bundle=joblib.load(args.model) # Load only this project's trusted locally generated artifact.
    if str(data['train'].date.max().date())!=bundle['training_cutoff']:raise ValueError('Training cutoff mismatch')
    for filename,digest in bundle['input_sha256'].items():
        if hashlib.sha256((args.data_dir/filename).read_bytes()).hexdigest()!=digest:raise ValueError(f'Input changed: {filename}')
    x=make_features(data['train'],data['test'],data['metaData'],bundle['training_cutoff'])
    if x.columns.tolist()!=bundle['features']:raise ValueError('Feature schema mismatch')
    preds={name:model.predict(x) for name,model in bundle['models'].items()}
    prediction=predict_ensemble(preds,bundle['ensemble'])
    if bundle.get('postprocessing'):
        from .postprocessing import apply_postprocessing
        prediction=apply_postprocessing(data['train'],data['test'],bundle['training_cutoff'],prediction,bundle['postprocessing'])
    report=export_predictions(data,prediction,args.output)
    print(json.dumps(clean(report),ensure_ascii=False,indent=2))

def main():
    if len(sys.argv)>1 and sys.argv[1]=='v2':
        from .pipeline_v2 import main as v2_main
        return v2_main(sys.argv[2:])
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','infer'])
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data'/'raw')
    parser.add_argument('--config',type=Path,default=ROOT/'configs'/'default.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'lightgbm_tuned')
    parser.add_argument('--model',type=Path,default=ROOT/'outputs'/'lightgbm_tuned'/'model_bundle.joblib')
    args=parser.parse_args()
    run(args) if args.command=='train' else infer(args)

if __name__=='__main__':main()
