"""Resumable four-fold residual CV, two-seed refit and log-space inference."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
import multiprocessing
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .features import load_data
from .features_v3 import training_matrix_v3,make_features_v3
from .models_v3 import ResidualTree,devices
from .ensemble_v3 import evaluate_ensemble
from .pipeline import save_json,environment,diagnostics
from .metrics import regression_metrics

ROOT=Path(__file__).resolve().parents[1]
PROTECTED=['submission_lgbm_0.247808.csv','outputs/lightgbm_tuned/model_bundle.joblib','configs/lightgbm_tuned.json']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def preserve(root,output):
    current={name:sha(root/name) for name in PROTECTED if (root/name).exists()}
    path=output/'protected_sha256.json'
    if path.exists():
        expected=json.loads(path.read_text())
        if current!=expected:raise RuntimeError('Protected artifact hash changed; refusing to continue')
    else:save_json(path,current)
    return current

def contract(args,cfg):
    args.output.mkdir(parents=True,exist_ok=True)
    if args.output.resolve()!=(ROOT/'outputs/ensemble_v3').resolve():raise ValueError('v3 writes only outputs/ensemble_v3')
    if args.config.resolve()!=(ROOT/'configs/ensemble_v3.json').resolve():raise ValueError('v3 requires configs/ensemble_v3.json')
    preserve(ROOT,args.output)
    inputs={p.name:sha(p) for p in sorted(args.data_dir.glob('*.csv'))}
    source={p.name:sha(p) for p in sorted([*(ROOT/'src').glob('*.py'), *(ROOT/'src').glob('*.json')])}
    fingerprint={'inputs':inputs,'config_sha256':sha(args.config),'source_sha256':source}
    path=args.output/'run_fingerprint.json'
    if path.exists() and json.loads(path.read_text())!=fingerprint:raise RuntimeError('Input/config/source mismatch. Use a fresh versioned run after review; never silently reuse stale caches.')
    save_json(path,fingerprint);save_json(args.output/'config_resolved.json',cfg)
    save_json(args.output/'environment.json',{**environment(),'hardware':devices()})

def cache_fold(data,cfg,fold,output):
    path=output/f'fold_{fold}';path.mkdir(parents=True,exist_ok=True);cache=path/'matrix.joblib'
    if cache.exists():return cache
    origin=pd.Timestamp(cfg['fold_origins'][fold-1]);train=data['train']
    valid=train.loc[(train.date>origin)&(train.date<=origin+pd.Timedelta(days=123))].reset_index(drop=True)
    if len(valid)!=157194:raise ValueError(f'Fold {fold} has {len(valid)} rows, expected 157194')
    print(f'FEATURES fold={fold} origin={origin.date()}',flush=True)
    x,y,audit,horizons=training_matrix_v3(train,data['metaData'],origin,cfg['max_training_rows'],cfg['seed'])
    xv=make_features_v3(train,valid,data['metaData'],origin)
    yv=np.column_stack([np.log1p(valid.boardings)-xv.anchor_board_log,np.log1p(valid.alightings)-xv.anchor_alight_log]).astype('float32')
    save_json(path/'feature_audit.json',{'origin':str(origin.date()),'rows':len(x),'valid_rows':len(xv),'features':list(x),'origins':audit,'horizon_counts':horizons,'train_year364_available':float(x.year364_clean_log.notna().mean()),'valid_year364_available':float(xv.year364_clean_log.notna().mean()),'train_nan_fraction':x.isna().mean().to_dict(),'max_training_label':max(a['label_max'] for a in audit)})
    joblib.dump({'x':x,'y':y,'xvalid':xv,'yvalid':yv,'valid':valid,'origin':origin},cache,compress=0)
    return cache

def worker(args,cfg):
    start=time.perf_counter();matrix=joblib.load(args.cache,mmap_mode='r')
    seed=args.seed;name=args.model;dest=args.destination;dest.mkdir(parents=True,exist_ok=True)
    hardware=devices()
    if name=='tabm':
        from .tabm_v3 import TabMRegressor
        model=TabMRegressor(cfg,seed,args.iterations,hardware)
    else:model=ResidualTree(name,cfg,seed,args.iterations,hardware)
    model.fit(matrix['x'],matrix['y'],matrix.get('xvalid'),matrix.get('yvalid'))
    if 'xvalid' in matrix:prediction=model.predict_log(matrix['xvalid'])
    else:prediction=model.predict_log(matrix['xtest'])
    np.save(dest/'prediction_log.npy',prediction)
    metadata={'model':name,'seed':seed,'best_iteration':model.best_iteration,'requested_iterations':args.iterations,'seconds':time.perf_counter()-start,'device':model.device,'train_rows':len(matrix['x'])}
    save_json(dest/'history.json',model.history)
    model.to_cpu();joblib.dump(model,dest/'model.joblib',compress=3)
    save_json(dest/'result.json',metadata);print('COMPLETED',metadata,flush=True)

def run_worker(args,cfg,cache,destination,name,seed,iterations=None):
    if (destination/'result.json').exists() and (destination/'prediction_log.npy').exists() and (destination/'model.joblib').exists():return
    destination.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','src.pipeline_v3','worker','--config',str(args.config),'--cache',str(cache),'--destination',str(destination),'--model',name,'--seed',str(seed)]
    if iterations is not None:command+=['--iterations',str(iterations)]
    env=os.environ.copy();env.update(PYTHONHASHSEED=str(seed),OMP_NUM_THREADS=str(cfg['threads']),OPENBLAS_NUM_THREADS=str(cfg['threads']),VECLIB_MAXIMUM_THREADS=str(cfg['threads']),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    print('TRAIN',destination,flush=True)
    with (destination/'train.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def train_jobs(args,cfg,jobs):
    """Schedule separate interpreters without sharing a GPU between training jobs.

    CPU/MPS: at most two CPU jobs plus one MPS job. Dual CUDA: one CPU
    LightGBM queue, one GPU-0 CatBoost/TabM queue, one GPU-1 XGBoost queue.
    """
    hardware=devices();cuda=hardware['cuda_count'] if cfg['device']!='cpu' else 0
    mps=hardware['mps'] and cfg['device']!='cpu';queues={}
    def queue(name):
        if cuda:
            if name=='lightgbm':return 'cpu'
            return 'cuda1' if name=='xgboost' and cuda>1 else 'cuda0'
        return 'mps' if name=='tabm' and mps else 'cpu'
    for job in jobs:queues.setdefault(queue(job[2]),[]).append(job)
    pools={key:ThreadPoolExecutor(max_workers=cfg.get('max_cpu_workers',2) if key=='cpu' and not cuda else 1) for key in queues}
    futures=[]
    try:
        for key,items in queues.items():
            for cache,destination,name,seed,iterations in items:
                futures.append(pools[key].submit(run_worker,args,cfg,cache,destination,name,seed,iterations))
        for future in as_completed(futures):future.result()
    finally:
        for pool in pools.values():pool.shutdown(wait=True,cancel_futures=True)

def collect_cv(data,cfg,output):
    all_frames=[];summaries=[]
    for fold in range(1,5):
        matrix=joblib.load(output/f'fold_{fold}/matrix.joblib',mmap_mode='r')
        valid=matrix['valid'];origin=matrix['origin'];frame=valid[['date','station_id','hour','boardings']].copy()
        frame['fold']=fold;frame['origin']=str(origin.date());frame['horizon']=(frame.date-origin).dt.days
        frame['true_log']=np.log1p(valid.boardings)
        for name in cfg['models']:
            dest=output/f'fold_{fold}'/name;pred=np.load(dest/'prediction_log.npy');frame[name]=pred
            reportpath=dest/'diagnostics';reportpath.mkdir(exist_ok=True)
            metric=diagnostics(data['train'],valid,np.expm1(pred),origin,reportpath)
            summaries.append({'fold':fold,'model':name,**metric})
        all_frames.append(frame)
    oof=pd.concat(all_frames,ignore_index=True)
    pred,lofo,report,corr=evaluate_ensemble(oof.true_log.to_numpy(),oof[cfg['models']].to_numpy(),oof.fold.to_numpy(),cfg['models'],cfg['fold_weights'],oof.date.to_numpy())
    oof['ensemble']=pred;oof['lofo_ensemble']=lofo
    oof.to_csv(output/'oof_predictions.csv.gz',index=False,compression='gzip')
    corr.to_csv(output/'residual_correlation.csv');save_json(output/'ensemble_report.json',report)
    for fold,frame in oof.groupby('fold'):
        origin=pd.Timestamp(cfg['fold_origins'][fold-1]);dest=output/f'fold_{fold}'/'ensemble';dest.mkdir(exist_ok=True)
        metric=diagnostics(data['train'],frame,np.expm1(frame.ensemble),origin,dest)
        summaries.append({'fold':fold,'model':'ensemble',**metric})
    comparison=pd.DataFrame(summaries)
    comparison['old_fold4_rmsle']=np.where(comparison.fold==4,.247808,np.nan)
    comparison['rmsle_delta_vs_old_fold4']=comparison.rmsle-comparison.old_fold4_rmsle
    comparison.to_csv(output/'cv_metrics.csv',index=False)
    print('CV REPORT',json.dumps(report,ensure_ascii=False),flush=True)
    return report

def final_cache(data,cfg,output):
    path=output/'final';path.mkdir(exist_ok=True);cache=path/'matrix.joblib'
    if cache.exists():return cache
    x,y,audit,horizons=training_matrix_v3(data['train'],data['metaData'],cfg['final_cutoff'],cfg['final_training_rows'],cfg['seed'])
    xtest=make_features_v3(data['train'],data['test'],data['metaData'],cfg['final_cutoff'])
    save_json(path/'feature_audit.json',{'rows':len(x),'origins':audit,'horizon_counts':horizons,'train_year364_available':float(x.year364_clean_log.notna().mean()),'test_year364_available':float(xtest.year364_clean_log.notna().mean())})
    joblib.dump({'x':x,'y':y,'xtest':xtest},cache,compress=0);return cache

def write_submission(data,pred,output):
    test=data['test'];submission=test[['example_id']].copy();submission['boardings']=np.expm1(pred)
    if len(submission)!=157194 or not submission.example_id.is_unique or not np.isfinite(submission.boardings).all() or (submission.boardings<0).any():raise ValueError('Submission integrity failure')
    path=output/'submission_ensemble_v3.csv';submission.to_csv(path,index=False,float_format='%.8f')
    read=pd.read_csv(path);integrity={'rows':len(read),'columns':list(read),'ids_in_test_order':bool(read.example_id.equals(test.example_id)),'unique_ids':bool(read.example_id.is_unique),'finite':bool(np.isfinite(read.boardings).all()),'nonnegative':bool((read.boardings>=0).all()),'mean':float(read.boardings.mean()),'min':float(read.boardings.min()),'max':float(read.boardings.max()),'mean_in_requested_315_345_range':bool(315<=read.boardings.mean()<=345),'minimum_below_3':bool(read.boardings.min()<3),'quantiles':read.boardings.quantile([0,.01,.1,.5,.9,.99,1]).to_dict(),'sha256':sha(path),'note':'Mean/minimum thresholds are diagnostics only. No mean forcing, floor forcing, bias calibration or domain zeroing was applied.'}
    save_json(output/'submission_validation.json',integrity)
    rich=test.copy();rich['boardings']=read.boardings;rich.to_csv(output/'test_predictions_with_keys.csv',index=False)
    rich.groupby(rich.date.dt.to_period('M')).boardings.agg(['mean','min','max','sum']).to_csv(output/'test_monthly_distribution.csv')
    preserve(ROOT,output);print('SUBMISSION',integrity,flush=True)

def write_candidates(data, predictions, report, output):
    """Two-seed family submissions and all-fold blend for controlled comparison."""
    dest=output/'candidates';dest.mkdir(exist_ok=True)
    values=dict(predictions)
    values['all_fold_blend']=sum(report['all_fold_comparison']['weights'][name]*p for name,p in predictions.items())
    audits={}
    for name,p in values.items():
        frame=data['test'][['example_id']].copy();frame['boardings']=np.expm1(p)
        if not np.isfinite(frame.boardings).all() or (frame.boardings<0).any():raise ValueError(name)
        path=dest/f'submission_{name}.csv';frame.to_csv(path,index=False,float_format='%.8f')
        audits[name]={'rows':len(frame),'mean':float(frame.boardings.mean()),'min':float(frame.boardings.min()),'sha256':sha(path)}
    save_json(dest/'validation.json',audits)

def _cache_task(task):
    data_dir,cfg,fold,output=task
    return cache_fold(load_data(data_dir),cfg,fold,output)

def run(args,cfg):
    start=time.perf_counter();contract(args,cfg);data=load_data(args.data_dir)
    jobs=[]
    with ProcessPoolExecutor(max_workers=cfg.get('feature_workers', 1),mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(_cache_task,[(args.data_dir,cfg,fold,args.output) for fold in range(1,5)]))
    for fold in range(1,5):
        cache=args.output/f'fold_{fold}/matrix.joblib'
        for name in cfg['models']:jobs.append((cache,args.output/f'fold_{fold}'/name,name,cfg['seed'],None))
    train_jobs(args,cfg,jobs)
    report=collect_cv(data,cfg,args.output)
    if args.cv_only:return
    cache=final_cache(data,cfg,args.output);predictions={};iteration_plan={};jobs=[]
    for name in cfg['models']:
        best=[json.loads((args.output/f'fold_{f}'/name/'result.json').read_text())['best_iteration'] for f in [3,4]]
        iterations=max(1,int(np.rint(np.dot(cfg['final_iteration_fold_weights'],best)*cfg['final_iteration_multiplier'])))
        iteration_plan[name]={'fold3':best[0],'fold4':best[1],'final_iterations':iterations}
        for seed in cfg['final_seeds']:
            dest=args.output/'final'/name/str(seed)
            jobs.append((cache,dest,name,seed,iterations))
    save_json(args.output/'final_iteration_plan.json',iteration_plan)
    train_jobs(args,cfg,jobs)
    for name in cfg['models']:
        predictions[name]=np.mean([np.load(args.output/'final'/name/str(seed)/'prediction_log.npy') for seed in cfg['final_seeds']],axis=0)
    write_candidates(data,predictions,report,args.output)
    pred=sum(report['weights'][name]*value for name,value in predictions.items())
    write_submission(data,pred,args.output)
    save_json(args.output/'bundle_manifest.json',{'config':cfg,'input_sha256':json.loads((args.output/'run_fingerprint.json').read_text())['inputs'],'weights':report['weights'],'models':{name:[str((Path('final')/name/str(seed)/'model.joblib')) for seed in cfg['final_seeds']] for name in cfg['models']},'feature_columns':list(joblib.load(cache,mmap_mode='r')['x']),'seconds':time.perf_counter()-start})

def predict_family(task):
    name,paths,output,xpath,hardware,cfg=task
    x=joblib.load(xpath,mmap_mode='r');logs=[]
    for relative in paths:
        model=joblib.load(Path(output)/relative)
        if name=='tabm':
            import torch
            torch.set_num_threads(cfg['threads'])
            model.device=model.device_name();model.model.to(model.device)
        elif name=='xgboost' and hardware['cuda_count'] and cfg['device']!='cpu':model.model.set_params(device=f'cuda:{min(1,hardware["cuda_count"]-1)}')
        logs.append(model.predict_log(x))
    return name,np.mean(logs,axis=0)

def infer(args,cfg):
    manifest=json.loads((args.output/'bundle_manifest.json').read_text())
    current={p.name:sha(p) for p in sorted(args.data_dir.glob('*.csv'))}
    if current!=manifest['input_sha256']:raise ValueError('Inference input hash mismatch')
    data=load_data(args.data_dir);x=make_features_v3(data['train'],data['test'],data['metaData'],cfg['final_cutoff'])
    if list(x)!=manifest['feature_columns']:raise ValueError('Inference feature schema mismatch')
    hardware=devices()
    xpath=args.output/'final/inference_features.joblib';joblib.dump(x,xpath,compress=0)
    # Independent families predict concurrently; LightGBM uses CPU, XGBoost GPU 1,
    # TabM GPU 0 when available. CatBoost persisted CPU inference is portable.
    tasks=[(name,manifest['models'][name],str(args.output),str(xpath),hardware,cfg) for name in cfg['models']]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:predictions=dict(pool.map(predict_family,tasks))
    pred=sum(manifest['weights'][name]*value for name,value in predictions.items())
    write_submission(data,pred,args.output)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['run','infer','worker'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/ensemble_v3.json');parser.add_argument('--data-dir',type=Path,default=ROOT/'data/raw');parser.add_argument('--output',type=Path,default=ROOT/'outputs/ensemble_v3')
    parser.add_argument('--cv-only',action='store_true');parser.add_argument('--cache',type=Path);parser.add_argument('--destination',type=Path);parser.add_argument('--model');parser.add_argument('--seed',type=int,default=2026);parser.add_argument('--iterations',type=int)
    args=parser.parse_args(argv);cfg=json.loads(args.config.read_text())
    if args.command=='worker':worker(args,cfg)
    elif args.command=='infer':infer(args,cfg)
    else:run(args,cfg)

if __name__=='__main__':main()
