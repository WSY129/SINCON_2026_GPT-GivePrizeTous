"""Two-stage v4 ablation, isolated workers, reproducible saved predictions."""
import argparse,hashlib,json,os,sys,time,subprocess,copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
import multiprocessing
import numpy as np
import pandas as pd
import joblib
from ..features import load_data
from ..pipeline import save_json,diagnostics
from ..models_v2 import devices
from .data import origins,origin_cache,matrix
from .ensemble import buckets,evaluate,blend,summary
from .convergence import final_iterations,validate_convergence

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/ensemble_v5'
BASE=['lightgbm','catboost_cpu','xgboost','tabm_v5']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def preserve():
    files=[]
    for folder in ['src','configs','outputs/ensemble_v2','outputs/ensemble_v3','outputs/ensemble_v4','outputs/lightgbm_tuned']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in str(p) and not p.name.startswith('._') and '/src/v5/' not in str(p) and p.name not in ('ensemble_v5.json','.DS_Store'):files.append(p)
    files+=[p for p in ROOT.glob('submission_lgbm_0.247808.csv')]
    current={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    path=OUT/'preserved_before.json'
    if path.exists():
        old=json.loads(path.read_text());changes=[k for k,v in old.items() if current.get(k)!=v]
        if changes:raise RuntimeError(f'Protected files changed: {changes[:5]}')
        save_json(OUT/'preservation_check.json',{'checked':len(old),'changed':changes})
    else:save_json(path,current)

def contract(cfg):
    OUT.mkdir(parents=True,exist_ok=True);preserve()
    inputs={p.name:sha(p) for p in (ROOT/'data/raw').glob('*.csv')}
    source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/v5').glob('*.py')}
    fp={'inputs':inputs,'source':source,'config':sha(ROOT/'configs/ensemble_v5.json')}
    path=OUT/'fingerprint.json'
    if path.exists() and json.loads(path.read_text())!=fp:raise ValueError('v5 source/config/input changed: refuse stale cache reuse')
    save_json(path,fp);save_json(OUT/'config_resolved.json',cfg)

def block_subset(x,y,keys,maximum=60000,seed=2026):
    n=len(x)//18;rng=np.random.default_rng(seed);days=np.sort(rng.choice(n,min(n,maximum),replace=False));ix=(days[:,None]*18+np.arange(18)).ravel()
    return x.iloc[ix].reset_index(drop=True),y[ix],keys.iloc[ix].reset_index(drop=True)

def worker(args,cfg):
    start=time.time();m=joblib.load(args.cache);dest=Path(args.dest);dest.mkdir(parents=True,exist_ok=True);hw=devices()
    cfg=copy.deepcopy(cfg);cfg['threads']=int(os.environ.get('SINCON_JOB_THREADS',cfg['threads']))
    x=m['x'];y=m['y'];xv=m.get('xvalid');yv=m.get('yvalid');query=m['query_x'];name=args.model
    if name=='tabm_v5':
        from .tabm import TabMRegressor
        model=TabMRegressor(cfg,args.seed,args.iterations,hw).fit(x,y,xv,yv)
        pred=model.predict_log(query)
    elif name=='st_itransformer':
        from .spatial import SpatialRegressor
        data=load_data(ROOT/'data/raw');dx,dy,dk=block_subset(m['daily_x'],m['daily_y'],m['daily_keys'],maximum=120000,seed=args.seed)
        vk=m['valid'][['date','station_id','hour']].copy();vk['origin']=m['origin']
        model=SpatialRegressor(cfg,args.seed,args.iterations,hw).fit(dx,dy,query,yv,dk,vk,data['train'],data['metaData'])
        pred=model.predict_log(query,vk)
    else:
        from .models import ResidualTree,LowDemand,Decomposition
        if name=='low_demand_specialist':model=LowDemand(cfg,args.seed,args.iterations,hw).fit(x,y,xv,yv)
        elif name=='decomp_gbdt':model=Decomposition(cfg,args.seed,args.iterations,hw).fit(x,y,xv,yv,daily_x=m['daily_x'],daily_y=m['daily_y'])
        else:model=ResidualTree(name,cfg,args.seed,args.iterations,hw).fit(x,y,xv,yv)
        pred=model.predict_log(query)
    if not np.isfinite(pred).all():raise FloatingPointError(name)
    convergence=validate_convergence(name,{'best_iteration':model.best_iteration},cfg)
    save_json(dest/'convergence.json',dict(convergence,checkpoint_epochs=getattr(model,'averaged_epochs',None),checkpoint_policy=getattr(model,'checkpoint_policy',None)))
    np.save(dest/'prediction_log.npy',pred);save_json(dest/'history.json',model.history)
    model.to_cpu();joblib.dump(model,dest/'model.joblib',compress=3)
    save_json(dest/'result.json',{'name':name,'seed':args.seed,'best_iteration':model.best_iteration,'seconds':time.time()-start,'rows':len(x),'device':model.device})

def train_one(job,cfg,deadline):
    cache,dest,name,seed,it=job
    if (dest/'result.json').exists():return
    dest.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','src.v5.pipeline','worker','--cache',str(cache),'--dest',str(dest),'--model',name,'--seed',str(seed)]
    if it is not None:cmd+=['--iterations',str(it)]
    env=os.environ.copy();env.update(PYTHONHASHSEED=str(seed),SINCON_JOB_THREADS=str(cfg['threads']),OMP_NUM_THREADS=str(cfg['threads']),OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONWARNINGS='ignore::FutureWarning')
    print('TRAIN',dest,flush=True)
    with (dest/'train.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.time()))
    print('DONE',dest,flush=True)

def train_jobs(jobs,cfg,deadline):
    # CPU CatBoost/LGB are concurrent with one GPU queue; no GPU contention.
    cpu=ThreadPoolExecutor(max_workers=2);gpu=ThreadPoolExecutor(max_workers=1);f=[]
    try:
        for job in jobs:
            pool=gpu if job[2] in ['xgboost','tabm_v5','st_itransformer'] else cpu
            f.append(pool.submit(train_one,job,cfg,deadline))
        for v in as_completed(f):v.result()
    finally:cpu.shutdown(wait=True,cancel_futures=True);gpu.shutdown(wait=True,cancel_futures=True)

def features_for(folds,cfg):
    allorig=set()
    for fold in folds:
        cutoff=cfg['final_cutoff'] if fold==0 else cfg['final_diagnostic_origin'] if fold==5 else cfg['fold_origins'][fold-1]
        allorig.update(origins(cutoff,7 if fold==4 else 14))
    tasks=[(str(ROOT/'data/raw'),str(OUT),str(o.date())) for o in sorted(allorig)]
    with ProcessPoolExecutor(max_workers=cfg['feature_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:list(pool.map(origin_cache,tasks))
    # Serial assembly limits copies of multi-million-row matrices.
    for fold in folds:matrix(ROOT/'data/raw',OUT,cfg,fold)

def collect(folds,names,cfg,label):
    frames=[]
    for fold in folds:
        m=joblib.load(OUT/f'fold_{fold}/matrix.joblib');v=m['valid'];f=v[['date','station_id','hour','boardings']].copy()
        f['fold']=fold;f['horizon']=(f.date-m['origin']).dt.days;f['true_log']=np.log1p(f.boardings)
        f['bucket']=buckets(m['query_x']);f['anchor_v5']=m['query_x'].anchor_v5.to_numpy()
        for name in names:f[name]=np.load(OUT/f'fold_{fold}'/name/'prediction_log.npy')
        frames.append(f)
    oof=pd.concat(frames,ignore_index=True);pred,report,lofo=evaluate(oof,names);oof['ensemble']=pred;oof['lofo']=lofo
    oof.to_csv(OUT/f'{label}_oof.csv.gz',index=False,compression='gzip');save_json(OUT/f'{label}_report.json',report)
    residual=oof[names].rsub(oof.true_log,axis=0);residual.corr().to_csv(OUT/f'{label}_residual_correlation.csv')
    station=[]
    for name in names+['ensemble','anchor_v5']:
        f=oof.loc[oof.fold==4]
        for sid,g in f.groupby('station_id'):
            error=g[name]-g.true_log;station.append({'model':name,'station_id':sid,'rmsle':float(np.sqrt(np.mean(error**2))),'log_sse':float(np.sum(error**2)),'mbe':float((np.expm1(g[name])-g.boardings).mean())})
    pd.DataFrame(station).to_csv(OUT/f'{label}_station_metrics.csv',index=False)
    return oof,report

def run(cfg):
    # A corrective restart consumes the original budget, never a fresh 120 min.
    start=float(os.environ.get('SINCON_V5_ORIGINAL_START',time.time()));deadline=start+cfg['budget_seconds'];contract(cfg)
    save_json(OUT/'run_status.json',{'status':'running','start':start,'deadline':deadline})
    from .audit import run_audit
    run_audit()
    features_for([1,2,3,4],cfg)
    jobs=[(OUT/f'fold_{f}/matrix.joblib',OUT/f'fold_{f}'/n,n,cfg['seed'],None) for f in range(1,5) for n in cfg['models']]
    train_jobs(jobs,cfg,deadline)
    oof,report=collect([1,2,3,4],cfg['models'],cfg,'step_b')
    # Preserve complete diagnostics on all validation rows.
    data=load_data(ROOT/'data/raw')
    for fold,g in oof.groupby('fold'):
        for name in cfg['models']+['ensemble']:
            dest=OUT/f'fold_{fold}'/name/'diagnostics';dest.mkdir(parents=True,exist_ok=True)
            diagnostics(data['train'],g,np.expm1(g[name].to_numpy()),pd.Timestamp(cfg['fold_origins'][fold-1]),dest)
    review_convergence(cfg,deadline)
    # Recollect if a bounded learning-rate diagnostic selected a better fit.
    oof,report=collect([1,2,3,4],cfg['models'],cfg,'step_b')
    features_for([0],cfg);jobs=[];plan={}
    for name in cfg['models']:
        best=[json.loads((OUT/f'fold_{f}'/name/'result.json').read_text())['best_iteration'] for f in [3,4]]
        it=final_iterations(best[0],best[1]);plan[name]={'fold3':best[0],'fold4':best[1],'final':it}
        for seed in cfg['final_seeds']:jobs.append((OUT/'final/matrix.joblib',OUT/'final'/name/str(seed),name,seed,it))
    save_json(OUT/'final_iteration_plan.json',plan);train_jobs(jobs,cfg,deadline)
    m=joblib.load(OUT/'final/matrix.joblib');p=np.column_stack([np.mean([np.load(OUT/'final'/n/str(s)/'prediction_log.npy') for s in cfg['final_seeds']],axis=0) for n in cfg['models']])
    pred=blend(p,buckets(m['query_x']),report['bucket_weights']);save_submission(m['valid'],pred,data['test'])
    save_json(OUT/'bundle_manifest.json',{'models':cfg['models'],'seeds':cfg['final_seeds'],'weights':report['bucket_weights'],'inputs':json.loads((OUT/'fingerprint.json').read_text())['inputs'],'seconds_before_reload':time.time()-start})
    # A fresh process reloads all checkpoints and recomputes features.
    subprocess.run([sys.executable,'-m','src.v5.pipeline','infer'],cwd=ROOT,check=True,timeout=max(1,deadline-time.time()))
    preserve();save_json(OUT/'run_status.json',{'status':'complete','seconds':time.time()-start,'within_budget':time.time()<=deadline})
    from .report import build_report
    build_report()

def review_convergence(cfg,deadline):
    reviews=[]
    for fold in [3,4]:
        for name in BASE:
            dest=OUT/f'fold_{fold}'/name
            result=json.loads((dest/'result.json').read_text());check=validate_convergence(name,result,cfg)
            reviews.append(dict(fold=fold,model=name,**check))
    save_json(OUT/'convergence_review.json',{'models':reviews,'interpretation':'Low best iterations are reported; longer training is not forced. Anchor already explains signal. Validation curves, effective sample diversity and retry scores are required to attribute cause.'})
    # At most one lower-rate diagnosis for each suspect model on Fold3/4.
    # Original outputs remain immutable; retry is diagnostic only, never silently selected.
    for row in reviews:
        if not row['below_diagnostic_minimum']:continue
        if deadline-time.time()<3600:row['retry']='skipped_to_reserve_final_budget';continue
        fold=row['fold'];name=row['model'];dest=OUT/'lr_diagnostics'/f'fold_{fold}'/name;dest.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-m','src.v5.pipeline','worker','--cache',str(OUT/f'fold_{fold}/matrix.joblib'),'--dest',str(dest),'--model',name,'--seed',str(cfg['seed']),'--lr-scale','0.5']
        with (dest/'train.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.time()))
        row['retry_result']=json.loads((dest/'result.json').read_text())
    save_json(OUT/'convergence_review.json',{'models':reviews,'policy':'Original CV fits kept; half-rate trials diagnose low iteration counts without forcing convergence or selecting on Public.'})


def save_submission(valid,pred,test):
    ordered=pd.Series(pred,index=valid.example_id).reindex(test.example_id).to_numpy();frame=test[['example_id']].copy();frame['boardings']=np.expm1(ordered)
    if len(frame)!=157194 or not frame.example_id.is_unique or not np.isfinite(frame.boardings).all() or (frame.boardings<0).any():raise ValueError('Submission integrity')
    path=OUT/'submission_ensemble_v5.csv';frame.to_csv(path,index=False,float_format='%.8f')
    save_json(OUT/'submission_validation.json',{'rows':len(frame),'ids_in_test_order':True,'finite_nonnegative':True,'mean':float(frame.boardings.mean()),'min':float(frame.boardings.min()),'max':float(frame.boardings.max()),'sha256':sha(path)})
    frame.assign(month=test.date.dt.strftime('%Y-%m')).groupby('month').boardings.agg(['mean','min','max']).to_csv(OUT/'test_monthly_distribution.csv')

def infer_one(args,cfg):
    m=joblib.load(args.cache);model=joblib.load(Path(args.dest)/'model.joblib');x=m['x']
    if args.model=='xgboost':
        # Checkpoints are portable CPU objects, but XGBoost 3.4.1's CPU
        # predictor differs from its CUDA predictor for this fitted model.
        # Restore the training backend when available; CPU remains runnable.
        hw=devices()
        device=f'cuda:{min(1,hw["cuda_count"]-1)}' if hw['cuda_count'] and cfg['device']!='cpu' else 'cpu'
        model.model.set_params(device=device);model.device=device
    if args.model in ['tabm_v5','st_itransformer']:
        model.device=model.device_name();model.model.to(model.device)
    if args.model=='st_itransformer':p=model.predict_log(x,m['keys'])
    else:p=model.predict_log(x)
    np.save(Path(args.dest)/'reload_prediction_log.npy',p)

def infer(cfg):
    from .features import make_features
    manifest=json.loads((OUT/'bundle_manifest.json').read_text());data=load_data(ROOT/'data/raw')
    if manifest['inputs']!={p.name:sha(p) for p in (ROOT/'data/raw').glob('*.csv')}:raise ValueError('Input hash mismatch')
    q=data['test'].sort_values(['date','station_id','hour']).reset_index(drop=True);x=make_features(data['train'],q,data['metaData'],cfg['final_cutoff']);keys=q[['date','station_id','hour']].copy();keys['origin']=pd.Timestamp(cfg['final_cutoff'])
    cache=OUT/'final/reload_features.joblib';joblib.dump({'x':x,'keys':keys},cache);logs=[];checks={}
    for name in manifest['models']:
        ps=[]
        for seed in manifest['seeds']:
            dest=OUT/'final'/name/str(seed)
            subprocess.run([sys.executable,'-m','src.v5.pipeline','infer-one','--cache',str(cache),'--dest',str(dest),'--model',name],cwd=ROOT,check=True)
            p=np.load(dest/'reload_prediction_log.npy');original=np.load(dest/'prediction_log.npy');ok=bool(np.allclose(p,original,rtol=1e-6,atol=1e-6));checks[f'{name}_{seed}']={'allclose':ok,'max_abs':float(np.max(abs(p-original)))}
            if not ok:raise ValueError('Reload differs')
            ps.append(p)
        logs.append(np.mean(ps,0))
    pred=blend(np.column_stack(logs),buckets(x),manifest['weights']);saved=pd.read_csv(OUT/'submission_ensemble_v5.csv');ordered=pd.Series(np.expm1(pred),index=q.example_id).reindex(saved.example_id).to_numpy()
    check=bool(np.allclose(ordered,saved.boardings,rtol=1e-6,atol=1e-6));reload=saved[['example_id']].copy();reload['boardings']=ordered;path=OUT/'submission_reload.csv';reload.to_csv(path,index=False,float_format='%.8f')
    save_json(OUT/'reload_validation.json',{'models':checks,'submission_allclose':check,'original_sha256':sha(OUT/'submission_ensemble_v5.csv'),'reload_sha256':sha(path),'byte_identical':sha(path)==sha(OUT/'submission_ensemble_v5.csv')})
    if not check:raise ValueError('Submission reload mismatch')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','worker','infer','infer-one']);p.add_argument('--cache',type=Path);p.add_argument('--dest');p.add_argument('--model');p.add_argument('--seed',type=int,default=2026);p.add_argument('--iterations',type=int);p.add_argument('--lr-scale',type=float,default=1.0)
    args=p.parse_args();cfg=json.loads((ROOT/'configs/ensemble_v5.json').read_text())
    if args.lr_scale!=1:
        for key in ['lightgbm','catboost','xgboost','tabm','st_itransformer']:cfg[key]['learning_rate']*=args.lr_scale
    if args.command=='worker':worker(args,cfg)
    elif args.command=='infer-one':infer_one(args,cfg)
    elif args.command=='infer':infer(cfg)
    else:run(cfg)
if __name__=='__main__':main()
