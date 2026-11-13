"""Bounded LightGBM search with temporal early stopping and frozen-round refits."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor,early_stopping,record_evaluation
from .features import load_data,make_features,training_matrix
from .models import Model,lightgbm_parameters
from .pipeline import save_json,environment,diagnostics,export_predictions
from .metrics import regression_metrics

ROOT=Path(__file__).resolve().parents[1]
CATEGORICAL=['station_id','canonical_line','hour','dow']

def clipped_log_rmsle(y_log,pred_log):
    """Targets are already log1p transformed; match deployed zero clipping."""
    return 'rmsle',float(np.sqrt(np.mean((np.maximum(np.asarray(pred_log),0)-np.asarray(y_log))**2))),False

def choose_common_round(curves,row_counts,max_rounds=1000):
    if len(curves)!=len(row_counts) or not curves or any(n<=0 for n in row_counts):
        raise ValueError('Nonempty curves and positive row counts required')
    common=min(max_rounds,*(len(c) for c in curves))
    if common<1:raise ValueError('No common observed iterations')
    values=np.array([c[:common] for c in curves],dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Invalid validation curve')
    pooled=np.sqrt(np.average(values**2,axis=0,weights=np.asarray(row_counts)))
    iteration=int(np.argmin(pooled))+1
    return iteration,float(pooled[iteration-1]),common,pooled

def candidate_parameters(seed=2026,trials=16):
    base=dict(learning_rate=.055,num_leaves=63,min_child_samples=80,reg_lambda=3.,
              reg_alpha=0.,max_depth=-1,colsample_bytree=.9,subsample=.9,subsample_freq=1,min_split_gain=0.)
    variants=[{},dict(learning_rate=.03,min_child_samples=160,reg_lambda=10.,max_depth=10),
              dict(learning_rate=.08,num_leaves=31,min_child_samples=160,reg_lambda=10.,max_depth=8),
              dict(learning_rate=.03,num_leaves=127,min_child_samples=160,reg_lambda=10.,max_depth=12)]
    candidates=[dict(base,**v) for v in variants]
    rng=np.random.default_rng(seed)
    while len(candidates)<trials:
        proposed=dict(base,learning_rate=float(rng.choice([.025,.04,.055,.08])),
                      num_leaves=int(rng.choice([31,63,95,127])),
                      min_child_samples=int(rng.choice([80,160,320])),
                      reg_lambda=float(rng.choice([3.,10.,30.])),
                      reg_alpha=float(rng.choice([0.,.1,1.])),
                      max_depth=int(rng.choice([8,10,12,-1])),
                      colsample_bytree=float(rng.choice([.75,.9,1.])),
                      subsample=float(rng.choice([.75,.9])),
                      min_split_gain=float(rng.choice([0.,.01])))
        if proposed not in candidates:candidates.append(proposed)
    return candidates[:trials]

def prepare_fold(train,meta,origin,cfg):
    origin=pd.Timestamp(origin)
    history=train.loc[train.date<=origin]
    valid=train.loc[(train.date>origin)&(train.date<=origin+pd.Timedelta(days=cfg['horizon_days']))].copy().reset_index(drop=True)
    if valid.empty or valid.date.max()!=origin+pd.Timedelta(days=cfg['horizon_days']):
        raise ValueError('Complete 123-day validation required')
    known=valid.station_id.isin(history.station_id.unique()).to_numpy()
    x,y,anchors=training_matrix(train,meta,origin,cfg['horizon_days'],cfg['origin_stride_days'],cfg['max_training_rows'],cfg['seed'])
    xv=make_features(train,valid,meta,origin)
    assert x.columns.equals(xv.columns)
    assert max(pd.Timestamp(a['label_end']) for a in anchors)<=origin
    return {'origin':origin,'valid':valid,'known':known,'x':x,'y':y,'xv':xv,'anchors':anchors}

def search_development(train,meta,cfg,output,trials,max_rounds,patience):
    if max_rounds!=1000:raise ValueError('User requested a hard cap of 1000 rounds')
    dev_end=pd.Timestamp(cfg['fold_origins'][1])+pd.Timedelta(days=cfg['horizon_days'])
    development=train.loc[train.date<=dev_end].copy()
    folds=[prepare_fold(development,meta,origin,cfg) for origin in cfg['fold_origins'][:2]]
    candidates=candidate_parameters(cfg['seed'],trials)
    design={'optimizer':'LightGBM gradient/tree boosting, not AdamW',
            'max_boosting_rounds':max_rounds,'early_stopping_rounds':patience,'min_delta':1e-5,
            'metric':'RMSLE after deployed zero clipping in log space','trials':trials,
            'candidate_parameters':candidates,'development_label_end':str(dev_end.date()),
            'selection':'minimum row-weighted pooled log-MSE on common observed iteration prefix',
            'station_policy':'validation stations observed on or before each origin; actual test has no unseen station',
            'global_optimum_guaranteed':False,
            'folds':[{'origin':str(f['origin'].date()),'all_rows':len(f['valid']),'selection_rows':int(f['known'].sum()),'excluded_cold_rows':int((~f['known']).sum()),'training_rows':len(f['y'])} for f in folds]}
    save_json(output/'search_design.json',design)
    rows=[];best=None
    for trial,parameters in enumerate(candidates,1):
        t=time.perf_counter();curves=[];fold_info=[]
        folder=output/'search'/f'trial_{trial:02d}';folder.mkdir(parents=True,exist_ok=True)
        print(f'Trial {trial}/{trials}: leaves={parameters["num_leaves"]}, lr={parameters["learning_rate"]}, L2={parameters["reg_lambda"]}',flush=True)
        for i,f in enumerate(folds,1):
            fit_cfg=dict(cfg,lightgbm_params=dict(parameters,n_estimators=max_rounds))
            params=lightgbm_parameters(fit_cfg);params['metric']='None'
            model=LGBMRegressor(**params)
            mask=f['known'];yv=np.log1p(f['valid'].loc[mask,'boardings'].to_numpy(dtype=np.float64))
            history={}
            model.fit(f['x'],np.log1p(f['y']),categorical_feature=CATEGORICAL,
                      eval_set=[(f['x'],np.log1p(f['y'])),(f['xv'].loc[mask],yv)],
                      eval_names=['training','validation'],eval_metric=clipped_log_rmsle,
                      callbacks=[record_evaluation(history),early_stopping(patience,first_metric_only=True,verbose=False,min_delta=1e-5)])
            curve=history['validation']['rmsle'];curves.append(curve)
            pd.DataFrame({'iteration':np.arange(1,len(curve)+1),'training_rmsle':history['training']['rmsle'],'validation_rmsle':curve}).to_csv(folder/f'fold_{i}_learning_curve.csv',index=False)
            info={'fold':i,'best_iteration':model.best_iteration_,'recorded_iterations':len(curve),
                  'retained_trees':model.n_estimators_,'early_stopped_before_cap':len(curve)<max_rounds,
                  'cap_reached':len(curve)==max_rounds,'best_validation_rmsle':model.best_score_['validation']['rmsle']}
            fold_info.append(info)
            print(f'  dev{i}: best={model.best_iteration_}, observed={len(curve)}, RMSLE={info["best_validation_rmsle"]:.6f}',flush=True)
            del model;gc.collect()
        iteration,value,common,pooled=choose_common_round(curves,[int(f['known'].sum()) for f in folds],max_rounds)
        pd.DataFrame({'iteration':np.arange(1,common+1),'pooled_rmsle':pooled}).to_csv(folder/'common_validation_curve.csv',index=False)
        row={'trial':trial,'selected_iteration':iteration,'pooled_development_rmsle':value,'common_observed_iterations':common,'seconds':time.perf_counter()-t,**parameters}
        rows.append(row)
        save_json(folder/'trial.json',{'summary':row,'folds':fold_info})
        pd.DataFrame(rows).to_csv(output/'search_results.csv',index=False)
        if best is None or (value,iteration)<(best['pooled_development_rmsle'],best['selected_iteration']):
            best={**row,'parameters':parameters,'folds':fold_info}
        print(f'  common selection={iteration}, pooled RMSLE={value:.6f} ({row["seconds"]:.1f}s)',flush=True)
    save_json(output/'selected_plan.json',{'selection_frozen_before_fold3':True,'selection_uses_fold3_labels':False,
                                         'existing_fold3_previously_analyzed':True,'global_optimum_guaranteed':False,
                                         'search_design':design,'best_trial':best})
    return best,folds

def fit_frozen_fold(f,cfg,path,train):
    path.mkdir(parents=True,exist_ok=True)
    model=Model('lightgbm_large',cfg).fit(f['x'],f['y'])
    pred=model.predict(f['xv'])
    frame=f['valid'][['date','station_id','hour','boardings']].copy()
    frame['horizon']=(frame.date-f['origin']).dt.days
    frame['known_at_origin']=f['known'];frame['lightgbm_large']=pred;frame['ensemble']=pred
    frame.to_csv(path/'oof.csv.gz',index=False,compression='gzip')
    save_json(path/'training_origins.json',f['anchors'])
    summary=diagnostics(train,f['valid'],pred,f['origin'],path)
    known=regression_metrics(f['valid'].loc[f['known'],'boardings'],pred[f['known']])
    save_json(path/'known_station_metrics.json',known)
    save_json(path/'fit_metadata.json',{'actual_iterations':model.estimator.n_estimators_,'rounds_frozen_from_development':True})
    return frame,summary,known

def run(args):
    started=time.perf_counter();out=args.output
    if (out/'model_bundle.joblib').exists():raise FileExistsError('Choose a new output directory to preserve the previous tuned run')
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(args.config.read_text())
    data=load_data(args.data_dir);train=data['train'];meta=data['metaData']
    fingerprint={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data_dir.glob('*.csv')}
    save_json(out/'input_sha256.json',fingerprint);save_json(out/'environment.json',environment())
    best,dev_folds=search_development(train,meta,cfg,out,args.trials,args.max_rounds,args.patience)
    selected=copy.deepcopy(cfg)
    selected['models']=['lightgbm_large']
    selected['lightgbm_params']=dict(best['parameters'],n_estimators=best['selected_iteration'])
    selected['ensemble_policy']={'method':'primary_constrained_log_mse','primary_model':'lightgbm_large','primary_min_weight':1.,'bias_correction':False}
    selected['tuning']={'method':'16-candidate bounded development search with early stopping','max_rounds':1000,'patience':args.patience,'min_delta':1e-5,'actual_trials':args.trials,'selected_iteration':best['selected_iteration'],'selected_trial':best['trial'],'search_output':str(out)}
    selected['evaluation_context']={'status':'reused_validation_after_prior_diagnostics','policy_selection_has_seen_audit':True,'weight_fit_uses_audit':False,'hyperparameter_selection_uses_audit':False,'note':'Hyperparameters and common boosting iteration selected using first two development folds, known stations only. Fold 3 already examined in earlier work; reused comparison only.'}
    save_json(out/'config_resolved.json',selected)
    # Development refits are needed because an early-stopped booster may retain
    # fewer trees than the common selected round recorded in its patience window.
    metrics=[];fold_records=[];known_scores=[]
    for i,f in enumerate(dev_folds,1):
        frame,summary,known=fit_frozen_fold(f,selected,out/f'fold_{i}',train)
        metrics.append({'fold':i,'role':'development','model':'lightgbm_large',**summary})
        known_scores.append((len(frame.loc[frame.known_at_origin]),known['rmsle']))
        fold_records.append({'fold':i,'origin':str(f['origin'].date()),'rows':len(frame),'selection_rows':int(f['known'].sum())})
    confirmed=float(np.sqrt(sum(n*v*v for n,v in known_scores)/sum(n for n,v in known_scores)))
    if abs(confirmed-best['pooled_development_rmsle'])>2e-7:raise AssertionError('Fixed-round refits did not reproduce selected development score')
    del dev_folds;gc.collect()
    # The third fold is accessed for evaluation only after selection is frozen.
    audit=prepare_fold(train,meta,selected['fold_origins'][2],selected)
    frame,summary,known=fit_frozen_fold(audit,selected,out/'fold_3',train)
    save_json(out/'audit_summary.json',summary)
    metrics.append({'fold':3,'role':'reused_comparison','model':'lightgbm_large',**summary})
    fold_records.append({'fold':3,'origin':str(audit['origin'].date()),'rows':len(frame),'selection_rows':0})
    pd.DataFrame(metrics).to_csv(out/'cv_scores.csv',index=False);save_json(out/'cv_folds.json',fold_records)
    reference=frame.lightgbm_large.to_numpy()
    print(f'Frozen-round comparison RMSLE={summary["rmsle"]:.9f}; round={best["selected_iteration"]}',flush=True)
    del audit;gc.collect()
    cutoff=train.date.max()
    x,y,anchors=training_matrix(train,meta,cutoff,selected['horizon_days'],selected['origin_stride_days'],selected['max_training_rows'],selected['seed'])
    xt=make_features(train,data['test'],meta,cutoff)
    model=Model('lightgbm_large',selected).fit(x,y)
    pred=model.predict(xt)
    plan={'weights':{'lightgbm_large':1.},'bias':0.,'blend_space':'log1p','bias_space':'log1p',
          'retained':['lightgbm_large'],'excluded':{},'metric':'rmsle','fit_folds':[1,2],
          'evaluation_context':selected['evaluation_context']}
    bundle={'config':selected,'ensemble':plan,'models':{'lightgbm_large':model},'features':x.columns.tolist(),
            'training_cutoff':str(cutoff.date()),'input_sha256':fingerprint}
    joblib.dump(bundle,out/'model_bundle.joblib',compress=3)
    model.estimator.booster_.save_model(str(out/'lightgbm_tuned.txt'))
    save_json(out/'ensemble_plan.json',plan);save_json(out/'final_training_origins.json',anchors)
    export_predictions(data,pred,out,reference)
    loaded=joblib.load(out/'model_bundle.joblib')
    restored=loaded['models']['lightgbm_large'].predict(xt)
    np.testing.assert_array_equal(pred,restored)
    export_predictions(data,restored,out/'reloaded',reference)
    old_hash=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()
    assert old_hash==hashlib.sha256((out/'reloaded/submission.csv').read_bytes()).hexdigest()
    save_json(out/'verification.json',{'reload_bitwise_identical':True,'submission_sha256':old_hash,
                                      'selected_development_rmsle':best['pooled_development_rmsle'],
                                      'refitted_development_rmsle':confirmed,'actual_final_trees':model.estimator.n_estimators_,
                                      'hard_cap_satisfied':model.estimator.n_estimators_<=1000,'final_fit_uses_frozen_rounds':True,
                                      'fold3_not_used_for_early_stopping_or_parameter_selection':True})
    save_json(out/'run_summary.json',{'status':'completed','elapsed_seconds':time.perf_counter()-started,
                                    'models_deployed':['lightgbm_large'],'feature_count':len(x.columns),
                                    'final_training_rows':len(y),'max_search_rounds':1000,
                                    'selected_rounds':best['selected_iteration'],'selected_trial':best['trial'],
                                    'trials':args.trials,'development_rmsle':confirmed,'audit_rmsle':summary['rmsle'],
                                    'previous_lightgbm_audit_rmsle':.2539725321304833,'global_optimum_guaranteed':False,
                                    'submission_rows':len(pred),'optimizer':'gradient tree boosting; not AdamW'})
    print(f'Completed: {out}; selected rounds={best["selected_iteration"]}; RMSLE={summary["rmsle"]:.9f}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/raw')
    p.add_argument('--config',type=Path,default=ROOT/'configs/three_tree.json')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/lightgbm_tuned')
    p.add_argument('--trials',type=int,default=16)
    p.add_argument('--max-rounds',type=int,default=1000)
    p.add_argument('--patience',type=int,default=100)
    args=p.parse_args()
    if args.trials<1 or args.patience<1:raise ValueError('Trials and patience must be positive')
    run(args)

if __name__=='__main__':main()
