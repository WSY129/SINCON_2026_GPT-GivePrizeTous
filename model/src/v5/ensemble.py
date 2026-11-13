"""Restricted convex blends; specialist eligibility is a hard constraint."""
import numpy as np
from scipy.optimize import minimize
BUCKETS=['new_stations','gyeyang_gyulhyeon','high_spike','late_or_low','regular']
DIRECT={'lightgbm','catboost_cpu','xgboost'}

def buckets(x):
    return np.select([x.station_id.isin([3107,3108,3109]),x.station_id.isin([3110,3111]),x.window_spike_amp>=.65,(x.hour>=21)|(x.anchor_v5<3.2)],[0,1,2,3],default=4).astype('int8')

def fold_row_weights(folds,selection):
    folds=np.asarray(folds);out=np.zeros(len(folds));mass={1:.15,2:.15,3:.35,4:.35}
    present=[f for f in sorted(set(folds)) if selection=='all' or f>=3]
    if not present:present=sorted(set(folds))
    for f in present:
        ix=folds==f;out[ix]=(mass[f] if selection=='all' else 1.)/ix.sum()
    return out/out.sum()

def fit_weights(y,p,folds,names,bucket='global',prior=None,shrink=0,direct_min=.20,spatial_max=.45):
    y=np.asarray(y,float);p=np.asarray(p,float);n=len(names)
    if not len(y):
        if prior is None:raise ValueError('No fitting rows')
        # Project prior onto this bucket's own constraints, never copy unmasked.
        y=np.zeros(1);p=np.zeros((1,n));folds=np.array([3]);shrink=max(shrink,1)
    direct_min=direct_min if bucket in ('global','regular') else 0.0
    direct=np.array([name in DIRECT for name in names],float)
    bounds=[(0,0) if name=='low_demand_specialist' and bucket!='late_or_low' else (0,spatial_max) if name=='st_itransformer' and bucket=='new_stations' else (0,1) for name in names]
    initial=direct/direct.sum() if direct.sum() else np.ones(n)/n
    if not direct.sum() and direct_min>0:raise ValueError('Direct tree floor infeasible')
    prior=initial if prior is None else np.asarray(prior,float)
    residual=p-y[:,None]
    grams=[residual.T@(residual*fold_row_weights(folds,part)[:,None]) for part in ['all','recent']]
    def fun(w):return sum(a*(w@g@w) for a,g in zip([.35,.65],grams))+shrink*np.sum((w-prior)**2)
    def jac(w):return sum(2*a*(g@w) for a,g in zip([.35,.65],grams))+2*shrink*(w-prior)
    constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)}, {'type':'ineq','fun':lambda w:w@direct-direct_min,'jac':lambda w:direct}]
    r=minimize(fun,initial,jac=jac,bounds=bounds,constraints=constraints,method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
    if not r.success:raise RuntimeError(r.message)
    w=r.x.copy();w[np.abs(w)<1e-12]=0
    if abs(w.sum()-1)>1e-7 or w.min()<-1e-9 or w@direct<direct_min-1e-7:raise AssertionError('Blend constraint')
    for j,(lo,hi) in enumerate(bounds):
        if not lo-1e-9<=w[j]<=hi+1e-9:raise AssertionError('Eligibility/cap violation')
    return w

def score(y,p):return float(np.sqrt(np.mean((np.asarray(y)-p)**2))) if len(y) else None

def summary(y,p,fold):
    recent=fold>=3;last=fold==4
    return {'oof_rmsle':score(y,p),'weighted_rmsle':float(np.sqrt(np.sum(fold_row_weights(fold,'all')*(y-p)**2))),'fold34_rmsle':score(y[recent],p[recent]),'fold4_rmsle':score(y[last],p[last])}

def blend(p,b,weights):
    out=np.empty(len(p))
    for k,name in enumerate(BUCKETS):out[b==k]=p[b==k]@np.asarray(weights[name])
    return out

def evaluate(frame,names):
    p=frame[names].to_numpy();y=frame.true_log.to_numpy();f=frame.fold.to_numpy();b=frame.bucket.to_numpy()
    global_w=fit_weights(y,p,f,names);final={}
    for k,name in enumerate(BUCKETS):
        ix=b==k;final[name]=fit_weights(y[ix],p[ix],f[ix],names,name,global_w,.0005).tolist()
    pred=blend(p,b,final)
    # Forward check uses disjoint dates. It does NOT choose a fallback on Fold 4.
    fit=(f==3)&~frame.date.isin(frame.loc[f==4,'date']).to_numpy();test=f==4
    wg=fit_weights(y[fit],p[fit],f[fit],names);cross=[];purged=p@wg
    for k,name in enumerate(BUCKETS):
        ix=fit&(b==k);va=test&(b==k);w=fit_weights(y[ix],p[ix],f[ix],names,name,wg,.0005)
        purged[va]=p[va]@w;cross.append({'bucket':name,'rows':int(va.sum()),'rmsle':score(y[va],purged[va]),'weights':w.tolist()})
    lofo=np.zeros(len(y))
    for leave in sorted(set(f)):
        va=f==leave;tr=(f!=leave)&~frame.date.isin(frame.loc[va,'date']).to_numpy()
        w=fit_weights(y[tr],p[tr],f[tr],names);lofo[va]=p[va]@w
    check=additional_checks(frame,names)
    report={'additional_checks':check,'models':{n:summary(y,p[:,i],f) for i,n in enumerate(names)},'global':summary(y,p@global_w,f),'ensemble':summary(y,pred,f),'global_weights':dict(zip(names,global_w.tolist())),'bucket_weights':final,'model_order':names,'objective':'0.35 * four-fold weighted log MSE + 0.65 * Fold3+4 pooled log MSE; per-bucket normalized within-fold; tiny L2 prior shrinkage','direct_tree_min_weight':.20,'new_station_spatial_cap':.45,'purged_chronological_check':{'fit_rows':int(fit.sum()),'test_rows':int(test.sum()),'global_rmsle':score(y[test],(p@wg)[test]),'raw_bucket_rmsle':score(y[test],purged[test]),'buckets':cross},'lofo_global_rmsle':score(y,lofo),'limitations':'OOF weight-fit scores are not independent. Forward checks purge repeated dates but reuse early stopping/feature design; LOFO may use later folds. No fallback is selected from Fold4 truth.'}
    return pred,report,lofo


def additional_checks(frame,names):
    y=frame.true_log.to_numpy();p=frame[names].to_numpy();f=frame.fold.to_numpy()
    train=(f==3)&~frame.date.isin(frame.loc[f==4,'date']).to_numpy();test=f==4
    comparison={}
    for floor in [0,.2]:
        w=fit_weights(y[train],p[train],f[train],names,direct_min=floor)
        comparison[str(floor)]={'weights':w.tolist(),'purged_fold4_rmsle':score(y[test],p[test]@w)}
    # Forward date blocks: each training meta row is strictly earlier than
    # the held-out block, with a 7-day embargo. This is meta-level validation;
    # base early stopping and feature design are not nested retrained here.
    import pandas as pd
    dates=pd.to_datetime(frame.date);unique=np.sort(dates.unique());blocks=[]
    for block in np.array_split(unique,5)[1:]:
        tr=(dates<block[0]-np.timedelta64(7,'D')).to_numpy();va=dates.isin(block).to_numpy()
        if tr.sum()<1000:continue
        for floor in [0,.2]:
            w=fit_weights(y[tr],p[tr],f[tr],names,direct_min=floor)
            blocks.append({'start':str(block[0]),'rows':int(va.sum()),'direct_floor':floor,'rmsle':score(y[va],p[va]@w)})
    return {'direct_floor_purged_comparison':comparison,'forward_date_blocks':blocks,'note':'Meta validation only; base model folds overlap and early stopping reused.'}
