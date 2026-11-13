"""Simplex log blends with past-feature gates and honest cross-check labels."""
import numpy as np
from scipy.optimize import minimize

BUCKETS=['new_stations','high_spike','late_or_low','regular']

def buckets(x):
    return np.select([x.station_id.isin([3107,3108,3109]),x.past_spike_amp>=.75,(x.hour>=21)|(x.anchor_v4<3.2)],[0,1,2],default=3).astype('int8')

def fit_weights(y,p,prior=None,shrink=0):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);n=p.shape[1]
    if not len(y):return np.ones(n)/n if prior is None else prior.copy()
    residual=p-y[:,None];gram=residual.T@residual/len(y)
    prior=np.ones(n)/n if prior is None else prior
    fun=lambda w:float(w@gram@w+shrink*np.sum((w-prior)**2))
    jac=lambda w:2*gram@w+2*shrink*(w-prior)
    r=minimize(fun,prior,jac=jac,bounds=[(0,1)]*n,constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)},method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
    if not r.success:raise RuntimeError(r.message)
    w=np.maximum(0,r.x);return w/w.sum()

def score(y,p):return float(np.sqrt(np.mean((np.asarray(y)-p)**2)))

def summary(y,p,fold):
    return {'oof_rmsle':score(y,p),'weighted_rmsle':float(np.sqrt(np.average((y-p)**2,weights=np.array([.15,.15,.35,.35])[fold-1]))),'fold34_rmsle':score(y[fold>=3],p[fold>=3]),'fold4_rmsle':score(y[fold==4],p[fold==4])}

def evaluate(frame,names):
    p=frame[names].to_numpy();y=frame.true_log.to_numpy();f=frame.fold.to_numpy();b=frame.bucket.to_numpy();recent=f>=3
    global_w=fit_weights(y[recent],p[recent]);all_w=fit_weights(y,p)
    fit=(f==3)&~frame.date.isin(frame.loc[f==4,'date']).to_numpy();test=f==4
    wg=fit_weights(y[fit],p[fit]);selected=np.argmin(np.mean((p[fit]-y[fit,None])**2,axis=0))
    final={};cross=[];pg=p@wg;purged=pg.copy()
    for k,name in enumerate(BUCKETS):
        wf=fit_weights(y[fit&(b==k)],p[fit&(b==k)],wg,.0005)
        ix=test&(b==k);local=score(y[ix],p[ix]@wf);global_score=score(y[ix],pg[ix]);single=score(y[ix],p[ix,selected])
        fallback=local>global_score
        final[name]=(global_w if fallback else fit_weights(y[recent&(b==k)],p[recent&(b==k)],global_w,.0005)).tolist()
        purged[ix]=p[ix]@wf
        cross.append({'bucket':name,'rows':int(ix.sum()),'bucket_rmsle':local,'global_rmsle':global_score,'training_selected_single':names[selected],'single_rmsle':single,'fallback_global':fallback,'fit_weights':wf.tolist()})
    pred=blend(p,b,final)
    lofo=np.zeros(len(y))
    for leave in sorted(set(f)):
        tr=f!=leave;va=~tr;w=fit_weights(y[tr],p[tr]);lofo[va]=p[va]@w
    report={'models':{n:summary(y,p[:,i],f) for i,n in enumerate(names)},'global':summary(y,p@global_w,f),'ensemble':summary(y,pred,f),'global_weights':dict(zip(names,global_w.tolist())),'all_fold_weights':dict(zip(names,all_w.tolist())),'bucket_weights':final,'model_order':names,'purged_chronological_check':{'fit_rows':int(fit.sum()),'test_rows':int(test.sum()),'global_rmsle':score(y[test],pg[test]),'raw_bucket_rmsle':score(y[test],purged[test]),'training_selected_single':names[selected],'single_rmsle':score(y[test],p[test,selected]),'buckets':cross},'lofo_global_rmsle':score(y,lofo),'limitations':'Fold 3+4 blend scores fit weights on those labels. Bucket fallback is selected using Fold 4; its selected score is not independent. Purged checks remove repeated dates but still reuse early-stopping and feature-design decisions.'}
    return pred,report,lofo

def blend(p,b,weights):
    out=np.empty(len(p))
    for k,name in enumerate(BUCKETS):out[b==k]=p[b==k]@np.asarray(weights[name])
    return out
