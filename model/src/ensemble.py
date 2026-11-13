"""Development-OOF convex blends: primary-constrained log-MSE or legacy greedy."""
import numpy as np
from .ensemble_v2 import convex_weights, evaluate_ensemble
import pandas as pd
from .metrics import score

def fit_ensemble(y,predictions,metric='rmsle',correlation_cutoff=0.95,policy=None):
    if policy and policy.get('method')=='primary_constrained_log_mse':
        if metric!='rmsle':raise ValueError('Constrained log-MSE blending requires RMSLE')
        return fit_primary_ensemble(y,predictions,policy)
    if policy:raise ValueError(f'Unknown ensemble policy: {policy}')
    names=list(predictions)
    scores={n:score(y,predictions[n],metric) for n in names}
    space={n:np.log1p(predictions[n]) if metric=='rmsle' else np.asarray(predictions[n]) for n in names}
    target=np.log1p(y) if metric=='rmsle' else y
    invert=lambda values:np.maximum(0,np.expm1(values)) if metric=='rmsle' else np.maximum(0,values)
    residual=pd.DataFrame({n:space[n]-target for n in names})
    corr=residual.corr()
    ranked=sorted(names,key=scores.get)
    retained=[];excluded={}
    for name in ranked:
        redundant=[other for other in retained if np.array_equal(space[name],space[other]) or corr.loc[name,other]>correlation_cutoff]
        if redundant:excluded[name]={'reason':'residual_correlation_above_threshold','with':redundant}
        else:retained.append(name)
    weights={n:0.0 for n in retained};weights[retained[0]]=1.0
    blend=space[retained[0]].copy();best=score(y,invert(blend),metric)
    for step in [0.25,0.1,0.05,0.02,0.01]:
        for _ in range(40):
            options=[]
            for name in retained:
                proposed=(1-step)*blend+step*space[name]
                options.append((score(y,invert(proposed),metric),name,proposed))
            trial,name,proposed=min(options,key=lambda item:item[0])
            if trial>=best-1e-8:break
            weights={n:w*(1-step) for n,w in weights.items()};weights[name]+=step
            blend=proposed;best=trial
    # A scalar adjustment is estimated only on development OOF, audit is untouched.
    bias=float(np.mean(target-blend)) if metric=='rmsle' else float(np.median(y-blend)) if metric=='mae' else 0.0
    corrected=invert(blend+bias)
    if score(y,corrected,metric)>=best:bias=0.0
    return {'weights':weights,'bias':bias,'bias_space':'log1p' if metric=='rmsle' else 'raw','blend_space':'log1p' if metric=='rmsle' else 'raw','retained':retained,'excluded':excluded,'individual_scores':scores,'development_score':score(y,invert(blend+bias),metric),'metric':metric},corr

def fit_primary_ensemble(y,predictions,policy):
    """Minimize development log-MSE with a fixed minimum primary-model share.

    The caller supplies development OOF only. Correlation filtering and intercept
    calibration are disabled; the primary model cannot be silently removed.
    """
    from scipy.optimize import minimize
    names=list(predictions)
    primary=policy['primary_model']
    minimum=float(policy['primary_min_weight'])
    if primary not in names:raise ValueError('Primary model missing from predictions')
    if not .5<=minimum<=1:raise ValueError('Primary minimum weight must be in [0.5, 1]')
    if policy.get('bias_correction',False):raise ValueError('This policy disables bias correction')
    truth=np.asarray(y,dtype=np.float64)
    if truth.ndim!=1 or len(truth)==0 or not np.isfinite(truth).all() or np.any(truth<0):
        raise ValueError('Expected finite nonnegative one-dimensional targets')
    arrays=[np.asarray(predictions[name],dtype=np.float64) for name in names]
    if any(p.shape!=truth.shape or not np.isfinite(p).all() or np.any(p<0) for p in arrays):
        raise ValueError('Predictions must match targets and be finite and nonnegative')
    residual=np.column_stack([np.log1p(p)-np.log1p(truth) for p in arrays])
    gram=residual.T@residual/len(truth)
    ix=names.index(primary)
    lower=np.zeros(len(names));lower[ix]=minimum
    start=lower+(1-minimum)/len(names)
    result=minimize(lambda w:float(w@gram@w),start,jac=lambda w:2*gram@w,
                    method='SLSQP',bounds=[(float(lo),1.) for lo in lower],
                    constraints=[{'type':'eq','fun':lambda w:w.sum()-1.,
                                  'jac':lambda w:np.ones_like(w)}],
                    options={'ftol':1e-13,'maxiter':500})
    if not result.success:raise RuntimeError(f'Ensemble optimization failed: {result.message}')
    w=np.asarray(result.x)
    if np.any(w<lower-1e-10) or abs(w.sum()-1)>1e-10:
        raise RuntimeError('Ensemble optimizer violated convex constraints')
    # SLSQP may return a boundary value a few ulps below its lower bound.
    w=np.maximum(w,lower)
    w[ix]+=1-w.sum()
    scores={name:score(truth,p,'rmsle') for name,p in zip(names,arrays)}
    plan={'weights':dict(zip(names,map(float,w))),'bias':0.,'bias_space':'log1p',
          'blend_space':'log1p','retained':names,'excluded':{},
          'individual_scores':scores,'metric':'rmsle','policy':dict(policy),
          'optimizer':'SLSQP: convex quadratic log-MSE',
          'correlation_filter_applied':False,'bias_correction_applied':False}
    plan['development_score']=score(truth,predict_ensemble(predictions,plan),'rmsle')
    return plan,pd.DataFrame(residual,columns=names).corr()

def predict_ensemble(predictions,plan):
    active=[(name,weight) for name,weight in plan['weights'].items() if weight>0]
    if len(active)==1 and active[0][1]==1.0 and plan['bias']==0.0:
        # Preserve exact standalone predictions, avoiding log/expm1 roundoff.
        return np.maximum(0,np.asarray(predictions[active[0][0]],dtype=np.float64))
    log_space=plan.get('blend_space')=='log1p'
    pred=sum(weight*(np.log1p(predictions[name]) if log_space else np.asarray(predictions[name])) for name,weight in plan['weights'].items() if weight>0)
    return np.maximum(0,np.expm1(pred+plan['bias']) if log_space else pred+plan['bias'])
