"""Convex log-space OOF blending without an intercept, including LOFO checks."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def row_weights(folds,fold_weights,exclude=None):
    weights=np.zeros(len(folds),dtype='float64')
    for fold,weight in enumerate(fold_weights,1):
        if fold==exclude:continue
        mask=folds==fold
        if mask.any():weights[mask]=weight/mask.sum()
    return weights/weights.sum()

def convex_weights(truth,predictions,weights=None):
    y=np.asarray(truth,dtype='float64');p=np.asarray(predictions,dtype='float64')
    if p.ndim!=2 or len(y)!=len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Invalid finite log prediction matrix')
    weights=np.full(len(y),1/len(y)) if weights is None else np.asarray(weights)/np.sum(weights)
    residual=p-y[:,None];gram=residual.T@(weights[:,None]*residual)
    n=p.shape[1]
    result=minimize(lambda w:float(w@gram@w),np.full(n,1/n),jac=lambda w:2*gram@w,method='SLSQP',bounds=[(0,1)]*n,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)}],options={'ftol':1e-13,'maxiter':1000})
    if not result.success:raise RuntimeError(result.message)
    w=np.maximum(0,result.x);w/=w.sum()
    return w

def evaluate_ensemble(truth,predictions,folds,names,fold_weights):
    y=np.asarray(truth);p=np.asarray(predictions);folds=np.asarray(folds)
    weights=row_weights(folds,fold_weights);w=convex_weights(y,p,weights)
    per_model={name:{'oof_rmsle':float(np.sqrt(np.mean((p[:,i]-y)**2))), 'weighted_rmsle':float(np.sqrt(np.sum(weights*(p[:,i]-y)**2))), 'fold34_rmsle':float(np.sqrt(np.mean((p[folds>=3,i]-y[folds>=3])**2)))} for i,name in enumerate(names)}
    lofo=np.zeros(len(y));reference=np.zeros(len(y));checks=[]
    for fold in range(1,5):
        fitweights=row_weights(folds,fold_weights,exclude=fold)
        wf=convex_weights(y,p,fitweights);ix=folds==fold
        best=int(np.argmin(np.sum(fitweights[:,None]*(p-y[:,None])**2,axis=0)))
        lofo[ix]=p[ix]@wf;reference[ix]=p[ix,best]
        checks.append({'held_out_fold':fold,'weights':dict(zip(names,wf.tolist())),'training_selected_best_model':names[best], 'ensemble_rmsle':float(np.sqrt(np.mean((lofo[ix]-y[ix])**2))), 'selected_single_rmsle':float(np.sqrt(np.mean((reference[ix]-y[ix])**2)))})
    pred=p@w
    report={'method':'SLSQP convex mean in log1p space; no intercept','weights':dict(zip(names,w.tolist())),'models':per_model,'ensemble':{'oof_rmsle':float(np.sqrt(np.mean((pred-y)**2))),'weighted_rmsle':float(np.sqrt(np.sum(weights*(pred-y)**2))),'fold34_rmsle':float(np.sqrt(np.mean((pred[folds>=3]-y[folds>=3])**2)))},'lofo':checks,'lofo_weighted_rmsle':float(np.sqrt(np.sum(weights*(lofo-y)**2))),'lofo_selected_single_weighted_rmsle':float(np.sqrt(np.sum(weights*(reference-y)**2))),'evaluation_limit':'Validation dates overlap between folds; LOFO is a sensitivity check, not an independent future holdout. Base models use their fold validation for early stopping.'}
    corr=pd.DataFrame(y[:,None]-p,columns=names).corr()
    return pred,lofo,report,corr
