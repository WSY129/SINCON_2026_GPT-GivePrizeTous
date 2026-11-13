"""Recent-regime convex blend, with all-fold and purged sensitivity checks."""
import numpy as np
import pandas as pd
from .ensemble_v2 import convex_weights, row_weights, evaluate_ensemble as evaluate_v2

def rmsle_log(y, p, weights=None):
    return float(np.sqrt(np.average((np.asarray(y)-np.asarray(p))**2, weights=weights)))

def evaluate_ensemble(truth, predictions, folds, names, fold_weights, dates=None):
    y=np.asarray(truth); p=np.asarray(predictions); folds=np.asarray(folds)
    all_pred, lofo, report, corr = evaluate_v2(y,p,folds,names,fold_weights)
    recent=folds>=3; w=convex_weights(y[recent],p[recent]); pred=p@w
    all_comparison={'weights':report['weights'],'ensemble':report['ensemble']}
    report['all_fold_comparison']=all_comparison
    report['weights']=dict(zip(names,w.tolist()))
    report['selection_rows']=int(recent.sum())
    report['selection_scope']='Fold 3 + Fold 4 pooled OOF, equal row weights'
    report['ensemble']={
        'oof_rmsle':rmsle_log(y,pred),
        'weighted_rmsle':rmsle_log(y,pred,row_weights(folds,fold_weights)),
        'fold34_rmsle':rmsle_log(y[recent],pred[recent]),
    }
    report['recent_cross_checks']=[]
    for source,target in [(3,4),(4,3)]:
        fit=folds==source; test=folds==target
        # Also report strict target-date exclusion because validation windows overlap.
        for purge in [False,True]:
            fit_mask=fit.copy()
            if purge:
                if dates is None:continue
                date_array=np.asarray(dates,dtype='datetime64[D]')
                fit_mask &= ~np.isin(date_array,date_array[test])
            if not fit_mask.any():continue
            wf=convex_weights(y[fit_mask],p[fit_mask])
            best=int(np.argmin(np.mean((p[fit_mask]-y[fit_mask,None])**2,axis=0)))
            report['recent_cross_checks'].append({
                'fit_fold':source,'test_fold':target,'purged_overlapping_target_dates':purge,
                'fit_rows':int(fit_mask.sum()),'weights':dict(zip(names,wf.tolist())),
                'ensemble_rmsle':rmsle_log(y[test],p[test]@wf),
                'training_selected_single':names[best],
                'single_rmsle':rmsle_log(y[test],p[test,best]),
                'chronological':source<target,
            })
    # The fitting objective includes every single-model vertex.
    best_single=min(report['models'][n]['fold34_rmsle'] for n in names)
    if report['ensemble']['fold34_rmsle'] > best_single + 1e-7:
        raise RuntimeError('Convex optimizer is worse than a feasible single model')
    report['residual_correlations_fold34']=pd.DataFrame(y[recent,None]-p[recent],columns=names).corr().to_dict()
    report['evaluation_limit'] += ' Recent-fold optimized scores reuse labels for weights; purged weight checks still reuse fold-selected early stopping and feature-design decisions. Fold 3/4 are not season-identical to July-October.'
    return pred,lofo,report,corr
