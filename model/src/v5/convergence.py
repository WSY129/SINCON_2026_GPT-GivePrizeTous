"""Iteration diagnostics are evidence to review, not convergence guarantees."""
import math
DIRECT={'lightgbm','catboost_cpu','xgboost'}

def validate_convergence(name,result,cfg):
    best=result['best_iteration'];floor=cfg['convergence_min_tree'] if name in DIRECT else cfg['convergence_min_tabm'] if name=='tabm_v5' else 1
    if not math.isfinite(float(best)) or best<1:raise ValueError('Invalid fitted iteration')
    return {'best_iteration':best,'diagnostic_minimum':floor,'below_diagnostic_minimum':best<floor,'note':'Early optimum alone does not establish underfitting or overfitting. Never force a higher best iteration.'}

def final_iterations(best3,best4):
    if min(best3,best4)<=0:raise ValueError('Invalid best iterations')
    return max(1,round(min(best4*1.25,max(best4,.3*best3+.7*best4)*1.10)))
