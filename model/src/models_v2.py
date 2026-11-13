"""Residual tree wrappers; torch is isolated to the TabM worker process."""
import os
import random
import numpy as np
import pandas as pd
from .features_v2 import CATS

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed)

def devices():
    # Probe in a fresh interpreter: loading macOS torch and LightGBM OpenMP
    # runtimes together can crash native dataset construction.
    import subprocess,sys,json
    result=subprocess.run([sys.executable,'-c',"import torch,json;print(json.dumps({'cuda_count':torch.cuda.device_count(),'mps':torch.backends.mps.is_available()}))"],capture_output=True,text=True,check=True)
    return json.loads(result.stdout)

class ResidualTree:
    def __init__(self,name,cfg,seed=2026,iterations=None,hardware=None):
        self.name=name;self.cfg=cfg;self.seed=seed;self.iterations=iterations
        self.hardware=hardware or {'cuda_count':0,'mps':False}
    def frame(self,x,fit=False):
        x=x.copy();self.cats=[c for c in CATS if c in x]
        if self.name=='catboost':
            for c in self.cats:x[c]=x[c].fillna(-1).astype('int32').astype(str)
        else:
            if fit:self.categories={c:sorted(int(v) for v in x[c].dropna().unique()) for c in self.cats}
            for c in self.cats:x[c]=pd.Categorical(x[c],categories=self.categories[c])
        return x
    def fit(self,x,y,xvalid=None,yvalid=None):
        seed_everything(self.seed);x=self.frame(x,fit=True)
        xv=None if xvalid is None else self.frame(xvalid)
        p=dict(self.cfg[self.name]);threads=self.cfg['threads'];patience=self.cfg['patience']
        gpu=self.hardware['cuda_count']>0 and self.cfg['device']!='cpu'
        self.device='cpu';self.history={}
        if self.name=='lightgbm':
            import lightgbm as lgb
            if self.iterations is not None:p['n_estimators']=self.iterations
            self.model=lgb.LGBMRegressor(**p,objective='regression_l2',random_state=self.seed,n_jobs=threads,verbosity=-1,deterministic=True,force_col_wise=True)
            callbacks=[lgb.log_evaluation(100),lgb.record_evaluation(self.history)]
            if xv is not None:callbacks.append(lgb.early_stopping(patience,verbose=True))
            self.model.fit(x,y[:,0],eval_set=None if xv is None else [(xv,yvalid[:,0])],eval_metric='rmse',callbacks=callbacks,categorical_feature=self.cats)
            self.best_iteration=int(self.model.best_iteration_ or p['n_estimators'])
        elif self.name=='catboost':
            from catboost import CatBoostRegressor
            if self.iterations is not None:p['iterations']=self.iterations
            if gpu:p.update(task_type='GPU',devices='0');self.device='cuda:0'
            self.model=CatBoostRegressor(**p,loss_function='RMSE',random_seed=self.seed,thread_count=threads,allow_writing_files=False,verbose=100)
            self.model.fit(x,y[:,0],cat_features=self.cats,eval_set=None if xv is None else (xv,yvalid[:,0]),early_stopping_rounds=patience if xv is not None else None)
            self.best_iteration=int(self.model.get_best_iteration()+1) if xv is not None else int(p['iterations'])
            self.history=self.model.get_evals_result()
        elif self.name=='xgboost':
            from .xgb_compat import InferenceConsistentXGBRegressor as XGBRegressor
            if self.iterations is not None:p['n_estimators']=self.iterations
            device=f'cuda:{min(1,self.hardware["cuda_count"]-1)}' if gpu else 'cpu';self.device=device
            self.model=XGBRegressor(**p,objective='reg:squarederror',eval_metric='rmse',random_state=self.seed,n_jobs=threads,device=device,early_stopping_rounds=patience if xv is not None else None)
            self.model.fit(x,y[:,0],eval_set=None if xv is None else [(xv,yvalid[:,0])],verbose=100)
            self.best_iteration=int(self.model.best_iteration+1) if xv is not None else int(p['n_estimators'])
            self.history=self.model.evals_result() if xv is not None else {}
        else:raise ValueError(self.name)
        return self
    def predict_log(self,x):
        residual=self.model.predict(self.frame(x))
        return np.maximum(0,x.anchor_board_log.to_numpy(dtype='float64')+residual)
    def to_cpu(self):
        if self.name=='xgboost':self.model.set_params(device='cpu')
        self.device='cpu';return self
