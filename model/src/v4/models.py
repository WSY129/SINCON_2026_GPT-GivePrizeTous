"""Version-isolated tree policies and auxiliary decomposition models."""
import copy
import numpy as np
import pandas as pd
from ..models_v2 import ResidualTree as BaseTree

class ResidualTree(BaseTree):
    def __init__(self,name,cfg,seed=2026,iterations=None,hardware=None):
        cfg=copy.deepcopy(cfg)
        if name=='catboost_cpu':name='catboost';cfg['device']='cpu'
        super().__init__(name,cfg,seed,iterations,hardware)
    def frame(self,x,fit=False):
        if self.name in ('lightgbm','xgboost'):x=x.drop(columns=['station_eff_dow'],errors='ignore')
        return super().frame(x,fit)


def softmax_shares(logits):
    logits=np.asarray(logits);e=np.exp(logits-logits.max(axis=1,keepdims=True))
    return e/e.sum(axis=1,keepdims=True)

class LowDemand:
    def __init__(self,cfg,seed=2026,iterations=None,hardware=None):
        self.cfg=copy.deepcopy(cfg);self.seed=seed;self.iterations=iterations;self.hardware=hardware
        self.cfg['lightgbm'].update(num_leaves=95,min_child_samples=8,reg_lambda=1)
        self.cfg['catboost'].update(depth=6,l2_leaf_reg=1)
    def fit(self,x,y,xvalid=None,yvalid=None):
        gate=(x.anchor_v4<3.3)|(x.hour>=21)
        vg=None if xvalid is None else (xvalid.anchor_v4<3.3)|(xvalid.hour>=21)
        self.models=[]
        for name in ('lightgbm','catboost_cpu'):
            m=ResidualTree(name,self.cfg,self.seed,self.iterations,self.hardware)
            m.fit(x.loc[gate],y[gate],None if vg is None else xvalid.loc[vg],None if vg is None else yvalid[vg])
            self.models.append(m)
        self.best_iteration=max(m.best_iteration for m in self.models)
        self.history={m.name:m.history for m in self.models};self.device='cpu';return self
    def predict_log(self,x):
        gate=(x.anchor_v4<3.3)|(x.hour>=21)
        p=x.anchor_board_log.to_numpy(dtype=float).copy()
        p[gate]=np.mean([m.predict_log(x.loc[gate]) for m in self.models],axis=0)
        return p
    def to_cpu(self):
        for m in self.models:m.to_cpu()
        return self


def daily_frame(x):
    """Inputs must be contiguous complete 18-hour station-days."""
    if len(x)%18:raise ValueError('Incomplete station-days')
    if not np.array_equal(x.hour.to_numpy().reshape(-1,18),np.tile(np.arange(6,24),(len(x)//18,1))):raise ValueError('Hours must be 6..23')
    values=x.to_numpy().reshape(-1,18,x.shape[1])
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);a=np.nanmean(values,axis=1)
    frame=pd.DataFrame(a,columns=x.columns)
    frame=frame.drop(columns=['hour','station_hour','hour_sin','hour_cos'],errors='ignore')
    base=np.log1p(np.expm1(x.anchor_board_log.to_numpy().reshape(-1,18)).sum(1))
    frame['anchor_board_log']=base
    return frame

class Decomposition:
    def __init__(self,cfg,seed=2026,iterations=None,hardware=None):
        self.cfg=cfg;self.seed=seed;self.iterations=iterations;self.hardware=hardware
    def fit(self,x,y,xvalid=None,yvalid=None,daily_x=None,daily_y=None):
        # daily_x contains all complete origin/station/day candidate blocks.
        dx=daily_frame(daily_x);true=np.expm1(daily_y[:,0]+daily_x.anchor_board_log.to_numpy()).reshape(-1,18)
        target=np.log1p(true.sum(1))-dx.anchor_board_log.to_numpy()
        dv=None if xvalid is None else daily_frame(xvalid)
        tv=None if dv is None else np.log1p(np.expm1(yvalid[:,0]+xvalid.anchor_board_log.to_numpy()).reshape(-1,18).sum(1))-dv.anchor_board_log.to_numpy()
        self.daily=[]
        for name in ('lightgbm','catboost_cpu'):
            m=ResidualTree(name,self.cfg,self.seed,self.iterations,self.hardware).fit(dx,np.column_stack([target,target]),dv,None if tv is None else np.column_stack([tv,tv]))
            self.daily.append(m)
        # Share model uses complete blocks; separate objective learns within-day structure.
        self.share=ResidualTree('lightgbm',self.cfg,self.seed,self.iterations,self.hardware)
        rng=np.random.default_rng(self.seed);nd=len(daily_x)//18
        chosen=np.sort(rng.choice(nd,min(nd,60000),replace=False));rows=(chosen[:,None]*18+np.arange(18)).ravel()
        true=true[chosen];sx=daily_x.iloc[rows].copy();base=np.log(softmax_shares(sx.anchor_board_log.to_numpy().reshape(-1,18))).ravel()
        sx['anchor_board_log']=0
        shares=np.log((true+.5)/(true.sum(1,keepdims=True)+9)).ravel()
        self.share_base_mode='softmax_anchor'
        sy=np.column_stack([shares-base,shares-base])
        sv=None if xvalid is None else xvalid.copy()
        if sv is not None:
            vb=np.log(softmax_shares(xvalid.anchor_board_log.to_numpy().reshape(-1,18))).ravel()
            vt=np.expm1(yvalid[:,0]+xvalid.anchor_board_log.to_numpy()).reshape(-1,18)
            vtarget=np.log((vt+.5)/(vt.sum(1,keepdims=True)+9)).ravel()-vb
            sv['anchor_board_log']=0
        self.share.fit(sx,sy,sv,None if sv is None else np.column_stack([vtarget,vtarget]))
        self.best_iteration=max([m.best_iteration for m in self.daily]+[self.share.best_iteration]);self.device='cpu'
        self.history={f'daily_{m.name}':m.history for m in self.daily};self.history['share']=self.share.history
        return self
    def predict_log(self,x):
        daily=np.mean([m.predict_log(daily_frame(x)) for m in self.daily],axis=0)
        # Raw residual prediction is intentional: share logits may be negative.
        logits=np.log(softmax_shares(x.anchor_board_log.to_numpy().reshape(-1,18))).ravel()+self.share.model.predict(self.share.frame(x.assign(anchor_board_log=0)))
        shares=softmax_shares(logits.reshape(-1,18))
        # Invert +0.5 smoothing exactly: total +18*0.5, not total alone.
        counts=np.maximum(0,(np.expm1(daily)[:,None]+9)*shares-.5)
        return np.log1p(counts).ravel()
    def to_cpu(self):
        for m in self.daily+[self.share]:m.to_cpu()
        return self
