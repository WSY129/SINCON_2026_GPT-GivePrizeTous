"""Small real-data fit/reload checks before expensive CV; not score estimates."""
import json,copy
from pathlib import Path
import numpy as np,pandas as pd,joblib
from ..features import load_data
from .features import make_features
from .models import ResidualTree,Decomposition,LowDemand
from .pipeline import ROOT,OUT

def main():
    d=load_data(ROOT/'data/raw');cfg=json.loads((ROOT/'configs/ensemble_v4.json').read_text());cfg['threads']=2;cfg['patience']=2
    for key,iterkey in [('lightgbm','n_estimators'),('catboost','iterations'),('xgboost','n_estimators')]:cfg[key][iterkey]=2
    dates=[('2025-12-30','2026-01-01','2026-01-02'),('2026-02-27','2026-03-01','2026-03-02')];parts=[]
    for o,a,b in dates:
        q=d['train'].loc[d['train'].date.between(a,b)&d['train'].station_id.isin([3107,3112,3127])].sort_values(['date','station_id','hour']).reset_index(drop=True)
        x=make_features(d['train'],q,d['metaData'],o);y=np.column_stack([np.log1p(q.boardings)-x.anchor_board_log,np.log1p(q.alightings)-x.anchor_alight_log]).astype('float32');parts.append((x,y))
    x,y=parts[0];v,t=parts[1];result={}
    for name in ['lightgbm','catboost_cpu','xgboost','low_demand_specialist','decomp_gbdt']:
        if name=='decomp_gbdt':m=Decomposition(cfg).fit(x,y,v,t,daily_x=x,daily_y=y)
        elif name=='low_demand_specialist':m=LowDemand(cfg).fit(x,y,v,t)
        else:m=ResidualTree(name,cfg).fit(x,y,v,t)
        pred=m.predict_log(v);assert pred.shape==(len(v),) and np.isfinite(pred).all();result[name]=True
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'smoke_trees.json').write_text(json.dumps(result));print(result)
if __name__=='__main__':main()
