"""CUDA forward/backward and serialization smoke checks, separately from OpenMP trees."""
import json
import numpy as np,pandas as pd,joblib
from ..features import load_data
from .features import make_features
from .pipeline import ROOT,OUT
from .tabm import TabMRegressor,multitask_loss
from .spatial import SpatialRegressor

def main():
    d=load_data(ROOT/'data/raw');cfg=json.loads((ROOT/'configs/ensemble_v5.json').read_text());cfg['threads']=2;cfg['tabm']['epochs']=1;cfg['st_itransformer']['epochs']=1
    parts=[]
    for o,a,b in [('2025-12-30','2026-01-01','2026-01-02'),('2026-02-27','2026-03-01','2026-03-02')]:
        q=d['train'].loc[d['train'].date.between(a,b)&d['train'].station_id.isin([3107,3112,3127])].sort_values(['date','station_id','hour']).reset_index(drop=True)
        x=make_features(d['train'],q,d['metaData'],o);y=np.column_stack([np.log1p(q.boardings)-x.anchor_board_log,np.log1p(q.alightings)-x.anchor_alight_log]).astype('float32');k=q[['date','station_id','hour']].copy();k['origin']=pd.Timestamp(o);parts.append((x,y,k))
    x,y,k=parts[0];v,t,vk=parts[1]
    for name in ['tabm','spatial']:
        if name=='tabm':m=TabMRegressor(cfg).fit(x,y,v,t);pred=m.predict_log(v)
        else:m=SpatialRegressor(cfg).fit(x,y,v,t,k,vk,d['train'],d['metaData']);pred=m.predict_log(v,vk)
        assert np.isfinite(pred).all() and pred.shape==(len(v),)
        m.to_cpu();path=OUT/f'smoke_{name}.joblib';joblib.dump(m,path);m2=joblib.load(path)
        p2=m2.predict_log(v) if name=='tabm' else m2.predict_log(v,vk)
        assert np.allclose(pred,p2,rtol=1e-4,atol=1e-4)
        print('OK',name,flush=True)
if __name__=='__main__':main()
