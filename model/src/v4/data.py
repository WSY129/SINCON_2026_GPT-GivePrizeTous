"""Shared origin cache and audited causal, horizon-stratified sampling."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from ..features import load_data
from ..domain_events import oneoff_mask
from .features import make_features


def origins(cutoff,step=14):
    end=pd.Timestamp(cutoff)-pd.Timedelta(days=14)
    out=list(pd.date_range('2025-02-11',end,freq=f'{step}D'))
    if out[-1]!=end:out.append(end)
    return out


def origin_cache(task):
    data_dir,output,origin=task;origin=pd.Timestamp(origin)
    path=Path(output)/'origin_cache'/f'{origin.date()}.joblib'
    if path.exists():return str(path)
    d=load_data(Path(data_dir));h=d['train'];labels=h.loc[(h.date>origin)&(h.date<=origin+pd.Timedelta(days=123))].sort_values(['date','station_id','hour']).reset_index(drop=True)
    x=make_features(h,labels,d['metaData'],origin)
    y=np.column_stack([np.log1p(labels.boardings)-x.anchor_board_log,np.log1p(labels.alightings)-x.anchor_alight_log]).astype('float32')
    valid=~oneoff_mask(labels).to_numpy()
    path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'x':x,'y':y,'keys':labels[['date','station_id','hour']],'eligible':valid,'origin':origin},path)
    print('ORIGIN',str(origin.date()),len(x),flush=True);return str(path)


def matrix(data_dir,output,cfg,fold):
    output=Path(output);dest=output/('final' if fold==0 else f'fold_{fold}');dest.mkdir(exist_ok=True)
    path=dest/'matrix.joblib'
    if path.exists():return path
    cutoff=pd.Timestamp(cfg['final_cutoff'] if fold==0 else cfg['fold_origins'][fold-1])
    os=origins(cutoff,7 if fold==4 else 14);candidate=[];blocks=[];audits=[]
    for oi,o in enumerate(os):
        d=joblib.load(output/'origin_cache'/f'{o.date()}.joblib')
        limit=(d['keys'].date<=cutoff).to_numpy();ix=np.flatnonzero(limit&d['eligible'])
        frame=pd.DataFrame({'oi':oi,'row':ix,'horizon':d['x'].horizon.to_numpy()[ix].astype(int),'rich':d['x'].n_yoy.to_numpy()[ix]>=2})
        candidate.append(frame)
        # Complete station-days retained for all daily labels; known-outage days excluded from training only.
        keep=limit.reshape(-1,18).all(1)&d['eligible'].reshape(-1,18).all(1)
        blocks.append(pd.DataFrame({'oi':oi,'day':np.flatnonzero(keep)}))
    candidates=pd.concat(candidate,ignore_index=True);daily=pd.concat(blocks,ignore_index=True)
    maxrows=cfg['final_training_rows'] if fold==0 else cfg['max_training_rows'];target=.85 if fold==0 else .80
    rng=np.random.default_rng(cfg['seed']);chosen=[]
    for horizon,part in candidates.groupby('horizon'):
        wanted=maxrows//123+(horizon<=maxrows%123)
        if fold in (0,3,4):
            rich=part.loc[part.rich];poor=part.loc[~part.rich]
            nr=min(len(rich),round(wanted*target));np_=min(len(poor),wanted-nr)
            nr=min(len(rich),wanted-np_)
            for subset,n in ((rich,nr),(poor,np_)):
                if n:chosen.extend(rng.choice(subset.index,n,replace=False).tolist())
        else:
            n=min(wanted,len(part));chosen.extend(rng.choice(part.index,n,replace=False).tolist())
    chosen=candidates.loc[chosen];xs=[];ys=[];keys=[];daily_x=[];daily_y=[];daily_keys=[]
    for oi,o in enumerate(os):
        d=joblib.load(output/'origin_cache'/f'{o.date()}.joblib');ix=chosen.loc[chosen.oi==oi,'row'].to_numpy()
        xs.append(d['x'].iloc[ix]);ys.append(d['y'][ix]);key=d['keys'].iloc[ix].copy();key['origin']=o;keys.append(key)
        day=daily.loc[daily.oi==oi,'day'].to_numpy();di=(day[:,None]*18+np.arange(18)).ravel()
        daily_x.append(d['x'].iloc[di]);daily_y.append(d['y'][di]);dk=d['keys'].iloc[di].copy();dk['origin']=o;daily_keys.append(dk)
        audits.append({'origin':str(o.date()),'rows':len(ix),'latest_label':None if not len(ix) else str(key.date.max().date())})
    train=load_data(Path(data_dir));valid=train['test'] if fold==0 else train['train'].loc[train['train'].date.between(cutoff+pd.Timedelta(days=1),cutoff+pd.Timedelta(days=123))]
    valid=valid.sort_values(['date','station_id','hour']).reset_index(drop=True)
    xv=make_features(train['train'],valid,train['metaData'],cutoff)
    m={'x':pd.concat(xs,ignore_index=True),'y':np.concatenate(ys),'keys':pd.concat(keys,ignore_index=True),'daily_x':pd.concat(daily_x,ignore_index=True),'daily_y':np.concatenate(daily_y),'daily_keys':pd.concat(daily_keys,ignore_index=True),'valid':valid,'origin':cutoff,'query_x':xv}
    if fold:
        m['xvalid']=xv;m['yvalid']=np.column_stack([np.log1p(valid.boardings)-xv.anchor_board_log,np.log1p(valid.alightings)-xv.anchor_alight_log]).astype('float32')
    else:m['xtest']=xv
    audit={'rows':len(m['x']),'candidate_rows':len(candidates),'rich_candidate_rows':int(candidates.rich.sum()),'rich_fraction':float(chosen.rich.mean()),'requested_rich_fraction':target if fold in (0,3,4) else None,'replacement':False,'quota_met':bool(chosen.rich.mean()>=target) if fold in (0,3,4) else None,'constraint_note':'Use all available rich candidates within each horizon quota, without duplicating labels or accessing future observations; an infeasible quota is reported, not fabricated.','horizon_counts':chosen.horizon.value_counts().sort_index().to_dict(),'origins':audits,'daily_candidate_days':len(daily),'query_rich_fraction':float((xv.n_yoy>=2).mean())}
    (dest/'feature_audit.json').write_text(json.dumps(audit,indent=2));joblib.dump(m,path)
    return path
