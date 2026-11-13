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


def cap_origins(candidates, daily, maximum=3, seed=2026):
    """At most three origins per target station-day, preserving all 18 hours.

    Select one origin from each sorted horizon tercile, so limiting duplicates
    does not silently remove long-horizon training cases.
    """
    days=candidates[['date','station_id','oi','horizon','rich']].drop_duplicates().copy()
    days['stratum']=np.minimum(2,(days.horizon-1)//41)
    days['priority']=np.random.default_rng(seed).random(len(days))-days.rich.astype(float)
    ix=days.groupby(['date','station_id','stratum']).priority.idxmin()
    allowed=days.loc[ix,['date','station_id','oi']]
    candidates=candidates.merge(allowed,on=['date','station_id','oi'],how='inner',validate='many_to_one')
    daily=daily.merge(allowed,on=['date','station_id','oi'],how='inner',validate='many_to_one')
    assert candidates.groupby(['date','station_id']).oi.nunique().max()<=maximum
    return candidates,daily


def matrix(data_dir,output,cfg,fold):
    output=Path(output);dest=output/('final' if fold==0 else 'final_diagnostic' if fold==5 else f'fold_{fold}');dest.mkdir(exist_ok=True)
    path=dest/'matrix.joblib'
    if path.exists():return path
    cutoff=pd.Timestamp(cfg['final_cutoff'] if fold==0 else cfg['final_diagnostic_origin'] if fold==5 else cfg['fold_origins'][fold-1])
    os=origins(cutoff,7 if fold==4 else 14);candidate=[];blocks=[];audits=[]
    for oi,o in enumerate(os):
        d=joblib.load(output/'origin_cache'/f'{o.date()}.joblib')
        limit=(d['keys'].date<=cutoff).to_numpy();ix=np.flatnonzero(limit&d['eligible'])
        frame=pd.DataFrame({'oi':oi,'row':ix,'horizon':d['x'].horizon.to_numpy()[ix].astype(int),'rich':d['x'].n_yoy.to_numpy()[ix]>=2})
        frame['date']=d['keys'].date.to_numpy()[ix];frame['station_id']=d['keys'].station_id.to_numpy()[ix]
        candidate.append(frame)
        # Complete station-days retained for all daily labels; known-outage days excluded from training only.
        keep=limit.reshape(-1,18).all(1)&d['eligible'].reshape(-1,18).all(1)
        day=np.flatnonzero(keep)
        blocks.append(pd.DataFrame({'oi':oi,'day':day,'date':d['keys'].date.to_numpy()[day*18],'station_id':d['keys'].station_id.to_numpy()[day*18]}))
    candidates=pd.concat(candidate,ignore_index=True);daily=pd.concat(blocks,ignore_index=True)
    uncapped=len(candidates)
    candidates,daily=cap_origins(candidates,daily,cfg['max_origins_per_target'],cfg['seed'])
    maxrows=cfg['final_training_rows'] if fold==0 else cfg['max_training_rows']
    rng=np.random.default_rng(cfg['seed']);chosen=[]
    candidates['season']=(candidates.date.dt.month%12)//3
    groups=list(candidates.groupby('season'));quota=maxrows//len(groups)
    for season,part in groups:
        # Mild preference only, no forced YoY percentage and no replacement.
        n=min(quota,len(part));prob=np.where(part.rich,1.25,1.0);prob=prob/prob.sum()
        chosen.extend(rng.choice(part.index,n,replace=False,p=prob).tolist())
    remaining=min(maxrows,len(candidates))-len(chosen)
    if remaining:
        rest=candidates.index.difference(chosen);chosen.extend(rng.choice(rest,remaining,replace=False).tolist())
    chosen=candidates.loc[chosen];xs=[];ys=[];keys=[];daily_x=[];daily_y=[];daily_keys=[]
    for oi,o in enumerate(os):
        d=joblib.load(output/'origin_cache'/f'{o.date()}.joblib');ix=chosen.loc[chosen.oi==oi,'row'].to_numpy()
        xs.append(d['x'].iloc[ix]);ys.append(d['y'][ix]);key=d['keys'].iloc[ix].copy();key['origin']=o;keys.append(key)
        day=daily.loc[daily.oi==oi,'day'].to_numpy();di=(day[:,None]*18+np.arange(18)).ravel()
        daily_x.append(d['x'].iloc[di]);daily_y.append(d['y'][di]);dk=d['keys'].iloc[di].copy();dk['origin']=o;daily_keys.append(dk)
        audits.append({'origin':str(o.date()),'rows':len(ix),'latest_label':None if not len(ix) else str(key.date.max().date())})
    train=load_data(Path(data_dir));valid=train['test'] if fold==0 else train['train'].loc[train['train'].date.between(cutoff+pd.Timedelta(days=1),cutoff+pd.Timedelta(days=123))]
    if fold==5:valid=valid.loc[valid.date<=pd.Timestamp(cfg['final_diagnostic_end'])]
    valid=valid.sort_values(['date','station_id','hour']).reset_index(drop=True)
    xv=make_features(train['train'],valid,train['metaData'],cutoff)
    m={'x':pd.concat(xs,ignore_index=True),'y':np.concatenate(ys),'keys':pd.concat(keys,ignore_index=True),'daily_x':pd.concat(daily_x,ignore_index=True),'daily_y':np.concatenate(daily_y),'daily_keys':pd.concat(daily_keys,ignore_index=True),'valid':valid,'origin':cutoff,'query_x':xv}
    m['sample_weight']=np.where(m['x'].n_yoy>=2,1.25,1.0).astype('float32')
    if fold:
        m['xvalid']=xv;m['yvalid']=np.column_stack([np.log1p(valid.boardings)-xv.anchor_board_log,np.log1p(valid.alightings)-xv.anchor_alight_log]).astype('float32')
    else:m['xtest']=xv
    audit={'uncapped_candidate_rows':uncapped,'max_origins_per_target_observed':int(chosen.groupby(['date','station_id']).oi.nunique().max()),'max_origins_per_target':cfg['max_origins_per_target'],'rows':len(m['x']),'candidate_rows':len(candidates),'rich_candidate_rows':int(candidates.rich.sum()),'rich_fraction':float(chosen.rich.mean()),'replacement':False,'season_counts':chosen.season.value_counts().to_dict(),'constraint_note':'At most one origin in each fixed 1-41/42-82/83-123 horizon band; no forced YoY quota.','horizon_counts':chosen.horizon.value_counts().sort_index().to_dict(),'origins':audits,'daily_candidate_days':len(daily),'query_rich_fraction':float((xv.n_yoy>=2).mean())}
    (dest/'feature_audit.json').write_text(json.dumps(audit,indent=2));joblib.dump(m,path)
    return path
