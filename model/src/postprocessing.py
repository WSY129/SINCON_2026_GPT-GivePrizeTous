"""Cutoff-safe, holiday-preserving corrections for nonnegative demand forecasts."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .holiday_calendar import calendar_features

ANCHOR_WINDOWS=(28,56,112)

def build_context(history,rows,origin):
    origin=pd.Timestamp(origin)
    query=rows[['date','station_id','hour']].copy().reset_index(drop=True)
    query['date']=pd.to_datetime(query.date)
    if (query.date<=origin).any():raise ValueError('Postprocessing queries must follow the origin')
    past=history.loc[pd.to_datetime(history.date)<=origin,['date','station_id','hour','boardings']].copy()
    past['date']=pd.to_datetime(past.date)
    if past.empty:raise ValueError('No observations before origin')
    query['dow']=query.date.dt.dayofweek
    query['is_holiday']=calendar_features(query.date,as_of=origin).is_holiday.to_numpy(dtype=bool)
    past['dow']=past.date.dt.dayofweek
    past['is_holiday']=calendar_features(past.date,as_of=origin).is_holiday.to_numpy(dtype=bool)
    past['log_boardings']=np.log1p(past.boardings)
    keys=['station_id','hour','dow']
    index=pd.MultiIndex.from_frame(query[keys])
    for days in ANCHOR_WINDOWS:
        available=past.loc[(past.date>origin-pd.Timedelta(days=days))&~past.is_holiday]
        profile=available.groupby(keys).log_boardings.mean()
        query[f'anchor_{days}']=profile.reindex(index).to_numpy()
    for days in [56,112]:
        available=past.loc[past.date>origin-pd.Timedelta(days=days)]
        cap=available.groupby(['station_id','hour']).log_boardings.quantile(.95)
        query[f'cap95_{days}']=cap.reindex(pd.MultiIndex.from_frame(query[['station_id','hour']])).to_numpy()
    return query

def candidate_specs():
    calibrations=[{'kind':'none','scale':0.}]
    for group in ['hour','hour_dow']:
        for scale in [.25,.5]:calibrations.append({'kind':group,'scale':scale})
    anchors=[{'days':0,'weight':0.}]+[{'days':d,'weight':w} for d in ANCHOR_WINDOWS for w in [.05,.1,.2]]
    candidates=[]
    for cal in calibrations:
        for anchor in anchors:
            name=f'{cal["kind"]}_{cal["scale"]:g}_anchor{anchor["days"]}_{anchor["weight"]:g}'
            candidates.append({'name':name,'calibration':cal,'anchor':anchor,'cap_days':0,'eligible':True})
    for scale in [.5,1.]:
        candidates.append({'name':f'global_bias_{scale:g}','calibration':{'kind':'global','scale':scale},'anchor':{'days':0,'weight':0.},'cap_days':0,'eligible':True})
    for days in [56,112]:
        candidates.append({'name':f'diagnostic_cap95_{days}','calibration':{'kind':'none','scale':0.},'anchor':{'days':0,'weight':0.},'cap_days':days,'eligible':False})
    return candidates

def group_keys(context,kind):
    if kind=='hour':return context.hour.astype(str).to_numpy()
    if kind=='hour_dow':return (context.hour.astype(str)+'|'+context.dow.astype(str)).to_numpy()
    raise ValueError(f'Unknown calibration group: {kind}')

def fit_plan(context,predictions,actual,spec,fold_ids=None):
    pred=np.asarray(predictions,dtype=np.float64);actual=np.asarray(actual,dtype=np.float64)
    if len(context)!=len(pred) or pred.shape!=actual.shape:raise ValueError('Postprocessing fit shape mismatch')
    if not np.isfinite(pred).all() or np.any(pred<0) or not np.isfinite(actual).all() or np.any(actual<0):
        raise ValueError('Finite nonnegative fit inputs required')
    kind=spec['calibration']['kind']
    result={'spec':spec,'offsets':{},'global_offset':0.,'shrinkage_pseudo_rows':1000,'offset_clip':.1,
            'holiday_policy':'retain_exact_model_prediction','missing_anchor_policy':'retain_calibrated_model_prediction'}
    if kind=='none':return result
    residual=np.log1p(actual)-np.log1p(pred)
    eligible=~context.is_holiday.to_numpy(dtype=bool)
    if not eligible.any():return result
    if kind=='global':
        result['global_offset']=float(np.clip(residual[eligible].mean(),-.1,.1)*spec['calibration']['scale'])
        return result
    folds=np.zeros(len(context),dtype=int) if fold_ids is None else np.asarray(fold_ids)
    data=pd.DataFrame({'key':group_keys(context,kind),'fold':folds,'residual':residual}).loc[eligible].copy()
    # Remove the seasonal/global bias separately per calibration fold so a sign
    # reversal between development periods is not transported into test forecasts.
    data['centered']=data.residual-data.groupby('fold').residual.transform('mean')
    grouped=data.groupby('key').centered.agg(['sum','size'])
    shifts=(grouped['sum']/(grouped['size']+1000)).clip(-.1,.1)*spec['calibration']['scale']
    result['offsets']={str(k):float(v) for k,v in shifts.items()}
    return result

def apply_plan(context,predictions,plan):
    pred=np.asarray(predictions,dtype=np.float64)
    if pred.ndim!=1 or len(context)!=len(pred) or not np.isfinite(pred).all() or np.any(pred<0):
        raise ValueError('Finite nonnegative predictions matching query rows required')
    spec=plan['spec'];kind=spec['calibration']['kind']
    if kind=='none' and spec['anchor']['weight']==0 and spec['cap_days']==0:return pred.copy()
    z=np.log1p(pred)
    if kind=='global':z=z+plan['global_offset']
    elif kind!='none':z=z+pd.Series(group_keys(context,kind)).map(plan['offsets']).fillna(0).to_numpy()
    anchor=spec['anchor']
    if anchor['weight']:
        if not 0<=anchor['weight']<=.2:raise ValueError('Historical anchor share must not exceed 20%')
        historical=context[f'anchor_{anchor["days"]}'].to_numpy()
        usable=np.isfinite(historical)
        z[usable]=(1-anchor['weight'])*z[usable]+anchor['weight']*historical[usable]
    if spec['cap_days']:
        cap=context[f'cap95_{spec["cap_days"]}'].to_numpy()
        usable=np.isfinite(cap);z[usable]=np.minimum(z[usable],cap[usable])
    out=np.expm1(np.maximum(z,0))
    # No closure or zero-demand rule exists for any of the target hours/holidays.
    holiday=context.is_holiday.to_numpy(dtype=bool);out[holiday]=pred[holiday]
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite postprocessed output')
    return out

def apply_postprocessing(history,rows,origin,predictions,plan):
    return apply_plan(build_context(history,rows,origin),predictions,plan)
