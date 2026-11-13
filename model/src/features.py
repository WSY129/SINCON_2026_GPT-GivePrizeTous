"""Origin-frozen direct forecasting features. Never read labels after origin."""
from pathlib import Path
from .features_v2 import make_features_v2, training_matrix_v2, partial_origins
import numpy as np
import pandas as pd
from .holiday_calendar import calendar_features

KEYS = ['station_id', 'hour']

def load_data(data_dir):
    import polars as pl
    root = Path(data_dir)
    result = {}
    for name in ['train', 'test', 'sample_submission', 'metaData']:
        lf = pl.scan_csv(root / f'{name}.csv')
        schema = lf.collect_schema()
        casts = [pl.col(c).cast(pl.Int32) for c in ['station_id', 'hour', 'boardings', 'alightings', 'canonical_line'] if c in schema]
        df = pd.DataFrame(lf.with_columns(casts).collect().to_dict(as_series=False))
        if 'date' in df: df['date'] = pd.to_datetime(df['date'])
        for c in ['station_id', 'hour', 'boardings', 'alightings', 'canonical_line']:
            if c in df: df[c] = df[c].astype('int32')
        result[name] = df
    tr, te = result['train'], result['test']
    for name, frame in [('train',tr),('test',te)]:
        if frame.duplicated(['date'] + KEYS).any(): raise ValueError(f'{name}: duplicate keys')
        if frame.isna().any().any(): raise ValueError(f'{name}: input cells missing; inspect first')
    if tr.date.max() >= te.date.min(): raise ValueError('Test is not strictly after training')
    if not np.isfinite(tr[['boardings','alightings']]).all().all() or (tr[['boardings','alightings']] < 0).any().any():
        raise ValueError('Training counts must be finite and nonnegative')
    return result

def make_features(history, query, metadata, origin):
    origin = pd.Timestamp(origin)
    hist = history.loc[history.date <= origin].copy()
    if hist.empty: raise ValueError('No history at this origin')
    if (query.date <= origin).any(): raise ValueError('Queries must follow origin')
    q = query[['date'] + KEYS].reset_index(drop=True).copy()
    q['_order'] = np.arange(len(q))
    q['dow'] = q.date.dt.dayofweek.astype('int8')
    hist['dow'] = hist.date.dt.dayofweek.astype('int8')
    q = q.merge(metadata[['station_id','canonical_line']],on='station_id',how='left',validate='many_to_one')
    if q.canonical_line.isna().any(): raise ValueError('Station absent from metadata')
    hist = hist.merge(metadata[['station_id','canonical_line']],on='station_id',how='left',validate='many_to_one')
    q['horizon'] = (q.date - origin).dt.days
    q['month'] = q.date.dt.month
    q['day_of_month'] = q.date.dt.day
    q['day_of_year'] = q.date.dt.dayofyear
    q['date_index'] = (q.date - pd.Timestamp('2025-01-01')).dt.days
    q['weekend'] = (q.dow >= 5).astype('int8')
    q = pd.concat([q,calendar_features(q.date,as_of=origin)],axis=1)
    for col, period in [('hour',24),('dow',7),('day_of_year',365.25)]:
        for trig in ['sin','cos']:
            q[f'{col}_{trig}'] = getattr(np,trig)(2*np.pi*q[col]/period)
    inception = hist.groupby('station_id',as_index=False).date.min().rename(columns={'date':'observed_first_date'})
    q = q.merge(inception,on='station_id',how='left',validate='many_to_one')
    q['history_days'] = (origin - q.observed_first_date).dt.days + 1
    q['cold_start'] = q.observed_first_date.isna().astype('int8')
    q = q.drop(columns='observed_first_date')
    # All aggregations are computed only once per origin, never using query labels.
    for days in [28,56,112]:
        h = hist.loc[hist.date > origin - pd.Timedelta(days=days)]
        stats = h.groupby(KEYS+['dow']).agg(
            **{f'board_mean_{days}':('boardings','mean'),
               f'board_median_{days}':('boardings','median'),
               f'board_std_{days}':('boardings','std'),
               f'alight_mean_{days}':('alightings','mean'),
               f'n_{days}':('boardings','size')}).reset_index()
        q = q.merge(stats,on=KEYS+['dow'],how='left',validate='many_to_one')
    short = hist.loc[hist.date > origin - pd.Timedelta(days=28)]
    stats = short.groupby(KEYS).agg(recent_mean=('boardings','mean'),recent_std=('boardings','std'),recent_alight=('alightings','mean')).reset_index()
    q = q.merge(stats,on=KEYS,how='left',validate='many_to_one')
    for keys,label in [(['canonical_line','hour','dow'],'line'),(['hour','dow'],'global')]:
        agg = short.groupby(keys,as_index=False).boardings.median().rename(columns={'boardings':f'{label}_median'})
        q = q.merge(agg,on=keys,how='left',validate='many_to_one')
    # Weekly seasonal copy for arbitrary horizon: repeat the last observed weekday.
    q['_lookup_date'] = origin - pd.to_timedelta((origin.dayofweek - q.dow)%7,unit='D')
    lookup = hist[['date']+KEYS+['boardings','alightings']].rename(columns={'date':'_lookup_date','boardings':'seasonal7','alightings':'seasonal7_alight'})
    q = q.merge(lookup,on=['_lookup_date']+KEYS,how='left',validate='many_to_one').drop(columns='_lookup_date')
    for days in [1,7,28]:
        snap = hist.loc[hist.date == origin - pd.Timedelta(days=days-1),KEYS+['boardings']].rename(columns={'boardings':f'origin_lag_{days}'})
        q = q.merge(snap,on=KEYS,how='left',validate='many_to_one')
    q['_lookup_date'] = q.date - pd.Timedelta(days=364)
    # This explicit guard keeps the feature valid if a longer forecast is requested.
    q.loc[q._lookup_date > origin,'_lookup_date'] = pd.NaT
    lookup = hist[['date']+KEYS+['boardings']].rename(columns={'date':'_lookup_date','boardings':'year364'})
    q = q.merge(lookup,on=['_lookup_date']+KEYS,how='left',validate='many_to_one').drop(columns='_lookup_date')
    q['trend_28_112'] = q.board_mean_28 / q.board_mean_112.clip(lower=1)
    q['boarding_alighting_ratio'] = q.board_mean_56 / q.alight_mean_56.clip(lower=1)
    q['baseline'] = q.board_median_56.fillna(q.line_median).fillna(q.global_median).fillna(float(hist.boardings.median()))
    q['seasonal7'] = q.seasonal7.fillna(q.baseline)
    q['weekday_mean'] = q.board_mean_56.fillna(q.baseline)
    q = q.sort_values('_order').drop(columns=['_order','date']).reset_index(drop=True)
    return q.astype('float32')

def training_matrix(train, metadata, cutoff, horizon=123, origin_stride=28, max_rows=220000, seed=2026):
    """Whole label windows end by cutoff; last training origin is horizon days earlier."""
    cutoff = pd.Timestamp(cutoff)
    start = train.date.min() + pd.Timedelta(days=27)
    end = cutoff - pd.Timedelta(days=horizon)
    origins = pd.date_range(start,end,freq=f'{origin_stride}D')
    if len(origins) == 0: raise ValueError('Insufficient history for full-horizon training')
    # Include the exact last permissible anchor so the newest observed labels
    # are represented even when the 28-day grid misses the cutoff boundary.
    if origins[-1] != end: origins = origins.append(pd.DatetimeIndex([end]))
    rng = np.random.default_rng(seed)
    per_origin = max(1,max_rows//len(origins))
    xs, ys, audit = [], [], []
    for origin in origins:
        labels = train.loc[(train.date > origin)&(train.date <= origin + pd.Timedelta(days=horizon))].copy()
        if len(labels)>per_origin:
            labels = labels.iloc[np.sort(rng.choice(len(labels),per_origin,replace=False))]
        x = make_features(train,labels,metadata,origin)
        xs.append(x); ys.append(labels.boardings.to_numpy(dtype='float32'))
        audit.append({'origin':str(origin.date()),'feature_max_date':str(origin.date()),'label_end':str((origin+pd.Timedelta(days=horizon)).date()),'rows':len(labels)})
    return pd.concat(xs,ignore_index=True),np.concatenate(ys),audit
