"""Causal seasonal kernels and regime anchors, retaining v3 covariates."""
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from ..features_v3 import make_features_v3, cleaned, lookup, NEW, CATS
from ..holiday_calendar import calendar_features


def dow_group(dow, holiday):
    return np.where(holiday | (dow == 6), 3, np.where(dow == 0, 0, np.where(dow == 5, 2, 1)))


def weighted_stats(values, weights):
    """Columnwise weighted quantiles and fractional-mass 15% trim."""
    weights = np.broadcast_to(weights, values.shape).copy()
    weights[~np.isfinite(values)] = 0
    count = (weights > 0).sum(axis=0)
    order = np.argsort(np.where(np.isfinite(values), values, np.inf), axis=0)
    v = np.take_along_axis(np.nan_to_num(values), order, axis=0)
    w = np.take_along_axis(weights, order, axis=0)
    total = w.sum(axis=0); denom = np.where(total > 0, total, 1)
    cumulative = np.cumsum(w, axis=0)
    def quantile(q):
        index = np.argmax(cumulative >= q * total, axis=0)
        result = v[index, np.arange(v.shape[1])].astype(float)
        result[total == 0] = np.nan
        return result
    mean = (v*w).sum(axis=0)/denom
    std = np.sqrt((w*(v-mean)**2).sum(axis=0)/denom)
    trimmed_w = np.maximum(0, np.minimum(cumulative, .85*total)-np.maximum(cumulative-w, .15*total))
    trimmed = (trimmed_w*v).sum(axis=0)/np.maximum(trimmed_w.sum(axis=0), 1e-12)
    for result in (mean, std, trimmed): result[total == 0] = np.nan
    return dict(mean=mean, std=std, median=quantile(.5), q85=quantile(.85), trimmed=trimmed, n=count)


def _fourier(dates):
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    return np.column_stack([fn(2*np.pi*k*doy/365.25) for k in range(1,4) for fn in (np.sin,np.cos)])


def fourier_predict(h, q, target, fallback):
    """Joint ridge: station/hour/effective-weekday effects + 6 annual harmonics."""
    keys = ['station_id','hour','eff_dow']; clean = h.loc[h.clean]
    codes, unique = pd.factorize(pd.MultiIndex.from_frame(clean[keys]))
    if not len(unique): return fallback.copy()
    onehot = sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes),len(unique)))
    design = sparse.hstack([onehot, sparse.csr_matrix(_fourier(clean.date))], format='csr')
    # Keep the demand level unpenalized. Penalizing every cell intercept toward
    # zero collapsed fallback demand at stations with a structural break.
    model = Ridge(alpha=50.0, fit_intercept=True, solver='lsqr', tol=1e-5, max_iter=80)
    model.fit(design, clean[target].to_numpy())
    coef = model.coef_
    qi = unique.get_indexer(pd.MultiIndex.from_frame(q[keys])); valid = qi >= 0
    result = fallback.copy(); result[valid] = model.intercept_ + coef[qi[valid]] + _fourier(q.date[valid]) @ coef[-6:]
    return np.maximum(0,result)


def make_features(history, query, meta, origin):
    origin = pd.Timestamp(origin)
    history = history.loc[history.date <= origin].copy()
    q = query[['date','station_id','hour']].copy().reset_index(drop=True)
    x = make_features_v3(history,q,meta,origin)
    q['eff_dow'] = x.eff_dow.astype(int); q['dow_group'] = dow_group(x.dow.to_numpy(), x.is_holiday.to_numpy(bool))
    q['canonical_line'] = x.canonical_line
    h = cleaned(history,origin)
    h['dow_group'] = dow_group(h.dow.to_numpy(),h.holiday.to_numpy())
    h['canonical_line'] = h.station_id.map(meta.set_index('station_id').canonical_line)
    raw = h.loc[~h.oneoff & (h.date > origin-pd.Timedelta(days=365))].copy()
    keys = ['station_id','hour','dow_group']
    grouped = raw.groupby(keys).board_log
    amp = (grouped.quantile(.9)-grouped.quantile(.5)).to_frame('amp')
    med = grouped.median().to_frame('med')
    raw['surge'] = raw.board_log.to_numpy() > lookup(med,raw,keys,'med')+.8
    prob = raw.groupby(keys).surge.mean().to_frame('prob')
    x['past_spike_amp'] = np.nan_to_num(lookup(amp,q,keys,'amp'))
    x['past_spike_prob'] = np.nan_to_num(lookup(prob,q,keys,'prob'))
    stations = sorted(meta.station_id.tolist()); cells = pd.MultiIndex.from_product([stations,range(6,24)],names=['station_id','hour'])
    dates = pd.date_range(history.date.min(),origin)
    ci = cells.get_indexer(pd.MultiIndex.from_frame(q[['station_id','hour']]))
    cleanmask = h.pivot(index='date',columns=['station_id','hour'],values='clean').reindex(index=dates,columns=cells).fillna(False).to_numpy(bool)
    rawmask = (~h.set_index(['date','station_id','hour']).oneoff).unstack(['station_id','hour']).reindex(index=dates,columns=cells).fillna(False).to_numpy(bool)
    daycal = calendar_features(pd.Series(dates),as_of=origin)
    daygroup = dow_group(dates.dayofweek.to_numpy(), daycal.is_holiday.to_numpy(bool))
    dayeff = dates.dayofweek.to_numpy()
    arrays = {t:h.pivot(index='date',columns=['station_id','hour'],values=t+'_log').reindex(index=dates,columns=cells).to_numpy() for t in ['board','alight']}
    earlynew = (dates >= '2025-06-28') & (dates <= '2025-09-25')
    newcell = np.isin(cells.get_level_values(0),NEW)
    stored = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for target, arr in arrays.items():
            results = {k:np.full(len(q),np.nan) for k in ['raw_mean','raw_std','q85','clean_med','trimmed','n','growth','trend','line_shape']}
            # Growth uses two observed windows, avoiding a 2024 denominator.
            growth = np.zeros((7,len(cells)))
            if origin >= pd.Timestamp('2026-01-11'):
                start = max(pd.Timestamp('2026-01-01'),origin-pd.Timedelta(days=55))
                recentwin = (dates>=start)&(dates<=origin)
                oldwin = (dates>=start-pd.Timedelta(days=364))&(dates<=origin-pd.Timedelta(days=364))
                for eff in range(7):
                    a=np.where(cleanmask & ((recentwin & (dayeff==eff))[:,None]),arr,np.nan)
                    b=np.where(cleanmask & ((oldwin & (dayeff==eff))[:,None]),arr,np.nan)
                    n=np.minimum(np.isfinite(a).sum(0),np.isfinite(b).sum(0))
                    growth[eff]=np.nan_to_num(n/(n+12)*(np.nanmedian(a,0)-np.nanmedian(b,0)))
                growth[:,newcell]=0
            recent28 = h.loc[h.clean & (h.date>origin-pd.Timedelta(days=28))].groupby(['station_id','hour','eff_dow'])[target+'_log'].median().to_frame('value')
            trendbase=lookup(recent28,q,['station_id','hour','eff_dow'],'value')
            slopes={}
            for key, part in h.loc[h.clean & h.station_id.isin(NEW) & (h.date>origin-pd.Timedelta(days=112))].groupby(['station_id','hour','eff_dow']):
                part=part.copy();part['week']=((part.date-origin).dt.days//7)
                weekly=part.groupby('week')[target+'_log'].median()
                if len(weekly)>=3:
                    u=weekly.index.to_numpy();v=weekly.to_numpy();a,b=np.triu_indices(len(u),1)
                    slopes[key]=np.clip(np.median((v[b]-v[a])/(u[b]-u[a])),-.04,.08)
            slope = np.array([slopes.get(tuple(v),0) for v in q[['station_id','hour','eff_dow']].to_numpy()])
            step=x.horizon.to_numpy()/7; whole=np.floor(step)
            damp=.92*(1-.92**whole)/(1-.92)+(step-whole)*.92**(whole+1)
            results['trend']=np.where(np.isfinite(trendbase),trendbase,x['anchor_'+target+'_log'])+slope*damp
            results['growth']=growth[q.eff_dow.to_numpy(),ci]
            for date, ix in q.groupby('date',sort=False).groups.items():
                ix=np.asarray(ix); center=date-pd.Timedelta(days=364)
                take=(dates>=center-pd.Timedelta(days=21))&(dates<=center+pd.Timedelta(days=21))&(dates<=origin)
                if not take.any():continue
                v=arr[take];w=np.maximum(0,1-np.abs((dates[take]-center).days.to_numpy())/22)[:,None]*np.ones((1,len(cells)))
                w[np.ix_(earlynew[take],newcell)]*=.25
                for eff, ids in q.loc[ix].groupby('eff_dow').groups.items():
                    ids=np.asarray(ids); weights=w*cleanmask[take]*(dayeff[take,None]==eff)
                    stats=weighted_stats(v,weights)
                    for key,stat in [('clean_med','median'),('trimmed','trimmed'),('n','n')]:results[key][ids]=stats[stat][ci[ids]]
                for grp,ids in q.loc[ix].groupby('dow_group').groups.items():
                    ids=np.asarray(ids);stats=weighted_stats(v,w*rawmask[take]*(daygroup[take,None]==grp))
                    for key,stat in [('raw_mean','mean'),('raw_std','std'),('q85','q85')]:results[key][ids]=stats[stat][ci[ids]]
            # Old-station line/hour seasonal shape relative to observed prior-year origin window.
            old = ~q.station_id.isin(NEW)
            seasonal=pd.DataFrame({'line':q.canonical_line,'hour':q.hour,'eff':q.eff_dow,'date':q.date,'value':results['trimmed']})
            base_end=origin-pd.Timedelta(days=364)
            base=h.loc[h.clean & ~h.station_id.isin(NEW) & h.date.between(base_end-pd.Timedelta(days=55),base_end)]
            basetab=base.groupby(['canonical_line','hour','eff_dow'])[target+'_log'].mean().to_frame('v')
            sea=seasonal.loc[old].groupby(['line','hour','eff','date']).value.mean().to_frame('v')
            targetline=lookup(sea,seasonal,['line','hour','eff','date'],'v')
            results['line_shape']=np.nan_to_num(targetline-lookup(basetab,q,['canonical_line','hour','eff_dow'],'v'))
            # Absent seasonal denominator means unknown, explicitly neutral rather than invented 2024 data.
            rolling=x['anchor_'+target+'_log'].to_numpy().copy()
            fourier=fourier_predict(h,q,target+'_log',rolling)
            n=np.nan_to_num(results['n']);w=n/(n+3)
            rawmean=np.where(np.isfinite(results['raw_mean']),results['raw_mean'],results['trimmed'])
            yoy=np.where(x.past_spike_amp>=.75,.65*rawmean+.35*results['trimmed'],results['trimmed'])+results['growth']
            baseline=np.where(n>0,w*np.nan_to_num(yoy)+(1-w)*rolling,.5*fourier+.5*rolling)
            baseline=np.where(q.station_id.isin(NEW),results['trend']+results['line_shape'],baseline)
            x['rolling_anchor_'+target+'_v2']=rolling
            x['anchor_'+target+'_log']=np.maximum(0,baseline)
            if target=='board':
                for key,value in results.items():stored[key]=value
                x['z_fourier_ridge']=fourier
    for key,name in [('raw_mean','z_yoy_raw_mean'),('raw_std','z_yoy_raw_std'),('q85','z_yoy_q85'),('clean_med','z_yoy_clean_med'),('trimmed','z_yoy_trimmed'),('n','n_yoy'),('growth','delta_yoy_shrunk'),('trend','z_new_station_trend'),('line_shape','line_hour_yoy_seas_delta')]:x[name]=stored[key]
    x['n_yoy']=x.n_yoy.fillna(0);x['anchor_v4']=x.anchor_board_log
    x['anchor_v4_minus_v2']=x.anchor_board_log-x.rolling_anchor_board_v2
    x['new_station_log_age']=np.log1p(np.maximum(0,(q.date-pd.Timestamp('2025-06-28')).dt.days))*q.station_id.isin(NEW)
    x=add_special(h,q,x,origin)
    return x.astype('float32')


def add_special(h,q,x,origin):
    normal=h.loc[h.clean & (h.dow<5)].groupby(['station_id','hour']).board_log.median().to_frame('v')
    weekday=lookup(normal,h,['station_id','hour'],'v')
    h=h.copy();h['normal_residual']=h.board_log-weekday
    tomorrow=calendar_features(q.date+pd.Timedelta(days=1),as_of=origin)
    x['is_holiday_eve_night']=tomorrow.is_holiday.to_numpy(bool)&(q.date.dt.dayofweek<5)&(q.hour>=17)
    pastnext=calendar_features(h.date+pd.Timedelta(days=1),as_of=origin)
    eve=h.loc[pastnext.is_holiday.to_numpy(bool)&(h.dow<5)&~h.oneoff]
    tab=eve.groupby(['station_id','hour']).normal_residual.mean().to_frame('v')
    x['holiday_eve_empirical_delta']=lookup(tab,q,['station_id','hour'],'v')*x.is_holiday_eve_night
    special={'labor_day_empirical_delta':['2025-05-01'],'lunar_m1_empirical_delta':['2025-01-28','2025-10-05'],'lunar_day_0_empirical_delta':['2025-01-29','2025-10-06'],'lunar_p1_empirical_delta':['2025-01-30','2025-10-07']}
    for name,ds in special.items():
        observed=h.loc[h.date.isin(pd.to_datetime(ds)) & ~h.oneoff]
        tab=observed.groupby(['station_id','hour']).normal_residual.mean().to_frame('v')
        fallback=observed.groupby(['canonical_line','hour']).normal_residual.mean().to_frame('v')
        v=lookup(tab,q,['station_id','hour'],'v');v=np.where(np.isfinite(v),v,lookup(fallback,q,['canonical_line','hour'],'v'))
        flag=x.is_labor_day if name.startswith('labor') else x[name.replace('_empirical_delta','')]
        x[name]=np.nan_to_num(v)*flag
    # Past afternoon alighting surge and evening response: same calendar month/group only.
    cleanraw=h.loc[~h.oneoff].copy();cleanraw['month']=cleanraw.date.dt.month
    daily=cleanraw.loc[cleanraw.hour.between(13,18)].groupby(['station_id','date']).alightings.sum().rename('afternoon').reset_index()
    daily['month']=daily.date.dt.month
    median=daily.groupby(['station_id','month']).afternoon.transform('median')
    daily['surge']=daily.afternoon>np.maximum(1,median)*1.8
    merged=cleanraw.merge(daily[['station_id','date','surge']],on=['station_id','date'],how='left')
    keys=['station_id','hour','month','dow_group'];part=q.copy();part['month']=part.date.dt.month
    probability=merged.groupby(keys).surge.mean().to_frame('v')
    lift=merged.loc[merged.surge.fillna(False)&merged.hour.between(19,23)].groupby(keys).normal_residual.mean().to_frame('v')
    x['alight_surge_probability']=lookup(probability,part,keys,'v')
    x['alight_surge_to_board_lift']=lookup(lift,part,keys,'v')
    return x
