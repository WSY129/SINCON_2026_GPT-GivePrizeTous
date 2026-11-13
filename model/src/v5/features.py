"""Causal seasonal kernels and regime anchors, retaining v3 covariates."""
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from ..features_v3 import make_features_v3, cleaned, lookup, NEW, CATS, calendar
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
    return dict(mean=mean, std=std, median=quantile(.5), q85=quantile(.85), q90=quantile(.9), trimmed=trimmed, n=count)


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


REGIME_START = pd.Timestamp('2025-07-01')


def protected_window(dates, center, origin, radius=21):
    dates = pd.DatetimeIndex(dates)
    mask = (dates >= center-pd.Timedelta(days=radius)) & (dates <= center+pd.Timedelta(days=radius)) & (dates <= origin)
    if center >= REGIME_START:
        mask &= dates >= REGIME_START
    return mask


def matched_growth(dates, arr, cleanmask, origin, effs, newcell, post_regime):
    """Matched dates only; 14 distinct usable overlap days per station/cell.

    The 14-day gate precedes weekday grouping (not 14 weekly observations).
    Both current and shifted windows have the same paired observations.
    """
    out = np.zeros((7, arr.shape[1])); counts = np.zeros(arr.shape[1],dtype=int)
    if origin < pd.Timestamp('2026-01-11'): return out, counts
    start = max(pd.Timestamp('2026-01-01'), origin-pd.Timedelta(days=55))
    recent = np.flatnonzero((dates >= start) & (dates <= origin))
    previous = dates.get_indexer(dates[recent]-pd.Timedelta(days=364))
    good = previous >= 0
    if post_regime: good &= (dates[recent]-pd.Timedelta(days=364)) >= REGIME_START
    recent, previous = recent[good], previous[good]
    if not len(recent): return out, counts
    valid = cleanmask[recent] & cleanmask[previous] & np.isfinite(arr[recent]) & np.isfinite(arr[previous])
    counts = valid.sum(0)
    for eff in range(7):
        mask = valid & (effs[recent,None] == eff)
        n = mask.sum(0)
        delta = np.nanmedian(np.where(mask,arr[recent],np.nan),axis=0)-np.nanmedian(np.where(mask,arr[previous],np.nan),axis=0)
        out[eff] = np.where(counts >= 14,np.nan_to_num(n/(n+12)*delta),0)
    out[:,newcell] = 0
    return out, counts


def recent_deseasonalized_slopes(h, origin, target):
    """Eight-week, within-weekday trend minus contemporaneous old-line trend."""
    part=h.loc[h.clean & (h.date>origin-pd.Timedelta(days=56))].copy()
    part['week']=(part.date-origin).dt.days//7
    old=part.loc[~part.station_id.isin(NEW+[3110,3111])].copy()
    keys=['station_id','hour','eff_dow']
    old['centered']=old[target+'_log']-old.groupby(keys)[target+'_log'].transform('median')
    season=old.groupby(['canonical_line','hour','eff_dow','week']).centered.median().rename('season')
    new=part.loc[part.station_id.isin(NEW)].join(season,on=['canonical_line','hour','eff_dow','week'])
    new['deseasonalized']=new[target+'_log']-new.season.fillna(0)
    slopes={}
    for key,g in new.groupby(keys):
        weekly=g.groupby('week').deseasonalized.median()
        if len(weekly)>=3:
            u=weekly.index.to_numpy();v=weekly.to_numpy();a,b=np.triu_indices(len(u),1)
            slopes[key]=float(np.clip(np.median((v[b]-v[a])/(u[b]-u[a])),0,.015))
    return slopes,new.groupby(keys).deseasonalized.median().to_frame('value')


def make_features(history, query, meta, origin):
    origin = pd.Timestamp(origin)
    history = history.loc[history.date <= origin].copy()
    if origin >= pd.Timestamp('2025-07-28'):
        history=history.loc[~(history.station_id.isin([3110,3111]) & (history.date<REGIME_START))]
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
            results = {k:np.full(len(q),np.nan) for k in ['raw_mean','raw_std','q85','clean_med','trimmed','n','growth','trend','line_shape','window_amp','narrow_med','narrow_n','growth_overlap_days','trend_slope']}
            growth, overlap = matched_growth(dates,arr,cleanmask,origin,dayeff,newcell,False)
            post_growth, post_overlap = matched_growth(dates,arr,cleanmask,origin,dayeff,newcell,True)
            slopes,recentbase=recent_deseasonalized_slopes(h,origin,target)
            trendbase=lookup(recentbase,q,['station_id','hour','eff_dow'],'value')
            slope = np.array([slopes.get(tuple(v),0) for v in q[['station_id','hour','eff_dow']].to_numpy()])
            step=x.horizon.to_numpy()/7; whole=np.floor(step)
            damp=.90*(1-.90**whole)/(1-.90)+(step-whole)*.90**(whole+1)
            results['trend']=np.where(np.isfinite(trendbase),trendbase,x['anchor_'+target+'_log'])+slope*damp
            post=(q.date-pd.Timedelta(days=364)>=REGIME_START).to_numpy()
            results['growth']=np.where(post,post_growth[q.eff_dow.to_numpy(),ci],growth[q.eff_dow.to_numpy(),ci])
            results['growth_overlap_days']=np.where(post,post_overlap[ci],overlap[ci])
            results['trend_slope']=slope
            for date, ix in q.groupby('date',sort=False).groups.items():
                ix=np.asarray(ix); center=date-pd.Timedelta(days=364)
                take=protected_window(dates,center,origin,21)
                if not take.any():continue
                v=arr[take];w=np.maximum(0,1-np.abs((dates[take]-center).days.to_numpy())/22)[:,None]*np.ones((1,len(cells)))
                for eff, ids in q.loc[ix].groupby('eff_dow').groups.items():
                    ids=np.asarray(ids); weights=w*cleanmask[take]*(dayeff[take,None]==eff)
                    stats=weighted_stats(v,weights)
                    for key,stat in [('clean_med','median'),('trimmed','trimmed'),('n','n')]:results[key][ids]=stats[stat][ci[ids]]
                    offsets=np.abs((dates[take]-center).days.to_numpy())
                    narrow=np.maximum(0,1-offsets/15)[:,None]*cleanmask[take]*(dayeff[take,None]==eff)
                    narrowstats=weighted_stats(v,narrow)
                    results['narrow_med'][ids]=narrowstats['median'][ci[ids]]
                    results['narrow_n'][ids]=narrowstats['n'][ci[ids]]
                for grp,ids in q.loc[ix].groupby('dow_group').groups.items():
                    ids=np.asarray(ids);stats=weighted_stats(v,np.ones_like(v)*rawmask[take]*(daygroup[take,None]==grp))
                    for key,stat in [('raw_mean','mean'),('raw_std','std'),('q85','q85')]:results[key][ids]=stats[stat][ci[ids]]
                    results['window_amp'][ids]=(stats['q90']-stats['median'])[ci[ids]]
            # Old-station line/hour seasonal shape relative to observed prior-year origin window.
            old = ~q.station_id.isin(NEW+[3110,3111])
            seasonal=pd.DataFrame({'line':q.canonical_line,'hour':q.hour,'eff':q.eff_dow,'date':q.date,'value':results['trimmed']})
            base_end=origin-pd.Timedelta(days=364)
            base=h.loc[h.clean & ~h.station_id.isin(NEW+[3110,3111]) & h.date.between(base_end-pd.Timedelta(days=55),base_end)]
            basetab=base.groupby(['canonical_line','hour','eff_dow'])[target+'_log'].mean().to_frame('v')
            sea=seasonal.loc[old].groupby(['line','hour','eff','date']).value.mean().to_frame('v')
            targetline=lookup(sea,seasonal,['line','hour','eff','date'],'v')
            results['line_shape']=np.nan_to_num(targetline-lookup(basetab,q,['canonical_line','hour','eff_dow'],'v'))
            # Seasonal change for new stations: old-line prior target versus available
            # previous-year recent window. Do not infer a PRE->POST annual growth.
            postbase=base.loc[base.date>=REGIME_START]
            posttab=postbase.groupby(['canonical_line','hour','eff_dow'])[target+'_log'].mean().to_frame('v')
            coverage=postbase.groupby(['canonical_line','hour']).date.nunique().to_frame('days')
            postdelta=targetline-lookup(posttab,q,['canonical_line','hour','eff_dow'],'v')
            enough=lookup(coverage,q,['canonical_line','hour'],'days')>=14
            # With no matched post-regime denominator use old-line target season
            # relative to the last 56 observed days, centered by station to avoid
            # composition changes. This is a seasonal profile, not growth addition.
            recentold=h.loc[h.clean & ~h.station_id.isin(NEW+[3110,3111]) & (h.date>origin-pd.Timedelta(days=56))]
            currenttab=recentold.groupby(['canonical_line','hour','eff_dow'])[target+'_log'].mean().to_frame('v')
            fallbackshape=np.nan_to_num(targetline-lookup(currenttab,q,['canonical_line','hour','eff_dow'],'v'))
            results['line_shape'][post]=np.where(enough[post],np.nan_to_num(postdelta[post]),fallbackshape[post])
            # Absent seasonal denominator means unknown, explicitly neutral rather than invented 2024 data.
            rolling=x['anchor_'+target+'_log'].to_numpy().copy()
            fourier=fourier_predict(h,q,target+'_log',rolling)
            cleanvalue=np.where(results['narrow_n']>=2,results['narrow_med'],results['clean_med'])
            results['n']=np.where(results['narrow_n']>=2,results['narrow_n'],results['n'])
            n=np.nan_to_num(results['n']);w=n/(n+3)
            rawmean=np.where(np.isfinite(results['raw_mean']),results['raw_mean'],cleanvalue)
            cleanvalue=np.where(np.isfinite(cleanvalue),cleanvalue,rawmean)
            alpha=np.clip((np.nan_to_num(results['window_amp'])-.45)/.45,0,1)
            yoy=alpha*rawmean+(1-alpha)*cleanvalue+results['growth']
            baseline=np.where(np.isfinite(yoy),yoy,.5*fourier+.5*rolling)
            baseline=np.where(q.station_id.isin(NEW),results['trend']+results['line_shape'],baseline)
            x['rolling_anchor_'+target+'_v2']=rolling
            x['anchor_'+target+'_log']=np.maximum(0,baseline)
            if target=='board':
                for key,value in results.items():stored[key]=value
                x['z_fourier_ridge']=fourier
    for key,name in [('raw_mean','z_yoy_raw_mean'),('raw_std','z_yoy_raw_std'),('q85','z_yoy_q85'),('clean_med','z_yoy_clean_med'),('trimmed','z_yoy_trimmed'),('n','n_yoy'),('growth','delta_yoy_shrunk'),('trend','z_new_station_trend'),('line_shape','line_hour_yoy_seas_delta'),('window_amp','window_spike_amp'),('narrow_med','z_yoy_narrow_median'),('growth_overlap_days','growth_overlap_days'),('trend_slope','new_station_weekly_slope')]:x[name]=stored[key]
    x['n_yoy']=x.n_yoy.fillna(0);x['anchor_v5']=x.anchor_board_log
    x['new_station_log_age']=np.log1p(np.maximum(0,(q.date-pd.Timestamp('2025-06-28')).dt.days))*q.station_id.isin(NEW)
    x=add_special(h,q,x,origin)
    x=add_anchor_holiday_lifts(h,q,x,origin)
    x['anchor_v5']=x.anchor_board_log
    x['rolling_anchor_v2']=x.rolling_anchor_board_v2
    x['anchor_v5_minus_v2']=x.anchor_board_log-x.rolling_anchor_board_v2
    assert {'year364_clean_log','yoy_seas_log_delta'}.issubset(x.columns)
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


def holiday_kinds(dates,origin):
    c=calendar(dates,origin)
    return np.select([c.lunar_m1.astype(bool),c.lunar_day_0.astype(bool),c.lunar_p1.astype(bool),c.is_election_day.astype(bool),c.is_substitute_holiday.astype(bool),c.is_holiday.astype(bool)],['lunar_m1','lunar_d0','lunar_p1','election','substitute','legal'],default='ordinary')


def add_anchor_holiday_lifts(h,q,x,origin):
    """Past holiday impact relative to its contemporaneous ordinary baseline."""
    h=h.copy();h['hol_type']=holiday_kinds(h.date,origin)
    q=q.copy();q['hol_type']=holiday_kinds(q.date,origin)
    for target in ['board','alight']:
        col=target+'_log';keys=['station_id','hour'];linekeys=['canonical_line','hour']
        labor=np.zeros(len(q));holiday=np.zeros(len(q))
        if origin>=pd.Timestamp('2025-05-02'):
            base=h.loc[h.clean & (h.dow<5) & h.date.between('2025-03-06','2025-04-30')].groupby(keys)[col].median().to_frame('v')
            obs=h.loc[h.date==pd.Timestamp('2025-05-01')].copy()
            obs['delta']=(obs[col]-lookup(base,obs,keys,'v')).clip(-1.8,.8)
            st=obs.groupby(keys).delta.mean().to_frame('v');ln=obs.groupby(linekeys).delta.mean().to_frame('v')
            a=lookup(st,q,keys,'v');b=lookup(ln,q,linekeys,'v');b=np.nan_to_num(b)
            labor=(.85*np.where(np.isfinite(a),a,b)+.15*b)*x.is_labor_day.to_numpy()
        records=[]
        for date,obs in h.loc[(h.hol_type!='ordinary') & ~h.oneoff].groupby('date'):
            base=h.loc[h.clean & (h.dow==6) & (h.date<date) & (h.date>=date-pd.Timedelta(days=56))]
            tab=base.groupby(keys)[col].median().to_frame('v')
            obs=obs.copy();obs['delta']=obs[col]-lookup(tab,obs,keys,'v');records.append(obs)
        if records:
            impacts=pd.concat(records).dropna(subset=['delta'])
            sk=keys+['hol_type'];lk=linekeys+['hol_type']
            st=impacts.groupby(sk).delta.agg(['mean','count']);ln=impacts.groupby(lk).delta.mean().to_frame('v')
            a=lookup(st,q,sk,'mean');n=np.nan_to_num(lookup(st,q,sk,'count'));b=np.nan_to_num(lookup(ln,q,lk,'v'))
            holiday=np.where(q.hol_type!='ordinary',n/(n+4)*np.nan_to_num(a)+4/(n+4)*b,0)
        x['labor_delta_'+target]=labor;x['holiday_sunday_lift_'+target]=holiday
        x['anchor_'+target+'_log']=np.maximum(0,x['anchor_'+target+'_log'].to_numpy()+labor+holiday)
    x['labor_delta']=x.labor_delta_board;x['holiday_sunday_lift']=x.holiday_sunday_lift_board
    return x
