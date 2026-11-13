"""Causal partial-horizon snapshots and cleaned log demand features for v2."""
import numpy as np
import pandas as pd
from .holiday_calendar import calendar_features

CATS=['station_id','canonical_line','hour','dow','eff_dow','holiday_subtype','station_hour','station_eff_dow']
NEW=[3107,3108,3109]
TRANSFERS={3118:3761,3761:3118,3124:3221,3221:3124,3213:3763,3763:3213}

def calendar(dates,origin):
    d=pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    c=calendar_features(d,as_of=origin).reset_index(drop=True)
    c['eff_dow']=np.where(c.is_holiday.astype(bool)&~c.is_labor_day.astype(bool),6,d.dt.dayofweek)
    c['holiday_subtype']=c.holiday_type
    for key in ['lunar_day_0','lunar_m1','lunar_p1']:c[key]=0
    centers=['2025-01-29','2025-10-06','2026-02-17','2026-09-25']
    for center in centers:
        for key,offset in [('lunar_day_0',0),('lunar_m1',-1),('lunar_p1',1)]:
            c.loc[(d==pd.Timestamp(center)+pd.Timedelta(days=offset))&c.is_lunar_holiday.astype(bool),key]=1
    # First ordinary weekday after a holiday block, allowing an intervening weekend.
    previous=[calendar_features(d-pd.Timedelta(days=k),as_of=origin) if (d-pd.Timedelta(days=k)).min()>=pd.Timestamp('2025-01-01') else None for k in range(1,5)]
    post=np.zeros(len(d),bool);between=np.ones(len(d),bool)
    for cprev in previous:
        if cprev is None:continue
        post|=between&cprev.is_holiday.to_numpy(dtype=bool)
        between&=cprev.is_rest_day.to_numpy(dtype=bool)
    c['post_holiday_workday']=post&~c.is_rest_day.to_numpy(dtype=bool)
    md=d.dt.month*100+d.dt.day
    # Generic seasonal proxies requested by user, not verified university schedules.
    c['summer_break_proxy']=md.between(622,831)
    c['winter_break_proxy']=(md>=1222)|(md<=228)
    c['semester_first2w_proxy']=md.between(302,315)|md.between(901,914)
    return c

def cleaned(history,origin):
    h=history.loc[history.date<=pd.Timestamp(origin)].copy().reset_index(drop=True)
    c=calendar_features(h.date,as_of=origin).reset_index(drop=True)
    h['dow']=h.date.dt.dayofweek;h['eff_dow']=h.dow
    h['holiday']=c.is_holiday.to_numpy(dtype=bool);h['rest']=c.is_rest_day.to_numpy(dtype=int)
    h['opening']=h.station_id.isin(NEW)&h.date.between('2025-06-28','2025-06-30')
    h['clean']=~(h.holiday|c.is_labor_day.to_numpy(dtype=bool)|h.opening)
    if pd.Timestamp(origin)>=pd.Timestamp('2025-07-28'):
        h.loc[h.station_id.isin([3110,3111])&(h.date<'2025-07-01'),'clean']=False
    h['board_log']=np.log1p(h.boardings);h['alight_log']=np.log1p(h.alightings)
    return h

def lookup(table,q,keys,col):
    return table[col].reindex(pd.MultiIndex.from_frame(q[keys]) if len(keys)>1 else q[keys[0]]).to_numpy()

def make_features_v2(history,query,meta,origin):
    origin=pd.Timestamp(origin)
    q=query[['date','station_id','hour']].copy().reset_index(drop=True)
    if (q.date<=origin).any():raise ValueError('Query must be after origin')
    h=cleaned(history,origin);line_map=meta.set_index('station_id').canonical_line
    h['canonical_line']=h.station_id.map(line_map)
    q['canonical_line']=q.station_id.map(line_map);q['dow']=q.date.dt.dayofweek
    q=pd.concat([q,calendar(q.date,origin)],axis=1)
    q['horizon']=(q.date-origin).dt.days;q['month']=q.date.dt.month;q['day_of_month']=q.date.dt.day
    q['day_of_year']=q.date.dt.dayofyear
    for col,period in [('hour',24),('dow',7),('day_of_year',365.25)]:
        q[col+'_sin']=np.sin(2*np.pi*q[col]/period);q[col+'_cos']=np.cos(2*np.pi*q[col]/period)
    first=h.groupby('station_id').date.min();q['history_days']=(origin-q.station_id.map(first)).dt.days+1
    q['cold_start']=q.history_days.isna().astype(int)
    keys=['station_id','hour','eff_dow'];profiles={}
    clean=h.loc[h.clean]
    for days in [28,56,112]:
        w=clean.loc[clean.date>origin-pd.Timedelta(days=days)]
        tab=w.groupby(keys).agg(board_mean=('board_log','mean'),board_median=('board_log','median'),board_std=('board_log','std'),alight_mean=('alight_log','mean'),alight_median=('alight_log','median'),alight_std=('alight_log','std'),count=('board_log','size'))
        profiles[days]=tab
        for col in tab:q[f'{col}_{days}']=lookup(tab,q,keys,col)
    recent=clean.loc[clean.date>origin-pd.Timedelta(days=112)]
    fallback_line=recent.groupby(['canonical_line','hour','eff_dow'])[['board_log','alight_log']].mean()
    fallback_global=recent.groupby(['hour','eff_dow'])[['board_log','alight_log']].mean()
    def anchor(rows,target):
        a=lookup(profiles[56],rows,keys,target+'_mean');b=lookup(profiles[112],rows,keys,target+'_mean')
        a=np.where(np.isfinite(a),a,b)
        b=lookup(fallback_line,rows,['canonical_line','hour','eff_dow'],target+'_log');a=np.where(np.isfinite(a),a,b)
        b=lookup(fallback_global,rows,['hour','eff_dow'],target+'_log');a=np.where(np.isfinite(a),a,b)
        a=np.where(np.isfinite(a),a,float(clean[target+'_log'].mean()))
        short=lookup(profiles[28],rows,keys,target+'_mean')
        return .6*a+.4*np.where(np.isfinite(short),short,a)
    q['anchor_board_log']=anchor(q,'board');q['anchor_alight_log']=anchor(q,'alight')
    cal=q.copy();cal['eff_dow']=cal.dow;q['cal_dow_log_mean']=lookup(profiles[56],cal,keys,'board_mean')
    cal['eff_dow']=6;q['sun_log_mean']=lookup(profiles[56],cal,keys,'board_mean')
    q['sun_minus_wkd_log']=q.sun_log_mean-q.cal_dow_log_mean
    q['growth_log']=q.board_mean_28-q.board_mean_112
    q['board_alight_asymmetry']=q.board_mean_56-q.alight_mean_56
    # Historical holiday elasticity, never future labels.
    ordinary=h.loc[h.clean&(h.dow<5)].groupby(['station_id','hour']).board_log.mean().to_frame('ordinary')
    holiday_mask=h.holiday&~h.opening&~((h.date.dt.month==5)&(h.date.dt.day==1))
    if origin>=pd.Timestamp('2025-07-28'):
        holiday_mask&=~(h.station_id.isin([3110,3111])&(h.date<'2025-07-01'))
    # Holiday observations are needed only for this explicit contrast feature;
    # the ordinary side uses the same cleaned structural regime as the anchors.
    holiday=h.loc[holiday_mask].groupby(['station_id','hour']).board_log.mean().to_frame('holiday')
    q['holiday_elasticity']=lookup(holiday,q,['station_id','hour'],'holiday')-lookup(ordinary,q,['station_id','hour'],'ordinary')
    shares=recent.groupby(['station_id','hour']).boardings.mean();shares=shares/shares.groupby(level=0).transform('sum')
    q['hour_share']=lookup(shares.to_frame('share'),q,['station_id','hour'],'share')
    # Exact prior-year lookup, replace holiday/opening observations with adjacent weeks.
    eligible=h.loc[~h.holiday&~h.opening&~((h.date.dt.month==5)&(h.date.dt.day==1))]
    prior=eligible.set_index(['date','station_id','hour'])[['board_log']]
    year=np.full(len(q),np.nan)
    for days in [364,357,371]:
        dates=q[['date','station_id','hour']].copy();dates['date']=dates.date-pd.Timedelta(days=days)
        val=lookup(prior,dates,['date','station_id','hour'],'board_log')
        val[dates.date>origin]=np.nan;year=np.where(np.isfinite(year),year,val)
    q['year364_clean_log']=year;q['year364_missing']=~np.isfinite(year)
    # Previous target month minus the 56-day window preceding origin one year earlier.
    baseline_end=origin-pd.DateOffset(years=1)
    base=clean.loc[(clean.date>baseline_end-pd.Timedelta(days=56))&(clean.date<=baseline_end)]
    base_station=base.groupby(['station_id','rest']).agg(value=('board_log','mean'),days=('date','nunique'))
    base_line=base.groupby(['canonical_line','hour','rest']).board_log.mean().to_frame('value')
    q['yoy_seas_log_delta']=np.nan;q['yoy_seas_fallback']=1
    for (year_now,month),indices in q.groupby([q.date.dt.year,q.date.dt.month]).groups.items():
        part=q.loc[indices].copy();part['rest']=part.is_rest_day
        previous=clean.loc[(clean.date.dt.year==year_now-1)&(clean.date.dt.month==month)]
        month_station=previous.groupby(['station_id','rest']).board_log.mean().to_frame('value')
        month_line=previous.groupby(['canonical_line','hour','rest']).board_log.mean().to_frame('value')
        a=lookup(month_station,part,['station_id','rest'],'value')-lookup(base_station,part,['station_id','rest'],'value')
        enough=part.station_id.map(base.groupby('station_id').date.nunique()).to_numpy()>=14
        b=lookup(month_line,part,['canonical_line','hour','rest'],'value')-lookup(base_line,part,['canonical_line','hour','rest'],'value')
        use=enough&np.isfinite(a);delta=np.where(use,a,b)
        q.loc[indices,'yoy_seas_fallback']=(~use).astype(int)
        q.loc[indices,'yoy_seas_log_delta']=delta
    q['yoy_seas_missing']=q.yoy_seas_log_delta.isna();q['yoy_seas_log_delta']=q.yoy_seas_log_delta.fillna(0)
    for label,ids in [('previous',q.station_id-1),('next',q.station_id+1),('transfer',q.station_id.map(TRANSFERS))]:
        other=q[['station_id','canonical_line','hour','eff_dow']].copy();other['station_id']=ids
        other['canonical_line']=ids.map(line_map)
        valid=other.canonical_line.notna()
        if label!='transfer':valid&=other.canonical_line==q.canonical_line
        value=anchor(other,'board');value[~valid]=np.nan
        q[f'{label}_anchor_log']=value
        q[f'{label}_growth_log']=lookup(profiles[28],other,keys,'board_mean')-lookup(profiles[112],other,keys,'board_mean')
        q.loc[~valid,f'{label}_growth_log']=np.nan
        q[f'{label}_log_ratio']=q.anchor_board_log-q[f'{label}_anchor_log']
    q['station_hour']=q.station_id*24+q.hour;q['station_eff_dow']=q.station_id*7+q.eff_dow
    return q.drop(columns='date').astype('float32')

def partial_origins(cutoff,start='2025-02-11'):
    end=pd.Timestamp(cutoff)-pd.Timedelta(days=14)
    origins=list(pd.date_range(start,end,freq='14D'))
    if origins and origins[-1]!=end:origins.append(end)
    return origins

def sampled_pairs(train,cutoff,max_rows,seed=2026):
    cutoff=pd.Timestamp(cutoff);origins=partial_origins(cutoff)
    dates={d:idx.to_numpy() for d,idx in train.loc[train.date<=cutoff].groupby('date').groups.items()}
    rng=np.random.default_rng(seed);pairs=[]
    for horizon in range(1,124):
        rows=[];os=[];weights=[]
        for oi,o in enumerate(origins):
            d=o+pd.Timedelta(days=horizon)
            if d>cutoff or d not in dates:continue
            ids=dates[d];rows.extend(ids);os.extend([oi]*len(ids))
            w=.5 if o<pd.Timestamp('2025-07-01') else 1.5 if o<pd.Timestamp('2025-12-30') else 2.5
            weights.extend([w]*len(ids))
        n=min(len(rows),max_rows//123+(horizon<=max_rows%123))
        if n:
            w=np.asarray(weights);ix=rng.choice(len(rows),n,replace=False,p=w/w.sum())
            pairs.append(pd.DataFrame({'row':np.asarray(rows)[ix],'origin_index':np.asarray(os)[ix],'horizon':horizon}))
    return pd.concat(pairs,ignore_index=True),origins

def training_matrix_v2(train,meta,cutoff,max_rows,seed=2026):
    pairs,origins=sampled_pairs(train,cutoff,max_rows,seed);xs=[];ys=[];audit=[]
    for oi,p in pairs.groupby('origin_index',sort=True):
        o=origins[oi];labels=train.loc[p.row].reset_index(drop=True)
        assert labels.date.min()>o and labels.date.max()<=pd.Timestamp(cutoff)
        x=make_features_v2(train,labels,meta,o)
        y=np.column_stack([np.log1p(labels.boardings)-x.anchor_board_log,np.log1p(labels.alightings)-x.anchor_alight_log]).astype('float32')
        xs.append(x);ys.append(y)
        audit.append({'origin':str(o.date()),'rows':len(p),'label_max':str(labels.date.max().date()),'min_horizon':int(p.horizon.min()),'max_horizon':int(p.horizon.max())})
    return pd.concat(xs,ignore_index=True),np.concatenate(ys),audit,pairs.horizon.value_counts().sort_index().to_dict()
