"""Day-level priors from ten years of daily card data (line-7 stations 2016-2019, transfer-partner stations 2016-2019 and
2022 onwards).  The file ends on 2026-06-30: no ridership of the evaluation period, of any line, is in it."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.v8 import core as c8

EXT = c8.ROOT / 'outputs/ensemble_v14/external'
CAL_FEATS = ['weekday', 'is_holiday', 'offday_block_len', 'offday_block_pos', 'is_before_offday', 'is_after_offday', 'is_bridge_day', 'is_labor_day',
             'is_seollal_period', 'is_chuseok_period', 'mj_day', 'days_to_mj', 'month']


class CalendarPrior:
    """log(day boardings / ordinary same-weekday level) as a function of calendar position."""

    def __init__(self):
        d = pd.read_csv(EXT / 'seoul_daily_2015_2026h1.csv', parse_dates=['date']); d['key'] = d.line + ' ' + d.station_name
        assert d.date.max() <= pd.Timestamp('2026-06-30')
        self.cal = pd.read_csv(EXT / 'calendar.csv', encoding='utf-8-sig', parse_dates=['date']).set_index('date')
        P = np.log(d.pivot_table(index='date', columns='key', values='boardings').clip(lower=1)); P = P[P.index >= '2016-01-01']; c = self.cal.reindex(P.index)
        self.ordinary = ((c.is_holiday == 0) & (c.is_bridge_day == 0) & (c.is_labor_day == 0) & (c.is_mj_period == 0)).to_numpy()
        Pm = P.copy(); Pm[~self.ordinary] = np.nan
        with np.errstate(all='ignore'):
            base = np.nanmedian(np.stack([Pm.shift(7 * w, freq='D').reindex(P.index).to_numpy() for w in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)]), 0)
        self.P, self.Y = P, P - base
        self.line7 = [k for k in P.columns if k.startswith('7호선') and '온수' not in k]
        self.covid = (P.index >= '2020-02-01') & (P.index <= '2022-04-30')

    def fit(self, origin):
        """model on rows dated before `origin`; returns dates -> prior (mean of the line-7 view and the partner view)."""
        rows = []
        for key in self.P.columns:
            m = (self.P.index < pd.Timestamp(origin)) & ~self.covid & self.Y[key].notna().to_numpy()
            f = self.cal.loc[self.P.index[m], CAL_FEATS].copy(); f['line7'] = int(key in self.line7); f['y'] = self.Y[key].to_numpy()[m]; rows.append(f)
        T = pd.concat(rows); T['days_to_mj'] = T.days_to_mj.clip(-8, 8)
        ms = [lgb.train(dict(objective='regression', learning_rate=0.05, num_leaves=15, min_data_in_leaf=15, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                             lambda_l2=1.0, verbose=-1, num_threads=8, seed=sd), lgb.Dataset(T[CAL_FEATS + ['line7']], T.y), 300) for sd in (1, 2, 3)]

        def predict(dates):
            f = self.cal.reindex(pd.DatetimeIndex(dates))[CAL_FEATS].copy(); f['days_to_mj'] = f.days_to_mj.clip(-8, 8); out = []
            for l7 in (1, 0):
                f['line7'] = l7; out.append(np.mean([m.predict(f[CAL_FEATS + ['line7']]) for m in ms], 0))
            return np.mean(out, 0)
        return predict

    def weekly_index(self, before_year):
        """[day type (weekday, Sat, Sun), ISO week 1..52] -> median over station-years (< before_year, no covid years) of the
        week's mean log boardings on ordinary days minus the station-year mean; smoothed over neighbouring weeks."""
        idx = self.P.index; dow = idx.dayofweek.to_numpy(); dt = np.where(dow < 5, 0, np.where(dow == 5, 1, 2)); wk = np.minimum(idx.isocalendar().week.to_numpy().astype(int), 52)
        out = np.zeros((3, 53))
        for t in range(3):
            acc = []
            for y in (2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025):
                if y >= before_year:
                    continue
                m = (idx.year == y) & (dt == t) & self.ordinary; tab = self.P[m].groupby(wk[m]).mean(); acc.append((tab - tab.mean()).reindex(range(1, 53)).to_numpy())
            with np.errstate(all='ignore'):
                v = np.nanmedian(np.concatenate(acc, 1), 1)
            v = np.where(np.isnan(v), np.nanmean(v), v); out[t, 1:] = 0.25 * np.roll(v, 1) + 0.5 * v + 0.25 * np.roll(v, -1)
        return out
