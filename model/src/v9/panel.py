"""Panel of ensemble_v9: one structural anchor and the inputs a model needs to learn season and weather.

anchor(origin, day, station, hour) = level at the origin (median of the last 8 same-weekday ordinary days)
                                   + multi-year seasonal change of the target month (operator history 2015-2024)
                                   + station x hour effect of the day's holiday category (measured before the origin)
target r = log1p(y) - anchor.

History before 2026 is first converted to the 2026 aggregation basis (the train file is 0.74-1.00 of the operator's
monthly totals in 2025 and equal to them in 2026), so levels, trends and year-over-year inputs are one series.
Inputs added to the v7 panel: the seasonal prior, the holiday effect, observed weather of the target day and hour
(allowed for the evaluation period by the organisers), last year's rain, rain inside the level window, weekend
ratios, and the games actually played at the two stadium stations.
"""
import json
import numpy as np
import pandas as pd
from src.v7 import panel as p7
from src.v7.panel import Panel, FI as FI7, _nan
from src.v8 import core as c8, pre

EXTRA = ['prior', 'prior_ok', 'hol_eff', 'cat', 'anchor_gap', 'wet', 'rain_mm', 'rain_h', 'rain_prev2', 'temp_h', 'tmean', 'tmax',
         'wet_ly', 'wet_share8', 'r_sat', 'r_sun', 'commute', 'basis', 'game_att', 'game_rel']
FEATURES = list(p7.FEATURES) + EXTRA
FI = {name: i for i, name in enumerate(FEATURES)}
YOY = p7.YOY
CATEGORICAL = p7.CATEGORICAL + ['cat', 'wet']
CAT_CODE = {k: i for i, k in enumerate(('wd', 'sat', 'sun') + c8.SPECIAL)}
HORIZON, MIN_HISTORY = p7.HORIZON, p7.MIN_HISTORY
BASIS_CUT = '2026-01-01'


class Panel9(Panel):
    def __init__(self, root='.', schedule=None, actual=True):
        """actual=False: nothing realised after the data ends is loaded (no evaluation-period weather, no game results)."""
        super().__init__(root, schedule)
        self.actual = actual
        f = pd.read_csv(c8.SHARE / 'ictr_to_train_factor_station_hour.csv')
        self.logf = np.zeros((self.S, self.H), np.float32)
        self.logf[np.searchsorted(self.stations, f.station_id.to_numpy()), f.hour.to_numpy() - 6] = np.log(f.factor_boardings.to_numpy())
        self.cut = self.day_index(BASIS_CUT)
        self.Z[:, :self.cut] -= self.logf[:, None, :]                       # 2025 history on the 2026 basis
        self.T = np.log1p(_nan(np.nansum, np.expm1(self.Z), 2)).astype(np.float32)
        self.T[np.isnan(self.Z).all(2)] = np.nan
        self.history9 = pre.History(self)
        h = json.loads((c8.EXT / 'open_meteo_incheon_hourly_20250101_20261031.json').read_text())['hourly']
        t = pd.to_datetime(h['time']); di = (t.normalize() - p7.START).days.to_numpy(); hr = t.hour.to_numpy()
        self.rain_h = np.zeros((len(self.days), 24), np.float32); self.temp_h = np.full((len(self.days), 24), np.nan, np.float32)
        self.rain_h[di, hr] = np.asarray(h['precipitation'], np.float32); self.temp_h[di, hr] = np.asarray(h['temperature_2m'], np.float32)
        if not actual:
            self.rain_h[self.D:] = 0.0; self.temp_h[self.D:] = np.nan
        self.rain_d = self.rain_h[:, 6:24].sum(1)
        self.wet = (self.rain_d >= 1).astype(np.float32) + (self.rain_d >= 10)
        games = []
        for name, sid, pad in (('games_ssg_munhak.csv', 3127, 0), ('games_bucheonfc.csv', 3754, 115)):
            g = pd.read_csv(c8.SHARE / name, parse_dates=['date'])
            if sid == 3127:
                g = g[(g.status == 'played') & g.attendance.notna() & g.end_time.notna()]; end = g.end_time
            else:
                g = g[(g.status == 'FE') & (g.attendance > 0)]; end = g.kickoff
            minute = end.map(lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1])) + pad
            games.append(pd.DataFrame({'sid': sid, 'day': (g.date - p7.START).dt.days.to_numpy(), 'att': g.attendance.to_numpy(), 'end': minute.to_numpy()}))
        self.games = pd.concat(games); self.games = self.games[(self.games.day >= 0) & (self.games.day < (len(self.days) if actual else self.D))]
        self._ctx = {}

    # ---- per-origin structural pieces --------------------------------------------------------------
    def context(self, oi):
        if oi in self._ctx:
            return self._ctx[oi]
        origin = self.days[oi]
        block = c8.Block(self, origin, [origin], np.zeros((1, self.S, self.H)))
        L = self.history9.levels(block)
        eff = pre.holiday_effects(block)
        Z = np.expm1(block.Z); k = block.hist_kind
        import warnings
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wd = np.nanmean(Z[:, k == 'wd'], 1)
            r_sat = np.nan_to_num(np.nanmean(Z[:, k == 'sat'], 1) / wd, nan=0.6); r_sun = np.nan_to_num(np.nanmean(Z[:, k == 'sun'], 1) / wd, nan=0.45)
            commute = np.nan_to_num((wd[:, [1, 2, 12]].sum(1) / wd.sum(1)), nan=0.3)
        ref = [((origin - pd.DateOffset(months=1)).year, (origin - pd.DateOffset(months=1)).month), (origin.year, origin.month)]
        self._ctx[oi] = dict(L=L, eff=eff, r_sat=np.log(np.clip(r_sat, 0.05, 3)), r_sun=np.log(np.clip(r_sun, 0.05, 3)), commute=commute, ref=ref, prior={})
        return self._ctx[oi]

    def prior(self, oi, year, month):
        """Seasonal change (S, H) of a target month relative to the two months ending at the origin, and validity."""
        ctx = self.context(oi)
        key = (year, month)
        if key in ctx['prior']:
            return ctx['prior'][key]
        L, ref, P = ctx['L'], ctx['ref'], pre.SEASONAL
        vals, wts = [], []
        for y in pre.NORMAL_YEARS:
            ry = [(y + (a - ref[-1][0]), m) for a, m in ref]
            yy = y + (year - ref[-1][0])
            if yy > 2024 or min(a for a, _ in ry) < 2015 or yy in (2020, 2021, 2022):
                continue
            change = L[:, :, yy - 2015, month - 1] - np.mean([L[:, :, a - 2015, m - 1] for a, m in ry], 0)
            w = pre.RECENCY[y] * (P['chuseok_bonus'] if month in (9, 10) and pre.CHUSEOK_MONTH[yy] == pre.CHUSEOK_MONTH.get(year, 0) else 1.0)
            vals.append(change); wts.append(w)
        if len(vals) < P['min_years']:
            out = (np.zeros((self.S, self.H), np.float32), np.zeros((self.S, self.H), np.float32))
        else:
            vals = np.stack(vals)
            prior, n = pre.weighted_median(vals, np.array(wts))
            st = c8.nanmean(vals, 2); recent = c8.nanmean(st[-2:], 0); old = c8.nanmed(st[:-2], 0) if len(vals) > 2 else recent
            shifted = np.abs(recent - old) > P['regime_gap']
            recent_prior = c8.nanmean(vals[-2:], 0)
            prior = np.where(shifted[:, None] & ~np.isnan(recent_prior), recent_prior, prior)
            station = c8.nanmean(prior, 1)[:, None]
            prior = station + P['hour_shrink'] * (prior - station)
            ok = (n >= P['min_years']) & ~np.isnan(prior)
            out = (np.where(ok, np.clip(prior, -0.6, 0.6), 0.0).astype(np.float32), ok.astype(np.float32))
        ctx['prior'][key] = out
        return out

    # ---- samples ------------------------------------------------------------------------------------
    def sample(self, oi, target_days):
        X7, mu7, y = super().sample(oi, target_days)
        target_days = np.asarray(target_days); n = len(target_days)
        ctx = self.context(oi); st = self.state(oi)
        kinds = c8.day_kinds(self.days[target_days], self.days[oi])
        dow = self.dow[target_days]
        E = np.zeros((n, self.S, self.H, len(EXTRA)), np.float32)
        anchor = np.zeros((n, self.S, self.H), np.float32)
        ex = {name: i for i, name in enumerate(EXTRA)}
        hours = np.arange(6, 24)
        cls_hist = self.cal(oi)['cls'][:oi + 1]; days_h = np.arange(oi + 1)
        wet_share = np.zeros(7, np.float32)
        for c in range(7):
            idx = days_h[(cls_hist == c) & (days_h > oi - 70)][-8:]
            wet_share[c] = (self.wet[idx] > 0).mean() if len(idx) else 0.0
        for k, d in enumerate(target_days):
            date = self.days[d]
            prior, ok = self.prior(oi, date.year, date.month)
            eff = ctx['eff'].get(kinds[k]) if kinds[k] in c8.SPECIAL else None
            level = st['mu'][dow[k]]                                          # ordinary level of that weekday
            a = level + prior + (eff if eff is not None else 0.0)
            anchor[k] = a
            E[k, :, :, ex['prior']] = prior; E[k, :, :, ex['prior_ok']] = ok
            E[k, :, :, ex['hol_eff']] = eff if eff is not None else 0.0
            E[k, :, :, ex['cat']] = CAT_CODE[kinds[k]]
            E[k, :, :, ex['anchor_gap']] = a - mu7[k]
            E[k, :, :, ex['wet']] = self.wet[d]; E[k, :, :, ex['rain_mm']] = np.log1p(self.rain_d[d])
            E[k, :, :, ex['rain_h']] = self.rain_h[d, hours][None, :]
            E[k, :, :, ex['rain_prev2']] = (self.rain_h[d, hours - 1] + self.rain_h[d, hours - 2])[None, :]
            E[k, :, :, ex['temp_h']] = self.temp_h[d, hours][None, :]
            E[k, :, :, ex['tmean']] = np.nanmean(self.temp_h[d, 6:24]); E[k, :, :, ex['tmax']] = np.nanmax(self.temp_h[d, 6:24])
            E[k, :, :, ex['wet_ly']] = self.wet[d - 364] if d - 364 >= 0 else np.nan
            E[k, :, :, ex['wet_share8']] = wet_share[self.cal(oi)['cls'][d]]
        E[..., ex['r_sat']] = ctx['r_sat'][None]; E[..., ex['r_sun']] = ctx['r_sun'][None]
        E[..., ex['commute']] = ctx['commute'][None, :, None]; E[..., ex['basis']] = self.logf[None]
        pos = {int(d): k for k, d in enumerate(target_days)}
        for g in self.games[self.games.day.isin(target_days)].itertuples():
            k = pos[int(g.day)]; s = int(np.searchsorted(self.stations, g.sid))
            E[k, s, :, ex['game_att']] = np.log1p(g.att)
            E[k, s, :, ex['game_rel']] = np.clip(hours - g.end / 60.0, -6, 4)
        missing = np.isnan(anchor)
        anchor = np.where(missing, mu7, anchor)
        return np.concatenate([X7, E], -1), anchor.astype(np.float32), y


def to_raw(panel, days, z):
    """Forecasts are on the 2026 basis; days before 2026 go back to the basis of the train file."""
    return z + np.where((np.asarray(days) < panel.cut)[:, None, None], panel.logf[None], 0.0)
