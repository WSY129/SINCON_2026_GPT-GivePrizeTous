"""ensemble_v11 panel: one row per (station, day, hour) — no pseudo-origin copies.

For a cutoff O (a validation origin or 2026-06-30) every day up to O is a training row and the 123 days after O are
prediction rows.  All rows share one anchor that is fixed at the cutoff:

    anchor(s, d, h) = level_O(s, h, weekday(d))              median of the last 8 clean ordinary days of that weekday
                    + curve(s, h, doy(d)) - curve averaged over the 8 weeks before O     (multi-year daily curve)
                    + holiday effect of the day's category (measured on days <= O)
                    + growth drift for stations younger than the data window (prediction rows only)

target = log1p(boardings) - anchor.  Inputs are exogenous: calendar, Fourier terms of the day of year, observed
hourly weather (rain, temperature, apparent temperature, radiation, sunshine, cloud, wind), realised games and
events, a cleaned year-over-year kernel, station traits and the age of the row (days before the cutoff).

Cleaning: station-days flagged as spikes by the team (known or unknown cause) and every listed event day are
removed before levels and year-over-year kernels are computed, so last year's one-off crowd is not carried over.
History before 2026 is converted to the 2026 aggregation basis when the cutoff is in 2026.
"""
import copy
import json
import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from src.v7.panel import Panel, START
from src.v7 import growth as growth7
from src.v8 import core as c8, pre
from src.v10 import seasonal as s10

EXT = c8.ROOT / 'outputs/ensemble_v11/external'
HORIZON = 123
HOURS = np.arange(6, 24)
FEATURES = ['sid', 'hour', 'dow', 'cat', 'line', 'age', 'f1s', 'f1c', 'f2s', 'f2c', 'f3s', 'f3c',
            'level', 'season', 'hol_eff', 'yoy', 'yoy_ok', 'growth',
            'rain_h', 'rain_prev2', 'rain_next1', 'rain_mm', 'wet', 'wet_ly', 'temp_h', 'app_h', 'tmax', 'tdev', 'hot', 'heatwave', 'comfort',
            'rad_h', 'rad_day', 'raddev', 'sun_hours', 'cloud_day', 'wind_max', 'outing',
            'game_att', 'game_start_rel', 'game_end_rel', 'margin', 'margin7', 'rainout', 'event',
            'r_sat', 'r_sun', 'commute', 'ab_ratio', 'basis']
FI = {n: i for i, n in enumerate(FEATURES)}
CATEGORICAL = ['sid', 'hour', 'dow', 'cat', 'line', 'wet']
CAT_SIZE = dict(sid=71, hour=18, dow=7, cat=12, line=3, wet=3)
DAILY = ['sid', 'dow', 'cat', 'line', 'age', 'f1s', 'f1c', 'f2s', 'f2c', 'f3s', 'f3c', 'season_d', 'hol_d', 'yoy_d', 'growth', 'rain_mm', 'wet', 'wet_ly',
         'tmax', 'tdev', 'hot', 'heatwave', 'comfort', 'rad_day', 'raddev', 'sun_hours', 'cloud_day', 'wind_max', 'outing', 'game_att', 'margin', 'rainout',
         'event', 'r_sat', 'r_sun', 'level_d']
CAT_CODE = {k: i for i, k in enumerate(('wd', 'sat', 'sun') + c8.SPECIAL)}


def _nan(fn, a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(a, axis=axis)


class Panel11:
    def __init__(self, root=None, actual=True):
        root = root or c8.ROOT
        p = Panel(root); self.p = p
        self.S, self.H, self.stations, self.days, self.D = p.S, p.H, p.stations, p.days, p.D
        train = pd.read_csv(root / 'data/raw/train.csv', parse_dates=['date'])
        self.A = np.full_like(p.Z, np.nan)
        self.A[np.searchsorted(p.stations, train.station_id.to_numpy()), (train.date - START).dt.days.to_numpy(), train.hour.to_numpy() - 6] = np.log1p(train.alightings.to_numpy())
        f = pd.read_csv(c8.SHARE / 'ictr_to_train_factor_station_hour.csv')
        self.logf = np.zeros((p.S, p.H), np.float32)
        self.logf[np.searchsorted(p.stations, f.station_id.to_numpy()), f.hour.to_numpy() - 6] = np.log(f.factor_boardings.to_numpy())
        self.cut = p.day_index('2026-01-01')
        self.history = s10.History10(p)
        self._weather(actual)
        self._events(actual)

    # ---------------------------------------------------------------------------------------- inputs
    def _weather(self, actual):
        h = json.loads((EXT / 'open_meteo_incheon_hourly_ext_20250101_20261031.json').read_text())['hourly']
        t = pd.to_datetime(h['time']); di = (t.normalize() - START).days.to_numpy(); hr = t.hour.to_numpy()
        n = len(self.days); last = int(di.max()) if actual else self.D - 1
        W = {}
        for key, name in (('precipitation', 'rain'), ('temperature_2m', 'temp'), ('apparent_temperature', 'app'), ('shortwave_radiation', 'rad'),
                          ('sunshine_duration', 'sun'), ('cloud_cover', 'cloud'), ('wind_speed_10m', 'wind')):
            a = np.full((n, 24), np.nan, np.float32); a[di, hr] = np.asarray(h[key], np.float32)
            a[last + 1:] = np.nan
            W[name] = a
        self.W = W; self.wlast = last
        rain = np.nan_to_num(W['rain'])
        self.rain_d = rain[:, 6:24].sum(1); self.wet = (self.rain_d >= 1).astype(np.float32) + (self.rain_d >= 10)
        self.tmax = _nan(np.nanmax, W['temp'][:, 6:24], 1); self.rad_day = np.nansum(W['rad'][:, 6:20], 1) / 1000.0
        self.sun_hours = np.nansum(W['sun'][:, 6:20], 1) / 3600.0; self.cloud_day = _nan(np.nanmean, W['cloud'][:, 9:19], 1)
        self.wind_max = _nan(np.nanmax, W['wind'][:, 6:24], 1)
        med = lambda x: pd.Series(x).rolling(29, center=True, min_periods=8).median().to_numpy()
        self.tdev = self.tmax - med(self.tmax); self.raddev = self.rad_day - med(self.rad_day)

    def _events(self, actual):
        from src.v7 import core as core7
        p = self.p; n = len(self.days); lim = n if actual else self.D
        mask = np.zeros((p.S, n), bool)
        spikes = pd.read_csv(c8.ROOT / 'outputs/ensemble_v10/external/spike_days_labeled.csv', parse_dates=['date'])
        mask[np.searchsorted(p.stations, spikes.station_id.to_numpy()), (spikes.date - START).dt.days.to_numpy()] = True
        self.unexplained = int((~spikes.explained).sum())
        schedule = core7.schedule_v7(); cfg = json.loads((c8.ROOT / 'configs/ensemble_v10.json').read_text())
        event = np.zeros((p.S, n), np.float32)
        listed = [(e.get('stations', [c8.MUNHAK]), e['date']) for e in list(schedule.events) + list(schedule.overlay_events())]
        listed += [(e['stations'], d) for e in cfg.get('station_events', []) for d in e['dates'] + e['past_dates']]
        listed += [([c8.MUNHAK], d) for d in cfg.get('munhak_past_events', [])] + [([c8.MUNHAK], e['date']) for e in cfg.get('munhak_events', [])]
        listed += [([3139], '2026-09-12')]
        for stations, d in listed:
            k = p.day_index(d)
            if 0 <= k < lim:
                for sid in stations:
                    event[np.searchsorted(p.stations, sid), k] = 1; mask[np.searchsorted(p.stations, sid), k] = True
        g = pd.read_csv(c8.SHARE / 'games_ssg_munhak.csv', parse_dates=['date']); g = g[g.date >= '2025-01-01']
        sc = pd.read_csv(EXT / 'games_ssg_scores.csv', parse_dates=['date']) if (EXT / 'games_ssg_scores.csv').exists() else pd.DataFrame(columns=['date', 'game_id'])
        g = g.merge(sc[[c for c in sc.columns if c not in ('date',)]], on='game_id', how='left')
        mins = lambda x: int(str(x).split(':')[0]) * 60 + int(str(x).split(':')[1])
        self.game = {name: np.full((p.S, n), np.nan, np.float32) for name in ('att', 'start', 'end', 'margin', 'margin7', 'rainout')}
        for name in ('att', 'rainout'):
            self.game[name][:] = 0
        m = int(np.searchsorted(p.stations, c8.MUNHAK))
        for r in g.itertuples():
            k = p.day_index(r.date)
            if not 0 <= k < lim:
                continue
            mask[m, k] = True
            if r.status == 'played' and pd.notna(r.attendance) and pd.notna(r.end_time):
                self.game['att'][m, k] += r.attendance
                self.game['start'][m, k] = mins(r.start_time); self.game['end'][m, k] = mins(r.end_time)
                self.game['margin'][m, k] = getattr(r, 'margin_final', np.nan); self.game['margin7'][m, k] = getattr(r, 'margin_after_7', np.nan)
            elif r.status == '우천취소':
                self.game['rainout'][m, k] = 1; self.game['start'][m, k] = mins(r.sched_time) if pd.notna(r.sched_time) else 1110
        b = pd.read_csv(c8.SHARE / 'games_bucheonfc.csv', parse_dates=['date']); b = b[(b.status == 'FE') & (b.attendance > 0) & (b.date >= '2025-01-01')]
        extra = pd.DataFrame([dict(date=pd.Timestamp(x['date']), kickoff=x['kickoff'], attendance=x['attendance']) for x in cfg.get('extra_matches_bucheon', [])])
        b = pd.concat([b[['date', 'kickoff', 'attendance']], extra], ignore_index=True)
        s = int(np.searchsorted(p.stations, 3754))
        for r in b.itertuples():
            k = p.day_index(r.date)
            if 0 <= k < lim:
                mask[s, k] = True; self.game['att'][s, k] = r.attendance
                self.game['start'][s, k] = mins(r.kickoff); self.game['end'][s, k] = mins(r.kickoff) + 115
        self.mask, self.event = mask, event

    # ---------------------------------------------------------------------------------------- pieces
    def curve(self, block):
        """Daily multi-year curve [S, H, 366] in log units (zero mean over the year)."""
        L = self.history.levels(block)                                    # [S, H, year, month]
        vals, wts = [], []
        for y in pre.NORMAL_YEARS:
            a = L[:, :, y - 2015, :]
            vals.append(a - _nan(np.nanmean, a, 2)[:, :, None]); wts.append(pre.RECENCY[y])
        vals = np.stack(vals)                                             # [years, S, H, 12]
        clim, n = pre.weighted_median(vals, np.array(wts))
        st = _nan(np.nanmean, vals, 2)                                    # [years, S, 12]
        recent = _nan(np.nanmean, st[-2:], 0); old = _nan(np.nanmedian, st[:-2], 0)
        shifted = np.abs(recent - old) > pre.SEASONAL['regime_gap']
        recent_c = _nan(np.nanmean, vals[-2:], 0)
        clim = np.where(shifted[:, None, :] & ~np.isnan(recent_c), recent_c, clim)
        station = _nan(np.nanmean, clim, 1)[:, None, :]
        clim = station + pre.SEASONAL['hour_shrink'] * (clim - station)
        clim = np.where((n >= 2) & ~np.isnan(clim), clim, 0.0)
        s = int(np.searchsorted(self.stations, 3139))
        clim[s, :, 7] = 0.5 * (clim[s, :, 6] + clim[s, :, 8])              # August at 3139 holds the festival crowds of earlier years
        mid = np.array([15.5, 45.5, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, 349.5])
        x = np.concatenate([mid, [mid[0] + 365.25]]); y = np.concatenate([clim, clim[:, :, :1]], 2)
        return CubicSpline(x, y, axis=2, bc_type='periodic')(np.where(np.arange(1, 367) < mid[0], np.arange(1, 367) + 365.25, np.arange(1, 367))).astype(np.float32)

    def build(self, origin, horizon=HORIZON):
        """Returns a dict of flat arrays for training rows (days <= origin) and dense arrays for prediction rows."""
        p = self.p; O = p.day_index(origin); S, H = self.S, self.H
        last = O + horizon; nd = last + 1
        basis2026 = O >= self.cut
        shift = np.where((np.arange(nd) < self.cut)[None, :, None], self.logf[:, None, :], 0.0) if basis2026 else 0.0
        Zb = p.Z[:, :nd] - shift; Ab = self.A[:, :nd] - shift
        Zb[:, O + 1:] = np.nan; Ab[:, O + 1:] = np.nan                     # nothing after the cutoff is ever read
        pz = copy.copy(p); pz.Z = Zb
        dates = self.days[:nd]
        kinds = c8.day_kinds(dates, origin); dow = p.dow[:nd]
        block = c8.Block(pz, origin, dates[O + 1:], np.zeros((horizon, S, H)))
        ordinary = np.isin(kinds, c8.ORDINARY)
        clean = np.where((self.mask[:, :nd] | ~ordinary[None, :])[:, :, None], np.nan, Zb)
        clean_a = np.where((self.mask[:, :nd] | ~ordinary[None, :])[:, :, None], np.nan, Ab)
        level = np.full((7, S, H), np.nan, np.float32); level_a = np.full((7, S, H), np.nan, np.float32)
        hist = np.arange(O + 1)
        for c in range(7):
            idx = hist[(dow[:O + 1] == c) & (hist > O - 70)]
            for arr, src in ((level, clean), (level_a, clean_a)):
                a = src[:, idx]
                ok = (~np.isnan(a[:, :, 0])).cumsum(1)
                keep = ok > (ok[:, -1:] - 8)                               # last 8 clean days per station
                arr[c] = _nan(np.nanmedian, np.where(keep[:, :, None], a, np.nan), 1)
        for arr, src in ((level, clean), (level_a, clean_a)):
            wd = _nan(np.nanmedian, src[:, hist[(dow[:O + 1] < 5) & (hist > O - 28)]], 1); we = _nan(np.nanmedian, src[:, hist[(dow[:O + 1] >= 5) & (hist > O - 56)]], 1)
            anyd = _nan(np.nanmedian, np.where(np.isnan(src[:, max(O - 20, 0):O + 1]), (Zb if src is clean else Ab)[:, max(O - 20, 0):O + 1], src[:, max(O - 20, 0):O + 1]), 1)
            for c in range(7):
                pooled = np.where(np.isnan(wd if c < 5 else we), anyd, wd if c < 5 else we)
                arr[c] = np.where(np.isnan(arr[c]), pooled, arr[c])
        C = self.curve(block)                                              # [S, H, 366]
        doy = dates.dayofyear.to_numpy()
        season = C[:, :, doy - 1] - C[:, :, doy[max(O - 55, 0):O + 1] - 1].mean(2)[:, :, None]        # [S, H, nd]
        season = np.moveaxis(season, 2, 1)                                 # [S, nd, H]
        eff = pre.holiday_effects(block)
        hol = np.zeros((S, nd, H), np.float32)
        for k in np.flatnonzero(~ordinary):
            if kinds[k] in eff:
                hol[:, k] = eff[kinds[k]]
        grow = np.zeros((S, nd), np.float32)
        slopes = growth7.slopes(p, origin, **growth7.VARIANTS[0])
        for sid, sl in slopes.items():
            grow[np.searchsorted(self.stations, sid), O + 1:] = np.clip(sl * (growth7.LAG + np.arange(1, horizon + 1)), -growth7.CAP, growth7.CAP)
        lev = np.moveaxis(level[dow], 0, 1)                                # [S, nd, H]
        lev_a = np.moveaxis(level_a[dow], 0, 1)
        anchor = lev + season + hol + grow[:, :, None]
        anchor_a = lev_a + season + hol + grow[:, :, None]
        # year-over-year kernel on the cleaned series, same weekday, triangular weights over +-2 weeks
        num = np.zeros((S, nd, H), np.float32); den = np.zeros((S, nd, H), np.float32)
        for off, w in ((-14, 1.0), (-7, 2.0), (0, 3.0), (7, 2.0), (14, 1.0)):
            src = np.arange(nd) - 364 + off
            ok = (src >= 0) & (src <= O)
            v = clean[:, np.clip(src, 0, nd - 1)]
            good = ok[None, :, None] & ~np.isnan(v)
            num += np.where(good, w * v, 0.0); den += np.where(good, w, 0.0)
        yoy = np.where(den >= 3, num / np.maximum(den, 1e-6), np.nan) - (lev + season)
        X = np.full((S, nd, H, len(FEATURES)), np.nan, np.float32)
        put = lambda name, v: X.__setitem__((..., FI[name]), v)
        put('sid', np.arange(S)[:, None, None]); put('hour', np.arange(H)[None, None, :]); put('dow', dow[None, :, None])
        put('cat', np.array([CAT_CODE[k] for k in kinds])[None, :, None]); put('line', p.line[:, None, None])
        put('age', np.clip(O - np.arange(nd), 0, None)[None, :, None])
        ang = 2 * np.pi * doy / 365.25
        for j in (1, 2, 3):
            put(f'f{j}s', np.sin(j * ang)[None, :, None]); put(f'f{j}c', np.cos(j * ang)[None, :, None])
        put('level', lev); put('season', season); put('hol_eff', hol); put('yoy', yoy); put('yoy_ok', (~np.isnan(yoy)).astype(np.float32)); put('growth', grow[:, :, None])
        W = self.W; rain = np.nan_to_num(W['rain'][:nd])
        put('rain_h', rain[:, HOURS][None]); put('rain_prev2', (rain[:, HOURS - 1] + rain[:, HOURS - 2])[None]); put('rain_next1', rain[:, np.minimum(HOURS + 1, 23)][None])
        put('rain_mm', np.log1p(self.rain_d[:nd])[None, :, None]); put('wet', self.wet[:nd][None, :, None])
        put('wet_ly', np.where(np.arange(nd) >= 364, self.wet[np.clip(np.arange(nd) - 364, 0, None)], np.nan)[None, :, None])
        put('temp_h', W['temp'][:nd][:, HOURS][None]); put('app_h', W['app'][:nd][:, HOURS][None])
        tmax = self.tmax[:nd]
        put('tmax', tmax[None, :, None]); put('tdev', self.tdev[:nd][None, :, None]); put('hot', np.clip(tmax - 30, 0, None)[None, :, None])
        put('heatwave', (tmax >= 33).astype(np.float32)[None, :, None])
        put('comfort', ((tmax >= 15) & (tmax <= 26) & (self.rain_d[:nd] < 1)).astype(np.float32)[None, :, None])
        put('rad_h', W['rad'][:nd][:, HOURS][None]); put('rad_day', self.rad_day[:nd][None, :, None]); put('raddev', self.raddev[:nd][None, :, None])
        put('sun_hours', self.sun_hours[:nd][None, :, None]); put('cloud_day', self.cloud_day[:nd][None, :, None]); put('wind_max', self.wind_max[:nd][None, :, None])
        put('outing', ((self.rain_d[:nd] < 1) & (self.sun_hours[:nd] >= 8)).astype(np.float32)[None, :, None])
        g = self.game
        put('game_att', np.log1p(g['att'][:, :nd])[:, :, None])
        put('game_start_rel', np.clip(HOURS[None, None, :] - g['start'][:, :nd, None] / 60.0, -8, 8)); put('game_end_rel', np.clip(HOURS[None, None, :] - g['end'][:, :nd, None] / 60.0, -8, 6))
        put('margin', g['margin'][:, :nd, None]); put('margin7', g['margin7'][:, :nd, None]); put('rainout', g['rainout'][:, :nd, None]); put('event', self.event[:, :nd, None])
        Zh = np.expm1(Zb[:, :O + 1]); kh = kinds[:O + 1]
        wdm = _nan(np.nanmean, Zh[:, kh == 'wd'], 1)
        with np.errstate(all='ignore'):
            put('r_sat', np.log(np.clip(np.nan_to_num(_nan(np.nanmean, Zh[:, kh == 'sat'], 1) / wdm, nan=0.6), 0.05, 3))[:, None, :])
            put('r_sun', np.log(np.clip(np.nan_to_num(_nan(np.nanmean, Zh[:, kh == 'sun'], 1) / wdm, nan=0.45), 0.05, 3))[:, None, :])
            put('commute', np.nan_to_num(wdm[:, [1, 2, 12]].sum(1) / wdm.sum(1), nan=0.3)[:, None, None])
            put('ab_ratio', np.nan_to_num(_nan(np.nanmean, Ab[:, max(O - 180, 0):O + 1] - Zb[:, max(O - 180, 0):O + 1], 1))[:, None, :])
        put('basis', self.logf[:, None, :])
        r = Zb - anchor; ra = Ab - anchor_a
        return dict(origin=str(origin)[:10], O=O, X=X, anchor=anchor.astype(np.float32), r=r.astype(np.float32), ra=ra.astype(np.float32),
                    age=np.clip(O - np.arange(nd), 0, None), kinds=kinds, basis2026=basis2026, mask=self.mask[:, :nd])


def flat_train(d, half_life=240.0, min_day=0):
    """Training rows (days <= cutoff with an observed target): X, r, ra, weights."""
    O = d['O']; X = d['X'][:, min_day:O + 1]; r = d['r'][:, min_day:O + 1]; ra = d['ra'][:, min_day:O + 1]
    ok = ~np.isnan(r) & ~np.isnan(d['anchor'][:, min_day:O + 1])
    w = np.broadcast_to((0.5 ** (d['age'][min_day:O + 1] / half_life))[None, :, None], r.shape)
    return X[ok], r[ok], np.nan_to_num(ra[ok]), w[ok].astype(np.float32), ok


def pred_rows(d):
    """Prediction block as [day, S, H, F] and anchor [day, S, H] (the layout used everywhere else in v8-v10)."""
    O = d['O']
    return np.moveaxis(d['X'][:, O + 1:], 0, 1), np.moveaxis(d['anchor'][:, O + 1:], 0, 1)
