"""Dense one-pass panel features for the v7 equilibrium models.

History is held as Z[station, day, hour] = log1p(boardings).  A sample is (origin day, target day) and its
features have shape (71 stations, 18 hours, F); everything is computed from days <= origin, calendar facts
known at the origin and schedule records with known_since <= origin.  One origin costs a few milliseconds.

Baseline: mu_eq(station, hour, eff_dow) = median of the last (up to) 8 days of the same effective weekday
class within 70 days before the origin.  Classes: 0-4 ordinary Mon-Fri, 5 Saturday, 6 Sunday or public holiday.
The training target for every row is r_eq = log1p(y) - mu_eq.
"""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from src.holiday_calendar import calendar_features

HOURS = np.arange(6, 24)
START = pd.Timestamp('2025-01-01')
CAL_END = pd.Timestamp('2026-12-31')
HORIZON = 123
MIN_HISTORY = 35
CAL_COLS = ['is_holiday', 'holiday_type', 'is_substitute_holiday', 'is_temporary_holiday', 'is_election_day',
            'is_lunar_holiday', 'is_labor_day', 'calendar_is_weekend', 'is_rest_day', 'is_holiday_eve',
            'is_after_holiday', 'is_bridge_day']
FEATURES = (['sid', 'line', 'hour', 'age', 'horizon', 'dow', 'cls', 'month', 'doy_sin', 'doy_cos'] + CAL_COLS +
            ['kbo_state', 'event', 'acad', 'acad_recent',
             'mu', 'm4', 'trend4', 'sd8', 'gap8', 'long16', 'nobs', 'mu_tot', 'trend_tot', 'share', 'mu_wd',
             'net_trend', 'r7', 'r14', 'r28',
             'ly_rel', 'ly_seas', 'ly_seas_tot', 'ly_seas_net', 'growth', 'growth_tot', 'yoy_ok'])
YOY = ['ly_rel', 'ly_seas', 'ly_seas_tot', 'ly_seas_net', 'growth', 'growth_tot', 'yoy_ok']
CATEGORICAL = ['sid', 'line', 'hour', 'dow', 'cls', 'month', 'holiday_type', 'kbo_state']
FI = {name: i for i, name in enumerate(FEATURES)}


def _nan(fn, a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(a, axis=axis)


class Panel:
    def __init__(self, root='.', schedule=None, academic=None, rain=None, last_day=None):
        """rain: optional Series(date -> daily precipitation mm) of observations; only days <= origin are read.
        academic: list of {station, start, end, known_since} semester periods."""
        root = Path(root)
        train = pd.read_csv(root / 'data/raw/train.csv', parse_dates=['date'])
        meta = pd.read_csv(root / 'data/raw/metaData.csv', parse_dates=['first_date'])
        if last_day is not None:
            train = train[train.date <= pd.Timestamp(last_day)]
        self.stations = np.sort(meta.station_id.to_numpy())
        self.S, self.H = len(self.stations), len(HOURS)
        self.days = pd.date_range(START, CAL_END)
        self.D = int((train.date.max() - START).days) + 1          # number of observed days
        Z = np.full((self.S, len(self.days), self.H), np.nan, np.float32)
        si = np.searchsorted(self.stations, train.station_id.to_numpy())
        di = (train.date - START).dt.days.to_numpy()
        Z[si, di, train.hour.to_numpy() - 6] = np.log1p(train.boardings.to_numpy())
        self.Z = Z
        self.T = np.log1p(_nan(np.nansum, np.expm1(Z), 2)).astype(np.float32)
        self.T[np.isnan(Z).all(2)] = np.nan
        meta = meta.set_index('station_id').loc[self.stations]
        self.line = meta.canonical_line.map({1: 0, 2: 1, 7: 2}).to_numpy()
        self.first = (meta.first_date - START).dt.days.to_numpy()
        self.dow = self.days.dayofweek.to_numpy()
        self.schedule = schedule
        self.academic = academic or []
        self.rain = None
        if rain is not None:
            self.rain = rain.reindex(self.days).to_numpy(float)
        self._cal, self._state = {}, {}

    def day_index(self, date):
        return int((pd.Timestamp(date) - START).days)

    # ---- calendar and schedule knowledge as of an origin ------------------------------------------------
    def cal(self, oi):
        if oi not in self._cal:
            origin = self.days[oi]
            frame = calendar_features(self.days, origin.strftime('%Y-%m-%d'))[CAL_COLS].to_numpy(np.float32)
            cls = np.where(frame[:, 0] > 0, 6, self.dow)
            kbo = np.full(len(self.days), -1, np.float32)
            event = np.zeros((self.S, len(self.days)), np.float32)
            if self.schedule is not None:
                state, _ = self.schedule.state(self.days, origin)
                kbo = np.asarray(state, np.float32)
                records = list(self.schedule.events) + list(self.schedule.overlay_events())
                for e in records:
                    if pd.Timestamp(e['known_since']) > origin:
                        continue
                    d = self.day_index(e['date'])
                    if 0 <= d < len(self.days):
                        for sid in e.get('stations', [3127]):
                            event[np.searchsorted(self.stations, sid), d] = 1
            acad = np.full((self.S, len(self.days)), -1, np.float32)
            for station in {a['station'] for a in self.academic}:
                acad[np.searchsorted(self.stations, station)] = 0
            for a in self.academic:
                if pd.Timestamp(a['known_since']) <= origin:
                    lo, hi = self.day_index(a['start']), self.day_index(a['end'])
                    acad[np.searchsorted(self.stations, a['station']), max(lo, 0):hi + 1] = 1
            self._cal[oi] = dict(frame=frame, cls=cls, kbo=kbo, event=event, acad=acad)
        return self._cal[oi]

    # ---- per-origin level state ------------------------------------------------------------------------
    def history(self, oi):
        """Observed log boardings up to the origin (optionally with the rain effect removed)."""
        return self.Z[:, :oi + 1], self.T[:, :oi + 1]

    def state(self, oi):
        if oi in self._state:
            return self._state[oi]
        cls = self.cal(oi)['cls'][:oi + 1]
        Z, T = self.history(oi)
        S, H = self.S, self.H
        keys = ['mu', 'm4', 'p4', 'sd8', 'mean8', 'long16']
        out = {k: np.full((7, S, H), np.nan, np.float32) for k in keys}
        out['nobs'] = np.zeros((7, S), np.float32)
        out['mu_tot'] = np.full((7, S), np.nan, np.float32)
        out['trend_tot'] = np.full((7, S), np.nan, np.float32)
        days = np.arange(oi + 1)
        for c in range(7):
            idx_all = days[(cls == c) & (days > oi - 140)]
            idx = idx_all[idx_all > oi - 70][-8:]
            if len(idx) == 0:
                continue
            a = Z[:, idx]
            out['mu'][c] = _nan(np.nanmedian, a, 1)
            out['mean8'][c] = _nan(np.nanmean, a, 1)
            out['sd8'][c] = _nan(np.nanstd, a, 1)
            out['m4'][c] = _nan(np.nanmean, a[:, -4:], 1)
            if len(idx) > 4:
                out['p4'][c] = _nan(np.nanmean, a[:, :-4], 1)
                out['trend_tot'][c] = _nan(np.nanmean, T[:, idx[-4:]], 1) - _nan(np.nanmean, T[:, idx[:-4]], 1)
            older = idx_all[idx_all < idx[0]][-8:]
            if len(older) >= 4:
                out['long16'][c] = _nan(np.nanmean, Z[:, older], 1)
            out['nobs'][c] = (~np.isnan(a[:, :, 0])).sum(1)
            out['mu_tot'][c] = _nan(np.nanmedian, T[:, idx], 1)
        wd = days[(cls <= 4) & (days > oi - 28)]
        we = days[(cls >= 5) & (days > oi - 56)]
        anyd = days[days > oi - 14]
        mu_wd = _nan(np.nanmedian, Z[:, wd], 1) if len(wd) else np.full((S, H), np.nan, np.float32)
        mu_we = _nan(np.nanmedian, Z[:, we], 1) if len(we) else np.full((S, H), np.nan, np.float32)
        mu_any = _nan(np.nanmedian, Z[:, anyd], 1)
        tot_any = _nan(np.nanmedian, T[:, anyd], 1)
        for c in range(7):
            weak = (out['nobs'][c] < 2)[:, None] | np.isnan(out['mu'][c])
            pooled = mu_wd if c <= 4 else mu_we
            pooled = np.where(np.isnan(pooled), mu_any, pooled)
            out['mu'][c] = np.where(weak, pooled, out['mu'][c])
            out['mu_tot'][c] = np.where(np.isnan(out['mu_tot'][c]), tot_any, out['mu_tot'][c])
        out['mu_wd'] = mu_wd
        # recent deviation of every day from its own class baseline
        recent = days[days > oi - 28]
        dev = Z[:, recent] - np.moveaxis(out['mu'][cls[recent]], 0, 1)
        for n in (7, 14, 28):
            out[f'r{n}'] = _nan(np.nanmean, dev[:, -n:], 1)
        out['net_trend'] = _nan(np.nanmean, out['m4'] - out['p4'], 1)          # (7, H)
        acad = self.cal(oi)['acad'][:, max(oi - 27, 0):oi + 1]
        out['acad_recent'] = np.where(acad[:, 0] < 0, -1, acad.mean(1))
        self._state[oi] = out
        return out

    # ---- samples ----------------------------------------------------------------------------------------
    def sample(self, oi, target_days):
        """Features for one origin and several target days.
        Returns X (n, S, H, F) float32, mu (n, S, H), y (n, S, H) (NaN where not observed / in the future)."""
        target_days = np.asarray(target_days)
        n, S, H = len(target_days), self.S, self.H
        cal, st = self.cal(oi), self.state(oi)
        st2 = self.state(oi - 364) if oi - 364 >= MIN_HISTORY - 1 else None
        X = np.full((n, S, H, len(FEATURES)), np.nan, np.float32)

        def put(name, value):
            X[..., FI[name]] = value

        put('sid', np.arange(S)[None, :, None]); put('line', self.line[None, :, None])
        put('hour', np.arange(H)[None, None, :])
        put('age', np.log1p(np.clip(oi - self.first, 0, 400))[None, :, None])
        put('horizon', (target_days - oi)[:, None, None])
        dow = self.dow[target_days]; cls = cal['cls'][target_days]
        put('dow', dow[:, None, None]); put('cls', cls[:, None, None])
        put('month', self.days.month.to_numpy()[target_days][:, None, None])
        doy = self.days.dayofyear.to_numpy()[target_days] * (2 * np.pi / 365.25)
        put('doy_sin', np.sin(doy)[:, None, None]); put('doy_cos', np.cos(doy)[:, None, None])
        for j, name in enumerate(CAL_COLS):
            put(name, cal['frame'][target_days, j][:, None, None])
        munhak = np.searchsorted(self.stations, 3127)
        X[:, :, :, FI['kbo_state']] = -1
        X[:, munhak, :, FI['kbo_state']] = cal['kbo'][target_days][:, None]
        put('event', cal['event'][:, target_days].T[:, :, None])
        put('acad', cal['acad'][:, target_days].T[:, :, None])
        put('acad_recent', st['acad_recent'][None, :, None])
        mu = st['mu'][cls]                                                    # (n, S, H)
        put('mu', mu)
        put('m4', st['m4'][cls] - mu); put('trend4', st['m4'][cls] - st['p4'][cls])
        put('sd8', st['sd8'][cls]); put('gap8', st['mean8'][cls] - mu); put('long16', st['long16'][cls] - mu)
        put('nobs', st['nobs'][cls][:, :, None])
        put('mu_tot', st['mu_tot'][cls][:, :, None]); put('trend_tot', st['trend_tot'][cls][:, :, None])
        put('share', mu - st['mu_tot'][cls][:, :, None]); put('mu_wd', st['mu_wd'][None] - mu)
        put('net_trend', st['net_trend'][cls][:, None, :])
        for name in ('r7', 'r14', 'r28'):
            put(name, st[name][None])
        # year-over-year block: same weekday 52 weeks earlier, class taken from the weekday only
        Z, T = self.history(oi)
        c2 = dow
        put('yoy_ok', 0)
        for k, d in enumerate(target_days):
            d2 = d - 364
            near = np.array([x for x in (d2 - 7, d2, d2 + 7) if 0 <= x <= oi], int)
            if d2 < 0 or len(near) == 0:
                continue
            ly = _nan(np.nanmedian, Z[:, near], 1); ly_tot = _nan(np.nanmedian, T[:, near], 1)
            X[k, :, :, FI['ly_rel']] = ly - st['mu'][c2[k]]
            if st2 is not None:
                seas = ly - st2['mu'][c2[k]]
                X[k, :, :, FI['ly_seas']] = seas
                X[k, :, :, FI['ly_seas_tot']] = (ly_tot - st2['mu_tot'][c2[k]])[:, None]
                X[k, :, :, FI['ly_seas_net']] = _nan(np.nanmean, seas, 0)[None, :]
                X[k, :, :, FI['growth']] = st['mu'][c2[k]] - st2['mu'][c2[k]]
                X[k, :, :, FI['growth_tot']] = (st['mu_tot'][c2[k]] - st2['mu_tot'][c2[k]])[:, None]
                X[k, :, :, FI['yoy_ok']] = 1
        y = np.full((n, S, H), np.nan, np.float32)
        seen = target_days < self.D
        y[seen] = np.moveaxis(self.Z[:, target_days[seen]], 0, 1)
        return X, mu.astype(np.float32), y

    def training_set(self, origin, stride=7, keep=1.0, seed=0, max_origins=None):
        """All pseudo-origins o' = origin - stride*k with labels no later than `origin`."""
        O = self.day_index(origin)
        rng = np.random.default_rng(seed)
        Xs, mus, ys, keys = [], [], [], []
        origins = [o for o in range(O - stride, MIN_HISTORY - 2, -stride)]
        if max_origins:
            origins = origins[:max_origins]
        for o in origins:
            days = np.arange(o + 1, min(o + HORIZON, O) + 1)
            if keep < 1:
                days = days[rng.random(len(days)) < keep]
            if len(days) == 0:
                continue
            X, mu, y = self.sample(o, days)
            Xs.append(X); mus.append(mu); ys.append(y)
            keys.append(np.stack([np.full(len(days), o), days], 1))
        return np.concatenate(Xs), np.concatenate(mus), np.concatenate(ys), np.concatenate(keys)

    def eval_set(self, origin, horizon=HORIZON):
        O = self.day_index(origin)
        days = np.arange(O + 1, O + horizon + 1)
        X, mu, y = self.sample(O, days)
        return X, mu, y, np.stack([np.full(len(days), O), days], 1)


def load_academic(config_path):
    path = Path(config_path)
    if not path.exists():
        return []
    return json.loads(path.read_text()).get('academic_periods', [])
