"""v10 weather module: the v8 daily rain effect plus hourly rain and temperature (observed values; allowed for the
evaluation period by the organisers).  Every effect is estimated on history <= origin.

f1  daily rain level of the target day (1-10 mm, >= 10 mm in service hours) minus the average rain effect
f2  the same quantity for the date 364 days earlier, with the opposite sign (echo of last year's rain)
f3  rain falling in the hour or the two hours before it, on top of the daily effect
f4  temperature anomaly (daily maximum minus the 29-day centred median) in the hot season (normal >= 24 C)
f5  temperature anomaly in the mild/cold season
Temperature effects are estimated on dry ordinary days only, separately for daytime (09-19h) and other hours.
"""
import json
import numpy as np
import pandas as pd
from src.v7.panel import START
from src.v8 import core as c8
from src.v8.core import nanmed, MUNHAK, ORDINARY

HOT = 24.0


class Weather10:
    def __init__(self, panel):
        h = json.loads((c8.EXT / 'open_meteo_incheon_hourly_20250101_20261031.json').read_text())['hourly']
        t = pd.to_datetime(h['time']); di = (t.normalize() - START).days.to_numpy(); hr = t.hour.to_numpy()
        n = len(panel.days)
        self.rain = np.zeros((n, 24)); self.temp = np.full((n, 24), np.nan)
        self.rain[di, hr] = np.asarray(h['precipitation'], float); self.temp[di, hr] = np.asarray(h['temperature_2m'], float)
        self.last = int(di.max())
        mm = self.rain[:, 6:24].sum(1)
        self.wet = (mm >= 1).astype(int) + (mm >= 10)
        hours = np.arange(6, 24)
        self.now = (self.rain[:, hours] + self.rain[:, hours - 1] + self.rain[:, hours - 2]) >= 1.0        # [day, 18]
        tmax = np.nanmax(self.temp[:self.last + 1, 6:24], 1)
        self.tmax = np.full(n, np.nan); self.tmax[:self.last + 1] = tmax
        norm = pd.Series(tmax).rolling(29, center=True, min_periods=10).median().to_numpy()
        self.tnorm = np.full(n, np.nan); self.tnorm[:self.last + 1] = norm
        self.tdev = self.tmax - self.tnorm
        self._fit = {}
        self.last_info = {}

    def fit(self, block, shrink=3.0):
        if block.origin in self._fit:
            return self._fit[block.origin]
        Z = block.Z; hk = block.hist_kind; O = block.O; days = np.arange(O + 1); dow = block.panel.dow[:O + 1]
        wet = self.wet[:O + 1]; ordinary = np.isin(hk, ORDINARY); off = (hk != 'wd').astype(int)
        S, H = Z.shape[0], Z.shape[2]
        dev = {}
        for t in days[ordinary]:
            near = days[(np.abs(days - t) <= 28) & (days != t) & (dow == dow[t]) & (wet == 0) & ordinary]
            if len(near) >= 2:
                dev[t] = np.nan_to_num(Z[:, t] - nanmed(Z[:, near], 1))
        # daily levels
        beta = np.zeros((3, 2, S, H)); cnt = np.zeros((3, 2))
        for t, d in dev.items():
            if wet[t] > 0:
                beta[wet[t], off[t]] += d; cnt[wet[t], off[t]] += 1
        mean = beta / np.maximum(cnt, 1)[:, :, None, None]
        net = np.median(mean, axis=2, keepdims=True); n = cnt[:, :, None, None]
        beta = (n * mean + shrink * net) / (n + shrink)
        share = np.array([(wet[ordinary] == l).mean() for l in range(3)])
        expected = share[1] * beta[1] + share[2] * beta[2]
        # hourly rain on top of the daily level
        num = np.zeros((2, 2, S)); den = np.zeros((2, 2))
        for t, d in dev.items():
            if wet[t] > 0:
                r = d - beta[wet[t], off[t]]; now = self.now[t]
                for flag in (0, 1):
                    sel = now == bool(flag)
                    if sel.any():
                        num[flag, off[t]] += r[:, sel].sum(1); den[flag, off[t]] += sel.sum()
        c = num[1] / np.maximum(den[1], 1)[:, None] - num[0] / np.maximum(den[0], 1)[:, None]            # (2, S)
        cnet = np.median(c, axis=1, keepdims=True); k = 20.0
        c = (den[1][:, None] * c + k * cnet) / (den[1][:, None] + k)
        pnow = float(den[1].sum() / max(den.sum(), 1))
        # temperature on dry days
        day_hours = (np.arange(6, 24) >= 9) & (np.arange(6, 24) <= 19)
        sxy = np.zeros((2, 2, 2, S)); sxx = np.zeros((2, 2)); nd = np.zeros((2, 2))
        for t, d in dev.items():
            if wet[t] == 0 and not np.isnan(self.tdev[t]):
                reg = int(self.tnorm[t] >= HOT); x = self.tdev[t]
                sxy[reg, off[t], 0] += x * d[:, day_hours].mean(1); sxy[reg, off[t], 1] += x * d[:, ~day_hours].mean(1)
                sxx[reg, off[t]] += x * x; nd[reg, off[t]] += 1
        slope = sxy / np.maximum(sxx, 1e-9)[:, :, None, None]
        snet = np.median(slope, axis=3, keepdims=True); kk = 30.0
        slope = (nd[:, :, None, None] * slope + kk * snet) / (nd[:, :, None, None] + kk)
        info = dict(wet_days={int(l): int(cnt[l].sum()) for l in (1, 2)}, rain_now_share_on_wet_days=round(pnow, 3),
                    hourly_rain_effect_network={g: round(float(np.median(c[i])), 4) for i, g in enumerate(('weekday', 'off'))},
                    temperature_slope_per_degree_network={f'{r}_{g}_{hb}': round(float(np.median(slope[i, j, m])), 4) for i, r in enumerate(('mild', 'hot'))
                                                          for j, g in enumerate(('weekday', 'off')) for m, hb in enumerate(('day', 'other'))},
                    dry_days={f'{r}_{g}': int(nd[i, j]) for i, r in enumerate(('mild', 'hot')) for j, g in enumerate(('weekday', 'off'))})
        self._fit[block.origin] = (beta, expected, c, pnow, slope, day_hours, info)
        return self._fit[block.origin]

    def features(self, block, pred):
        beta, expected, c, pnow, slope, day_hours, info = self.fit(block)
        n = len(block.dates); S, H = pred.shape[1:]
        f = [np.zeros_like(pred) for _ in range(5)]
        off = (block.kind != 'wd').astype(int)
        for k in range(n):
            d = block.di[k]; w = self.wet[d]; g = off[k]
            f[0][k] = beta[w, g] - expected[g]
            p = d - 364
            if p >= 0:
                f[1][k] = -(beta[self.wet[p], g] - expected[g])
            if w > 0:
                f[2][k] = c[g][:, None] * (self.now[d].astype(float) - pnow)[None, :]
            elif not np.isnan(self.tdev[d]):
                reg = int(self.tnorm[d] >= HOT)
                eff = np.where(day_hours[None, :], slope[reg, g, 0][:, None], slope[reg, g, 1][:, None]) * self.tdev[d]
                f[3 + (1 - reg)][k] = eff                      # f4 hot season, f5 mild season
        s = int(np.searchsorted(block.panel.stations, MUNHAK))
        for x in f:
            x[:, s] = 0; x[block.blocked] = 0
        self.last_info = info
        return f
