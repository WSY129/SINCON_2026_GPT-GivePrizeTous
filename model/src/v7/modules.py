"""Add-on modules of v7.  Each one maps (frame rows of one origin, current log forecast) -> corrected forecast,
using only observations up to that origin; free coefficients are fitted on local out-of-fold residuals.
"""
import json
import numpy as np
import pandas as pd
from . import growth
from .panel import FI, _nan

MUNHAK = 3127


def row_index(panel, frame, origin):
    O = panel.day_index(origin)
    di = (frame.date.to_numpy() - np.datetime64(origin)).astype('timedelta64[D]').astype(int) - 1
    si = np.searchsorted(panel.stations, frame.station_id.to_numpy()); hi = frame.hour.to_numpy() - 6
    return O, di, si, hi


# ------------------------------------------------------------------ recent-window contamination (gap)
def gap_feature(panel, frame, origin, threshold=0.1):
    """mean - median of the last 8 same-class days: positive when one-off spikes inflate mean-based levels."""
    O, di, si, hi = row_index(panel, frame, origin)
    X, _, _, _ = panel.eval_set(origin, int(di.max()) + 1)
    gap = np.nan_to_num(X[di, si, hi, FI['gap8']])
    keep = (np.abs(gap) > threshold) & (frame.station_id.to_numpy() != MUNHAK) & (frame.is_holiday.to_numpy() == 0)
    return np.where(keep, gap, 0.0)


# ------------------------------------------------------------------ rain g-computation
class Rain:
    def __init__(self, daily_csv, climate_csv, wet_mm=5.0):
        d = pd.read_csv(daily_csv, parse_dates=['date']).set_index('date')
        self.wet = (d.precipitation_sum >= wet_mm)
        c = pd.read_csv(climate_csv, parse_dates=['date'])
        c = c[c.date.dt.year <= 2024]                                    # climatology from years before the data window
        self.climate = (c.precipitation_sum >= wet_mm).groupby(c.date.dt.month).mean()

    def features(self, panel, frame, origin):
        """g1 = effect * wet share of the level window (what the levels absorbed);
        g2 = effect * climatological wet share of the target month (what the target will see on average)."""
        O, di, si, hi = row_index(panel, frame, origin)
        cal = panel.cal(O); cls = cal['cls']
        wet = self.wet.reindex(panel.days[:O + 1]).fillna(False).to_numpy()
        days = np.arange(O + 1)
        beta = np.zeros((2, panel.S, panel.H)); count = np.zeros(2)
        for t in days[wet & (cal['frame'][:O + 1, 0] == 0)]:
            near = days[(np.abs(days - t) <= 28) & (cls[:O + 1] == cls[t]) & ~wet]
            if len(near) < 3:
                continue
            dev = panel.Z[:, t] - _nan(np.nanmedian, panel.Z[:, near], 1)
            g = int(cls[t] >= 5)
            beta[g] += np.nan_to_num(dev); count[g] += 1
        beta = beta / (count[:, None, None] + 3.0)
        share = np.zeros(7)
        for c in range(7):
            idx = days[(cls[:O + 1] == c) & (days > O - 70)][-8:]
            share[c] = wet[idx].mean() if len(idx) else 0.0
        target = O + 1 + di
        tc = cls[target]; g = (tc >= 5).astype(int)
        effect = beta[g, si, hi]
        clim = self.climate.reindex(panel.days.month[target]).to_numpy()
        return effect * share[tc], effect * clim, dict(wet_days=int(wet.sum()), weekday_events=int(count[0]), weekend_events=int(count[1]),
                                                        mean_effect_weekday=float(beta[0].mean()), mean_effect_weekend=float(beta[1].mean()))


def least_squares(features, residual, rows):
    """Coefficients k minimising |residual + features @ k| on the selected rows."""
    X = np.stack(features, 1)[rows]
    return np.linalg.lstsq(X, -residual[rows], rcond=None)[0]


# ------------------------------------------------------------------ growth drift wrapper
def growth_columns(panel, frame, origins):
    """slope arrays (aligned with frame) for every variant."""
    out = []
    for variant in growth.VARIANTS:
        col = np.zeros(len(frame))
        for k, origin in enumerate(origins, start=1):
            rows = (frame.fold == k).to_numpy()
            col[rows] = growth.slope_column(panel, frame[rows], origin, **variant)
        out.append(col)
    return out


def growth_fit(slope_cols, horizon, residual, rows):
    """Best variant and (a, b) on the selected rows (in-sample on those rows)."""
    best = None
    for i, slope in enumerate(slope_cols):
        use = rows & (slope != 0)
        a, b = growth.fit_ab(slope[use], horizon[use], residual[use])
        sse = float(((residual[use] + growth.correction(slope[use], horizon[use], a, b)) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, i, a, b)
    return dict(variant=best[1], a=best[2], b=best[3])
