"""Growth drift of stations that opened inside the data window (the Geomdan extension: 3107, 3108, 3109).

Every earlier version anchors on recent levels, so a station that is still filling up is forecast too low and
the gap widens with the horizon.  The drift is the station's own recent trend relative to the old stations:

    slope_s = OLS slope (per day) of  T_s(t) - mean_old T(t)  over ordinary weekdays in the last `window` days
    correction(s, h) = slope_s * (a + b * h)            (log units, added to the forecast, capped)

Days in the first three weeks after a station opened are left out of the slope (opening-week novelty), and a
station younger than 111 days (three settling weeks + a 90-day record) gets no correction at all: at the first
validation origin the new stations were 63 days old and a slope fitted there has the wrong sign.
T is the log of the daily total and slope uses observations <= origin only.  `pool` shrinks each station's slope
towards the common slope of the new stations (0 = own slope, 1 = common slope).  The common slope is the median of the
stations' slopes: at origin 2026-05-15 one station's spring-park demand inflated its own slope fivefold.  The shipped setting has no fitted
coefficient: common slope, a = 28 days (the level lag of an 8-week mean), b = 1 (the trend continues); a
least-squares fit of (a, b) on out-of-fold residuals is kept only as a diagnostic (fit_ab).
"""
import numpy as np
import pandas as pd

VARIANTS = [dict(window=56, pool=1.0, robust=True)]   # 8-week window, common slope = median over the new stations
LAG, PERSISTENCE = 28.0, 1.0                # a = half the window (level lag of a window mean), b = trend continues
CAP = 0.25


def new_stations(panel):
    return panel.stations[panel.first > 0]


def slopes(panel, origin, window=56, pool=1.0, min_days=15, settle=21, min_age=111, robust=True):
    O = panel.day_index(origin)
    cls = panel.cal(O)['cls']
    days = np.arange(max(O - window + 1, 0), O + 1); days = days[cls[days] <= 4]
    old = np.flatnonzero((panel.first == 0) & (panel.stations != 3127))
    ref = np.nanmean(panel.T[old][:, days], 0)
    out = {}
    for sid in new_stations(panel):
        s = int(np.searchsorted(panel.stations, sid))
        if O - panel.first[s] < min_age:                 # too young for a trend: the module abstains
            continue
        rel = panel.T[s, days] - ref; ok = ~np.isnan(rel) & (days >= panel.first[s] + settle)
        if ok.sum() < min_days:
            continue
        x = days[ok] - days[ok].mean(); y = rel[ok] - rel[ok].mean()
        out[int(sid)] = float((x * y).sum() / (x * x).sum())
    if out and pool > 0:
        centre = float(np.median(list(out.values())) if robust else np.mean(list(out.values())))
        out = {k: (1 - pool) * v + pool * centre for k, v in out.items()}
    return out


def slope_column(panel, frame, origin, **variant):
    """Slope of each row's station (0 for stations that are not new)."""
    table = slopes(panel, origin, **variant)
    return frame.station_id.map(table).fillna(0.0).to_numpy()


def fit_ab(slope, horizon, residual):
    """Least squares of  -residual ~ slope * (a + b * horizon)  on rows with a non-zero slope."""
    use = slope != 0
    X = np.stack([slope[use], slope[use] * horizon[use]], 1)
    coef = np.linalg.lstsq(X, -residual[use], rcond=None)[0]
    return float(coef[0]), float(coef[1])


def correction(slope, horizon, a, b):
    return np.clip(slope * (a + b * horizon), -CAP, CAP)
