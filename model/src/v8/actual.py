"""Stage 2 modules of v8 — these use facts realised inside the evaluation period, which the organisers allow
as long as they are not subway ridership: games actually played (attendance, end time, cancellations), events
actually held and observed weather.  The insurance submission never imports this module.

A1 munhak    exit crowd at 3127 = attendance x flow kernel aligned on the minute the game ended
A2 bucheon   the same for Bucheon FC home matches at 3754 (kick-off + 115 minutes)
A3 isle      one-day festival at 3139 with a scaled festival profile (no comparable past edition)
A4 weather   wet-day effect by station x day group x hour, applied with the observed rain of the target day,
             plus the echo of last year's rain on the same weekday (+364 days)
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from src.holiday_calendar import calendar_features
from src.v6.munhak import MunhakSpecialist, day_group
from . import core
from .core import nanmed, nanmean, MUNHAK, ORDINARY

BINS = np.array([-120, -60, -30, -15, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90, 150])   # minutes relative to the end of the game
WIDTH = np.diff(BINS)
BUCHEON, SONGDO = 3754, 3139


def minutes(text):
    h, m = str(text).split(':')[:2]
    return int(h) * 60 + int(m)


def overlap(end_minute):
    """[18 hours, bins]: minutes of each service hour that fall into each offset bin, divided by the bin width."""
    lo = end_minute + BINS[:-1]; hi = end_minute + BINS[1:]
    h0 = (np.arange(6, 24) * 60)[:, None]
    return np.clip(np.minimum(h0 + 60, hi[None]) - np.maximum(h0, lo[None]), 0, None) / WIDTH[None]


def load_games():
    ssg = pd.read_csv(core.SHARE / 'games_ssg_munhak.csv', parse_dates=['date'])
    ssg = ssg[(ssg.status == 'played') & ssg.attendance.notna() & ssg.end_time.notna()].copy()
    ssg['end'] = ssg.end_time.map(minutes)
    bfc = pd.read_csv(core.SHARE / 'games_bucheonfc.csv', parse_dates=['date'])
    bfc = bfc[(bfc.status == 'FE') & (bfc.attendance > 0)].copy()
    bfc['end'] = bfc.kickoff.map(minutes) + 115
    return ssg[['date', 'attendance', 'end']], bfc[['date', 'attendance', 'end']]


def fit_kernel(excess, design, normal=None, groups=None):
    """Share of the attendance boarding in each offset bin.  Start: non-negative least squares on counts.
    With `normal` (ordinary counts per game, [n, 18]) the fit is refined in log1p space, which is the metric,
    together with a multiplicative scale per day group (weekday = 1)."""
    theta, _ = nnls(np.concatenate(design), np.concatenate(excess))
    if normal is None:
        return theta, np.ones(3)
    from scipy.optimize import least_squares
    D = np.stack(design); N = np.stack(normal); Y = np.log1p(np.clip(N + np.stack(excess), 0, None)); G = np.asarray(groups)
    m = len(theta)

    def resid(p):
        scale = np.array([1.0, p[m], p[m + 1]])[G][:, None]
        smooth = 0.15 * np.diff(p[:m] / WIDTH * 10) * np.sqrt(len(D))            # mild smoothness of the flow rate
        return np.concatenate([(np.log1p(np.clip(N + scale * (D @ p[:m]), 0, None)) - Y).ravel(), smooth])
    fit = least_squares(resid, np.concatenate([np.maximum(theta, 1e-4), [1.0, 1.0]]),
                        bounds=(np.concatenate([np.zeros(m), [0.5, 0.5]]), np.concatenate([np.ones(m), [1.6, 1.6]])))
    return fit.x[:m], np.array([1.0, fit.x[m], fit.x[m + 1]])


def munhak_feature(block, pred, train, schedule, games):
    """(regression forecast - current forecast) on station 3127 for season days; other rows 0."""
    origin = pd.Timestamp(block.origin)
    model = MunhakSpecialist(schedule).fit(train[train.date <= origin], origin)
    s = int(np.searchsorted(block.panel.stations, MUNHAK))
    hist_days = block.panel.days[:block.O + 1]
    hol = calendar_features(pd.Series(hist_days), as_of=block.origin).is_holiday.to_numpy(bool)
    group = day_group(hist_days, hol)
    counts = np.expm1(block.Z[s])                                           # (days, 18)
    normal = np.expm1(model.normal)
    past = games[games.date <= origin]
    excess, design, base, grp = [], [], [], []
    for date, g in past.groupby('date'):
        t = block.panel.day_index(date)
        if t < 0 or np.isnan(counts[t]).any():
            continue
        excess.append(counts[t] - normal[group[t]]); base.append(normal[group[t]]); grp.append(group[t])
        design.append(sum(a * overlap(e) for a, e in zip(g.attendance, g.end)))
    theta, scale = fit_kernel(excess, design, base, grp)
    # crowd days that are neither a game in the list nor an announced concert: rate and profile of what is left
    md_h = hist_days.month * 100 + hist_days.day
    game_days = {block.panel.day_index(d) for d in past.date}
    concert_days = {block.panel.day_index(pd.Timestamp(e['date'])) for e in schedule.venue_events('munhak_main')}
    idx = np.arange(block.O + 1)
    free = (md_h >= 320) & (md_h <= 1031) & ~np.isin(idx, list(game_days | concert_days)) & ~np.isnan(counts).any(1)
    spike = counts.max(1) >= 1200
    rate = np.array([(spike[free & (group == g)].sum() + 0.5) / (free & (group == g)).sum().clip(1) if (free & (group == g)).any() else 0.0 for g in range(3)])
    rate = np.minimum(rate, model.event_rate)
    feat = np.zeros_like(pred)
    hol_t = calendar_features(pd.Series(block.dates), as_of=block.origin).is_holiday.to_numpy(bool)
    gt = day_group(block.dates, hol_t)
    concerts = {pd.Timestamp(e['date']) for e in schedule.venue_events('munhak_main')}
    future = games[games.date.isin(block.dates)]
    by_date = {d: g for d, g in future.groupby('date')}
    md = block.dates.month * 100 + block.dates.day
    for k, d in enumerate(block.dates):
        if d in concerts:
            continue
        if d in by_date:
            g = by_date[d]
            z = np.log1p(np.clip(normal[gt[k]] + scale[gt[k]] * (sum(a * overlap(e) for a, e in zip(g.attendance, g.end)) @ theta), 0, None))
        elif 320 <= md[k] <= 1031:
            r = rate[gt[k]]
            z = (1 - r) * model.normal[gt[k]] + r * model.event[gt[k]]
        else:
            continue
        feat[k, s] = z - pred[k, s]
    info = dict(games_fitted=len(excess), exit_share_of_attendance=round(float(theta.sum()), 4),
                group_scale=[round(float(x), 3) for x in scale], residual_event_rate=[round(float(x), 4) for x in rate],
                kernel={int(b): round(float(t), 4) for b, t in zip(BINS[:-1], theta)},
                game_days_in_block=int(len(by_date)))
    return [feat], info


def bucheon_feature(block, pred, matches, seasonal_days=()):
    s = int(np.searchsorted(block.panel.stations, BUCHEON))
    origin = pd.Timestamp(block.origin)
    Z = block.Z[s]; hk = block.hist_kind; days = np.arange(block.O + 1); dow = block.panel.dow[:block.O + 1]
    past = matches[matches.date <= origin]
    busy = {block.panel.day_index(d) for d in list(past.date) + list(seasonal_days)}
    clean = np.array([t not in busy for t in days]) & np.isin(hk, ORDINARY)
    excess, design = [], []
    for row in past.itertuples():
        t = block.panel.day_index(row.date)
        if t < 0 or np.isnan(Z[t]).any():
            continue
        near = days[(np.abs(days - t) <= 28) & (dow == dow[t]) & clean]
        if len(near) < 2:
            continue
        excess.append(np.expm1(Z[t]) - np.expm1(nanmed(Z[near], 0)))
        design.append(row.attendance * overlap(row.end))
    feat = np.zeros_like(pred)
    if len(excess) < 5:
        return [feat], dict(matches_fitted=len(excess))
    theta, _ = fit_kernel(excess, design)
    for row in matches[matches.date.isin(block.dates)].itertuples():
        k = int(np.flatnonzero(block.dates == row.date)[0])
        base = block.neighbour(pred, k, allow=ORDINARY)[s]
        if np.isnan(base).any():
            continue
        z = np.log1p(np.clip(np.expm1(base) + row.attendance * overlap(row.end) @ theta, 0, None))
        feat[k, s] = z - pred[k, s]
    return [feat], dict(matches_fitted=len(excess), exit_share_of_attendance=round(float(theta.sum()), 4),
                        match_days_in_block=int(matches.date.isin(block.dates).sum()))


def festival_lift(block, past_days, scale):
    """Scaled mean log lift [18] of past festival days at 3139 against ordinary same-weekday days."""
    s = int(np.searchsorted(block.panel.stations, SONGDO))
    Z = block.Z[s]; hk = block.hist_kind; days = np.arange(block.O + 1); dow = block.panel.dow[:block.O + 1]
    idx = [block.panel.day_index(d) for d in past_days]
    clean = np.isin(hk, ORDINARY) & ~np.isin(days, idx)
    lifts = []
    for t in idx:
        if 0 <= t <= block.O:
            near = days[(np.abs(days - t) <= 35) & (dow == dow[t]) & clean]
            if len(near) >= 2:
                lifts.append(Z[t] - nanmed(Z[near], 0))
    return scale * np.clip(np.mean(lifts, 0), 0, None) if lifts else np.zeros(18)


# ------------------------------------------------------------------------------------------- weather
class Weather:
    def __init__(self, path=None):
        d = pd.read_csv(path or core.EXT / 'weather_incheon_daily_20250101_20261031.csv', parse_dates=['date'])
        self.wet = d.set_index('date').wet

    def level(self, dates):
        return self.wet.reindex(pd.DatetimeIndex(dates)).fillna(0).to_numpy(int)

    def effects(self, block, shrink=3.0):
        """beta[level 1..2, group weekday/off, S, H]: mean deviation of wet days from dry same-class days, <= origin."""
        Z = block.Z; hk = block.hist_kind; O = block.O; days = np.arange(O + 1)
        wet = self.level(block.panel.days[:O + 1]); dow = block.panel.dow[:O + 1]
        off = ~np.isin(hk, ('wd',))
        beta = np.zeros((3, 2, Z.shape[0], Z.shape[2])); count = np.zeros((3, 2))
        for t in days[(wet > 0) & np.isin(hk, ORDINARY)]:
            near = days[(np.abs(days - t) <= 28) & (dow == dow[t]) & (wet == 0) & np.isin(hk, ORDINARY)]
            if len(near) < 2:
                continue
            dev = Z[:, t] - nanmed(Z[:, near], 1)
            beta[wet[t], int(off[t])] += np.nan_to_num(dev); count[wet[t], int(off[t])] += 1
        mean = beta / np.maximum(count, 1)[:, :, None, None]
        net = np.median(mean, axis=2, keepdims=True)
        n = count[:, :, None, None]
        beta = (n * mean + shrink * net) / (n + shrink)
        share = np.array([(wet[np.isin(hk, ORDINARY)] == l).mean() for l in range(3)])
        expected = (share[1, None, None, None] * beta[1] + share[2, None, None, None] * beta[2])       # (2, S, H)
        return beta, expected, dict(wet_days={int(l): int(count[l].sum()) for l in (1, 2)}, wet_share=[round(float(x), 3) for x in share],
                                    mean_effect={f'level{l}_{g}': round(float(beta[l, i].mean()), 4) for l in (1, 2) for i, g in enumerate(('weekday', 'off'))})

    def features(self, block, pred):
        """f1: effect of the observed rain of the target day minus the average rain effect;
        f2: minus the same quantity for the date 364 days earlier (last year's rain echoed by year-over-year inputs)."""
        beta, expected, info = self.effects(block)
        wet_t = self.level(block.dates); prev = block.di - 364
        wet_p = self.level(block.panel.days[np.clip(prev, 0, None)])
        off = (~np.isin(block.kind, ('wd',))).astype(int)
        f1 = np.zeros_like(pred); f2 = np.zeros_like(pred)
        for k in range(len(block.dates)):
            f1[k] = beta[wet_t[k], off[k]] - expected[off[k]]
            if prev[k] >= 0:
                f2[k] = -(beta[wet_p[k], off[k]] - expected[off[k]])
        s = int(np.searchsorted(block.panel.stations, MUNHAK))
        f1[:, s] = 0; f2[:, s] = 0
        f1[block.blocked] = 0; f2[block.blocked] = 0
        self.last_info = info
        return [f1, f2]
