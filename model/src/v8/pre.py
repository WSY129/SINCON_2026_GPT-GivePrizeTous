"""Stage 1 modules of v8 — everything here respects the first-half cutoff (facts and observations <= origin).

P1 seasonal   multi-year seasonal index from the transit operator's monthly station x hour history (2015-2024)
P2 holiday    station x hour effects of public holidays, lunar holidays, bridge days measured on past holidays
P3 ghost      removal of the dip that last year's holiday left on this year's ordinary day (+364 days)
"""
import numpy as np
import pandas as pd
from . import core
from .core import nanmed, nanmean, MUNHAK, ORDINARY

YEARS = np.arange(2015, 2025)
NORMAL_YEARS = [2016, 2017, 2018, 2019, 2023, 2024]                     # 2020-2022: pandemic and recovery
RECENCY = {2016: 1.0, 2017: 1.0, 2018: 1.0, 2019: 1.5, 2023: 2.5, 2024: 3.0}
CHUSEOK_MONTH = {2015: 9, 2016: 9, 2017: 10, 2018: 9, 2019: 9, 2020: 10, 2021: 9, 2022: 9, 2023: 9, 2024: 9, 2025: 10, 2026: 9}
SEASONAL = dict(cap=0.20, regime_gap=0.12, hour_shrink=0.5, chuseok_bonus=2.0, min_years=2)


class History:
    """Monthly station x hour boardings of the operator (already scaled to the train basis by the team)."""

    def __init__(self, panel):
        ic = pd.read_csv(core.SHARE / 'ictr_hourly_monthly_2015_2024.csv')
        ic = ic[(ic.quality == 'ok') & ic.station_id.isin(panel.stations)]
        y = ic.data_month.str[:4].astype(int).to_numpy() - 2015; m = ic.data_month.str[5:].astype(int).to_numpy() - 1
        s = np.searchsorted(panel.stations, ic.station_id.to_numpy()); h = ic.hour.to_numpy() - 6
        self.B = np.full((panel.S, panel.H, len(YEARS), 12), np.nan)
        self.B[s, h, y, m] = ic.boardings.to_numpy()
        cal = pd.read_csv(core.SHARE / 'calendar_2015_2026.csv', parse_dates=['date'])
        cal = cal[cal.date.dt.year <= 2024]
        lunar = (cal.is_chuseok == 1) | (cal.is_seollal == 1)
        kind = np.where(lunar, 3, np.where((cal.is_holiday == 1) | (cal.dow == 6), 2, np.where(cal.dow == 5, 1, 0)))
        self.counts = np.zeros((len(YEARS), 12, 4))
        np.add.at(self.counts, (cal.date.dt.year.to_numpy() - 2015, cal.date.dt.month.to_numpy() - 1, kind), 1)

    def levels(self, block):
        """log weekday-equivalent daily level [S, H, year, month]; day-type ratios come from data <= origin."""
        Z = np.expm1(block.Z); k = block.hist_kind
        mean = lambda sel: nanmean(Z[:, sel], 1)
        wd = mean(k == 'wd'); r_sat = np.clip(mean(k == 'sat') / wd, 0.05, 3); r_sun = np.clip(mean(k == 'sun') / wd, 0.05, 3)
        r_sat = np.nan_to_num(r_sat, nan=0.6); r_sun = np.nan_to_num(r_sun, nan=0.45)
        c = self.counts
        W = c[None, None, :, :, 0] + c[None, None, :, :, 1] * r_sat[:, :, None, None] \
            + (c[None, None, :, :, 2] + 0.7 * c[None, None, :, :, 3]) * r_sun[:, :, None, None]
        return np.log((self.B + 1) / W)


def weighted_median(values, weights):
    """values [n, ...] with NaN, weights [n] -> weighted median along axis 0 and the count of valid entries."""
    order = np.argsort(np.where(np.isnan(values), np.inf, values), axis=0)
    v = np.take_along_axis(values, order, 0)
    w = np.broadcast_to(np.asarray(weights, float).reshape((-1,) + (1,) * (values.ndim - 1)), values.shape)
    w = np.where(np.isnan(v), 0.0, np.take_along_axis(w, order, 0))
    cum = np.cumsum(w, 0); total = cum[-1]
    pick = np.argmax(cum >= 0.5 * total[None], axis=0)
    out = np.take_along_axis(v, pick[None], 0)[0]
    n = (~np.isnan(values)).sum(0)
    return np.where(total > 0, out, np.nan), n


def month_plan(block):
    """Reference months (the two calendar months ending at the origin) and the target (year, month) of every day."""
    o = pd.Timestamp(block.origin)
    ref = [((o - pd.DateOffset(months=1)).year, (o - pd.DateOffset(months=1)).month), (o.year, o.month)]
    ym = np.array([d.year * 12 + d.month - 1 for d in block.dates])
    counts = pd.Series(ym).value_counts()
    full = sorted(k for k, v in counts.items() if v >= 10)
    target = np.array([min(full, key=lambda f: abs(f - x)) for x in ym])
    return ref, target, full


def seasonal_feature(history, block, pred, params=SEASONAL, plain=False):
    """delta[day, S, H] = clip(prior change - change implied by the forecast), from the reference months."""
    L = history.levels(block)
    ref, target, months = month_plan(block)
    S, H = pred.shape[1:]
    hk = block.hist_kind; hd = block.panel.days[:block.O + 1]
    in_ref = np.isin(hd.year * 12 + hd.month - 1, [y * 12 + m - 1 for y, m in ref]) & (hk == 'wd') & (np.arange(block.O + 1) > block.O - 75)
    ref_level = nanmean(block.Z[:, in_ref], 1)                                         # (S, H)
    delta = np.zeros_like(pred)
    ordinary_wd = block.kind == 'wd'
    target_year = block.dates[-1].year
    for ymi in months:
        ty, tm = divmod(ymi, 12)                                                       # target year, month index
        days = (target == ymi)
        use = days & ordinary_wd
        if use.sum() < 3:
            continue
        implied = pred[use].mean(0) - ref_level
        vals, wts = [], []
        for y in NORMAL_YEARS:
            ry = [(y + (ry_ - ref[-1][0]), rm) for ry_, rm in ref]                     # same offsets as the block's
            yy = y + (ty - ref[-1][0])
            if yy > 2024 or min(a for a, _ in ry) < 2015 or (yy in (2020, 2021, 2022)):
                continue
            change = L[:, :, yy - 2015, tm] - np.mean([L[:, :, a - 2015, m - 1] for a, m in ry], 0)
            w = 1.0 if plain else RECENCY[y]
            if not plain and tm + 1 in (9, 10) and CHUSEOK_MONTH[yy] == CHUSEOK_MONTH.get(ty, 0):
                w *= params['chuseok_bonus']
            vals.append(change); wts.append(w)
        if len(vals) < params['min_years']:
            continue
        vals = np.stack(vals); wts = np.array(wts)
        prior, n = weighted_median(vals, np.ones(len(wts)) if plain else wts)
        if not plain:
            years = [y for y in NORMAL_YEARS][:len(vals)]
            st = nanmean(vals, 2)                                                      # station-level change per year
            recent = nanmean(st[-2:], 0); old = nanmed(st[:-2], 0) if len(vals) > 2 else recent
            shifted = np.abs(recent - old) > params['regime_gap']                      # structural change: trust recent years
            recent_prior = nanmean(vals[-2:], 0)
            prior = np.where(shifted[:, None] & ~np.isnan(recent_prior), recent_prior, prior)
            station = nanmean(prior, 1)[:, None]
            prior = station + params['hour_shrink'] * (prior - station)
        d = prior - implied
        d = np.where((n >= params['min_years']) & ~np.isnan(d), d, 0.0)
        if not plain:
            d = np.clip(d, -params['cap'], params['cap'])
        d[np.searchsorted(block.panel.stations, MUNHAK)] = 0.0
        delta[days] = d[None]
    return delta


# ------------------------------------------------------------------------------------------- holidays
GROUPS = {'regular': ('hol_wd', 'hol_sat', 'labor'), 'lunar_day': ('lunar0',), 'lunar_adjacent': ('lunar_adj',), 'newyear': ('newyear',), 'bridge': ('bridge',),
          'long_weekend': ('long_sat', 'long_sun')}
BASE_KIND = lambda dow: 'wd' if dow < 5 else ('sat' if dow == 5 else 'sun')
ORD = ORDINARY


def holiday_effects(block, shrink=1.0):
    """Mean log effect [S, H] of each special day category, measured on past days <= origin against the median of
    ordinary same-weekday days within four weeks, shrunk towards the network median of that category."""
    Z = block.Z; hk = block.hist_kind; O = block.O
    dow = block.panel.dow[:O + 1]; days = np.arange(O + 1)
    out = {}
    for cat in core.SPECIAL:
        effects = []
        for t in days[hk == cat]:
            near = days[(np.abs(days - t) <= 28) & (dow == dow[t]) & (hk == BASE_KIND(dow[t]))]
            if len(near) < 2:
                near = days[(np.abs(days - t) <= 49) & (dow == dow[t]) & (hk == BASE_KIND(dow[t]))]
            if len(near) < 2:
                continue
            effects.append(Z[:, t] - nanmed(Z[:, near], 1))
        if not effects:
            continue
        e = np.stack(effects); n = (~np.isnan(e)).sum(0)
        mean = nanmean(e, 0)
        net = nanmed(mean, 0)[None, :]
        out[cat] = np.where(n > 0, (n * np.nan_to_num(mean) + shrink * net) / (n + shrink), net)
    return out


ANOMALY = 0.4          # a station is "atypical" for a category when its mean effect differs this much (log) from the network's


def holiday_features(block, pred):
    """Two features per group: (specialist forecast - current forecast) on the special days of that group, for
    typical stations and for stations whose own measured effect is far from the network effect (cemetery
    visits at 부평삼거리 on lunar holidays, industrial-estate stations on any holiday)."""
    eff = holiday_effects(block)
    names = list(GROUPS)
    feats = {(g, a): np.zeros_like(pred) for g in names for a in (0, 1)}
    munhak = np.searchsorted(block.panel.stations, MUNHAK)
    for k, cat in enumerate(block.kind):
        if cat not in eff:
            continue
        base = block.neighbour(pred, k, allow=(BASE_KIND(block.dates[k].dayofweek),))
        spec = base + eff[cat]
        group = next(g for g, cats in GROUPS.items() if cat in cats)
        d = np.where(np.isnan(spec), 0.0, spec - pred[k])
        d[block.blocked[k]] = 0.0; d[munhak] = 0.0
        odd = np.abs((eff[cat] - nanmed(eff[cat], 0)[None]).mean(1)) > ANOMALY
        feats[(group, 1)][k, odd] = d[odd]; feats[(group, 0)][k, ~odd] = d[~odd]
    return [feats[(g, a)] for g in names for a in (0, 1)]


def ghost_days(block):
    """Ordinary days whose date 364 days earlier was a special day (as known at the origin)."""
    prev = block.di - 364
    ok = prev >= 0
    kinds = core.day_kinds(block.panel.days[np.where(ok, prev, 0)], block.origin)
    return ok & np.isin(kinds, core.SPECIAL) & np.isin(block.kind, ORDINARY)


def ghost_feature(block, pred):
    ghost = ghost_days(block)
    feat = np.zeros_like(pred)
    munhak = np.searchsorted(block.panel.stations, MUNHAK)
    clean = np.isin(block.kind, ORDINARY) & ~ghost
    for k in np.flatnonzero(ghost):
        base = block.neighbour(pred, k, ok=clean & (block.dates.dayofweek == block.dates[k].dayofweek))
        d = np.where(np.isnan(base), 0.0, base - pred[k])
        d[block.blocked[k]] = 0.0; d[munhak] = 0.0
        feat[k] = d
    return [feat]
