"""v10 event modules (realised facts of the evaluation period are allowed; subway ridership is not used).

bucheon   v8 skipped four Saturday matches: every Saturday within four weeks was itself a match day, so the
          ordinary-Saturday level could not be found and the row was left unchanged.  The search now reaches
          seven weeks on each side.  Cup home matches listed in configs/ensemble_v10.json are added.
munhak    the v8 regression plus the non-league events actually held at the stadium; on season days with no
          game and no listed event the left-over spike probability is measured after removing the days that
          are now identified.
recurring an event with an earlier edition inside the data (science festival at the Asiad stadium, Bupyeong
          Pungmul festival) gets the lift measured on that edition x 0.85, as the v6 overlay does.
"""
import json
import numpy as np
import pandas as pd
from src.holiday_calendar import calendar_features
from src.v6.munhak import MunhakSpecialist, day_group
from src.v8 import core as c8, actual
from src.v8.core import nanmed, MUNHAK, ORDINARY

WIDE = (7, 14, 21, 28, 35, 42, 49)
CONFIG = c8.ROOT / 'configs/ensemble_v10.json'


def config():
    return json.loads(CONFIG.read_text())


def station_level(block, pred, k, s, offsets=WIDE, n=4):
    """Median forecast [18] of the n nearest same-weekday ordinary days that are not event days *of this station*.
    (v8 picked the four nearest ordinary days first and only then dropped the station's event days, which left
    nothing when a run of consecutive Saturdays were all match days.)"""
    found = []
    for o in offsets:
        for sign in (-1, 1):
            j = k + sign * o
            if 0 <= j < len(block.dates) and block.kind[j] in ORDINARY and not block.blocked[j, s]:
                found.append(pred[j, s])
        if len(found) >= n:
            break
    return np.median(found, 0) if found else None


def bucheon_matches(cfg=None):
    cfg = cfg or config()
    _, bfc = actual.load_games()
    extra = pd.DataFrame([dict(date=pd.Timestamp(m['date']), attendance=float(m['attendance']), end=actual.minutes(m['kickoff']) + 115)
                          for m in cfg.get('extra_matches_bucheon', []) if m.get('attendance')])
    return pd.concat([bfc, extra], ignore_index=True) if len(extra) else bfc


def bucheon_feature(block, pred, matches):
    s = int(np.searchsorted(block.panel.stations, actual.BUCHEON))
    origin = pd.Timestamp(block.origin)
    Z = block.Z[s]; hk = block.hist_kind; days = np.arange(block.O + 1); dow = block.panel.dow[:block.O + 1]
    past = matches[matches.date <= origin]
    busy = {block.panel.day_index(d) for d in past.date}
    clean = np.array([t not in busy for t in days]) & np.isin(hk, ORDINARY)
    excess, design = [], []
    for row in past.itertuples():
        t = block.panel.day_index(row.date)
        if t < 0 or np.isnan(Z[t]).any():
            continue
        near = days[(np.abs(days - t) <= 28) & (dow == dow[t]) & clean]
        if len(near) < 2:
            near = days[(np.abs(days - t) <= 49) & (dow == dow[t]) & clean]
        if len(near) < 2:
            continue
        excess.append(np.expm1(Z[t]) - np.expm1(nanmed(Z[near], 0)))
        design.append(row.attendance * actual.overlap(row.end))
    feat = np.zeros_like(pred)
    theta, _ = actual.fit_kernel(excess, design)
    done, skipped = 0, []
    for row in matches[matches.date.isin(block.dates)].itertuples():
        k = int(np.flatnonzero(block.dates == row.date)[0])
        base = station_level(block, pred, k, s)
        if base is None:
            skipped.append(str(row.date.date())); continue
        feat[k, s] = np.log1p(np.clip(np.expm1(base) + row.attendance * actual.overlap(row.end) @ theta, 0, None)) - pred[k, s]
        done += 1
    return [feat], dict(matches_fitted=len(excess), exit_share_of_attendance=round(float(theta.sum()), 4), match_days_applied=done, skipped=skipped)


def lift_profile(block, sid, past_days, window=35):
    """Mean log lift [18] of past event days at a station against ordinary same-weekday days, and the sample size."""
    s = int(np.searchsorted(block.panel.stations, sid))
    Z = block.Z[s]; hk = block.hist_kind; days = np.arange(block.O + 1); dow = block.panel.dow[:block.O + 1]
    idx = [block.panel.day_index(pd.Timestamp(d)) for d in past_days]
    clean = np.isin(hk, ORDINARY) & ~np.isin(days, idx)
    lifts = []
    for t in idx:
        if 0 <= t <= block.O and not np.isnan(Z[t]).any():
            near = days[(np.abs(days - t) <= window) & (dow == dow[t]) & clean]
            if len(near) >= 2:
                lifts.append(Z[t] - nanmed(Z[near], 0))
    return (np.mean(lifts, 0), len(lifts)) if lifts else (np.zeros(18), 0)


def apply_recurring(block, pred, events):
    """Adds shrink x past lift on the listed dates (test block only; these editions cannot be validated on folds)."""
    out = pred.copy(); log = []
    for e in events:
        for sid in e['stations']:
            lift, n = lift_profile(block, sid, e['past_dates'])
            if n == 0:
                continue
            s = int(np.searchsorted(block.panel.stations, sid))
            for d in e['dates']:
                d = pd.Timestamp(d)
                if d not in block.dates:
                    continue
                k = int(np.flatnonzero(block.dates == d)[0])
                block.blocked[k, s] = True
                base = station_level(block, out, k, s)
                if base is None:
                    base = out[k, s]
                if e.get('mode') == 'add':             # no earlier echo on this date: add the lift to the current forecast
                    new = out[k, s] + e.get('shrink', 0.85) * np.clip(lift, 0, None)
                else:                                  # same weekday as last year's edition: replace echo by level + lift
                    new = base + e.get('shrink', 0.85) * lift
                log.append(dict(event=e['name'], station=int(sid), date=str(d.date()), past_editions=n, daily_before=round(float(np.expm1(out[k, s]).sum())),
                                daily_after=round(float(np.expm1(new).sum())), max_log_lift=round(float(lift.max()), 3)))
                out[k, s] = new
    return out, log


def munhak_feature(block, pred, train, schedule, games, cfg=None):
    """v8 regression with (1) listed non-league events and (2) a left-over spike rate measured after removing
    every identified event day from the history."""
    cfg = cfg or config()
    origin = pd.Timestamp(block.origin)
    model = MunhakSpecialist(schedule).fit(train[train.date <= origin], origin)
    s = int(np.searchsorted(block.panel.stations, MUNHAK))
    hist_days = block.panel.days[:block.O + 1]
    hol = calendar_features(pd.Series(hist_days), as_of=block.origin).is_holiday.to_numpy(bool)
    group = day_group(hist_days, hol)
    counts = np.expm1(block.Z[s]); normal = np.expm1(model.normal)
    past = games[games.date <= origin]
    excess, design, base, grp = [], [], [], []
    for date, g in past.groupby('date'):
        t = block.panel.day_index(date)
        if t < 0 or np.isnan(counts[t]).any():
            continue
        excess.append(counts[t] - normal[group[t]]); base.append(normal[group[t]]); grp.append(group[t])
        design.append(sum(a * actual.overlap(e) for a, e in zip(g.attendance, g.end)))
    theta, scale = actual.fit_kernel(excess, design, base, grp)
    md_h = hist_days.month * 100 + hist_days.day
    idx = np.arange(block.O + 1)
    game_days = {block.panel.day_index(d) for d in past.date}
    concert_days = {block.panel.day_index(pd.Timestamp(e['date'])) for e in schedule.venue_events('munhak_main')}
    known_days = {block.panel.day_index(pd.Timestamp(d)) for d in cfg.get('munhak_past_events', [])}
    spike = counts.max(1) >= 1200
    free = (md_h >= 320) & (md_h <= 1031) & ~np.isin(idx, list(game_days | concert_days)) & ~np.isnan(counts).any(1)
    empty = free & ~np.isin(idx, list(known_days))
    rate = np.array([(spike[empty & (group == g)].sum() + 0.5) / max((empty & (group == g)).sum(), 1) for g in range(3)])
    rate = np.minimum(rate, model.event_rate)
    # listed non-league games (불꽃야구 직관): attendance is not reported, so use the attendance-equivalent implied by
    # the boardings of identified earlier editions (<= origin) through the same exit kernel
    implied = [float(np.clip(counts[t] - normal[group[t]], 0, None).sum() / max(theta.sum(), 1e-6)) for t in sorted(known_days) if 0 <= t <= block.O and not np.isnan(counts[t]).any()]
    equivalent = float(np.median(implied)) if implied else 8000.0
    feat = np.zeros_like(pred)
    hol_t = calendar_features(pd.Series(block.dates), as_of=block.origin).is_holiday.to_numpy(bool)
    gt = day_group(block.dates, hol_t)
    concerts = {pd.Timestamp(e['date']) for e in schedule.venue_events('munhak_main')}
    by_date = {d: g for d, g in games[games.date.isin(block.dates)].groupby('date')}
    listed = {pd.Timestamp(e['date']): e for e in cfg.get('munhak_events', [])}
    md = block.dates.month * 100 + block.dates.day
    used = []
    for k, d in enumerate(block.dates):
        if d in concerts:
            continue
        if d in by_date:
            g = by_date[d]
            z = np.log1p(np.clip(normal[gt[k]] + scale[gt[k]] * (sum(a * actual.overlap(e) for a, e in zip(g.attendance, g.end)) @ theta), 0, None))
        elif d in listed:
            e = listed[d]
            end = actual.minutes(e['end']) if e.get('end') else actual.minutes(e['start']) + 190
            z = np.log1p(np.clip(normal[gt[k]] + float(e.get('attendance') or equivalent) * actual.overlap(end) @ theta, 0, None))
            used.append(str(d.date()))
        elif 320 <= md[k] <= 1031:
            r = rate[gt[k]]
            z = (1 - r) * model.normal[gt[k]] + r * model.event[gt[k]]
        else:
            continue
        feat[k, s] = z - pred[k, s]
    return [feat], dict(games_fitted=len(excess), exit_share_of_attendance=round(float(theta.sum()), 4), group_scale=[round(float(x), 3) for x in scale],
                        left_over_spike_rate=[round(float(x), 4) for x in rate], v6_event_rate=[round(float(x), 4) for x in model.event_rate],
                        identified_past_event_days=int(len(known_days)), attendance_equivalent=round(equivalent), listed_events_used=used, game_days_in_block=int(len(by_date)))
