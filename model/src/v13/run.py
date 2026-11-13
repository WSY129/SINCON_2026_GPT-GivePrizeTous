"""Builds submission_ensemble_v13.csv = v12 + three corrections fitted on local out-of-fold residuals (CPU only).

    python -m src.v13.run

Every correction is fitted for a fold on the other folds only AND without the rows whose dates fall inside that
fold: the folds overlap by two months, so the same day's truth would otherwise leak into the fit.  A correction is
kept only if the all-fold score improves and no fold gets worse.  No leaderboard value is read or used.

  events   concerts and volleyball days that carry no lift in v12 yet: mean residual of earlier days of the same kind
           at the same station, by hour (changes fold forecasts only: no such day is left in the test period)
  games    hours around the end of home games (Munhak baseball, Bucheon football): mean residual by hour offset from
           the end hour and by the third of the hour in which the game ended
  school   change in the share of schools on vacation between the anchor window (8 weeks before the cutoff) and the
           target day, times a school-hour effect (06-08h and 15-17h): one global slope each + shrunk station slopes
  low      calibration of very small forecasts (under 5 boardings), which v12 under-predicts in every fold
"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v11 import stack as s11
from src.v11.panel import Panel11

OUT = c8.ROOT / 'outputs/ensemble_v13'
EXT = OUT / 'external'
KINDS = ['concert', 'festival', 'seasonal', 'sports', 'kovo', 'art', 'bac', 'univfest']
EVENT_FEATURES = ['station', 'hour', 'day_group'] + ['is_' + k for k in KINDS] + ['start_hour', 'hours_since_start', 'typical', 'lift_in_v12', 'v12']
EVENT_KINDS_USED = ['concert', 'kovo']     # kinds corrected (the only ones that gain in every fold where they occur)
LIFT = 0.5                 # an event day whose v12 forecast is already this far (log) above its typical day at some hour is left alone
EVENT_SHRINK_DAYS, EVENT_MIN_DAYS = 1.0, 2
GAME_STATIONS = [3127, 3754]
GAME_SHRINK, GAME_MIN = 5.0, 5
AM, PM = (6, 8), (15, 17)                    # school-hour groups (inclusive)
RIDGE_GLOBAL, RIDGE_STATION = 1.0, 300.0
LOW_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
LOW_SHRINK = 50.0
TOL = 1e-5


def hhmm(x):
    s = str(x)
    return int(s[:2]) + int(s[3:5]) / 60 if len(s) >= 5 and s[:2].isdigit() and s[3:5].isdigit() else np.nan


def day_group(kind):
    k = np.asarray(kind)
    return np.where(k == 'wd', 0, np.where(k == 'sat', 1, np.where(k == 'sun', 2, 3)))


def load_events(stations):
    """{(station id, 'YYYY-MM-DD'): dict(kinds=set, start=hour or nan)} from published schedules only."""
    known = {int(s) for s in stations}; ev = {}

    def add(sid, d, kind, start=np.nan):
        if not str(sid).strip().isdigit() or int(sid) not in known:
            return
        e = ev.setdefault((int(sid), str(d)[:10]), dict(kinds=set(), start=np.nan)); e['kinds'].add(kind)
        if np.isfinite(start):
            e['start'] = start if not np.isfinite(e['start']) else min(e['start'], start)
    for e in json.loads((c8.ROOT / 'outputs/ensemble_v7/domain_schedule_v7.json').read_text())['events']:   # own research (v6/v7): start times
        if e.get('venue') in ('asiad_main', 'munhak_main') and e.get('family') != 'sustainable_wave_festival':
            for sid in e['stations']:
                add(sid, e['date'], 'concert', hhmm(e.get('start')))
        elif e.get('venue') == 'songdo_festival_park':
            for sid in e['stations']:
                add(sid, e['date'], 'festival', hhmm(e.get('start')))
    for fn in ('events_venues.csv', 'events_venues_added.csv'):
        for r in pd.read_csv(EXT / fn).itertuples():
            if pd.isna(r.station_ids) or str(r.confidence).startswith(('data', 'low')):      # dates inferred from ridership, or venue unconfirmed
                continue
            for d in pd.date_range(r.start_date, r.end_date if pd.notna(r.end_date) else r.start_date):
                for sid in str(r.station_ids).split(';'):
                    add(sid, d, str(r.type), hhmm(r.start_time))
    kv = pd.read_csv(EXT / 'games_kovo_incheon.csv'); kv = kv[kv.station_id.notna()]
    for r in kv.itertuples():
        add(int(r.station_id), r.date, 'kovo', hhmm(r.start_time))
    ac = pd.read_csv(EXT / 'artcenter_incheon_performances.csv'); ac = ac[ac.hall.astype(str).str.contains('대공연장')]
    for r in ac.itertuples():
        for d in pd.date_range(r.start_date, r.end_date):
            add(3125, d, 'art', float(r.start_hour_first) if pd.notna(r.start_hour_first) else np.nan)
    bac = pd.read_csv(EXT / 'bac_bucheon_performances.csv'); bac = bac[bac.hall.astype(str).str.contains('콘서트홀')]
    for r in bac.itertuples():
        for d in pd.date_range(r.start_date, r.end_date):
            for sid in (3756, 3757):
                add(sid, d, 'bac')
    for r in pd.read_csv(EXT / 'incheon_univ_festival_labels.csv', encoding='utf-8-sig').itertuples():
        add(r.station_id, r.date, 'univfest')
    cfg10 = json.loads((c8.ROOT / 'configs/ensemble_v10.json').read_text()); cfg11 = json.loads((c8.ROOT / 'configs/ensemble_v11.json').read_text())
    for cfg in (cfg10, cfg11):                                                               # realised events researched for v10 / v11
        for e in cfg.get('station_events', []):
            if e.get('mode') == 'signed':                                                    # dips (study weeks) stay with the v11 handling
                continue
            for d in e['dates'] + e['past_dates']:
                for sid in e['stations']:
                    add(sid, d, 'festival')
    for e in cfg10.get('munhak_events', []):
        add(3127, e['date'], 'sports', hhmm(e.get('start')))
    for d in cfg10.get('munhak_past_events', []):
        add(3127, d, 'sports')
    return ev


def event_cells(block, pred, ev, stations):
    """Design rows of every cell of the listed event station-days inside a block."""
    n, ns, nh = pred.shape; idx = {str(d)[:10]: i for i, d in enumerate(block.dates)}; sidx = {int(s): j for j, s in enumerate(stations)}; grp = day_group(block.kind)
    inside = [(key, e) for key, e in sorted(ev.items()) if key[1] in idx]
    is_ev = np.zeros((n, ns), bool)
    for (sid, d), _ in inside:
        is_ev[idx[d], sidx[sid]] = True
    typical = np.zeros((4, ns, nh))
    for g in range(4):
        days = grp == g
        if not days.any():
            continue
        for s in np.flatnonzero(is_ev.any(0)):
            m = days & ~is_ev[:, s]
            typical[g, s] = np.median(pred[m if m.sum() >= 3 else days, s], 0)
    X, pos, dates = [], [], []
    for (sid, d), e in inside:
        i, s = idx[d], sidx[sid]; start = e['start'] if np.isfinite(e['start']) else -1.0
        for h in range(nh):
            t = typical[grp[i], s, h]
            X.append([s, h + 6, grp[i]] + [float(k in e['kinds']) for k in KINDS] + [start, (h + 6 - start) if start >= 0 else -9.0, t, pred[i, s, h] - t, pred[i, s, h]])
            pos.append((i, s, h)); dates.append(block.dates[i])
    return np.array(X, np.float32).reshape(-1, len(EVENT_FEATURES)), pos, np.array(dates, dtype='datetime64[ns]')


def vacation_change(block, share):
    """share on the target day minus the mean share over the weekdays of the 8 weeks before the cutoff; 0 off weekdays."""
    O = pd.Timestamp(block.origin); a = pd.date_range(O - pd.Timedelta(days=55), O); a = a[a.dayofweek < 5]
    base = float(share.reindex([str(x)[:10] for x in a]).fillna(0).mean())
    return (share.reindex([str(x)[:10] for x in block.dates]).fillna(0).to_numpy() - base) * (np.asarray(block.kind) == 'wd'), base


def school_fit(parts, ns, new_station):
    """parts: list of (delta [days], residual [days, S, H]).  Slopes [2 hour groups, S] = global + shrunk station part.
    The residual of the school hours is taken against the mean residual of the other hours of the same station-day,
    so a level error of a station (growth, seasonal drift) cannot pass for a school effect.  Stations opened inside the
    data window get the global slope only: their morning commute is still growing, which looks like a school effect."""
    out = np.zeros((2, ns)); glob = np.zeros(2); rest = np.ones(18, bool); rest[AM[0] - 6:AM[1] - 6 + 1] = False; rest[PM[0] - 6:PM[1] - 6 + 1] = False
    for g, (lo, hi) in enumerate((AM, PM)):
        nh = hi - lo + 1; A = np.zeros((1 + ns, 1 + ns)); b = np.zeros(1 + ns)
        for delta, res in parts:
            contrast = res[:, :, lo - 6:hi - 6 + 1] - res[:, :, rest].mean(2, keepdims=True)
            q = float((delta ** 2).sum()) * nh; r = (delta[:, None] * contrast.sum(2)).sum(0)                              # [S]
            A[0, 0] += q * ns; A[0, 1:] += q; A[1:, 0] += q; A[1:, 1:] += np.eye(ns) * q; b[0] += r.sum(); b[1:] += r
        A += np.diag([RIDGE_GLOBAL] + [RIDGE_STATION] * ns); beta = np.linalg.solve(A, b); glob[g] = beta[0]; out[g] = beta[0] + np.where(new_station, 0.0, beta[1:])
    return out, glob


def school_apply(delta, slopes, shape):
    c = np.zeros(shape)
    for g, (lo, hi) in enumerate((AM, PM)):
        c[:, :, lo - 6:hi - 6 + 1] = delta[:, None, None] * slopes[g][None, :, None]
    return c


def low_fit(parts):
    """parts: list of (forecast log1p, residual).  Mean residual per forecast-count bin, shrunk towards zero."""
    add = np.zeros(len(LOW_EDGES) - 1)
    for i, (lo, hi) in enumerate(zip(LOW_EDGES[:-1], LOW_EDGES[1:])):
        num = den = 0.0
        for f, r in parts:
            m = (np.expm1(f) >= lo) & (np.expm1(f) < hi); num += float(r[m].sum()); den += float(m.sum())
        add[i] = num / (den + LOW_SHRINK)
    return add


def low_apply(f, add):
    c = np.zeros_like(f); cnt = np.expm1(f)
    for i, (lo, hi) in enumerate(zip(LOW_EDGES[:-1], LOW_EDGES[1:])):
        c[(cnt >= lo) & (cnt < hi)] = add[i]
    return c


def main():
    S = s11.Stack(); p = S.panel; folds = S.folds; test = S.test; st = p.stations; ns = len(st)
    names = pd.read_csv(c8.ROOT / 'data/raw/metaData.csv').set_index('station_id').station_name
    o12 = np.load(c8.ROOT / 'outputs/ensemble_v12/oof_v12.npz'); v12 = {k: o12[f'fold{k}'].astype(float) for k in folds}
    sub12 = pd.read_csv(c8.ROOT / 'outputs/ensemble_v12/submission_ensemble_v12.csv')
    t12 = np.log1p(sub12.boardings.to_numpy())[test.row_index].reshape(test.pred.shape)
    keep = lambda k: {j: np.asarray((folds[j].dates < folds[k].dates.min()) | (folds[j].dates > folds[k].dates.max())) for j in folds if j != k}   # date-blocked rows of the other folds
    out = dict(information='local OOF only; every fit is nested and date-blocked on aligned truth; no leaderboard value is used',
               protocol='for fold k a correction is fitted on the other folds without the days that fall inside fold k; kept only if the all-fold score improves and no fold gets worse')
    out['generations'] = {'v12': dict(aligned=S.score(v12), raw_truth=S.score(v12, raw=True))}
    cur = {k: v12[k].copy() for k in folds}; cur_t = t12.copy(); steps = {}

    def judge(name, new, extra):
        a, b = S.score(cur), S.score(new)
        ok = b['all_oof'] < a['all_oof'] - 1e-7 and all(b[f'fold_{k}'] <= a[f'fold_{k}'] + TOL for k in folds)
        steps[name] = dict(before=a, after=b, adopted=bool(ok), **extra)
        print(f"{name:8s} {a['all_oof']:.5f} -> {b['all_oof']:.5f}  " + ' '.join(f"F{k} {b[f'fold_{k}'] - a[f'fold_{k}']:+.5f}" for k in folds) + ('  ADOPTED' if ok else '  rejected'), flush=True)
        return ok

    # ---- 1. events ------------------------------------------------------------------------------------------------------
    # mean residual of earlier event days of the same kind at the same station, by hour and weekday / weekend, shrunk by
    # one day; needs two analogue days.  Only for event days that carry no lift in v12 yet ("open" days), and only for
    # concerts and volleyball: performances, festivals and ballpark sports events gain nothing or lose on the folds.
    ev = load_events(st)
    cells = {k: event_cells(folds[k], v12[k], ev, st) for k in folds}; cells_t = event_cells(test, t12, ev, st)
    LI, ST_, HR_, DG_ = (EVENT_FEATURES.index(c) for c in ('lift_in_v12', 'station', 'hour', 'day_group'))
    res = {k: np.array([S.truth[k][i, s, h] - cur[k][i, s, h] for i, s, h in cells[k][1]]) for k in folds}
    day_open = lambda X: np.repeat(X[:, LI].reshape(-1, 18).max(1) < LIFT, 18)
    key_of = lambda x: (int(x[ST_]), int(x[HR_]), int(x[DG_] > 0))

    def event_means(kind, train):
        """train: list of (X, residual) -> {(station, hour, weekend): (sum, cells)} over open days of that kind."""
        c = EVENT_FEATURES.index('is_' + kind); acc = {}
        for X, r in train:
            m = (X[:, c] > 0) & day_open(X)
            for x, v in zip(X[m], r[m]):
                a = acc.setdefault(key_of(x), [0.0, 0]); a[0] += float(v); a[1] += 1
        return acc

    def event_apply(kind, acc, X):
        c = EVENT_FEATURES.index('is_' + kind); m = (X[:, c] > 0) & day_open(X); f = np.zeros(len(X))
        f[m] = [(lambda a: a[0] / (a[1] + EVENT_SHRINK_DAYS) if a[1] >= EVENT_MIN_DAYS else 0.0)(acc.get(key_of(x), (0.0, 0))) for x in X[m]]
        return f
    new = {k: cur[k].copy() for k in folds}; gain_kind = {kd: {} for kd in EVENT_KINDS_USED}; used_days = {kd: {} for kd in EVENT_KINDS_USED}
    for k in folds:
        lo, hi = np.datetime64(folds[k].dates.min()), np.datetime64(folds[k].dates.max()); f = np.zeros(len(res[k]))
        train = [(cells[j][0][(cells[j][2] < lo) | (cells[j][2] > hi)], res[j][(cells[j][2] < lo) | (cells[j][2] > hi)]) for j in folds if j != k]
        for kd in EVENT_KINDS_USED:
            fk = event_apply(kd, event_means(kd, train), cells[k][0]); f += fk
            gain_kind[kd][k] = round(float((res[k] ** 2 - (res[k] - fk) ** 2).sum()), 1); used_days[kd][k] = int((fk != 0).sum() // 18)
        for (i, s, h), v in zip(cells[k][1], f):
            new[k][i, s, h] = max(cur[k][i, s, h] + v, 0.0)
    gain = sum(((cur[k] - S.truth[k]) ** 2).sum(axis=(0, 2)) - ((new[k] - S.truth[k]) ** 2).sum(axis=(0, 2)) for k in folds)
    info = dict(kinds_used=EVENT_KINDS_USED, lift_threshold=LIFT, shrink_days=EVENT_SHRINK_DAYS, min_analogue_days=EVENT_MIN_DAYS,
                listed_event_station_days={k: len(cells[k][1]) // 18 for k in folds}, corrected_station_days=used_days, sse_gain_by_kind_and_fold=gain_kind,
                sse_gain_by_station={f'{int(st[i])} {names[int(st[i])]}': round(float(gain[i]), 1) for i in np.argsort(-gain)[:4]})
    if judge('events', new, info):
        cur = new; f = np.zeros(len(cells_t[1])); train = [(cells[j][0], res[j]) for j in folds]
        for kd in EVENT_KINDS_USED:
            f += event_apply(kd, event_means(kd, train), cells_t[0])
        for (i, s, h), v in zip(cells_t[1], f):
            cur_t[i, s, h] = max(cur_t[i, s, h] + v, 0.0)
        steps['events'].update(test_station_days_changed=int((f != 0).sum() // 18),
                               note='no concert day of the test period is still without a lift in v12 and there is no volleyball at 3759 in July-October, so this step changes fold forecasts only')

    # ---- 1b. hours around the end of home games (Munhak baseball, Bucheon football) ------------------------------------------
    # mean residual by venue, hour offset from the end hour (-1 .. +3) and the third of the hour in which the game ended
    wx = Panel11(); venues = [int(np.searchsorted(st, v)) for v in GAME_STATIONS]

    def game_cells(block):
        rows = []
        for s in venues:
            att = np.nan_to_num(wx.game['att'][s][block.di]); end = wx.game['end'][s][block.di]
            for i in np.flatnonzero(att > 0):
                eh, band = int(end[i] // 60), int(end[i] % 60) // 20
                rows += [(i, s, eh + off - 6, (s, off, band)) for off in range(-1, 4) if 0 <= eh + off - 6 < 18]
        return rows

    def game_means(train):
        acc = {}
        for key, v in train:
            a = acc.setdefault(key, [0.0, 0]); a[0] += v; a[1] += 1
        return {key: a[0] / (a[1] + GAME_SHRINK) for key, a in acc.items() if a[1] >= GAME_MIN}
    gc = {k: game_cells(folds[k]) for k in folds}; gc_t = game_cells(test)
    before = {k: cur[k].copy() for k in folds}; new = {k: cur[k].copy() for k in folds}; gsse = {}
    for k in folds:
        m = keep(k); mean = game_means([(key, S.truth[j][i, s, h] - before[j][i, s, h]) for j in m for i, s, h, key in gc[j] if m[j][i]]); a = b_ = 0.0
        for i, s, h, key in gc[k]:
            new[k][i, s, h] = max(before[k][i, s, h] + mean.get(key, 0.0), 0.0)
            a += (before[k][i, s, h] - S.truth[k][i, s, h]) ** 2; b_ += (new[k][i, s, h] - S.truth[k][i, s, h]) ** 2
        gsse[k] = [round(a, 1), round(b_, 1)]
    info = dict(stations=GAME_STATIONS, offsets=[-1, 3], shrink_cells=GAME_SHRINK, min_cells=GAME_MIN, game_days={k: len({(i, s) for i, s, h, key in gc[k]}) for k in folds},
                game_days_test=len({(i, s) for i, s, h, key in gc_t}), sse_of_these_cells_before_after=gsse)
    if judge('games', new, info):
        cur = new; mean = game_means([(key, S.truth[j][i, s, h] - before[j][i, s, h]) for j in folds for i, s, h, key in gc[j]])
        for i, s, h, key in gc_t:
            cur_t[i, s, h] = max(cur_t[i, s, h] + mean.get(key, 0.0), 0.0)
        steps['games']['test_added_log'] = {f'{int(st[key[0]])} end+{key[1]}h third{key[2]}': round(float(v), 3) for key, v in sorted(mean.items()) if abs(v) >= 0.05}

    # ---- 2. school vacation -----------------------------------------------------------------------------------------------
    sv = pd.read_csv(EXT / 'school_vacation_daily.csv'); sv = sv[(sv.area == '전체') & (sv.level == '전체')].set_index('date').share_vacation
    dv = {k: vacation_change(folds[k], sv) for k in folds}; dv_t = vacation_change(test, sv)
    before = {k: cur[k].copy() for k in folds}; new = {}; glob_k = {}
    for k in folds:
        m = keep(k); sl, gl = school_fit([(dv[j][0][m[j]], (S.truth[j] - before[j])[m[j]]) for j in m], ns, p.first > 0); glob_k[k] = gl
        new[k] = np.maximum(before[k] + school_apply(dv[k][0], sl, before[k].shape), 0)
    info = dict(anchor_share={k: round(dv[k][1], 3) for k in folds}, anchor_share_test=round(dv_t[1], 3), hour_groups=dict(am=list(AM), pm=list(PM)),
                global_slopes_nested={k: [round(float(x), 4) for x in glob_k[k]] for k in folds}, ridge_station=RIDGE_STATION)
    if judge('school', new, info):
        cur = new; sl, gl = school_fit([(dv[j][0], S.truth[j] - before[j]) for j in folds], ns, p.first > 0)
        prev = cur_t.copy(); cur_t = np.maximum(cur_t + school_apply(dv_t[0], sl, cur_t.shape), 0)
        order = np.argsort(sl.sum(0)); wd = np.asarray(test.kind) == 'wd'; vac = wd & (dv_t[0] > 0.5)
        steps['school'].update(test_global_slopes=[round(float(x), 4) for x in gl],
                               strongest_stations={f'{int(st[i])} {names[int(st[i])]}': [round(float(sl[0, i]), 3), round(float(sl[1, i]), 3)] for i in order[:6]},
                               test_days_above_half=int(vac.sum()), test_first_last=[str(test.dates[vac][0])[:10], str(test.dates[vac][-1])[:10]] if vac.any() else None,
                               test_mean_log_change_school_hours=round(float((cur_t - prev)[vac][:, :, [0, 1, 2, 9, 10, 11]].mean()), 4) if vac.any() else 0.0,
                               test_mean_log_change_day_total=round(float(np.log(np.expm1(cur_t[vac]).sum(2) / np.expm1(prev[vac]).sum(2)).mean()), 4) if vac.any() else 0.0)

    # ---- 3. very small forecasts ------------------------------------------------------------------------------------------
    before = {k: cur[k].copy() for k in folds}; new = {}; add_k = {}
    for k in folds:
        m = keep(k); add_k[k] = low_fit([(before[j][m[j]], (S.truth[j] - before[j])[m[j]]) for j in m]); new[k] = before[k] + low_apply(before[k], add_k[k])
    info = dict(edges=LOW_EDGES, shrink=LOW_SHRINK, added_log_nested={k: [round(float(x), 3) for x in add_k[k]] for k in folds},
                cells_below_5={k: int((np.expm1(before[k]) < LOW_EDGES[-1]).sum()) for k in folds})
    if judge('low', new, info):
        cur = new; add = low_fit([(before[j], S.truth[j] - before[j]) for j in folds]); n_low = int((np.expm1(cur_t) < LOW_EDGES[-1]).sum())
        cur_t = cur_t + low_apply(cur_t, add); steps['low'].update(test_added_log=[round(float(x), 3) for x in add], test_cells_changed=n_low)

    # ---- results ------------------------------------------------------------------------------------------------------------
    out['steps'] = steps
    out['generations']['v13'] = dict(aligned=S.score(cur), raw_truth=S.score(cur, raw=True))
    a12, a13 = out['generations']['v12']['aligned'], out['generations']['v13']['aligned']
    out['folds_worse_than_v12'] = {k: round(a13[f'fold_{k}'] - a12[f'fold_{k}'], 6) for k in folds if a13[f'fold_{k}'] > a12[f'fold_{k}'] + TOL}
    sse12 = sum(((v12[k] - S.truth[k]) ** 2).sum(axis=(0, 2)) for k in folds); sse13 = sum(((cur[k] - S.truth[k]) ** 2).sum(axis=(0, 2)) for k in folds); n = 4 * 123 * 18
    order = np.argsort(sse13 - sse12)
    out['stations'] = {f'{int(st[i])} {names[int(st[i])]}': dict(v12=round(float(np.sqrt(sse12[i] / n)), 4), v13=round(float(np.sqrt(sse13[i] / n)), 4)) for i in list(order[:12]) + list(order[-4:])}
    np.savez_compressed(OUT / 'oof_v13.npz', **{f'fold{k}': cur[k].astype(np.float32) for k in folds})
    flat = np.empty(cur_t.size); flat[test.row_index] = np.maximum(cur_t, 0).reshape(-1)
    frame = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
    assert len(frame) == 157194 and np.isfinite(frame.boardings).all() and (frame.boardings >= 0).all() and (frame.example_id.to_numpy() == sub12.example_id.to_numpy()).all()
    path = OUT / 'submission_ensemble_v13.csv'; frame.to_csv(path, index=False, float_format='%.8f')
    ref = np.log1p(sub12.boardings.to_numpy())
    out['submission'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(frame)), mean_boardings=float(frame.boardings.mean()),
                             log_rmsd_vs_v12=float(np.sqrt(np.mean((flat - ref) ** 2))), mean_log_diff_vs_v12=float((flat - ref).mean()), rows_changed_vs_v12=int((np.abs(flat - ref) > 1e-9).sum()))
    (OUT / 'gates_v13.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(f"v13: aligned {a12['all_oof']:.5f} -> {a13['all_oof']:.5f} | raw {out['generations']['v12']['raw_truth']['all_oof']:.5f} -> {out['generations']['v13']['raw_truth']['all_oof']:.5f} | worse folds {out['folds_worse_than_v12']}")
    print(json.dumps(out['submission'], indent=1), flush=True)
    return out


if __name__ == '__main__':
    main()
