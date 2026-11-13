"""Builds submission_ensemble_v16.csv = v15 + five tested corrections (CPU, seconds; no leaderboard value is used).

    python -m src.v16.run

Each step is fitted nested by fold and date-blocked (for fold k: other folds without the days inside fold k) on the
residuals of the current forecast, and kept only if the all-fold score improves and no fold gets worse.
  concert   Munhak main-stadium concert days: hours 20-23 blended with the mean log profile of the other concert days
  lunar     Lunar New Year / Chuseok period: station x day class x hour group mean residual (classes: the day itself,
            the adjacent days, the weekend days within three days of it); applied where the mean is at least GATE
  growth    stations opened inside the data window: weekday correction = c x (8-week slope of an hour group minus the
            all-day slope) x (28 + horizon)
  mart      Sundays on which large stores are closed (2nd / 4th): slope on the centred closure flag, 11-18h
  night     station 3139 on days without an event: mean residual at 20 / 21 / 22h by type of day
"""
import hashlib
import json
import os
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v11 import stack as s11
from src.v13 import run as R
from src.v15.events import load_events

OUT = c8.ROOT / 'outputs/ensemble_v16'
V15 = c8.ROOT / 'outputs/ensemble_v15'
TOL = 1e-5
HG = np.repeat([0, 1, 2, 3], [4, 6, 4, 4])                                  # hour index -> 06-09, 10-15, 16-19, 20-23
NEW = [3107, 3108, 3109]
LUNAR0 = np.array(['2025-10-06', '2026-02-17', '2026-09-25'], dtype='datetime64[D]')
LUNAR_KAPPA, LUNAR_GATE = 6.0, float(os.environ.get('V16_LUNAR_GATE', '0.15'))
LUNAR_STATIONS = [int(x) for x in os.environ.get('V16_LUNAR_STATIONS', '3122').split(',') if x]      # empty = every station
LUNAR_CLASSES = [int(x) for x in os.environ.get('V16_LUNAR_CLASSES', '0,1,2').split(',')]
GROWTH_MODE = os.environ.get('V16_GROWTH', 'slope')                          # slope | mean (station x hour-group mean residual on weekdays)
GROWTH_STATIONS = [int(x) for x in os.environ.get('V16_GROWTH_STATIONS', '3107,3108,3109').split(',')]
GROWTH_GROUPS = [int(x) for x in os.environ.get('V16_GROWTH_GROUPS', '0,1,2,3').split(',')]
MART_GATE = float(os.environ.get('V16_MART_GATE', '0.05'))
MART_STATIONS = [int(x) for x in os.environ.get('V16_MART_STATIONS', '3131,3206').split(',') if x]      # the two stations named with the hypothesis; empty = every station
NIGHT_HOURS = [int(x) for x in os.environ.get('V16_NIGHT_HOURS', '22').split(',')]
CONCERT_HOURS, CONCERT_PRIOR_DAYS = (14, 15, 16, 17), 2.0
MART_HOURS, MART_RIDGE = slice(5, 13), 30.0
NIGHT_KAPPA = 20.0
ORDER = ['concert', 'lunar', 'growth', 'mart', 'night']


def main():
    S = s11.Stack(); folds = S.folds; test = S.test; p = S.panel; st = np.asarray(p.stations).astype(int); ns = len(st); sidx = {int(s): i for i, s in enumerate(st)}
    names = pd.read_csv(c8.ROOT / 'data/raw/metaData.csv').set_index('station_id').station_name
    base = {k: np.load(V15 / 'oof_v15.npz')[f'fold{k}'].astype(float) for k in folds}
    sub15 = pd.read_csv(V15 / 'submission_ensemble_v15.csv'); base_t = np.log1p(sub15.boardings.to_numpy())[test.row_index].reshape(test.pred.shape)
    bl = np.load(V15 / 'base_keep/blocked.npz'); blocked = {**{k: bl[f'fold{k}'] for k in folds}, 'test': bl['test']}; blocks = {**folds, 'test': test}
    shape = {n: (len(b.dates), ns, 18) for n, b in blocks.items()}; origin = {n: (b.origin if n != 'test' else c8.FINAL) for n, b in blocks.items()}
    keep = lambda k: {j: np.asarray((folds[j].dates < folds[k].dates.min()) | (folds[j].dates > folds[k].dates.max())) for j in folds if j != k}
    days = pd.DatetimeIndex(p.days); Zal = p.Z + np.where((np.arange(p.Z.shape[1]) >= p.day_index('2026-01-01'))[None, :, None], S.logf[:, None, :], 0.0)      # training ridership on the 2025 basis

    # ------------------------------------------------------------------------------------------------ generic group mean
    def group_mean(cur, cur_t, keys, kappa, min_n=4, gate=0.0):
        nk = max(int(v.max()) for v in keys.values()) + 1

        def acc(parts):
            s = np.zeros(nk); n = np.zeros(nk)
            for key, r in parts:
                m = key >= 0; s += np.bincount(key[m], weights=r[m], minlength=nk); n += np.bincount(key[m], minlength=nk)
            raw = np.where(n >= min_n, s / np.maximum(n, 1), 0.0)
            return np.where(np.abs(raw) >= gate, raw * n / (n + kappa), 0.0)

        def apply(z, key, d):
            out = z.copy(); m = key >= 0; out[m] = np.maximum(z[m] + d[key[m]], 0); return out
        new = {}
        for k in folds:
            m = keep(k); new[k] = apply(cur[k], keys[k], acc([(keys[j][m[j]], (S.truth[j] - cur[j])[m[j]]) for j in m]))
        d = acc([(keys[j], S.truth[j] - cur[j]) for j in folds])
        return new, apply(cur_t, keys['test'], d), d

    # ------------------------------------------------------------------------------------------------ steps
    ev = load_events(st); concert_days = sorted(d for (s, d), e in ev.items() if s == 3127 and 'concert' in e['kinds']); m_j = sidx[3127]

    def step_concert(cur, cur_t):
        hist = {d: Zal[m_j, days.get_loc(pd.Timestamp(d))] for d in concert_days if pd.Timestamp(d) in days and np.isfinite(Zal[m_j, days.get_loc(pd.Timestamp(d))]).all()}

        def run(name, z):
            b = blocks[name]; out = z.copy(); lo, hi = np.datetime64(b.dates.min()), np.datetime64(b.dates.max()); used = {}
            train = [v for d, v in hist.items() if not (lo <= np.datetime64(d) <= hi)]
            basis2026 = name in ('test', 4)
            for i, d in enumerate(pd.DatetimeIndex(b.dates)):
                if str(d)[:10] in concert_days and len(train) >= 2:
                    w = len(train) / (len(train) + CONCERT_PRIOR_DAYS); est = np.mean(train, 0) - (S.logf[m_j] if basis2026 else 0.0)
                    for h in CONCERT_HOURS:
                        out[i, m_j, h] = w * est[h] + (1 - w) * z[i, m_j, h]
                    used[str(d)[:10]] = dict(days_used=len(train), before=[int(round(np.expm1(z[i, m_j, h]))) for h in CONCERT_HOURS], after=[int(round(np.expm1(out[i, m_j, h]))) for h in CONCERT_HOURS])
            return out, used
        res = {k: run(k, cur[k]) for k in folds}; t, used_t = run('test', cur_t)
        return {k: v[0] for k, v in res.items()}, t, dict(hours=[h + 6 for h in CONCERT_HOURS], concert_days_in_training_period=[d for d in concert_days if d in hist], folds={k: v[1] for k, v in res.items()}, test=used_t)

    def lunar_keys():
        out = {}
        for n, b in blocks.items():
            kind = np.asarray(b.kind); d = pd.DatetimeIndex(b.dates).values.astype('datetime64[D]'); near = np.abs((d[:, None] - LUNAR0[None, :]).astype(int)).min(1) <= 3
            cls = np.where(kind == 'lunar0', 0, np.where(kind == 'lunar_adj', 1, np.where(np.isin(kind, ['long_sat', 'long_sun', 'hol_sat']) & near, 2, -1)))
            key = (np.arange(ns)[None, :, None] * 3 + cls[:, None, None]) * 4 + HG[None, None, :]
            okc = np.isin(cls, LUNAR_CLASSES)[:, None, None] & (np.isin(st, LUNAR_STATIONS)[None, :, None] if LUNAR_STATIONS else True)
            out[n] = np.where(np.broadcast_to(okc, shape[n]), key, -1)
        return out

    def step_lunar(cur, cur_t):
        keys = lunar_keys(); new, t, d = group_mean(cur, cur_t, keys, LUNAR_KAPPA, gate=LUNAR_GATE)
        nz = np.flatnonzero(d); top = sorted(nz, key=lambda i: -abs(d[i]))[:14]; CL = ['명절 당일', '명절 전후일', '명절 옆 주말']; GR = ['06-09', '10-15', '16-19', '20-23']
        dt = pd.DatetimeIndex(test.dates); days_t = [f"{str(dt[i])[:10]} {test.kind[i]}" for i in range(len(dt)) if (keys['test'][i] >= 0).any()]
        return new, t, dict(kappa_cells=LUNAR_KAPPA, gate=LUNAR_GATE, keys_applied_test=int(len(nz)), test_days=days_t,
                            largest_test={f"{int(st[i // 12])} {names[int(st[i // 12])]} {CL[(i // 4) % 3]} {GR[i % 4]}": round(float(d[i]), 3) for i in top},
                            test_mean_log_change_on_those_days=round(float((t - cur_t)[(keys['test'] >= 0).any((1, 2))].mean()), 4))

    def growth_feature(n):
        b = blocks[n]; O = p.day_index(origin[n]); idx = np.arange(O - 55, O + 1); wd = days[idx].dayofweek < 5; x = np.zeros(shape[n]); info = {}
        tgt = np.flatnonzero(np.asarray(b.kind) == 'wd'); hz = (tgt + 1 + 28).astype(float)
        for sid in NEW:
            j = sidx[sid]; B = np.expm1(Zal[j, idx][wd]); ok = np.isfinite(B).all(1); t = idx[wd][ok].astype(float); B = B[ok]
            if len(t) < 15:
                continue
            sl = lambda v: float(np.polyfit(t, np.log(v + 1.0), 1)[0]); all_ = sl(B.sum(1)); info[sid] = []
            for g in range(4):
                ds = sl(B[:, HG == g].sum(1)) - all_; info[sid].append(round(ds * 100, 3))
                x[np.ix_(tgt, [j], np.flatnonzero(HG == g))] = (ds * hz)[:, None, None]
        return x, info

    def step_growth_mean(cur, cur_t):
        keys = {}
        for n, b in blocks.items():
            key = np.full(shape[n], -1); wd = np.flatnonzero(np.asarray(b.kind) == 'wd')
            for a, sid in enumerate(GROWTH_STATIONS):
                for g in GROWTH_GROUPS:
                    key[np.ix_(wd, [sidx[sid]], np.flatnonzero(HG == g))] = a * 4 + g
            keys[n] = key
        new, t, d = group_mean(cur, cur_t, keys, 40.0)
        return new, t, dict(mode='station x hour-group mean residual on weekdays', stations=GROWTH_STATIONS, hour_groups=GROWTH_GROUPS, test_added_log=[round(float(v), 3) for v in d])

    def step_growth(cur, cur_t):
        X = {n: growth_feature(n) for n in blocks}; coef = {}
        fit = lambda parts: float(np.clip(sum((x * r).sum() for x, r in parts) / max(sum((x ** 2).sum() for x, r in parts), 1e-9), 0, 1.5))
        new = {}
        for k in folds:
            m = keep(k); coef[k] = fit([(X[j][0][m[j]], (S.truth[j] - cur[j])[m[j]]) for j in m]); new[k] = np.maximum(cur[k] + coef[k] * X[k][0], 0)
        c = fit([(X[j][0], S.truth[j] - cur[j]) for j in folds]); t = np.maximum(cur_t + c * X['test'][0], 0); wd = np.asarray(test.kind) == 'wd'
        return new, t, dict(coefficient_nested={k: round(v, 3) for k, v in coef.items()}, coefficient_test=round(c, 3), slope_minus_all_day_percent_per_day={str(n): X[n][1] for n in blocks},
                            test_mean_log_change_weekdays={str(s): [round(float((t - cur_t)[wd][:, sidx[s]][:, HG == g].mean()), 3) for g in range(4)] for s in NEW})

    cal = pd.read_csv(OUT / 'external/mart_closure_calendar.csv', encoding='utf-8-sig').set_index('date').is_mart_closed_sunday

    def mart_x(n):
        b = blocks[n]; sun = np.asarray(b.kind) == 'sun'; c = np.array([float(cal.get(str(d)[:10], 0)) for d in b.dates]); x = np.zeros(len(b.dates)); x[sun] = c[sun] - c[sun].mean()
        return x

    def step_mart(cur, cur_t):
        X = {n: mart_x(n) for n in blocks}

        def fit(parts):
            num = sum((x[:, None, None] * r[:, :, MART_HOURS]).sum((0, 2)) for x, r in parts); den = sum((x ** 2).sum() * (MART_HOURS.stop - MART_HOURS.start) for x, r in parts)
            g = num.sum() / max(den * ns, 1e-9); sl = g + (num - g * den) / (den + MART_RIDGE); return np.where((np.abs(sl) >= MART_GATE) & (np.isin(st, MART_STATIONS) if MART_STATIONS else True), sl, 0.0)

        def apply(z, x, sl):
            out = z.copy(); out[:, :, MART_HOURS] = np.maximum(z[:, :, MART_HOURS] + x[:, None, None] * sl[None, :, None], 0); return out
        new = {}
        for k in folds:
            m = keep(k); new[k] = apply(cur[k], X[k], fit([(X[j][m[j]], (S.truth[j] - cur[j])[m[j]]) for j in m]))
        sl = fit([(X[j], S.truth[j] - cur[j]) for j in folds]); order = np.argsort(sl)
        return new, apply(cur_t, X['test'], sl), dict(hours='11-18', ridge=MART_RIDGE, stations=MART_STATIONS, closed_minus_open_log_effect_test={f'{int(st[i])} {names[int(st[i])]}': round(float(sl[i]), 3) for i in list(order[:6]) + list(order[-3:])},
                                                     dongchun=round(float(sl[sidx[3131]]), 3), dokjeong=round(float(sl[sidx[3206]]), 3))

    def step_night(cur, cur_t):
        j = sidx[3139]; keys = {}
        for n, b in blocks.items():
            g = R.day_group(b.kind); ok = (g < 3) & ~blocked[n][:, j]; key = np.full(shape[n], -1)
            for h in NIGHT_HOURS:
                key[ok, j, h - 6] = (h - 20) * 3 + g[ok]
            keys[n] = key
        new, t, d = group_mean(cur, cur_t, keys, NIGHT_KAPPA)
        return new, t, dict(station=3139, kappa_cells=NIGHT_KAPPA, hours=NIGHT_HOURS, test_added_log={f'{h}h': dict(zip(['weekday', 'saturday', 'sunday'], [round(float(v), 3) for v in d[(h - 20) * 3:(h - 20) * 3 + 3]])) for h in NIGHT_HOURS})

    STEPS = dict(concert=step_concert, lunar=step_lunar, growth=step_growth if GROWTH_MODE == 'slope' else step_growth_mean, mart=step_mart, night=step_night)

    # ------------------------------------------------------------------------------------------------ judge and chain
    def verdict(a, b):
        return bool(b['all_oof'] < a['all_oof'] - 1e-7 and all(b[f'fold_{k}'] <= a[f'fold_{k}'] + TOL for k in folds))
    s15 = S.score(base); alone = {}
    for n in ORDER:
        new, _, _ = STEPS[n](base, base_t); b = S.score(new); alone[n] = dict(after=b, would_pass=verdict(s15, b))
        print(f"alone  {n:8s} {s15['all_oof']:.5f} -> {b['all_oof']:.5f}  " + ' '.join(f"F{k} {b[f'fold_{k}'] - s15[f'fold_{k}']:+.5f}" for k in folds), flush=True)
    cur = {k: v.copy() for k, v in base.items()}; cur_t = base_t.copy(); steps = {}
    for n in ORDER:
        new, t, info = STEPS[n](cur, cur_t); a, b = S.score(cur), S.score(new); ok = verdict(a, b)
        steps[n] = dict(before=a, after=b, adopted=ok, **info)
        print(f"chain  {n:8s} {a['all_oof']:.5f} -> {b['all_oof']:.5f}  " + ' '.join(f"F{k} {b[f'fold_{k}'] - a[f'fold_{k}']:+.5f}" for k in folds) + ('  ADOPTED' if ok else '  rejected'), flush=True)
        if ok:
            steps[n]['test_log_rms_change'] = round(float(np.sqrt(((t - cur_t) ** 2).mean())), 5); steps[n]['test_cells_changed'] = int((np.abs(t - cur_t) > 1e-9).sum()); cur, cur_t = new, t
    out = dict(information='local OOF only; every step is nested and date-blocked on aligned truth; no leaderboard value is used',
               protocol='for fold k a correction is fitted on the other folds without the days that fall inside fold k; kept only if the all-fold score improves and no fold gets worse',
               generations={'v15': dict(aligned=s15, raw_truth=S.score(base, raw=True)), 'v16': dict(aligned=S.score(cur), raw_truth=S.score(cur, raw=True))}, alone_on_v15=alone, steps=steps)
    a15, a16 = out['generations']['v15']['aligned'], out['generations']['v16']['aligned']
    out['folds_worse_than_v15'] = {k: round(a16[f'fold_{k}'] - a15[f'fold_{k}'], 6) for k in folds if a16[f'fold_{k}'] > a15[f'fold_{k}'] + TOL}
    np.savez_compressed(OUT / 'oof_v16.npz', **{f'fold{k}': cur[k].astype(np.float32) for k in folds})
    flat = np.empty(cur_t.size); flat[test.row_index] = np.maximum(cur_t, 0).reshape(-1); sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
    assert len(sub) == 157194 and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all() and (sub.example_id.to_numpy() == sub15.example_id.to_numpy()).all()
    path = OUT / 'submission_ensemble_v16.csv'; sub.to_csv(path, index=False, float_format='%.8f'); ref = np.log1p(sub15.boardings.to_numpy())
    out['submission'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(sub)), mean_boardings=float(sub.boardings.mean()),
                             log_rmsd_vs_v15=float(np.sqrt(np.mean((flat - ref) ** 2))), rows_changed_vs_v15=int((np.abs(flat - ref) > 1e-9).sum()))
    (OUT / 'gates_v16.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(f"v16: aligned v15 {a15['all_oof']:.5f} -> v16 {a16['all_oof']:.5f} | raw {out['generations']['v15']['raw_truth']['all_oof']:.5f} -> {out['generations']['v16']['raw_truth']['all_oof']:.5f} | worse folds {out['folds_worse_than_v15']}")
    return out


if __name__ == '__main__':
    main()
