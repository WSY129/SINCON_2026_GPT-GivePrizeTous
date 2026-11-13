"""Builds submission_ensemble_v17.csv = v15 + four corrections (CPU, seconds; no leaderboard value is used).

    python -m src.v17.run

Same protocol as v16 (nested by fold, date-blocked, kept only if the all-fold score improves and no fold gets worse).
Against v16:
  lunar    v16 added the mean fold residual of station 3122 on the lunar-holiday days.  The folds had forecast those days
           far too low, the final model (trained on both holidays) does not, so the same addition overshot in the test
           period (holiday itself, 09-16h: 6,191 -> 12,619 boardings against 8,285 / 9,968 observed).  Now the forecast is
           blended with a target = ordinary-Sunday level of the forecast + the excess observed on the same day of the
           other lunar holidays (in boardings, on the basis of the target period), weight n / (n + 1).
  night    station 3139, 22h, days without an event: mean residual at half strength (v16: full strength, rejected)
  concert, mart   unchanged
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

OUT = c8.ROOT / 'outputs/ensemble_v17'
BASE = c8.ROOT / os.environ.get('V17_BASE', 'outputs/ensemble_v15')            # folder holding oof_v15.npz, submission_ensemble_v15.csv, base_keep/blocked.npz
TOL = 1e-5
LUNAR0 = ['2025-01-29', '2025-10-06', '2026-02-17', '2026-09-25']
LUNAR_STATION, LUNAR_HOURS, LUNAR_OFFSETS = 3122, slice(3, 13), (-3, -2, -1, 0, 1, 2, 3)      # 09-18h
CONCERT_HOURS, CONCERT_PRIOR_DAYS = (14, 15, 16, 17), 2.0
MART_HOURS, MART_RIDGE, MART_STATIONS = slice(5, 13), 30.0, [3131, 3206]
NIGHT_KAPPA, NIGHT_HOUR, NIGHT_STRENGTH = 20.0, 22, 0.5
ORDER = ['concert', 'lunar', 'mart', 'night']


def main():
    S = s11.Stack(); folds = S.folds; test = S.test; p = S.panel; st = np.asarray(p.stations).astype(int); ns = len(st); sidx = {int(s): i for i, s in enumerate(st)}
    base = {k: np.load(BASE / 'oof_v15.npz')[f'fold{k}'].astype(float) for k in folds}
    sub15 = pd.read_csv(BASE / 'submission_ensemble_v15.csv'); base_t = np.log1p(sub15.boardings.to_numpy())[test.row_index].reshape(test.pred.shape)
    bl = np.load(BASE / 'base_keep/blocked.npz'); blocked = {**{k: bl[f'fold{k}'] for k in folds}, 'test': bl['test']}; blocks = {**folds, 'test': test}
    shape = {n: (len(b.dates), ns, 18) for n, b in blocks.items()}; basis2026 = {n: n in ('test', 4) for n in blocks}
    keep = lambda k: {j: np.asarray((folds[j].dates < folds[k].dates.min()) | (folds[j].dates > folds[k].dates.max())) for j in folds if j != k}
    days = pd.DatetimeIndex(p.days); kinds = np.asarray(c8.day_kinds(days, c8.FINAL))
    Zal = p.Z + np.where((np.arange(p.Z.shape[1]) >= p.day_index('2026-01-01'))[None, :, None], S.logf[:, None, :], 0.0)      # training ridership on the 2025 basis

    def group_mean(cur, cur_t, keys, kappa, strength=1.0, min_n=4):
        nk = max(int(v.max()) for v in keys.values()) + 1

        def acc(parts):
            s = np.zeros(nk); n = np.zeros(nk)
            for key, r in parts:
                m = key >= 0; s += np.bincount(key[m], weights=r[m], minlength=nk); n += np.bincount(key[m], minlength=nk)
            return np.where(n >= min_n, strength * s / (n + kappa), 0.0)

        def apply(z, key, d):
            out = z.copy(); m = key >= 0; out[m] = np.maximum(z[m] + d[key[m]], 0); return out
        new = {}
        for k in folds:
            m = keep(k); new[k] = apply(cur[k], keys[k], acc([(keys[j][m[j]], (S.truth[j] - cur[j])[m[j]]) for j in m]))
        d = acc([(keys[j], S.truth[j] - cur[j]) for j in folds])
        return new, apply(cur_t, keys['test'], d), d

    # ---- Munhak main-stadium concerts (as v16) -------------------------------------------------------------------------------
    ev = load_events(st); concert_days = sorted(d for (s, d), e in ev.items() if s == 3127 and 'concert' in e['kinds']); m_j = sidx[3127]

    def step_concert(cur, cur_t):
        hist = {d: Zal[m_j, days.get_loc(pd.Timestamp(d))] for d in concert_days if pd.Timestamp(d) in days and np.isfinite(Zal[m_j, days.get_loc(pd.Timestamp(d))]).all()}

        def run(name, z):
            b = blocks[name]; out = z.copy(); lo, hi = np.datetime64(b.dates.min()), np.datetime64(b.dates.max()); used = {}
            train = [v for d, v in hist.items() if not (lo <= np.datetime64(d) <= hi)]
            for i, d in enumerate(pd.DatetimeIndex(b.dates)):
                if str(d)[:10] in concert_days and len(train) >= 2:
                    w = len(train) / (len(train) + CONCERT_PRIOR_DAYS); est = np.mean(train, 0) - (S.logf[m_j] if basis2026[name] else 0.0)
                    for h in CONCERT_HOURS:
                        out[i, m_j, h] = w * est[h] + (1 - w) * z[i, m_j, h]
                    used[str(d)[:10]] = dict(days_used=len(train), before=[int(round(np.expm1(z[i, m_j, h]))) for h in CONCERT_HOURS], after=[int(round(np.expm1(out[i, m_j, h]))) for h in CONCERT_HOURS])
            return out, used
        res = {k: run(k, cur[k]) for k in folds}; t, used_t = run('test', cur_t)
        return {k: v[0] for k, v in res.items()}, t, dict(hours=[h + 6 for h in CONCERT_HOURS], folds={k: v[1] for k, v in res.items()}, test=used_t)

    # ---- grave-visit days at station 3122: blend towards "ordinary Sunday + excess of the other lunar holidays" ---------------
    l_j = sidx[LUNAR_STATION]; lun_idx = {d: days.get_loc(pd.Timestamp(d)) for d in LUNAR0 if pd.Timestamp(d) in days}; Bal = np.expm1(Zal[l_j])

    def hist_excess(d0, off):
        """Excess over the ordinary-Sunday level [18] on day d0 + off (2025 basis), and whether that day was a working day."""
        i0 = lun_idx[d0]; i = i0 + off
        sun = [k for k in range(max(i0 - 42, 0), min(i0 + 43, len(days))) if kinds[k] == 'sun' and abs(k - i0) > 10 and np.isfinite(Bal[k]).all()]
        if not (0 <= i < len(days)) or len(sun) < 2 or not np.isfinite(Bal[i]).all():
            return None
        return Bal[i] - np.median(Bal[sun], 0), kinds[i] == 'wd'

    def step_lunar(cur, cur_t):
        def run(name, z):
            b = blocks[name]; out = z.copy(); d = pd.DatetimeIndex(b.dates); kind = np.asarray(b.kind); lo, hi = d.min(), d.max(); log = {}
            fb = np.exp(-S.logf[l_j]) if basis2026[name] else 1.0
            for d0 in LUNAR0:
                t0 = pd.Timestamp(d0)
                if not (lo <= t0 <= hi):
                    continue
                i0 = int(np.flatnonzero(d == t0)[0]); sun = [k for k in range(len(d)) if kind[k] == 'sun' and 10 < abs(k - i0) <= 42] or [k for k in range(len(d)) if kind[k] == 'sun']
                usual = np.median(np.expm1(z[sun, l_j]), 0)
                for off in LUNAR_OFFSETS:
                    i = i0 + off
                    if not (0 <= i < len(d)) or (abs(off) >= 2 and kind[i] == 'wd'):
                        continue
                    an = [hist_excess(x, off) for x in lun_idx if x != d0 and not (lo <= pd.Timestamp(x) + pd.Timedelta(days=off) <= hi)]
                    an = [a[0] for a in an if a is not None and not (abs(off) >= 2 and a[1])]
                    if not an:
                        continue
                    w = len(an) / (len(an) + 1.0); target = np.log1p(np.maximum(usual + np.mean(an, 0) * fb, 1.0))
                    out[i, l_j, LUNAR_HOURS] = w * target[LUNAR_HOURS] + (1 - w) * z[i, l_j, LUNAR_HOURS]
                    log[str(d[i])[:10]] = dict(offset=off, kind=str(kind[i]), analog_days=len(an), weight=round(w, 2), hours='09-18', before=[int(round(v)) for v in np.expm1(z[i, l_j, LUNAR_HOURS])],
                                               target=[int(round(v)) for v in np.expm1(target[LUNAR_HOURS])], after=[int(round(v)) for v in np.expm1(out[i, l_j, LUNAR_HOURS])])
            return out, log
        res = {k: run(k, cur[k]) for k in folds}; t, log_t = run('test', cur_t)
        return {k: v[0] for k, v in res.items()}, t, dict(station=LUNAR_STATION, folds={k: {d: dict(before_sum=sum(v['before']), after_sum=sum(v['after']), analog_days=v['analog_days']) for d, v in r[1].items()} for k, r in res.items()}, test=log_t)

    # ---- large-store closure Sundays (as v16) --------------------------------------------------------------------------------
    cal = pd.read_csv(OUT / 'external/mart_closure_calendar.csv', encoding='utf-8-sig').set_index('date').is_mart_closed_sunday

    def mart_x(n):
        b = blocks[n]; sun = np.asarray(b.kind) == 'sun'; c = np.array([float(cal.get(str(d)[:10], 0)) for d in b.dates]); x = np.zeros(len(b.dates)); x[sun] = c[sun] - c[sun].mean()
        return x

    def step_mart(cur, cur_t):
        X = {n: mart_x(n) for n in blocks}; width = MART_HOURS.stop - MART_HOURS.start

        def fit(parts):
            num = sum((x[:, None, None] * r[:, :, MART_HOURS]).sum((0, 2)) for x, r in parts); den = sum((x ** 2).sum() * width for x, r in parts)
            g = num.sum() / max(den * ns, 1e-9); sl = g + (num - g * den) / (den + MART_RIDGE); return np.where(np.isin(st, MART_STATIONS), sl, 0.0)

        def apply(z, x, sl):
            out = z.copy(); out[:, :, MART_HOURS] = np.maximum(z[:, :, MART_HOURS] + x[:, None, None] * sl[None, :, None], 0); return out
        new = {}
        for k in folds:
            m = keep(k); new[k] = apply(cur[k], X[k], fit([(X[j][m[j]], (S.truth[j] - cur[j])[m[j]]) for j in m]))
        sl = fit([(X[j], S.truth[j] - cur[j]) for j in folds])
        return new, apply(cur_t, X['test'], sl), dict(hours='11-18', closed_minus_open_log_effect_test={str(s): round(float(sl[sidx[s]]), 3) for s in MART_STATIONS})

    def step_night(cur, cur_t):
        j = sidx[3139]; keys = {}
        for n, b in blocks.items():
            g = R.day_group(b.kind); ok = (g < 3) & ~blocked[n][:, j] & np.array([(3139, str(x)[:10]) not in ev for x in b.dates]); key = np.full(shape[n], -1); key[ok, j, NIGHT_HOUR - 6] = g[ok]; keys[n] = key
        new, t, d = group_mean(cur, cur_t, keys, NIGHT_KAPPA, NIGHT_STRENGTH)
        return new, t, dict(station=3139, hour=NIGHT_HOUR, strength=NIGHT_STRENGTH, test_added_log=dict(zip(['weekday', 'saturday', 'sunday'], [round(float(v), 3) for v in d[:3]])))

    STEPS = dict(concert=step_concert, lunar=step_lunar, mart=step_mart, night=step_night)
    verdict = lambda a, b: bool(b['all_oof'] < a['all_oof'] - 1e-7 and all(b[f'fold_{k}'] <= a[f'fold_{k}'] + TOL for k in folds))
    cur = {k: v.copy() for k, v in base.items()}; cur_t = base_t.copy(); steps = {}; s15 = S.score(base)
    for n in ORDER:
        new, t, info = STEPS[n](cur, cur_t); a, b = S.score(cur), S.score(new); ok = verdict(a, b)
        steps[n] = dict(before=a, after=b, adopted=ok, fold_changes={k: round(b[f'fold_{k}'] - a[f'fold_{k}'], 7) for k in folds}, **info)
        print(f"{n:8s} {a['all_oof']:.5f} -> {b['all_oof']:.5f}  " + ' '.join(f"F{k} {b[f'fold_{k}'] - a[f'fold_{k}']:+.6f}" for k in folds) + ('  ADOPTED' if ok else '  rejected'), flush=True)
        if ok:
            steps[n]['test_cells_changed'] = int((np.abs(t - cur_t) > 1e-9).sum()); cur, cur_t = new, t
    load = lambda v: {k: np.load(c8.ROOT / f'outputs/ensemble_{v}/oof_{v}.npz')[f'fold{k}'].astype(float) for k in folds}; v15, v16 = load('v15'), load('v16')
    out = dict(information='local OOF only; every step is nested and date-blocked on aligned truth; no leaderboard value is used',
               protocol='for fold k a correction is fitted on the other folds without the days that fall inside fold k; kept only if the all-fold score improves and no fold gets worse',
               base=str(BASE.relative_to(c8.ROOT)), steps=steps,
               generations={n: dict(aligned=S.score(z), raw_truth=S.score(z, raw=True)) for n, z in (('v15', v15), ('v16', v16), ('v17 base', base), ('v17', cur))})
    a15, a17 = out['generations']['v15']['aligned'], out['generations']['v17']['aligned']
    out['folds_worse_than_v15'] = {k: round(a17[f'fold_{k}'] - a15[f'fold_{k}'], 6) for k in folds if a17[f'fold_{k}'] > a15[f'fold_{k}'] + TOL}
    np.savez_compressed(OUT / 'oof_v17.npz', **{f'fold{k}': cur[k].astype(np.float32) for k in folds})
    flat = np.empty(cur_t.size); flat[test.row_index] = np.maximum(cur_t, 0).reshape(-1); sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
    assert len(sub) == 157194 and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all() and (sub.example_id.to_numpy() == sub15.example_id.to_numpy()).all()
    path = OUT / 'submission_ensemble_v17.csv'; sub.to_csv(path, index=False, float_format='%.8f')
    ref = np.log1p(pd.read_csv(c8.ROOT / 'outputs/ensemble_v15/submission_ensemble_v15.csv').boardings.to_numpy())
    out['submission'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(sub)), mean_boardings=float(sub.boardings.mean()),
                             log_rmsd_vs_v15=float(np.sqrt(np.mean((flat - ref) ** 2))), rows_changed_vs_v15=int((np.abs(flat - ref) > 1e-9).sum()))
    (OUT / 'gates_v17.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    g = out['generations']; print('v15', round(a15['all_oof'], 5), '| v16', round(g['v16']['aligned']['all_oof'], 5), '| v17', round(a17['all_oof'], 5), '| raw', round(g['v17']['raw_truth']['all_oof'], 5), '| worse folds vs v15', out['folds_worse_than_v15'])
    return out


if __name__ == '__main__':
    main()
