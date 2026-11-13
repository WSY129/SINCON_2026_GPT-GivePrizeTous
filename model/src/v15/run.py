"""Builds submission_ensemble_v15.csv (CPU only; no leaderboard value is used).

    python -m src.v15.train ...      (once: LightGBM with three seeds and XGBoost on the v15 panel)
    python -m src.v15.run [--reuse]  (--reuse: take the two pipeline runs from the previous call)

1. members  lgb_h (three seeds, v15 panel: v14 inputs, corrected event labels, three bad station-days out of the targets),
            XGBoost of depth 10 (xgb_i, two seeds) and depth 8 (xgb_h), same panel, and lgb_d (v14) join the v12 stacking
            pipeline, which is otherwise unchanged
2. event-day rule  v12-v14 kept the v10 value on every listed event station-day.  v10 is worth keeping only where it
            carries a researched lift, so the pipeline is run twice (with and without that rule) and v10 is kept only on
            the event station-days whose v10 forecast is at least LIFT (log) above that station's typical day at some hour.
            The same rule is applied to the folds and to the test period.
3. day-type weights  the pipeline uses one set of member weights for all days; `mix` moves its output towards / away from
            single members with one coefficient per member and type of day (ridge-shrunk to "no change")
4. steps    events / school / low / holiday corrections as in v14, fitted nested and date-blocked on the new base
Every step from 3 on is kept only if the all-fold score improves and no fold gets worse.
"""
import hashlib
import json
import os
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v11 import stack as s11
from src.v12 import run as run12, meta
from src.v13 import run as R
from src.v14.priors import CalendarPrior
from src.v14.run import holiday_shift, TYPICAL
from . import mix
from .events import load_events

OUT = c8.ROOT / 'outputs/ensemble_v15'
DIRS = [c8.ROOT / 'outputs/ensemble_v14/models', OUT / 'models']
LIFT = 0.5
TOL = 1e-5
MEMBERS = os.environ.get('V15_MEMBERS', 'lgb_h,xgb_i,xgb_h,lgb_d').split(',')                                      # lgb_c of v14 is superseded by lgb_h (same setting, corrected panel, three seeds)
MIX_OLD = ['lgb_b', 'xgb', 'cb', 'low_tw']                                 # v11 members offered to the day-type weights next to v10 and the retrained ones
MIX_RIDGE = float(os.environ.get('V15_MIX_RIDGE', '0.002'))
MIX_HOURS = os.environ.get('V15_MIX_HOURS', 'all')


# ------------------------------------------------------------------------------------------------ retrained members
def files(name, cut):
    return sorted(p for d in DIRS for p in d.glob(f'{cut}__{name}__*.npz'))


def new_names():
    return [n for n in MEMBERS if all(files(n, f) for f in list(s11.FOLDS.values()) + ['final'])]


def curves(name):
    return {k: np.mean([json.loads(p.with_suffix('.json').read_text())['valid_rmsle'] for p in files(name, f)], 0) for k, f in s11.FOLDS.items()}


def strict_index(name, k=None):
    c = curves(name); use = [3, 4] if k is None else [j for j in c if j != k]
    return int(round(np.mean([int(np.argmin(c[j])) for j in use])))


def _mean(paths, i):
    out = []
    for p in paths:
        d = np.load(p); out.append(np.maximum(d['anchor'] + meta.smooth(d['pred'], min(i, d['pred'].shape[0] - 1)), 0))
    return np.mean(out, 0).astype(float)


def load_fold(name, k):
    return _mean(files(name, s11.FOLDS[k]), strict_index(name, k))


def load_final(name):
    return _mean(files(name, 'final'), strict_index(name))


def run_base(S, new, sub, keep_events, reuse=False):
    """The v12 pipeline with the retrained members; its output folder is redirected, nothing of v12 is written."""
    saved = (s11.Stack, meta.load_fold, meta.load_final, run12.OUT, run12.block_events); (OUT / sub).mkdir(parents=True, exist_ok=True); tag = OUT / sub / 'members.json'
    state = dict(members=new, files=sorted(p.name for n in new for c in list(s11.FOLDS.values()) + ['final'] for p in files(n, c)))
    if not (reuse and tag.exists() and json.loads(tag.read_text()) == state):
        try:
            s11.Stack = lambda: S
            meta.load_fold = lambda n, k, smoothed=True: load_fold(n, k) if n in new else saved[1](n, k, smoothed)
            meta.load_final = lambda n, smoothed=True: load_final(n) if n in new else saved[2](n, smoothed)
            run12.OUT = OUT / sub; assert run12.OUT != saved[3]
            if not keep_events:
                run12.block_events = lambda b, schedule, extra=None: None
            run12.main()
        finally:
            s11.Stack, meta.load_fold, meta.load_final, run12.OUT, run12.block_events = saved
        np.savez_compressed(OUT / sub / 'blocked.npz', **{f'fold{k}': S.folds[k].blocked for k in S.folds}, test=S.test.blocked); tag.write_text(json.dumps(state))
    g = json.loads((OUT / sub / 'gates_v12.json').read_text()); bl = np.load(OUT / sub / 'blocked.npz')
    oof = np.load(OUT / sub / 'oof_v12.npz'); frame = pd.read_csv(OUT / sub / 'submission_ensemble_v12.csv')
    blocked = {**{k: bl[f'fold{k}'] for k in S.folds}, 'test': bl['test']}
    return g, {k: oof[f'fold{k}'].astype(float) for k in S.folds}, np.log1p(frame.boardings.to_numpy())[S.test.row_index].reshape(S.test.pred.shape), blocked, frame


def lifted_days(block, v10, blocked):
    """[day, S]: listed event station-days on which the v10 forecast is at least LIFT above the station's typical day at some hour."""
    grp = R.day_group(block.kind); out = np.zeros(blocked.shape, bool)
    for s in np.flatnonzero(blocked.any(0)):
        for g in range(4):
            days = grp == g; sel = days & blocked[:, s]
            if not sel.any():
                continue
            m = days & ~blocked[:, s]; typ = np.median(v10[m if m.sum() >= 3 else days, s], 0)
            out[sel, s] = (v10[sel, s] - typ[None, :]).max(1) >= LIFT
    return out


# ------------------------------------------------------------------------------------------------ main
def main(reuse=False):
    S = s11.Stack(); new = new_names(); S.names = S.names + new; S.members = ['v10'] + S.names
    for k in S.folds:
        for n in new:
            S.pred[k][n] = load_fold(n, k)
    p = S.panel; folds = S.folds; test = S.test; st = p.stations; ns = len(st)
    names = pd.read_csv(c8.ROOT / 'data/raw/metaData.csv').set_index('station_id').station_name
    gA, A, tA, blocked, frame = run_base(S, new, 'base_keep', True, reuse)
    gB, B, tB, _, _ = run_base(S, new, 'base_nokeep', False, reuse)
    v10_t = np.log1p(pd.read_csv(c8.ROOT / 'outputs/ensemble_v10/submission_ensemble_v10.csv').boardings.to_numpy())[test.row_index].reshape(test.pred.shape)
    skip = np.isin(st, [3127, 3754])                                         # Munhak and Bucheon have their own rules inside the pipeline
    lift = {k: lifted_days(folds[k], S.pred[k]['v10'], blocked[k]) & ~skip[None, :] for k in folds}; lift_t = lifted_days(test, v10_t, blocked['test']) & ~skip[None, :]
    base = {k: np.where(lift[k][:, :, None], A[k], B[k]) for k in folds}; tb = np.where(lift_t[:, :, None], tA, tB)
    load = lambda v: {k: np.load(c8.ROOT / f'outputs/ensemble_{v}/oof_{v}.npz')[f'fold{k}'].astype(float) for k in folds}
    v12, v13, v14 = load('v12'), load('v13'), load('v14')
    keep = lambda k: {j: np.asarray((folds[j].dates < folds[k].dates.min()) | (folds[j].dates > folds[k].dates.max())) for j in folds if j != k}
    out = dict(information='local OOF only; every fit is nested (and date-blocked for the correction steps) on aligned truth; no leaderboard value is used',
               protocol='for fold k a correction is fitted on the other folds without the days that fall inside fold k; kept only if the all-fold score improves and no fold gets worse')
    ref = ('lgb_b', 'xgb') + tuple(new)
    out['retrained_models'] = dict(members=new, seeds={n: dict(folds=len(files(n, 'f1')), final=len(files(n, 'final'))) for n in new},
                                   single_model_aligned={n: S.score({k: S.pred[k][n] for k in folds}) for n in ref}, checkpoint={n: {k: strict_index(n, k) for k in folds} for n in new},
                                   stage1_weights_test=gB['test']['stage1_weights'], segment_weights_test=gB['test']['segment_weights'], models_used_test=gB['test']['models_used'])
    out['event_day_rule'] = dict(lift_threshold=LIFT, listed_station_days={k: int((blocked[k] & ~skip[None, :]).sum()) for k in folds}, kept_on_v10={k: int(lift[k].sum()) for k in folds},
                                 listed_station_days_test=int((blocked['test'] & ~skip[None, :]).sum()), kept_on_v10_test=int(lift_t.sum()),
                                 kept_test=sorted({f"{str(test.dates[i])[:10]} {int(st[s])} {names[int(st[s])]}" for i, s in zip(*np.nonzero(lift_t))}),
                                 released_test=sorted({f"{str(test.dates[i])[:10]} {int(st[s])} {names[int(st[s])]}" for i, s in zip(*np.nonzero(blocked['test'] & ~skip[None, :] & ~lift_t))}),
                                 always_v10=S.score(A), never=S.score(B), only_where_v10_has_a_lift=S.score(base))
    out['generations'] = {n: dict(aligned=S.score(z), raw_truth=S.score(z, raw=True)) for n, z in (('v12', v12), ('v13', v13), ('v14', v14), ('v15 base (stack + event-day rule)', base))}
    print('event-day rule: always v10', round(S.score(A)['all_oof'], 5), '| never', round(S.score(B)['all_oof'], 5), '| only with a lift', round(S.score(base)['all_oof'], 5), flush=True)
    cur = {k: base[k].copy() for k in folds}; cur_t = tb.copy(); steps = {}

    def judge(name, new_, extra):
        a, b = S.score(cur), S.score(new_)
        ok = b['all_oof'] < a['all_oof'] - 1e-7 and all(b[f'fold_{k}'] <= a[f'fold_{k}'] + TOL for k in folds)
        steps[name] = dict(before=a, after=b, adopted=bool(ok), **extra)
        print(f"{name:8s} {a['all_oof']:.5f} -> {b['all_oof']:.5f}  " + ' '.join(f"F{k} {b[f'fold_{k}'] - a[f'fold_{k}']:+.5f}" for k in folds) + ('  ADOPTED' if ok else '  rejected'), flush=True)
        return ok

    # ---- day-type weights ------------------------------------------------------------------------------------------------------
    mm = [n for n in ['v10'] + new + MIX_OLD if n in S.pred[1]]; grp = {k: R.day_group(folds[k].kind) for k in folds}; grp_t = R.day_group(test.kind); blocks = mix.HOUR_BLOCKS[MIX_HOURS]
    hb = np.zeros(18, int)
    for j, (a, b) in enumerate(blocks):
        hb[a:b] = j
    free_station = ~skip & ~(np.asarray(p.first) > 0)                         # not Munhak / Bucheon, not the stations opened inside the data window
    use = {k: np.broadcast_to((~blocked[k] & free_station[None, :])[:, :, None], cur[k].shape) for k in folds}; use_t = np.broadcast_to((~blocked['test'] & free_station[None, :])[:, :, None], cur_t.shape)
    gid = lambda g, shape: np.broadcast_to(g[:, None, None] * len(blocks) + hb[None, None, :], shape)
    part = lambda j, m=None: (lambda u: (mix.design(cur[j], [S.pred[j][n] for n in mm], u), (S.truth[j] - cur[j])[u], gid(grp[j], cur[j].shape)[u]))(use[j] if m is None else use[j] & m[:, None, None])
    new_ = {}; d_k = {}
    for k in folds:
        m = keep(k); d_k[k] = mix.fit([part(j, m[j]) for j in m], 4 * len(blocks), MIX_RIDGE)
        new_[k] = mix.apply(cur[k], [S.pred[k][n] for n in mm], use[k], gid(grp[k], cur[k].shape), d_k[k])
    if judge('mix', new_, dict(members=mm, ridge=MIX_RIDGE, hour_blocks=MIX_HOURS, groups=['weekday', 'saturday', 'sunday', 'special day'])):
        cur = new_; d_all = mix.fit([part(j) for j in folds], 4 * len(blocks), MIX_RIDGE)
        mt = [v10_t if n == 'v10' else (load_final(n) if n in new else meta.load_final(n)) for n in mm]; prev = cur_t.copy()
        cur_t = mix.apply(cur_t, mt, use_t, gid(grp_t, cur_t.shape), d_all)
        steps['mix'].update(test_delta={g: {n: round(float(v), 3) for n, v in zip(mm, d_all[i * len(blocks)])} for i, g in enumerate(['weekday', 'saturday', 'sunday', 'special day'])} if len(blocks) == 1 else
                            [[round(float(v), 3) for v in row] for row in d_all], test_log_rms_change=round(float(np.sqrt(((cur_t - prev) ** 2).mean())), 4), test_mean_log_change=round(float((cur_t - prev).mean()), 5))

    # ---- events: concerts and volleyball without a lift ---------------------------------------------------------------------
    # "without a lift" is judged on the v10 forecast (as in the event-day rule), the residual is the one of the current forecast
    ev = load_events(st); cells = {k: R.event_cells(folds[k], S.pred[k]['v10'], ev, st) for k in folds}; cells_t = R.event_cells(test, v10_t, ev, st)
    LI, ST_, HR_, DG_ = (R.EVENT_FEATURES.index(c) for c in ('lift_in_v12', 'station', 'hour', 'day_group'))
    res = {k: np.array([S.truth[k][i, s, h] - cur[k][i, s, h] for i, s, h in cells[k][1]]) for k in folds}
    day_open = lambda X: np.repeat(X[:, LI].reshape(-1, 18).max(1) < R.LIFT, 18); key_of = lambda x: (int(x[ST_]), int(x[HR_]), int(x[DG_] > 0))

    def event_means(kind, train):
        c = R.EVENT_FEATURES.index('is_' + kind); acc = {}
        for X, r in train:
            m = (X[:, c] > 0) & day_open(X)
            for x, v in zip(X[m], r[m]):
                a = acc.setdefault(key_of(x), [0.0, 0]); a[0] += float(v); a[1] += 1
        return acc

    def event_apply(kind, acc, X):
        c = R.EVENT_FEATURES.index('is_' + kind); m = (X[:, c] > 0) & day_open(X); f = np.zeros(len(X))
        f[m] = [(lambda a: a[0] / (a[1] + R.EVENT_SHRINK_DAYS) if a[1] >= R.EVENT_MIN_DAYS else 0.0)(acc.get(key_of(x), (0.0, 0))) for x in X[m]]
        return f
    new_ = {k: cur[k].copy() for k in folds}; gain_kind = {kd: {} for kd in R.EVENT_KINDS_USED}
    for k in folds:
        lo, hi = np.datetime64(folds[k].dates.min()), np.datetime64(folds[k].dates.max()); f = np.zeros(len(res[k]))
        train = [(cells[j][0][(cells[j][2] < lo) | (cells[j][2] > hi)], res[j][(cells[j][2] < lo) | (cells[j][2] > hi)]) for j in folds if j != k]
        for kd in R.EVENT_KINDS_USED:
            fk = event_apply(kd, event_means(kd, train), cells[k][0]); f += fk; gain_kind[kd][k] = round(float((res[k] ** 2 - (res[k] - fk) ** 2).sum()), 1)
        for (i, s, h), v in zip(cells[k][1], f):
            new_[k][i, s, h] = max(cur[k][i, s, h] + v, 0.0)
    if judge('events', new_, dict(kinds_used=R.EVENT_KINDS_USED, sse_gain_by_kind_and_fold=gain_kind)):
        cur = new_; f = np.zeros(len(cells_t[1]))
        for kd in R.EVENT_KINDS_USED:
            f += event_apply(kd, event_means(kd, [(cells[j][0], res[j]) for j in folds]), cells_t[0])
        for (i, s, h), v in zip(cells_t[1], f):
            cur_t[i, s, h] = max(cur_t[i, s, h] + v, 0.0)
        steps['events']['test_station_days_changed'] = int((f != 0).sum() // 18)

    # ---- school vacation -----------------------------------------------------------------------------------------------------
    sv = pd.read_csv(R.EXT / 'school_vacation_daily.csv'); sv = sv[(sv.area == '전체') & (sv.level == '전체')].set_index('date').share_vacation
    dv = {k: R.vacation_change(folds[k], sv) for k in folds}; dv_t = R.vacation_change(test, sv)
    before = {k: cur[k].copy() for k in folds}; new_ = {}; glob_k = {}
    for k in folds:
        m = keep(k); sl, gl = R.school_fit([(dv[j][0][m[j]], (S.truth[j] - before[j])[m[j]]) for j in m], ns, p.first > 0); glob_k[k] = gl
        new_[k] = np.maximum(before[k] + R.school_apply(dv[k][0], sl, before[k].shape), 0)
    if judge('school', new_, dict(global_slopes_nested={k: [round(float(x), 4) for x in glob_k[k]] for k in folds})):
        cur = new_; sl, gl = R.school_fit([(dv[j][0], S.truth[j] - before[j]) for j in folds], ns, p.first > 0); prev = cur_t.copy()
        cur_t = np.maximum(cur_t + R.school_apply(dv_t[0], sl, cur_t.shape), 0); vac = (np.asarray(test.kind) == 'wd') & (dv_t[0] > 0.5)
        steps['school'].update(test_global_slopes=[round(float(x), 4) for x in gl], test_mean_log_change_school_hours=round(float((cur_t - prev)[vac][:, :, [0, 1, 2, 9, 10, 11]].mean()), 4))

    # ---- very small forecasts --------------------------------------------------------------------------------------------------
    before = {k: cur[k].copy() for k in folds}; new_ = {}
    for k in folds:
        m = keep(k); new_[k] = before[k] + R.low_apply(before[k], R.low_fit([(before[j][m[j]], (S.truth[j] - before[j])[m[j]]) for j in m]))
    if judge('low', new_, dict(edges=R.LOW_EDGES)):
        cur = new_; add = R.low_fit([(before[j], S.truth[j] - before[j]) for j in folds]); n_low = int((np.expm1(cur_t) < R.LOW_EDGES[-1]).sum())
        cur_t = cur_t + R.low_apply(cur_t, add); steps['low'].update(test_added_log=[round(float(x), 3) for x in add], test_cells_changed=n_low)

    # ---- special days: calendar prior ----------------------------------------------------------------------------------------
    CP = CalendarPrior(); before = {k: cur[k].copy() for k in folds}
    hs = {k: holiday_shift(folds[k], before[k], CP.fit(folds[k].origin)(folds[k].dates)) for k in folds}

    def gamma(pairs):
        num = den = 0.0
        for j, mask in pairs:
            special, shift, typ = hs[j]; r = S.truth[j] - before[j]
            for i in np.flatnonzero(special & mask):
                num += float(r[i][typ[i]].sum() * shift[i]); den += shift[i] ** 2 * r[i][typ[i]].size
        return float(np.clip(num / max(den, 1e-9), 0, 1))

    def holiday_apply(pred, h, g):
        special, shift, typ = h; z = pred.copy()
        for i in np.flatnonzero(special):
            z[i][typ[i]] = np.maximum(z[i][typ[i]] + g * shift[i], 0)
        return z
    gam = {k: gamma(list(keep(k).items())) for k in folds}; new_ = {k: holiday_apply(before[k], hs[k], gam[k]) for k in folds}
    if judge('holiday', new_, dict(gamma_nested={k: round(v, 3) for k, v in gam.items()}, special_days={k: int(hs[k][0].sum()) for k in folds}, typical_band=TYPICAL)):
        cur = new_; g = gamma([(j, np.ones(len(folds[j].dates), bool)) for j in folds]); ht = holiday_shift(test, cur_t, CP.fit(c8.FINAL)(test.dates))
        cur_t = holiday_apply(cur_t, ht, g)
        steps['holiday'].update(gamma_test=round(g, 3), test_days=[dict(date=str(test.dates[i])[:10], kind=str(test.kind[i]), prior_minus_forecast=round(float(ht[1][i]), 3),
                                                                          applied_log_shift=round(float(g * ht[1][i]), 3), typical_stations=int(ht[2][i].sum())) for i in np.flatnonzero(ht[0])])

    # ---- results ----------------------------------------------------------------------------------------------------------------
    out['steps'] = steps; out['generations']['v15 base (stack + event-day rule)'] = dict(aligned=S.score(base), raw_truth=S.score(base, raw=True)); out['generations']['v15'] = dict(aligned=S.score(cur), raw_truth=S.score(cur, raw=True))
    a14, a15 = out['generations']['v14']['aligned'], out['generations']['v15']['aligned']
    out['folds_worse_than_v14'] = {k: round(a15[f'fold_{k}'] - a14[f'fold_{k}'], 6) for k in folds if a15[f'fold_{k}'] > a14[f'fold_{k}'] + TOL}
    sse14 = sum(((v14[k] - S.truth[k]) ** 2).sum(axis=(0, 2)) for k in folds); sse15 = sum(((cur[k] - S.truth[k]) ** 2).sum(axis=(0, 2)) for k in folds); n = 4 * 123 * 18
    order = np.argsort(sse15 - sse14)
    out['stations'] = {f'{int(st[i])} {names[int(st[i])]}': dict(v14=round(float(np.sqrt(sse14[i] / n)), 4), v15=round(float(np.sqrt(sse15[i] / n)), 4)) for i in list(order[:12]) + list(order[-5:])}
    np.savez_compressed(OUT / 'oof_v15.npz', **{f'fold{k}': cur[k].astype(np.float32) for k in folds})
    flat = np.empty(cur_t.size); flat[test.row_index] = np.maximum(cur_t, 0).reshape(-1)
    sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
    assert len(sub) == 157194 and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all() and (sub.example_id.to_numpy() == frame.example_id.to_numpy()).all()
    path = OUT / 'submission_ensemble_v15.csv'; sub.to_csv(path, index=False, float_format='%.8f')
    ref14 = np.log1p(pd.read_csv(c8.ROOT / 'outputs/ensemble_v14/submission_ensemble_v14.csv').boardings.to_numpy())
    out['submission'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(sub)), mean_boardings=float(sub.boardings.mean()),
                             log_rmsd_vs_v14=float(np.sqrt(np.mean((flat - ref14) ** 2))), mean_log_diff_vs_v14=float((flat - ref14).mean()))
    (OUT / 'gates_v15.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(f"v15: aligned v14 {a14['all_oof']:.5f} -> v15 {a15['all_oof']:.5f} | raw {out['generations']['v14']['raw_truth']['all_oof']:.5f} -> {out['generations']['v15']['raw_truth']['all_oof']:.5f} | worse folds {out['folds_worse_than_v14']}")
    print(json.dumps(out['submission'], indent=1), flush=True)
    return out


if __name__ == '__main__':
    import sys
    main(reuse='--reuse' in sys.argv)
