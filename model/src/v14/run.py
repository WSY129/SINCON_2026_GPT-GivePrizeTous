"""Builds submission_ensemble_v14.csv (CPU only; no leaderboard value is used).

    python -m src.v14.train ...      (once: retrains the two LightGBM models on the v14 panel)
    python -m src.v14.run

1. base   the v12 stacking pipeline, unchanged, with the two retrained LightGBM models (lgb_c, lgb_d) added to the
          member list -> outputs/ensemble_v14/base/
2. steps  corrections fitted on the out-of-fold residuals of the base, nested and date-blocked (for fold k: the other
          folds without the days that fall inside fold k); a step is kept only if the all-fold score improves and no
          fold gets worse:
            events   concert / volleyball days without a lift, mean residual by station, hour and weekday / weekend
            school   change of the school-vacation share against the anchor window x school hours
            low      forecasts under 5 boardings
            holiday  special days: forecast pulled towards a calendar prior learned from ten years of daily card data of
                     the line-7 stations and the transfer-partner stations (data before each cutoff only)
"""
import hashlib
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.v8 import core as c8
from src.v11 import stack as s11
from src.v12 import run as run12, meta
from src.v13 import run as R

OUT = c8.ROOT / 'outputs/ensemble_v14'
MODELS = OUT / 'models'
EXT = OUT / 'external'
NEW = ['lgb_c', 'lgb_d']
CAL_FEATS = ['weekday', 'is_holiday', 'offday_block_len', 'offday_block_pos', 'is_before_offday', 'is_after_offday', 'is_bridge_day', 'is_labor_day',
             'is_seollal_period', 'is_chuseok_period', 'mj_day', 'days_to_mj', 'month']
TYPICAL = 0.35           # a station counts as typical on a special day if its implied day ratio is within this (log) of the median
TOL = 1e-5


# ------------------------------------------------------------------------------------------------ retrained members
def curves(name):
    return {k: json.loads((MODELS / f'{f}__{name}__2026.json').read_text())['valid_rmsle'] for k, f in s11.FOLDS.items()}


def strict_index(name, k=None):
    c = curves(name); use = [3, 4] if k is None else [j for j in c if j != k]
    return int(round(np.mean([int(np.argmin(c[j])) for j in use])))


def load_fold(name, k):
    d = np.load(MODELS / f'{s11.FOLDS[k]}__{name}__2026.npz'); i = min(strict_index(name, k), d['pred'].shape[0] - 1)
    return np.maximum(d['anchor'] + meta.smooth(d['pred'], i), 0).astype(float)


def load_final(name):
    out = []
    for p in sorted(MODELS.glob(f'final__{name}__*.npz')):
        d = np.load(p); i = min(strict_index(name), d['pred'].shape[0] - 1); out.append(np.maximum(d['anchor'] + meta.smooth(d['pred'], i), 0))
    return np.mean(out, 0).astype(float)


def base_stack():
    """The v12 pipeline with the retrained models as extra members.  Nothing of v12 is written: its output folder is
    redirected to outputs/ensemble_v14/base for the duration of the call."""
    S = s11.Stack(); S.names = S.names + NEW; S.members = ['v10'] + S.names
    for k in S.folds:
        for n in NEW:
            S.pred[k][n] = load_fold(n, k)
    saved = (s11.Stack, meta.load_fold, meta.load_final, run12.OUT); (OUT / 'base').mkdir(parents=True, exist_ok=True)
    try:
        s11.Stack = lambda: S
        meta.load_fold = lambda n, k, smoothed=True: load_fold(n, k) if n in NEW else saved[1](n, k, smoothed)
        meta.load_final = lambda n, smoothed=True: load_final(n) if n in NEW else saved[2](n, smoothed)
        run12.OUT = OUT / 'base'; assert run12.OUT != saved[3]
        g = run12.main()
    finally:
        s11.Stack, meta.load_fold, meta.load_final, run12.OUT = saved
    return S, g


# ------------------------------------------------------------------------------------------------ calendar prior
class CalendarPrior:
    """log(day boardings / ordinary same-weekday level) as a function of calendar position, learned from the Seoul card
    data of 2016-2019 and 2022 onwards (rows before the cutoff only)."""

    def __init__(self):
        d = pd.read_csv(EXT / 'seoul_daily_2015_2026h1.csv', parse_dates=['date']); d['key'] = d.line + ' ' + d.station_name
        assert d.date.max() <= pd.Timestamp('2026-06-30')                    # no evaluation-period ridership of any line
        self.cal = pd.read_csv(EXT / 'calendar.csv', encoding='utf-8-sig', parse_dates=['date']).set_index('date')
        P = np.log(d.pivot_table(index='date', columns='key', values='boardings').clip(lower=1)); P = P[P.index >= '2016-01-01']; c = self.cal.reindex(P.index)
        ordinary = ((c.is_holiday == 0) & (c.is_bridge_day == 0) & (c.is_labor_day == 0) & (c.is_mj_period == 0)).to_numpy()
        Pm = P.copy(); Pm[~ordinary] = np.nan
        base = np.nanmedian(np.stack([Pm.shift(7 * w, freq='D').reindex(P.index).to_numpy() for w in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)]), 0)
        self.P, self.Y = P, P - base
        self.line7 = [k for k in P.columns if k.startswith('7호선') and '온수' not in k]
        self.covid = (P.index >= '2020-02-01') & (P.index <= '2022-04-30')

    def fit(self, origin):
        rows = []
        for key in self.P.columns:
            m = (self.P.index < pd.Timestamp(origin)) & ~self.covid & self.Y[key].notna().to_numpy()
            f = self.cal.loc[self.P.index[m], CAL_FEATS].copy(); f['line7'] = int(key in self.line7); f['y'] = self.Y[key].to_numpy()[m]; rows.append(f)
        T = pd.concat(rows); T['days_to_mj'] = T.days_to_mj.clip(-8, 8)
        ms = [lgb.train(dict(objective='regression', learning_rate=0.05, num_leaves=15, min_data_in_leaf=15, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                             lambda_l2=1.0, verbose=-1, num_threads=8, seed=sd), lgb.Dataset(T[CAL_FEATS + ['line7']], T.y), 300) for sd in (1, 2, 3)]

        def predict(dates):
            f = self.cal.reindex(pd.DatetimeIndex(dates))[CAL_FEATS].copy(); f['days_to_mj'] = f.days_to_mj.clip(-8, 8); out = []
            for l7 in (1, 0):                                                # mean of the line-7 view and the partner-station view
                f['line7'] = l7; out.append(np.mean([m.predict(f[CAL_FEATS + ['line7']]) for m in ms], 0))
            return np.mean(out, 0)
        return predict


def implied_ratio(block, pred):
    """log day total of a forecast minus the same station's ordinary same-weekday level inside the block [day, S]."""
    day = np.log(np.expm1(pred).sum(2)); kind = np.asarray(block.kind); dw = block.dates.dayofweek.to_numpy(); out = np.zeros_like(day)
    for w in range(7):
        out[dw == w] = day[dw == w] - np.median(day[(dw == w) & np.isin(kind, ['wd', 'sat', 'sun'])], 0)[None, :]
    return out


def holiday_shift(block, pred, prior):
    """(special-day mask, day shift [day] = prior - median implied ratio, typical-station mask [day, S])."""
    I = implied_ratio(block, pred); special = ~np.isin(np.asarray(block.kind), ['wd', 'sat', 'sun']); med = np.median(I, 1)
    return special, (prior - med) * special, np.abs(I - med[:, None]) < TYPICAL


# ------------------------------------------------------------------------------------------------ main
def main():
    S, g12 = base_stack(); p = S.panel; folds = S.folds; test = S.test; st = p.stations; ns = len(st)
    names = pd.read_csv(c8.ROOT / 'data/raw/metaData.csv').set_index('station_id').station_name
    ob = np.load(OUT / 'base/oof_v12.npz'); base = {k: ob[f'fold{k}'].astype(float) for k in folds}
    subb = pd.read_csv(OUT / 'base/submission_ensemble_v12.csv'); tb = np.log1p(subb.boardings.to_numpy())[test.row_index].reshape(test.pred.shape)
    o13 = np.load(c8.ROOT / 'outputs/ensemble_v13/oof_v13.npz'); v13 = {k: o13[f'fold{k}'].astype(float) for k in folds}
    o12 = np.load(c8.ROOT / 'outputs/ensemble_v12/oof_v12.npz'); v12 = {k: o12[f'fold{k}'].astype(float) for k in folds}
    keep = lambda k: {j: np.asarray((folds[j].dates < folds[k].dates.min()) | (folds[j].dates > folds[k].dates.max())) for j in folds if j != k}
    out = dict(information='local OOF only; every fit is nested (and date-blocked for the correction steps) on aligned truth; no leaderboard value is used',
               protocol='for fold k a correction is fitted on the other folds without the days that fall inside fold k; kept only if the all-fold score improves and no fold gets worse')
    single = {n: S.score({k: S.pred[k][n] for k in folds}) for n in ('lgb_a', 'lgb_b') + tuple(NEW)}
    out['retrained_models'] = dict(single_model_aligned=single, checkpoint={n: {k: strict_index(n, k) for k in folds} for n in NEW}, checkpoint_final={n: strict_index(n) for n in NEW},
                                   stage1_weights_test=g12['test']['stage1_weights'], segment_weights_test=g12['test']['segment_weights'], models_used_test=g12['test']['models_used'])
    out['generations'] = {'v12': dict(aligned=S.score(v12), raw_truth=S.score(v12, raw=True)), 'v13': dict(aligned=S.score(v13), raw_truth=S.score(v13, raw=True)),
                          'v14 base (v12 pipeline + retrained models)': dict(aligned=S.score(base), raw_truth=S.score(base, raw=True))}
    cur = {k: base[k].copy() for k in folds}; cur_t = tb.copy(); steps = {}

    def judge(name, new, extra):
        a, b = S.score(cur), S.score(new)
        ok = b['all_oof'] < a['all_oof'] - 1e-7 and all(b[f'fold_{k}'] <= a[f'fold_{k}'] + TOL for k in folds)
        steps[name] = dict(before=a, after=b, adopted=bool(ok), **extra)
        print(f"{name:8s} {a['all_oof']:.5f} -> {b['all_oof']:.5f}  " + ' '.join(f"F{k} {b[f'fold_{k}'] - a[f'fold_{k}']:+.5f}" for k in folds) + ('  ADOPTED' if ok else '  rejected'), flush=True)
        return ok

    # ---- events: concerts and volleyball without a lift (same estimator as v13) ------------------------------------------
    ev = R.load_events(st); cells = {k: R.event_cells(folds[k], base[k], ev, st) for k in folds}; cells_t = R.event_cells(test, tb, ev, st)
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
    new = {k: cur[k].copy() for k in folds}; gain_kind = {kd: {} for kd in R.EVENT_KINDS_USED}
    for k in folds:
        lo, hi = np.datetime64(folds[k].dates.min()), np.datetime64(folds[k].dates.max()); f = np.zeros(len(res[k]))
        train = [(cells[j][0][(cells[j][2] < lo) | (cells[j][2] > hi)], res[j][(cells[j][2] < lo) | (cells[j][2] > hi)]) for j in folds if j != k]
        for kd in R.EVENT_KINDS_USED:
            fk = event_apply(kd, event_means(kd, train), cells[k][0]); f += fk; gain_kind[kd][k] = round(float((res[k] ** 2 - (res[k] - fk) ** 2).sum()), 1)
        for (i, s, h), v in zip(cells[k][1], f):
            new[k][i, s, h] = max(cur[k][i, s, h] + v, 0.0)
    if judge('events', new, dict(kinds_used=R.EVENT_KINDS_USED, sse_gain_by_kind_and_fold=gain_kind)):
        cur = new; f = np.zeros(len(cells_t[1]))
        for kd in R.EVENT_KINDS_USED:
            f += event_apply(kd, event_means(kd, [(cells[j][0], res[j]) for j in folds]), cells_t[0])
        for (i, s, h), v in zip(cells_t[1], f):
            cur_t[i, s, h] = max(cur_t[i, s, h] + v, 0.0)
        steps['events']['test_station_days_changed'] = int((f != 0).sum() // 18)

    # ---- school vacation -----------------------------------------------------------------------------------------------
    sv = pd.read_csv(R.EXT / 'school_vacation_daily.csv'); sv = sv[(sv.area == '전체') & (sv.level == '전체')].set_index('date').share_vacation
    dv = {k: R.vacation_change(folds[k], sv) for k in folds}; dv_t = R.vacation_change(test, sv)
    before = {k: cur[k].copy() for k in folds}; new = {}; glob_k = {}
    for k in folds:
        m = keep(k); sl, gl = R.school_fit([(dv[j][0][m[j]], (S.truth[j] - before[j])[m[j]]) for j in m], ns, p.first > 0); glob_k[k] = gl
        new[k] = np.maximum(before[k] + R.school_apply(dv[k][0], sl, before[k].shape), 0)
    if judge('school', new, dict(global_slopes_nested={k: [round(float(x), 4) for x in glob_k[k]] for k in folds})):
        cur = new; sl, gl = R.school_fit([(dv[j][0], S.truth[j] - before[j]) for j in folds], ns, p.first > 0); prev = cur_t.copy()
        cur_t = np.maximum(cur_t + R.school_apply(dv_t[0], sl, cur_t.shape), 0); vac = (np.asarray(test.kind) == 'wd') & (dv_t[0] > 0.5)
        steps['school'].update(test_global_slopes=[round(float(x), 4) for x in gl], test_mean_log_change_school_hours=round(float((cur_t - prev)[vac][:, :, [0, 1, 2, 9, 10, 11]].mean()), 4))

    # ---- very small forecasts --------------------------------------------------------------------------------------------
    before = {k: cur[k].copy() for k in folds}; new = {}
    for k in folds:
        m = keep(k); new[k] = before[k] + R.low_apply(before[k], R.low_fit([(before[j][m[j]], (S.truth[j] - before[j])[m[j]]) for j in m]))
    if judge('low', new, dict(edges=R.LOW_EDGES)):
        cur = new; add = R.low_fit([(before[j], S.truth[j] - before[j]) for j in folds]); n_low = int((np.expm1(cur_t) < R.LOW_EDGES[-1]).sum())
        cur_t = cur_t + R.low_apply(cur_t, add); steps['low'].update(test_added_log=[round(float(x), 3) for x in add], test_cells_changed=n_low)

    # ---- special days: calendar prior from ten years of daily card data ---------------------------------------------------
    CP = CalendarPrior(); before = {k: cur[k].copy() for k in folds}
    hs = {k: holiday_shift(folds[k], before[k], CP.fit(folds[k].origin)(folds[k].dates)) for k in folds}

    def gamma(ks_masks):
        num = den = 0.0
        for j, mask in ks_masks:
            special, shift, typ = hs[j]; r = S.truth[j] - before[j]
            for i in np.flatnonzero(special & mask):
                num += float(r[i][typ[i]].sum() * shift[i]); den += shift[i] ** 2 * r[i][typ[i]].size
        return float(np.clip(num / max(den, 1e-9), 0, 1))

    def holiday_apply(pred, h, g):
        special, shift, typ = h; z = pred.copy()
        for i in np.flatnonzero(special):
            z[i][typ[i]] = np.maximum(z[i][typ[i]] + g * shift[i], 0)
        return z
    gam = {k: gamma([(j, m) for j, m in keep(k).items()]) for k in folds}; new = {k: holiday_apply(before[k], hs[k], gam[k]) for k in folds}
    if judge('holiday', new, dict(gamma_nested={k: round(v, 3) for k, v in gam.items()}, special_days={k: int(hs[k][0].sum()) for k in folds}, typical_band=TYPICAL)):
        cur = new; g = gamma([(j, np.ones(len(folds[j].dates), bool)) for j in folds]); ht = holiday_shift(test, cur_t, CP.fit(c8.FINAL)(test.dates)); prev = cur_t.copy()
        cur_t = holiday_apply(cur_t, ht, g)
        steps['holiday'].update(gamma_test=round(g, 3), test_days=[dict(date=str(test.dates[i])[:10], kind=str(test.kind[i]), prior_minus_forecast=round(float(ht[1][i]), 3),
                                                                          applied_log_shift=round(float(g * ht[1][i]), 3), typical_stations=int(ht[2][i].sum())) for i in np.flatnonzero(ht[0])])

    # ---- results ------------------------------------------------------------------------------------------------------------
    out['steps'] = steps; out['generations']['v14'] = dict(aligned=S.score(cur), raw_truth=S.score(cur, raw=True))
    a13, a14 = out['generations']['v13']['aligned'], out['generations']['v14']['aligned']
    out['folds_worse_than_v13'] = {k: round(a14[f'fold_{k}'] - a13[f'fold_{k}'], 6) for k in folds if a14[f'fold_{k}'] > a13[f'fold_{k}'] + TOL}
    sse13 = sum(((v13[k] - S.truth[k]) ** 2).sum(axis=(0, 2)) for k in folds); sse14 = sum(((cur[k] - S.truth[k]) ** 2).sum(axis=(0, 2)) for k in folds); n = 4 * 123 * 18
    order = np.argsort(sse14 - sse13)
    out['stations'] = {f'{int(st[i])} {names[int(st[i])]}': dict(v13=round(float(np.sqrt(sse13[i] / n)), 4), v14=round(float(np.sqrt(sse14[i] / n)), 4)) for i in list(order[:12]) + list(order[-5:])}
    np.savez_compressed(OUT / 'oof_v14.npz', **{f'fold{k}': cur[k].astype(np.float32) for k in folds})
    flat = np.empty(cur_t.size); flat[test.row_index] = np.maximum(cur_t, 0).reshape(-1)
    frame = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
    assert len(frame) == 157194 and np.isfinite(frame.boardings).all() and (frame.boardings >= 0).all() and (frame.example_id.to_numpy() == subb.example_id.to_numpy()).all()
    path = OUT / 'submission_ensemble_v14.csv'; frame.to_csv(path, index=False, float_format='%.8f')
    ref = np.log1p(pd.read_csv(c8.ROOT / 'outputs/ensemble_v13/submission_ensemble_v13.csv').boardings.to_numpy())
    out['submission'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(frame)), mean_boardings=float(frame.boardings.mean()),
                             log_rmsd_vs_v13=float(np.sqrt(np.mean((flat - ref) ** 2))), mean_log_diff_vs_v13=float((flat - ref).mean()))
    (OUT / 'gates_v14.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(f"v14: aligned v12 {out['generations']['v12']['aligned']['all_oof']:.5f} | v13 {a13['all_oof']:.5f} -> v14 {a14['all_oof']:.5f} | raw {out['generations']['v13']['raw_truth']['all_oof']:.5f} -> {out['generations']['v14']['raw_truth']['all_oof']:.5f} | worse folds {out['folds_worse_than_v13']}")
    print(json.dumps(out['submission'], indent=1), flush=True)
    return out


if __name__ == '__main__':
    main()
