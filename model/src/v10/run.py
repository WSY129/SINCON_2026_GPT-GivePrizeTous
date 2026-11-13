"""Builds submission_ensemble_v10.csv and the result table (local OOF only; no leaderboard value is used).

    python -m src.v10.run

Chain on the v7 forecast:  P1 seasonal index (v10 fixes) -> P2 holiday effects -> A4 weather (daily + hourly rain,
temperature) -> A2 Bucheon actual matches (all matches) -> A1 Munhak actual games and events -> A3 one-off and
recurring events -> blend with the v9 tree models.

Every coefficient is fitted on *aligned* truth: folds whose origin is in 2025 are scored on the 2025 aggregation
basis (their 2026 truth is converted with the station x hour factor), because the test has no such break.  Each
step is reported on both the aligned and the raw truth.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v7 import core as core7
from src.v8 import core as c8, pre, actual
from src.v8.run import block_events
from . import seasonal, games
from .weather import Weather10

OUT = c8.ROOT / 'outputs/ensemble_v10'
V9 = c8.ROOT / 'outputs/ensemble_v9/models'
MEMBERS = ['lgb', 'cb', 'xgb']
REPORT = [3127, 3226, 3754, 3107, 3136, 3139, 3209, 3210, 3120, 3119, 3111, 3110, 3137]
ISLE = dict(date='2026-09-12', scale=0.6)


def main():
    panel, folds, test = c8.load_blocks()
    train = core7.load_train(); schedule = core7.schedule_v7(); cfg = games.config()
    f = pd.read_csv(c8.SHARE / 'ictr_to_train_factor_station_hour.csv'); logf = np.zeros((panel.S, panel.H))
    logf[np.searchsorted(panel.stations, f.station_id.to_numpy()), f.hour.to_numpy() - 6] = np.log(f.factor_boardings.to_numpy())
    early = {k: b.origin < '2026-01-01' for k, b in folds.items()}
    jump = {k: (np.where((b.dates.year == 2026)[:, None, None], logf[None], 0.0) if early[k] else 0.0) for k, b in folds.items()}
    raw = {k: b.truth.copy() for k, b in folds.items()}
    for k, b in folds.items():
        b.truth = raw[k] + jump[k]                                    # aligned truth, used for every fit below
    ssg, _ = actual.load_games(); matches = games.bucheon_matches(cfg)
    recurring = cfg.get('station_events', [])
    extra = [(actual.BUCHEON, list(matches.date))] + [(sid, [pd.Timestamp(d) for d in e['dates'] + e['past_dates']]) for e in recurring for sid in e['stations']]
    for b in list(folds.values()) + [test]:
        b.blocked[:] = False; block_events(b, schedule, extra=extra)
    history = seasonal.History10(panel); weather = Weather10(panel)

    def both(pred):
        """aligned and raw scores of fold forecasts given on the fold's own basis."""
        a = c8.scores(folds, pred)
        saved = {k: b.truth for k, b in folds.items()}
        for k, b in folds.items():
            b.truth = raw[k]
        r = c8.scores(folds, pred)
        for k, b in folds.items():
            b.truth = saved[k]
        return a, r

    cur = {k: b.pred.copy() for k, b in folds.items()}; tcur = test.pred.copy()
    out = dict(information='local OOF only; nested coefficients fitted on aligned truth; no leaderboard value is used', steps=[])
    rows = []

    def record(label, pred, adopted, extra=None):
        a, r = both(pred)
        st = c8.station_scores(panel, folds, pred, REPORT)
        rows.append(dict(step=label, adopted=adopted, **{f'aligned_{k}': v for k, v in a.items()}, **{f'raw_{k}': v for k, v in r.items()}, **{f'st_{k}': v for k, v in st.items()}))
        out['steps'].append(dict(step=label, adopted=adopted, aligned=a, raw_truth=r, **(extra or {})))
        return a

    prev = record('v7', cur, '')

    def gate(label, new, tnew, extra):
        nonlocal cur, tcur, prev
        a, _ = both(new)
        worse = [k for k in (1, 2, 3, 4) if a[f'fold_{k}'] > prev[f'fold_{k}'] + 1e-5]
        adopt = a['all_oof'] < prev['all_oof'] - 1e-6 and not worse
        record(label, new, bool(adopt), dict(extra, folds_worse=worse))
        print(f"{label:30s} aligned {prev['all_oof']:.5f} -> {a['all_oof']:.5f}  " + ' '.join(f"F{k} {a[f'fold_{k}']:.5f}" for k in (1, 2, 3, 4)) + f"  {'ADOPT' if adopt else 'reject'}", flush=True)
        if adopt:
            cur, tcur, prev = new, tnew, a

    def module(label, fn, bounds):
        info = {}

        def feature(block, pred):
            res = fn(block, pred)
            if isinstance(res, tuple):
                info[block.name] = res[1]; res = res[0]
            return res
        new, per_fold, c_all, changed = c8.apply_module(folds, cur, feature, bounds)
        feats = feature(test, tcur)
        delta = sum(c * x for c, x in zip(c_all, feats))
        gate(label, new, tcur + delta, dict(coefficients_nested=per_fold, coefficients_all_folds=c_all, details=info,
                                            test_cells_changed=int((delta != 0).sum())))

    new, info, tnew = seasonal.apply(folds, cur, history, test, tcur)
    gate('P1 다년 계절 지수 (v10)', new, tnew, dict(details=info, line7_cells_added=history.added, gate=[list(x) for x in seasonal.GATE]))
    module('P2 역별 공휴일 효과', pre.holiday_features, (0.0, 1.2))
    module('A4 실측 기상 (일·시간 강수, 기온)', lambda b, p: weather.features(b, p), (0.0, 1.5))
    out['weather'] = weather.last_info
    module('A2 부천 실제 경기 (전 경기)', lambda b, p: games.bucheon_feature(b, p, matches), (0.0, 1.2))
    module('A1 문학 실제 경기·행사', lambda b, p: games.munhak_feature(b, p, train, schedule, ssg, cfg), (0.0, 1.2))

    # ---- events of the evaluation period that cannot be validated on folds --------------------------------
    d = pd.Timestamp(ISLE['date']); k = int(np.flatnonzero(test.dates == d)[0]); s = int(np.searchsorted(panel.stations, actual.SONGDO))
    past = [pd.Timestamp(e['date']) for e in schedule.overlay_events() if 3139 in e.get('stations', []) and pd.Timestamp(e['date']) <= pd.Timestamp(c8.FINAL) and pd.Timestamp(e['date']).dayofweek == 5]
    test.blocked[k, s] = True
    tcur[k, s] = test.neighbour(tcur, k, allow=c8.ORDINARY)[s] + actual.festival_lift(test, past, ISLE['scale'])
    tcur, event_log = games.apply_recurring(test, tcur, recurring)
    out['unvalidated_events'] = dict(isle=dict(date=ISLE['date'], scale=ISLE['scale']), recurring=event_log)
    base_final = prev

    # ---- blend with the v9 tree models ---------------------------------------------------------------------
    munhak = int(np.searchsorted(panel.stations, c8.MUNHAK))
    v9 = {n: {k: np.load(V9 / f'f{k}__{n}.npz')['lofo'].astype(float) + (logf[None] if early[k] else 0.0) for k in folds} for n in MEMBERS}
    truth = {k: b.truth for k, b in folds.items()}
    free = {}
    for k, b in folds.items():
        m = ~np.broadcast_to(b.blocked[:, :, None], b.truth.shape).copy(); m[:, munhak] = False; free[k] = m

    def seg_of(b):
        sg = np.where(np.isin(b.kind, c8.SPECIAL), 2, 0)
        sg = np.where((sg == 0) & (weather.wet[b.di] > 0), 1, sg)
        return np.broadcast_to(sg[:, None, None], b.pred.shape)
    seg = {k: seg_of(b) for k, b in folds.items()}; seg_t = seg_of(test)

    def errs(k, sid):
        m = free[k] & (seg[k] == sid)
        return np.stack([cur[k][m] - truth[k][m]] + [v9[n][k][m] - truth[k][m] for n in MEMBERS], 1)

    def fit(train_folds):
        return {sid: core7.convex(np.concatenate([errs(j, sid) for j in train_folds])) for sid in (0, 1, 2)}

    def mix(w, base, members, sg, keep):
        res = np.zeros_like(base)
        for sid in (0, 1, 2):
            res += (sg == sid) * (w[sid][0] * base + sum(wi * p for wi, p in zip(w[sid][1:], members)))
        return np.where(keep, res, base)
    standard = {k: mix(fit([j for j in folds if j != k]), cur[k], [v9[n][k] for n in MEMBERS], seg[k], free[k]) for k in folds}
    matched = {4: mix(fit([3]), cur[4], [v9[n][4] for n in MEMBERS], seg[4], free[4]), 3: mix(fit([4]), cur[3], [v9[n][3] for n in MEMBERS], seg[3], free[3])}
    w34 = fit([3, 4])
    for k in (1, 2):
        matched[k] = mix(w34, cur[k], [v9[n][k] for n in MEMBERS], seg[k], free[k])
    singles = {'v10 기반(결합 전)': both(cur)}
    for n in MEMBERS:
        singles[f'v9 {n} 단독'] = both({k: v9[n][k] for k in folds})
    a_std, r_std = both(standard); a_m, r_m = both(matched)
    best_single = {s: min(v[i]['all_oof'] for v in singles.values()) for i, s in enumerate(('aligned', 'raw'))}
    gain34 = min(base_final['fold_3'] - a_m['fold_3'], base_final['fold_4'] - a_m['fold_4'])
    worse = [k for k in (1, 2, 3, 4) if a_m[f'fold_{k}'] > base_final[f'fold_{k}'] + 1e-5]
    adopt = bool(a_m['all_oof'] < base_final['all_oof'] - 1e-6 and not worse and gain34 >= 1e-4)
    out['blend'] = dict(members=MEMBERS, singles={n: dict(aligned=v[0], raw_truth=v[1]) for n, v in singles.items()},
                        standard_nested=dict(aligned=a_std, raw_truth=r_std), regime_matched=dict(aligned=a_m, raw_truth=r_m),
                        beats_every_single=dict(standard_nested_aligned=bool(a_std['all_oof'] < best_single['aligned']), standard_nested_raw=bool(r_std['all_oof'] < best_single['raw']),
                                                regime_matched_aligned=bool(a_m['all_oof'] < best_single['aligned']), regime_matched_raw=bool(r_m['all_oof'] < best_single['raw'])),
                        weights_test={int(sid): dict(zip(['v10'] + MEMBERS, [round(float(x), 2) for x in w34[sid]])) for sid in (0, 1, 2)},
                        segments={0: '평소 맑은 날', 1: '비 온 날', 2: '공휴일·명절·연휴 범주'}, min_gain_fold_3_4=float(gain34), folds_worse=worse, adopted=adopt)
    record('+ v9 결합 (Fold 3·4 가중치)', matched, adopt)
    rows.append(dict(step='참고: v9 결합 (일반 4-fold 중첩 가중치)', adopted='', **{f'aligned_{k}': v for k, v in a_std.items()}, **{f'raw_{k}': v for k, v in r_std.items()}))
    print(f"blend: base {base_final['all_oof']:.5f} -> matched {a_m['all_oof']:.5f} (F3 {a_m['fold_3']:.5f} F4 {a_m['fold_4']:.5f}), standard {a_std['all_oof']:.5f} (F3 {a_std['fold_3']:.5f} F4 {a_std['fold_4']:.5f})  {'ADOPT' if adopt else 'reject'}  weights {out['blend']['weights_test']}", flush=True)
    if adopt:
        finals = [np.mean([np.load(p)['pred'] for p in sorted(V9.glob(f'final__{n}__*.npz'))], 0) for n in MEMBERS]
        keep = ~np.broadcast_to(test.blocked[:, :, None], tcur.shape).copy(); keep[:, munhak] = False
        tcur = mix(w34, tcur, finals, seg_t, keep)
    flat = np.empty(tcur.size); flat[test.row_index] = np.maximum(tcur, 0).reshape(-1)
    frame = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
    assert len(frame) == 157194 and np.isfinite(frame.boardings).all() and (frame.boardings >= 0).all()
    path = OUT / 'submission_ensemble_v10.csv'; frame.to_csv(path, index=False, float_format='%.8f')
    ref = {n: np.log1p(pd.read_csv(c8.ROOT / p).boardings.to_numpy()) for n, p in (('v8', 'outputs/ensemble_v8/submission_ensemble_v8.csv'), ('v9', 'outputs/ensemble_v9/submission_ensemble_v9.csv'))}
    out['submission'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(frame)), mean_boardings=float(frame.boardings.mean()),
                             **{f'log_rmsd_vs_{n}': float(np.sqrt(np.mean((flat - v) ** 2))) for n, v in ref.items()},
                             **{f'rows_changed_vs_{n}': int((np.abs(flat - v) > 1e-9).sum()) for n, v in ref.items()})
    np.savez_compressed(OUT / 'oof_v10.npz', **{f'fold{k}': (matched if adopt else cur)[k].astype(np.float32) for k in folds})
    (OUT / 'gates_v10.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    pd.DataFrame(rows).to_csv(OUT / 'VALIDATION_V10.csv', index=False, float_format='%.5f')
    print(json.dumps(out['submission'], indent=1))
    return out


if __name__ == '__main__':
    main()
