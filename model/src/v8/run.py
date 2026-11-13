"""Builds the two v8 submissions and the validation table (local out-of-fold numbers only).

    python -m src.v8.run            # writes outputs/ensemble_v8/{submission_precutoff.csv, submission_ensemble_v8.csv,
                                    #                              gates_v8.json, VALIDATION_V8.csv}

precutoff : v7 forecast -> P1 seasonal index -> P2 holiday effects (-> P3 ghost removal if it passes)
main (v8) : the same chain -> A4 observed rain -> A2 Bucheon actual matches -> A1 Munhak actual games -> A3 festival
A module is kept only if the nested score improves and no fold gets worse.
"""
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from src.v6.schedule import Schedule
from src.v7 import core as core7
from . import core, pre, actual

REPORT = [3127, 3226, 3754, 3107, 3136, 3139, 3209, 3121, 3110, 3111]
ISLE = dict(date='2026-09-12', scale=0.6)


def block_events(block, schedule, extra=()):
    """Mark station-days that belong to scheduled or actual events so they are never used as ordinary neighbours."""
    panel = block.panel
    for e in list(schedule.events) + list(schedule.overlay_events()):
        d = pd.Timestamp(e['date'])
        if d in block.dates:
            k = int(np.flatnonzero(block.dates == d)[0])
            for sid in e.get('stations', [core.MUNHAK]):
                block.blocked[k, np.searchsorted(panel.stations, sid)] = True
    for sid, dates in extra:
        for d in dates:
            if d in block.dates:
                block.blocked[int(np.flatnonzero(block.dates == d)[0]), np.searchsorted(panel.stations, sid)] = True


def main(mode='both'):
    panel, folds, test = core.load_blocks()
    train = core7.load_train()
    schedule = core7.schedule_v7()
    history = pre.History(panel)
    use_actual = mode in ('both', 'main')            # the insurance chain never opens a file with realised facts
    ssg, bfc = actual.load_games() if use_actual else (None, None)
    weather = actual.Weather() if use_actual else None
    out = dict(information='local OOF only (nested: every coefficient applied to a fold is fitted on the other three); no leaderboard value is used',
               baseline_v7=core.scores(folds, {k: b.pred for k, b in folds.items()}), chains={})
    out['baseline_v7']['stations'] = core.station_scores(panel, folds, {k: b.pred for k, b in folds.items()}, REPORT)
    rows = []

    def run_chain(name, steps):
        for b in list(folds.values()) + [test]:
            b.blocked[:] = False
            block_events(b, schedule, extra=[(actual.BUCHEON, list(bfc.date))] if name == 'main' else ())
        cur = {k: b.pred.copy() for k, b in folds.items()}; tcur = test.pred.copy()
        prev = core.scores(folds, cur); log = []
        rows.append(dict(chain=name, step='v7', adopted='', **prev, **{f'st_{k}': v for k, v in core.station_scores(panel, folds, cur, REPORT).items()}))
        for label, fn, bounds in steps:
            info = {}

            def feature(block, pred, fn=fn, info=info):
                res = fn(block, pred)
                if isinstance(res, tuple):
                    info[block.name] = res[1]; res = res[0]
                return res
            new, per_fold, c_all, changed = core.apply_module(folds, cur, feature, bounds)
            sc = core.scores(folds, new)
            worse = [k for k in (1, 2, 3, 4) if sc[f'fold_{k}'] > prev[f'fold_{k}'] + 1e-5]
            adopt = sc['all_oof'] < prev['all_oof'] - 1e-6 and not worse
            entry = dict(step=label, before=prev, after=sc, coefficients_nested=per_fold, coefficients_all_folds=c_all,
                         cells_changed=changed, folds_worse=worse, adopted=bool(adopt), details=info)
            if adopt:
                feats = feature(test, tcur)
                delta = sum(c * f for c, f in zip(c_all, feats))
                entry['test_cells_changed'] = int((delta != 0).sum()); entry['test_mean_log_change'] = float(delta[delta != 0].mean()) if (delta != 0).any() else 0.0
                tcur = tcur + delta; cur = new; prev = sc
            rows.append(dict(chain=name, step=label, adopted=bool(adopt), **sc, **{f'st_{k}': v for k, v in core.station_scores(panel, folds, new, REPORT).items()}))
            print(f"[{name}] {label:24s} all {entry['before']['all_oof']:.5f} -> {sc['all_oof']:.5f}  " + ' '.join(f"F{k} {sc[f'fold_{k}']:.5f}" for k in (1, 2, 3, 4))
                  + f"  coef {c_all}  {'ADOPT' if adopt else 'reject'}", flush=True)
            log.append(entry)
        return cur, tcur, log, prev

    pre_steps = [('P1 다년 계절 지수', lambda b, p: [pre.seasonal_feature(history, b, p)], (0.0, 1.5)),
                 ('P2 역별 공휴일 효과', pre.holiday_features, (0.0, 1.2)),
                 ('P3 전년 공휴일 유령 제거', pre.ghost_feature, (0.0, 1.0))]
    main_steps = pre_steps + [
        ('A4 실측 강수', (lambda b, p: weather.features(b, p)), (0.0, 1.5)),
        ('A2 부천 실제 경기', lambda b, p: actual.bucheon_feature(b, p, bfc), (0.0, 1.2)),
        ('A1 문학 실제 경기', lambda b, p: actual.munhak_feature(b, p, train, schedule, ssg), (0.0, 1.2))]

    def save(name, tcur, file):
        flat = np.empty(tcur.size); flat[test.row_index] = np.maximum(tcur, 0).reshape(-1)
        frame = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
        assert len(frame) == 157194 and np.isfinite(frame.boardings).all() and (frame.boardings >= 0).all()
        path = core.OUT / file; frame.to_csv(path, index=False, float_format='%.8f')
        v7 = np.log1p(pd.read_csv(core.ROOT / 'outputs/ensemble_v7/submission_ensemble_v7.csv').boardings.to_numpy())
        d = flat - v7
        return dict(file=file, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(frame)), mean_boardings=float(frame.boardings.mean()),
                    log_rmsd_vs_v7=float(np.sqrt(np.mean(d ** 2))), mean_log_diff_vs_v7=float(d.mean()), rows_changed_vs_v7=int((np.abs(d) > 1e-9).sum()))

    nested = {'v7': {k: b.pred.copy() for k, b in folds.items()}}
    if mode in ('both', 'pre'):
        cur, tcur, log, final = run_chain('precutoff', pre_steps); nested['precutoff'] = cur
        out['chains']['precutoff'] = dict(steps=log, final=final, submission=save('precutoff', tcur, 'submission_precutoff.csv'))
    if mode in ('both', 'main'):
        cur, tcur, log, final = run_chain('main', main_steps); nested['main'] = cur
        # A3: one-day festival at 3139 with no comparable past edition -> scaled festival profile, not validated on folds
        d = pd.Timestamp(ISLE['date']); k = int(np.flatnonzero(test.dates == d)[0]); s = int(np.searchsorted(panel.stations, actual.SONGDO))
        past = [pd.Timestamp(e['date']) for e in schedule.overlay_events() if 3139 in e.get('stations', []) and pd.Timestamp(e['date']) <= pd.Timestamp(core.FINAL) and pd.Timestamp(e['date']).dayofweek == 5]
        test.blocked[k, s] = True
        lift = actual.festival_lift(test, past, ISLE['scale'])
        base = test.neighbour(tcur, k, allow=core.ORDINARY)[s]
        before = tcur[k, s].copy(); tcur[k, s] = base + lift
        log.append(dict(step='A3 이슬라이브 페스티벌(3139, 2026-09-12)', adopted=True, validated=False, scale=ISLE['scale'], past_days=[str(x.date()) for x in past],
                        boardings_before=[round(float(x)) for x in np.expm1(before)], boardings_after=[round(float(x)) for x in np.expm1(tcur[k, s])]))
        out['chains']['main'] = dict(steps=log, final=final, submission=save('main', tcur, 'submission_ensemble_v8.csv'))
        out['weather'] = weather.last_info
    # Sensitivity: the train file has a station x hour level break between 2025 and 2026 (2025 is 0.74-1.00 of the
    # operator's own monthly totals, 2026 equals them).  Folds whose origin is in 2025 are scored here against
    # 2026 truth converted to the 2025 basis, which removes an error no forecast made in 2025 could avoid.
    f = pd.read_csv(core.SHARE / 'ictr_to_train_factor_station_hour.csv'); logf = np.zeros((panel.S, panel.H))
    logf[np.searchsorted(panel.stations, f.station_id.to_numpy()), f.hour.to_numpy() - 6] = np.log(f.factor_boardings.to_numpy())
    saved = {k: b.truth for k, b in folds.items()}
    for k, b in folds.items():
        if b.origin < '2026-01-01':
            b.truth = b.truth + np.where((b.dates.year == 2026)[:, None, None], logf[None], 0.0)
    out['aligned_truth_sensitivity'] = {name: core.scores(folds, p) for name, p in nested.items()}
    for k, b in folds.items():
        b.truth = saved[k]
    np.savez_compressed(core.OUT / 'oof_v8.npz', **{f'{name}_fold{k}': p[k].astype(np.float32) for name, p in nested.items() for k in p})
    (core.OUT / 'gates_v8.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    pd.DataFrame(rows).to_csv(core.OUT / 'VALIDATION_V8.csv', index=False, float_format='%.5f')
    return out


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'both')
