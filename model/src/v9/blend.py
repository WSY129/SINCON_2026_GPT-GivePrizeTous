"""Blend the v9 structural-anchor models with the v8 chains and build the v9 submissions (local OOF only).

    python -m src.v9.blend         # needs outputs/ensemble_v9/models (strict fold predictions and finals)

main      : v8 main forecast  +  v9 models trained with observed weather and game inputs
precutoff : v8 insurance forecast  +  v9 models whose names end in _w0 (no realised fact of the forecast period)

Scoring basis.  The train file changes aggregation basis on 2026-01-01.  v9 history is converted to the 2026
basis; forecasts made by v2..v8 at an origin in 2025 are on the 2025 basis.  All fold comparisons here are done
on one consistent basis per fold ("aligned"): folds with a 2025 origin are scored on the 2025 basis (2026 truth
converted with the station x hour factor), Fold 4 on the 2026 basis.  Raw-truth numbers are reported next to them.
Rows of the Munhak station and of scheduled/actual event station-days keep the v8 value.
"""
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from src.v7 import core as core7
from src.v8 import core as c8, actual
from src.v8.run import block_events

OUT = c8.ROOT / 'outputs/ensemble_v9'
MODELS = OUT / 'models'


def fold_models():
    names = sorted({p.stem.split('__')[1] for p in MODELS.glob('f[1-4]__*.npz')})
    return [n for n in names if all((MODELS / f'f{k}__{n}.npz').exists() and 'lofo' in np.load(MODELS / f'f{k}__{n}.npz').files for k in range(1, 5))]


def main():
    panel, folds, test = c8.load_blocks()
    schedule = core7.schedule_v7()
    _, bfc = actual.load_games()
    f = pd.read_csv(c8.SHARE / 'ictr_to_train_factor_station_hour.csv'); logf = np.zeros((panel.S, panel.H))
    logf[np.searchsorted(panel.stations, f.station_id.to_numpy()), f.hour.to_numpy() - 6] = np.log(f.factor_boardings.to_numpy())
    oof8 = np.load(c8.OUT / 'oof_v8.npz')
    names = fold_models()
    munhak = int(np.searchsorted(panel.stations, c8.MUNHAK))
    out = dict(information='local OOF only; nested convex weights (weights applied to a fold are fitted on the other three); no leaderboard value is used',
               models=names, chains={})
    raw_truth = {k: b.truth.copy() for k, b in folds.items()}
    early = {k: b.origin < '2026-01-01' for k, b in folds.items()}
    aligned_truth = {k: (raw_truth[k] + np.where((b.dates.year == 2026)[:, None, None], logf[None], 0.0)) if early[k] else raw_truth[k] for k, b in folds.items()}
    v9 = {n: {k: np.load(MODELS / f'f{k}__{n}.npz')['lofo'].astype(float) + (logf[None] if early[k] else 0.0) for k in folds} for n in names}
    test8 = {'main': pd.read_csv(c8.OUT / 'submission_ensemble_v8.csv'), 'precutoff': pd.read_csv(c8.OUT / 'submission_precutoff.csv')}

    def score(pred, truth):
        mse = {k: float(np.mean((pred[k] - truth[k]) ** 2)) for k in folds}
        n = {k: truth[k].size for k in folds}
        return dict(all_oof=float(np.sqrt(sum(mse[k] * n[k] for k in folds) / sum(n.values()))),
                    weighted_4fold=float(np.sqrt(sum(c8.FOLD_WEIGHTS[k] * mse[k] for k in folds))),
                    fold_3_4=float(np.sqrt((mse[3] * n[3] + mse[4] * n[4]) / (n[3] + n[4]))), **{f'fold_{k}': float(np.sqrt(v)) for k, v in mse.items()})

    def to_raw(pred):            # forecasts of early folds are on the 2025 basis; 2026 rows get the break back
        return {k: pred[k] - (np.where((b.dates.year == 2026)[:, None, None], logf[None], 0.0) if early[k] else 0.0) for k, b in folds.items()}

    out['single_models_aligned'] = {n: score(v9[n], aligned_truth) for n in names}
    for chain in ('main', 'precutoff'):
        members = [n for n in names if n.endswith('_w0') == (chain == 'precutoff')]
        if not members:
            continue
        for b in list(folds.values()) + [test]:
            b.blocked[:] = False
            block_events(b, schedule, extra=[(actual.BUCHEON, list(bfc.date))])
        base = {k: oof8[f'{chain}_fold{k}'].astype(float) for k in folds}
        free = {}
        for k, b in folds.items():
            m = ~np.broadcast_to(b.blocked[:, :, None], b.truth.shape).copy(); m[:, munhak] = False
            free[k] = m
        cols = ['v8'] + members
        weather = actual.Weather() if chain == 'main' else None

        def segments(b):
            """0 ordinary dry day, 1 wet day (observed rain; main chain only), 2 special day (holiday categories)."""
            seg = np.where(np.isin(b.kind, c8.SPECIAL), 2, 0)
            if weather is not None:
                seg = np.where((seg == 0) & (weather.level(b.dates) > 0), 1, seg)
            return np.broadcast_to(seg[:, None, None], b.pred.shape)
        seg = {k: segments(b) for k, b in folds.items()}; seg_test = segments(test)
        ids = sorted(set(np.unique(seg_test)) | {int(x) for k in folds for x in np.unique(seg[k])})

        def errors(k, sid):
            m = free[k] & (seg[k] == sid)
            return np.stack([base[k][m] - aligned_truth[k][m]] + [v9[n][k][m] - aligned_truth[k][m] for n in members], 1)

        def fit(train_folds):
            return {sid: core7.convex(np.concatenate([errors(j, sid) for j in train_folds])) for sid in ids}

        def mix(w, k=None, dense8=None, finals=None):
            src8 = base[k] if k is not None else dense8
            src9 = [v9[n][k] for n in members] if k is not None else finals
            sg = seg[k] if k is not None else seg_test
            out = np.zeros_like(src8)
            for sid in ids:
                out += (sg == sid) * (w[sid][0] * src8 + sum(wi * p for wi, p in zip(w[sid][1:], src9)))
            return out
        blended, weights = {}, {}
        for k in folds:                       # standard protocol, reported for reference: weights from the other three folds
            w = fit([j for j in folds if j != k]); weights[k] = {int(sid): dict(zip(cols, [round(float(x), 2) for x in w[sid]])) for sid in ids}
            blended[k] = np.where(free[k], mix(w, k=k), base[k])
        # Regime-matched protocol.  In Folds 1-2 the base versions had no prior-year observation of the target months,
        # so weights fitted there over-weight v9; the test, like Folds 3-4, has prior-year observations.  Weights are
        # therefore fitted on Folds 3+4, checked by swapping them between Fold 3 and Fold 4, and checked on Folds 1-2.
        matched = {4: np.where(free[4], mix(fit([3]), k=4), base[4]), 3: np.where(free[3], mix(fit([4]), k=3), base[3])}
        w_all = fit([3, 4])
        for k in (1, 2):
            matched[k] = np.where(free[k], mix(w_all, k=k), base[k])
        res = dict(members=members, weights_standard_nested=weights,
                   weights_all_folds={int(sid): dict(zip(cols, [round(float(x), 2) for x in w_all[sid]])) for sid in ids},
                   weights_from='Fold 3 + Fold 4', segments={0: '평소 맑은 날', 1: '비 온 날(주력만)', 2: '공휴일·명절·연휴 범주'},
                   aligned=dict(v8=score(base, aligned_truth), v9_blend=score(matched, aligned_truth), standard_nested=score(blended, aligned_truth)),
                   raw_truth=dict(v8=score(base, raw_truth), v9_blend=score(to_raw(matched), raw_truth)),
                   protocol='Fold 4 <- weights of Fold 3; Fold 3 <- weights of Fold 4; Folds 1, 2 <- weights of Folds 3+4',
                   stations_aligned={int(s): dict(v8=c8.rmse(np.concatenate([(base[k][:, i] - aligned_truth[k][:, i]).ravel() for k in folds])),
                                                  v9_blend=c8.rmse(np.concatenate([(matched[k][:, i] - aligned_truth[k][:, i]).ravel() for k in folds])))
                                     for s, i in ((s, int(np.searchsorted(panel.stations, s))) for s in (3226, 3754, 3107, 3136, 3139, 3209, 3110, 3111))})
        np.savez_compressed(OUT / f'oof_v9_{chain}.npz', **{f'fold{k}': matched[k].astype(np.float32) for k in folds})
        a, b2 = res['aligned']['v8'], res['aligned']['v9_blend']
        worse = [k for k in (1, 2, 3, 4) if b2[f'fold_{k}'] > a[f'fold_{k}'] + 1e-5]
        gain34 = min(a['fold_3'] - b2['fold_3'], a['fold_4'] - b2['fold_4'])          # the two test-like folds must both gain visibly
        res['adopted'] = bool(b2['all_oof'] < a['all_oof'] - 1e-6 and not worse and gain34 >= 1e-4); res['folds_worse'] = worse; res['min_gain_fold_3_4'] = float(gain34)
        sn = res['aligned']['standard_nested']
        print(f"[{chain}] aligned all {a['all_oof']:.5f} -> {b2['all_oof']:.5f}  " + ' '.join(f"F{k} {a[f'fold_{k}']:.5f}->{b2[f'fold_{k}']:.5f}" for k in (1, 2, 3, 4))
              + f"  weights {res['weights_all_folds']}  {'ADOPT' if res['adopted'] else 'reject'}  (standard nested: all {sn['all_oof']:.5f} F3 {sn['fold_3']:.5f} F4 {sn['fold_4']:.5f})", flush=True)
        # ---- test forecast --------------------------------------------------------------------------
        finals = {n: sorted(MODELS.glob(f'final__{n}__*.npz')) for n in members}
        if res['adopted'] and all(len(v) >= 2 for v in finals.values()):
            t8 = np.empty(test.pred.size); t8[:] = np.log1p(test8[chain].boardings.to_numpy())
            dense8 = t8[test.row_index].reshape(test.pred.shape)
            mixed = mix(w_all, dense8=dense8, finals=[np.mean([np.load(p)['pred'] for p in finals[n]], 0) for n in members])
            m = ~np.broadcast_to(test.blocked[:, :, None], dense8.shape).copy(); m[:, munhak] = False
            if chain == 'main':
                d = pd.Timestamp('2026-09-12'); m[int(np.flatnonzero(test.dates == d)[0]), int(np.searchsorted(panel.stations, 3139))] = False
            dense = np.where(m, mixed, dense8)
            flat = np.empty(dense.size); flat[test.row_index] = np.maximum(dense, 0).reshape(-1)
            frame = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
            assert len(frame) == 157194 and np.isfinite(frame.boardings).all() and (frame.boardings >= 0).all()
            name = 'submission_ensemble_v9.csv' if chain == 'main' else 'submission_precutoff_v9.csv'
            frame.to_csv(OUT / name, index=False, float_format='%.8f')
            diff = flat - t8
            res['submission'] = dict(file=name, sha256=hashlib.sha256((OUT / name).read_bytes()).hexdigest(), rows=int(len(frame)), mean_boardings=float(frame.boardings.mean()),
                                     log_rmsd_vs_v8=float(np.sqrt(np.mean(diff ** 2))), mean_log_diff_vs_v8=float(diff.mean()),
                                     final_files={n: [p.name for p in v] for n, v in finals.items()})
            print(f"   wrote {name}: mean {frame.boardings.mean():.2f}, log RMSD vs v8 chain {res['submission']['log_rmsd_vs_v8']:.4f}", flush=True)
        out['chains'][chain] = res
    (OUT / 'blend_v9.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return out


if __name__ == '__main__':
    main()
