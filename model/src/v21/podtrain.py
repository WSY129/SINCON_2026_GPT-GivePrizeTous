"""Standalone trainer for the frozen v21 panels (numpy, lightgbm, xgboost only; no project code needed).
    python podtrain.py --panel DIR --out DIR --cutoffs f1,f2 --models lgb_h --seeds 2026 --threads 8
Same protocol as v15: no early stopping, predictions stored at the 8 checkpoints of ROUNDS, validation RMSLE per checkpoint
(aligned truth) recorded so the checkpoint can be chosen from the other folds.  Outputs <cutoff>__<model>__<seed>.npz/.json
in the v15 format (pred [8, day, S, H], anchor)."""
import argparse, json, os, time, hashlib
from pathlib import Path
import numpy as np

ROUNDS = [100, 200, 400, 700, 1000, 1500, 2200, 3000]; LR = 0.03
LGB = dict(objective='regression', learning_rate=LR, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
           verbose=-1, deterministic=True, force_col_wise=True)
XGB = dict(tree_method='hist', max_depth=8, eta=LR, subsample=0.8, colsample_bytree=0.8, min_child_weight=50, reg_lambda=5.0, objective='reg:squarederror')
SPECS = {
    'lgb_h': ('lgb', {}),                                                           # v15 setting on the v21 panel
    'lgb_x': ('lgb', dict(extra_trees=True, feature_fraction=0.5, lambda_l2=20.0, min_data_in_leaf=100)),   # randomised splits, strong column sampling
    'lgb_i': ('lgb', dict(num_leaves=31, min_data_in_leaf=100)),
    'xgb_i': ('xgb', dict(max_depth=10)),
    'xgb_h': ('xgb', {}),
    'xgb_g': ('xgb', dict(max_depth=10, device='cuda')),
    'xgb_g11': ('xgb', dict(max_depth=11, device='cuda')),                  # GPU, depth 11 (the GPU track model)
}


def fit(z, model, seed, threads):
    kind, extra = SPECS[model]; X, y, w, Xp = z['Xtr'], z['r'], z['w'], z['Xp']; shape = tuple(z['shape'])
    if kind == 'lgb':
        import lightgbm as lgb
        m = lgb.train(dict(LGB, **extra, num_threads=threads, seed=seed), lgb.Dataset(X, y, weight=w, free_raw_data=False), ROUNDS[-1])
        pred = [m.predict(Xp, num_iteration=k, num_threads=threads) for k in ROUNDS]
    else:
        import xgboost as xgb
        p = dict(XGB, **extra, seed=seed, nthread=threads)
        m = xgb.train(p, xgb.DMatrix(X, y, weight=w, nthread=threads), ROUNDS[-1]); dp = xgb.DMatrix(Xp, nthread=threads)
        pred = [m.predict(dp, iteration_range=(0, k)) for k in ROUNDS]
    pred = np.stack([q.reshape(shape) for q in pred]).astype(np.float32)
    curve = None
    if 'truth' in z:
        t = z['truth']; curve = [float(np.sqrt(np.mean((np.maximum(z['ap'] + q, 0) - t) ** 2))) for q in pred]
    return pred, curve


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--panel'); ap.add_argument('--out'); ap.add_argument('--cutoffs'); ap.add_argument('--models'); ap.add_argument('--seeds')
    ap.add_argument('--threads', type=int, default=8); ap.add_argument('--drop-last', type=int, default=0); ap.add_argument('--suffix', default=''); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for cut in a.cutoffs.split(','):
        z = dict(np.load(Path(a.panel) / f'{cut}.npz'))
        if a.drop_last:                                                     # ablation: drop the last N inputs (the v21 drift features)
            z['Xtr'] = z['Xtr'][:, :-a.drop_last]; z['Xp'] = z['Xp'][:, :-a.drop_last]
        for model in a.models.split(','):
            for seed in [int(s) for s in a.seeds.split(',')]:
                tag = f'{cut}__{model}{a.suffix}__{seed}'
                if (out / f'{tag}.npz').exists():
                    continue
                t0 = time.time(); pred, curve = fit(z, model, seed, a.threads)
                np.savez_compressed(out / f'{tag}.npz', pred=pred, anchor=z['ap'])
                (out / f'{tag}.json').write_text(json.dumps(dict(model=model, cutoff=cut, seed=seed, rounds=ROUNDS, valid_rmsle=curve, seconds=round(time.time() - t0, 1),
                                                             threads=a.threads, host=os.uname().nodename, spec=SPECS[model][1], n_train=int(len(z['r'])), drop_last=a.drop_last), indent=1))
                print(f'[{cut}] {model} seed {seed} {time.time() - t0:.0f}s valid {None if curve is None else [round(c, 4) for c in curve]}', flush=True)


if __name__ == '__main__':
    main()
