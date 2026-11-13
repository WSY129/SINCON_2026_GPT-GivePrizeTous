"""Members of v15 on the v15 panel (CPU): LightGBM and XGBoost with corrected event inputs, several seeds each.

    python -m src.v15.train --cutoffs f1,f2,f3,f4 --models lgb_h,xgb_h --seeds 2026,2027,2028
    python -m src.v15.train --cutoffs final --models lgb_h,xgb_h --seeds 2026,2027,2028

Same protocol as v11 / v14: no early stopping, eight checkpoints stored, the checkpoint is chosen from the other folds.
All models use the first 74 inputs of the v14 panel (v11's 50 + the 24 added ones).
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from src.v11 import train as T
from src.v14.panel import FEATURES14, N_FIRST
from .panel import Panel15

THREADS = int(os.environ.get('V11_THREADS', '12'))
GRID = T.ROUNDS
BASE = dict(objective='regression', learning_rate=T.LR, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
SPECS = {
    'lgb_h': dict(kind='lgb'),                                                 # the v14 lgb_c setting on the v15 panel (corrected events, bad days out)
    'lgb_i': dict(kind='lgb', params=dict(num_leaves=31, min_data_in_leaf=100)),
    'xgb_h': dict(kind='xgb'),                                                 # v11 XGBoost setting (depth 8)
    'xgb_i': dict(kind='xgb', params=dict(max_depth=10)),                      # deeper trees: better than depth 8 on folds 2, 3 and 4
    'xgb_j': dict(kind='xgb', params=dict(max_depth=10, min_child_weight=200)),
    'xgb_k': dict(kind='xgb', params=dict(objective='reg:pseudohubererror', huber_slope=1.0)),
}


def fit(D, model, seed):
    spec = SPECS[model]; X = D.Xtr[:, :N_FIRST]; Xp = D.Xp.reshape(-1, D.Xp.shape[-1])[:, :N_FIRST]; w = D.w
    if spec.get('half_life'):
        d = D.d; O = d['O']; w = np.broadcast_to((0.5 ** (d['age'][:O + 1] / spec['half_life']))[None, :, None], d['r'][:, :O + 1].shape)[D.ok].astype(np.float32)
    if spec['kind'] == 'lgb':
        cats = [FEATURES14.index(c) for c in spec.get('cat', [])]
        if cats:
            X = X.copy(); Xp = Xp.copy()
            for c in cats:
                X[:, c] = np.nan_to_num(X[:, c], nan=0); Xp[:, c] = np.nan_to_num(Xp[:, c], nan=0)
        m = lgb.train(dict(BASE, **spec.get('params', {}), num_threads=THREADS, seed=seed), lgb.Dataset(X, D.r, weight=w, categorical_feature=cats if cats else 'auto'), GRID[-1])
        predict = lambda k: m.predict(Xp, num_iteration=k, num_threads=THREADS)
    else:
        import xgboost as xgb
        m = xgb.train(dict(dict(tree_method='hist', device='cpu', max_depth=8, eta=T.LR, subsample=0.8, colsample_bytree=0.8, min_child_weight=50, reg_lambda=5.0,
                                objective='reg:squarederror', seed=seed, nthread=THREADS), **spec.get('params', {})), xgb.DMatrix(X, D.r, weight=w, nthread=THREADS), GRID[-1])
        dp = xgb.DMatrix(Xp, nthread=THREADS); predict = lambda k: m.predict(dp, iteration_range=(0, k))
    out = np.stack([predict(k).reshape(D.shape) for k in GRID]).astype(np.float32)
    return out, [D.score(p) for p in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoffs', default='f1,f2,f3,f4'); ap.add_argument('--models', default='lgb_g'); ap.add_argument('--seeds', default='2026')
    ap.add_argument('--out', default='outputs/ensemble_v15/models')
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True); panel = Panel15()
    for name in args.cutoffs.split(','):
        D = T.Data(panel, name)
        for seed in [int(s) for s in args.seeds.split(',')]:
            for model in args.models.split(','):
                tag = f'{name}__{model}__{seed}'
                if (out / f'{tag}.npz').exists() or (Path('outputs/ensemble_v14/models') / f'{tag}.npz').exists():
                    continue
                t0 = time.time(); pred, curve = fit(D, model, seed)
                np.savez_compressed(out / f'{tag}.npz', pred=pred, anchor=D.ap.astype(np.float32))
                (out / f'{tag}.json').write_text(json.dumps(dict(model=model, cutoff=name, seed=seed, rounds=GRID, valid_rmsle=curve, seconds=round(time.time() - t0, 1), n_train=int(len(D.r)),
                                                                features=N_FIRST, spec={k: v for k, v in SPECS[model].items()}), indent=1, default=float))
                print(f'[{name}] {model} seed {seed} {time.time() - t0:.0f}s valid {[None if c is None else round(c, 4) for c in curve]}', flush=True)


if __name__ == '__main__':
    main()
