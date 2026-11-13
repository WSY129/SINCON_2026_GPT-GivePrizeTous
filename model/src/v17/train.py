"""Two stack candidates asked for in the v17 directive, on the v15 panel (CPU):

    python -m src.v17.train --cutoffs f1,f2,f3,f4,final --models lgb_r,cb_n

lgb_r   LightGBM, Huber loss (alpha 1.0), 127 leaves, at least 400 rows per leaf
cb_n    CatBoost symmetric trees (depth 8, RMSE) on the 74 inputs of v15
Same protocol as v11 / v15: no early stopping, eight checkpoints stored, the checkpoint is chosen from the other folds.
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from src.v11 import train as T
from src.v14.panel import N_FIRST
from src.v15.panel import Panel15

THREADS = int(os.environ.get('V11_THREADS', '12'))
GRID = T.ROUNDS


def fit(D, model, seed):
    X = D.Xtr[:, :N_FIRST]; Xp = D.Xp.reshape(-1, D.Xp.shape[-1])[:, :N_FIRST]
    if model == 'lgb_r':
        m = lgb.train(dict(objective='huber', alpha=1.0, learning_rate=T.LR, num_leaves=127, min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                           num_threads=THREADS, seed=seed, verbose=-1), lgb.Dataset(X, D.r, weight=D.w), GRID[-1])
        predict = lambda k: m.predict(Xp, num_iteration=k, num_threads=THREADS)
    else:
        from catboost import CatBoostRegressor, Pool
        m = CatBoostRegressor(task_type='CPU', depth=8, learning_rate=T.LR, iterations=GRID[-1], l2_leaf_reg=5.0, loss_function='RMSE', random_seed=seed, verbose=0, thread_count=THREADS, allow_writing_files=False)
        m.fit(Pool(X, D.r, weight=D.w)); predict = lambda k: m.predict(Xp, ntree_end=k, thread_count=THREADS)
    out = np.stack([predict(k).reshape(D.shape) for k in GRID]).astype(np.float32)
    return out, [D.score(p) for p in out]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cutoffs', default='f1,f2,f3,f4,final'); ap.add_argument('--models', default='lgb_r,cb_n'); ap.add_argument('--seeds', default='2026')
    args = ap.parse_args(); out = Path('outputs/ensemble_v17/models'); out.mkdir(parents=True, exist_ok=True); panel = Panel15()
    for name in args.cutoffs.split(','):
        D = T.Data(panel, name)
        for seed in [int(s) for s in args.seeds.split(',')]:
            for model in args.models.split(','):
                tag = f'{name}__{model}__{seed}'
                if (out / f'{tag}.npz').exists():
                    continue
                t0 = time.time(); pred, curve = fit(D, model, seed)
                np.savez_compressed(out / f'{tag}.npz', pred=pred, anchor=D.ap.astype(np.float32))
                (out / f'{tag}.json').write_text(json.dumps(dict(model=model, cutoff=name, seed=seed, rounds=GRID, valid_rmsle=curve, seconds=round(time.time() - t0, 1), features=N_FIRST), indent=1, default=float))
                print(f'[{name}] {model} seed {seed} {time.time() - t0:.0f}s valid {[None if c is None else round(c, 4) for c in curve]}', flush=True)


if __name__ == '__main__':
    main()
