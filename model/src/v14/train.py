"""Retrains the LightGBM direct models of v11 on the v14 panel (CPU).

    python -m src.v14.train --cutoffs f1,f2,f3,f4 --models lgb_c,lgb_d
    python -m src.v14.train --cutoffs final --seeds 2026,2027 --models lgb_c,lgb_d

lgb_c = the v11 lgb_b setting (127 leaves), lgb_d = the v11 lgb_a setting (31 leaves); both with the 24 added inputs.
lgb_e / lgb_f = the same two with the two day-level priors added and 6,000 rounds.
Same protocol as v11: no early stopping, forecasts stored at eight checkpoints, checkpoint chosen from the other folds.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from src.v11 import train as T
from .panel import Panel14, FEATURES14

from .panel import N_FIRST
import lightgbm as lgb

LEAVES = {'lgb_c': 127, 'lgb_d': 31, 'lgb_e': 127, 'lgb_f': 31}
ROUNDS_LONG = [700, 1000, 1500, 2200, 3000, 4000, 5000, 6000]


def fit(D, model, seed):
    """lgb_c / lgb_d: v11 settings on the first 74 inputs, 3,000 rounds.  lgb_e / lgb_f: all inputs, 6,000 rounds."""
    long = model in ('lgb_e', 'lgb_f'); n = D.Xtr.shape[1] if long else N_FIRST; X = D.Xtr[:, :n]; Xp = D.Xp.reshape(-1, D.Xp.shape[-1])[:, :n]
    if not long:
        predict = T.tree_fit('lgb', X, D.r, D.w, seed, leaves=LEAVES[model]); grid = T.ROUNDS
    else:
        grid = ROUNDS_LONG
        m = lgb.train(dict(objective='regression', learning_rate=T.LR, num_leaves=LEAVES[model], min_data_in_leaf=100 if LEAVES[model] <= 31 else 200, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=T.THREADS, seed=seed, verbose=-1), lgb.Dataset(X, D.r, weight=D.w), grid[-1])
        predict = lambda Z, k: m.predict(Z, num_iteration=k, num_threads=T.THREADS)
    out = np.stack([predict(Xp, k).reshape(D.shape) for k in grid]).astype(np.float32)
    return out, [D.score(p) for p in out], grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoffs', default='f1,f2,f3,f4'); ap.add_argument('--models', default='lgb_c,lgb_d'); ap.add_argument('--seeds', default='2026')
    ap.add_argument('--out', default='outputs/ensemble_v14/models')
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    panel = Panel14()
    for name in args.cutoffs.split(','):
        t0 = time.time(); D = T.Data(panel, name); print(f'[{name}] panel {time.time() - t0:.1f}s train rows {len(D.r)} features {D.Xtr.shape[1]}', flush=True)
        for seed in [int(s) for s in args.seeds.split(',')]:
            for model in args.models.split(','):
                tag = f'{name}__{model}__{seed}'
                if (out / f'{tag}.npz').exists():
                    continue
                t0 = time.time(); pred, curve, grid = fit(D, model, seed)
                np.savez_compressed(out / f'{tag}.npz', pred=pred, anchor=D.ap.astype(np.float32))
                (out / f'{tag}.json').write_text(json.dumps(dict(model=model, cutoff=name, seed=seed, rounds=grid, valid_rmsle=curve, seconds=round(time.time() - t0, 1),
                                                                n_train=int(len(D.r)), features=int(D.Xtr.shape[1] if model in ('lgb_e', 'lgb_f') else N_FIRST)), indent=1, default=float))
                print(f'[{name}] {model} seed {seed} {time.time() - t0:.0f}s valid {[None if c is None else round(c, 4) for c in curve]}', flush=True)


if __name__ == '__main__':
    main()
