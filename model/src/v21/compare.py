"""Single-model comparison, v21 vs v15, on the local validator (aligned truth, log1p RMSLE).
For each model and fold: predictions averaged over the seeds present, RMSLE at every checkpoint; reports the best checkpoint
and the checkpoint chosen from the other folds (strict, as in the stack).  No leaderboard value is used.
    .venv/bin/python -m src.v21.compare"""
import glob, json, re
from pathlib import Path
import numpy as np

ROOT = Path('.'); PANEL = ROOT / 'outputs/ensemble_v21/panel'
DIRS21 = [ROOT / 'outputs/ensemble_v21/models_mac', ROOT / 'outputs/ensemble_v21/models_pod', ROOT / 'outputs/ensemble_v21/models_gpu']
DIRS15 = [ROOT / 'outputs/ensemble_v15/models', ROOT / 'outputs/ensemble_v14/models']
W = {'f1': 0.15, 'f2': 0.15, 'f3': 0.35, 'f4': 0.35}
_truth = {}


def truth(f):
    if f not in _truth:
        _truth[f] = np.load(PANEL / f'{f}.npz')['truth'].astype(np.float64)
    return _truth[f]


def curve(dirs, model, f):
    files = sorted(p for d in dirs for p in d.glob(f'{f}__{model}__*.npz'))
    if not files:
        return None, 0
    acc = None
    for p in files:
        z = np.load(p); full = np.maximum(z['anchor'][None].astype(np.float64) + z['pred'], 0)
        acc = full if acc is None else acc + full
    acc /= len(files); t = truth(f)
    return np.sqrt(((acc - t[None]) ** 2).mean(axis=(1, 2, 3))), len(files)


def table(dirs, model):
    c = {f: curve(dirs, model, f) for f in W}
    if any(v[0] is None for v in c.values()):
        done = {f: (round(float(v[0].min()), 5), v[1]) for f, v in c.items() if v[0] is not None}
        return dict(partial=done)
    out = {}
    for f in W:
        others = [g for g in W if g != f]; k = int(round(np.mean([int(np.argmin(c[g][0])) for g in others])))
        out[f] = dict(best=round(float(c[f][0].min()), 5), strict=round(float(c[f][0][k]), 5), seeds=c[f][1])
    out['weighted_strict'] = round(float(np.sqrt(sum(W[f] * out[f]['strict'] ** 2 for f in W))), 5)
    return out


if __name__ == '__main__':
    models21 = sorted({re.sub(r'^f\d__|^final__', '', p.stem).rsplit('__', 1)[0] for d in DIRS21 for p in d.glob('f*__*.npz')})
    for m in models21:
        print(m, 'v21', json.dumps(table(DIRS21, m)))
        base = {'lgb_x': 'lgb_h', 'lgb_i': 'lgb_d', 'xgb_g11': 'xgb_i', 'cb_g8': 'xgb_i'}.get(m, m)
        if any(d.glob(f'f1__{base}__*.npz') for d in DIRS15):
            print(m, f'v15 ({base})', json.dumps(table(DIRS15, base)))
