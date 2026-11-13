"""Writes outputs/ensemble_v21/final/: the validated v21 = w * (v21 chain) + (1 - w) * v19 (= v20e base) in log space.
Fold k uses the weight chosen on the other three folds (src.v21.blend2); the test uses the weight chosen on all four.
    V21_CHAIN=<chain folder> .venv/bin/python -m src.v21.final"""
import hashlib, json, os
from pathlib import Path
import numpy as np, pandas as pd
from src.v8 import core as c8
from src.v11 import stack as s11
from . import blend2

CHAIN = c8.ROOT / os.environ.get('V21_CHAIN', 'outputs/ensemble_v21/chain_all_tol1.5e-4_lgb_hk+xgb_ik+xgb_g11k')
V19 = c8.ROOT / 'outputs/ensemble_v19'
OUT = c8.ROOT / 'outputs/ensemble_v21/final'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = blend2.run(CHAIN / 'v19/oof_v19.npz', V19 / 'oof_v19.npz', verbose=False)
    S = s11.Stack(); F = list(S.folds)
    A = np.load(CHAIN / 'v19/oof_v19.npz'); B = np.load(V19 / 'oof_v19.npz')
    oof = {f'fold{k}': (res['nested'][k]['w_A'] * A[f'fold{k}'].astype(float) + (1 - res['nested'][k]['w_A']) * B[f'fold{k}'].astype(float)).astype(np.float32) for k in F}
    np.savez_compressed(OUT / 'oof_v21.npz', **oof)
    sa = pd.read_csv(CHAIN / 'v19/submission_ensemble_v19.csv'); sb = pd.read_csv(V19 / 'submission_ensemble_v19.csv'); assert (sa.example_id == sb.example_id).all()
    w = res['w_test']; z = w * np.log1p(sa.boardings.to_numpy()) + (1 - w) * np.log1p(sb.boardings.to_numpy())
    sub = pd.DataFrame({'example_id': sa.example_id, 'boardings': np.expm1(np.maximum(z, 0))})
    assert len(sub) == 157194 and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all()
    path = OUT / 'submission_ensemble_v21.csv'; sub.to_csv(path, index=False, float_format='%.8f')
    sc = S.score({k: oof[f'fold{k}'].astype(float) for k in F}); sc_raw = S.score({k: oof[f'fold{k}'].astype(float) for k in F}, raw=True)
    v20e = dict(all_oof=0.16541, weighted_4fold=0.16333, fold_3_4=0.16017, fold_4=0.15891, fold_1=0.1803, fold_2=0.16005, fold_3=0.16143)
    out = dict(information='v21 = nested blend of the v21 chain and v19 (the base of v20e); local validator only, no leaderboard value used',
               chain=str(CHAIN.relative_to(c8.ROOT)), weights=res, aligned=sc, raw_truth=sc_raw, v20e_aligned=v20e,
               folds_worse_than_v20e={k: round(sc[k] - v, 6) for k, v in v20e.items() if k.startswith('fold_') and sc[k] > v + 1e-6},
               submission=dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(sub), mean_boardings=float(sub.boardings.mean())))
    (OUT / 'gates_v21.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('aligned', 'folds_worse_than_v20e')}, indent=1, default=float)); print(out['submission'])


if __name__ == '__main__':
    main()
