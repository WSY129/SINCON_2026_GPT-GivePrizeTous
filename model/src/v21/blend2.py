"""Second-stage blend of two complete pipelines in log space: w * A + (1 - w) * B, with w chosen for each fold on the
other three folds only (pooled squared error, days inside the scored fold left out), and for the test on all four folds.  Local validator only.
    .venv/bin/python -m src.v21.blend2 <oof_A.npz> <oof_B.npz>"""
import sys, json
import numpy as np
from src.v11 import stack as s11

GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def run(pa, pb, verbose=True):
    S = s11.Stack(); F = list(S.folds); T = S.truth
    A = {k: np.load(pa)[f'fold{k}'].astype(float) for k in F}; B = {k: np.load(pb)[f'fold{k}'].astype(float) for k in F}
    sse = {k: np.array([(((w * A[k] + (1 - w) * B[k]) - T[k]) ** 2).sum() for w in GRID]) for k in F}
    n = {k: T[k].size for k in F}
    dates = {k: set(str(d)[:10] for d in S.folds[k].dates) for k in F}
    def sse_blocked(j, k):                                                   # fold j without the days that fold k also covers (folds overlap by about two months)
        keep = np.array([str(d)[:10] not in dates[k] for d in S.folds[j].dates])
        return np.array([(((w * A[j][keep] + (1 - w) * B[j][keep]) - T[j][keep]) ** 2).sum() for w in GRID])
    nested = {}
    for k in F:
        tot = sum(sse_blocked(j, k) for j in F if j != k); wk = GRID[int(np.argmin(tot))]
        nested[k] = (float(wk), float(np.sqrt(sse[k][list(GRID).index(wk)] / n[k])))
    w_test = float(GRID[int(np.argmin(sum(sse[j] for j in F)))])
    allo = float(np.sqrt(sum(sse[k][list(GRID).index(nested[k][0])] for k in F) / sum(n.values())))
    fw = {1: 0.15, 2: 0.15, 3: 0.35, 4: 0.35}
    w4 = float(np.sqrt(sum(fw[k] * nested[k][1] ** 2 for k in F)))
    f34 = float(np.sqrt((sse[3][list(GRID).index(nested[3][0])] + sse[4][list(GRID).index(nested[4][0])]) / (n[3] + n[4])))
    res = dict(nested={k: dict(w_A=v[0], rmsle=round(v[1], 5)) for k, v in nested.items()}, w_test=w_test, all_oof=round(allo, 5), weighted_4fold=round(w4, 5), fold_3_4=round(f34, 5), fold_4=round(nested[4][1], 5))
    if verbose:
        print(json.dumps(res, indent=1))
        for w in (0.0, 0.5, 0.6, 0.7, 1.0):
            i = list(GRID).index(w); print('fixed w', w, {k: round(float(np.sqrt(sse[k][i] / n[k])), 5) for k in F})
    return res


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
