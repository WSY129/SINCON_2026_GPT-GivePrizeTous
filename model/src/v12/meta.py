"""v12 meta-learning pieces on top of the stored v11 forecasts (CPU only; no retraining of the base models).

smooth      average of the neighbouring stored checkpoints (0.25 / 0.5 / 0.25 around the checkpoint chosen from the
            other folds) instead of a single checkpoint
segments    0 Munhak station, 1 holiday categories, 2 low-demand cells (v10 forecast < 40 boardings),
            3 weather days (rain, heat wave >= 33 C, or a dry sunny off-day), 4 ordinary cells
two stages  convex weights on all cells, then convex weights per segment with a ridge towards the stage-1 weights
EB gate     per station, a continuous weight between the stack and v10, shrunk towards the stack
"""
import numpy as np
from scipy.optimize import minimize
from src.v8 import core as c8
from src.v11 import stack as s11

SEGMENTS = {0: '문학경기장', 1: '공휴일·명절·연휴', 2: '저수요(40명 미만)', 3: '강수·폭염·쾌청 여가', 4: '일반'}
KERNEL = (0.25, 0.5, 0.25)


def smooth(pred, i):
    idx = [j for j in (i - 1, i, i + 1) if 0 <= j < pred.shape[0]]
    w = np.array([KERNEL[j - i + 1] for j in idx]); w = w / w.sum()
    return sum(wi * pred[j] for wi, j in zip(w, idx))


def load_fold(name, k, smoothed=True):
    d = np.load(s11.MODELS / f'{s11.FOLDS[k]}__{name}__2026.npz')
    i = min(s11.strict_index(name, k), d['pred'].shape[0] - 1)
    return np.maximum(d['anchor'] + (smooth(d['pred'], i) if smoothed else d['pred'][i]), 0).astype(float)


def load_final(name, smoothed=True):
    out = []
    for p in sorted(s11.MODELS.glob(f'final__{name}__*.npz')):
        d = np.load(p); i = min(s11.strict_index(name), d['pred'].shape[0] - 1)
        out.append(np.maximum(d['anchor'] + (smooth(d['pred'], i) if smoothed else d['pred'][i]), 0))
    return np.mean(out, 0).astype(float)


def segments(block, v10, wx):
    """Segment id [day, S, H] of a block; wx = Panel11 (weather arrays by day index)."""
    di = block.di
    seg = np.full(v10.shape, 4, np.int8)
    off = block.kind != 'wd'
    weather = (wx.wet[di] > 0) | (wx.tmax[di] >= 33) | (off & (wx.rain_d[di] < 1) & (wx.sun_hours[di] >= 8))
    seg[weather] = 3
    seg[v10 < np.log1p(40.0)] = 2
    seg[np.isin(block.kind, c8.SPECIAL)] = 1
    seg[:, int(np.searchsorted(block.panel.stations, c8.MUNHAK))] = 0
    return seg


def simplex_ridge(E, lam=0.0, centre=None):
    """Non-negative weights summing to one that minimise mean squared error + lam * |w - centre|^2."""
    m = E.shape[1]; G = E.T @ E / max(len(E), 1); c = np.full(m, 1.0 / m) if centre is None else centre
    res = minimize(lambda w: w @ G @ w + lam * ((w - c) ** 2).sum(), c, jac=lambda w: 2 * G @ w + 2 * lam * (w - c), method='SLSQP', bounds=[(0, 1)] * m,
                   constraints=[dict(type='eq', fun=lambda w: w.sum() - 1, jac=lambda w: np.ones(m))], options=dict(maxiter=500, ftol=1e-13))
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def fit_two_stage(errors, seg, lam2):
    """errors [n, members], seg [n] -> (stage-1 weights, {segment: weights})."""
    w1 = simplex_ridge(errors)
    return w1, {sid: (simplex_ridge(errors[seg == sid], lam2, w1) if (seg == sid).sum() > 500 else w1) for sid in SEGMENTS}


def apply_two_stage(preds, seg, weights):
    """preds: list of member arrays; seg: array of the same shape."""
    out = np.zeros_like(preds[0])
    for sid, w in weights.items():
        m = seg == sid
        if m.any():
            out[m] = sum(wi * p[m] for wi, p in zip(w, preds))
    return out


def eb_station_weights(stack, base, truth, strength=0.25):
    """alpha[S] in [0, 1]: forecast = base + alpha * (stack - base); least squares per station shrunk towards 1."""
    num = sum(((t - b) * (s - b)).sum(axis=(0, 2)) for s, b, t in zip(stack, base, truth))
    den = sum(((s - b) ** 2).sum(axis=(0, 2)) for s, b in zip(stack, base))
    kappa = strength * float(np.median(den[den > 0]))
    return np.clip((num + kappa) / (den + kappa), 0.0, 1.0)
