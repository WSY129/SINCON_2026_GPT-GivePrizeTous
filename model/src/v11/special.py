"""Station specialists of v11: separate small LightGBM models for the stations that carry most of the error.

Each specialist sees only its own stations' rows of the unique-day panel (same anchor, same exogenous inputs)
plus inputs that only make sense there:
  munhak  (3127)             exit-flow kernel of the game (attendance x minutes after the end), rain around the
                             scheduled start of a rained-out game (crowd that came and went home)
  bucheon (3754)             the same kernel for football (kick-off + 115 min)
  parks   (3226, 3137)       weekend/holiday x radiation, sunshine, temperature band, rain
  geomdan (3107, 3108, 3109) trend of the new stations; lake-park leisure at 3107 through the weather inputs
A specialist replaces the stacked forecast on a station only if it was better there in the other folds.
"""
import numpy as np
import lightgbm as lgb
from scipy.optimize import nnls
from src.v8 import actual
from .panel import FEATURES, FI

GROUPS = {'munhak': [3127], 'bucheon': [3754], 'parks': [3226, 3137], 'geomdan': [3107, 3108, 3109]}
EXTRA = ['kern', 'rain_start', 'off_day', 'off_rad', 'off_sun', 'off_comfort', 'off_rain']
SEEDS = (11, 12, 13)
PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=25, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, num_threads=8)
ROUNDS = 500


def extra_features(panel, d, s):
    """Extra inputs for station index s: [day, H, len(EXTRA)] over all days of the build."""
    X = d['X'][s]; anchor = d['anchor'][s]; O = d['O']; nd = X.shape[0]
    E = np.zeros((nd, 18, len(EXTRA)), np.float32)
    att = panel.game['att'][s, :nd]; end = panel.game['end'][s, :nd]; start = panel.game['start'][s, :nd]
    played = (att > 0) & ~np.isnan(end)
    if played[:O + 1].sum() >= 10:
        design = [att[t] * actual.overlap(end[t]) for t in np.flatnonzero(played[:O + 1]) if not np.isnan(d['r'][s, t]).any()]
        excess = [np.expm1(np.clip(anchor[t] + d['r'][s, t], 0, None)) - np.expm1(np.clip(anchor[t], 0, None)) for t in np.flatnonzero(played[:O + 1]) if not np.isnan(d['r'][s, t]).any()]
        theta, _ = nnls(np.concatenate(design), np.concatenate(excess))
        for t in np.flatnonzero(played):
            E[t, :, 0] = np.log1p(np.clip(np.expm1(np.clip(anchor[t], 0, None)) + att[t] * actual.overlap(end[t]) @ theta, 0, None)) - anchor[t]
    rain = np.nan_to_num(panel.W['rain'][:nd])
    for t in np.flatnonzero(panel.game['rainout'][s, :nd] > 0):
        h0 = int(np.clip(np.nan_to_num(start[t], nan=1110) // 60, 2, 23))
        E[t, :, 1] = rain[t, h0 - 2:h0 + 1].sum()
    off = (X[:, :, FI['dow']] >= 5) | (X[:, :, FI['cat']] >= 3)
    E[:, :, 2] = off
    for j, name in ((3, 'rad_day'), (4, 'sun_hours'), (5, 'comfort'), (6, 'rain_mm')):
        E[:, :, j] = off * np.nan_to_num(X[:, :, FI[name]])
    return E


def fit_group(panel, d, stations):
    """Residual forecast [day, len(stations), H] for the prediction days of a build."""
    O = d['O']; idx = [int(np.searchsorted(panel.stations, sid)) for sid in stations]
    Xs, ys, ws, Xp = [], [], [], []
    for s in idx:
        E = extra_features(panel, d, s)
        X = np.concatenate([d['X'][s], E], -1)
        r = d['r'][s, :O + 1]; ok = ~np.isnan(r)
        w = np.broadcast_to((0.5 ** (d['age'][:O + 1] / 240.0))[:, None], r.shape)
        Xs.append(X[:O + 1][ok]); ys.append(r[ok]); ws.append(w[ok]); Xp.append(X[O + 1:])
    Xtr = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    Xp = np.stack(Xp, 1)                                                  # [day, station, H, F]
    pred = np.zeros(Xp.shape[:3])
    for seed in SEEDS:
        m = lgb.train(dict(PARAMS, seed=seed), lgb.Dataset(Xtr, y, weight=w), ROUNDS)
        pred += m.predict(Xp.reshape(-1, Xp.shape[-1])).reshape(Xp.shape[:3]) / len(SEEDS)
    anchor = np.moveaxis(d['anchor'][idx][:, O + 1:], 0, 1)
    return np.maximum(anchor + pred, 0)
