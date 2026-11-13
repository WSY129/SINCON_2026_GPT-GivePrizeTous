"""Shared machinery of ensemble_v8: dense fold/test arrays on top of the v7 forecasts, day categories as known
at an origin, and nested (leave-one-fold-out) fitting of module coefficients.  No leaderboard value is used.

v8 is a chain of corrections applied to the v7 forecast: every module maps (history <= origin, current
forecast) -> one or more "feature" arrays (proposed log changes); a scalar coefficient per feature is fitted on
local out-of-fold residuals of the other folds.
"""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from src.holiday_calendar import holiday_records, calendar_features
from src.v7.panel import Panel, START

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/ensemble_v8'
EXT = OUT / 'external'
SHARE = EXT / 'team_share'
ORIGINS = ['2025-08-30', '2025-10-30', '2025-12-30', '2026-02-27']
FINAL = '2026-06-30'
MUNHAK = 3127
SPECIAL = ('hol_wd', 'hol_sat', 'lunar0', 'lunar_adj', 'newyear', 'labor', 'bridge', 'long_sat', 'long_sun')
ORDINARY = ('wd', 'sat', 'sun')
FOLD_WEIGHTS = {1: 0.15, 2: 0.15, 3: 0.35, 4: 0.35}


def nanmed(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmedian(a, axis=axis)


def nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmean(a, axis=axis)


def day_kinds(dates, origin):
    """Category of each date using only holiday facts published by `origin`."""
    dates = pd.DatetimeIndex(dates)
    rec = holiday_records(); rec = rec[rec.known_since <= str(origin)[:10]]
    known = {pd.Timestamp(d): (k, n) for d, k, n in zip(rec.date, rec.kind, rec.name)}
    bridge = calendar_features(pd.Series(dates), as_of=str(origin)[:10]).is_bridge_day.to_numpy()
    out = []
    for d, b in zip(dates, bridge):
        if d in known:
            kind, name = known[d]
            if kind in ('seollal', 'chuseok'):
                out.append('lunar0' if name in ('설날', '추석') else 'lunar_adj')
            elif kind == 'new_year':
                out.append('newyear')
            elif d.dayofweek == 6:
                out.append('sun')
            elif d.dayofweek == 5:
                out.append('hol_sat')
            else:
                out.append('hol_wd')
        elif d.month == 5 and d.day == 1 and d.dayofweek < 5:
            out.append('labor')
        elif b:
            out.append('bridge')
        else:
            out.append('wd' if d.dayofweek < 5 else ('sat' if d.dayofweek == 5 else 'sun'))
    # weekend days inside a long weekend made by a Friday or Monday holiday
    holiday = {d for d in dates if d in known}
    one = pd.Timedelta(days=1)
    for i, d in enumerate(dates):
        if out[i] == 'sat' and ((d - one) in known or (d + 2 * one) in known):
            out[i] = 'long_sat'
        elif out[i] == 'sun' and d not in known and ((d + one) in known or (d - 2 * one) in known or (d - one) in known):
            out[i] = 'long_sun'
    return np.array(out)


class Block:
    """One forecast block (a validation fold or the test window) as dense arrays [day, station, hour]."""

    def __init__(self, panel, origin, dates, pred, truth=None, name=''):
        self.panel, self.origin, self.name = panel, str(origin)[:10], name
        self.dates = pd.DatetimeIndex(dates)
        self.O = panel.day_index(origin)
        self.di = np.array([panel.day_index(d) for d in self.dates])
        self.pred = pred.astype(float); self.truth = truth
        self.kind = day_kinds(self.dates, origin)
        self.hist_kind = day_kinds(panel.days[:self.O + 1], origin)
        self.horizon = self.di - self.O
        self.blocked = np.zeros((len(self.dates), panel.S), bool)        # station-days owned by event modules

    @property
    def Z(self):
        return self.panel.Z[:, :self.O + 1]                              # history up to the origin only

    def neighbour(self, pred, k, allow=ORDINARY, offsets=(7, 14, 21, 28), ok=None):
        """Median forecast of nearby same-weekday ordinary days around block day k -> (S, H)."""
        good = np.isin(self.kind, allow) if ok is None else ok
        idx = [k + s * o for o in offsets for s in (-1, 1) if 0 <= k + s * o < len(self.dates) and good[k + s * o]]
        idx = idx[:4] if len(idx) >= 4 else idx
        if not idx:
            return np.full(pred.shape[1:], np.nan)
        stack = pred[idx].copy()
        stack[self.blocked[idx]] = np.nan
        return nanmed(stack, 0)


def load_blocks(panel=None):
    """v7 strict out-of-fold forecasts of the four folds and the v7 test forecast as Blocks."""
    panel = panel or Panel(ROOT)
    oof = pd.read_pickle(ROOT / 'outputs/ensemble_v7/oof_v6_v7.pkl').sort_values(['fold', 'date', 'station_id', 'hour'])
    S, H = panel.S, panel.H
    folds = {}
    for k, origin in enumerate(ORIGINS, start=1):
        g = oof[oof.fold == k]
        dates = pd.DatetimeIndex(sorted(g.date.unique()))
        assert len(g) == len(dates) * S * H
        folds[k] = Block(panel, origin, dates, g.v7_strict.to_numpy().reshape(len(dates), S, H),
                         g.true_log.to_numpy().reshape(len(dates), S, H), name=f'fold{k}')
    test = pd.read_csv(ROOT / 'data/raw/test.csv', parse_dates=['date'])
    sub = pd.read_csv(ROOT / 'outputs/ensemble_v7/submission_ensemble_v7.csv')
    test['pred'] = np.log1p(sub.boardings.to_numpy())
    order = test.sort_values(['date', 'station_id', 'hour'])
    dates = pd.DatetimeIndex(sorted(test.date.unique()))
    tb = Block(panel, FINAL, dates, order.pred.to_numpy().reshape(len(dates), S, H), name='test')
    tb.row_index = order.index.to_numpy()                                 # position of each dense cell in test.csv
    tb.example_id = test.example_id.to_numpy()
    return panel, folds, tb


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def scores(folds, preds):
    """Local validator numbers for a dict fold -> dense forecast."""
    mse = {k: float(np.mean((preds[k] - b.truth) ** 2)) for k, b in folds.items()}
    n = {k: b.truth.size for k, b in folds.items()}
    out = dict(all_oof=float(np.sqrt(sum(mse[k] * n[k] for k in mse) / sum(n.values()))),
               weighted_4fold=float(np.sqrt(sum(FOLD_WEIGHTS[k] * mse[k] for k in mse))),
               fold_3_4=float(np.sqrt((mse[3] * n[3] + mse[4] * n[4]) / (n[3] + n[4]))))
    out.update({f'fold_{k}': float(np.sqrt(v)) for k, v in mse.items()})
    return out


def station_scores(panel, folds, preds, stations):
    out = {}
    for sid in stations:
        s = int(np.searchsorted(panel.stations, sid))
        e = np.concatenate([(preds[k][:, s] - b.truth[:, s]).ravel() for k, b in folds.items()])
        out[int(sid)] = rmse(e)
    return out


def fit_coefficients(feats, resid, masks):
    """Least squares c minimising |resid + sum_j c_j feat_j| over the given folds (lists of dense arrays)."""
    m = len(feats[0])
    A = np.zeros((m, m)); b = np.zeros(m)
    for f, r, ok in zip(feats, resid, masks):
        X = np.stack([x[ok] for x in f], 1)
        A += X.T @ X; b += X.T @ (-r[ok])
    c = np.zeros(m)
    use = np.diag(A) > 1e-9
    if use.any():
        c[use] = np.linalg.solve(A[np.ix_(use, use)] + 1e-9 * np.eye(int(use.sum())), b[use])
    return c


def apply_module(folds, cur, feature_fn, bounds=(0.0, 1.5), mask_fn=None):
    """Nested application of one module.
    feature_fn(block, pred) -> list of dense feature arrays (0 where the module does not act).
    Returns (new forecasts per fold, coefficients per held-out fold, coefficients fitted on all folds)."""
    feats = {k: feature_fn(b, cur[k]) for k, b in folds.items()}
    resid = {k: cur[k] - b.truth for k, b in folds.items()}
    masks = {k: (np.ones(b.truth.shape, bool) if mask_fn is None else mask_fn(b)) for k, b in folds.items()}
    new, per_fold = {}, {}
    for k in folds:
        others = [j for j in folds if j != k]
        c = np.clip(fit_coefficients([feats[j] for j in others], [resid[j] for j in others], [masks[j] for j in others]), *bounds)
        new[k] = cur[k] + sum(ci * f for ci, f in zip(c, feats[k])); per_fold[k] = [round(float(x), 4) for x in c]
    c_all = np.clip(fit_coefficients(list(feats.values()), list(resid.values()), list(masks.values())), *bounds)
    changed = {k: int(sum((f != 0) for f in feats[k]).astype(bool).sum()) for k in folds}
    return new, per_fold, [round(float(x), 4) for x in c_all], changed
