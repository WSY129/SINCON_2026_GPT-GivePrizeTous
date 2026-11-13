"""Version-level blend weights from local out-of-fold predictions only.

No leaderboard score enters this module. Each existing ensemble version contributes its
leave-one-fold-out (LOFO) ensemble prediction, i.e. the prediction it made for a fold with its own
internal blender fitted on the other folds. Convex weights (>= 0, sum 1, no intercept) minimise the
log-space MSE pooled over the four time-ordered validation folds, which all have the same number of
rows. Two row segments get their own weights: named holidays and all other rows. Rows of stations
that a specialist replaces are left out of the fit.
"""
import itertools
import numpy as np
import pandas as pd
from ..holiday_calendar import calendar_features

KEY = ['date', 'station_id', 'hour', 'fold']
SOURCES = {
    'v2': ('outputs/ensemble_v2/oof_predictions.csv.gz', 'lofo_ensemble', 'ensemble'),
    'v3': ('outputs/ensemble_v3/oof_predictions.csv.gz', 'lofo_ensemble', 'ensemble'),
    'v4': ('outputs/ensemble_v4/step_b_oof.csv.gz', 'lofo', 'ensemble'),
    'v5': ('outputs/ensemble_v5/step_b_oof.csv.gz', 'lofo', 'ensemble'),
}


def load_oof(root, versions, fold_origins):
    """Merged OOF frame: true_log plus, per version, the LOFO column (name) and the in-sample column (name_fit)."""
    frame = None
    for name in versions:
        path, lofo, fitted = SOURCES[name]
        part = pd.read_csv(root / path, parse_dates=['date'])
        keep = part[KEY + (['true_log', 'horizon'] if frame is None else [])].copy()
        keep[name] = part[lofo].clip(lower=0); keep[name + '_fit'] = part[fitted].clip(lower=0)
        frame = keep if frame is None else frame.merge(keep, on=KEY, how='inner', validate='one_to_one')
    if len(frame) != 4 * 157194: raise ValueError('OOF folds are incomplete')
    holiday = np.zeros(len(frame), bool)
    for fold, origin in enumerate(fold_origins, start=1):       # holidays as they were known at each fold origin
        rows = (frame.fold == fold).to_numpy()
        holiday[rows] = calendar_features(frame.loc[rows, 'date'], as_of=origin).is_holiday.to_numpy(bool)
    frame['is_holiday'] = holiday
    return frame


def simplex(k, step):
    n = int(round(1 / step))
    for combo in itertools.combinations(range(n + k - 1), k - 1):
        parts = np.diff((-1,) + combo + (n + k - 1,)) - 1
        yield parts / n


def fit_convex(errors, step=0.01):
    """Exact grid search of w'Gw on the simplex; G is the k x k second-moment matrix of the errors."""
    gram = errors.T @ errors / len(errors)
    grid = np.array(list(simplex(errors.shape[1], step)))
    value = np.einsum('ij,jk,ik->i', grid, gram, grid)
    best = int(np.argmin(value))
    return grid[best], float(np.sqrt(value[best]))


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def fit(root, cfg):
    spec = cfg['blend']; versions = spec['versions']; step = spec['grid_step']
    frame = load_oof(root, versions, cfg['fold_origins'])
    use = ~frame.station_id.isin(spec['exclude_stations']).to_numpy()
    errors = frame[versions].to_numpy() - frame[['true_log']].to_numpy()
    segment = np.where(frame.is_holiday, 'holiday', 'ordinary')
    report = dict(method='convex least squares on local leave-one-fold-out OOF predictions; four folds pooled with equal rows; no leaderboard input',
                  versions=versions, grid_step=step, excluded_stations=spec['exclude_stations'], segments={})
    weights = {}
    for name in ('ordinary', 'holiday'):
        rows = use & (segment == name)
        w, score = fit_convex(errors[rows], step)
        weights[name] = dict(zip(versions, [float(v) for v in w]))
        lofo = {}
        for fold in (1, 2, 3, 4):                               # weights from the other folds, scored on the held-out fold
            train_rows = rows & (frame.fold != fold).to_numpy(); test_rows = rows & (frame.fold == fold).to_numpy()
            wf, _ = fit_convex(errors[train_rows], step)
            lofo[str(fold)] = dict(weights=dict(zip(versions, [float(v) for v in wf])), rmsle=rmse(errors[test_rows] @ wf),
                                   rmsle_full_fit_weights=rmse(errors[test_rows] @ w),
                                   single=dict(zip(versions, [rmse(errors[test_rows][:, i]) for i in range(len(versions))])))
        report['segments'][name] = dict(rows=int(rows.sum()), weights=weights[name], pooled_rmsle=score,
                                        single_pooled=dict(zip(versions, [rmse(errors[rows][:, i]) for i in range(len(versions))])),
                                        equal_weight_pooled=rmse(errors[rows].mean(axis=1)), leave_one_fold_out=lofo)
    # Reference only: other purely local choices of folds, to show how much the answer depends on that choice.
    rows = use & (segment == 'ordinary'); alt = {}
    for label, folds in {'folds_2_and_4': (2, 4), 'folds_3_and_4': (3, 4), 'folds_1_and_2': (1, 2)}.items():
        w, score = fit_convex(errors[rows & frame.fold.isin(folds).to_numpy()], step)
        alt[label] = dict(weights=dict(zip(versions, [float(v) for v in w])), rmsle_on_those_folds=score, rmsle_all_folds=rmse(errors[rows] @ w))
    report['ordinary_alternatives_reference_only'] = alt
    return weights, report, frame


def apply(frame_or_logs, weights, is_holiday, versions):
    """Blend log predictions with the segment weights. ``frame_or_logs`` maps version -> array."""
    ordinary = sum(weights['ordinary'][v] * np.asarray(frame_or_logs[v]) for v in versions)
    holiday = sum(weights['holiday'][v] * np.asarray(frame_or_logs[v]) for v in versions)
    return np.where(np.asarray(is_holiday, bool), holiday, ordinary)
