"""Local validation machinery of ensemble_v7 (no leaderboard information anywhere).

frame: one row per (fold, date, station, hour) of the four validation folds with the true log value, the
leave-one-fold-out predictions of v2/v4/v5 (read from their saved OOF files) and of the v7 equilibrium models.
Blend weights are convex and are evaluated *nested*: the weights applied to a fold are fitted on the other three.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.v6 import blend as blend6
from src.v6.pipeline import apply_specialists
from src.v6.schedule import Schedule

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/ensemble_v7'
VERSIONS = ['v2', 'v4', 'v5']
MUNHAK = 3127
REPORT_STATIONS = [3127, 3226, 3754, 3107, 3136, 3139, 3209]
FOLD_WEIGHTS = {1: 0.15, 2: 0.15, 3: 0.35, 4: 0.35}


def config():
    return json.loads((ROOT / 'configs/ensemble_v7.json').read_text())


def schedule_v7(cfg=None):
    """v6 schedule plus the v7 additions listed in configs/ensemble_v7.json (written to the v7 output folder)."""
    cfg = cfg or config()
    base = json.loads((ROOT / cfg['schedule_v6']).read_text())
    base['events'] = base['events'] + cfg.get('extra_events', [])
    path = OUT / 'domain_schedule_v7.json'
    path.write_text(json.dumps(base, ensure_ascii=False, indent=1))
    return Schedule(path)


def load_train():
    return pd.read_csv(ROOT / 'data/raw/train.csv', parse_dates=['date'])


def eq_models(folder=None):
    folder = Path(folder or OUT / 'eq')
    names = sorted({p.stem.split('__')[1] for p in folder.glob('f[1-4]__*.npz')})
    return [n for n in names if all((folder / f'f{k}__{n}.npz').exists() for k in range(1, 5))]


def load_frame(cfg=None, models=None, folder=None, key='lofo'):
    cfg = cfg or config()
    folder = Path(folder or OUT / 'eq')
    frame = blend6.load_oof(ROOT, VERSIONS, cfg['fold_origins'])
    frame = frame.sort_values(['fold', 'date', 'station_id', 'hour']).reset_index(drop=True)
    stations = np.sort(frame.station_id.unique())
    si = np.searchsorted(stations, frame.station_id.to_numpy()); hi = frame.hour.to_numpy() - 6
    for name in (eq_models(folder) if models is None else models):
        col = np.full(len(frame), np.nan)
        for k, origin in enumerate(cfg['fold_origins'], start=1):
            rows = np.flatnonzero((frame.fold == k).to_numpy())
            di = (frame.date.to_numpy()[rows] - np.datetime64(origin)).astype('timedelta64[D]').astype(int) - 1
            col[rows] = np.load(folder / f'f{k}__{name}.npz')[key][di, si[rows], hi[rows]]
        frame[name] = col
    return frame


def convex(E):
    """Convex weights minimising the mean squared error of E @ w (E: rows x models of log errors)."""
    m = E.shape[1]
    if m == 1:
        return np.ones(1)
    G = E.T @ E / len(E)
    res = minimize(lambda w: w @ G @ w, np.full(m, 1 / m), jac=lambda w: 2 * G @ w, method='SLSQP',
                   bounds=[(0, 1)] * m, constraints=[dict(type='eq', fun=lambda w: w.sum() - 1, jac=lambda w: np.ones(m))],
                   options=dict(maxiter=500, ftol=1e-12))
    w = np.round(np.clip(res.x, 0, None) / max(res.x.clip(0).sum(), 1e-12), 2)      # 0.01 grid, as in v6
    w[np.argmax(w)] += round(1.0 - w.sum(), 2)
    return w


def fit_weights(frame, cols, holiday_cols=None, rows=None):
    """Ordinary and holiday weights; station 3127 is left out of the fit (its rows are replaced later)."""
    holiday_cols = holiday_cols or [c for c in cols if c in VERSIONS]
    use = (frame.station_id != MUNHAK).to_numpy() if rows is None else rows & (frame.station_id != MUNHAK).to_numpy()
    y = frame.true_log.to_numpy(); hol = frame.is_holiday.to_numpy().astype(bool)
    out = {}
    for seg, mask, names in (('ordinary', use & ~hol, cols), ('holiday', use & hol, holiday_cols)):
        E = frame[names].to_numpy()[mask] - y[mask, None]
        out[seg] = dict(zip(names, convex(E).round(4).tolist()))
    return out


def apply_weights(values, weights, is_holiday):
    """values: dict name -> log predictions (aligned arrays)."""
    hol = np.asarray(is_holiday).astype(bool)
    ordinary = sum(w * values[n] for n, w in weights['ordinary'].items() if w)
    holiday = sum(w * values[n] for n, w in weights['holiday'].items() if w)
    return np.where(hol, holiday, ordinary)


def blend_columns(frame, cols, name, holiday_cols=None):
    """Adds <name>_in (weights fitted on all folds) and <name> (nested: weights from the other folds)."""
    values = {c: frame[c].to_numpy() for c in cols}
    full = fit_weights(frame, cols, holiday_cols)
    frame[name + '_in'] = apply_weights(values, full, frame.is_holiday)
    nested = np.full(len(frame), np.nan); per_fold = {}
    for k in sorted(frame.fold.unique()):
        w = fit_weights(frame, cols, holiday_cols, rows=(frame.fold != k).to_numpy())
        rows = (frame.fold == k).to_numpy(); per_fold[int(k)] = w
        nested[rows] = apply_weights({c: v[rows] for c, v in values.items()}, w, frame.is_holiday.to_numpy()[rows])
    frame[name] = nested
    return full, per_fold


def specialists(frame, col, train, schedule, cfg=None, use=('munhak', 'events'), hook=None):
    """Runs the v6 specialist code on a base column, fold by fold, with the knowledge of each origin."""
    cfg = cfg or config()
    out = np.full(len(frame), np.nan)
    for k, origin in enumerate(cfg['fold_origins'], start=1):
        rows = np.flatnonzero((frame.fold == k).to_numpy()); q = frame.iloc[rows]
        fn = hook or apply_specialists
        out[rows], _ = fn(train, q, q[col].to_numpy(), origin, schedule, use=use)
    return out


def rmse(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float('nan')


def metrics(frame, col, stations=REPORT_STATIONS):
    err2 = (frame[col] - frame.true_log) ** 2
    mse = err2.groupby(frame.fold).mean()
    out = dict(all_oof=float(np.sqrt(err2.mean())),
               weighted_4fold=float(np.sqrt(sum(w * mse[f] for f, w in FOLD_WEIGHTS.items()))),
               fold_3_4=float(np.sqrt(err2[frame.fold.isin([3, 4])].mean())),
               **{f'fold_{f}': float(np.sqrt(mse[f])) for f in (1, 2, 3, 4)})
    for sid in stations:
        out[f'st_{sid}'] = float(np.sqrt(err2[frame.station_id == sid].mean()))
    return out


def gate(frame, before, after):
    """Zero-risk rule: overall strict score and both Fold 2 and Fold 4 must improve."""
    a, b = metrics(frame, before, []), metrics(frame, after, [])
    changed = int((np.abs(frame[after] - frame[before]) > 1e-9).sum())
    passed = b['all_oof'] < a['all_oof'] - 1e-6 and b['fold_2'] < a['fold_2'] - 1e-6 and b['fold_4'] < a['fold_4'] - 1e-6
    return dict(before=a, after=b, rows_changed=changed, adopt=bool(passed))
