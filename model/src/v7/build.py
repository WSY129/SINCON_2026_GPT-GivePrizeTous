"""Build the single v7 submission from the modules that passed the local gates.

    python -m src.v7.build        # needs outputs/ensemble_v7/gates.json (python -m src.v7.gates)

Base = convex log-space blend (weights fitted on all four local folds) of the v2/v4/v5 submissions and, if the
equilibrium family was adopted, the 2-seed averages of its final models; then the v6 specialist code with the
v7 schedule additions; then the new-station growth drift.  No leaderboard score is read.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.holiday_calendar import calendar_features
from src.v6.schedule import Schedule, STATE_NAMES
from . import core, growth
from .gates import FOOTBALL
from .panel import Panel
from .pipeline import apply_specialists, MunhakV7
from src.v6.munhak import MunhakSpecialist


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def eq_test(test, name, folder, origin):
    files = sorted(folder.glob(f'final__{name}__*.npz'))
    if len(files) < 2:
        raise FileNotFoundError(f'two final seeds are required for {name}, found {len(files)}')
    pred = np.mean([np.load(p)['pred'] for p in files], 0)
    stations = np.sort(test.station_id.unique())
    di = (test.date.to_numpy() - np.datetime64(origin)).astype('timedelta64[D]').astype(int) - 1
    return pred[di, np.searchsorted(stations, test.station_id.to_numpy()), test.hour.to_numpy() - 6], [p.name for p in files]


def forecast(cfg, gates, train, test, origin, panel, steps=None):
    """Returns (log forecast, details).  `steps` defaults to the adopted modules."""
    steps = gates['adopted'] if steps is None else steps
    weights = gates['final_coefficients_all_folds']['blend'] if steps == gates['adopted'] else None
    if weights is None:
        raise ValueError('forecast() is built for the adopted configuration')
    values, used = {}, {}
    for v, path in cfg['base_submissions'].items():
        values[v] = np.log1p(pd.read_csv(core.ROOT / path).boardings.to_numpy())
    for name, w in weights['ordinary'].items():
        if name not in values and w:
            values[name], used[name] = eq_test(test, name, core.OUT / 'eq', origin)
    is_holiday = calendar_features(test.date, as_of=origin).is_holiday.to_numpy(bool)
    base = core.apply_weights(values, weights, is_holiday)
    schedule = core.schedule_v7(cfg) if 'beer' in steps else Schedule(core.ROOT / cfg['schedule_v6'])
    out, details = apply_specialists(train, test, base, origin, schedule,
                                     munhak_cls=MunhakV7 if 'monday' in steps else MunhakSpecialist,
                                     overlay=FOOTBALL if 'bucheon' in steps else None)
    details.update(base=base, after_specialists=out.copy(), weights=weights, eq_files=used)
    if 'growth' in steps:
        slope = growth.slope_column(panel, test, origin, **growth.VARIANTS[0])
        horizon = (test.date.to_numpy() - np.datetime64(origin)).astype('timedelta64[D]').astype(int).astype(float)
        add = growth.correction(slope, horizon, growth.LAG, growth.PERSISTENCE)
        out = out + add
        details['growth'] = dict(slopes=growth.slopes(panel, origin, **growth.VARIANTS[0]), rows=int((add != 0).sum()),
                                 mean_log_correction=float(add[add != 0].mean()) if (add != 0).any() else 0.0,
                                 max_log_correction=float(add.max()))
    return np.maximum(out, 0), details


def main():
    cfg = core.config(); out = core.OUT
    gates = json.loads((out / 'gates.json').read_text())
    train = core.load_train()
    test = pd.read_csv(core.ROOT / 'data/raw/test.csv', parse_dates=['date'])
    origin = cfg['final_cutoff']
    assert train.date.max() <= pd.Timestamp(origin)
    panel = Panel(core.ROOT, Schedule(core.ROOT / cfg['schedule_v6']))
    log, details = forecast(cfg, gates, train, test, origin, panel)
    frame = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(log)})
    v6_ids = pd.read_csv(core.ROOT / 'outputs/ensemble_v6/submission_ensemble_v6.csv', usecols=['example_id']).example_id
    assert len(frame) == 157194 and frame.example_id.is_unique and (frame.example_id.to_numpy() == v6_ids.to_numpy()).all()
    assert np.isfinite(frame.boardings).all() and (frame.boardings >= 0).all()
    path = out / 'submission_ensemble_v7.csv'; frame.to_csv(path, index=False, float_format='%.8f')

    # ---- difference against the v6 submission ----------------------------------------------------------
    v6 = np.log1p(pd.read_csv(core.ROOT / 'outputs/ensemble_v6/submission_ensemble_v6.csv').boardings.to_numpy())
    d = test[['date', 'station_id', 'hour']].copy(); d['diff'] = log - v6; d['month'] = d.date.dt.month
    rmsd = lambda x: float(np.sqrt(np.mean(np.square(x))))
    by_station = d.groupby('station_id')['diff'].agg(mean='mean', rmsd=rmsd, max_abs=lambda x: float(np.abs(x).max()))
    diff = dict(log_rmsd_all=rmsd(d['diff']), mean_log_diff=float(d['diff'].mean()), rows_changed=int((d['diff'].abs() > 1e-6).sum()),
                rows_changed_over_0_05=int((d['diff'].abs() > 0.05).sum()), mean_boardings_v6=float(np.expm1(v6).mean()),
                mean_boardings_v7=float(frame.boardings.mean()),
                by_month={int(k): dict(mean=float(g['diff'].mean()), rmsd=rmsd(g['diff'])) for k, g in d.groupby('month')},
                by_hour={int(k): dict(mean=float(g['diff'].mean()), rmsd=rmsd(g['diff'])) for k, g in d.groupby('hour')},
                by_station_top15={int(k): {a: round(float(b), 5) for a, b in r.items()} for k, r in by_station.sort_values('rmsd', ascending=False).head(15).iterrows()},
                stations_changed=int((by_station.rmsd > 1e-6).sum()))
    quota = json.loads((out / 'munhak_quota.json').read_text()) if (out / 'munhak_quota.json').exists() else {}
    (out / 'MUNHAK_QUOTA_AND_DIFF_AUDIT.json').write_text(json.dumps(dict(
        information='local data only; no leaderboard score is used', munhak_quota=quota, diff_v7_minus_v6_log=diff),
        ensure_ascii=False, indent=1, default=float))
    states = details['munhak_states']; states.to_csv(out / 'munhak_test_states.csv', index=False)
    report = pd.DataFrame(details.get('event_report', []))
    report.to_csv(out / 'event_actions_test.csv', index=False)
    meta = dict(file=path.name, sha256=sha(path), rows=int(len(frame)), mean_boardings=float(frame.boardings.mean()),
                adopted_modules=gates['adopted'], blend_weights=details['weights'], eq_final_files=details['eq_files'],
                growth=details.get('growth'), munhak_rows_replaced=details.get('munhak_rows_replaced'),
                event_actions=int(len(report)), diff_vs_v6_log_rmsd=diff['log_rmsd_all'],
                v6_submission_sha256=sha(core.ROOT / 'outputs/ensemble_v6/submission_ensemble_v6.csv'),
                inputs={v: sha(core.ROOT / p) for v, p in cfg['base_submissions'].items()})
    (out / 'submission_validation.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: meta[k] for k in ('sha256', 'rows', 'mean_boardings', 'adopted_modules', 'growth', 'diff_vs_v6_log_rmsd')}, ensure_ascii=False, indent=1, default=float))
    print('diff by month', {k: round(v['mean'], 4) for k, v in diff['by_month'].items()}, 'rows changed', diff['rows_changed'])
    if len(report):
        print(report[report.station == 3139].to_string()[:2500])


if __name__ == '__main__':
    main()
