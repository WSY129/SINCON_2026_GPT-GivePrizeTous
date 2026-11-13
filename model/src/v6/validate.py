"""Local validation of ensemble_v6 on the four time-ordered folds. No leaderboard information is used.

For each fold origin the base blend (version-level LOFO predictions with the fitted segment weights)
is passed through the same specialist code as the final forecast, with only the observations and the
schedule records that were available at that origin.

Usage (from the project root):  python -m src.v6.validate
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import blend
from .pipeline import apply_specialists
from .schedule import Schedule

ROOT = Path(__file__).resolve().parents[2]
STATIONS = {3127: '문학경기장', 3754: '부천종합운동장', 3209: '아시아드경기장', 3210: '서구청', 3139: '송도달빛축제공원'}


def rmse(x):
    x = np.asarray(x, float); return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float('nan')


def validator_metrics(frame, column):
    """The four numbers the project's earlier reports call the local validator, plus each fold."""
    err2 = (frame[column] - frame.true_log) ** 2
    mse = err2.groupby(frame.fold).mean()
    weights = {1: 0.15, 2: 0.15, 3: 0.35, 4: 0.35}
    return dict(all_oof=float(np.sqrt(err2.mean())), weighted_4fold=float(np.sqrt(sum(weights[f] * mse[f] for f in weights))),
                fold_3_4=float(np.sqrt(err2[frame.fold.isin([3, 4])].mean())), fold_4=float(np.sqrt(mse[4])),
                **{f'fold_{f}': float(np.sqrt(mse[f])) for f in (1, 2, 3)})


def run(cfg=None, verbose=True):
    cfg = cfg or json.loads((ROOT / 'configs/ensemble_v6.json').read_text())
    versions = cfg['blend']['versions']
    weights, blend_report, frame = blend.fit(ROOT, cfg)
    train = pd.read_csv(ROOT / cfg['input_data'] / 'train.csv', parse_dates=['date'])
    schedule = Schedule(ROOT / cfg['schedule'])
    frame = frame.sort_values(['fold', 'date', 'station_id', 'hour']).reset_index(drop=True)
    frame['base'] = blend.apply({v: frame[v].to_numpy() for v in versions}, weights, frame.is_holiday.to_numpy(), versions)
    # Same weights applied to each version's saved ensemble column (the convention of the earlier reports).
    frame['base_saved'] = blend.apply({v: frame[v + '_fit'].to_numpy() for v in versions}, weights, frame.is_holiday.to_numpy(), versions)
    frame['munhak_only'] = np.nan; frame['final'] = np.nan; frame['final_saved'] = np.nan
    event_reports = {}; munhak_state_rows = []
    for fold, origin in enumerate(cfg['fold_origins'], start=1):
        rows = np.flatnonzero((frame.fold == fold).to_numpy()); q = frame.iloc[rows]
        a, _ = apply_specialists(train, q, q.base.to_numpy(), origin, schedule, use=('munhak',))
        b, details = apply_specialists(train, q, q.base.to_numpy(), origin, schedule)
        c, _ = apply_specialists(train, q, q.base_saved.to_numpy(), origin, schedule)
        frame.iloc[rows, frame.columns.get_loc('munhak_only')] = a
        frame.iloc[rows, frame.columns.get_loc('final')] = b
        frame.iloc[rows, frame.columns.get_loc('final_saved')] = c
        event_reports[str(fold)] = details.get('event_report', [])
        if 'munhak_states' in details:                              # per-state view of station 3127 for this fold
            st_ = details['munhak_states'][['date', 'state', 'concert']]
            mq = q.loc[q.station_id == 3127, ['date', 'hour', 'true_log', 'base']].merge(st_, on='date', how='left')
            mq['final'] = b[np.flatnonzero((q.station_id == 3127).to_numpy())]; mq['fold'] = fold
            mq['label'] = np.where(mq.concert.fillna(False), 'CONCERT', mq.state.fillna('OFF'))
            dmax = np.expm1(mq.true_log).groupby(mq.date).transform('max'); mq['crowd_day'] = dmax >= 1200
            munhak_state_rows.append(mq)
    truth = frame.true_log.to_numpy()
    stages = versions + ['base', 'munhak_only', 'final']
    overall = {s: {**{str(f): rmse(g[s] - g.true_log) for f, g in frame.groupby('fold')}, 'all': rmse(frame[s] - truth)} for s in stages}
    station = {}
    for sid, name in STATIONS.items():
        g = frame[frame.station_id == sid]
        station[str(sid)] = dict(name=name, rows_per_fold=int(len(g) / 4),
                                 base={**{str(f): rmse(x.base - x.true_log) for f, x in g.groupby('fold')}, 'all': rmse(g.base - g.true_log)},
                                 final={**{str(f): rmse(x.final - x.true_log) for f, x in g.groupby('fold')}, 'all': rmse(g.final - g.true_log)},
                                 rows_changed={str(f): int((np.abs(x.final - x.base) > 1e-9).sum()) for f, x in g.groupby('fold')})
    # Effect of each event action on the rows it touched.
    actions = []
    for fold, reports in event_reports.items():
        for r in reports:
            g = frame[(frame.fold == int(fold)) & (frame.station_id == r['station']) & (frame.date == pd.Timestamp(r['date']))]
            actions.append(dict(fold=int(fold), **{k: r[k] for k in ('station', 'date', 'action')}, event=r.get('event', ''), comparable_events=r.get('comparable_events'),
                                rmsle_before=rmse(g.munhak_only - g.true_log), rmsle_after=rmse(g.final - g.true_log)))
    local_validator = dict(
        definition='RMSLE on the four time-ordered validation folds (157,194 rows each). all_oof pools every row; weighted_4fold = sqrt(0.15*MSE1 + 0.15*MSE2 + 0.35*MSE3 + 0.35*MSE4); fold_3_4 pools folds 3 and 4.',
        saved_ensemble_columns={**{v: validator_metrics(frame, v + '_fit') for v in versions}, 'v6_base_blend': validator_metrics(frame, 'base_saved'), 'v6_final': validator_metrics(frame, 'final_saved')},
        leave_one_fold_out_columns={**{v: validator_metrics(frame, v) for v in versions}, 'v6_base_blend': validator_metrics(frame, 'base'),
                                    'v6_blend_plus_munhak': validator_metrics(frame, 'munhak_only'), 'v6_final': validator_metrics(frame, 'final')})
    munhak_by_state = []
    if munhak_state_rows:
        ms = pd.concat(munhak_state_rows)
        for (fold, label), g in ms.groupby(['fold', 'label']):
            munhak_by_state.append(dict(fold=int(fold), state=label, days=int(g.date.nunique()), days_with_crowd=int(g[g.crowd_day].date.nunique()),
                                        rmsle_blend=rmse(g.base - g.true_log), rmsle_final=rmse(g.final - g.true_log), bias_final=float((g.final - g.true_log).mean())))
    other = frame[~frame.station_id.isin(list(STATIONS))]
    summary = dict(information='local OOF only; no leaderboard score is read anywhere in src/v6',
                   fold_origins=cfg['fold_origins'], blend=blend_report, local_validator=local_validator, overall_rmsle=overall, station_rmsle=station, munhak_by_state=munhak_by_state, event_actions=actions,
                   untouched_rows_identical=bool(np.allclose(other.final, other.base)))
    out = ROOT / cfg['output']; out.mkdir(parents=True, exist_ok=True)
    (out / 'validation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    if verbose:
        print('weights:', json.dumps(weights))
        print('[로컬 검증기] 저장 앙상블 열 기준'); print(pd.DataFrame(local_validator['saved_ensemble_columns']).T.round(4).to_string())
        print('[로컬 검증기] fold 제외(LOFO) 열 기준'); print(pd.DataFrame(local_validator['leave_one_fold_out_columns']).T.round(4).to_string())
        for sid, s in station.items():
            print(sid, s['name'], 'base', {k: round(v, 4) for k, v in s['base'].items()}, '-> final', {k: round(v, 4) for k, v in s['final'].items()}, 'rows changed', s['rows_changed'])
        for a in actions:
            print('  ', a)
    return summary, frame


if __name__ == '__main__':
    run()
