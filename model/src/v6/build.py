"""Build the single ensemble_v6 submission.

    base      log-space blend of the existing v2 / v4 / v5 submissions; weights come from local
              out-of-fold predictions only (src/v6/blend.py)
    munhak    schedule-conditioned mixture specialist for station 3127 (KBO home schedule + announced concerts)
    events    lift overlay for announced concerts (3209, 3210), K League home matches (3754) and Pentaport (3139)

Usage (from the project root):  python -m src.v6.build
Writes only under outputs/ensemble_v6/. v1-v5 artefacts are read, never modified.
"""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from ..holiday_calendar import calendar_features
from . import blend
from .pipeline import apply_specialists
from .schedule import Schedule

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg = json.loads((ROOT / 'configs/ensemble_v6.json').read_text()); out = ROOT / cfg['output']; out.mkdir(parents=True, exist_ok=True)
    origin = pd.Timestamp(cfg['final_cutoff']); versions = cfg['blend']['versions']
    train = pd.read_csv(ROOT / cfg['input_data'] / 'train.csv', parse_dates=['date'])
    test = pd.read_csv(ROOT / cfg['input_data'] / 'test.csv', parse_dates=['date'])
    if train.date.max() > origin: raise ValueError('training data beyond the information cutoff')

    weights, blend_report, _ = blend.fit(ROOT, cfg)
    (out / 'blend_weights.json').write_text(json.dumps(dict(weights=weights, report=blend_report), ensure_ascii=False, indent=1))
    logs = {}
    for name in versions:
        s = pd.read_csv(ROOT / cfg['base_submissions'][name])
        if not (s.example_id.values == test.example_id.values).all(): raise ValueError(f'{name}: id order')
        logs[name] = np.log1p(s.boardings.to_numpy())
    is_holiday = calendar_features(test.date, as_of=origin).is_holiday.to_numpy(bool)
    base = blend.apply(logs, weights, is_holiday, versions)

    schedule = Schedule(ROOT / cfg['schedule'])
    final, details = apply_specialists(train, test, base, origin, schedule)

    frame = test[['example_id']].copy(); frame['boardings'] = np.expm1(final)
    if len(frame) != 157194 or not frame.example_id.is_unique or not np.isfinite(frame.boardings).all() or (frame.boardings < 0).any():
        raise ValueError('Submission integrity')
    path = out / 'submission_ensemble_v6.csv'; frame.to_csv(path, index=False, float_format='%.8f')
    back = pd.read_csv(path)
    validation = dict(file=path.name, rows=len(back), ids_in_test_order=bool((back.example_id.values == test.example_id.values).all()),
                      finite_nonnegative=bool(np.isfinite(back.boardings).all() and (back.boardings >= 0).all()),
                      mean=float(back.boardings.mean()), min=float(back.boardings.min()), max=float(back.boardings.max()), sha256=sha(path),
                      monthly_mean={m: float(g.mean()) for m, g in back.boardings.groupby(test.date.dt.strftime('%Y-%m'))})
    (out / 'submission_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=1))

    states = details['munhak_states']; states.to_csv(out / 'munhak_test_states.csv', index=False)
    changed = np.abs(final - base) > 1e-9
    audit = dict(information_cutoff=str(origin.date()),
                 kbo_records=[dict(season=r['season'], stage=r['stage'], known_since=str(r['known'].date()), coverage=[str(r['start'].date()), str(r['end'].date())],
                                   home_dates=len(r['home']), reserve_dates=len(r['reserve'])) for r in schedule.visible(origin)],
                 events_total=len(schedule.events),
                 events_in_test_window=[dict(name=e.get('name'), venue=e['venue'], date=e['date'], start=e.get('start'), known_since=e['known_since'])
                                        for e in schedule.events if pd.Timestamp(e['date']) > origin],
                 munhak_test_state_days=states.state.value_counts().to_dict(), munhak_concert_days=states.loc[states.concert, 'date'].dt.strftime('%Y-%m-%d').tolist(),
                 munhak_fit=details['munhak_fit'], event_overlay=details['event_report'],
                 rows_changed_by_station={int(k): int(v) for k, v in test.loc[changed].groupby('station_id').size().items()},
                 rows_changed_total=int(changed.sum()))
    (out / 'schedule_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=1))
    # Row-level record of every prediction a specialist changed (base blend vs final, in boardings).
    rows = test.loc[changed, ['example_id', 'date', 'station_id', 'hour']].copy()
    rows['base_boardings'] = np.expm1(base[changed]); rows['final_boardings'] = np.expm1(final[changed])
    rows.to_csv(out / 'specialist_rows.csv', index=False, float_format='%.3f')
    from .events import EventOverlay
    overlay = EventOverlay(schedule.overlay_events()).fit(train, origin)
    (out / 'event_lift_summary.json').write_text(json.dumps({str(sid): overlay.family_summary(sid) for sid in (3754, 3209, 3210, 3139)}, ensure_ascii=False, indent=1))

    # How far the final file is from each input submission (log RMSD), overall and on specialist stations.
    seg = np.select([test.station_id == 3127, test.station_id == 3754, test.station_id.isin([3209, 3210]), test.station_id == 3139], ['3127', '3754', '3209_3210', '3139'], 'other')
    diff = {name: dict(all=float(np.sqrt(((final - logs[name]) ** 2).mean())), **{k: float(np.sqrt(((final - logs[name])[seg == k] ** 2).mean())) for k in np.unique(seg)}) for name in versions}
    diff['base_blend'] = dict(all=float(np.sqrt(((final - base) ** 2).mean())), **{k: float(np.sqrt(((final - base)[seg == k] ** 2).mean())) for k in np.unique(seg)})
    (out / 'input_distance_audit.json').write_text(json.dumps(diff, ensure_ascii=False, indent=1))
    print(json.dumps(validation, ensure_ascii=False, indent=1)); print('weights', json.dumps(weights))
    print('rows changed by station', audit['rows_changed_by_station'])
    for r in details['event_report']: print('  ', r)


if __name__ == '__main__':
    main()
