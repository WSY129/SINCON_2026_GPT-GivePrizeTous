"""Builds submission_ensemble_v13e.csv = v13 + the event-specific entries of configs/ensemble_v13.json.

    python -m src.v13.extras

These entries use realised facts of the evaluation period (event held, sold out, start time, duration) and ridership
before the cutoff only.  No fold contains such a day, so they are NOT validated locally; the file is kept apart from
the validated v13 for that reason.  No leaderboard value is used.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8

OUT = c8.ROOT / 'outputs/ensemble_v13'


def scenario_log(scenarios, hour, fallback):
    """mean over the scenarios of log1p(boardings) at one hour; a scenario without that hour keeps the fallback."""
    p = np.array([s['p'] for s in scenarios]); assert abs(p.sum() - 1) < 1e-9
    return float(sum(s['p'] * np.log1p(s['hours'][str(hour)]) if str(hour) in s['hours'] else s['p'] * fallback for s in scenarios))


def main():
    cfg = json.loads((c8.ROOT / 'configs/ensemble_v13.json').read_text()); test = pd.read_csv(c8.ROOT / 'data/raw/test.csv')
    sub = pd.read_csv(OUT / 'submission_ensemble_v13.csv'); assert (sub.example_id == test.example_id).all()
    z = np.log1p(sub.boardings.to_numpy()); log = []
    for e in cfg['extras']:
        day = (test.station_id == e['station']) & (test.date == e['date']); hours = sorted({int(h) for s in e.get('scenarios', []) for h in s['hours']} | {int(h) for h in e.get('multiply', {})})
        before, after = {}, {}
        for h in hours:
            i = np.flatnonzero((day & (test.hour == h)).to_numpy()); assert len(i) == 1; i = int(i[0]); before[h] = int(round(np.expm1(z[i])))
            if 'scenarios' in e:
                w = e['weight_on_scenarios']; z[i] = w * scenario_log(e['scenarios'], h, z[i]) + (1 - w) * z[i]
            else:
                z[i] = z[i] + np.log(e['multiply'][str(h)])
            after[h] = int(round(np.expm1(z[i])))
        log.append(dict(name=e['name'], station=e['station'], date=e['date'], before=before, after=after))
    frame = pd.DataFrame({'example_id': sub.example_id, 'boardings': np.expm1(z)})
    assert len(frame) == 157194 and np.isfinite(frame.boardings).all() and (frame.boardings >= 0).all()
    path = OUT / 'submission_ensemble_v13e.csv'; frame.to_csv(path, index=False, float_format='%.8f')
    out = dict(information='v13 + event-specific entries; not validated on any fold; no leaderboard value is used', entries=log,
               submission=dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(frame)), mean_boardings=float(frame.boardings.mean()),
                               rows_changed_vs_v13=int((np.abs(np.log1p(frame.boardings.to_numpy()) - np.log1p(sub.boardings.to_numpy())) > 1e-9).sum())))
    (OUT / 'gates_v13e.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for r in log:
        print(r['date'], r['station'], r['name'][:34], r['before'], '->', r['after'])
    print(json.dumps(out['submission'], indent=1))
    return out


if __name__ == '__main__':
    main()
