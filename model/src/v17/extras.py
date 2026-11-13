"""Builds submission_ensemble_v17e.csv = v17 + the entries of configs/ensemble_v17.json that no fold can validate.

    python -m src.v17.extras

  summer_leave   collective summer leave fortnight: per station and hour group, half of the gap between the fall observed
                 in 2025 (training data) and the fall v17 already forecasts for 2026
  extras         event-specific entries: sold-out fan match, stadium concert, stadium festival, marathon, and two
                 probability-weighted entries (a monthly Saturday surge of unknown cause, a district festival)
No leaderboard value is used.  Only ridership before the cutoff and realised facts of the evaluation period enter.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v13.extras import scenario_log
from src.v14.extras import fall

OUT = c8.ROOT / 'outputs/ensemble_v17'


def main():
    cfg = json.loads((c8.ROOT / 'configs/ensemble_v17.json').read_text()); panel, folds, test = c8.load_blocks(); st = panel.stations
    names = pd.read_csv(c8.ROOT / 'data/raw/metaData.csv').set_index('station_id').station_name
    sub = pd.read_csv(OUT / 'submission_ensemble_v17.csv'); assert (sub.example_id.to_numpy() == test.example_id).all()
    z = np.log1p(sub.boardings.to_numpy())[test.row_index].reshape(test.pred.shape); z0 = z.copy()
    # ---- summer leave -----------------------------------------------------------------------------------------------------
    L = cfg['summer_leave']; groups = L['hour_groups']; days = pd.DatetimeIndex(panel.days); assert L['leave_2025'][1] <= '2026-06-30'
    past = np.moveaxis(np.expm1(panel.Z), 0, 1)                              # [day, S, H] observed boardings (training period only)
    kinds = c8.day_kinds(days, c8.FINAL); f25, _ = fall(past, days, kinds, L['leave_2025'], L['reference_2025'], groups)
    f26, lv = fall(np.expm1(z), test.dates, test.kind, L['leave_2026'], L['reference_2026'], groups)
    shift = np.clip(L['share_of_gap'] * np.minimum(f25 - f26, 0.0), L['largest_shift'], 0.0)
    shift[:, panel.first > 0] = 0.0; shift[:, np.isin(st, L['excluded_stations'])] = 0.0; shift = np.nan_to_num(shift)
    for g, (lo, hi) in enumerate(groups):
        z[np.ix_(np.flatnonzero(lv), np.arange(len(st)), np.arange(lo - 6, hi - 6 + 1))] += shift[g][None, :, None]
    old = (panel.first == 0) & ~np.isin(st, L['excluded_stations']); day_before = np.expm1(z0[lv]).sum(2); day_after = np.expm1(z[lv]).sum(2)
    order = np.argsort(shift.mean(0))
    leave = dict(days=[str(d)[:10] for d in test.dates[lv]], stations=int(old.sum()),
                 median_fall_2025=[round(float(np.median(f25[g][old])), 3) for g in range(len(groups))], median_fall_v17=[round(float(np.median(f26[g][old])), 3) for g in range(len(groups))],
                 median_shift_by_hour_group=[round(float(np.median(shift[g][old])), 3) for g in range(len(groups))],
                 mean_day_total_change=round(float(np.log(day_after[:, old].sum() / day_before[:, old].sum())), 4),
                 largest={f'{int(st[i])} {names[int(st[i])]}': [round(float(x), 3) for x in shift[:, i]] for i in order[:8]})
    # ---- event entries ------------------------------------------------------------------------------------------------------
    flat = np.empty(z.size); flat[test.row_index] = z.reshape(-1); t = pd.read_csv(c8.ROOT / 'data/raw/test.csv'); log = []
    for e in cfg['extras']:
        day = (t.station_id == e['station']) & (t.date == e['date']); hours = sorted({int(h) for s in e.get('scenarios', []) for h in s['hours']} | {int(h) for h in e.get('multiply', {})})
        before, after = {}, {}
        for h in hours:
            i = int(np.flatnonzero((day & (t.hour == h)).to_numpy())[0]); before[h] = int(round(np.expm1(flat[i])))
            flat[i] = e['weight_on_scenarios'] * scenario_log(e['scenarios'], h, flat[i]) + (1 - e['weight_on_scenarios']) * flat[i] if 'scenarios' in e else flat[i] + np.log(e['multiply'][str(h)])
            after[h] = int(round(np.expm1(flat[i])))
        log.append(dict(name=e['name'], station=e['station'], date=e['date'], before=before, after=after, probabilities=[sc['p'] for sc in e.get('scenarios', [])]))
    frame = pd.DataFrame({'example_id': sub.example_id, 'boardings': np.expm1(flat)})
    assert len(frame) == 157194 and np.isfinite(frame.boardings).all() and (frame.boardings >= 0).all()
    path = OUT / 'submission_ensemble_v17e.csv'; frame.to_csv(path, index=False, float_format='%.8f')
    out = dict(information='v17 + entries that no fold can validate; no leaderboard value is used', summer_leave=leave, entries=log,
               submission=dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(frame)), mean_boardings=float(frame.boardings.mean()),
                               rows_changed_vs_v17=int((np.abs(np.log1p(frame.boardings.to_numpy()) - np.log1p(sub.boardings.to_numpy())) > 1e-9).sum())))
    (OUT / 'gates_v17e.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(leave, ensure_ascii=False)); [print(r['date'], r['station'], r['name'][:30], r['before'], '->', r['after']) for r in log]; print(json.dumps(out['submission'], indent=1))
    return out


if __name__ == '__main__':
    main()
