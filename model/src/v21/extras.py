"""Builds submission_ensemble_v21e.csv = v21 (validated, with the bus holiday step) + the entries that no fold can validate:
  1. the v20e entries unchanged (configs/ensemble_v20.json: summer leave fortnight, 14 entries = 13 scenario entries + the marathon multiplier), re-run on the v21 base
  2. Songdo beer festival 2026 (3139, 8/22-8/30, 18-23h) moved down: in 2026 the bus taps at the festival-park exit stops
     (38696 + 38697) were 0.05-0.46 of the 2025 festival excess on the same weekday (2025 is training data), admission became
     paid in 2026 (10,000 / 12,000 won, yanolja ticket page), and the stop lost about 30 % of its weekday volume after 8/4.
     Scenario A = forecast as is, scenario B = ordinary level + rho x (forecast - ordinary level) with rho = bus ratio / 0.7
     (floor 0.2); probability 0.5 each, combined in log space as in every other entry.
  3. Grave visits at 부평삼거리 (3122) on Chuseok day 2026-09-25, 08-16h: scenario B scales the forecast so that the 08-16h total
     equals the 2026 Seollal day (2026-02-17, training data: 10,124 boardings; a short 3-day holiday like the 4-day Chuseok 2026,
     against 8,451 on Chuseok day 2025 inside a 10-day holiday); bus taps at 39491 + 39493 were 1.4-1.9 x the 2025 Chuseok day.
     Probability 0.5 each, log space.
No leaderboard value is used.  Run: .venv/bin/python -m src.v21.extras"""
import hashlib, json, shutil
import numpy as np, pandas as pd
from src.v8 import core as c8

OUT = c8.ROOT / 'outputs/ensemble_v21/final'
BEER = {'2026-08-22': 0.44, '2026-08-23': 0.2, '2026-08-24': 0.2, '2026-08-25': 0.2, '2026-08-26': 0.23, '2026-08-27': 0.2,
        '2026-08-28': 0.66, '2026-08-29': 0.30, '2026-08-30': 0.36}
HOURS = range(18, 24)


def main(base_name='submission_ensemble_v21_bus.csv'):
    from src.v20 import extras as v20
    tmp = OUT / 'e_base'; tmp.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT / base_name, tmp / 'submission_ensemble_v19.csv')
    v20.BASE = tmp; v20.OUT = OUT / 'e_out'; v20.OUT.mkdir(parents=True, exist_ok=True)
    v20.main()
    sub = pd.read_csv(v20.OUT / 'submission_ensemble_v20e.csv'); te = pd.read_csv(c8.ROOT / 'data/raw/test.csv'); assert (sub.example_id == te.example_id).all()
    z = np.log1p(sub.boardings.to_numpy()); rows = []
    te['dow'] = pd.to_datetime(te.date).dt.dayofweek
    ordinary = te[(te.station_id == 3139) & te.date.between('2026-08-01', '2026-09-30') & ~te.date.isin(list(BEER) + ['2026-07-31', '2026-08-01', '2026-08-02', '2026-09-12'])]
    for d, rho in BEER.items():
        dw = pd.Timestamp(d).dayofweek; same = ordinary[ordinary.dow.isin([dw] if dw >= 5 else [0, 1, 2, 3, 4])]
        for h in HOURS:
            m = ((te.date == d) & (te.station_id == 3139) & (te.hour == h)).to_numpy(); i = np.flatnonzero(m)[0]
            base = float(np.median(np.expm1(z[same[same.hour == h].index.to_numpy()])))
            a = float(np.expm1(z[i])); b = min(a, base + rho * max(a - base, 0.0))     # only lowers
            new = 0.5 * np.log1p(a) + 0.5 * np.log1p(b); rows.append(dict(date=d, hour=h, ordinary=round(base), before=round(a), scenario_b=round(b), after=round(float(np.expm1(new)))))
            z[i] = new
    m = ((te.station_id == 3122) & (te.date == '2026-09-25') & te.hour.between(8, 16)).to_numpy()
    tot = float(np.expm1(z[m]).sum()); scale = 10124.0 / tot
    if scale > 1:
        z[m] = 0.5 * z[m] + 0.5 * np.log1p(np.expm1(z[m]) * scale)
    chuseok = dict(station=3122, date='2026-09-25', hours='08-16', before_total=round(tot), scenario_b_total=10124, after_total=round(float(np.expm1(z[m]).sum())))
    print('3122 Chuseok day:', chuseok)
    out = pd.DataFrame({'example_id': sub.example_id, 'boardings': np.expm1(z)})
    assert len(out) == 157194 and np.isfinite(out.boardings).all() and (out.boardings >= 0).all()
    path = OUT / 'submission_ensemble_v21e.csv'; out.to_csv(path, index=False, float_format='%.8f')
    beer = pd.DataFrame(rows); beer.to_csv(OUT / 'v21e_beer_festival.csv', index=False)
    g = json.loads((v20.OUT / 'gates_v20e.json').read_text())
    info = dict(base=base_name, v20e_entries=len(g.get('entries', [])), beer=rows, chuseok_3122=chuseok, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows_changed_vs_base=int((np.abs(np.log1p(out.boardings.to_numpy()) - np.log1p(pd.read_csv(OUT / base_name).boardings.to_numpy())) > 1e-6).sum()))
    (OUT / 'gates_v21e.json').write_text(json.dumps(info, indent=1, default=float))
    print(beer.to_string(index=False)); print({k: v for k, v in info.items() if k != 'beer'})


if __name__ == '__main__':
    main()
