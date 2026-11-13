"""v21 step 'busholiday': on special days, stations whose bus stop was busier (or quieter) than the network average get
their subway forecast moved in the same direction.  Bus = hourly bus taps at one stop next to each of 29 stations
(stcis.go.kr, queried 2026-11-13).  Evaluation-period subway ridership is not used anywhere.

  x(s, d)  = log((bus(s, d) + 5) / (bus baseline(s, same day type, same window) + 5)), bus = boardings + alightings 06-23h
  r(s, d)  = mean over hours of (truth - forecast) of the v21 out-of-fold forecast
  model    r(s, d) - mean_s r(s, d) = beta * (x(s, d) - mean_s x(s, d))      (cross-section only; the day-level error is not touched)
Validation: Chuseok 2025 days are in fold 1, Seollal 2026 days in folds 2 and 3.  Fold 1 uses beta fitted on Seollal, folds
2 and 3 use beta fitted on Chuseok (never on themselves).  The test uses beta fitted on both.  Shift applied to all hours."""
import json, os
import numpy as np, pandas as pd
from src.v8 import core as c8
from src.v11 import stack as s11

ROOT = c8.ROOT; OUT = ROOT / 'outputs/ensemble_v21/final'
BUS = [ROOT / 'data/external_bus/stcis_20261113/stcis_bus_hourly_long_holidays.csv', ROOT / 'data/external_bus/stcis_20261113/stcis_bus_hourly_long.csv']
CLIP = float(os.environ.get('V21_BUS_CLIP', '1.0'))
SPECIAL = {'chuseok25': ['2025-10-03', '2025-10-04', '2025-10-05', '2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-10-10'],
           'seollal26': ['2026-02-14', '2026-02-15', '2026-02-16', '2026-02-17', '2026-02-18'],
           'chuseok26': ['2026-09-24', '2026-09-25', '2026-09-26', '2026-09-27'],
           'october26': ['2026-10-03', '2026-10-04', '2026-10-05', '2026-10-09', '2026-10-10', '2026-10-11']}
ORDINARY = {'chuseok25': ['2025-10-01', '2025-10-02', '2025-10-11', '2025-10-12'],
            'seollal26': ['2026-02-07', '2026-02-08', '2026-02-09', '2026-02-10', '2026-02-11', '2026-02-12', '2026-02-13'],
            'chuseok26': ['2026-09-14', '2026-09-15', '2026-09-16', '2026-09-17', '2026-09-18', '2026-09-19', '2026-09-20', '2026-09-21', '2026-09-22', '2026-09-23'],
            'october26': ['2026-10-01', '2026-10-02', '2026-10-06', '2026-10-07', '2026-10-08', '2026-09-19', '2026-09-20']}


def dtype(d):
    w = pd.Timestamp(d).dayofweek; return 'sat' if w == 5 else ('sun' if w == 6 else 'wd')


def bus_anomaly():
    stops = pd.read_csv(ROOT / 'outputs/ensemble_v21/work/bus_stop_per_station.csv', dtype={'ars': str}, encoding='utf-8-sig').head(30)
    b = pd.concat([pd.read_csv(p, dtype={'ars': str}) for p in BUS if p.exists()])
    key = ['ars', 'date', 'hour', 'next_day']
    dup = b[b.duplicated(key, keep=False)]
    assert dup.groupby(key)[['ride', 'alight']].nunique().max().max() <= 1, 'overlapping bus rows disagree'
    b = b.drop_duplicates(key)                                               # the two files overlap on 54 stop-days (identical values)
    b = b[~b.next_day & b.hour.between(6, 23)]
    day = b.groupby(['ars', 'date']).apply(lambda q: q.ride.sum() + q.alight.sum()).rename('bus').reset_index()
    rows = []
    for r in stops.itertuples():
        x = day[day.ars == r.ars].set_index('date').bus
        for grp, days in SPECIAL.items():
            for d in days:
                if d not in x.index:
                    continue
                base = [x[o] for o in ORDINARY[grp] if o in x.index and dtype(o) == dtype(d)]
                if not base and dtype(d) != 'wd':                                  # no ordinary weekend in the window: weekday baseline x weekend ratio of the test window
                    continue
                if not base:
                    continue
                rows.append(dict(station_id=int(r.station_id), ars=r.ars, group=grp, date=d, bus=float(x[d]), base=float(np.median(base)), x=float(np.log((x[d] + 5) / (np.median(base) + 5)))))
    A = pd.DataFrame(rows)
    A['x_dev'] = (A.x - A.groupby('date').x.transform('mean')).clip(-CLIP, CLIP)
    return A


def main():
    A = bus_anomaly(); S = s11.Stack(); F = S.folds; st = list(S.panel.stations.astype(int))
    oof = {k: np.load(OUT / 'oof_v21.npz')[f'fold{k}'].astype(float) for k in F}
    rec = []
    for k, blk in F.items():
        dates = [str(d)[:10] for d in blk.dates]
        for a in A[A.group.isin(['chuseok25', 'seollal26'])].itertuples():
            if a.date in dates:
                i = dates.index(a.date); s = st.index(a.station_id)
                rec.append(dict(fold=k, group=a.group, date=a.date, station_id=a.station_id, x_dev=a.x_dev, r=float((S.truth[k][i, s] - oof[k][i, s]).mean())))
    R = pd.DataFrame(rec); R['r_dev'] = R.r - R.groupby(['fold', 'date']).r.transform('mean')
    def beta(df):
        return float(np.clip((df.x_dev * df.r_dev).sum() / max((df.x_dev ** 2).sum(), 1e-9), 0, 1.5))
    b_ch, b_se, b_all = beta(R[R.group == 'chuseok25']), beta(R[R.group == 'seollal26']), beta(R)
    corr = {g: round(float(np.corrcoef(R[R.group == g].x_dev, R[R.group == g].r_dev)[0, 1]), 3) for g in ('chuseok25', 'seollal26')}
    new = {k: v.copy() for k, v in oof.items()}
    for k, blk in F.items():
        dates = [str(d)[:10] for d in blk.dates]
        for a in A[A.group.isin(['chuseok25', 'seollal26'])].itertuples():
            if a.date in dates:
                bk = b_se if a.group == 'chuseok25' else b_ch
                new[k][dates.index(a.date), st.index(a.station_id)] += bk * a.x_dev
    s0, s1 = S.score(oof), S.score(new)
    ok = s1['all_oof'] < s0['all_oof'] - 1e-7 and all(s1[f'fold_{k}'] <= s0[f'fold_{k}'] + 1e-5 for k in F)
    D = R.groupby(['fold', 'group', 'date']).r.mean().reset_index().merge(A.groupby('date').x.mean().rename('x_mean').reset_index(), on='date')
    day_corr = {g: round(float(np.corrcoef(D[D.group == g].x_mean, D[D.group == g].r)[0, 1]), 3) for g in ('chuseok25', 'seollal26')}
    day_corr['all'] = round(float(np.corrcoef(D.x_mean, D.r)[0, 1]), 3)
    out = dict(beta_chuseok=b_ch, beta_seollal=b_se, beta_all=b_all, corr_xdev_rdev=corr, day_level_corr_xmean_rmean=day_corr, n=len(R), before=s0, after=s1, adopted=bool(ok),
               delta={f'fold_{k}': round(s1[f'fold_{k}'] - s0[f'fold_{k}'], 6) for k in F})
    print(json.dumps(out, indent=1, default=float))
    A.to_csv(OUT / 'bus_holiday_anomaly.csv', index=False); R.to_csv(OUT / 'bus_holiday_training_residuals.csv', index=False)
    (OUT / 'gates_busholiday.json').write_text(json.dumps(out, indent=1, default=float))
    return out, A


def apply_test(beta, A):
    """Test shift on the special days of the evaluation period (Chuseok 2026, October holidays) at the stations with bus data."""
    sub = pd.read_csv(OUT / 'submission_ensemble_v21.csv'); te = pd.read_csv(ROOT / 'data/raw/test.csv'); assert (sub.example_id == te.example_id).all()
    z = np.log1p(sub.boardings.to_numpy()); rows = []
    for a in A[A.group.isin(['chuseok26', 'october26'])].itertuples():
        m = ((te.date == a.date) & (te.station_id == a.station_id)).to_numpy()
        z[m] = np.maximum(z[m] + beta * a.x_dev, 0); rows.append(dict(date=a.date, station_id=a.station_id, x_dev=round(a.x_dev, 3), shift=round(beta * a.x_dev, 4)))
    return pd.DataFrame({'example_id': sub.example_id, 'boardings': np.expm1(z)}), pd.DataFrame(rows)


def write():
    out, A = main()
    if not out['adopted']:
        print('busholiday rejected: nothing written'); return
    S = s11.Stack(); F = S.folds; st = list(S.panel.stations.astype(int))
    oof = {k: np.load(OUT / 'oof_v21.npz')[f'fold{k}'].astype(float) for k in F}
    for k, blk in F.items():
        dates = [str(d)[:10] for d in blk.dates]
        for a in A[A.group.isin(['chuseok25', 'seollal26'])].itertuples():
            if a.date in dates:
                oof[k][dates.index(a.date), st.index(a.station_id)] += (out['beta_seollal'] if a.group == 'chuseok25' else out['beta_chuseok']) * a.x_dev
    np.savez_compressed(OUT / 'oof_v21_bus.npz', **{f'fold{k}': oof[k].astype(np.float32) for k in F})
    sub, shifts = apply_test(out['beta_all'], A)
    assert len(sub) == 157194 and np.isfinite(sub.boardings).all()
    sub.to_csv(OUT / 'submission_ensemble_v21_bus.csv', index=False, float_format='%.8f'); shifts.to_csv(OUT / 'bus_holiday_test_shifts.csv', index=False)
    shifts['group'] = np.where(shifts.date < '2026-10-01', 'chuseok26', 'october26')
    summ = {g: dict(station_days=int(len(q)), stations=int(q.station_id.nunique()), mean_abs=round(float(q['shift'].abs().mean()), 4), mean=round(float(q['shift'].mean()), 4)) for g, q in shifts.groupby('group')}
    g = json.loads((OUT / 'gates_busholiday.json').read_text()); g['test_shifts'] = summ; (OUT / 'gates_busholiday.json').write_text(json.dumps(g, indent=1, default=float))
    print('test shifts:', summ)


if __name__ == '__main__':
    write()
