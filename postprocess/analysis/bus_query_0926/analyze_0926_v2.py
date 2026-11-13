"""9/26 판정 v2: 정류장을 늘리고 품질 필터와 두 집계(중앙값·합계)를 함께 본다.

python3 analyze_0926_v2.py --bus <stcis long csv> --train data/train.csv --test <test.csv> \
    --sub v10=<submission csv> --sub v20e=<submission csv> --stops <city,ars,station_id csv>

정류장별 추정 = 지하철(역) 2025-10-07 비율 x 버스 2026-09-26 비율 / 버스 2025-10-07 비율
비율은 모두 '그날 / 평소 토요일' (2026: 9/5, 9/12, 9/19 중앙값, 2025: 9/20, 9/27 평균).
품질 필터: 기준 토요일 하루 승차(06~23) 모두 100 이상, 기준 토요일끼리 최대/최소 1.6 이하.
평가 기간 지하철은 읽지 않는다(train <= 2026-06-30).
"""
import argparse
import numpy as np
import pandas as pd

B26 = ['2026-09-05', '2026-09-12', '2026-09-19']
B25 = ['2025-09-20', '2025-09-27']
BANDS = {'am(06-08)': [6, 7, 8], 'pm(17-18)': [17, 18], 'day(06-23)': list(range(6, 24))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bus', required=True); ap.add_argument('--train', required=True); ap.add_argument('--test', required=True)
    ap.add_argument('--sub', action='append', default=[]); ap.add_argument('--stops', required=True); ap.add_argument('--out', default='result_v2.csv')
    a = ap.parse_args()
    stops = pd.read_csv(a.stops, dtype=str); s2st = dict(zip(stops.ars.str.zfill(5), stops.station_id.astype(int)))
    b = pd.read_csv(a.bus, dtype={'ars': str}); b = b[(b.next_day.astype(str) == 'False') & b.ars.isin(s2st)]
    tr = pd.read_csv(a.train); tr = tr[tr.date <= '2026-06-30']
    te = pd.read_csv(a.test)
    for kv in a.sub:
        k, p = kv.split('='); te[k] = te.example_id.map(pd.read_csv(p).set_index('example_id').boardings)
    models = [kv.split('=')[0] for kv in a.sub]

    day = b[b.hour.between(6, 23)].groupby(['ars', 'date']).ride.sum().unstack()
    ok = []
    for ars in day.index:
        r = day.loc[ars]
        need = B26 + B25 + ['2026-09-26', '2025-10-07']
        if any(pd.isna(r.get(d)) for d in need):
            continue
        base26 = r[B26]
        if (base26 < 100).any() or base26.max() / base26.min() > 1.6 or (r[B25] < 100).any():
            continue
        ok.append(ars)
    print(f'정류장 {len(s2st)}곳 중 필터 통과 {len(ok)}곳: {ok}')

    rows = []
    for band, hrs in BANDS.items():
        x = b[b.hour.isin(hrs) & b.ars.isin(ok)].groupby(['date', 'ars']).ride.sum().unstack()
        bus26 = x.loc['2026-09-26'] / x.loc[B26].median()
        bus25 = x.loc['2025-10-07'] / x.loc[B25].mean()
        sts = sorted({s2st[s] for s in ok})
        s = tr[tr.station_id.isin(sts) & tr.hour.isin(hrs)].groupby(['date', 'station_id']).boardings.sum().unstack()
        sub25 = s.loc['2025-10-07'] / s.loc[B25].mean()
        est = pd.Series({ars: sub25[s2st[ars]] * bus26[ars] / bus25[ars] for ars in ok})
        m = te[te.station_id.isin(sts) & te.hour.isin(hrs)].groupby(['date', 'station_id'])[models].sum()
        mr = {k: (m.loc['2026-09-26'][k] / m.loc[B26][k].groupby(level=1).median()) for k in models}
        # pooled (sum over stops/stations)
        pooled_est = (sub25.mean() * x.loc['2026-09-26'].sum() / x.loc[B26].sum(axis=1).median()
                      / (x.loc['2025-10-07'].sum() / x.loc[B25].sum(axis=1).mean()))
        q1, med, q3 = np.percentile(est, [25, 50, 75])
        row = dict(band=band, n=len(est), est_median=med, est_q1=q1, est_q3=q3, est_pooled=pooled_est)
        for k in models:
            row[f'{k}_median'] = mr[k].median()
            row[f'share_closer_to_{k}'] = None
        if len(models) == 2:
            k1, k2 = models
            closer1 = sum(abs(np.log(est[ars]) - np.log(mr[k1][s2st[ars]])) < abs(np.log(est[ars]) - np.log(mr[k2][s2st[ars]])) for ars in ok)
            row[f'share_closer_to_{k1}'] = closer1 / len(ok); row[f'share_closer_to_{k2}'] = 1 - closer1 / len(ok)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 200)
    print(out.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
