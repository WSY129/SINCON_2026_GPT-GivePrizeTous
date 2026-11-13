"""Bus (stcis) vs subway (training period) at Songdo festival park (3139) and Bupyeong-samgeori (3122):
2025 festivals / Chuseok in the TRAINING period give a bus -> subway ratio; 2026 bus shows the evaluation-period size.
No evaluation-period subway ridership is read."""
import pandas as pd, numpy as np
from pathlib import Path
OUT = Path('outputs/handoff_for_gemini/02_bus_raw/tables')
b = pd.read_csv('data/external_bus/stcis_20261113/stcis_bus_hourly_long.csv', dtype={'ars': str}); b = b[~b.next_day]
t = pd.read_csv('data/raw/train.csv'); sub = pd.read_csv('outputs/ensemble_v20/submission_ensemble_v20e.csv'); te = pd.read_csv('data/raw/test.csv'); te['v20e'] = sub.boardings
L = []
# ---- Songdo: hours 20-23, stops 38696+38697 ----
bs = b[b.ars.isin(['38696', '38697']) & b.hour.between(20, 23)].groupby(['date', 'hour']).ride.sum().unstack()
fest = {'2025-08-01': 'Pentaport', '2025-08-02': 'Pentaport', '2025-08-03': 'Pentaport', **{f'2025-08-{d}': 'Beer' for d in range(22, 31)},
        '2026-07-31': 'Pentaport', '2026-08-01': 'Pentaport', '2026-08-02': 'Pentaport', **{f'2026-08-{d}': 'Beer' for d in range(22, 31)}, '2026-09-12': 'Islive'}
def base(year):
    d = [x for x in bs.index if x.startswith(year) and x not in fest]
    wk = [x for x in d if pd.Timestamp(x).dayofweek >= 5]; wd = [x for x in d if pd.Timestamp(x).dayofweek < 5]
    return bs.loc[wd].median(), bs.loc[wk].median()
rows = []
sw = t[(t.station_id == 3139) & t.hour.between(20, 23)].pivot_table(index='date', columns='hour', values='boardings')
sbase = sw[(sw.index >= '2025-07-05') & (sw.index <= '2025-09-20') & ~sw.index.isin(list(fest))]
for d in sorted(x for x in bs.index if x in fest):
    y = d[:4]; bw, bk = base(y); bb = bk if pd.Timestamp(d).dayofweek >= 5 else bw
    r = dict(date=d, weekday=pd.Timestamp(d).day_name()[:3], event=fest[d])
    for h in (20, 21, 22, 23):
        r[f'bus_{h}'] = int(bs.loc[d, h]); r[f'bus_excess_{h}'] = round(float(bs.loc[d, h] - bb[h]), 1)
        if y == '2025':
            r[f'subway_{h}'] = int(sw.loc[d, h]); r[f'subway_excess_{h}'] = round(float(sw.loc[d, h] - sbase[h].median()), 0)
        else:
            r[f'v20e_{h}'] = round(float(te[(te.station_id == 3139) & (te.date == d) & (te.hour == h)].v20e.iloc[0]), 0)
    be = sum(max(r[f'bus_excess_{h}'], 0) for h in (20, 21, 22, 23)); r['bus_excess_20_23'] = be
    if y == '2025': r['subway_excess_20_23'] = sum(max(r[f'subway_excess_{h}'], 0) for h in (20, 21, 22, 23)); r['ratio'] = round(r['subway_excess_20_23'] / be, 1) if be > 5 else np.nan
    else: r['v20e_20_23'] = sum(r[f'v20e_{h}'] for h in (20, 21, 22, 23))
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv(OUT / 'songdo_festivals_bus_vs_subway_2025_and_2026.csv', index=False)
L.append(('Songdo 3139: bus excess (38696+38697) vs subway excess, 20-23h', S[[c for c in ['date', 'weekday', 'event', 'bus_excess_20_23', 'subway_excess_20_23', 'ratio', 'v20e_20_23', 'bus_21', 'bus_22', 'subway_22', 'v20e_22'] if c in S]]))
# ---- Bupyeong-samgeori: daytime 09-18, stops 39491+39493 ----
bp = b[b.ars.isin(['39491', '39493']) & b.hour.between(9, 18)].groupby('date').agg(ride=('ride', 'sum'), alight=('alight', 'sum'))
sp = t[(t.station_id == 3122) & t.hour.between(9, 18)].groupby('date').boardings.sum()
rows = []
for d in bp.index:
    r = dict(date=d, weekday=pd.Timestamp(d).day_name()[:3], bus_ride_09_18=int(bp.loc[d, 'ride']), bus_alight_09_18=int(bp.loc[d, 'alight']))
    if d <= '2026-06-30': r['subway_3122_09_18'] = int(sp.get(d, np.nan))
    else: r['v20e_3122_09_18'] = round(float(te[(te.station_id == 3122) & (te.date == d) & te.hour.between(9, 18)].v20e.sum()), 0)
    rows.append(r)
P = pd.DataFrame(rows); P.to_csv(OUT / 'bupyeong_chuseok_bus_vs_subway_2025_and_2026.csv', index=False)
L.append(('Bupyeong-samgeori 3122: daytime 09-18 bus (39491+39493) vs subway 2025 / v20e 2026', P))
with open(OUT / 'SUMMARY_TABLES.md', 'a') as f:
    for title, df in L: f.write(f'## {title}\n\n```\n{df.to_string(index=False)}\n```\n\n')
for title, df in L: print(title); print(df.to_string(index=False)); print()
