"""Section 1-2 tables of handoff_for_gemini from the bus-stop hourly boardings (stcis.go.kr) and the TRAINING-period subway
ridership (data/raw/train.csv, up to 2026-06-30).  No evaluation-period subway ridership exists here or is used.
    python outputs/handoff_for_gemini/scripts/bus_tables.py <stcis_long.csv> <out_dir>"""
import sys
import numpy as np, pandas as pd
from pathlib import Path

bus = pd.read_csv(sys.argv[1], dtype={'ars': str}); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
bus = bus[~bus.next_day]
tr = pd.read_csv('data/raw/train.csv')
def sub(sid, dates, hours):
    x = tr[(tr.station_id == sid) & tr.date.isin(dates) & tr.hour.isin(hours)]
    return x.pivot_table(index='date', columns='hour', values='boardings')
def b(ars, dates, hours, col='ride'):
    x = bus[(bus.ars == ars) & bus.date.isin(dates) & bus.hour.isin(hours)]
    return x.pivot_table(index='date', columns='hour', values=col, aggfunc='sum')
def wk(d): return pd.Timestamp(d).dayofweek >= 5
notes = []

# 1a. eight calibration days: hourly bus (ride, alight) at 37144 vs subway 3127 boardings and alightings, 16-23h, with baselines
cal = ['2026-05-30', '2026-06-02', '2026-06-03', '2026-06-04', '2026-06-05', '2026-06-06', '2026-06-07', '2026-06-13']
H = list(range(14, 24))
win = sorted(bus[(bus.ars == '37144') & (bus.date >= '2026-05-30') & (bus.date <= '2026-06-13')].date.unique())
base_days = [d for d in win if d not in cal]
rows = []
for d in cal:
    same = [x for x in base_days if wk(x) == wk(d)]
    for h in H:
        g = lambda df: (float(df.loc[d, h]) if (d in df.index and h in df.columns) else np.nan)
        br, ba = b('37144', [d], [h]), b('37144', [d], [h], 'alight')
        sb = tr[(tr.station_id == 3127) & (tr.date == d) & (tr.hour == h)]
        sbase = tr[(tr.station_id == 3127) & tr.date.isin(same) & (tr.hour == h)]
        bbase = bus[(bus.ars == '37144') & bus.date.isin(same) & (bus.hour == h)]
        rows.append(dict(date=d, weekday=pd.Timestamp(d).day_name()[:3], hour=h, bus_ride=g(br), bus_alight=g(ba), bus_ride_base_median=float(bbase.ride.median()),
                         subway_board=int(sb.boardings.iloc[0]) if len(sb) else np.nan, subway_alight=int(sb.alightings.iloc[0]) if len(sb) else np.nan,
                         subway_board_base_median=float(sbase.boardings.median()), base_days=len(same)))
c = pd.DataFrame(rows); c['bus_excess'] = c.bus_ride - c.bus_ride_base_median; c['subway_excess'] = c.subway_board - c.subway_board_base_median
c.to_csv(OUT / 'munhak_37144_vs_3127_calibration_days_hourly.csv', index=False)
s = c[c.hour.between(16, 23)].groupby('date')[['bus_excess', 'subway_excess']].apply(lambda q: q.clip(lower=0).sum())
s['ratio_subway_per_bus'] = (s.subway_excess / s.bus_excess).round(2); notes.append(('night/evening games, sum of positive excess 16-23h', s))

# 1b. day games at 14:00 (exit about 17h): 37144 bus vs 3127 subway, 15-19h, baseline = non-game days of the same day type in the same windows
games = ['2026-04-26', '2026-05-02', '2026-05-03', '2026-05-17', '2026-06-06', '2026-03-28', '2026-03-29']
allb = sorted(bus[bus.ars == '37144'].date.unique()); known_crowd = set(cal + games + ['2026-04-04', '2026-04-05', '2026-04-11', '2026-05-05'])
rows = []
for d in games:
    if d not in allb: continue
    pool = [x for x in allb if x not in known_crowd and wk(x) == wk(d) and abs((pd.Timestamp(x) - pd.Timestamp(d)).days) <= 28 and x <= '2026-06-30']
    for h in range(15, 20):
        bb = bus[(bus.ars == '37144') & (bus.date == d) & (bus.hour == h)]; bp = bus[(bus.ars == '37144') & bus.date.isin(pool) & (bus.hour == h)]
        sb = tr[(tr.station_id == 3127) & (tr.date == d) & (tr.hour == h)]; sp = tr[(tr.station_id == 3127) & tr.date.isin(pool) & (tr.hour == h)]
        rows.append(dict(date=d, weekday=pd.Timestamp(d).day_name()[:3], hour=h, bus_ride=int(bb.ride.sum()), bus_alight=int(bb.alight.sum()), bus_base=float(bp.ride.median()) if len(bp) else np.nan,
                         subway_board=int(sb.boardings.sum()) if len(sb) else np.nan, subway_base=float(sp.boardings.median()) if len(sp) else np.nan, base_days=len(pool)))
dg = pd.DataFrame(rows)
if len(dg):
    dg['bus_excess'] = dg.bus_ride - dg.bus_base; dg['subway_excess'] = dg.subway_board - dg.subway_base
    dg.to_csv(OUT / 'munhak_37144_vs_3127_day_games_hourly.csv', index=False)
    s2 = dg.groupby('date')[['bus_excess', 'subway_excess']].apply(lambda q: q.clip(lower=0).sum()); s2['ratio_subway_per_bus'] = (s2.subway_excess / s2.bus_excess).round(2)
    notes.append(('day games (14:00 start), sum of positive excess 15-19h', s2))

# 1c. Munhak complex: every stop, every queried day, ride 14-23h and day total, with the event kind of the day
ev = {'2026-04-04': 'concert(main)', '2026-04-05': 'concert(main)', '2026-04-11': 'concert(main)', '2026-08-29': 'Sustainable Wave(main)', '2026-08-30': 'Sustainable Wave(main)',
      '2026-09-12': 'PLAVE(main)', '2026-09-13': 'PLAVE(main)', '2026-10-17': 'fan match(ballpark)', '2026-10-31': 'day game(ballpark)'}
for d in cal: ev.setdefault(d, 'baseball(ballpark)')
for d in games: ev.setdefault(d, 'day game(ballpark)')
mk = bus[bus.ars.isin(['37144', '37157', '38310', '38311', '38329', '38423'])]
w = mk.pivot_table(index=['ars', 'stop_name', 'date', 'weekday'], columns='hour', values='ride', aggfunc='sum').reset_index()
w['day_total_ride'] = mk.groupby(['ars', 'date']).ride.sum().reindex(pd.MultiIndex.from_frame(w[['ars', 'date']])).to_numpy()
w['event'] = w.date.map(ev).fillna('')
w[['ars', 'stop_name', 'date', 'weekday', 'event'] + [h for h in range(14, 24)] + ['day_total_ride']].to_csv(OUT / 'munhak_complex_stops_ride_by_day.csv', index=False, encoding='utf-8-sig')
evd = w[w.event != '']
cmp = evd.assign(evening=evd[[20, 21, 22, 23]].sum(1), afternoon=evd[[16, 17, 18]].sum(1)).groupby(['event', 'ars', 'stop_name'])[['evening', 'afternoon']].mean().round(0)
base = w[w.event == ''].assign(evening=lambda q: q[[20, 21, 22, 23]].sum(1), afternoon=lambda q: q[[16, 17, 18]].sum(1)).groupby('ars')[['evening', 'afternoon']].median()
cmp = cmp.join(base, on='ars', rsuffix='_ordinary_median'); cmp.to_csv(OUT / 'munhak_complex_event_vs_ordinary.csv', encoding='utf-8-sig'); notes.append(('Munhak complex stops: mean ride 20-23h / 16-18h on event days vs ordinary-day median', cmp))

# 1d. Seonhak spring festival vs Neungheodae (October): ride and alight 18-23h at 38305/38306, subway 3128 for the spring days (training)
sh = bus[bus.ars.isin(['38305', '38306']) & bus.hour.between(12, 23)]
t = sh.pivot_table(index=['ars', 'stop_name', 'date', 'weekday'], columns='hour', values=['ride', 'alight'], aggfunc='sum')
t.columns = [f'{a}_{h}' for a, h in t.columns]; t = t.reset_index()
t['festival'] = t.date.map({'2026-03-28': 'spring festival', '2026-03-29': 'spring festival', '2026-10-09': '능허대', '2026-10-10': '능허대'}).fillna('')
t.to_csv(OUT / 'seonhak_38305_38306_ride_alight_12_23.csv', index=False, encoding='utf-8-sig')
s3 = sub(3128, sorted(set(t.date[t.date <= '2026-06-30'])), list(range(12, 24))); s3.to_csv(OUT / 'seonhak_3128_subway_boardings_training_days.csv')

# 2. the six gaps: daily totals and hourly tables per stop group (bus only for the evaluation period)
def group(ars_list, name, hours=range(6, 24)):
    x = bus[bus.ars.isin(ars_list)]
    if not len(x): return
    hw = x.pivot_table(index=['date', 'weekday'], columns='hour', values='ride', aggfunc='sum').reset_index()
    tot = x.groupby(['date']).agg(ride=('ride', 'sum'), alight=('alight', 'sum')).reset_index()
    hw = hw.merge(tot, on='date'); hw.to_csv(OUT / f'gap_{name}_ride_by_hour_and_day_total.csv', index=False, encoding='utf-8-sig')
    per = x.groupby(['ars', 'stop_name', 'date', 'weekday']).agg(ride=('ride', 'sum'), alight=('alight', 'sum')).reset_index(); per.to_csv(OUT / f'gap_{name}_per_stop_day_totals.csv', index=False, encoding='utf-8-sig')
    return hw
dal = group(['38696', '38697', '38801', '38824'], 'songdo_dalbit')
idg = group(['39395', '39398', '39471'], 'incheon_grand_park')
gd = group(['89584', '89585', '96008', '96009'], 'geomdan_lake_park')
bc = group(['11373', '13295', '13297'], 'bucheon_stadium')
ju = group(['37583', '37584', '40841'], 'juan_bupyeong_summer_leave')
bp = group(['39491', '39493'], 'bupyeong_samgeori_family_park')
if gd is not None:                                                        # weekday trend vs June, total and 06-09h
    g = gd[~gd.weekday.isin(['토', '일'])].copy(); g['month'] = g.date.str[:7]; g['am_06_09'] = g[[6, 7, 8, 9]].sum(1)
    m = g.groupby('month')[['ride', 'alight', 'am_06_09']].mean(); m = (m / m.iloc[0]).round(3).add_suffix('_vs_june'); m.to_csv(OUT / 'gap_geomdan_weekday_trend_vs_june.csv'); notes.append(('Geomdan stops, weekday mean relative to June', m))
if ju is not None:
    per = pd.read_csv(OUT / 'gap_juan_bupyeong_summer_leave_per_stop_day_totals.csv', dtype={'ars': str}); per = per[~per.weekday.isin(['토', '일'])]
    per['period'] = np.select([per.date <= '2026-07-24', per.date.between('2026-07-27', '2026-07-31'), per.date.between('2026-08-03', '2026-08-07')], ['mid-July (7/13-24)', 'leave week 1 (7/27-31)', 'leave week 2 (8/3-7)'], '')
    q = per[per.period != ''].groupby(['ars', 'stop_name', 'period'])[['ride', 'alight']].mean().unstack('period')
    q.to_csv(OUT / 'gap_summer_leave_weekday_means.csv', encoding='utf-8-sig'); notes.append(('Summer leave: weekday mean per stop', q.round(0)))
    s25 = tr[tr.station_id.isin([3217, 3118, 3761]) & tr.date.between('2025-07-14', '2025-08-08')].copy(); s25['dow'] = pd.to_datetime(s25.date).dt.dayofweek
    s25 = s25[s25.dow < 5]; s25['period'] = np.select([s25.date <= '2025-07-25', s25.date.between('2025-07-28', '2025-08-01'), s25.date.between('2025-08-04', '2025-08-08')], ['mid-July 2025', 'leave week 1 2025', 'leave week 2 2025'], '')
    s25 = s25[s25.period != ''].groupby(['station_id', 'period', 'date']).boardings.sum().groupby(['station_id', 'period']).mean().unstack().round(0); s25.to_csv(OUT / 'gap_summer_leave_subway_2025_training.csv'); notes.append(('Same stations, subway 2025 (training) weekday day totals', s25))

with open(OUT / 'SUMMARY_TABLES.md', 'w') as f:
    for title, df in notes:
        f.write(f'## {title}\n\n```\n{df.to_string()}\n```\n\n')
print(open(OUT / 'SUMMARY_TABLES.md').read()[:6000])
