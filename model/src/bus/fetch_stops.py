"""Hourly bus boardings for chosen stops and days (MOLIT stop-by-route API, stop filter), bus rows only.

    python -m src.bus.fetch_stops --sgg 28237 --stops data/external_bus/pilot_stops.csv --start 20260228 --end 20260630

Writes data/external_bus/hourly_<sgg>.csv with date, sttn_id, hour, ride, goff (summed over routes and passenger types).
Rows of other transport modes are dropped before anything is kept.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from .fetch_daily import call, items, OUT


def one(args):
    op, key, sgg, stop, day = args; rows = []; no = 1
    while True:
        for attempt in range(4):
            try:
                got, total = items(call(op, key, pageNo=no, numOfRows=1000, opr_ymd=day, ctpv_cd=sgg[:2], sgg_cd=sgg, sttn_id=stop)); break
            except Exception as e:
                got, total = None, 0; time.sleep(2 + 4 * attempt)
        if got is None:
            return stop, day, None
        rows += [(r['tzon'], r['ride_nope'], r['goff_nope']) for r in got if r.get('trfc_mns_se_cd') == 'B']
        if no * 1000 >= total or not got:
            break
        no += 1
    return stop, day, rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sgg', required=True); ap.add_argument('--stops', required=True); ap.add_argument('--start', required=True); ap.add_argument('--end', required=True)
    ap.add_argument('--threads', type=int, default=6); ap.add_argument('--op', default='getDailyStopbyRouteTripVolume'); ap.add_argument('--tag', default='')
    a = ap.parse_args(); key = os.environ['DATA_GO_KR_KEY']; stops = pd.read_csv(a.stops, dtype=str); stops = stops[stops.sgg == a.sgg].sttn_id.tolist()
    path = OUT / f'hourly_{a.sgg}{a.tag}.csv'; done = set()
    if path.exists():
        old = pd.read_csv(path, dtype=dict(sttn_id=str, date=str)); done = set(zip(old.sttn_id, old.date))
    jobs = [(a.op, key, a.sgg, s, d) for d in pd.date_range(a.start, a.end).strftime('%Y%m%d') for s in stops if (s, d) not in done]
    print(f'{len(stops)} stops, {len(jobs)} stop-days to fetch', flush=True); t0 = time.time(); out = []; failed = 0
    with ThreadPoolExecutor(a.threads) as ex:
        for i, (stop, day, rows) in enumerate(ex.map(one, jobs)):
            if rows is None:
                failed += 1; continue
            d = pd.DataFrame(rows, columns=['hour', 'ride', 'goff']) if rows else pd.DataFrame(dict(hour=['00'], ride=[0], goff=[0]))
            g = d.groupby('hour', as_index=False).sum(); g.insert(0, 'sttn_id', stop); g.insert(0, 'date', day); out.append(g)
            if (i + 1) % 500 == 0:
                pd.concat(out).to_csv(path, mode='a', header=not path.exists(), index=False); out = []; print(f'{i + 1}/{len(jobs)} {time.time() - t0:.0f}s failed {failed}', flush=True)
    if out:
        pd.concat(out).to_csv(path, mode='a', header=not path.exists(), index=False)
    print(f'done {len(jobs)} in {time.time() - t0:.0f}s, failed {failed}', flush=True)


if __name__ == '__main__':
    main()
