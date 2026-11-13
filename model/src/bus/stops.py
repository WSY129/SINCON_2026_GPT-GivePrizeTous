"""Lists the bus stops of one district from one day of the MOLIT stop-by-route API (pages fetched in parallel).

    python -m src.bus.stops --sgg 28237 --day 20260603

Bus rows only (trfc_mns_se_cd == 'B'); rows of other modes are dropped before anything is kept.
Writes data/external_bus/stops_<sgg>_<day>.csv: one row per stop with its name, ARS number, dong and the day's boardings.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from .fetch_daily import call, items, OUT


def page(args):
    op, key, day, sgg, no = args
    for attempt in range(4):
        try:
            rows, total = items(call(op, key, pageNo=no, numOfRows=1000, opr_ymd=day, ctpv_cd=sgg[:2], sgg_cd=sgg)); return [r for r in rows if r.get('trfc_mns_se_cd') == 'B'], total
        except Exception:
            time.sleep(2 + 3 * attempt)
    return [], 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sgg', required=True); ap.add_argument('--day', default='20260603'); ap.add_argument('--threads', type=int, default=6); ap.add_argument('--op', default='getDailyStopbyRouteTripVolume')
    a = ap.parse_args(); key = os.environ['DATA_GO_KR_KEY']; t0 = time.time(); first, total = page((a.op, key, a.day, a.sgg, 1)); n = (total + 999) // 1000
    print(f'{a.sgg} {a.day}: {total} rows, {n} pages', flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        rest = list(ex.map(page, [(a.op, key, a.day, a.sgg, i) for i in range(2, n + 1)]))
    d = pd.DataFrame(first + [r for rows, _ in rest for r in rows]); OUT.mkdir(parents=True, exist_ok=True)
    g = d.groupby(['sttn_id', 'sttn_nm', 'sttn_ars_no', 'emd_nm'], dropna=False).agg(ride=('ride_nope', 'sum'), goff=('goff_nope', 'sum'), routes=('rte_id', 'nunique')).reset_index().sort_values('ride', ascending=False)
    g.to_csv(OUT / f'stops_{a.sgg}_{a.day}.csv', index=False, encoding='utf-8-sig'); print(f'{len(d)} bus rows, {len(g)} stops, {time.time() - t0:.0f}s; pages missing {sum(1 for rows, _ in rest if not rows)}'); print(g.head(12).to_string())


if __name__ == '__main__':
    main()
