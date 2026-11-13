"""Daily boardings per bus stop from the MOLIT transport-card OpenAPI on data.go.kr (dataset 15142065,
"국토교통부_정류장별 경유노선별 이용량", end point https://apis.data.go.kr/1613000/StopbyRouteTripVolume).

NOT TESTED: no service key was available when this was written.  The operation path and the response fields are not
printed on the dataset page, so start with --probe and look at what comes back.

    export DATA_GO_KR_KEY=...            (the key stays in your shell; it is never written to a file or printed)
    python -m src.bus.fetch_daily --probe --day 20250801 --sgg 28260
    python -m src.bus.fetch_daily --start 20250701 --end 20261031 --mode-field <field> --bus-value <value>

Rule of the competition: subway ridership of the evaluation period must not be used.  The dataset description says a
"stop" can also be a subway station, so rows are written only when --mode-field / --bus-value identify them as bus rows;
everything else is dropped before anything is saved.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
import pandas as pd

BASE = 'https://apis.data.go.kr/1613000/StopbyRouteTripVolume'
OUT = Path('data/external_bus')
SGG = {'28': ['28110', '28140', '28177', '28185', '28200', '28237', '28245', '28260'], '41': ['41192', '41194', '41196']}      # Incheon districts (before the July 2026 reorganisation), Bucheon


def call(op, key, **params):
    q = urllib.parse.urlencode(dict(serviceKey=key, dataType='JSON', **params), safe='%')
    with urllib.request.urlopen(f'{BASE}/{op}?{q}', timeout=60) as r:
        return json.loads(r.read().decode('utf-8'))


def items(payload):
    root = payload.get('Response', payload.get('response', payload)); body = root.get('body', {}); it = body.get('items', {})
    it = it.get('item', it) if isinstance(it, dict) else it
    return (it if isinstance(it, list) else [it] if it else []), int(body.get('totalCount', 0) or 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--op', default='getDailyStopbyRouteTripVolume'); ap.add_argument('--probe', action='store_true')
    ap.add_argument('--day', default='20250801'); ap.add_argument('--sgg', default='28260'); ap.add_argument('--start'); ap.add_argument('--end')
    ap.add_argument('--mode-field'); ap.add_argument('--bus-value'); ap.add_argument('--rows', type=int, default=1000); ap.add_argument('--stop', default='')
    ap.add_argument('--probe-all', action='store_true'); ap.add_argument('--probe-dates', default='')
    a = ap.parse_args(); key = os.environ['DATA_GO_KR_KEY']
    if a.probe_dates:
        for day in a.probe_dates.split(','):
            try:
                t0 = time.time(); p = call(a.op, key, pageNo=1, numOfRows=1, opr_ymd=day, ctpv_cd=a.sgg[:2], sgg_cd=a.sgg); got, total = items(p)
                print(f'DATE {day}: ' + (f'available, {total} rows' if got else json.dumps(p, ensure_ascii=False)[:120]) + f' ({time.time() - t0:.1f}s)', flush=True)
            except Exception as e:
                print(f'DATE {day}: ERROR {type(e).__name__} {str(e)[:160]}', flush=True)
        return
    if a.probe_all:
        for day, rows, stop in (('20250801', 1000, ''), ('20250801', 10000, ''), ('20250801', 100000, ''), ('20260915', 10, ''), ('20261030', 10, ''), ('20250801', 1000, '2800358')):
            try:
                t0 = time.time(); p = call(a.op, key, pageNo=1, numOfRows=rows, opr_ymd=day, ctpv_cd=a.sgg[:2], sgg_cd=a.sgg, **({'sttn_id': stop} if stop else {})); got, total = items(p)
                body = p.get('Response', p.get('response', {})).get('body', {}); head = p.get('Response', p.get('response', {})).get('header', {})
                d = pd.DataFrame(got); extra = f"modes {d.trfc_mns_se_cd.value_counts().to_dict()} hours {sorted(d.tzon.unique())[0]}-{sorted(d.tzon.unique())[-1]} stops {d.sttn_id.nunique()} routes {d.rte_id.nunique()} user types {d.users_type_cd.nunique()} ride sum {int(d.ride_nope.sum())}" if len(d) else json.dumps(p, ensure_ascii=False)[:300]
                print(f'PROBE day {day} asked {rows} stop {stop or "all"}: {time.time() - t0:.1f}s header {head} body {dict((k, v) for k, v in body.items() if k != "items")} parsed total {total} rows {len(got)} | {extra}', flush=True)
            except Exception as e:
                print(f'PROBE day {day} asked {rows} stop {stop or "all"}: ERROR {type(e).__name__} {str(e)[:200]}', flush=True)
        return
    if a.probe:
        t0 = time.time(); p = call(a.op, key, pageNo=1, numOfRows=a.rows, opr_ymd=a.day, ctpv_cd=a.sgg[:2], sgg_cd=a.sgg, **({'sttn_id': a.stop} if a.stop else {})); rows, total = items(p)
        body = p.get('Response', p.get('response', {})).get('body', {})
        print(f'day {a.day} district {a.sgg} stop {a.stop or "all"}: totalCount {total}, rows in this page {len(rows)} (asked {a.rows}), {time.time() - t0:.1f}s; body keys {[k for k in body if k != "items"]} {dict((k, body[k]) for k in body if k != "items")}')
        if rows:
            d = pd.DataFrame(rows); print('fields:', list(d.columns)); print('modes:', d.trfc_mns_se_cd.value_counts().to_dict() if 'trfc_mns_se_cd' in d else '-', '| hours:', sorted(d.tzon.unique())[:30] if 'tzon' in d else '-', '| stops', d.sttn_id.nunique(), 'routes', d.rte_id.nunique(), 'user types', d.users_type_cd.nunique() if 'users_type_cd' in d else '-')
            print(d.head(2).to_string())
        else:
            print(json.dumps(p, ensure_ascii=False)[:600])
        return
    assert a.mode_field and a.bus_value, 'name the field and value that mark bus rows (see --probe)'
    OUT.mkdir(parents=True, exist_ok=True)
    for day in pd.date_range(a.start, a.end).strftime('%Y%m%d'):
        path = OUT / f'{day}.csv'
        if path.exists():
            continue
        keep = []
        for ctpv, codes in SGG.items():
            for sgg in codes:
                page = 1
                while True:
                    rows, total = items(call(a.op, key, pageNo=page, numOfRows=a.rows, opr_ymd=day, ctpv_cd=ctpv, sgg_cd=sgg))
                    keep += [r for r in rows if str(r.get(a.mode_field)) == a.bus_value]          # bus rows only; subway rows never reach the disk
                    if page * a.rows >= total or not rows:
                        break
                    page += 1; time.sleep(0.05)
        pd.DataFrame(keep).to_csv(path, index=False, encoding='utf-8-sig'); print(day, len(keep), flush=True)


if __name__ == '__main__':
    main()
