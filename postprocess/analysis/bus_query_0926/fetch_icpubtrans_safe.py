"""인천시 대중교통 통계 API(15113194)·도로 교통량 API(15113145) 안전 조회기.

대회 규칙: 2026-07-01~10-31 지하철 승하차 수만 금지. 이 API에는 지하철 수치를 주는 기능이 섞여 있어서
아래처럼 막는다.
  - 버스 전용 기능(BusRidePsg_Route_InfoPage 등)과 도로 교통량(NodeLink_Trfc_DD): 모든 날짜 허용.
    응답 필드 이름에 subway/station/역사가 섞이면 저장하지 않고 멈춘다.
  - 지하철 포함 기능: PublicTransportPsg_Dong_InfoPage(시간대별 지하철 승객수), STAT-BusStop_Station_*(역사 단위),
    UserPattern_Dong_InfoPage, UserPatternInfo_v2. YMD가 20260630 이전일 때만 호출하고,
    응답에 그 이후 날짜가 하나라도 보이면 저장하지 않고 멈춘다.
    예) python3 fetch_icpubtrans_safe.py op --op PublicTransportPsg_Dong_InfoPage --ymd 20251007 20260218 --out dong.csv
키는 환경변수 DATA_GO_KR_KEY 에서만 읽고 출력·저장하지 않는다.

예)
  export DATA_GO_KR_KEY=...   # 직접 입력
  python3 fetch_icpubtrans_safe.py probe --ymd 20251007 --route 1       # 학습 기간 날짜로 먼저 시험
  python3 fetch_icpubtrans_safe.py ride --routes routes.txt --ymd 20260926 20260919 20251007 --out ride.csv
  python3 fetch_icpubtrans_safe.py road --ymd 20260926 20260919 --out road.csv
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

BUS = 'https://apis.data.go.kr/6280000/ICPubTrans_v2'
ROAD = 'https://apis.data.go.kr/6280000/ICRoadVolStat'
ALLOWED = {'BusRidePsg_Route_InfoPage': BUS, 'Bus_Route_BusStop_OPHour': BUS, 'Bus_Route_Section_OPHour': BUS,
           'BusAboardPsg_Route_InfoPage': BUS, 'NodeLink_Trfc_DD': ROAD}
# 지하철 수치가 섞인 기능: 2026-06-30 이전 날짜에만 부를 수 있다 (예측 기간 지하철 승하차 금지 규칙)
SUBWAY_OPS = {'PublicTransportPsg_Dong_InfoPage': BUS, 'STAT-BusStop_Station_Transfer_DD': BUS,
              'STAT-BusStop_Station_Passenger_DD': BUS, 'UserPattern_Dong_InfoPage': BUS, 'UserPatternInfo_v2': BUS}
CUTOFF = '20260630'
BAD_FIELD = re.compile(r'subway|station|역사|지하철|도시철도', re.I)
DATE_FIELD = re.compile(r'^(ymd|statDate|YMD)$')


def key():
    k = os.environ.get('DATA_GO_KR_KEY', '').strip()
    if not k:
        sys.exit('DATA_GO_KR_KEY 환경변수가 없다.')
    return k


def call(op, **params):
    subway = op in SUBWAY_OPS
    if not subway and op not in ALLOWED:
        sys.exit(f'미허용 기능: {op}')
    ymd = str(params.get('YMD', ''))
    if subway and not (re.fullmatch(r'\d{8}', ymd) and ymd <= CUTOFF):
        sys.exit(f'지하철 포함 기능 {op}는 {CUTOFF} 이전 날짜만 허용 (요청 YMD={ymd!r})')
    k = key()
    q = urllib.parse.urlencode(dict(params, serviceKey=k, pageNo=params.get('pageNo', 1), numOfRows=params.get('numOfRows', 1000), type='json'), safe='%')
    url = f'{(SUBWAY_OPS if subway else ALLOWED)[op]}/{op}?{q}'
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(url, timeout=30).read().decode('utf-8', 'ignore')
            break
        except Exception as e:
            msg = str(e).replace(k, '***')
            print(f'  재시도 {attempt + 1}: {msg[:120]}', file=sys.stderr); time.sleep(3 + 3 * attempt)
    else:
        return None, 'network'
    try:
        js = json.loads(raw)
    except ValueError:
        return None, raw.replace(k, '***')[:200]
    body = js.get('response', js).get('body', js.get('body', {}))
    items = body.get('items', [])
    if isinstance(items, dict):
        items = items.get('item', [])
    if isinstance(items, dict):
        items = [items]
    for it in items:
        for f, v in it.items():
            if not subway and BAD_FIELD.search(f):     # 버스·도로 기능 응답에 지하철 성격 필드가 섞이면 저장 금지
                sys.exit(f'응답에 지하철 성격 필드 {f!r} — 저장하지 않고 중단')
            if subway and DATE_FIELD.match(f) and re.sub(r'\D', '', str(v))[:8] > CUTOFF:
                sys.exit(f'지하철 포함 응답에 {CUTOFF} 이후 날짜 {v!r} — 저장하지 않고 중단')
    return items, body.get('totalCount')


def fetch_all(op, **params):
    out, page = [], 1
    while True:
        items, total = call(op, pageNo=page, **params)
        if items is None:
            return out, total
        out += items
        if not items or total is None or len(out) >= int(total):
            return out, total
        page += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['probe', 'ride', 'road', 'op'])
    ap.add_argument('--op', default='', help="mode op: 기능 이름. 지하철 포함 기능은 20260630 이전 --ymd만 허용")
    ap.add_argument('--ymd', nargs='+', required=True)
    ap.add_argument('--route', default='')
    ap.add_argument('--routes', default='')
    ap.add_argument('--out', default='out.csv')
    a = ap.parse_args()
    if a.mode == 'probe':
        for y in a.ymd:
            items, total = call('BusRidePsg_Route_InfoPage', YMD=y, routeName=a.route, numOfRows=5)
            print(y, 'total', total, '| 필드', sorted(items[0]) if items else items)
            if items:
                print('  예시', {k: items[0][k] for k in list(items[0])[:6]})
        return
    rows = []
    if a.mode == 'ride':
        routes = [r.strip() for r in open(a.routes, encoding='utf-8') if r.strip()] if a.routes else [a.route]
        for y in a.ymd:
            for r in routes:
                items, total = fetch_all('BusRidePsg_Route_InfoPage', YMD=y, routeName=r)
                rows += [dict(it, _ymd=y, _route=r) for it in items]
                print(y, r, len(items), flush=True); time.sleep(0.3)
    elif a.mode == 'op':
        if a.op in SUBWAY_OPS and any(not (re.fullmatch(r'\d{8}', y) and y <= CUTOFF) for y in a.ymd):
            sys.exit(f'{a.op}: 모든 --ymd가 {CUTOFF} 이전이어야 한다 (아무 것도 호출하지 않음)')
        for y in a.ymd:
            extra = dict(routeName=a.route) if a.route else {}
            items, total = fetch_all(a.op, YMD=y, **extra)
            rows += [dict(it, _ymd=y) for it in items]
            print(y, a.op, len(items), flush=True)
    else:
        for y in a.ymd:
            items, total = fetch_all('NodeLink_Trfc_DD', YMD=y)
            rows += [dict(it, _ymd=y) for it in items]
            print(y, len(items), flush=True)
    if rows:
        cols = sorted({k for r in rows for k in r})
        with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print('saved', len(rows), a.out)


if __name__ == '__main__':
    main()
