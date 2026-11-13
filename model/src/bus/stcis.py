"""Hourly BUS boardings / alightings of single stops from the transport-card statistics site (https://www.stcis.go.kr,
노선·정류장 지표 > 정류장별 이용량), the same two requests the public page makes.  No login.

    python -m src.bus.stcis --start 2025-08-01 --end 2026-10-31          # writes data/external_bus/stcis/<ARS>.csv

Only stops of kind bus are queried (the stop search also lists the subway station, which is never requested).
One request at a time with a pause; windows never cross a month, the longest is 14 days (site limit).
"""
import argparse
import re
import time
from pathlib import Path
import http.cookiejar
import urllib.parse
import urllib.request
import pandas as pd

BASE = 'https://www.stcis.go.kr'
OUT = Path('data/external_bus/stcis')
STOPS = [('28', '37144', 3127), ('28', '42319', 3209), ('28', '42289', 3210), ('28', '38697', 3139), ('41', '13296', 3754), ('28', '38306', 3128), ('28', '38305', 3128),
         ('28', '38395', 3136), ('28', '39395', 3226), ('28', '40209', 3120), ('28', '40336', 3119), ('28', '40539', 3759), ('28', '39292', 3124), ('28', '39207', 3125),
         ('28', '38563', 3137), ('41', '55057', 3757), ('28', '39491', 3122), ('28', '38619', 3138)]
TPL = dict(indiCd='Z01723', indiNm='노선·정류장 지표(정류장별 이용량)', siteGb='P', pgngYn='N', rdAlocSel='Y', rdStgptSel='Y', searchDateGubun='3', daybyTblNm='DM_STTNBY_USECNT_001',
           mnbyTblNm='DM_MMBY_STTNBY_USECNT_001', dstrctTblNm='', mnbyDstrctTblNm='', yrbyDstrctTblNm='', yrbyTblNm='', excclcAreaCd='', routeId='', routeSdCd='', routeSggCd='',
           searchAlocZoneEmd='', searchAlocZoneSd='', searchAlocZoneSgg='', searchODAreaGubun='', searchODAreaGubun_2='', searchStgptZoneEmd='', searchStgptZoneSd='', searchStgptZoneSgg='',
           selectZoneSd='', selectZoneSgg='', sttnIdGrp='', sttnSdCd='', tcboId='', zoneDstrct='', zoneEmd='', zoneSd='', zoneSgg='', searchFromYear='2025', searchToYear='2025')


class Site:
    def __init__(self, pause=2.0):
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())); self.pause = pause
        self.op.addheaders = [('User-Agent', 'Mozilla/5.0')]
        self.op.open(f'{BASE}/pivotIndi/wpsPivotIndicator.do?siteGb=P&indiClss=IC03&indiSel=IC0308', timeout=60).read()

    def post(self, path, data):
        body = urllib.parse.urlencode(data).encode('utf-8')
        for attempt in range(4):
            try:
                req = urllib.request.Request(BASE + path, data=body, headers={'X-Requested-With': 'XMLHttpRequest', 'Referer': f'{BASE}/pivotIndi/wpsPivotIndicator.do',
                                                                              'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'})
                text = self.op.open(req, timeout=90).read().decode('utf-8', 'ignore'); time.sleep(self.pause); return text
            except Exception:
                time.sleep(3 + 5 * attempt)
        raise RuntimeError(f'failed {path}')

    def lookup(self, sd, ars, day):
        html = self.post('/pivotIndi/sttnListAjax.do', dict(searchDateGubun='3', searchFromDay=day, searchToDay=day, searchFromMonth=day[:7], searchToMonth=day[:7], searchFromYear=day[:4],
                                                             searchToYear=day[:4], searchPopSttnZoneSd=sd, searchPopSttnZoneSgg='', searchPopSttnZoneEmd='', popupSearchSttnNma='', popupSearchSttnArsno=ars, indiCd='Z01723'))
        vals = re.findall(r'name=["\']chkSttn["\'][^>]*value=["\']([^"\']+)["\']', html) + re.findall(r'value=["\']([^"\']+)["\'][^>]*name=["\']chkSttn["\']', html)
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S)
        out = []
        for v in dict.fromkeys(vals):
            p = v.split('|')
            if len(p) >= 7 and p[5] == ars:
                row = next((r for r in rows if v in r), '')
                if '지하철' not in row:                                       # bus stops only
                    out.append(p)
        return out

    def fetch(self, p, f, t):
        data = dict(TPL, sttnId=p[2], sttnSggCd=p[3], tcboIdSttn=p[0], excclcAreaCdSttn=p[1], searchFromDay=f, searchToDay=t, searchFromDayDD=f.replace('-', ''), searchFromMonth=f[:7], searchToMonth=t[:7])
        if p[6] != '1':
            data.update(sttnId='', sttnIdGrp=p[2])
        html = self.post('/pivotIndi/indicatorAjax.do', data)
        text = re.sub(r'<script.*?</script>|<style.*?</style>', ' ', html, flags=re.S); text = re.sub(r'<[^>]+>', ' ', text); text = re.sub(r'\s+', ' ', text)
        text = text[text.rfind('하차') + 2:]
        days = re.findall(r'(\d{4}-\d{2}-\d{2})\(.\)', text); nums = [int(x) for x in re.findall(r'-?\d+', text[text.rfind(')') + 1:])]
        rows = []
        for i, d in enumerate(days):
            v = nums[i * 48:(i + 1) * 48]
            if len(v) == 48:
                for j, h in enumerate(list(range(4, 24)) + list(range(0, 4))):
                    rows.append((d, h, v[2 * j], v[2 * j + 1]))
        return rows


def windows(start, end):
    for m in pd.period_range(start[:7], end[:7], freq='M'):
        last = m.days_in_month
        for a, b in ((1, 14), (15, 28), (29, last)):
            if a <= last:
                f, t = f'{m}-{a:02d}', f'{m}-{min(b, last):02d}'
                if f <= end and t >= start:
                    yield str(m), f, t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--start', default='2025-08-01'); ap.add_argument('--end', default='2026-10-31'); ap.add_argument('--stops', default='')
    ap.add_argument('--list', default='', help='csv with columns city, ars, station_id (extra stops)')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); site = Site(); want = set(a.stops.split(',')) if a.stops else None
    stops = STOPS if not a.list else [(str(r.city), str(r.ars).zfill(5), int(r.station_id)) for r in pd.read_csv(a.list, dtype=dict(city=str, ars=str)).itertuples()]
    for sd, ars, station in stops:
        if want and ars not in want:
            continue
        path = OUT / f'{ars}.csv'; have = set(pd.read_csv(path, dtype=str).date) if path.exists() else set(); ids = {}; added = 0; t0 = time.time()
        for month, f, t in windows(a.start, a.end):
            if all(str(d.date()) in have for d in pd.date_range(f, t)):
                continue
            if month not in ids:
                ids[month] = site.lookup(sd, ars, f'{month}-01')
            if not ids[month]:
                print(f'{ars} {month}: stop not found', flush=True); continue
            rows = [(station, ars, ids[month][0][2], *r) for r in site.fetch(ids[month][0], f, t)]
            if rows:                                                         # written window by window, so an interruption loses nothing
                pd.DataFrame(rows, columns=['station_id', 'ars', 'sttn_id', 'date', 'hour', 'ride', 'goff']).to_csv(path, mode='a', header=not path.exists(), index=False); added += len(rows) // 24
        print(f'{ars} (station {station}): {added} days added, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
