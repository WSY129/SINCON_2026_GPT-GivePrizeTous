"""9/26(추석 다음 토요일) 수준 판정: 버스 비율 -> 지하철 비율 환산.

입력
  --bus    stcis 행 단위 CSV (stcis_bus_hourly_long.csv 형식: ars,date,hour,next_day,ride,...)
  --train  대회 train.csv (지하철은 2026-06-30까지만 사용)
출력: 정류장·시간대별 비율 표와, 2026-09-26의 '추정 지하철 비율'(평소 토요일 대비)

방법
  1) 학습 기간 짝: 2025-10-04(추석 연휴 토) / 2025 평소 토(9/20, 9/27)
       bus_r25 = 버스 비율, sub_r25 = 같은 역 지하철 비율 (train)
       k = log(sub_r25) / log(bus_r25)   (버스 변화 1로그당 지하철 변화)
  2) 평가 기간: bus_r26 = 2026-09-26 / 2026 평소 토(9/5, 9/12, 9/19 중 있는 것)
       추정 지하철 비율 = exp(k * log(bus_r26))  (k가 불안정하면 k=1도 같이 출력)
  10/3(개천절 토)도 같은 방식으로 함께 출력한다.
"""
import argparse
import csv
import math
import statistics as st
from collections import defaultdict

BANDS = {'am(06-08)': (6, 7, 8), 'pm(17-18)': (17, 18), 'day(06-23)': tuple(range(6, 24))}
STOP_TO_STATION = {'41426': '3114', '37583': '3217', '38039': '3132', '38231': '3129', '11315': '3755',
                   '41290': '3113', '40336': '3119', '39317': '3222', '39318': '3225', '11238': '3756'}
BASE25 = ['2025-09-20', '2025-09-27']
HOL25 = '2025-10-04'
BASE26 = ['2026-09-05', '2026-09-12', '2026-09-19']
TARGETS26 = ['2026-09-26', '2026-10-03']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bus', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--out', default='result_0926.csv')
    a = ap.parse_args()

    bus = defaultdict(int)  # (ars, date, hour) -> ride
    for r in csv.DictReader(open(a.bus, encoding='utf-8-sig')):
        if r['ars'] in STOP_TO_STATION and r['next_day'] in ('False', 'false', '0'):
            bus[(r['ars'], r['date'], int(r['hour']))] += int(r['ride'])
    sub = defaultdict(int)
    want = set(STOP_TO_STATION.values())
    for r in csv.DictReader(open(a.train, encoding='utf-8-sig')):
        if r['station_id'] in want and r['date'] <= '2026-06-30':
            sub[(r['station_id'], r['date'], int(r['hour']))] += int(r['boardings'])

    def tot(d, key, date, hrs):
        vals = [d.get((key, date, h)) for h in hrs]
        return None if all(v is None for v in vals) else sum(v or 0 for v in vals)

    def ratio(d, key, date, bases, hrs):
        t = tot(d, key, date, hrs)
        b = [x for x in (tot(d, key, x, hrs) for x in bases) if x]
        return None if t is None or not b else t / st.median(b)

    rows = []
    for ars, sid in STOP_TO_STATION.items():
        for band, hrs in BANDS.items():
            b25 = ratio(bus, ars, HOL25, BASE25, hrs)
            s25 = ratio(sub, sid, HOL25, BASE25, hrs)
            k = None
            if b25 and s25 and abs(math.log(b25)) > 0.05:
                k = math.log(s25) / math.log(b25)
            for tgt in TARGETS26:
                b26 = ratio(bus, ars, tgt, BASE26, hrs)
                est_k = math.exp(k * math.log(b26)) if (k is not None and b26) else None
                rows.append(dict(ars=ars, station_id=sid, band=band, target=tgt,
                                 bus_ratio_2025_1004=b25, subway_ratio_2025_1004=s25, k=k,
                                 bus_ratio_target=b26, est_subway_ratio_k=est_k,
                                 est_subway_ratio_k1=b26))
    fmt = lambda v: '' if v is None else (f'{v:.3f}' if isinstance(v, float) else v)
    with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})

    print('band       target      n  median_bus_ratio  median_est(k)  median_est(k=1)')
    for band in BANDS:
        for tgt in TARGETS26:
            sel = [r for r in rows if r['band'] == band and r['target'] == tgt]
            br = [r['bus_ratio_target'] for r in sel if r['bus_ratio_target']]
            ek = [r['est_subway_ratio_k'] for r in sel if r['est_subway_ratio_k']]
            m = lambda xs: f'{st.median(xs):.3f}' if xs else '-'
            print(f'{band:10} {tgt}  {len(br):>2}  {m(br):>16}  {m(ek):>13}  {m(br):>15}')
    print('비교 기준(train 전체 역): 2025-10-04 아침 0.73, 2026-02-14 0.85. '
          'v10-트리 차이 0.17로그 = 약 19%.')


if __name__ == '__main__':
    main()
