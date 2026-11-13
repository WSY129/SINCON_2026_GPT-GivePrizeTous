"""서울시 지하철 호선별 역별 시간대별 승하차 (OA-12252, CardSubwayTime) 에서
7호선 부천·인천 구간 9개 역의 월별·시간대별 승하차를 받는다.
2022-01 부터는 인천교통공사로 넘어가 서울 데이터에서 빠짐 -> 2015-01 ~ 2021-12 만.
샘플키(sample)는 한 번에 5건까지만 되지만 역 이름으로 거르면 1건이라 충분함.
"""
import json, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/seoul_line7"; RAW.mkdir(parents=True, exist_ok=True)
STATIONS = {3753: "까치울", 3754: "부천종합운동장", 3755: "춘의", 3756: "신중동", 3757: "부천시청",
            3758: "상동", 3759: "삼산체육관", 3760: "굴포천", 3761: "부평구청"}
q = urllib.parse.quote

def get(ym, name):
    f = RAW / f"{ym}_{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    url = f"http://openapi.seoul.go.kr:8088/sample/json/CardSubwayTime/1/5/{ym}/{q('7호선')}/{q(name)}"
    for i in range(4):
        try:
            txt = urllib.request.urlopen(url, timeout=20).read().decode()
            js = json.loads(txt); f.write_text(txt); time.sleep(0.15); return js
        except Exception as e:
            time.sleep(2 * (i + 1))
    return {}

rows = []
for y in range(2015, 2022):
    for m in range(1, 13):
        ym = f"{y}{m:02d}"
        for sid, name in STATIONS.items():
            js = get(ym, name)
            for r in js.get("CardSubwayTime", {}).get("row", []):
                for h in range(0, 24):
                    on, off = r.get(f"HR_{h}_GET_ON_NOPE"), r.get(f"HR_{h}_GET_OFF_NOPE")
                    if on is None and off is None:
                        continue
                    rows.append(dict(month=f"{y}-{m:02d}", station_id=sid, station_name=name, hour=h,
                                     boardings=on, alightings=off))
df = pd.DataFrame(rows)
out = ROOT / "data/external/seoul_line7_monthly_hourly_2015_2021.csv"
df.to_csv(out, index=False)
print(out, len(df), df.month.nunique(), "months")
print(df.groupby(df.month.str[:4]).boardings.sum().astype(int))
