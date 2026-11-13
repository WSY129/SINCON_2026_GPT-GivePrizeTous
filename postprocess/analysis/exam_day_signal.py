"""학교 시험 날 = 하교가 12~13시로 당겨지는 날. 역별로 train에서 찾아 두고, 역 근처 학교와 2026 시험 날짜를 붙인다.

신호: log((12시+13시 승차) / (16시+17시 승차)) 를 같은 역 앞뒤 21평일 중앙값과 비교 (공휴일·주말 제외).
검바위는 시험 날 12시 승차가 평소의 약 4.3배, 16시는 0.46배.

Output:
  data/external/exam_signal_station_day.csv      역 × 평일 신호 (train 2025-01 ~ 2026-06)
  data/external/schools_near_exam_stations.csv   신호가 있는 역 700m 안 중·고 (OSM) + NEIS 시험 날짜(24번 파일)
  data/external/raw/osm/schools_near_exam_stations.json   OSM 원본 (Overpass)
"""
import json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external"
RAW = EXT / "raw/osm"
RAW.mkdir(parents=True, exist_ok=True)
STATIONS = [3208, 3122, 3224, 3209, 3206, 3204, 3216, 3215]  # 신호가 시험 기간에 몰린 역

tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
cal = pd.read_csv(EXT / "calendar_2015_2026.csv", parse_dates=["date"])
meta = pd.read_csv(ROOT / "data/metaData.csv").set_index("station_id").station_name
hol = set(cal.date[cal.is_holiday == 1])

# 1) 신호
x = tr[(~tr.date.isin(hol)) & (tr.date.dt.day_of_week < 5)]
p = x.pivot_table(index=["station_id", "date"], columns="hour", values="boardings")
sig = np.log((p[12] + p[13] + 1) / (p[16] + p[17] + 1)).rename("raw").reset_index()
sig["base"] = sig.groupby("station_id").raw.transform(lambda v: v.rolling(21, center=True, min_periods=8).median())
sig["signal"] = sig.raw - sig.base
sig["station_name"] = sig.station_id.map(meta)
sig["exam_like"] = sig.signal > 0.4
sig.round(4).to_csv(EXT / "exam_signal_station_day.csv", index=False)

# 2) 역 근처 학교 (OSM)
raw_f = RAW / "schools_near_exam_stations.json"
c = pd.read_csv(EXT / "station_coords.csv")
cs = c[c.station_id.isin(STATIONS)]
if not raw_f.exists():
    q = "[out:json][timeout:60];(" + "".join(
        f'nwr(around:700,{r.lat},{r.lon})["amenity"="school"];' for r in cs.itertuples()) + ");out center tags;"
    out = subprocess.run(["curl", "-s", "-m", "90", "-A", "sincon-analysis/1.0", "--data-urlencode", "data@-",
                          "https://overpass-api.de/api/interpreter"], input=q, capture_output=True, text=True).stdout
    raw_f.write_text(out)
j = json.loads(raw_f.read_text())
rows = []
for e in j["elements"]:
    nm = e.get("tags", {}).get("name", "")
    if not ("중학교" in nm or "고등학교" in nm):
        continue
    lat = e.get("lat", e.get("center", {}).get("lat"))
    lon = e.get("lon", e.get("center", {}).get("lon"))
    d = np.hypot((cs.lat - lat) * 111000, (cs.lon - lon) * 88000)
    k = d.idxmin()
    rows.append(dict(station_id=int(cs.loc[k, "station_id"]), station_name=cs.loc[k, "station_name"],
                     school=nm, dist_m=int(d.min())))
sc = pd.DataFrame(rows).sort_values(["station_id", "dist_m"]).drop_duplicates(["station_id", "school"])

ev = pd.read_csv(EXT / "school_events_2025_2026.csv", parse_dates=["date"])
ex = ev[ev.EVENT_NM.str.contains("시험|고사|평가", na=False)
        & ~ev.EVENT_NM.str.contains("학력평가|모의|연합|수행|진단|성취도|체력|수능|직업기초", na=False)]
names = set(ev.SCHUL_NM)


def ds(nm, y):
    g = ex[ex.SCHUL_NM == nm]
    return " ".join(d.strftime("%m-%d") for d in sorted(g.date.unique()) if d.year == y)


sc["in_neis_file"] = sc.school.isin(names)
sc["neis_exam_2025"] = sc.school.map(lambda n: ds(n, 2025))
sc["neis_exam_2026"] = sc.school.map(lambda n: ds(n, 2026))
sc["source"] = "OpenStreetMap (Overpass) © OpenStreetMap contributors (ODbL); 시험 날짜 = school_events_2025_2026.csv (NEIS)"
sc.to_csv(EXT / "schools_near_exam_stations.csv", index=False)

print(sig[sig.exam_like & sig.station_id.isin(STATIONS)].groupby("station_name").size().sort_values(ascending=False))
print(sc.to_string())
