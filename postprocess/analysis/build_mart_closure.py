"""대형마트 의무휴업일(매월 둘째·넷째 일요일) 달력 + 역별 효과 (train으로 추정).
근거: 인천은 대형마트가 둘째·넷째 일요일에 쉬어요 (경기일보 2022-12-11, 롯데마트 인천터미널점만 예외로 보도).
      부천시 2026년 고시는 찾지 못했지만 train에서 부천시청·굴포천도 같은 날 줄어요.
효과: 공휴일이 아닌 일요일만 써서, 역 × 달 평균을 뺀 log(승차)를 '의무휴업 일요일 vs 다른 일요일'로 비교.
Output:
  data/external/mart_closure_calendar.csv           date, is_mart_closed_sunday, nth_sunday, confidence, note  (2025-01 ~ 2026-10 일요일)
  data/external/mart_closure_effect_by_station.csv  station_id, station_name, band(하루/11-14/15-18/19-23), pct, t, n_closed, n_open
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv", parse_dates=["date"])
hol = set(cal[cal.is_holiday == 1].date)

# 1) 달력
sun = pd.DataFrame({"date": pd.date_range("2025-01-01", "2026-10-31")})
sun = sun[sun.date.dt.day_of_week == 6].copy()
sun["nth_sunday"] = (sun.date.dt.day - 1) // 7 + 1
sun["is_mart_closed_sunday"] = sun.nth_sunday.isin([2, 4]).astype(int)
sun["confidence"] = "high"
sun["note"] = ""
m = sun.date == "2026-09-27"
sun.loc[m, ["confidence", "note"]] = ["medium", "추석 연휴 직후 일요일. 9/27 영업 여부는 지자체별로 다르다고 보도됨(인천·부천 점포별 확인 못 함)"]
m = sun.date.isin(hol)
sun.loc[m, "note"] = sun.loc[m, "note"] + " 공휴일과 겹침"
sun["date"] = sun.date.dt.strftime("%Y-%m-%d")
sun[["date", "nth_sunday", "is_mart_closed_sunday", "confidence", "note"]].to_csv(
    ROOT / "data/external/mart_closure_calendar.csv", index=False)

# 2) 역별 효과
t = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
meta = pd.read_csv(ROOT / "data/metaData.csv")[["station_id", "station_name"]]
t = t[(t.date.dt.day_of_week == 6) & (~t.date.isin(hol))]
bands = {"하루": (6, 23), "11-14": (11, 14), "15-18": (15, 18), "19-23": (19, 23)}
out = []
for b, (lo, hi) in bands.items():
    d = t[(t.hour >= lo) & (t.hour <= hi)].groupby(["date", "station_id"]).boardings.sum().reset_index()
    d["closed"] = (((d.date.dt.day - 1) // 7 + 1).isin([2, 4])).astype(int)
    d["lb"] = np.log1p(d.boardings)
    d["r"] = d.lb - d.groupby(["station_id", d.date.dt.to_period("M")]).lb.transform("mean")
    g = d.groupby(["station_id", "closed"]).r.agg(["mean", "size"]).unstack()
    sd = d.groupby("station_id").r.std()
    diff = g[("mean", 1)] - g[("mean", 0)]
    n1, n0 = g[("size", 1)], g[("size", 0)]
    out.append(pd.DataFrame({"band": b, "pct": ((np.exp(diff) - 1) * 100).round(1),
                             "t": (diff / (sd * np.sqrt(1 / n1 + 1 / n0))).round(1),
                             "n_closed": n1, "n_open": n0}).reset_index())
res = pd.concat(out).merge(meta, on="station_id")[["station_id", "station_name", "band", "pct", "t", "n_closed", "n_open"]]
res.to_csv(ROOT / "data/external/mart_closure_effect_by_station.csv", index=False)
day = res[res.band == "하루"].sort_values("pct")
print(day.head(12).to_string(index=False))
print("역 수 |t|>=3 이고 감소:", ((day.t <= -3)).sum(), "/ 중앙값", day.pct.median())
print(sun[(sun.date >= "2026-07-01") & (sun.is_mart_closed_sunday == 1)].date.tolist())
