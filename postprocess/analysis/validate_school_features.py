"""학교 변수 표(school_features_station_day.csv)가 train 승차를 실제로 바꾸는지 확인.
각 날을 같은 역·같은 요일 앞뒤 3주 '평범한 날'(학교 행사 없음, 공휴일·앞뒤 날 아님) 중앙값과 비교한 로그 비율의 중앙값.
Output: analysis/school_features_validation.csv
사용: .venv/bin/python -I analysis/validate_school_features.py
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
f = pd.read_csv(ROOT / "data/external/school_features_station_day.csv", parse_dates=["date"])
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv", parse_dates=["date"])
hol = set(cal.date[cal.is_holiday == 1]); adj = hol | {d + pd.Timedelta(days=k) for d in hol for k in (-1, 1)}
meta = pd.read_csv(ROOT / "data/metaData.csv").set_index("station_id").station_name
f["state"] = np.select(
    [f.hs_exam_main == 1, f.vacation_share >= 0.5, f.closed_n > 0, f.ceremony_n > 0, f.trip_n > 0, f.hs_exam_n + f.ms_exam_n > 0],
    ["시험(주 학교)", "방학", "재량휴업", "방학식·개학식 등", "체험학습·수학여행", "다른 학교 시험"], "평범")
rows = []
for sid, g in f.groupby("station_id"):
    x = tr[(tr.station_id == sid) & (tr.date.dt.day_of_week < 5) & ~tr.date.isin(adj)]
    P = x.pivot(index="date", columns="hour", values="boardings")
    st = g.set_index("date").state
    normal = [d for d in P.index if st.get(d) == "평범"]
    for d in P.index:
        nb = [q for q in normal if q.day_of_week == d.day_of_week and 0 < abs((q - d).days) <= 21]
        if len(nb) < 2:
            continue
        U = P.loc[nb].median()
        rows.append(dict(station=meta[sid], state=st.get(d), day=np.log(P.loc[d].sum() / U.sum()),
                         h7=np.log((P.loc[d, 7] + 1) / (U[7] + 1)), h12_13=np.log((P.loc[d, [12, 13]].sum() + 1) / (U[[12, 13]].sum() + 1)),
                         h16_17=np.log((P.loc[d, [16, 17]].sum() + 1) / (U[[16, 17]].sum() + 1))))
r = pd.DataFrame(rows)
t = r.groupby(["station", "state"]).agg(days=("day", "size"), day=("day", "median"), h7=("h7", "median"),
                                         h12_13=("h12_13", "median"), h16_17=("h16_17", "median")).round(2)
t.to_csv(ROOT / "analysis/school_features_validation.csv")
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
print(t.to_string())
