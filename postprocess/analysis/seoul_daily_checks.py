"""서울시 일별 자료 검증 + 추석 전후 날짜별 효과.
1) 7호선 9개 역: 일별 합 vs 기존 월별·시간대별 자료(OA-12252) 월 합
2) 환승 상대역 vs train 같은 날 승차 상관 (2025-01~2026-06)
3) 추석 D-3 ~ D+3 날짜별 배율 = 그날 승차 ÷ 앞뒤 4주 같은 요일 중앙값(공휴일 제외)
Output: analysis/chuseok_daily_effect_seoul.csv (year, station, role, offset, date, dow, ratio)
"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "data/external/seoul_daily_2015_2026h1.csv", parse_dates=["date"])
cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv", parse_dates=["date"])
hol = set(cal[cal.is_holiday == 1].date)

# 1)
m = pd.read_csv(ROOT / "data/external/seoul_line7_monthly_hourly_2015_2021.csv")
mm = m.groupby(["month", "station_id"]).boardings.sum()
t = d[d.role == "target"].copy(); t["month"] = t.date.dt.strftime("%Y-%m")
dm = t.groupby(["month", "station_id"]).boardings.sum()
r = (dm / mm).dropna()
print("1) 일별 합 ÷ 월별 자료: 중앙값 %.4f, 범위 %.3f~%.3f, n=%d" % (r.median(), r.min(), r.max(), len(r)))

# 2)
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
td = tr.groupby(["date", "station_id"]).boardings.sum().unstack()
pairs = {"계양": 3110, "검암": 3207, "부평": 3120, "주안": 3218, "원인재": 3130}
meta = pd.read_csv(ROOT / "data/metaData.csv").set_index("station_id").station_name
for nm, sid in pairs.items():
    if sid not in td or meta[sid].split("(")[0] != nm:
        sid = meta[meta.str.startswith(nm)].index[0]
    p = d[(d.role == "partner") & (d.station_name == nm)].set_index("date").boardings
    x = pd.concat([p, td[sid]], axis=1, keys=["p", "t"]).dropna()
    print("2) %s: 상대역 하루 %d명 vs train %s(%d) %d명, 일별 상관 %.2f, log 변화 상관(같은 요일 전주 대비) %.2f" % (
        nm, x.p.mean(), meta[sid], sid, x.t.mean(), x.p.corr(x.t), np.log(x.p).diff(7).corr(np.log(x.t).diff(7))))

# 3)
ch = cal[cal.holiday_name == "추석"].date
rows = []
for (role, nm), g in d.groupby(["role", "station_name"]):
    s = g.set_index("date").boardings.astype(float)
    for c in ch:
        if c < s.index.min() or c > s.index.max():
            continue
        for off in range(-3, 5):
            x = c + pd.Timedelta(days=off)
            ref = [x + pd.Timedelta(weeks=w) for w in (-5, -4, -3, -2, 2, 3, 4, 5)]
            ref = [s[y] for y in ref if y in s.index and y not in hol]
            if x in s.index and len(ref) >= 4:
                rows.append((c.year, nm, role, off, x.strftime("%Y-%m-%d"), x.day_of_week, round(s[x] / np.median(ref), 3)))
e = pd.DataFrame(rows, columns=["year", "station", "role", "offset", "date", "dow", "ratio"])
e.to_csv(ROOT / "analysis/chuseok_daily_effect_seoul.csv", index=False)
ok = e[~e.year.isin([2020, 2021])]
print("3) 추석 전후 배율 (코로나 2020·2021 제외, 역·연도 중앙값)")
print(ok.groupby(["role", "offset"]).ratio.median().unstack().round(2).to_string())
print(ok[ok.role == "target"].pivot_table(index="year", columns="offset", values="ratio", aggfunc="median").round(2).to_string())
print(ok[ok.role == "partner"].pivot_table(index="year", columns="offset", values="ratio", aggfunc="median").round(2).to_string())
print(ok[ok.offset.isin([-1, 0, 1])].groupby(["role", "station"]).ratio.median().round(2).to_string())
