"""7. test 기간 공휴일의 닮은꼴: 같은 요일에 놓인 (명절 아닌) 공휴일이 하루·시간대별로 얼마나 줄었나.
train(2025-01~2026-06, 71역 시간대) + 서울시 일별(7호선 9역 2015~2021, 환승 상대역 2015~2026-06).
배율 기준: 같은 역의 앞뒤 4주 같은 요일(공휴일 제외) 중앙값 / 일요일 기준: 앞뒤 4주 일요일 중앙값.
"""
import numpy as np
import pandas as pd
from _load import EXT, OUT, cal, meta, train

ROOT = OUT.parents[1]
c15 = pd.read_csv(EXT / "calendar_2015_2026.csv", parse_dates=["date"])
hol = c15[c15.is_holiday == 1].copy()
holset = set(hol.date)
lunar = hol.holiday_name.fillna("").str.contains("설|추석")
plain = hol[~lunar & ~hol.holiday_name.fillna("").str.contains("신정|성탄")]  # 연말연시는 따로 움직여서 뺌
plain = plain[plain.dow.isin([0, 4])]  # 월요일·금요일 공휴일 (test 의 7/17 금, 8/17 월, 10/5 월, 10/9 금)


def ratio(series, d, same_dow=True):
    ref_days = [d + pd.Timedelta(weeks=w) for w in (-4, -3, -2, -1, 1, 2, 3, 4)]
    if not same_dow:  # 가장 가까운 일요일들
        sun = d + pd.Timedelta(days=(6 - d.day_of_week))
        ref_days = [sun + pd.Timedelta(weeks=w) for w in (-4, -3, -2, -1, 1, 2, 3, 4)]
    ref = [series[x] for x in ref_days if x in series.index and x not in holset]
    if d not in series.index or len(ref) < 4:
        return np.nan
    return series[d] / np.median(ref)


rows = []
# 서울 일별
sd = pd.read_csv(EXT / "seoul_daily_2015_2026h1.csv", parse_dates=["date"])
for (role, nm), g in sd.groupby(["role", "station_name"]):
    s = g.set_index("date").boardings.astype(float)
    for r in plain.itertuples():
        if r.date.year in (2020, 2021):
            continue
        rows.append({"src": "서울일별_" + ("7호선9역" if role == "target" else "환승상대역"), "date": r.date, "name": r.holiday_name,
                     "dow": "월" if r.dow == 0 else "금", "station": nm,
                     "vs_same_dow": ratio(s, r.date), "vs_sunday": ratio(s, r.date, False)})
# train (역별 하루 합)
t = train()
td = t.groupby(["station_id", "date"]).boardings.sum()
for sid, s in td.groupby(level=0):
    s = s.droplevel(0)
    for r in plain[plain.date.between("2025-01-01", "2026-06-30")].itertuples():
        rows.append({"src": "train", "date": r.date, "name": r.holiday_name, "dow": "월" if r.dow == 0 else "금", "station": sid,
                     "vs_same_dow": ratio(s, r.date), "vs_sunday": ratio(s, r.date, False)})
x = pd.DataFrame(rows).dropna()
byday = x.groupby(["src", "dow", "date", "name"])[["vs_same_dow", "vs_sunday"]].median().round(3).reset_index()
byday.to_csv(OUT / "a7_holiday_analogs_by_day.csv", index=False)
summ = x.groupby(["src", "dow"])[["vs_same_dow", "vs_sunday"]].median().round(3)
summ["n_days"] = x.groupby(["src", "dow"]).date.nunique()
print(summ.to_string())
print(byday[byday.src == "train"].to_string(index=False))

# train 시간대: 금·월 공휴일 ÷ 일요일 (네트워크)
C = cal()
tt = t.merge(C[["date", "dow", "is_holiday", "bridge", "is_semi_holiday", "is_lunar_holiday"]], on="date")
hd = plain[plain.date.between("2025-01-01", "2026-06-30")].date
sun = tt[(tt.dow == 6) & (tt.is_holiday == 0)]
sunp = sun.groupby(["date", "hour"]).boardings.sum().groupby("hour").mean()
out = {}
for dw, nm in ((4, "금요일 공휴일"), (0, "월요일 공휴일")):
    hh = tt[tt.date.isin(hd) & (tt.dow == dw)].groupby(["date", "hour"]).boardings.sum().groupby("hour").mean()
    out[nm] = (hh / sunp).round(2)
hp = pd.DataFrame(out)
hp.to_csv(OUT / "a7_holiday_hour_vs_sunday.csv")
print("\n시간대: 금·월 공휴일 ÷ 보통 일요일\n", hp.T.to_string())
print("train 의 금·월 공휴일:", sorted(set((d.strftime('%Y-%m-%d'), n) for d, n in zip(plain.date, plain.holiday_name) if pd.Timestamp('2025-01-01') <= d <= pd.Timestamp('2026-06-30'))))
