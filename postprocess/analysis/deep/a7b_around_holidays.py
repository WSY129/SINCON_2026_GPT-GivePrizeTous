"""7b. 공휴일 주변 날: 토요일 공휴일 / 연휴 전날(평일) / 사흘 연휴 가운데 일요일·토요일.
train 시간대별 + 서울 일별(하루). 기준: 같은 역의 앞뒤 4주 같은 요일(공휴일 아닌 날) 중앙값.
"""
import numpy as np
import pandas as pd
from _load import EXT, OUT, cal, train

c15 = pd.read_csv(EXT / "calendar_2015_2026.csv", parse_dates=["date"])
c15 = c15.sort_values("date").reset_index(drop=True)
off = (c15.dow >= 5) | (c15.is_holiday == 1)
c15["off"] = off
c15["prev_off"] = off.shift(1, fill_value=False)
c15["next_off"] = off.shift(-1, fill_value=False)
blk = (off != off.shift()).cumsum()
c15["blen"] = np.where(off, off.groupby(blk).transform("sum"), 0)
lunar = c15.holiday_name.fillna("").str.contains("설|추석")
endyear = c15.date.dt.strftime("%m-%d").between("12-24", "12-31") | c15.date.dt.strftime("%m-%d").between("01-01", "01-02")
holset = set(c15[c15.is_holiday == 1].date)
blk_lunar = c15.groupby(blk).apply(lambda g: (g.holiday_name.fillna("").str.contains("설|추석")).any(), include_groups=False)
c15["lunar_block"] = blk.map(blk_lunar)
c15["lunar_near"] = c15.lunar_block | c15.lunar_block.shift(-1, fill_value=False) | c15.lunar_block.shift(1, fill_value=False)

kinds = {
    "토요일 공휴일": (c15.dow == 5) & (c15.is_holiday == 1) & ~lunar,
    "연휴 전날(평일, 다음날부터 3일+)": (~c15.off) & c15.next_off & (c15.blen.shift(-1) >= 3) & ~c15.lunar_near & ~endyear,
    "사흘 연휴 가운데 일요일(월 공휴일)": (c15.dow == 6) & (c15.blen == 3) & c15.next_off & ~c15.lunar_near,
    "사흘 연휴 가운데 토요일(금 공휴일)": (c15.dow == 5) & (c15.blen == 3) & c15.prev_off & ~c15.lunar_near & (c15.is_holiday == 0),
}
DAYS = {k: c15[v].date for k, v in kinds.items()}


def ratio(s, d):
    ref = [s.get(d + pd.Timedelta(weeks=w)) for w in (-4, -3, -2, -1, 1, 2, 3, 4)
           if (d + pd.Timedelta(weeks=w)) not in holset and (d + pd.Timedelta(weeks=w)) in s.index]
    ref = [r for r in ref if r is not None]
    return s[d] / np.median(ref) if d in s.index and len(ref) >= 4 else np.nan


rows = []
sd = pd.read_csv(EXT / "seoul_daily_2015_2026h1.csv", parse_dates=["date"])
for (role, nm), g in sd.groupby(["role", "station_name"]):
    s = g.set_index("date").boardings.astype(float)
    for k, ds in DAYS.items():
        for d in ds:
            if d.year in (2020, 2021):
                continue
            rows.append({"src": "서울일별", "kind": k, "date": d, "r": ratio(s, d)})
t = train()
td = t.groupby(["station_id", "date"]).boardings.sum()
for sid, s in td.groupby(level=0):
    s = s.droplevel(0)
    for k, ds in DAYS.items():
        for d in ds[(ds >= "2025-01-01") & (ds <= "2026-06-30")]:
            rows.append({"src": "train", "kind": k, "date": d, "r": ratio(s, d)})
x = pd.DataFrame(rows).dropna()
summ = x.groupby(["kind", "src"]).agg(배율_중앙=("r", "median"), 날수=("date", "nunique")).round(3).unstack("src")
print(summ.to_string())
x.groupby(["kind", "src", "date"]).r.median().round(3).to_csv(OUT / "a7b_around_holidays_by_day.csv")

# train 시간대 (네트워크): 그날 ÷ 앞뒤 4주 같은 요일
tt = t.groupby(["date", "hour"]).boardings.sum().unstack()
hrows = {}
for k, ds in DAYS.items():
    rs = []
    for d in ds[(ds >= "2025-01-01") & (ds <= "2026-06-30")]:
        ref = [d + pd.Timedelta(weeks=w) for w in (-4, -3, -2, -1, 1, 2, 3, 4)]
        ref = [r for r in ref if r in tt.index and r not in holset]
        if d in tt.index and len(ref) >= 4:
            rs.append(tt.loc[d] / tt.loc[ref].median())
    if rs:
        hrows[k] = pd.concat(rs, axis=1).median(axis=1)
hp = pd.DataFrame(hrows).round(2)
hp.to_csv(OUT / "a7b_around_holidays_hour.csv")
print("\n시간대 (train 네트워크, 같은 요일 = 1)\n", hp.T.to_string())
for k, ds in DAYS.items():
    print(k, "train:", [d.strftime("%m-%d") + f"({d.year % 100})" for d in ds[(ds >= "2025-01-01") & (ds <= "2026-06-30")]],
          "| test:", [d.strftime("%m-%d") for d in ds[(ds >= "2026-07-01") & (ds <= "2026-10-31")]])
