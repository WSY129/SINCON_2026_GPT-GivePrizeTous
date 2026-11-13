"""10-year (2015-2026) holiday/calendar table with COVID flag. Source: python `holidays` package (KR) + manual additions."""
import pandas as pd, holidays
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
# 임시공휴일·선거일 중 패키지에 빠질 수 있는 것 보완 (출처: 인사혁신처 발표)
EXTRA = {"2025-01-27": "임시공휴일", "2025-06-03": "대통령선거", "2026-06-03": "지방선거",
         "2026-05-01": "노동절", "2026-07-17": "제헌절"}
d = pd.DataFrame({"date": pd.date_range("2015-01-01", "2026-12-31")})
kr = {}
for y in range(2015, 2027):
    kr.update({pd.Timestamp(k): v for k, v in holidays.KR(years=y).items()})
for k, v in EXTRA.items():
    kr.setdefault(pd.Timestamp(k), v)
d["holiday_name"] = d.date.map(kr)
d["is_holiday"] = d.holiday_name.notna().astype(int)
d["dow"] = d.date.dt.day_of_week
name = d.holiday_name.fillna("")
d["is_chuseok"] = name.str.contains("추석|Chuseok").astype(int)
d["is_seollal"] = name.str.contains("설날|Seollal|Korean New Year").astype(int)
# 명절 기준일(당일)과의 거리: -3..+3
for col, key in [("chuseok", "추석"), ("seollal", "설날")]:
    days = d[name.isin([key]) | name.str.fullmatch(key)].date
    days = d[name.str.contains(key) & ~name.str.contains("전날|다음날|대체")].date
    d[f"days_from_{col}"] = d.date.apply(lambda x: min(((x - t).days for t in days), key=abs))
    d.loc[d[f"days_from_{col}"].abs() > 7, f"days_from_{col}"] = None
off = (d.dow >= 5) | (d.is_holiday == 1)
blk = (off != off.shift()).cumsum()
d["off_block_len"] = off.groupby(blk).transform("sum").where(off, 0)
d["is_last_day_of_long_break"] = (off & ~off.shift(-1, fill_value=False) & (d.off_block_len >= 3)).astype(int)
# 코로나 구간: 2020-02-23 위기경보 '심각' ~ 2022-04-18 사회적 거리두기 전면 해제
d["covid_period"] = d.date.between("2020-02-23", "2022-04-17").astype(int)
d.to_csv(ROOT / "data/external/calendar_2015_2026.csv", index=False)
c = d[d.days_from_chuseok == 0][["date", "dow", "off_block_len", "covid_period"]]
print("추석 당일 목록"); print(c.to_string(index=False))
