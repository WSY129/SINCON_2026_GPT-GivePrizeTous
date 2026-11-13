"""학교 학사일정 → 날짜별 '방학 중인 학교 비율' (평일만).
입력: data/external/school_events_2025_2026.csv (fetch_neis.py 결과), data/external/raw/neis/schools.csv
방학일 = 행사명에 '방학'이 들어가고 '방학식'은 아닌 날 (방학식 날은 수업하는 날). 재량휴업일도 따로 셉니다.
받은 기간(방학 무렵)만 값이 있어요: 2025-07-07~09-07, 2025-12-15~2026-03-08, 2026-07-06~10-31
Output: data/external/school_vacation_daily.csv
  date, area(전체/인천/부천/구 이름), level(전체/중학교/고등학교), n_schools, share_vacation, share_closed(방학+재량휴업)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ev = pd.read_csv(ROOT / "data/external/school_events_2025_2026.csv", dtype=str)
sch = pd.read_csv(ROOT / "data/external/raw/neis/schools.csv", dtype=str)
sch["city"] = sch.ORG_RDNMA.str.contains("부천").map({True: "부천", False: "인천"})
sch["gu"] = sch.ORG_RDNMA.str.split().str[1]
sch.loc[sch.city == "부천", "gu"] = "부천시"
sch = sch.rename(columns={"SCHUL_KND_SC_NM": "level"})[["SD_SCHUL_CODE", "level", "city", "gu"]]

name = ev.EVENT_NM.fillna("").str.replace(" ", "")
ev["vac"] = name.str.contains("방학") & ~name.str.contains("방학식")
ev["closed"] = ev.vac | ((ev.SBTR_DD_SC_NM == "휴업일") & ~name.str.contains("토요휴업"))
day = ev.groupby(["SD_SCHUL_CODE", "date"])[["vac", "closed"]].max().reset_index()

windows = [("2025-07-07", "2025-09-07"), ("2025-12-15", "2026-03-08"), ("2026-07-06", "2026-10-30")]
dates = pd.DatetimeIndex([])
for a, b in windows:
    dates = dates.append(pd.bdate_range(a, b))
grid = pd.MultiIndex.from_product([sch.SD_SCHUL_CODE, dates.strftime("%Y-%m-%d")], names=["SD_SCHUL_CODE", "date"]).to_frame(index=False)
g = grid.merge(day, on=["SD_SCHUL_CODE", "date"], how="left").fillna({"vac": False, "closed": False}).merge(sch, on="SD_SCHUL_CODE")
# 일정이 하나도 없는 학교는 뺌 (못 받은 학교)
g = g[g.SD_SCHUL_CODE.isin(ev.SD_SCHUL_CODE.unique())]

out = []
for area_col in ("all", "city", "gu"):
    for lv in ("전체", "중학교", "고등학교"):
        x = g if lv == "전체" else g[g.level == lv]
        keys = ["date"] if area_col == "all" else ["date", area_col]
        r = x.groupby(keys).agg(n_schools=("vac", "size"), share_vacation=("vac", "mean"), share_closed=("closed", "mean")).reset_index()
        r["area"] = "전체" if area_col == "all" else r[area_col]
        r["level"] = lv
        out.append(r[["date", "area", "level", "n_schools", "share_vacation", "share_closed"]])
res = pd.concat(out)
res[["share_vacation", "share_closed"]] = res[["share_vacation", "share_closed"]].round(3)
res.to_csv(ROOT / "data/external/school_vacation_daily.csv", index=False)
tot = res[(res.area == "전체") & (res.level == "전체")].set_index("date").share_vacation
print("schools used:", g.SD_SCHUL_CODE.nunique())
for a, b in windows:
    s = tot[a:b]
    print(a, b, "| 50% 넘는 첫날", s[s >= 0.5].index.min(), "| 마지막날", s[s >= 0.5].index.max(), "| 최대", s.max())
