"""2. 달력·방학·휴가: 시간대별로 얼마나 바뀌나."""
import numpy as np
import pandas as pd
from _load import OUT, cal, meta, plt, school_share, train

C = cal()
t = train().merge(C, on="date")
m = meta()
types = pd.read_csv(OUT / "a1_station_types.csv")[["station_id", "type"]]
t = t.merge(m[["station_id", "name"]], on="station_id").merge(types, on="station_id")
t["lb"] = np.log1p(t.boardings)

# 기준: 같은 역·같은 시각의 '보통 날'(공휴일·징검다리 아님) 평균 — 평일/토/일 구분
normal = t[(t.is_holiday == 0) & (t.is_semi_holiday == 0) & (t.bridge == 0)]
base = normal.groupby(["station_id", "daytype", "hour"]).boardings.mean().rename("base")
wkbase = base.xs(0, level="daytype").rename("wk")
sunbase = base.xs(2, level="daytype").rename("sun")

# 2-1 날 종류별: 평일 기준 대비 & 일요일 기준 대비 (네트워크 시간대 합)
C2 = C.copy()
ds = C2.date.dt.strftime("%Y-%m-%d")
C2["kind"] = "평일"
C2.loc[C2.dow == 5, "kind"] = "토요일"
C2.loc[C2.dow == 6, "kind"] = "일요일"
C2.loc[C2.is_holiday == 1, "kind"] = "공휴일(일반)"
C2.loc[C2.is_lunar_holiday == 1, "kind"] = "설·추석 당일·연휴"
C2.loc[C2.bridge == 1, "kind"] = "징검다리 평일"
C2.loc[C2.is_semi_holiday == 1, "kind"] = "근로자의날(2025)"
lastday = (C2.is_holiday == 1) & (C2.next_off == 0) & (C2.off_block_len >= 3)
C2.loc[lastday, "kind"] = "연휴 마지막날"
t2 = t.merge(C2[["date", "kind"]], on="date")
net = t2.groupby(["kind", "date", "hour"]).boardings.sum().groupby(["kind", "hour"]).mean().unstack()
ref_wk = net.loc["평일"]
rel = (net / ref_wk).round(2)
rel["하루합(평일=1)"] = (net.sum(axis=1) / ref_wk.sum()).round(3)
rel.to_csv(OUT / "a2_daykind_hour_vs_weekday.csv")
print(rel[["하루합(평일=1)", 7, 8, 12, 15, 18, 21]].sort_values("하루합(평일=1)").to_string())

# 2-2 공휴일 반응: 역 유형별 '공휴일 ÷ 그 역 일요일 기준' (하루 합)
hol = t2[t2.kind.isin(["공휴일(일반)", "설·추석 당일·연휴", "연휴 마지막날"])]
hd = hol.groupby(["station_id", "kind", "date"]).boardings.sum().groupby(["station_id", "kind"]).mean()
sd = normal[normal.daytype == 2].groupby(["station_id", "date"]).boardings.sum().groupby("station_id").mean()
hr = (hd / sd.reindex(hd.index.get_level_values(0)).values).rename("vs_sunday").reset_index()
hr = hr.merge(m[["station_id", "name"]], on="station_id").merge(types, on="station_id")
piv = hr.pivot_table(index=["name", "type"], columns="kind", values="vs_sunday").round(2)
piv.to_csv(OUT / "a2_holiday_vs_sunday_by_station.csv")
print("\n설·추석 일요일 대비 상위/하위:")
s = piv["설·추석 당일·연휴"].sort_values()
print(s.head(6).to_dict()); print(s.tail(6).to_dict())

# 2-3 학교 방학: 2025 여름 평일, 방학 비율 → 역·시간대별 탄력 (log 승차 ~ share)
ss = school_share()
w = t[(t.date >= "2025-07-07") & (t.date <= "2025-09-05") & (t.daytype == 0) & (t.is_holiday == 0)].merge(ss, on="date")
def slope(g):
    x = g.share_vacation.values; y = g.lb.values
    if x.std() < 1e-6:
        return np.nan
    return np.polyfit(x, y, 1)[0]
el = w.groupby(["station_id", "hour"]).apply(slope, include_groups=False).rename("dlog").reset_index()
el["pct_change_full_vacation"] = (np.expm1(el.dlog) * 100).round(1)
elp = el.pivot(index="station_id", columns="hour", values="pct_change_full_vacation")
elp = elp.join(m.set_index("station_id")["name"]).set_index("name")
elp.to_csv(OUT / "a2_school_vacation_effect_station_hour.csv")
net_el = w.groupby(["date", "hour"]).boardings.sum().reset_index().merge(ss, on="date")
net_el["lb"] = np.log1p(net_el.boardings)
nh = net_el.groupby("hour").apply(slope, include_groups=False)
print("\n학교 방학(0→100%) 전체 시간대 효과 %:", (np.expm1(nh) * 100).round(1).to_dict())
day_eff = w.groupby(["station_id", "date"]).boardings.sum().reset_index().merge(ss, on="date")
day_eff["lb"] = np.log1p(day_eff.boardings)
de = day_eff.groupby("station_id").apply(slope, include_groups=False)
de = (np.expm1(de) * 100).round(1).rename("하루합 효과%").to_frame().join(m.set_index("station_id")["name"]).sort_values("하루합 효과%")
de.to_csv(OUT / "a2_school_vacation_effect_station_day.csv")
print("방학 하루합 효과 큰 역:", de.head(8).set_index("name")["하루합 효과%"].to_dict())

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(nh.index, np.expm1(nh) * 100, color=["#c44" if v < 0 else "#48c" for v in nh])
ax.set_xlabel("시"); ax.set_ylabel("방학 비율 0→100% 일 때 변화(%)"); ax.grid(alpha=.3, axis="y")
ax.set_title("중·고 여름방학 효과 (2025 여름 평일, 네트워크 전체)")
fig.tight_layout(); fig.savefig(OUT / "a2_school_vacation_hour.png", dpi=130); plt.close()

# 2-4 여름 집단휴가 주간 (학교 방학 효과를 뺀 나머지): 2025-07-28~08-08 vs 방학 비율 같은 다른 날
w["peak_week"] = w.date.between("2025-07-28", "2025-08-08")
hi = w[w.share_vacation > 0.75]
pw = hi.groupby(["station_id", "peak_week", "hour"]).boardings.mean().unstack("peak_week")
pw = (pw[True] / pw[False]).rename("ratio").reset_index()
am = pw[pw.hour.isin([7, 8])].groupby("station_id").ratio.mean()
am = am.rename("휴가주간 아침 비율").to_frame().join(m.set_index("station_id")["name"]).join(types.set_index("station_id")).sort_values("휴가주간 아침 비율")
am.round(3).to_csv(OUT / "a2_summer_holiday_week_am.csv")
print("\n방학 중(>75%)인 날끼리 비교, 7/28~8/8 휴가 주간 아침 승차 비율 하위:", am.head(8).set_index("name")["휴가주간 아침 비율"].round(2).to_dict())
print("유형별 평균:", am.groupby("type")["휴가주간 아침 비율"].mean().round(3).to_dict())
