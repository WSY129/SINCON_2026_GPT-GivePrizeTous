"""4. 날씨·미세먼지·인구·공항·정책 사건."""
import numpy as np
import pandas as pd
from _load import EXT, OUT, cal, event_days, meta, plt, train, weather_daily

C = cal()
t = train().merge(C[["date", "daytype", "is_holiday", "bridge", "is_semi_holiday", "month"]], on="date")
m = meta()
ev = event_days()[["date", "station_id"]].drop_duplicates().assign(has_ev=1)
t = t.merge(ev, on=["date", "station_id"], how="left").fillna({"has_ev": 0})
t = t[(t.has_ev == 0) & (t.is_holiday == 0) & (t.bridge == 0) & (t.is_semi_holiday == 0)]
# 기준: 같은 역·요일유형·시각·월(계절 영향 제거)
base = t.groupby(["station_id", "daytype", "hour", "month"]).boardings.median().rename("base")
t = t.join(base, on=["station_id", "daytype", "hour", "month"])
net = t.groupby(["date", "hour", "daytype"])[["boardings", "base"]].sum().reset_index()
net["r"] = net.boardings / net.base

# 4-1 시간당 강수
w = weather_daily()
if w is not None:
    x = net.merge(w[["date", "hour", "precipitation", "apparent_temperature", "snowfall"]], on=["date", "hour"])
    x["rain_bin"] = pd.cut(x.precipitation, [-0.01, 0, 1, 5, 10, 200], labels=["0", "0~1mm", "1~5mm", "5~10mm", "10mm+"])
    rb = x.groupby(["daytype", "rain_bin"], observed=True).r.agg(["median", "size"]).unstack(0)
    rb.round(3).to_csv(OUT / "a4_rain_hourly.csv")
    print("시간당 강수별 네트워크 승차 배율 (평일 0 / 토 1 / 일 2):\n", rb.round(3).to_string())
    # 하루 단위: 일 강수·최고 체감온도
    wd = w.groupby("date").agg(rain=("precipitation", "sum"), tmax=("apparent_temperature", "max"), snow=("snowfall", "sum")).reset_index()
    dn = net.groupby(["date", "daytype"])[["boardings", "base"]].sum().reset_index().merge(wd, on="date")
    dn["r"] = dn.boardings / dn.base
    dn["rain_bin"] = pd.cut(dn.rain, [-0.01, 0.5, 5, 20, 50, 500], labels=["~0.5", "0.5~5", "5~20", "20~50", "50+"])
    dn["t_bin"] = pd.cut(dn.tmax, [-30, 0, 10, 20, 28, 33, 50], labels=["<0", "0~10", "10~20", "20~28", "28~33", "33+"])
    rd = dn.groupby(["daytype", "rain_bin"], observed=True).r.median().unstack(0).round(3)
    td = dn.groupby(["daytype", "t_bin"], observed=True).r.median().unstack(0).round(3)
    print("\n일 강수별 하루 배율:\n", rd.to_string()); print("\n일 최고 체감온도별 하루 배율:\n", td.to_string())
    rd.to_csv(OUT / "a4_rain_daily.csv"); td.to_csv(OUT / "a4_temp_daily.csv")
    print("\n적설 있는 날:", dn[dn.snow > 0.5][["date", "daytype", "snow", "r"]].round(2).to_string(index=False))

# 4-2 미세먼지 (일 평균 PM10)
aq = pd.read_csv(EXT / "airquality_hourly.csv", parse_dates=["time"])
aq = aq[aq.area == "incheon"]; aq["date"] = aq.time.dt.normalize()
pm = aq.groupby("date").pm10.mean().rename("pm10").reset_index()
dn2 = net.groupby(["date", "daytype"])[["boardings", "base"]].sum().reset_index().merge(pm, on="date")
dn2["r"] = dn2.boardings / dn2.base
dn2["pm_bin"] = pd.cut(dn2.pm10, [0, 30, 50, 80, 150, 1000], labels=["좋음~30", "30~50", "50~80", "80~150(나쁨)", "150+"])
pmr = dn2.groupby(["daytype", "pm_bin"], observed=True).r.agg(["median", "size"]).unstack(0).round(3)
print("\nPM10 별 하루 배율:\n", pmr.to_string()); pmr.to_csv(OUT / "a4_pm10_daily.csv")

# 4-3 인구: 역세권 인구 vs 평일 하루 승차 (단면), 검단 3역 월별 추세 탄력
tt = train().merge(C[["date", "daytype", "is_holiday"]], on="date")
wkd = tt[(tt.daytype == 0) & (tt.is_holiday == 0)]
pop = pd.read_csv(EXT / "station_catchment_pop_monthly.csv")
mon = wkd.groupby(["station_id", pd.Grouper(key="date", freq="MS"), "date"]).boardings.sum().groupby(["station_id", "date"]).mean().reset_index()
mon["ym"] = mon.date.dt.strftime("%Y-%m")
mp = mon.merge(pop[["ym", "station_id", "pop_1km", "boundary_change"]], on=["ym", "station_id"])
cs = mp[mp.ym == "2026-06"]
r = np.corrcoef(np.log(cs.pop_1km), np.log(cs.boardings))[0, 1]
print(f"\n단면: log(역세권 1km 인구) vs log(평일 승차) 상관 r={r:.2f} (71역, 2026-06)")
g3 = mp[mp.station_id.isin([3107, 3108, 3109]) & (mp.ym >= "2025-08")]
for sid, g in g3.groupby("station_id"):
    el = np.polyfit(np.log(g.pop_1km), np.log(g.boardings), 1)[0]
    print(f"  {m.set_index('station_id').name[sid]} 2025-08→2026-06: 인구 {g.pop_1km.iloc[0]:.0f}→{g.pop_1km.iloc[-1]:.0f} ({g.pop_1km.iloc[-1]/g.pop_1km.iloc[0]-1:+.1%}), "
          f"평일 승차 {g.boardings.iloc[0]:.0f}→{g.boardings.iloc[-1]:.0f} ({g.boardings.iloc[-1]/g.boardings.iloc[0]-1:+.1%}), 탄력 {el:.2f}")
mp.to_csv(OUT / "a4_station_month_pop_boardings.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(cs.pop_1km / 1000, cs.boardings / 1000, s=12)
for rr in cs.itertuples():
    if rr.boardings > 25000 or rr.pop_1km > 85000:
        ax[0].annotate(m.set_index("station_id").name[rr.station_id], (rr.pop_1km / 1000, rr.boardings / 1000), fontsize=7)
ax[0].set_xlabel("역세권 1km 인구 (천 명)"); ax[0].set_ylabel("평일 하루 승차 (천 명)"); ax[0].set_title(f"단면 (2026-06, r={r:.2f} 로그)")
for sid, g in mp[mp.station_id.isin([3107, 3108, 3109])].groupby("station_id"):
    ax[1].plot(pd.to_datetime(g.ym), g.boardings / g.boardings.iloc[-1], marker="o", label=m.set_index("station_id").name[sid] + " 승차")
    ax[1].plot(pd.to_datetime(g.ym), g.pop_1km / g.pop_1km.iloc[-1], ls="--", label=m.set_index("station_id").name[sid] + " 인구")
ax[1].set_title("검단 3역: 평일 승차·역세권 인구 (2026-06=1)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "a4_population.png", dpi=130); plt.close()

# 4-4 공항: 계양 월 평일 승차 vs 인천공항 월 여객
ap = pd.read_csv(EXT / "airport_icn_monthly_2015_2026.csv")
ky = mon[mon.station_id == 3110].merge(ap[["ym", "passengers"]], on="ym")
ky2 = ky[ky.ym >= "2025-08"]  # 검단 개통 뒤
rr = np.corrcoef(ky2.passengers, ky2.boardings)[0, 1]
print(f"\n계양역 평일 승차 vs 인천공항 월 여객 (2025-08~2026-06) r={rr:.2f}")
# 공항철도 환승 시간대: 계양 21~23시 비중
kh = tt[(tt.station_id == 3110) & (tt.daytype == 0)].groupby("hour").boardings.mean()
print("계양 평일 시간대 승차 (밤 비중):", (kh[[21, 22, 23]].sum() / kh.sum()).round(3))

# 4-5 정책·사건 날
pe = {"2026-01-13": "서울 버스 파업", "2026-01-14": "서울 버스 파업", "2025-12-04": "첫눈 폭설", "2025-08-13": "극한호우"}
allt = train().merge(C[["date", "daytype", "month"]], on="date").join(base, on=["station_id", "daytype", "hour", "month"])
allt = allt.merge(m[["station_id", "canonical_line", "name"]], on="station_id")
for d, nm in pe.items():
    x = allt[allt.date == d]
    byl = x.groupby("canonical_line")[["boardings", "base"]].sum()
    print(f"{d} {nm}: 노선별 하루 배율", (byl.boardings / byl.base).round(3).to_dict())
    if "버스" in nm:
        s7 = x[x.canonical_line == 7].groupby("hour")[["boardings", "base"]].sum()
        print("   7호선 시간대 배율", (s7.boardings / s7.base).round(2).to_dict())
# 검단 개통·요금 인상 전후: 같은 달 비교가 불가해서 6월 vs 7월 평일, 노선별
pp = wkd.merge(m[["station_id", "canonical_line"]], on="station_id")
pp = pp[pp.date.between("2025-06-02", "2025-07-25")]
pp["after"] = pp.date >= "2025-06-30"
a = pp[~pp.station_id.isin([3107, 3108, 3109])].groupby(["canonical_line", "after", "date"]).boardings.sum().groupby(["canonical_line", "after"]).mean().unstack()
print("\n2025-06-28 (검단 개통·요금 인상) 전후 평일 하루 승차 (기존 역만):", (a[True] / a[False] - 1).round(3).to_dict())
