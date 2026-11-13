"""3. 행사·경기·공연: 시간대별로 몇 명이 더 타나 (같은 역·같은 요일유형·같은 시각의 '행사 없는 날' 대비)."""
import numpy as np
import pandas as pd
from _load import OUT, cal, event_days, meta, plt, train

C = cal()
t = train().merge(C[["date", "daytype", "is_holiday", "bridge", "is_semi_holiday"]], on="date")
m = meta()
ev = event_days()
ev = ev[ev.date.between(t.date.min(), t.date.max())]
evkey = ev[["date", "station_id"]].drop_duplicates().assign(has_ev=1)
t = t.merge(evkey, on=["date", "station_id"], how="left").fillna({"has_ev": 0})

clean = t[(t.has_ev == 0) & (t.is_holiday == 0) & (t.bridge == 0) & (t.is_semi_holiday == 0)]
base = clean.groupby(["station_id", "daytype", "hour"]).boardings.median().rename("base")
t = t.join(base, on=["station_id", "daytype", "hour"])
t["excess"] = t.boardings - t.base
t["ratio"] = (t.boardings + 1) / (t.base + 1)

# 행사 1건(날짜×역)마다 종류 하나 (여러 개면 우선순위)
prio = ["SSG경기", "SSG우천취소", "콘서트", "축제", "기타스포츠", "부천FC", "V리그", "예술회관대공연장", "부천아트센터", "컨벤시아대형", "꽃·계절"]
ev["p"] = ev.kind.map({k: i for i, k in enumerate(prio)}).fillna(99)
one = ev.sort_values("p").drop_duplicates(["date", "station_id"])
x = t.merge(one[["date", "station_id", "kind", "name", "start_hour", "size"]], on=["date", "station_id"])

# 3-1 종류별: 하루 추가 승차(중앙값), 시간대별 배율(중앙값)
day = x.groupby(["kind", "date", "station_id"]).agg(excess=("excess", "sum"), base=("base", "sum")).reset_index()
summ = day.groupby("kind").agg(n=("excess", "size"), 하루추가승차_중앙=("excess", "median"),
                               하루추가승차_상위10=("excess", lambda s: s.quantile(.9)),
                               하루배율_중앙=("excess", lambda s: 0)).reset_index()
summ["하루배율_중앙"] = day.assign(r=(day.excess + day.base) / day.base).groupby("kind").r.median().values
hourly = x.groupby(["kind", "hour"]).ratio.median().unstack().round(2)
summ = summ.merge(hourly, left_on="kind", right_index=True).sort_values("하루추가승차_중앙", ascending=False)
summ.round(2).to_csv(OUT / "a3_event_kind_summary.csv", index=False)
print(summ[["kind", "n", "하루추가승차_중앙", "하루추가승차_상위10", "하루배율_중앙", 17, 18, 21, 22]].round(2).to_string())

# 3-2 시작시각 기준 정렬: 시작 h 기준 상대시각별 추가 승차 (승차는 '끝난 뒤' 몰림)
y = x.dropna(subset=["start_hour"]).copy()
y["rel"] = y.hour - y.start_hour
rel = y.groupby(["kind", "rel"]).excess.median().unstack("kind")
rel = rel.loc[-6:8]
rel.round(0).to_csv(OUT / "a3_excess_by_hour_from_start.csv")
fig, ax = plt.subplots(figsize=(11, 5))
for k in ["SSG경기", "콘서트", "부천FC", "V리그", "예술회관대공연장", "부천아트센터", "SSG우천취소", "기타스포츠"]:
    if k in rel:
        ax.plot(rel.index, rel[k], marker="o", ms=3, label=k)
ax.axvline(0, c="k", lw=.8); ax.set_xlabel("행사 시작 시각 기준 (시간)"); ax.set_ylabel("추가 승차 (중앙값, 명/시간)")
ax.set_title("행사 날 시간대별 추가 승차 — 시작 시각 기준"); ax.grid(alpha=.3); ax.legend(ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "a3_event_profile.png", dpi=130); plt.close()

# 3-3 SSG: 관중 수 → 경기 뒤 승차 (끝난 시각 이후 2시간)
g = pd.read_csv(OUT.parents[1] / "data/external/games_ssg_munhak.csv", parse_dates=["date"])
g = g[g.status == "played"].dropna(subset=["attendance", "end_time"])
g["end_h"] = g.end_time.str[:2].astype(int)
mm = t[t.station_id.isin([3127])].merge(g[["date", "end_h", "attendance"]], on="date")
post = mm[(mm.hour >= mm.end_h) & (mm.hour <= mm.end_h + 1)].groupby(["date", "attendance"]).excess.sum().reset_index()
b, a = np.polyfit(post.attendance, post.excess, 1)
r = np.corrcoef(post.attendance, post.excess)[0, 1]
print(f"\nSSG 경기 뒤 2시간 추가 승차(문학경기장역) ≈ {b:.3f} × 관중 + {a:.0f}  (r={r:.2f}, n={len(post)})")
pre = mm[(mm.hour >= mm.end_h - 5) & (mm.hour < mm.end_h - 2)].groupby("date").excess.sum()
print(f"경기 전 하차는 승차 데이터에 없지만, 경기 전 3시간 추가 승차 중앙값 {pre.median():.0f}명 (거의 없음)")
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(post.attendance, post.excess, s=10, alpha=.6)
xs = np.linspace(post.attendance.min(), post.attendance.max(), 10); ax.plot(xs, b * xs + a, c="r")
ax.set_xlabel("관중 수"); ax.set_ylabel("경기 종료 후 2시간 추가 승차 (문학경기장역)"); ax.set_title(f"SSG 관중 → 경기 후 승차 (r={r:.2f})")
fig.tight_layout(); fig.savefig(OUT / "a3_ssg_attendance.png", dpi=130); plt.close()

# 3-4 콘서트 이웃역 파급 (아시아드 콘서트 날 근처 역들 하루 배율)
conc = one[one.kind == "콘서트"][["date", "station_id"]]
asiad_days = conc[conc.station_id == 3209].date.unique()
nb = t[t.date.isin(asiad_days)].groupby(["station_id"]).apply(lambda g: (g.boardings.sum() + 1) / (g.base.sum() + 1), include_groups=False)
nb = nb.rename("배율").to_frame().join(m.set_index("station_id")["name"]).sort_values("배율", ascending=False)
print("\n아시아드 콘서트 날 역별 하루 배율 상위:", nb.head(8).set_index("name")["배율"].round(2).to_dict())
nb.round(3).to_csv(OUT / "a3_asiad_concert_spillover.csv")
