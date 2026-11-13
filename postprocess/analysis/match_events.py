"""Match spike days to known events and measure effect sizes."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "analysis/daily_ratio.csv", parse_dates=["date"])
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
ev = pd.read_csv(ROOT / "data/external/events_venues.csv", parse_dates=["start_date", "end_date"])
ssg = pd.read_csv(ROOT / "data/external/games_ssg_munhak.csv", parse_dates=["date"])
bfc = pd.read_csv(ROOT / "data/external/games_bucheonfc.csv", parse_dates=["date"])
last = d.date.max()

# --- label table (station, date, label)
lab = []
for _, e in ev.dropna(subset=["station_ids"]).iterrows():
    for s in str(e.station_ids).split(";"):
        for day in pd.date_range(e.start_date, e.end_date):
            lab.append((int(s), day, e.type + ":" + e.event))
for _, g in ssg[ssg.status == "played"].iterrows():
    lab.append((3127, g.date, f"SSG:{g.away}({int(g.attendance)})"))
for _, g in bfc[bfc.attendance > 0].iterrows():
    lab.append((3754, g.date, f"부천FC:{g.away}({int(g.attendance)})"))
L = pd.DataFrame(lab, columns=["station_id", "date", "label"]).groupby(["station_id", "date"]).label.agg(" / ".join).reset_index()
L.to_csv(ROOT / "data/external/event_labels_by_station_day.csv", index=False)

sp = pd.read_csv(ROOT / "analysis/spike_days.csv", parse_dates=["date"]).merge(L, on=["station_id", "date"], how="left")
sp["explained"] = sp.label.notna()
print("== 급증일 설명률 (역별)")
print(sp.groupby(["station_id", "station_name"]).explained.agg(["sum", "count"]).sort_values("count", ascending=False).head(10).to_string())
sp.to_csv(ROOT / "analysis/spike_days_labeled.csv", index=False)

# --- SSG effect: 문학 boardings by hour, game day vs non-game same weekday type
m = tr[tr.station_id == 3127].copy()
gd = ssg[ssg.status == "played"].set_index("date")
m["game"] = m.date.isin(gd.index)
m = m[m.date.dt.month.between(3, 10)]
m["wk"] = np.where(m.date.dt.day_of_week >= 5, "weekend", "weekday")
prof = m.groupby(["wk", "game", "hour"]).boardings.mean().unstack(["wk", "game"]).round(0)
print("\n== 문학경기장역 시간대별 평균 승차 (경기일 vs 비경기일)"); print(prof.to_string())
# attendance relation: post-game boardings (end hour .. +1) vs attendance
rows = []
for dt, g in gd.iterrows():
    if dt > last or pd.isna(g.end_time): continue
    eh = int(str(g.end_time)[:2])
    day = m[m.date == dt].set_index("hour").boardings
    rows.append(dict(date=dt, att=g.attendance, end_hour=eh, post=day.reindex([eh, eh + 1]).sum(),
                     peak_hour=day.idxmax() if len(day) else None))
R = pd.DataFrame(rows)
R.to_csv(ROOT / "analysis/ssg_attendance_vs_boardings.csv", index=False)
print("\n경기 종료 후 2시간 승차 vs 관중수 상관:", round(R[["att", "post"]].corr().iloc[0, 1], 3),
      " 비율(승차/관중) 중앙값:", round((R.post / R.att).median(), 3))
print("피크 시각 = 종료 시각인 경기 비율:", round((R.peak_hour == R.end_hour).mean(), 2),
      " 종료+1:", round((R.peak_hour == R.end_hour + 1).mean(), 2))

# --- Bucheon FC
b = tr[tr.station_id == 3754].copy()
rows = []
for _, g in bfc[(bfc.attendance > 0) & (bfc.date <= last)].iterrows():
    kh = int(g.kickoff[:2]); day = b[b.date == g.date].set_index("hour").boardings
    rows.append(dict(date=g.date, att=g.attendance, kickoff=g.kickoff, post=day.reindex([kh + 2, kh + 3]).sum()))
B = pd.DataFrame(rows); print("\n부천FC 경기 후 승차 vs 관중 상관:", round(B[["att", "post"]].corr().iloc[0, 1], 3),
      " 비율 중앙값:", round((B.post / B.att).median(), 3))

# --- holidays at 부평삼거리 (인천가족공원)
x = d[d.station_id == 3122].set_index("date")
print("\n부평삼거리 명절 tot_ratio:", x.loc[["2025-01-28","2025-01-29","2025-01-30","2025-10-05","2025-10-06","2025-10-07","2026-02-16","2026-02-17","2026-02-18"], "tot_ratio"].round(2).to_dict())

# --- unexplained big spikes
u = sp[~sp.explained & ((sp.tot_ratio >= 1.5) | (sp.night_b_ratio >= 3))]
print("\n== 아직 원인 미상인 큰 급증일", len(u)); print(u[["date","station_name","dow","tot_ratio","night_b_ratio"]].round(1).to_string(index=False))
