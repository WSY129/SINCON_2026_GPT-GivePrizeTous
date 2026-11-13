"""6. test 기간(2026-07-01~10-31) 요인 달력: 날짜마다 알려진 요인 + 역별 행사 목록."""
import numpy as np
import pandas as pd
from _load import EXT, OUT, cal, event_days, meta

C = cal()
C = C[C.date.between("2026-07-01", "2026-10-31")].copy()
m = meta().set_index("station_id")
sys_hol = __import__("common").HOLIDAYS
C["ds"] = C.date.dt.strftime("%Y-%m-%d")
C["요일"] = C.dow.map(dict(enumerate("월화수목금토일")))
C["공휴일"] = C.ds.map(sys_hol).fillna("")
v = pd.read_csv(EXT / "school_vacation_daily.csv", parse_dates=["date"])
v = v[(v.area == "전체") & (v.level == "전체")][["date", "share_vacation", "share_closed"]]
C = C.merge(v, on="date", how="left")
C["중고_방학비율"] = C.share_vacation.round(2)
C["중고_휴업비율"] = C.share_closed.round(2)  # 방학 + 재량휴업 (주말은 비어 있음)
C["산업단지_휴가주간"] = np.where(C.date.between("2026-07-27", "2026-08-07"), "예(8/3~7 가장 유력)", "")
C["연휴"] = np.where(C.off_block_len >= 3, "연휴(" + C.off_block_len.astype(str) + "일)", "")
C["징검다리"] = np.where(C.bridge == 1, "징검다리", "")
ap = pd.read_csv(EXT / "airport_icn_peak_days.csv")
peak = {"2026-08-02": "공항 최다(입국 12.2만)", "2026-07-25": "공항 출국 최다", "2026-09-27": "공항 최다·입국 최다(14.0만)", "2026-09-24": "공항 출국 최다"}
C["공항"] = C.ds.map(peak).fillna("")
pol = pd.read_csv(EXT / "transport_policy_events.csv")
pol = pol[pol.start_date.fillna("").ne("")]
pol = pol[(pol.start_date <= "2026-10-31") & (pol.end_date.fillna(pol.start_date) >= "2026-07-01")]
pmap = {}
for r in pol.itertuples():
    end = r.end_date if isinstance(r.end_date, str) and r.end_date else r.start_date
    for d in pd.date_range(max(r.start_date, "2026-07-01"), min(end, "2026-10-31")):
        pmap.setdefault(d.strftime("%Y-%m-%d"), []).append(str(r.event)[:30])
C["정책·날씨·기타"] = C.ds.map(lambda d: "; ".join(pmap.get(d, [])))

ev = event_days()
ev = ev[ev.date.between("2026-07-01", "2026-10-31") & (ev.kind != "컨벤시아대형")]
ev["역"] = ev.station_id.map(m.name)
ev["txt"] = ev.kind + ":" + ev.name.astype(str).str[:24] + np.where(ev.start_hour.notna(), "(" + ev.start_hour.astype("Int64").astype(str) + "시)", "") + "@" + ev["역"]
main = ev.sort_values("station_id").drop_duplicates(["date", "kind", "name"])
C = C.merge(main.groupby("date").txt.apply(lambda s: " | ".join(s)).rename("행사").reset_index(), on="date", how="left").fillna({"행사": ""})
conv = pd.read_csv(EXT / "convensia_events.csv")
conv = conv[(conv.n_halls >= 2)]
cmap = {}
for r in conv.itertuples():
    for d in pd.date_range(r.start_date, r.end_date):
        cmap.setdefault(d.strftime("%Y-%m-%d"), []).append(str(r.event_nm)[:20])
C["컨벤시아_대형(효과 작음)"] = C.ds.map(lambda d: "; ".join(cmap.get(d, [])))
cols = ["ds", "요일", "공휴일", "연휴", "징검다리", "중고_방학비율", "중고_휴업비율", "산업단지_휴가주간", "행사", "공항", "정책·날씨·기타", "컨벤시아_대형(효과 작음)"]
C[cols].rename(columns={"ds": "date"}).to_csv(OUT / "a6_test_period_driver_calendar.csv", index=False)
ev[["date", "station_id", "역", "kind", "name", "start_hour", "end_time", "end_hour", "end_basis", "size"]].sort_values(["date", "station_id"]).to_csv(OUT / "a6_test_period_station_events.csv", index=False)
print("test 기간 행사 있는 날:", C[C.행사 != ""].shape[0], "/", len(C))
print(ev.groupby("kind").date.nunique().to_dict())
print(C[cols].head(3).to_string())
