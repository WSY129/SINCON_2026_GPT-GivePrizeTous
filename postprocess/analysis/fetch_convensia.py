"""송도컨벤시아 행사 일정 (전시·회의·이벤트) — 송도컨벤시아 OpenAPI (키 필요 없음).
https://songdoconvensia.visitincheon.or.kr/page/openApiEvent.do
※ 주최 측 요청으로 공개하지 않는 행사는 빠져 있어요.
Output:
  data/external/convensia_events.csv       행사 1건 = 1행
  data/external/convensia_daily.csv        날짜별 진행 중인 전시·회의 수, 대형 전시(홀 2개 이상) 여부
"""
import json, subprocess
from datetime import date
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/convensia"
RAW.mkdir(parents=True, exist_ok=True)
URL = "https://songdoconvensia.visitincheon.or.kr/openApi/event.do?resultType=json&stdate={}&eddate={}"

rows = []
for y in range(2015, 2027):
    for m in range(1, 13):
        if (y, m) > (2026, 12):
            break
        a = date(y, m, 1)
        b = (pd.Timestamp(a) + pd.offsets.MonthEnd(0)).date()
        f = RAW / f"{y}{m:02d}.json"
        if not f.exists():
            out = subprocess.run(["curl", "-s", "-m", "60", "-A", "Mozilla/5.0",
                                  URL.format(a.strftime("%Y%m%d"), b.strftime("%Y%m%d"))],
                                 capture_output=True, text=True).stdout
            f.write_text(out)
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        rows += d.get("itemList") or []

ev = pd.DataFrame(rows).drop_duplicates(["event_nm", "st_event_dt", "end_event_dt", "location"])
ev["start_date"] = pd.to_datetime(ev.st_event_dt, format="%Y%m%d", errors="coerce")
ev["end_date"] = pd.to_datetime(ev.end_event_dt, format="%Y%m%d", errors="coerce")
ev = ev.dropna(subset=["start_date", "end_date"])
ev["n_days"] = (ev.end_date - ev.start_date).dt.days + 1
ev["n_halls"] = ev.location.fillna("").str.count(r"Hall\s*\d")
ev["is_public_show"] = (ev.category == "전시") & (ev.n_halls >= 1)
ev = ev.sort_values("start_date")
out = ev[["start_date", "end_date", "n_days", "category", "event_nm", "location", "n_halls", "pay", "host", "homepage"]].copy()
for c in ("start_date", "end_date"):
    out[c] = out[c].dt.strftime("%Y-%m-%d")
out.to_csv(ROOT / "data/external/convensia_events.csv", index=False)

# 날짜별 집계 (장기 상설 이벤트 30일 초과는 따로 표시만 하고 집계에서 뺌)
days = []
for e in ev[ev.n_days <= 30].itertuples():
    for d in pd.date_range(e.start_date, e.end_date):
        days.append({"date": d, "category": e.category, "n_halls": e.n_halls})
dd = pd.DataFrame(days)
daily = dd.groupby("date").agg(n_events=("category", "size"),
                               n_exhibitions=("category", lambda s: (s == "전시").sum()),
                               n_meetings=("category", lambda s: (s == "회의").sum()),
                               halls_in_use=("n_halls", "sum")).reset_index()
daily["big_show"] = (daily.halls_in_use >= 2).astype(int)
daily["date"] = daily.date.dt.strftime("%Y-%m-%d")
daily.to_csv(ROOT / "data/external/convensia_daily.csv", index=False)
print(len(out), out.start_date.min(), out.start_date.max())
print(out.category.value_counts().to_dict())
print(daily[(daily.date >= "2026-07-01") & (daily.date <= "2026-10-31")].big_show.sum(), "big-show days in test period")
