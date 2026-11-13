"""2025-01 ~ 2026-06 train 으로 '행사·경기·축제·공휴일 때문에 특정 역이 평소보다 얼마나 변했나' 를 계산.

평소(기준값)
- 행사·경기: 같은 역, 같은 요일, 앞뒤 4주(±28일) 중 공휴일·행사일을 뺀 날의 중앙값
- 공휴일: 같은 역, 같은 달의 (a) 평범한 평일 중앙값, (b) 일요일 중앙값

출력
- analysis/event_effects_by_day.csv      : 행사(경기) 하루하루의 효과
- analysis/event_effects_summary.csv     : 행사 종류 × 역 요약
- analysis/holiday_effects_by_station.csv: 공휴일 종류 × 역 요약
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
meta = pd.read_csv(ROOT / "data/metaData.csv")
cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv", parse_dates=["date"])
lab = pd.read_csv(ROOT / "data/external/event_labels_by_station_day.csv", parse_dates=["date"])
kovo = pd.read_csv(ROOT / "data/external/games_kovo_incheon.csv", parse_dates=["date"])
names = meta.set_index("station_id").station_name

# ── 행사 라벨 (배구 포함) ──────────────────────────────────────────────
k = kovo[kovo.station_id.notna()]
lab = pd.concat([lab, pd.DataFrame({"station_id": k.station_id.astype(int), "date": k.date,
                                    "label": "배구:" + k.home + "-" + k.away + "(" + k.start_time + ")"})])
lab = lab[(lab.date >= "2025-01-01") & (lab.date <= "2026-06-30")]
# 아시아드 콘서트는 옆 역 서구청(3210)에도 이미 라벨이 있음
def group(lbl):
    head = lbl.split(":")[0]
    return {"SSG": "SSG 야구", "부천FC": "부천FC 축구", "배구": "흥국생명 배구"}.get(head, head)
lab["kind"] = lab.label.map(group)
lab["event"] = lab.label.str.split(":", n=1).str[1].str.split(" / ").str[0]

hol = set(cal.loc[cal.is_holiday == 1, "date"])
ev_days = set(zip(lab.station_id, lab.date))

# ── 하루 합계, 시간대 값 ──────────────────────────────────────────────
day = tr.groupby(["station_id", "date"]).boardings.sum().rename("b").reset_index()
day["dow"] = day.date.dt.day_of_week
hourly = tr.pivot_table(index=["station_id", "date"], columns="hour", values="boardings")

def baseline_dates(sid, dt):
    cands = [dt + pd.Timedelta(days=7 * w) for w in (-4, -3, -2, -1, 1, 2, 3, 4)]
    return [c for c in cands if c not in hol and (sid, c) not in ev_days and (sid, c) in hourly.index]

rows = []
for (sid, dt), g in lab.groupby(["station_id", "date"]):
    if (sid, dt) not in hourly.index:
        continue
    base = baseline_dates(sid, dt)
    if len(base) < 3:
        continue
    h_today = hourly.loc[(sid, dt)]
    h_base = hourly.loc[[(sid, b) for b in base]].median()
    tot_r = h_today.sum() / h_base.sum()
    night_r = h_today.loc[20:23].sum() / max(h_base.loc[20:23].sum(), 1)
    diff = h_today - h_base
    peak_h = int(diff.idxmax())
    rows.append(dict(station_id=sid, station_name=names.get(sid), date=dt.date(), dow="월화수목금토일"[dt.day_of_week],
                     is_holiday=int(dt in hol), kind=g.kind.iloc[0], event=g.event.iloc[0],
                     day_boardings=int(h_today.sum()), usual_day=int(h_base.sum()), day_ratio=round(tot_r, 2),
                     extra_boardings=int(h_today.sum() - h_base.sum()), night_ratio=round(night_r, 2),
                     peak_hour=peak_h, peak_hour_boardings=int(h_today[peak_h]), peak_hour_usual=int(h_base[peak_h]),
                     peak_hour_ratio=round(h_today[peak_h] / max(h_base[peak_h], 1), 1)))
E = pd.DataFrame(rows).sort_values(["kind", "station_id", "date"])
E.to_csv(ROOT / "analysis/event_effects_by_day.csv", index=False, encoding="utf-8-sig")

S = (E.groupby(["kind", "station_id", "station_name"])
       .agg(days=("date", "size"), day_ratio_median=("day_ratio", "median"), day_ratio_max=("day_ratio", "max"),
            extra_boardings_median=("extra_boardings", "median"), night_ratio_median=("night_ratio", "median"),
            peak_hour_mode=("peak_hour", lambda s: int(s.mode().iloc[0])), peak_hour_ratio_median=("peak_hour_ratio", "median"))
       .reset_index().sort_values(["kind", "days"], ascending=[True, False]))
S.to_csv(ROOT / "analysis/event_effects_summary.csv", index=False, encoding="utf-8-sig")

# ── 공휴일 × 역 ───────────────────────────────────────────────────────
c = cal.set_index("date")
day = day.join(c[["holiday_name", "is_holiday", "is_chuseok", "is_seollal", "is_last_day_of_long_break"]], on="date")
day["m"] = day.date.dt.to_period("M")
normal_wd = day[(day.is_holiday == 0) & (day.dow < 5)].groupby(["station_id", "m"]).b.median().rename("wd")
sunday = day[(day.is_holiday == 0) & (day.dow == 6)].groupby(["station_id", "m"]).b.median().rename("sun")
day = day.join(normal_wd, on=["station_id", "m"]).join(sunday, on=["station_id", "m"])

def hgroup(r):
    if r.is_seollal == 1 and "설날" == str(r.holiday_name).strip():
        return "설날 당일"
    if r.is_chuseok == 1 and "추석" == str(r.holiday_name).strip():
        return "추석 당일"
    if r.is_seollal == 1:
        return "설 연휴 (당일 제외)"
    if r.is_chuseok == 1 or "추석" in str(r.holiday_name):
        return "추석 연휴 (당일 제외)"
    if r.is_holiday == 1:
        return "기타 공휴일"
    return None

day["hgroup"] = day.apply(hgroup, axis=1)
extra = {pd.Timestamp("2025-05-01"): "근로자의 날 2025 (공휴일 아님)", pd.Timestamp("2025-10-10"): "징검다리 평일 2025-10-10"}
day.loc[day.date.isin(extra), "hgroup"] = day.loc[day.date.isin(extra), "date"].map(extra)
H = day[day.hgroup.notna()].copy()
H["vs_weekday"] = H.b / H.wd
H["vs_sunday"] = H.b / H.sun
HS = (H.groupby(["hgroup", "station_id"]).agg(days=("date", "nunique"), vs_weekday=("vs_weekday", "median"), vs_sunday=("vs_sunday", "median"))
        .reset_index())
HS["station_name"] = HS.station_id.map(names)
HS = HS.round(3)
HS.to_csv(ROOT / "analysis/holiday_effects_by_station.csv", index=False, encoding="utf-8-sig")

# 공휴일 다음 평일, 연휴 마지막 날
print("events", E.shape, "summary", S.shape, "holiday", HS.shape)
