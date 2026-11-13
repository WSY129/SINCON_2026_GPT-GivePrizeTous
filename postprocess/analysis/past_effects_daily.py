"""과거 일별 자료(48번·47번)로 행사·경기·공휴일 효과를 2019 ~ 2026-06까지 다시 계산한다.

18~20번(station_effects_2025_2026.py)과 같은 방법을 더 긴 기간에 쓴다.
  평소(행사): 같은 역·같은 요일·앞뒤 4주 중 공휴일·그 역 행사일·코로나 구간을 뺀 날의 중앙값 (3일 이상)
  평소(공휴일): 같은 역·같은 달의 평범한 평일 중앙값, 일요일 중앙값
쓰는 값: 48번 boardings (6~23시, F 기준 = 2026 train 기준). 비율만 보므로 기준 차이는 영향 없음.
시간대 값: 2019 ~ 2020-02는 47번, 2025-01 ~ 2026-06은 train. 2022 ~ 2024는 하루 합만 있음.
코로나 구간(2020-02-23 ~ 2022-04-17)은 뺀다.

출력 (analysis/)
  event_effects_by_day_2019_2026.csv      행사·경기 하루하루 효과 (연도 열 포함)
  event_effects_summary_2019_2026.csv     행사 종류 × 역 × 기간(2019 / 2022-24 / 2025-26) 요약
  holiday_effects_by_station_2019_2026.csv 공휴일 종류 × 역 × 연도
  chuseok_2019_layout_hourly.csv          2019 추석(2026과 요일 배치가 같음) 역 × 날짜 × 시간 비율
  chuseok_saturday_analogs.csv            "추석 다음 날 = 토요일" 등 9/26과 비슷한 날의 역별 비율
사용: .venv/bin/python -I analysis/past_effects_daily.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external"
OUT = ROOT / "analysis"

meta = pd.read_csv(ROOT / "data/metaData.csv")
names = meta.set_index("station_id").station_name
cal = pd.read_csv(EXT / "calendar_2015_2026.csv", parse_dates=["date"])
cal = cal.set_index("date")
hol = set(cal.index[cal.is_holiday == 1])
covid = set(cal.index[cal.covid_period == 1])

# ── 자료 ────────────────────────────────────────────────────────────
D = pd.read_csv(EXT / "ictr_daily_station_2019_2026h1.csv", parse_dates=["date"])
D = D[~D.date.isin(covid)]
day = D.set_index(["station_id", "date"])["boardings"]

H47 = pd.read_csv(EXT / "ictr_daily_hourly_2019_2020.csv", parse_dates=["date"])
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
hourly = pd.concat([
    H47[~H47.date.isin(covid)].pivot_table(index=["station_id", "date"], columns="hour", values="boardings"),
    tr.pivot_table(index=["station_id", "date"], columns="hour", values="boardings"),
])
hourly = hourly[~hourly.index.duplicated(keep="last")]

# ── 행사 라벨 (18번과 같은 것: 8·9·11번 + 배구) ─────────────────────────
lab = pd.read_csv(EXT / "event_labels_by_station_day.csv", parse_dates=["date"])
kovo = pd.read_csv(EXT / "games_kovo_incheon.csv", parse_dates=["date"])
k = kovo[kovo.station_id.notna()]
lab = pd.concat([lab, pd.DataFrame({"station_id": k.station_id.astype(int), "date": k.date,
                                    "label": "배구:" + k.home + "-" + k.away + "(" + k.start_time.astype(str) + ")"})])
lab = lab[(lab.date >= "2019-01-01") & (lab.date <= "2026-06-30") & ~lab.date.isin(covid)]
KIND = {"SSG": "SSG 야구", "부천FC": "부천FC 축구", "배구": "흥국생명 배구"}
lab["kind"] = lab.label.str.split(":").str[0].map(lambda h: KIND.get(h, h))
lab["event"] = lab.label.str.split(":", n=1).str[1].str.split(" / ").str[0]
lab["attendance"] = np.where(lab.kind == "SSG 야구",
                             pd.to_numeric(lab.event.str.extract(r"\((\d+)\)")[0], errors="coerce"), np.nan)
ev_days = set(zip(lab.station_id, lab.date))


def base_dates(sid, dt):
    c = [dt + pd.Timedelta(days=7 * w) for w in (-4, -3, -2, -1, 1, 2, 3, 4)]
    return [x for x in c if x not in hol and x not in covid and (sid, x) not in ev_days and (sid, x) in day.index]


def period(y):
    return "2019" if y <= 2020 else ("2022-24" if y <= 2024 else "2025-26")


# ── 1. 행사·경기 하루하루 ─────────────────────────────────────────────
rows = []
for (sid, dt), g in lab.groupby(["station_id", "date"]):
    if (sid, dt) not in day.index:
        continue
    b = base_dates(sid, dt)
    if len(b) < 3:
        continue
    today, usual = day.loc[(sid, dt)], day.loc[[(sid, x) for x in b]].median()
    r = dict(station_id=sid, station_name=names.get(sid), date=dt.date(), year=dt.year, period=period(dt.year),
             dow="월화수목금토일"[dt.day_of_week], is_holiday=int(dt in hol), kind=g.kind.iloc[0], event=g.event.iloc[0],
             attendance=g.attendance.iloc[0], day_boardings=round(today), usual_day=round(usual),
             day_ratio=round(today / usual, 3), extra_boardings=round(today - usual))
    hb = [x for x in b if (sid, x) in hourly.index]
    if (sid, dt) in hourly.index and len(hb) >= 3:
        ht, hu = hourly.loc[(sid, dt)], hourly.loc[[(sid, x) for x in hb]].median()
        diff = ht - hu
        ph = int(diff.idxmax())
        r.update(night_ratio=round(ht.loc[20:23].sum() / max(hu.loc[20:23].sum(), 1), 2), peak_hour=ph,
                 peak_hour_ratio=round(ht[ph] / max(hu[ph], 1), 1))
    rows.append(r)
E = pd.DataFrame(rows).sort_values(["kind", "station_id", "date"])
E.to_csv(OUT / "event_effects_by_day_2019_2026.csv", index=False, encoding="utf-8-sig")

S = (E.groupby(["kind", "station_id", "station_name", "period"])
       .agg(days=("date", "size"), day_ratio_median=("day_ratio", "median"), day_ratio_p75=("day_ratio", lambda s: s.quantile(.75)),
            extra_median=("extra_boardings", "median"))
       .round(3).reset_index())
S.to_csv(OUT / "event_effects_summary_2019_2026.csv", index=False, encoding="utf-8-sig")

# SSG: 관중 1명당 하루 추가 승차 (문학경기장역)
ssg = E[(E.kind == "SSG 야구") & (E.station_id == 3127) & E.attendance.notna() & (E.attendance > 0)]
slope = ssg.groupby("year").apply(lambda g: pd.Series({
    "games": len(g), "extra_per_fan": np.polyfit(g.attendance, g.extra_boardings, 1)[0],
    "corr": g.attendance.corr(g.extra_boardings)}), include_groups=False).round(3)

# ── 2. 공휴일 × 역 × 연도 ──────────────────────────────────────────────
dd = D[["station_id", "date", "boardings"]].rename(columns={"boardings": "b"}).copy()
dd = dd.join(cal[["holiday_name", "is_holiday", "is_chuseok", "is_seollal"]], on="date")
dd["dow"], dd["m"], dd["year"] = dd.date.dt.day_of_week, dd.date.dt.to_period("M"), dd.date.dt.year
wd = dd[(dd.is_holiday == 0) & (dd.dow < 5)].groupby(["station_id", "m"]).b.median().rename("wd")
sun = dd[(dd.is_holiday == 0) & (dd.dow == 6)].groupby(["station_id", "m"]).b.median().rename("sun")
dd = dd.join(wd, on=["station_id", "m"]).join(sun, on=["station_id", "m"])


def hgroup(r):
    n = str(r.holiday_name).strip()
    if r.is_seollal == 1 and n == "설날":
        return "설날 당일"
    if r.is_chuseok == 1 and n == "추석":
        return "추석 당일"
    if r.is_seollal == 1:
        return "설 연휴 (당일 제외)"
    if r.is_chuseok == 1 or "추석" in n:
        return "추석 연휴 (당일 제외)"
    if r.is_holiday == 1:
        return "기타 공휴일"
    return None


dd["hgroup"] = dd.apply(hgroup, axis=1)
Hh = dd[dd.hgroup.notna()].copy()
Hh["vs_weekday"], Hh["vs_sunday"] = Hh.b / Hh.wd, Hh.b / Hh.sun
HS = (Hh.groupby(["hgroup", "station_id", "year"]).agg(days=("date", "nunique"), vs_weekday=("vs_weekday", "median"),
                                                        vs_sunday=("vs_sunday", "median")).round(3).reset_index())
HS["station_name"] = HS.station_id.map(names)
HS.to_csv(OUT / "holiday_effects_by_station_2019_2026.csv", index=False, encoding="utf-8-sig")

# ── 3. 2019 추석 (2026과 같은 요일 배치: 전날 목 / 당일 금 / 다음날 토 / 일 / 월) ──
LAYOUT = {"2019-09-11": "D-2(수)", "2019-09-12": "전날(목)", "2019-09-13": "당일(금)", "2019-09-14": "다음날(토)",
          "2019-09-15": "D+2(일, 연휴 끝)", "2019-09-16": "D+3(월)"}
TO2026 = {"2019-09-11": "2026-09-23", "2019-09-12": "2026-09-24", "2019-09-13": "2026-09-25",
          "2019-09-14": "2026-09-26", "2019-09-15": "2026-09-27", "2019-09-16": "2026-09-28"}
hr = []
for ds, lbl in LAYOUT.items():
    dt = pd.Timestamp(ds)
    for sid in sorted({s for s, d in hourly.index if d == dt}):
        b = [x for x in base_dates(sid, dt) if (sid, x) in hourly.index]
        if len(b) < 3:
            continue
        ht, hu = hourly.loc[(sid, dt)], hourly.loc[[(sid, x) for x in b]].median()
        for h in range(6, 24):
            hr.append(dict(date_2019=ds, date_2026=TO2026[ds], day=lbl, station_id=sid, station_name=names.get(sid), hour=h,
                           boardings=ht[h], usual=hu[h], ratio=round(ht[h] / hu[h], 3) if hu[h] > 0 else np.nan))
CH = pd.DataFrame(hr)
CH.to_csv(OUT / "chuseok_2019_layout_hourly.csv", index=False, encoding="utf-8-sig")

# ── 4. 9/26과 비슷한 날 (역별 하루 비율, 평소 같은 요일 대비) ────────────────
ANALOG = {"2019-09-14": "추석 다음날 = 토 (2026과 같은 배치, 연휴 4일)",
          "2023-09-30": "추석 다음날 = 토 (연휴 6일)",
          "2022-09-10": "추석 당일 = 토",
          "2024-09-21": "추석 연휴 끝난 뒤 첫 토",
          "2025-10-04": "추석 연휴 중 토 (D-2, 연휴 7일)",
          "2025-10-11": "추석 연휴 끝난 뒤 첫 토"}
an = []
for ds, lbl in ANALOG.items():
    dt = pd.Timestamp(ds)
    for sid in sorted({s for s, d in day.index if d == dt}):
        b = base_dates(sid, dt)
        if len(b) < 3:
            continue
        an.append(dict(date=ds, kind=lbl, station_id=sid, station_name=names.get(sid),
                       day_ratio=round(day.loc[(sid, dt)] / day.loc[[(sid, x) for x in b]].median(), 3)))
AN = pd.DataFrame(an)
AN.to_csv(OUT / "chuseok_saturday_analogs.csv", index=False, encoding="utf-8-sig")

# ── 화면 요약 ─────────────────────────────────────────────────────────
pd.set_option("display.width", 220)
print("events", E.shape, "by period:", E.groupby("period").size().to_dict())
print("\n[SSG 문학경기장] 관중 1명당 하루 추가 승차\n", slope.to_string())
print("\n[행사 종류별 하루 배율 중앙값, 기간별]")
print(E.groupby(["kind", "period"]).day_ratio.median().unstack().round(2).to_string())


def block(df, hours):
    g = df[df.hour.isin(hours)].groupby(["day", "station_id"])[["boardings", "usual"]].sum()
    r = (g.boardings / g.usual).groupby(level=0)
    return pd.DataFrame({"median": r.median(), "p25": r.quantile(.25), "p75": r.quantile(.75),
                         "total": g.boardings.groupby(level=0).sum() / g.usual.groupby(level=0).sum()}).round(3)


for nm, hs in (("06~08시", [6, 7, 8]), ("17~18시", [17, 18]), ("06~23시", list(range(6, 24)))):
    print(f"\n[2019 추석, 역별 비율 (평소 같은 요일 대비)] {nm}\n", block(CH, hs).reindex(LAYOUT.values()).to_string())
print("\n[9/26과 비슷한 날, 역별 하루 비율]\n",
      AN.groupby(["date", "kind"]).day_ratio.describe()[["count", "25%", "50%", "75%"]].round(3).to_string())
