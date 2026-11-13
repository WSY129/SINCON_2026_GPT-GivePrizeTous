"""분석용 공통 로더: train + 달력 + 모든 외부 자료를 (date, station_id, hour) 단위로 붙인다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import calendar  # noqa: E402

EXT = ROOT / "data/external"
OUT = ROOT / "outputs/analysis"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def meta():
    m = pd.read_csv(ROOT / "data/metaData.csv")
    m["name"] = m.station_name.str.replace(r"\(.*\)", "", regex=True)
    return m


def train():
    t = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
    return t


def cal():
    c = calendar("2024-12-01", "2026-11-30")
    return c


def rng(a, b):
    return pd.date_range(a, b)


def event_days():
    """역×날짜 → 행사 종류 표 (경기·공연·축제·컨벤시아·공연장·배구·부천FC·마라톤 등). 시작시각 포함."""
    rows = []

    def add(d, sid, kind, name, start=None, size=None, end=None, end_basis=None):
        # end: "HH:MM" 문자열(없으면 None). end_hour = 종료 시각이 속한 시간대(승차 hour와 같은 기준)
        end = end if isinstance(end, str) and end[:2].isdigit() else None
        rows.append({"date": pd.Timestamp(d), "station_id": int(sid), "kind": kind, "name": name,
                     "start_hour": start, "size": size, "end_time": end,
                     "end_hour": int(end[:2]) if end else None, "end_basis": end_basis if end else None})

    def plus(hhmm, minutes):
        if not (isinstance(hhmm, str) and hhmm[:2].isdigit()):
            return None
        t = pd.Timestamp("2000-01-01 " + hhmm[:5]) + pd.Timedelta(minutes=minutes)
        return t.strftime("%H:%M")

    g = pd.read_csv(EXT / "games_ssg_munhak.csv")
    for r in g.itertuples():
        st = int(str(r.sched_time)[:2]) if isinstance(r.sched_time, str) else None
        kind = "SSG경기" if r.status == "played" else "SSG우천취소"
        for sid in (3127, 3128):
            add(r.date, sid, kind, f"SSG vs {r.away}", st, r.attendance,
                r.end_time if r.status == "played" else None, "실제(KBO 기록)")
    f = pd.read_csv(EXT / "games_bucheonfc.csv")
    for r in f.itertuples():
        st = int(str(r.kickoff)[:2]) if isinstance(r.kickoff, str) else None
        for sid in (3754, 3755):
            add(r.date, sid, "부천FC", f"부천 vs {r.away}", st, r.attendance, plus(r.kickoff, 115), "추정(킥오프+115분)")
    k = pd.read_csv(EXT / "games_kovo_incheon.csv")
    for r in k.itertuples():
        st = int(str(r.start_time)[:2]) if isinstance(r.start_time, str) else None
        sids = (3759, 3758) if "삼산" in r.place else (3114, 3115)
        for sid in sids:
            add(r.date, sid, "V리그", f"{r.home} vs {r.away}", st, None, r.end_time, "실제(KOVO 기록)")
    for fn in ("events_venues.csv", "events_venues_added.csv"):
        e = pd.read_csv(EXT / fn)
        for r in e.itertuples():
            st = int(str(r.start_time)[:2]) if isinstance(r.start_time, str) and r.start_time[:2].isdigit() else None
            kind = {"concert": "콘서트", "festival": "축제", "seasonal": "꽃·계절", "sports": "기타스포츠"}.get(r.type, r.type)
            for d in rng(r.start_date, r.end_date):
                for sid in str(r.station_ids).split(";"):
                    if sid.strip().isdigit():
                        add(d, sid, kind, r.event, st, None, getattr(r, "end_time", None), "행사 파일(공식/추정은 time_note)")
    a = pd.read_csv(EXT / "artcenter_incheon_performances.csv")
    a = a[(a.hall == "대공연장") & ~a.title.astype(str).str.contains("취소|점검")]
    for r in a.itertuples():
        sh = getattr(r, "start_hour_first", None)
        sh = int(sh) if sh is not None and pd.notna(sh) else 19  # 상세 시각이 없는 옛 공연만 19시로 둬요
        if (pd.Timestamp(r.end_date) - pd.Timestamp(r.start_date)).days <= 7:
            for d in rng(r.start_date, r.end_date):
                add(d, 3125, "예술회관대공연장", r.title, sh)
    b = pd.read_csv(EXT / "bac_bucheon_performances.csv")
    b = b[b.hall == "콘서트홀"]
    for r in b.itertuples():
        if (pd.Timestamp(r.end_date) - pd.Timestamp(r.start_date)).days <= 7:
            for d in rng(r.start_date, r.end_date):
                for sid in (3757, 3756):
                    add(d, sid, "부천아트센터", r.title, 19)
    c = pd.read_csv(EXT / "convensia_events.csv")
    c = c[(c.n_halls >= 2) & (c.n_days <= 30)]
    for r in c.itertuples():
        for d in rng(r.start_date, r.end_date):
            for sid in (3137, 3136, 3138):
                add(d, sid, "컨벤시아대형", r.event_nm, 10)
    ev = pd.DataFrame(rows)
    return ev


def school_share():
    v = pd.read_csv(EXT / "school_vacation_daily.csv", parse_dates=["date"])
    return v[(v.area == "전체") & (v.level == "전체")][["date", "share_vacation"]]


def weather_daily():
    f = EXT / "weather_hourly_2015_2026.csv"
    if not f.exists():
        return None
    w = pd.read_csv(f, parse_dates=["time"])
    w = w[w.area == "incheon"]
    w["date"] = w.time.dt.normalize()
    w["hour"] = w.time.dt.hour
    return w
