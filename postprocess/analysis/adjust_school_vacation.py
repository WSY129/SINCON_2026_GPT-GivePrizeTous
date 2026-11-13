"""[쓰지 않음 2026-11-13 밤: 시간대별 보정이 잡음 수준(요일마다 하교 시각이 달라 16~17시 차이가 시간대별로 안 잡힘). 제출 파일 만들지 말 것]
학교 앞 8개 역의 방학 날(학교 홈페이지 일정 기준) 시간대 모양을 train 쪽으로 절반만 맞춘다.

근거: school_features_validation.csv — 방학 날 ÷ 평범한 날 (같은 역·요일 앞뒤 3주)
  train 16~17시: 부평삼거리 −0.62, 독정 −0.39, 인천가좌 −0.13, 가재울 −0.12
  기반 제출    : 부평삼거리 −0.41, 독정 −0.47, 인천가좌 −0.24, 가재울 −0.18  (검바위·만수·아시아드·마전은 거의 같음)
방법 (역 s, 시간 h)
  t[h] = train 방학 날 로그 비율 중앙값, p[h] = 기반 제출 test 방학 날 로그 비율 중앙값 (같은 방법)
  c[h] = t[h] − p[h], |c| < 0.08 은 0, ±0.4 제한. 새 값 = 기존 × exp(0.5 · c[h])  (test 방학 평일, 공휴일·앞뒤 날 제외)
사용: .venv/bin/python -I analysis/adjust_school_vacation.py [기반 파일]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926_2019a50_jul2_0923d0927m_a50_ssg_0719_bfc_0711_octfest_night_exam_3208a75_kopis.csv"
OUT = BASE.with_name(BASE.stem + "_vac.csv")
ALPHA, MIN_C, CLIP = 0.5, 0.08, 0.4

f = pd.read_csv(ROOT / "data/external/school_features_station_day.csv", parse_dates=["date"])
f["state"] = np.select([f.hs_exam_main == 1, f.vacation_share >= 0.5, f.closed_n > 0, f.ceremony_n > 0, f.trip_n > 0,
                        f.hs_exam_n + f.ms_exam_n > 0], ["exam", "vac", "closed", "cer", "trip", "oexam"], "normal")
cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv", parse_dates=["date"])
hol = set(cal.date[cal.is_holiday == 1]); adj = hol | {d + pd.Timedelta(days=k) for d in hol for k in (-1, 1)}
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
te = pd.read_csv(ROOT / "data/test.csv", parse_dates=["date"])
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()
te["boardings"] = sub.boardings.values


def vac_effect(df, sid, st):
    x = df[(df.station_id == sid) & (df.date.dt.day_of_week < 5) & ~df.date.isin(adj)]
    P = x.pivot(index="date", columns="hour", values="boardings")
    normal = [d for d in P.index if st.get(d) == "normal"]
    L, days = [], []
    for d in P.index:
        if st.get(d) != "vac":
            continue
        nb = [q for q in normal if q.day_of_week == d.day_of_week and 0 < abs((q - d).days) <= 21]
        if len(nb) >= 2:
            L.append(np.log((P.loc[d] + 1) / (P.loc[nb].median() + 1))); days.append(d)
    return (pd.DataFrame(L).median() if L else None), days


out = te.copy()
rows = []
for sid, g in f.groupby("station_id"):
    st = g.set_index("date").state
    t, _ = vac_effect(tr, sid, st)
    p, days = vac_effect(te, sid, st)
    if t is None or p is None:
        continue
    c = (t - p).where((t - p).abs() >= MIN_C, 0.0).clip(-CLIP, CLIP)
    if (c == 0).all():
        continue
    idx = out.index[(out.station_id == sid) & out.date.isin(days)]
    before = out.loc[idx, "boardings"].sum()
    out.loc[idx, "boardings"] = out.loc[idx, "boardings"].values * np.exp(ALPHA * c.reindex(out.loc[idx, "hour"]).values)
    rows.append(dict(station=sid, days=len(days), changed_hours=",".join(str(h) for h in c.index[c != 0]),
                     c16_17=round(c[[16, 17]].mean(), 2), day_change=round(out.loc[idx, "boardings"].sum() / before, 3)))
print(pd.DataFrame(rows).to_string(index=False))

res = out[["example_id", "boardings"]]
assert len(res) == len(te) and (res.example_id.values == te.example_id.values).all()
assert res.boardings.notna().all() and np.isfinite(res.boardings).all() and (res.boardings >= 0).all()
res.to_csv(OUT, index=False)
dd = np.log1p(res.boardings.values) - np.log1p(sub.boardings.values)
print(f"바뀐 칸 {(dd != 0).sum():,}, 두 제출 거리(RMSLE) {np.sqrt((dd ** 2).mean()):.5f}")
print("저장:", OUT.relative_to(ROOT))
