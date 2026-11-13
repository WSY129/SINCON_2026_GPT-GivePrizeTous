"""10월 축제 3개 날을 2025년 같은 축제의 시간대별 효과로 절반만 올린다.

29번(events_venues_added.csv)에만 있고 12번 라벨에는 없는 축제라 모델이 모를 수 있다 (2026-11-13 밤 점검).
  부평풍물대축제 10/2 ~ 4 (부평 3120·부평시장 3119·부평구청 3118·3761) ← 2025-09-26 ~ 28 (금 ~ 일)
  드림파크 국화축제 10/17 · 10/24 토 (검암 3207)                      ← 2025-10-25 · 11-01 토
  인천퀴어문화축제 10/17 토 (인천시청 3124·3221)                       ← 2025-09-06 토
방법 (역 s, 시간 h)
  2025 효과 f[h] = 2025 그날 ÷ 평소(같은 요일 앞뒤 4주, 공휴일·행사일 제외 중앙값) — train 시간대 값
  작은 칸은 그날 하루 비율 쪽으로 줄임 (w = 평소 / (평소 + 30)), 축제 시간 안에서만, 1 ~ 3으로 제한
  새 값 = 기존 × f[h]^0.5  (모델에 축제 효과가 없다고 보고 절반만 더함. 10/3 개천절 효과는 기존 예측에 이미 있음)
제외: 10/18(일) 국화축제 — 큰비(56mm)라 기존 예측이 낮은 게 맞음
사용: .venv/bin/python -I analysis/adjust_oct_festivals.py [기반 파일]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926_2019a50_jul2_0923d0927m_a50_ssg_0719_bfc_0711.csv"
OUT = BASE.with_name(BASE.stem + "_octfest.csv")
K, ALPHA = 30, 0.5
JOBS = [  # (축제, 역들, [(2026 날짜, 2025 같은 날)], 축제 시간)
    ("부평풍물대축제", [3120, 3119, 3118, 3761], [("2026-10-02", "2025-09-26"), ("2026-10-03", "2025-09-27"), ("2026-10-04", "2025-09-28")], range(10, 24)),
    ("드림파크 국화축제", [3207], [("2026-10-17", "2025-10-25"), ("2026-10-24", "2025-11-01")], range(10, 20)),
    ("인천퀴어문화축제", [3124, 3221], [("2026-10-17", "2025-09-06")], range(12, 20)),
]

cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv").set_index("date")
te = pd.read_csv(ROOT / "data/test.csv")
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()
tr = pd.read_csv(ROOT / "data/train.csv")
TH = tr.pivot_table(index=["station_id", "date"], columns="hour", values="boardings")
tdates = set(tr.date)

lab = pd.read_csv(ROOT / "data/external/event_labels_by_station_day.csv")
busy = set(zip(lab.station_id, lab.date))
add = pd.read_csv(ROOT / "data/external/events_venues_added.csv")
for r in add[add.station_ids.notna()].itertuples():
    for sid in [int(float(x)) for x in str(r.station_ids).split(";") if x.strip()]:
        for d in pd.date_range(r.start_date, r.end_date):
            busy.add((sid, d.strftime("%Y-%m-%d")))

out = sub.set_index("example_id").copy()
key = te.set_index(["station_id", "date", "hour"]).example_id
rows = []
for name, sids, pairs, hours in JOBS:
    for d26, d25 in pairs:
        t = pd.Timestamp(d25)
        for sid in sids:
            cand = [(t + pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d") for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
            cand = [x for x in cand if x in tdates and (sid, x) not in busy and cal.loc[x, "is_holiday"] == 0 and (sid, x) in TH.index]
            if len(cand) < 3 or (sid, d25) not in TH.index:
                continue
            usual = TH.loc[[(sid, x) for x in cand]].median()
            today = TH.loc[(sid, d25)]
            day_r = today.loc[list(hours)].sum() / usual.loc[list(hours)].sum()
            f = {}
            for h in hours:
                w = usual[h] / (usual[h] + K)
                r_h = today[h] / usual[h] if usual[h] > 0 and today[h] > 0 else day_r
                f[h] = float(np.clip(np.exp(w * np.log(r_h) + (1 - w) * np.log(day_r)), 1.0, 3.0))
            ids = [key[(sid, d26, h)] for h in hours]
            before = out.loc[ids, "boardings"].sum()
            out.loc[ids, "boardings"] = out.loc[ids, "boardings"].values * np.array([f[h] ** ALPHA for h in hours])
            rows.append(dict(festival=name, date_2026=d26, from_2025=d25, station=sid, effect_2025=round(day_r, 2),
                             hours=f"{min(hours)}~{max(hours)}", change=round(out.loc[ids, "boardings"].sum() / before, 3)))
print(pd.DataFrame(rows).to_string(index=False))

out = out.reset_index()
assert (out.example_id.values == te.example_id.values).all()
assert out.boardings.notna().all() and (out.boardings >= 0).all()
out.to_csv(OUT, index=False)
dd = np.log1p(out.boardings) - np.log1p(sub.boardings)
print(f"바뀐 칸 {(dd != 0).sum():,}, 두 제출 거리(RMSLE) {np.sqrt((dd ** 2).mean()):.5f}")
print("저장:", OUT.relative_to(ROOT))
