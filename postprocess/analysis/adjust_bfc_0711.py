"""두 가지 행사 날 추가 승차를 과거 쪽으로 절반만 올린다 (adjust_ssg_0719.py와 같은 방법).

① 부천FC 홈경기 날 부천종합운동장역(3754)
   과거 2025 ~ 2026 상반기 경기: 하루 추가 승차 ≈ 관중 × 0.265 (상관 0.82).
   2026 상반기 K리그1 큰 경기(관중 8,500 ~ 10,000)도 추가 2,400 ~ 3,800명.
   기반 제출: 7/19(9,218명) 0.52배, 8/8(5,187명) 0.58배, 8/16(9,268명) 0.47배. 7/22·10/24는 1.0배.
② 7/11 흠뻑쇼 날 아시아드경기장역(3209)
   과거 하루 공연 해: 2022 추가 3,887명(2.17배), 2023 4,369명(2.35배), 2024 토 3,820명. 2025(이틀 공연) 2,428 · 2,467명.
   목표 = 2022 ~ 2024 토요일 추가의 중앙값 3,887명. 기반 제출 2,145명(1.65배).
방법: 평소 U[h] = 같은 요일 앞뒤 4주 중 행사·공휴일 아닌 날의 예측 중앙값, 예측 추가 e[h] = max(예측[h] − U[h], 0), 합계 E
      새 합계 = sqrt(목표 × E) (절반), 배율 k = 새 합계 / E (0.8 ~ 1.6 제한), 새 값[h] = U[h] + k·e[h]
사용: .venv/bin/python -I analysis/adjust_bfc_0711.py [기반 파일]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926_2019a50_jul2_0923d0927m_a50_ssg_0719.csv"
OUT = BASE.with_name(BASE.stem + "_bfc_0711.csv")

cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv").set_index("date")
te = pd.read_csv(ROOT / "data/test.csv")
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()
m = te.merge(sub, on="example_id")
dates = set(m.date)
lab = pd.read_csv(ROOT / "data/external/event_labels_by_station_day.csv")
fc = pd.read_csv(ROOT / "data/external/games_bucheonfc.csv")
out = sub.set_index("example_id").copy()

jobs = []  # (역, 날짜, 목표 추가, 설명)
for _, g in fc[(fc.date >= "2026-07-01") & (fc.date <= "2026-10-31") & fc.attendance.notna()].iterrows():
    jobs.append((3754, g.date, 0.265 * g.attendance, f"부천FC {g.away} ({int(g.attendance):,}명)"))
jobs.append((3209, "2026-07-11", 3887.0, "흠뻑쇼"))

rows = []
for sid, d, T, desc in jobs:
    if cal.loc[d, "is_holiday"] == 1:
        continue
    mh = m[m.station_id == sid]
    H = mh.pivot_table(index="date", columns="hour", values="boardings")
    busy = set(lab[lab.station_id == sid].date) | (set(fc.date) if sid == 3754 else set())
    t = pd.Timestamp(d)
    for span in (4, 8):  # 앞뒤 4주에 평소 날이 3일 미만이면 8주로 넓힘 (8/22 ~ 9/19 토요일 홈경기 5연속)
        cand = [(t + pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d") for k in range(-span, span + 1) if k]
        cand = [x for x in cand if x in dates and x not in busy and cal.loc[x, "is_holiday"] == 0]
        if len(cand) >= 3:
            break
    if len(cand) < 3:
        print("평소 날 부족, 건너뜀:", sid, d)
        continue
    U = H.loc[cand].median()
    e = (H.loc[d] - U).clip(lower=0)
    E = e.sum()
    k = float(np.clip(np.sqrt(T * E) / E, 0.8, 1.6))
    new = U + k * e
    hrs = e.index[e > 0]
    ids = mh[(mh.date == d) & mh.hour.isin(hrs)].set_index("hour").example_id
    out.loc[ids.values, "boardings"] = new.loc[ids.index].values
    rows.append(dict(station=sid, date=d, event=desc, pred_extra=round(E), target=round(T), k=round(k, 3),
                     new_extra=round((new - U).clip(lower=0).sum()), day_ratio_before=round(H.loc[d].sum() / U.sum(), 2),
                     day_ratio_after=round(new.sum() / U.sum(), 2)))
print(pd.DataFrame(rows).to_string(index=False))

out = out.reset_index()
assert (out.example_id.values == te.example_id.values).all()
assert out.boardings.notna().all() and (out.boardings >= 0).all()
out.to_csv(OUT, index=False)
dd = np.log1p(out.boardings) - np.log1p(sub.boardings)
print(f"바뀐 칸 {(dd != 0).sum():,}, 두 제출 거리(RMSLE) {np.sqrt((dd ** 2).mean()):.5f}")
print("저장:", OUT.relative_to(ROOT))
