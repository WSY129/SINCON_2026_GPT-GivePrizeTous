"""추석 주변 날(9/23 낮, 9/27 아침)을 2019 같은 요일 배치의 실제 비율로 조정한다. (adjust_0926_2019.py와 같은 방법)

근거 (README 6-3절 (20), 2026-11-13 밤 점검)
  9/23(수, 추석 2일 전) 09~16시: 2019-09-11 실측 1.19 vs 예측 1.02. 하루로는 2019 1.06, 2022 1.02, 2023(수) 1.03 → 매년 평소보다 많음
  9/27(일, 연휴 끝)   06~08시: 2019-09-15 실측 0.83 vs 예측 0.95 (일요일이 연휴 끝인 해는 2019뿐)
방법 (역 s, 시간 h)
  U = 기반 제출에서 같은 요일 앞뒤 4주(공휴일 제외) 예측 중앙값
  r = 2019 비율, 로그에서 시간대 전체 중앙값 쪽으로 줄임 (w = u19 / (u19 + 30)), 2019 자료 없는 역은 시간대 중앙값
  새 값 = exp(α·log(U·r) + (1−α)·log 기존)
제외: 2019 그날 경기가 있던 역 (9/11 문학 SSG, 9/15 문학 SSG·부천FC)
사용: .venv/bin/python -I analysis/adjust_chuseok_days_2019.py [alpha=0.5] [기반 파일]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
BASE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926_2019a50_jul2.csv"
OUT = BASE.with_name(BASE.stem + f"_0923d0927m_a{int(ALPHA * 100)}.csv")
K = 30
# (2026 날짜, 2019 같은 날 라벨, 바꿀 시간, 제외 역)
JOBS = [("2026-09-23", "D-2(수)", list(range(9, 17)), {3127}),
        ("2026-09-27", "D+2(일, 연휴 끝)", [6, 7, 8], {3127, 3754})]

cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv").set_index("date")
te = pd.read_csv(ROOT / "data/test.csv")
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()
m = te.merge(sub, on="example_id")
dates = set(m.date)
ch = pd.read_csv(ROOT / "analysis/chuseok_2019_layout_hourly.csv")

out = sub.copy().set_index("example_id")
for d26, lbl, hours, excl in JOBS:
    t = pd.Timestamp(d26)
    same = [(t + pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d") for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
    same = [x for x in same if x in dates and cal.loc[x, "is_holiday"] == 0]
    U = m[m.date.isin(same)].groupby(["station_id", "hour"]).boardings.median().rename("U")
    c = ch[(ch.day == lbl) & ~ch.station_id.isin(excl) & (ch.usual > 0) & (ch.boardings > 0)].copy()
    glob = c.groupby("hour").ratio.median().rename("r_glob")
    c = c.join(glob, on="hour")
    w = c.usual / (c.usual + K)
    c["r"] = np.exp(w * np.log(c.ratio) + (1 - w) * np.log(c.r_glob))
    d = m[(m.date == d26) & m.hour.isin(hours) & ~m.station_id.isin(excl)].copy()
    d = d.join(U, on=["station_id", "hour"]).join(c.set_index(["station_id", "hour"]).r, on=["station_id", "hour"]).join(glob, on="hour")
    d["r"] = d.r.fillna(d.r_glob)
    d["new"] = np.exp(ALPHA * np.log(d.U * d.r) + (1 - ALPHA) * np.log(d.boardings))
    out.loc[d.example_id, "boardings"] = d.set_index("example_id").new
    before = d.boardings.sum() / d.U.sum()
    after = d.new.sum() / d.U.sum()
    ch_h = c[c.hour.isin(hours)]
    tgt = ch_h.boardings.sum() / ch_h.usual.sum()
    print(f"{d26} {min(hours):02d}~{max(hours):02d}시: 평소 대비 기존 {before:.3f} → 새 {after:.3f} (2019 실측 {tgt:.3f}), 바뀐 칸 {len(d)}")

out = out.reset_index()
assert out.boardings.notna().all() and (out.boardings >= 0).all() and len(out) == len(te)
assert (out.example_id.values == te.example_id.values).all()
out.to_csv(OUT, index=False)
print("저장:", OUT.relative_to(ROOT))
