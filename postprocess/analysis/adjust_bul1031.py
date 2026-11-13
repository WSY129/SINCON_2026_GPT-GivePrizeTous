"""10/31(토) 14시 문학 불꽃야구2 (vs 독립리그 대표팀) 날 문학경기장역(3127) 추가 승차를 과거 쪽으로 절반만 올린다.
adjust_bfc_0711.py 와 같은 방법.

근거: 과거 문학 불꽃야구 직관 4경기 (train, 평소 = 같은 요일 앞뒤 4주 중 경기·행사·공휴일 아닌 날 중앙값)
  2025-07-27 (일 18시, 2.3만 매진) 하루 추가 2,655명 — 시작 +4시간에 2,136명
  2025-09-14 (일 17시)             2,498명 — 시작 +3시간에 2,161명
  2026-05-30 (토 17시)             2,914명 — 시작 +3시간에 1,970명
  2026-06-13 (토 17시)             1,271명 — 시작 +3시간에   910명
  목표 = 중앙값 2,577명. 기반 제출의 추가는 약 1,000명 (17시 868명).
  10/31 관중·종료 시각은 확인 못 함 (예고 기사뿐, 나무위키 관중 칸 빈칸). 종료 추정 17:30.
방법: 평소 U[h] = 같은 요일 앞뒤 4주(부족하면 8주) 중 경기·행사·공휴일 아닌 날의 예측 중앙값,
      예측 추가 e[h] = max(예측[h] − U[h], 0), 합계 E
      새 합계 = sqrt(목표 × E) (절반), 배율 k = 새 합계 / E (0.8 ~ 1.6 제한), 새 값[h] = U[h] + k·e[h]
      → 모델이 잡은 시간 배분(16 ~ 18시)은 그대로 두고 크기만 키움. 선학(3128)은 과거 경기 날 변화가 없어 그대로.
사용: .venv/bin/python -I analysis/adjust_bul1031.py [기반 파일]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926_2019a50_jul2_0923d0927m_a50_ssg_0719_bfc_0711_octfest_night_exam.csv"
OUT = BASE.with_name(BASE.stem + "_bul1031.csv")
SID, DAY, TARGET = 3127, "2026-10-31", 2577.0

cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv").set_index("date")
te = pd.read_csv(ROOT / "data/test.csv")
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()
m = te.merge(sub, on="example_id")
dates = set(m.date)

busy = set(pd.read_csv(ROOT / "data/external/games_ssg_munhak.csv").date)  # 우천취소 포함
lab = pd.read_csv(ROOT / "data/external/event_labels_by_station_day.csv")
busy |= set(lab[lab.station_id == SID].date)
add = pd.read_csv(ROOT / "data/external/events_venues_added.csv")
for r in add[add.station_ids.notna()].itertuples():
    if str(SID) in str(r.station_ids):
        busy |= {d.strftime("%Y-%m-%d") for d in pd.date_range(r.start_date, r.end_date)}

mh = m[m.station_id == SID]
H = mh.pivot_table(index="date", columns="hour", values="boardings")
t = pd.Timestamp(DAY)
for span in (4, 8):
    cand = [(t + pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d") for k in range(-span, span + 1) if k]
    cand = [x for x in cand if x in dates and x not in busy and cal.loc[x, "is_holiday"] == 0]
    if len(cand) >= 3:
        break
U = H.loc[cand].median()
e = (H.loc[DAY] - U).clip(lower=0)
E = e.sum()
k = float(np.clip(np.sqrt(TARGET * E) / E, 0.8, 1.6))
new = U + k * e
hrs = e.index[e > 0]
print("평소 날:", cand)
print(pd.DataFrame({"평소": U, "기존": H.loc[DAY], "새 값": new}).loc[12:23].round(0).astype(int).T.to_string())
print(f"예측 추가 {E:,.0f} → {(new - U).clip(lower=0).sum():,.0f} (목표 {TARGET:,.0f}), k = {k:.3f}, "
      f"하루 평소 대비 {H.loc[DAY].sum() / U.sum():.2f} → {new.sum() / U.sum():.2f}")

out = sub.set_index("example_id").copy()
ids = mh[(mh.date == DAY) & mh.hour.isin(hrs)].set_index("hour").example_id
out.loc[ids.values, "boardings"] = new.loc[ids.index].values
out = out.reset_index()
assert (out.example_id.values == te.example_id.values).all()
assert out.boardings.notna().all() and np.isfinite(out.boardings).all() and (out.boardings >= 0).all()
out.to_csv(OUT, index=False)
dd = np.log1p(out.boardings) - np.log1p(sub.boardings)
print(f"바뀐 칸 {(dd != 0).sum():,}, 두 제출 거리(RMSLE) {np.sqrt((dd ** 2).mean()):.5f}")
print("저장:", OUT.relative_to(ROOT))
