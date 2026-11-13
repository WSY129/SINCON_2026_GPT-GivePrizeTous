"""두 가지를 조정한다 (근거: past_effects_daily.py로 만든 2019 ~ 2026 효과 표, 2026-11-13 밤 점검)

① SSG 홈경기 날 문학경기장역(3127): 경기로 늘어나는 승차를 과거 관계 쪽으로 올림
   과거 2022 ~ 2025 경기 294개: 하루 추가 승차 ≈ 관중 × (평일 0.308 / 토 0.280 / 일 0.272)
   기반 제출의 경기 날 추가 승차는 이 관계의 약 0.76배 (일요일이 특히 낮음)
   방법: 평소 U[h] = 같은 요일 앞뒤 4주 중 경기·행사·공휴일 아닌 날의 예측 중앙값
         예측 추가 e[h] = max(예측[h] − U[h], 0), 합계 E. 목표 T = 기울기 × 관중
         새 합계 E' = exp(α·log T + (1−α)·log E) (α=0.5), 배율 k = E'/E (0.8 ~ 1.6로 제한)
         새 값[h] = U[h] + k·e[h] (예측이 평소보다 많은 시간만)
   제외: 공휴일 경기(7/17·9/24·10/9) — 과거 공휴일 경기는 11개뿐이고 추석 전날은 성격이 다름
② 7/19(일, 7/17 금요일 공휴일 뒤 3일 연휴 끝): 과거 같은 배치의 일요일은 평소의 0.87 ~ 0.99 (중앙값 0.951)
   기반 제출은 비 안 온 일요일 대비 r. r > 0.951이면 전 역·전 시간에 sqrt(0.951 / r) 곱함 (절반만)
   제외: 그날 경기가 있는 문학경기장(SSG)·부천종합운동장(부천FC)
사용: .venv/bin/python -I analysis/adjust_ssg_0719.py [기반 파일]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926_2019a50_jul2_0923d0927m_a50.csv"
OUT = BASE.with_name(BASE.stem + "_ssg_0719.csv")
ALPHA = 0.5
SLOPE = {"평일": 0.308, "토": 0.280, "일": 0.272}
MUNHAK = 3127

cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv").set_index("date")
te = pd.read_csv(ROOT / "data/test.csv")
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()
m = te.merge(sub, on="example_id")
dates = set(m.date)
out = sub.set_index("example_id").copy()

# ── ① SSG ──────────────────────────────────────────────────────────────
games = pd.read_csv(ROOT / "data/external/games_ssg_munhak.csv")
games = games[(games.date >= "2026-07-01") & (games.date <= "2026-10-31")]
lab = pd.read_csv(ROOT / "data/external/event_labels_by_station_day.csv")
busy = set(lab[lab.station_id == MUNHAK].date) | set(games.date)  # 경기(취소 포함)·축제·공연 날은 평소에서 뺌
mh = m[m.station_id == MUNHAK]
hourly = mh.pivot_table(index="date", columns="hour", values="boardings")
rows = []
for _, g in games[games.status == "played"].iterrows():
    d = g.date
    if cal.loc[d, "is_holiday"] == 1:
        continue
    t = pd.Timestamp(d)
    cand = [(t + pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d") for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
    cand = [x for x in cand if x in dates and x not in busy and cal.loc[x, "is_holiday"] == 0]
    if len(cand) < 3:
        continue
    U = hourly.loc[cand].median()
    pred = hourly.loc[d]
    e = (pred - U).clip(lower=0)
    E = e.sum()
    typ = "일" if t.day_of_week == 6 else ("토" if t.day_of_week == 5 else "평일")
    T = SLOPE[typ] * g.attendance
    k = float(np.clip(np.exp(ALPHA * np.log(T) + (1 - ALPHA) * np.log(E)) / E, 0.8, 1.6))
    new = U + k * e
    hrs = e.index[e > 0]
    ids = mh[(mh.date == d) & mh.hour.isin(hrs)].set_index("hour").example_id
    out.loc[ids.values, "boardings"] = new.loc[ids.index].values
    rows.append(dict(date=d, type=typ, attendance=int(g.attendance), pred_extra=round(E), target=round(T),
                     k=round(k, 3), new_extra=round((new - U).clip(lower=0).sum()), hours=f"{hrs.min()}~{hrs.max()}"))
R = pd.DataFrame(rows)
print(R.to_string(index=False))
print(f"① SSG: {len(R)}경기, 예측 추가 합 {R.pred_extra.sum():,} → {R.new_extra.sum():,} (과거 관계 {R.target.sum():,})")

# ── ② 7/19 ────────────────────────────────────────────────────────────
w = pd.read_csv(ROOT / "data/external/weather_hourly_2015_2026.csv")
rain = w[(w.area == "incheon") & w.hour.between(6, 23)].groupby("date").precipitation.sum()
D19 = "2026-07-19"
skip = {MUNHAK, 3754}
day = m[~m.station_id.isin(skip)].groupby(["date", "station_id"]).boardings.sum()
t = pd.Timestamp(D19)
cand = [(t + pd.Timedelta(days=7 * k)).strftime("%Y-%m-%d") for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
cand = [x for x in cand if x in dates and cal.loc[x, "is_holiday"] == 0 and rain.get(x, 0) < 1]
r = day.loc[D19].sum() / day.loc[cand].groupby(level="station_id").median().sum()
TARGET = 0.951
f = float(np.sqrt(TARGET / r)) if r > TARGET else 1.0
mask = (te.date == D19) & ~te.station_id.isin(skip)
out.loc[te.example_id[mask].values, "boardings"] *= f
print(f"② 7/19: 비 안 온 일요일 {cand} 대비 {r:.3f} → 배율 {f:.4f} 적용 ({mask.sum()}칸)")

out = out.reset_index()
assert (out.example_id.values == te.example_id.values).all()
assert out.boardings.notna().all() and (out.boardings >= 0).all()
out.to_csv(OUT, index=False)
d = np.log1p(out.boardings) - np.log1p(sub.boardings)
print(f"바뀐 칸 {(d != 0).sum():,}, 두 제출 거리(RMSLE) {np.sqrt((d ** 2).mean()):.5f}")
print("저장:", OUT.relative_to(ROOT))
