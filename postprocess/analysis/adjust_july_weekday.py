"""2026년 7월 평일(공휴일 제외) 예측을 일정 비율 올린다.

근거 (README 6-3절 (20) 이후 점검, 2026-11-13 밤)
  7월 평일 ÷ 6월 평일 (역 중앙값): 과거 2019 0.956, 2022 0.980, 2023 0.956, 2024 0.938, 2025 0.939
  기반 제출 2026 예측 0.918 → 과거 모든 해보다 낮음. 비 안 온 날만 봐도 예측 0.898 vs 과거 0.924 ~ 0.976.
  가장 낮은 최근 해(2025)에 맞추는 정도 = 약 +2%.
대상: 2026-07 평일(7/17 제헌절 제외) 전 시간, 검단 연장 영향 역(3107~3111)·문학경기장(3127) 제외.
사용: .venv/bin/python -I analysis/adjust_july_weekday.py <factor, 예: 1.02> [기반 파일]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTOR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.02
BASE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926_2019a50.csv"
OUT = BASE.with_name(BASE.stem + f"_jul{round((FACTOR - 1) * 100)}.csv")
EXCL = {3107, 3108, 3109, 3110, 3111, 3127}

cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv").set_index("date")
te = pd.read_csv(ROOT / "data/test.csv")
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()

is_jul = te.date.str[:7] == "2026-07"
is_wd = te.date.map(lambda d: cal.loc[d, "is_holiday"] == 0 and cal.loc[d, "dow"] < 5)
mask = (is_jul & is_wd & ~te.station_id.isin(EXCL)).values

out = sub.copy()
out.loc[mask, "boardings"] = out.loc[mask, "boardings"] * FACTOR
assert out.boardings.notna().all() and (out.boardings >= 0).all() and len(out) == len(te)
out.to_csv(OUT, index=False)
d = np.log1p(out.boardings) - np.log1p(sub.boardings)
print(f"x{FACTOR}: 바뀐 칸 {mask.sum():,} ({mask.mean():.1%}), 날짜 {te.date[mask].nunique()}일, "
      f"log1p 변화 평균 {d[mask].mean():+.4f}, 두 제출 거리(RMSLE) {np.sqrt((d ** 2).mean()):.5f}")
print("저장:", OUT.relative_to(ROOT))
