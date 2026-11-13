"""2026년 8월 평일(공휴일 제외) 예측을 일정 비율 올린다. 7월 평일 ×1.02 (adjust_july_weekday.py, 퍼블릭 개선)와 같은 방법.

근거 (2026-11-13 밤 점검, 검단 연장 영향 역·문학·부천종합운동장·검단 2호선 북부·인천대공원 제외 57개 역)
  작년 같은 달 대비 평일 증가 (공휴일·앞뒤 날 제외, 역 중앙값)
    train 실측 3월 +3.2% · 4월 +4.4% · 5월 +5.9% · 6월 +3.6%
    기반 제출   7월 +2.7% (×1.02 반영 후) · 8월 +1.3% · 9월 +3.9% · 10월 +3.2%
  비 안 온 평일만: 3 ~ 6월 +3.1 ~ +5.3%, 8월 제출 +1.2%. 비 온 날 수는 2025년 8월 15일, 2026년 8월 17일로 비슷함.
  역별: 57개 역 중 54개가 (8월 증가율) < (그 역의 3 ~ 6월 증가율), 중앙 −3.3%. 시간대도 거의 전부 2 ~ 4%p 낮음.
  과거 8월 평일 ÷ 6월 평일: 2019 0.917, 2022 0.906, 2023 0.905, 2024 0.905, 2025 0.917 → 제출 0.900 (가장 낮음)
  → 두 근거의 중간쯤인 +2%.
대상: 2026-08 평일(8/17 광복절 대체휴일 제외) 전 시간, 검단 연장 영향 역(3107~3111)·문학경기장(3127) 제외.
사용: .venv/bin/python -I analysis/adjust_aug_weekday.py <factor, 예: 1.02> [기반 파일]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTOR = float(sys.argv[1]) if len(sys.argv) > 1 else 1.02
BASE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926_2019a50_jul2_0923d0927m_a50_ssg_0719_bfc_0711_octfest_night_exam.csv"
OUT = BASE.with_name(BASE.stem + f"_aug{round((FACTOR - 1) * 100)}.csv")
EXCL = {3107, 3108, 3109, 3110, 3111, 3127}

cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv").set_index("date")
te = pd.read_csv(ROOT / "data/test.csv")
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()

is_aug = te.date.str[:7] == "2026-08"
is_wd = te.date.map(lambda d: cal.loc[d, "is_holiday"] == 0 and cal.loc[d, "dow"] < 5)
mask = (is_aug & is_wd & ~te.station_id.isin(EXCL)).values

out = sub.copy()
out.loc[mask, "boardings"] = out.loc[mask, "boardings"] * FACTOR
assert out.boardings.notna().all() and (out.boardings >= 0).all() and len(out) == len(te)
assert (out.example_id.values == te.example_id.values).all()
out.to_csv(OUT, index=False)
d = np.log1p(out.boardings) - np.log1p(sub.boardings)
print(f"x{FACTOR}: 바뀐 칸 {mask.sum():,} ({mask.mean():.1%}), 날짜 {te.date[mask].nunique()}일, "
      f"log1p 변화 평균 {d[mask].mean():+.4f}, 두 제출 거리(RMSLE) {np.sqrt((d ** 2).mean()):.5f}")
print("저장:", OUT.relative_to(ROOT))
