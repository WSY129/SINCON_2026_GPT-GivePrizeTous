"""KOPIS(공연예술통합전산망) 공연장 일별 예매 수로 두 날을 고친다 (2026-11-13 밤).
  8/29(토) 문학경기장(3127) 20~22시 ×0.6 : Sustainable Wave Festival 예매 4,968장 (kopis_big_days.csv) — 기존 가정(이틀 약 4만 명)보다 작음
  10/31(토) 남동구청(3225) 20~22시 ×1.4 : 인천남동체육관 공연 예매 2,267장 — 기존 예측에 행사 없음
근거 표: analysis/kopis_check_events.py → analysis/kopis_big_days.csv (data/external/kopis_venue_daily.csv)
사용: .venv/bin/python -I analysis/adjust_kopis.py <기반 파일>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(sys.argv[1])
OUT = BASE.with_name(BASE.stem + "_kopis.csv")
JOBS = [("2026-08-29", 3127, [20, 21, 22], 0.6), ("2026-10-31", 3225, [20, 21, 22], 1.4)]
te = pd.read_csv(ROOT / "data/test.csv"); sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()
out = sub.copy()
for d, sid, hrs, k in JOBS:
    m = ((te.date == d) & (te.station_id == sid) & te.hour.isin(hrs)).values
    out.loc[m, "boardings"] = out.loc[m, "boardings"] * k
assert out.boardings.notna().all() and (out.boardings >= 0).all()
out.to_csv(OUT, index=False)
print("바뀐 칸", int((out.boardings.values != sub.boardings.values).sum()), "→", OUT.name)
