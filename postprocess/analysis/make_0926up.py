"""v20e → submission_v20e_0926up.csv : 9/26(추석 다음날 토) 06~08시·17~21시만 max(v20e, v10).
근거: analysis/bus_query_0926/RESULT_P3_FINAL.md (stcis 버스 5개 정류장 — 9/26 아침·저녁이 v20e보다 높게 추정됨).
입력: 팀 저장소 outputs/ensemble_v20/submission_ensemble_v20e.csv, outputs/ensemble_v10/submission_ensemble_v10.csv
사용: .venv/bin/python -I analysis/make_0926up.py <v20e 파일> <v10 파일> <출력 폴더>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V20E, V10, OUTDIR = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
te = pd.read_csv(ROOT / "data/test.csv")
a = pd.read_csv(V20E); b = pd.read_csv(V10)
assert (a.example_id.values == te.example_id.values).all() and (b.example_id.values == te.example_id.values).all()
m = ((te.date == "2026-09-26") & te.hour.isin([6, 7, 8, 17, 18, 19, 20, 21])).values
out = a.copy()
out.loc[m, "boardings"] = np.maximum(a.boardings.values[m], b.boardings.values[m])
OUTDIR.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTDIR / "submission_v20e_0926up.csv", index=False)
print("바뀐 칸", int((out.boardings.values != a.boardings.values).sum()), "→", OUTDIR / "submission_v20e_0926up.csv")
