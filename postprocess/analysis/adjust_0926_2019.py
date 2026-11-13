"""9/26(추석 다음날 토) 예측을 2019-09-14(요일 배치가 같은 날) 실제 비율로 아침·저녁만 조정한다.

근거: analysis/chuseok_2019_layout_hourly.csv (README 6-3절 (20)). 평가 기간 지하철 자료는 쓰지 않는다.
방법 (역 s, 시간 h)
  1) 평소 토요일 예측 U[s,h] = 기반 제출의 2026-08-29 ~ 10-24 토요일(10/3 개천절 제외 7일) 중앙값
  2) 2019 비율 r[s,h] = 2019-09-14 ÷ 평소 같은 요일(앞뒤 4주) — 로그에서 시간대 전체 중앙값 쪽으로 줄임
     가중 w = u19 / (u19 + 30) (u19 = 2019 평소 승차, 적은 칸일수록 전체 중앙값을 따름)
     2019 자료 없는 역(7호선·검단 3역)은 시간대 전체 중앙값만 씀
     문학경기장은 조정하지 않음 (2019-09-14 SSG 경기 있었고, 2026 평소 토요일 예측에 경기 저녁이 섞임)
  3) 목표 T = U × r,  새 값 = exp(α·log T + (1−α)·log 기존)  — 아침 06~09시, 저녁 18~23시만
사용: .venv/bin/python -I analysis/adjust_0926_2019.py [alpha=0.5]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
BASE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926up.csv"
OUT = BASE.with_name(f"submission_v20e_0926_2019a{int(ALPHA * 100)}.csv")
HOURS = [6, 7, 8, 9, 18, 19, 20, 21, 22, 23]
SATS = ["2026-08-29", "2026-09-05", "2026-09-12", "2026-09-19", "2026-10-10", "2026-10-17", "2026-10-24"]
K = 30
EXCLUDE_2019 = {3127}  # 2019-09-14 문학 SSG 경기

te = pd.read_csv(ROOT / "data/test.csv")
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()
m = te.merge(sub, on="example_id")

U = m[m.date.isin(SATS)].groupby(["station_id", "hour"]).boardings.median().rename("U")
c = pd.read_csv(ROOT / "analysis/chuseok_2019_layout_hourly.csv")
c = c[(c.day == "다음날(토)") & ~c.station_id.isin(EXCLUDE_2019) & (c.usual > 0) & (c.boardings > 0)]
glob = c.groupby("hour").ratio.median().rename("r_glob")
c = c.join(glob, on="hour")
w = c.usual / (c.usual + K)
c["r"] = np.exp(w * np.log(c.ratio) + (1 - w) * np.log(c.r_glob))
r = c.set_index(["station_id", "hour"]).r

# 문학경기장은 손대지 않음: 평소 토요일 예측(U)에 SSG 경기 날 저녁이 섞여 부풀어 있고, 9/26엔 경기 없음
d = m[(m.date == "2026-09-26") & m.hour.isin(HOURS) & ~m.station_id.isin(EXCLUDE_2019)].copy()
d = d.join(U, on=["station_id", "hour"]).join(r, on=["station_id", "hour"]).join(glob, on="hour")
d["r"] = d.r.fillna(d.r_glob)
d["T"] = d.U * d.r
d["new"] = np.exp(ALPHA * np.log(d["T"]) + (1 - ALPHA) * np.log(d.boardings))

out = sub.copy()
out.loc[out.example_id.isin(d.example_id), "boardings"] = out.example_id.map(d.set_index("example_id").new)
assert out.boardings.notna().all() and (out.boardings >= 0).all() and len(out) == len(te)
out.to_csv(OUT, index=False)

# 요약: 9/26 ÷ 평소 토요일(예측) — 문학경기장 제외 합계
mm = te.merge(out, on="example_id")
nine = mm[mm.date == "2026-09-26"].set_index(["station_id", "hour"]).boardings
base9 = m[m.date == "2026-09-26"].set_index(["station_id", "hour"]).boardings
ref = U[U.index.get_level_values(0) != 3127]
for nm, hs in (("06~08", [6, 7, 8]), ("17~18", [17, 18]), ("18~23", list(range(18, 24))), ("06~23", list(range(6, 24)))):
    sel = ref[ref.index.get_level_values(1).isin(hs)]
    print(f"{nm}: 기존 {base9.reindex(sel.index).sum() / sel.sum():.3f} → 새 {nine.reindex(sel.index).sum() / sel.sum():.3f}")
lg = np.log1p(d.new) - np.log1p(d.boardings)
print(f"바뀐 칸 {len(d):,}, log1p 변화 평균 {lg.mean():+.3f} (최소 {lg.min():+.3f}, 최대 {lg.max():+.3f})")
print("저장:", OUT.relative_to(ROOT))
