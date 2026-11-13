"""9 ~ 10월 밤 20 ~ 23시 비중을 과거만큼 줄인다 (절반만).

근거 (2026-11-13 밤 점검): 6월 대비 그 달의 20 ~ 23시 비중 변화 log(그 달 ÷ 6월), 역 중앙값
  10월 평일 2019 −0.084, 2025 −0.065, 예측 −0.049 / 10월 일요일 −0.105, −0.094, −0.054 / 9월도 같은 방향
방법: 요일 종류(평일·토·일) × 달(9·10)마다 gap = 예측 변화 − 2025 변화 (덜 극단적인 해 기준)
      gap > 0이면 그 달 그 요일 종류의 20 ~ 23시에 exp(−gap/2) 곱함
대상: 공휴일 아닌 날. 제외: 검단 연장 영향 역(3107 ~ 3111)·문학경기장, 경기·행사가 있는 역·날
사용: .venv/bin/python -I analysis/adjust_autumn_night.py [기반 파일]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926_2019a50_jul2_0923d0927m_a50_ssg_0719_bfc_0711_octfest.csv"
OUT = BASE.with_name(BASE.stem + "_night.csv")
EXCL = {3107, 3108, 3109, 3110, 3111, 3127}
NIGHT = [20, 21, 22, 23]

cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv").set_index("date")
te = pd.read_csv(ROOT / "data/test.csv")
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()
p = te.merge(sub, on="example_id")
tr = pd.read_csv(ROOT / "data/train.csv")

busy = set()
lab = pd.read_csv(ROOT / "data/external/event_labels_by_station_day.csv")
busy |= set(zip(lab.station_id, lab.date))
add = pd.read_csv(ROOT / "data/external/events_venues_added.csv")
for r in add[add.station_ids.notna()].itertuples():
    for sid in [int(float(x)) for x in str(r.station_ids).split(";") if x.strip()]:
        for d in pd.date_range(r.start_date, r.end_date):
            busy.add((sid, d.strftime("%Y-%m-%d")))
fc = pd.read_csv(ROOT / "data/external/games_bucheonfc.csv")
busy |= {(3754, d) for d in fc.date}


def kind(d):
    w = cal.loc[d, "dow"]
    return "wd" if w < 5 else ("sat" if w == 5 else "sun")


def night_change(df, y, m, k):
    def share(mm):
        d = df[(df.date.str[:7] == f"{y}-{mm:02d}") & ~df.station_id.isin(EXCL)]
        ok = d.date.map(lambda x: cal.loc[x, "is_holiday"] == 0 and kind(x) == k)
        g = d[ok].groupby(["station_id", "hour"]).boardings.sum()
        return g / g.groupby(level=0).transform("sum")
    base_df = tr if y == 2026 else df
    cur = share(m)
    d6 = base_df[(base_df.date.str[:7] == f"{y}-06") & ~base_df.station_id.isin(EXCL)]
    ok = d6.date.map(lambda x: cal.loc[x, "is_holiday"] == 0 and kind(x) == k)
    g6 = d6[ok].groupby(["station_id", "hour"]).boardings.sum()
    jun = g6 / g6.groupby(level=0).transform("sum")
    r = np.log(cur / jun).replace([np.inf, -np.inf], np.nan)
    return r.groupby(level=1).median().loc[NIGHT].mean()


out = sub.set_index("example_id").copy()
key = te.set_index(["station_id", "date", "hour"]).example_id
rows = []
for m in (9, 10):
    for k in ("wd", "sat", "sun"):
        hist = night_change(tr, 2025, m, k)
        pred = night_change(p, 2026, m, k)
        gap = pred - hist
        f = float(np.exp(-gap / 2)) if gap > 0 else 1.0
        n = 0
        if f < 1:
            days = [d for d in sorted(set(p.date)) if d[:7] == f"2026-{m:02d}" and cal.loc[d, "is_holiday"] == 0 and kind(d) == k]
            for d in days:
                for sid in sorted(set(te.station_id) - EXCL):
                    if (sid, d) in busy:
                        continue
                    ids = [key[(sid, d, h)] for h in NIGHT]
                    out.loc[ids, "boardings"] *= f
                    n += len(ids)
        rows.append(dict(month=m, kind=k, change_2025=round(hist, 3), change_pred=round(pred, 3), gap=round(gap, 3), factor=round(f, 4), cells=n))
print(pd.DataFrame(rows).to_string(index=False))

out = out.reset_index()
assert (out.example_id.values == te.example_id.values).all()
assert out.boardings.notna().all() and (out.boardings >= 0).all()
out.to_csv(OUT, index=False)
dd = np.log1p(out.boardings) - np.log1p(sub.boardings)
print(f"바뀐 칸 {(dd != 0).sum():,}, 두 제출 거리(RMSLE) {np.sqrt((dd ** 2).mean()):.5f}")
print("저장:", OUT.relative_to(ROOT))
