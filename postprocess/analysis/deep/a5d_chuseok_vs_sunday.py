"""5d. 추석 연휴 날(공휴일인 날)에만 '추석 ÷ 보통 일요일' 배율(서울 일별, 2025 제외)을 기본 예측에 곱하기 (검증 B)."""
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
HERE = Path(__file__).resolve().parent
exec((HERE / "a5b_chuseok_feature.py").read_text(encoding="utf-8").split("rows = []")[0])
sd = pd.read_csv(EXT / "seoul_daily_2015_2026h1.csv", parse_dates=["date"])
c15 = pd.read_csv(EXT / "calendar_2015_2026.csv", parse_dates=["date"])
hol = set(c15[c15.is_holiday == 1].date)
core = c15[c15.holiday_name.fillna("").str.contains("추석") & (c15.is_holiday == 1)][["date", "holiday_name"]]
core["kind"] = np.where(core.holiday_name.str.contains("대체"), "대체", np.where(core.holiday_name.str.contains("전날"), "전날", np.where(core.holiday_name.str.contains("다음날"), "다음날", "당일")))
rows = []
for (role, nm), g in sd.groupby(["role", "station_name"]):
    s = g.set_index("date").boardings.astype(float)
    for r in core.itertuples():
        if r.date.year in (2020, 2021, 2025) or r.date not in s.index:
            continue
        sun = r.date + pd.Timedelta(days=(6 - r.date.day_of_week))
        ref = [s[x] for x in [sun + pd.Timedelta(weeks=w) for w in (-5, -4, -3, -2, 2, 3, 4, 5)] if x in s.index and x not in hol]
        if len(ref) >= 4:
            rows.append((r.kind, s[r.date] / np.median(ref)))
fac = pd.DataFrame(rows, columns=["kind", "r"]).groupby("kind").r.median()
print("추석 ÷ 일요일 배율 (2025 제외):", fac.round(3).to_dict())
tr = t[(t.date < "2025-07-01") | (t.date > "2025-10-31")]
va = t[t.date.between("2025-07-01", "2025-10-31")].copy()
va = va[va.station_id.isin(tr.station_id.unique())]
b = lgb.train(P, lgb.Dataset(tr[G_BASE], tr.y, categorical_feature=["sid"]), ROUNDS)
p = np.clip(b.predict(va[G_BASE]), 0, None)
k = va.date.map(core.set_index("date").kind)
adj = np.log(k.map(fac).fillna(1.0).values)
y = va.y.values
corem = k.notna().values
wide = va.date.between("2025-10-01", "2025-10-12").values
out = []
for nm, q in [("기본 모델", p), ("추석 연휴 날만 × 배율", p + adj), ("추석 연휴 날만 × 배율의 절반(로그)", p + adj / 2)]:
    out.append({"방법": nm, "7~10월 전체": round(float(np.sqrt(np.mean((y - q) ** 2))), 5),
                "추석 연휴 날": round(float(np.sqrt(np.mean((y[corem] - q[corem]) ** 2))), 4),
                "10/1~10/12": round(float(np.sqrt(np.mean((y[wide] - q[wide]) ** 2))), 4)})
res = pd.DataFrame(out); print(res.to_string(index=False)); res.to_csv(OUT / "a5d_chuseok_vs_sunday.csv", index=False)
fac.round(3).to_csv(OUT / "a5d_chuseok_vs_sunday_factor.csv")
