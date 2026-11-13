"""5b. 추석 날짜별 배율(서울시 일별 자료)을 모델 변수로 넣으면 RMSLE가 얼마나 바뀌나.

변수
- chuseok_off   : 그 해 추석 당일까지 남은 날 (-4 ~ +4, 범위 밖은 99)
- chuseok_prior : 같은 날짜 위치(offset)의 과거 배율 중앙값 (평소 같은 요일 = 1, 범위 밖은 1)
                  7호선 부천·인천 9개 역(2015~2019) + 환승 상대역 6곳(2015~2024), 2020·2021 코로나 제외
- chuseok_prior_line7 / _partner : 위 두 묶음 따로
검증 B(2025-07~10)에는 2025년 서울 자료를 뺀 배율만 써요 (정답 미리 보기 방지).
test용 파일은 2025년까지 다 넣어 다시 계산해요 → data/external/chuseok_prior_daily.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "a5_ablation.py").read_text(encoding="utf-8")
exec(src.split("G = {")[0])  # a5 의 자료 붙이기 부분만 그대로 실행 → t, C, P, ROUNDS 등

P = dict(objective="regression", learning_rate=0.06, num_leaves=255, min_data_in_leaf=40, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)  # a5 와 같은 설정
ROUNDS = 700
ROOT = HERE.parents[1]
eff = pd.read_csv(ROOT / "analysis/chuseok_daily_effect_seoul.csv")
cal15 = pd.read_csv(EXT / "calendar_2015_2026.csv", parse_dates=["date"])
chuseok_days = cal15[cal15.holiday_name == "추석"].date.tolist()


def prior_table(exclude_years):
    e = eff[~eff.year.isin(exclude_years)]
    line7 = e[e.role == "target"].groupby("offset").ratio.median()
    partner = e[e.role == "partner"].groupby("offset").ratio.median()
    both = e.groupby("offset").ratio.median()
    return pd.DataFrame({"chuseok_prior": both, "chuseok_prior_line7": line7, "chuseok_prior_partner": partner})


def add_feats(df, pt):
    d = pd.DataFrame({"date": pd.to_datetime(sorted(df.date.unique()))})
    off = []
    for x in d.date:
        cand = [(x - c).days for c in chuseok_days if abs((x - c).days) <= 4]
        off.append(cand[0] if cand else 99)
    d["chuseok_off"] = off
    d = d.join(pt, on="chuseok_off")
    for c in pt.columns:
        d[c] = d[c].fillna(1.0)
    return df.drop(columns=[c for c in d.columns if c != "date" and c in df.columns]).merge(d, on="date", how="left")


G_BASE = (["sid", "hour", "dow", "daytype", "is_holiday", "month", "trend", "new_age", "post_0628"]
          + ["is_lunar_holiday", "is_semi_holiday", "bridge", "off_block_len", "last_day_break", "first_day_after_break", "prev_off", "next_off", "days_to_hol"]
          + ["share_vacation", "summer_peak_week"] + EVF + ["ev_any_day"]
          + ["precipitation", "apparent_temperature", "snowfall", "wind_speed_10m", "rain_day", "tmax_day", "rain_prev3h", "special_day"]
          + ["log_pop1k", "pop_500m"])
G_CH = ["chuseok_off", "chuseok_prior", "chuseok_prior_line7", "chuseok_prior_partner"]


def run(tr, va, feats, seed=7):
    p = dict(P, seed=seed)
    b = lgb.train(p, lgb.Dataset(tr[feats], tr.y, categorical_feature=["sid"]), ROUNDS)
    return np.clip(b.predict(va[feats]), 0, None)


def rm(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


rows = []
# 검증 B: 2025 추석 포함, 배율은 2025 제외
ptB = prior_table([2020, 2021, 2025])
tB = add_feats(t, ptB)
tr = tB[(tB.date < "2025-07-01") | (tB.date > "2025-10-31")]
va = tB[tB.date.between("2025-07-01", "2025-10-31")]
va = va[va.station_id.isin(tr.station_id.unique())]
win = va.date.between("2025-10-01", "2025-10-12")
for seed in (7, 11, 23):
    p0 = run(tr, va, G_BASE, seed)
    p1 = run(tr, va, G_BASE + G_CH, seed)
    rows.append({"fold": "B 2025-07~10", "seed": seed, "base_all": rm(va.y, p0), "with_chuseok_all": rm(va.y, p1),
                 "base_chuseok_window": rm(va.y[win], p0[win]), "with_chuseok_window": rm(va.y[win], p1[win])})
    if seed == 7:
        dd = va[win].assign(p0=p0[win], p1=p1[win]).groupby("date").apply(
            lambda g: pd.Series({"base": rm(g.y, g.p0), "with": rm(g.y, g.p1),
                                 "actual_vs_pred_base": np.expm1(g.y).sum() / np.expm1(g.p0).sum(),
                                 "actual_vs_pred_with": np.expm1(g.y).sum() / np.expm1(g.p1).sum()}), include_groups=False)
        dd = dd.join(tB.drop_duplicates("date").set_index("date")[["chuseok_off", "chuseok_prior"]])
        dd.round(4).to_csv(OUT / "a5b_chuseok_window_by_day.csv")
        print(dd.round(3).to_string(), flush=True)
    print(rows[-1], flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "a5b_chuseok_feature.csv", index=False)
print(res.mean(numeric_only=True).round(5).to_string())

# test 용 변수 파일 (2025년까지 다 넣어 계산)
ptT = prior_table([2020, 2021])
days = pd.DataFrame({"date": pd.date_range("2025-01-01", "2026-10-31")})
f = add_feats(days.assign(x=0), ptT).drop(columns="x")
f["date"] = f.date.dt.strftime("%Y-%m-%d")
f.to_csv(EXT / "chuseok_prior_daily.csv", index=False)
print(f[f.chuseok_off != 99].to_string())
