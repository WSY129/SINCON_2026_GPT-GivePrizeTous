"""5. 자료 묶음을 하나씩 더할 때 RMSLE가 얼마나 줄어드나 (LightGBM, log1p 승차).
검증 A: 2026-04-01~06-30 을 떼어 내고 그 전으로 학습 (시간 순서 그대로, test 와 같은 상황)
검증 B: 2025-07-01~10-31 을 떼어 내고 나머지로 학습 (여름방학·추석·가을 행사가 들어 있는 기간 — 앞뒤 자료를 다 쓰므로 낙관적)
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from _load import EXT, OUT, cal, event_days, meta, school_share, train, weather_daily

C = cal()
C["ds"] = C.date.dt.strftime("%Y-%m-%d")
off = C.is_off == 1
C["last_day_break"] = ((C.is_off == 1) & (C.next_off == 0) & (C.off_block_len >= 3)).astype(int)
C["first_day_after_break"] = ((C.is_off == 0) & (C.prev_off == 1)).astype(int)
hol_dates = C.loc[C.is_holiday == 1, "date"].values
C["days_to_hol"] = [min([((h - d) / np.timedelta64(1, "D")) for h in hol_dates if h >= d] or [99]) for d in C.date.values]
C["days_to_hol"] = C.days_to_hol.clip(upper=10)

t = train()
m = meta()
t = t.merge(C, on="date")
t["sid"] = t.station_id.astype("category")
t["trend"] = (t.date - pd.Timestamp("2025-01-01")).dt.days
t["new_age"] = np.where(t.station_id.isin([3107, 3108, 3109]), (t.date - pd.Timestamp("2025-06-28")).dt.days, -1)
t["post_0628"] = (t.date >= "2025-06-28").astype(int)
t["y"] = np.log1p(t.boardings)

# 학교 방학
t = t.merge(school_share(), on="date", how="left").fillna({"share_vacation": 0})
t["summer_peak_week"] = t.date.between("2025-07-28", "2025-08-08").astype(int)

# 행사
ev = event_days()
kinds = ["SSG경기", "SSG우천취소", "콘서트", "축제", "꽃·계절", "기타스포츠", "부천FC", "V리그", "예술회관대공연장", "부천아트센터", "컨벤시아대형"]
ev = ev[ev.kind.isin(kinds)]
for k in kinds:
    e = ev[ev.kind == k].groupby(["date", "station_id"]).agg(sh=("start_hour", "min"), sz=("size", "max")).reset_index()
    e[f"ev_{k}"] = 1
    e = e.rename(columns={"sh": f"sh_{k}", "sz": f"sz_{k}"})
    t = t.merge(e, on=["date", "station_id"], how="left")
    t[f"ev_{k}"] = t[f"ev_{k}"].fillna(0)
    t[f"rel_{k}"] = (t.hour - t[f"sh_{k}"]).fillna(-99)
    t = t.drop(columns=[f"sh_{k}"])
    if t[f"sz_{k}"].notna().sum() == 0:
        t = t.drop(columns=[f"sz_{k}"])
EVF = [c for c in t.columns if c.startswith(("ev_", "rel_", "sz_"))]
# 같은 날 다른 역 행사(파급): 그날 전체 행사 수
t["ev_any_day"] = t.groupby("date")[[f"ev_{k}" for k in ("SSG경기", "콘서트", "축제")]].transform("max").sum(axis=1)

# 날씨 (시간)
w = weather_daily()
w = w[["date", "hour", "precipitation", "apparent_temperature", "snowfall", "wind_speed_10m"]]
wd = w.groupby("date").agg(rain_day=("precipitation", "sum"), tmax_day=("apparent_temperature", "max")).reset_index()
t = t.merge(w, on=["date", "hour"], how="left").merge(wd, on="date", how="left")
t["rain_prev3h"] = t.groupby(["station_id", "date"]).precipitation.transform(lambda s: s.rolling(3, min_periods=1).sum())

# 인구
pop = pd.read_csv(EXT / "station_catchment_pop_monthly.csv")[["ym", "station_id", "pop_1km", "pop_500m"]]
t["ym"] = t.date.dt.strftime("%Y-%m")
t = t.merge(pop, on=["ym", "station_id"], how="left")
t["log_pop1k"] = np.log1p(t.pop_1km)

# 공항·미세먼지
ap = pd.read_csv(EXT / "airport_icn_monthly_2015_2026.csv")[["ym", "passengers"]].rename(columns={"passengers": "icn_pax"})
t = t.merge(ap, on="ym", how="left")
aq = pd.read_csv(EXT / "airquality_hourly.csv", parse_dates=["time"])
aq = aq[aq.area == "incheon"]; aq["date"] = aq.time.dt.normalize()
t = t.merge(aq.groupby("date").pm10.mean().rename("pm10").reset_index(), on="date", how="left")

# 정책·특이일
t["special_day"] = t.ds.isin(["2025-08-13", "2025-12-04", "2026-01-13", "2026-01-14"]).astype(int)

G = {
    "G0 기본(역·시각·요일·공휴일·추세·검단 개통)": ["sid", "hour", "dow", "daytype", "is_holiday", "month", "trend", "new_age", "post_0628"],
    "G1 달력 상세(명절·징검다리·연휴 마지막날 등)": ["is_lunar_holiday", "is_semi_holiday", "bridge", "off_block_len", "last_day_break", "first_day_after_break", "prev_off", "next_off", "days_to_hol"],
    "G2 학교 방학·휴가 주간": ["share_vacation", "summer_peak_week"],
    "G3 행사·경기·공연": EVF + ["ev_any_day"],
    "G4 날씨(시간·일)": ["precipitation", "apparent_temperature", "snowfall", "wind_speed_10m", "rain_day", "tmax_day", "rain_prev3h", "special_day"],
    "G5 역세권 인구": ["log_pop1k", "pop_500m"],
    "G6 공항·미세먼지": ["icn_pax", "pm10"],
}
P = dict(objective="regression", learning_rate=0.06, num_leaves=255, min_data_in_leaf=40, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
ROUNDS = 700


def rmsle(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def fit_eval(feats, tr, va):
    d = lgb.Dataset(tr[feats], tr.y, categorical_feature=["sid"])
    b = lgb.train(P, d, ROUNDS)
    p = np.clip(b.predict(va[feats]), 0, None)
    return rmsle(va.y, p), b, p


folds = {
    "A: 2026-04~06 (앞쪽만 학습)": (t[t.date < "2026-04-01"], t[t.date >= "2026-04-01"]),
    "B: 2025-07~10 (앞뒤 학습)": (t[(t.date < "2025-07-01") | (t.date > "2025-10-31")], t[t.date.between("2025-07-01", "2025-10-31")]),
}
rows = []
for fname, (tr, va) in folds.items():
    va = va[va.station_id.isin(tr.station_id.unique())]
    feats = []
    prev = None
    for gname, gf in G.items():
        feats = feats + gf
        s, b, p = fit_eval(feats, tr, va)
        rows.append({"fold": fname, "step": "누적 +" + gname, "rmsle": s, "gain_vs_prev": None if prev is None else prev - s})
        print(fname, "누적 +", gname, round(s, 5), "" if prev is None else f"(−{prev - s:.5f})", flush=True)
        prev = s
    full = s
    va = va.assign(pred=p)
    if fname.startswith("A"):
        imp = pd.Series(b.feature_importance("gain"), index=feats).sort_values(ascending=False)
        imp.round(0).to_csv(OUT / "a5_feature_importance_gain.csv")
        for gname, gf in list(G.items())[1:]:
            fs = [f for f in feats if f not in gf]
            s, _, _ = fit_eval(fs, tr, va)
            rows.append({"fold": fname, "step": "전체 − " + gname, "rmsle": s, "gain_vs_prev": s - full})
            print(fname, "전체 −", gname, round(s, 5), f"(빼면 +{s - full:.5f})", flush=True)
    # 오차가 큰 역·시각 (전체 모델 기준)
    va["se"] = (va.y - va.pred) ** 2
    worst = va.groupby("station_id").se.mean().pow(.5).sort_values(ascending=False).head(10)
    worst = worst.rename("rmsle").to_frame().join(m.set_index("station_id")["name"])
    worst.to_csv(OUT / f"a5_worst_stations_{fname[0]}.csv")
    print(fname, "오차 큰 역:", worst.set_index("name").rmsle.round(3).to_dict())
    bh = va.groupby("hour").se.mean().pow(.5).round(3)
    print(fname, "시각별 RMSLE:", bh.to_dict(), flush=True)
pd.DataFrame(rows).to_csv(OUT / "a5_ablation.csv", index=False)
