"""5c. 추석 배율을 '곱하는 보정'으로 쓰면 나아지나 (검증 B: 2025-07~10, 2025 추석 포함).
트리 모델은 학습 자료에 없는 추석 날짜를 배우지 못해서(5b), 이렇게 해요:
  보정 예측 = '평소 같은 요일이었다면' 예측 (공휴일 관련 변수를 보통 날 값으로 바꿔 예측) × 서울 일별 배율
배율은 2025년을 뺀 해로만 계산(정답 미리 보기 방지). 7호선 역은 line7 배율, 나머지는 환승 상대역 배율도 비교.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

HERE = Path(__file__).resolve().parent
exec((HERE / "a5b_chuseok_feature.py").read_text(encoding="utf-8").split("rows = []")[0])  # 변수·함수 정의까지만

ptB = prior_table([2020, 2021, 2025])
tB = add_feats(t, ptB)
tr = tB[(tB.date < "2025-07-01") | (tB.date > "2025-10-31")]
va = tB[tB.date.between("2025-07-01", "2025-10-31")].copy()
va = va[va.station_id.isin(tr.station_id.unique())]
b = lgb.train(P, lgb.Dataset(tr[G_BASE], tr.y, categorical_feature=["sid"]), ROUNDS)
p_base = np.clip(b.predict(va[G_BASE]), 0, None)

# '평소 같은 요일'로 바꾼 입력
cf = va.copy()
dow = cf.date.dt.day_of_week
cf["is_holiday"] = 0; cf["is_lunar_holiday"] = 0; cf["is_semi_holiday"] = 0; cf["bridge"] = 0
cf["daytype"] = np.where(dow == 6, 2, np.where(dow == 5, 1, 0))
cf["off_block_len"] = np.where(dow >= 5, 2, 0)
cf["last_day_break"] = 0; cf["first_day_after_break"] = 0
cf["prev_off"] = (dow.isin([6, 0])).astype(int)
cf["next_off"] = (dow.isin([4, 5])).astype(int)
cf["days_to_hol"] = 10
p_norm = np.clip(b.predict(cf[G_BASE]), 0, None)

win = (va.chuseok_off != 99).values
is7 = va.station_id.isin([3753, 3754, 3755, 3756, 3757, 3758, 3759, 3760, 3761, 3762, 3763]).values
variants = {"기본 모델 그대로": p_base}
for col, nm in [("chuseok_prior", "전체 중앙값"), ("chuseok_prior_partner", "환승 상대역"), ("chuseok_prior_line7", "7호선 9역")]:
    pr = va[col].values
    variants[f"평소 예측 × 배율({nm})"] = np.where(win, p_norm + np.log(pr), p_base)
pr_mix = np.where(is7, va.chuseok_prior_line7.values, va.chuseok_prior_partner.values)
variants["평소 예측 × 배율(7호선 역은 7호선, 나머지는 상대역)"] = np.where(win, p_norm + np.log(pr_mix), p_base)
variants["(기본 + 보정) 평균"] = np.where(win, (p_base + (p_norm + np.log(pr_mix))) / 2, p_base)

y = va.y.values
wide = va.date.between("2025-10-01", "2025-10-12").values
rows = []
for k, p in variants.items():
    rows.append({"방법": k, "RMSLE_7~10월 전체": round(float(np.sqrt(np.mean((y - p) ** 2))), 5),
                 "RMSLE_추석 D-4~D+4": round(float(np.sqrt(np.mean((y[win] - p[win]) ** 2))), 4),
                 "RMSLE_10/1~10/12": round(float(np.sqrt(np.mean((y[wide] - p[wide]) ** 2))), 4)})
res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(OUT / "a5c_chuseok_multiplier.csv", index=False)
best = variants["평소 예측 × 배율(7호선 역은 7호선, 나머지는 상대역)"]
day = va.assign(b=p_base, c=best).loc[win].groupby("date").apply(
    lambda g: pd.Series({"off": g.chuseok_off.iloc[0], "실제÷예측_기본": np.expm1(g.y).sum() / np.expm1(g.b).sum(),
                         "실제÷예측_보정": np.expm1(g.y).sum() / np.expm1(g.c).sum(),
                         "RMSLE_기본": np.sqrt(np.mean((g.y - g.b) ** 2)), "RMSLE_보정": np.sqrt(np.mean((g.y - g.c) ** 2))}), include_groups=False)
print(day.round(3).to_string())
day.round(4).to_csv(OUT / "a5c_chuseok_multiplier_by_day.csv")
