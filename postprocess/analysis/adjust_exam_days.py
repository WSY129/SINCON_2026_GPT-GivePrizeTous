"""학교 시험 날 학교 앞 역의 시간대 모양을 바꾼다 (12~13시 ↑, 16~17시 ↓). 절반만 반영.

근거
  시험 날은 하교가 12~13시로 당겨진다. train에서 검바위는 시험 날 12시 승차가 평소의 약 4.3배, 16시 0.46배.
  시험 날짜는 학교 홈페이지 '월간 일정' (analysis/fetch_school_schedule_web.py → school_schedule_web.csv).
  서인천고(검바위 143 m) 일정이 train 급증일과 2025-04 · 07 · 10 · 12, 2026-04 모두 하루도 빠짐없이 맞음.
  기존 제출은 시험 날과 같은 주 다른 날의 모양이 거의 같다 (= 모델이 시험 날을 모름, 아래 '기존 차이' 열).
방법 (역 s, 시간 h)
  m[s,h] = train 시험 날 ÷ 평소(같은 요일 앞뒤 3주, 공휴일·시험 날 제외 중앙값) 의 로그 중앙값
           |m| < 0.05 는 0, −1.0 ~ 1.5 로 제한
  새 값 = 기존 × exp(α · m[s,h]),  α = 0.5
  검증: 시험 기간 하나씩 빼고 나머지로 m 을 구해 맞혀 봄 (leave-one-period-out). 시험 날 하루(18칸) 제곱오차 합
        α 0 → 0.5:  검바위 5.12 → 2.32, 부평삼거리 1.77 → 1.48, 만수 0.75 → 0.54, 아시아드 0.78 → 0.55,
                    마전 0.74 → 0.57, 독정 1.10 → 0.85, 인천가좌 0.85 → 0.68  (α 0.75 도 비슷하게 좋음, 부평삼거리만 0.5가 나음)
제외: 가재울(신호 약함), 공휴일·주말, 직업기초능력평가·모의고사 같은 시험
사용: .venv/bin/python -I analysis/adjust_exam_days.py [기반 파일] [α] [역:α,...]   예) ... 0.5 3208:0.75
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/v20e_0926v10/submission_v20e_0926_2019a50_jul2_0923d0927m_a50_ssg_0719_bfc_0711_octfest_night.csv"
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
# 역별 α 덮어쓰기 (예: "3208:0.75" — 검증에서 검바위는 α 0.75가 더 좋았음 2.32 → 1.72)
ALPHA_ST = {int(k): float(v) for k, v in (p.split(":") for p in sys.argv[3].split(","))} if len(sys.argv) > 3 else {}
suffix = "_exam" + ("" if ALPHA == 0.5 else str(int(ALPHA * 100))) + "".join(f"_{k}a{int(v * 100)}" for k, v in ALPHA_ST.items())
OUT = BASE.with_name(BASE.stem + suffix + ".csv")
SCHOOLS = {  # 역: 근처 학교 (거리 m). 중학교만 시험인 날은 train에서 효과가 거의 없어서 고등학교만 (마전은 바로 앞 마전중)
    3208: ["서인천고등학교"],                       # 검바위 143 (대인고·간재울중만 시험인 날은 12시 +0.0 ~ 0.3로 약함)
    3122: ["제일고등학교"],                         # 부평삼거리 136
    3224: ["문일여자고등학교"],                     # 만수 308 (만수중·만수여중만인 날은 효과 없음)
    3209: ["인천디자인고등학교"],                   # 아시아드경기장 640
    3204: ["마전중학교"],                           # 마전 173
    3206: ["백석고등학교", "한국주얼리고등학교"],     # 독정 530 · 453
    3215: ["가림고등학교", "가좌고등학교"],          # 인천가좌 556 · 511
}

cal = pd.read_csv(ROOT / "data/external/calendar_2015_2026.csv", parse_dates=["date"])
hol = set(cal.date[cal.is_holiday == 1])
w = pd.read_csv(ROOT / "data/external/school_schedule_web.csv", parse_dates=["date"])
ex = w[w.event.str.contains("고사|시험|지필", na=False)
       & ~w.event.str.contains("모의|수능|수학능력|학력|자격|검정|직업기초", na=False)]
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date"])
te = pd.read_csv(ROOT / "data/test.csv", parse_dates=["date"])
sub = pd.read_csv(BASE)
assert (sub.example_id.values == te.example_id.values).all()
te["boardings"] = sub.boardings.values

def weekdays(dates):
    return sorted(d for d in set(dates) if d.day_of_week < 5 and d not in hol)

out = te.copy()
mult_rows, day_rows = [], []
for sid, schools in SCHOOLS.items():
    E = set(ex[ex.school.isin(schools)].date)
    # 1) train 효과
    x = tr[(tr.station_id == sid) & (~tr.date.isin(hol)) & (tr.date.dt.day_of_week < 5)]
    P = x.pivot(index="date", columns="hour", values="boardings")
    L = []
    for d in [d for d in P.index if d in E]:
        nb = [q for q in P.index if q.day_of_week == d.day_of_week and 0 < abs((q - d).days) <= 21 and q not in E]
        if len(nb) >= 2:
            L.append(np.log((P.loc[d] + 1) / (P.loc[nb].median() + 1)))
    m = pd.DataFrame(L).median()
    m = m.where(m.abs() >= 0.05, 0.0).clip(-1.0, 1.5)
    mult_rows.append(dict(station_id=sid, n_train_exam_days=len(L), **{f"h{h}": round(v, 3) for h, v in m.items()}))
    # 2) test 시험 날
    days = weekdays(d for d in E if pd.Timestamp("2026-07-01") <= d <= pd.Timestamp("2026-10-31"))
    S = out[out.station_id == sid].pivot(index="date", columns="hour", values="boardings")
    for d in days:
        # 기존 예측이 시험 날을 알았는지: 시험 날 ÷ 같은 주(±7일) 다른 평일 중 같은 요일 아닌 날도 포함한 중앙값, 12~13시 vs 16~17시
        nb = [q for q in S.index if 0 < abs((q - d).days) <= 7 and q.day_of_week < 5 and q not in hol and q not in E]
        before_shape = np.log(S.loc[d, [12, 13]].sum() / S.loc[d, [16, 17]].sum()) - \
            np.median([np.log(S.loc[q, [12, 13]].sum() / S.loc[q, [16, 17]].sum()) for q in nb]) if nb else np.nan
        idx = out.index[(out.station_id == sid) & (out.date == d)]
        f = np.exp(ALPHA_ST.get(sid, ALPHA) * m.reindex(out.loc[idx, "hour"]).values)
        before = out.loc[idx, "boardings"].sum()
        out.loc[idx, "boardings"] = out.loc[idx, "boardings"].values * f
        sch = ", ".join(sorted(set(ex[(ex.date == d) & ex.school.isin(schools)].school)))
        day_rows.append(dict(station_id=sid, date=d.date(), schools=sch, 기존차이_12_13vs16_17=round(before_shape, 2),
                             h12_change=round(f[list(out.loc[idx, "hour"]).index(12)], 2),
                             h16_change=round(f[list(out.loc[idx, "hour"]).index(16)], 2),
                             day_change=round(out.loc[idx, "boardings"].sum() / before, 3)))

pd.set_option("display.width", 250)
mult = pd.DataFrame(mult_rows)
days = pd.DataFrame(day_rows)
print(mult.to_string(index=False))
print(days.to_string(index=False))
if ALPHA == 0.5 and not ALPHA_ST:  # 기본판만 표를 남김
    mult.to_csv(ROOT / "analysis/exam_day_multipliers.csv", index=False)
    days.to_csv(ROOT / "analysis/exam_days_test_2026.csv", index=False)

res = out[["example_id", "boardings"]]
assert len(res) == len(te) and (res.example_id.values == te.example_id.values).all()
assert res.boardings.notna().all() and np.isfinite(res.boardings).all() and (res.boardings >= 0).all()
res.to_csv(OUT, index=False)
dd = np.log1p(res.boardings.values) - np.log1p(sub.boardings.values)
print(f"바뀐 칸 {(dd != 0).sum():,}, 두 제출 거리(RMSLE) {np.sqrt((dd ** 2).mean()):.5f}")
print("저장:", OUT.relative_to(ROOT))
