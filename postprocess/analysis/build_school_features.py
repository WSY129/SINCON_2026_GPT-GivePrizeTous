"""역 × 날짜 학교 변수 표 (모델 입력용). 학교 앞 8개 역, 2025-01-01 ~ 2026-10-31 (train + test).

왜 8개 역만: train 71개 역에서 '시험 날 신호'(12~13시 ÷ 16~17시 급증, exam_signal_station_day.csv)가
뚜렷한 역이 이 8개뿐이다. 나머지 역은 학교 일정이 승차 모양을 거의 바꾸지 않는다 (README 6-3절 (21) · (22)).
입력: data/external/school_schedule_web.csv (학교 홈페이지 '월간 일정'), schools_near_exam_stations.csv (역 700 m 안 학교)
행사 분류 (행사 이름 기준)
  exam      시험(고사·시험·지필) — 모의고사·수능·학력평가·자격·직업기초 제외
  vacation  방학 (방학식 날 제외)
  closed    재량휴업일·개교기념일 (토요휴업일 제외)
  ceremony  방학식·종업식·졸업식·개학식·입학식 (보통 일찍 끝남)
  trip      수학여행·체험학습·수련활동 (일부 학년이 학교에 안 옴)
열
  hs_exam_main  역의 '주 고등학교'(아래 MAIN) 시험 날 0/1 ← 가장 강한 변수 (검바위 12시 ×4.3)
  hs_exam_n / ms_exam_n  700 m 안 고등학교 / 중학교 중 시험인 학교 수
  vacation_share  일정을 받은 학교 중 방학인 비율 (평일만 의미 있음)
  closed_n, ceremony_n, trip_n  그 행사인 학교 수
  n_schools  일정을 받은 학교 수
사용: .venv/bin/python -I analysis/build_school_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external"
MAIN = {  # 시험 날 효과가 train에서 확인된 학교 (adjust_exam_days.py와 같음)
    3208: ["서인천고등학교"], 3122: ["제일고등학교"], 3224: ["문일여자고등학교"], 3209: ["인천디자인고등학교"],
    3204: ["마전중학교"], 3206: ["백석고등학교", "한국주얼리고등학교"], 3215: ["가림고등학교", "가좌고등학교"], 3216: [],
}
CATS = {
    "exam": (r"고사|시험|지필", r"모의|수능|수학능력|학력|자격|검정|직업기초"),
    "vacation": (r"방학", r"방학식"),
    "closed": (r"재량휴업|개교기념", r"토요휴업"),
    "ceremony": (r"방학식|종업식|졸업식|개학|입학식", r"^$"),
    "trip": (r"수학여행|체험학습|현장체험|수련", r"^$"),
}

w = pd.read_csv(EXT / "school_schedule_web.csv", parse_dates=["date"])
sc = pd.read_csv(EXT / "schools_near_exam_stations.csv")
level = {s: ("고" if "고등학교" in s else "중") for s in sc.school}
for c, (inc, exc) in CATS.items():
    w[c] = w.event.str.contains(inc, na=False) & ~w.event.str.contains(exc, na=False)
flags = w.groupby(["school", "date"])[list(CATS)].any().reset_index()

# 방학 날을 하나하나 적지 않은 학교가 많다 → 방학식·종업식 다음 날부터 다음 개학 전날까지를 방학으로 채움 (75일 이내)
fill = []
for s, g in w.groupby("school"):
    starts = sorted(set(g.date[g.event.str.contains(r"방학식|종업식", na=False)]))
    ends = sorted(set(g.date[g.event.str.contains(r"개학", na=False) & ~g.event.str.contains(r"3학년", na=False)]))
    for a in starts:
        nxt = [e for e in ends if 0 < (e - a).days <= 75]
        if nxt:
            fill += [(s, d) for d in pd.date_range(a + pd.Timedelta(days=1), nxt[0] - pd.Timedelta(days=1))]
fill = pd.DataFrame(fill, columns=["school", "date"]).assign(vacation_fill=True).drop_duplicates()
flags = flags.merge(fill, on=["school", "date"], how="outer")
flags[list(CATS)] = flags[list(CATS)].fillna(False).astype(bool)
flags["vacation"] = flags.vacation | flags.vacation_fill.fillna(False).astype(bool)

dates = pd.date_range("2025-01-01", "2026-10-31")
rows = []
for sid in sorted(sc.station_id.unique()):
    schools = sorted(set(sc[sc.station_id == sid].school) & set(w.school))
    f = flags[flags.school.isin(schools)]
    for d in dates:
        g = f[f.date == d]
        hs = g[g.school.map(level) == "고"]
        ms = g[g.school.map(level) == "중"]
        rows.append(dict(
            date=d.strftime("%Y-%m-%d"), station_id=sid,
            hs_exam_main=int(bool(g[g.school.isin(MAIN.get(sid, [])) & g.exam].shape[0])),
            hs_exam_n=int(hs.exam.sum()), ms_exam_n=int(ms.exam.sum()),
            vacation_share=round(g.vacation.sum() / max(len(schools), 1), 3),
            closed_n=int(g.closed.sum()), ceremony_n=int(g.ceremony.sum()), trip_n=int(g.trip.sum()),
            n_schools=len(schools)))
out = pd.DataFrame(rows)
out.to_csv(EXT / "school_features_station_day.csv", index=False)
print(out.groupby("station_id")[["hs_exam_main", "hs_exam_n", "ms_exam_n", "closed_n", "ceremony_n", "trip_n"]].sum())
print("방학 비율 > 0.5 인 날 수:", out[out.vacation_share > 0.5].groupby("station_id").size().to_dict())
print("저장:", (EXT / "school_features_station_day.csv").relative_to(ROOT), len(out), "줄")
