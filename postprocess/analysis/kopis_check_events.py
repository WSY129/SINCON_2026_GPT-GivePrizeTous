"""KOPIS 공연장 × 일별 티켓 수(kopis_venue_daily.csv)를 기존 행사 목록·급증일·train 비율과 맞춰 본다.

1) 큰 공연(하루 티켓 ≥ BIG)이 기존 행사 목록(11·29번, 배구 10번, 컨벤시아 32번)에 있는지 → 없으면 '새로 찾음'
2) 급증일(37번) 중 원인 '미확인'인 날에 근처 공연장 공연이 있었는지
3) train 기간: 공연 날 그 역의 하루·밤 비율(13번) vs 티켓 수
4) test 기간(2026-07 ~ 10) 큰 공연 목록

출력: analysis/kopis_big_days.csv, analysis/kopis_unexplained_spike_match.csv, analysis/kopis_ticket_vs_ratio.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external"
BIG = 1000

kv = pd.read_csv(EXT / "kopis_venue_daily.csv", dtype={"station_ids": str})
kv["date"] = pd.to_datetime(kv.date)


def expand(df, s="start_date", e="end_date"):
    rows = []
    for _, r in df.iterrows():
        for d in pd.date_range(r[s], r[e] if pd.notna(r[e]) else r[s]):
            rows.append({**r.to_dict(), "date": d})
    return pd.DataFrame(rows)


# --- 기존 목록을 '공연장 이름 키워드 × 날짜'로
known = []
ev = pd.concat([pd.read_csv(EXT / "events_venues.csv"), pd.read_csv(EXT / "events_venues_added.csv")], ignore_index=True)
for _, r in expand(ev).iterrows():
    known.append((r.date, str(r.venue), str(r.event)))
kovo = pd.read_csv(EXT / "games_kovo_incheon.csv")
for _, r in kovo.iterrows():
    known.append((pd.to_datetime(r.date), str(r.place), f"V리그 {r.home}-{r.away}"))
conv = pd.read_csv(EXT / "convensia_events.csv")
for _, r in expand(conv).iterrows():
    known.append((r.date, "송도컨벤시아", f"[{r.category}] {r.event_nm}"))
known = pd.DataFrame(known, columns=["date", "k_venue", "k_event"])
known["date"] = pd.to_datetime(known.date)

KEY = {"인천문학경기장": "문학", "인천아시아드주경기장": "아시아드", "송도컨벤시아": "컨벤시아",
       "송도달빛축제공원": "달빛", "삼산월드체육관": "삼산", "부천실내체육관": "부천", "인천남동체육관": "남동",
       "아트센터 인천": "아트센터"}

big = kv[kv.tickets >= BIG].copy()
big["known_event"] = ""
for i, r in big.iterrows():
    k = KEY.get(r.venue)
    if not k:
        continue
    m = known[(known.date == r.date) & known.k_venue.str.contains(k)]
    big.at[i, "known_event"] = " / ".join(m.k_event.unique()[:3])
big["dow"] = big.date.dt.dayofweek
big["period"] = np.where(big.date >= "2026-07-01", "test", np.where(big.date >= "2025-01-01", "train", "past"))
big.sort_values(["venue", "date"]).to_csv(ROOT / "analysis/kopis_big_days.csv", index=False)

# --- 급증일 미확인과 맞추기
sp = pd.read_csv(ROOT / "analysis/spike_days_labeled_v2.csv", dtype={"station_id": str})
sp["date"] = pd.to_datetime(sp.date)
st_rows = kv.assign(station_id=kv.station_ids.fillna("").str.split(";")).explode("station_id")
st_rows = st_rows[st_rows.station_id != ""]
m = sp.merge(st_rows[["date", "station_id", "venue", "tickets", "shows"]], on=["date", "station_id"], how="left")
m.to_csv(ROOT / "analysis/kopis_unexplained_spike_match.csv", index=False)

# --- train 비율 vs 티켓
dr = pd.read_csv(ROOT / "analysis/daily_ratio.csv", dtype={"station_id": str})
dr["date"] = pd.to_datetime(dr.date)
tv = st_rows.merge(dr[["date", "station_id", "tot_ratio", "night_b_ratio", "is_holiday"]], on=["date", "station_id"])
tv.to_csv(ROOT / "analysis/kopis_ticket_vs_ratio.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_rows", 400)
    print("== 큰 공연(하루 티켓 ≥ %d) 공연장별 날 수 · 기존 목록에 없는 날" % BIG)
    g = big.groupby(["venue", "period"]).agg(days=("date", "size"), new=("known_event", lambda s: (s == "").sum()))
    print(g)
    print("\n== train·test 기간 큰 공연 중 기존 목록에 없는 날 (경기장·체육관·컨벤시아·달빛·아트센터인천)")
    nb = big[(big.period != "past") & (big.known_event == "") & big.venue.isin(KEY)]
    print(nb[["date", "venue", "station_ids", "shows", "tickets", "dow", "period"]].to_string(index=False))
    print("\n== 급증일 원인 별 공연장 공연이 겹친 수")
    mm = m[m.tickets.notna()]
    print(mm.groupby("cause_group").size())
    print("\n== '미확인' 급증일 중 공연장 공연이 있던 날")
    print(mm[mm.cause_group == "미확인"][["date", "station_name", "tot_ratio", "night_b_ratio", "venue", "tickets"]].to_string(index=False))
    print("\n== 공연장별: 티켓 구간별 그 역 밤 승차 비율 중앙값 (train, 공휴일 아님)")
    t = tv[tv.is_holiday == 0].copy()
    t["bin"] = pd.cut(t.tickets, [0, 300, 1000, 3000, 10000, 1e6], labels=["<300", "300-1k", "1k-3k", "3k-10k", "10k+"])
    print(t.groupby(["venue", "station_id", "bin"], observed=True).agg(n=("date", "size"), tot=("tot_ratio", "median"), night=("night_b_ratio", "median")).round(2))
