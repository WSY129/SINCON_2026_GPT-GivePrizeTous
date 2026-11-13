"""KOVO V리그 인천 홈경기 (2015-16 ~ 2026-27 시즌).
API: https://user-api.kovo.co.kr/games/schedules?match=001&season=NNN  (kovo.co.kr 새 사이트가 쓰는 공개 API)
인천 경기장: 계양체육관(대한항공, 흥국생명 2021-22 전), 삼산월드체육관(흥국생명 2021-22~) 등 -> city=='인천' 로 거름
"""
import json, time, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/kovo"; RAW.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Origin": "https://kovo.co.kr"}

def get(season):
    f = RAW / f"season_{season}.json"
    if f.exists():
        return json.loads(f.read_text())
    txt = subprocess.run(["curl", "-s", "-m", "40", "-A", H["User-Agent"], "-H", "Origin: " + H["Origin"],
                          f"https://user-api.kovo.co.kr/games/schedules?match=001&season={season}"],
                         capture_output=True, text=True).stdout
    f.write_text(txt); time.sleep(0.5)
    return json.loads(txt)

# 경기장 -> 가까운 역 (station_id)
# 계양체육관은 걸어갈 만한 역이 없음 (계양~작전역 경기일 효과 없음 확인) -> 빈칸
VENUE_STATION = {"계양": "", "삼산": "3759"}
rows = []
for s in range(12, 24):
    for g in get(f"{s:03d}")["payload"]:
        if g.get("city") != "인천" and "인천" not in (g.get("place") or ""):
            continue
        place = g.get("place") or ""
        st = next((v for k, v in VENUE_STATION.items() if k in place), "")
        rows.append(dict(date=g["gameDate"], start_time=(g.get("startTime") or "")[:5], end_time=(g.get("endTime") or "")[:5],
                         gender=g["gender"], home=g["homeTeam"]["name"], away=g["awayTeam"]["name"],
                         place=place, station_id=st, league=g["league"]["name"], season=g["season"]["value"],
                         attendance=g.get("audience") or g.get("spectators") or g.get("attendance")))
df = pd.DataFrame(rows).sort_values(["date", "start_time"])
out = ROOT / "data/external/games_kovo_incheon.csv"
df.to_csv(out, index=False)
print(out, len(df))
print(df.groupby([df.date.str[:4], "home", "place"]).size().unstack(["home", "place"]).fillna(0).astype(int))
