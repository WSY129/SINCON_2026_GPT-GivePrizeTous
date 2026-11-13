"""부천아트센터(BAC, 2023-05 개관, 부천시 원미구 소향로 165 — 역: 부천시청 3757) 연간 공연 일정.
누리집 '연간일정'이 쓰는 방식 그대로: 필터(JSON) → /api/historyBack/create 로 키 받기 → /product/performance/year/search?q=키
키 필요 없음. 공연·전시·교육 중 '공연'(Category 1469)만 받아요.
Output: data/external/bac_bucheon_performances.csv (start_date, end_date, title, hall, production_type, genre_id, url)
"""
import json, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/bac"
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://www.bac.or.kr"


def curl(*args):
    return subprocess.run(["curl", "-s", "-m", "60", "-A", "Mozilla/5.0", *args], capture_output=True, text=True).stdout


rows = []
for y in range(2023, 2027):
    f = RAW / f"year_{y}.json"
    if not f.exists():
        flt = {"Year": y, "Categories": None, "Category": 1469, "ProductionTypeID": None, "TheaterID": None, "SearchText": None}
        key = json.loads(curl("-X", "POST", BASE + "/api/historyBack/create", "--data-urlencode", "value=" + json.dumps(flt)))["Tag"]
        f.write_text(curl(BASE + "/product/performance/year/search?q=" + key))
    d = json.loads(f.read_text())
    for mo in d.get("Schedule") or []:
        for it in mo.get("Days") or []:
            rows.append({"start_date": pd.to_datetime(it["PlayBeginDate"]).strftime("%Y-%m-%d"),
                         "end_date": pd.to_datetime(it["PlayEndDate"]).strftime("%Y-%m-%d"),
                         "title": it["Title"], "hall": it["Place"], "production_type": it.get("ProductionTypeName"),
                         "genre_id": it.get("GenreID"), "url": it.get("DetailsUrl")})
df = pd.DataFrame(rows).drop_duplicates(["start_date", "end_date", "title", "hall"]).sort_values("start_date")
df.to_csv(ROOT / "data/external/bac_bucheon_performances.csv", index=False)
print(len(df), df.start_date.min(), df.start_date.max())
print(df.groupby(df.start_date.str[:4]).size().to_dict(), df.hall.value_counts().to_dict())
