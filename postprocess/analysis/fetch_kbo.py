"""Fetch SSG (SK) schedule/results 2025-2026 from KBO site schedule endpoint. Source: koreabaseball.com"""
import json, re, time, subprocess
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw"
URL = "https://www.koreabaseball.com/ws/Schedule.asmx/GetScheduleList"
def fetch(season, month):
    f = RAW / f"kbo_SK_{season}_{month:02d}.json"
    for attempt in range(4):
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                f.unlink(); time.sleep(3 * (attempt + 1))
        out = subprocess.run(["curl", "-s", "-m", "30", "-X", "POST", URL, "-H", "User-Agent: Mozilla/5.0",
            "-H", "Referer: https://www.koreabaseball.com/Schedule/Schedule.aspx", "-H", "X-Requested-With: XMLHttpRequest",
            "-H", "Content-Type: application/x-www-form-urlencoded; charset=UTF-8",
            "--data", f"leId=1&srIdList=0%2C1%2C9%2C6%2C3%2C4%2C5%2C7&seasonId={season}&gameMonth={month:02d}&teamId=SK"],
            capture_output=True, text=True).stdout
        f.write_text(out); time.sleep(1)
    try:
        return json.loads(f.read_text())
    except Exception:
        print("skip", season, month); f.unlink(); return {}
strip = lambda s: re.sub(r"<[^>]+>", " ", s or "").split()
rows = []
for season in range(2016, 2027):
    for m in range(3, 12):
        js = fetch(season, m)
        day = None
        for r in js.get("rows", []):
            cells = r["row"]
            c = {x["Class"]: x["Text"] for x in cells if x["Class"]}
            if "day" in c: day = c["day"]
            if "play" not in c: continue
            texts = [x["Text"] for x in cells]
            p = strip(c["play"])
            away, home = p[0], p[-1]
            score = re.findall(r">(\d+)<", c["play"])
            rows.append(dict(season=season, day=day, time=" ".join(strip(c.get("time", ""))),
                 away=away, home=home, away_score=score[0] if len(score) == 2 else None,
                 home_score=score[1] if len(score) == 2 else None,
                 game_id=(re.findall(r"gameId=(\w+)", c.get("relay", "")) or [None])[0],
                 extra=" | ".join(" ".join(strip(t)) for t in texts[-4:])))
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df.season.astype(str) + "." + df.day.str[:5], format="%Y.%m.%d")
df.to_csv(ROOT / "data/external/raw/kbo_SK_all.csv", index=False)
print(len(df)); print(df.head(3).to_string()); print(df.extra.value_counts().head(15).to_string())
