"""Attendance / start / end time for every game at 문학 (SSG home) 2025-2026. Source: koreabaseball.com GameCenter."""
import json, time, subprocess
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/kbo_box"; RAW.mkdir(parents=True, exist_ok=True)
URL = "https://www.koreabaseball.com/ws/Schedule.asmx/GetScoreBoardScroll"
def get(season, gid):
    f = RAW / f"{gid}.json"
    if f.exists() and f.stat().st_size > 50:
        return json.loads(f.read_text())
    for sr in (0, 1, 3, 4, 5, 7, 9, 6):
        out = subprocess.run(["curl", "-s", "-m", "30", "-X", "POST", URL, "-H", "User-Agent: Mozilla/5.0",
            "-H", "Referer: https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx", "-H", "X-Requested-With: XMLHttpRequest",
            "-H", "Content-Type: application/x-www-form-urlencoded; charset=UTF-8",
            "--data", f"leId=1&srId={sr}&seasonId={season}&gameId={gid}"], capture_output=True, text=True).stdout
        time.sleep(0.4)
        try:
            d = json.loads(out)
            if d.get("G_ID"):
                d["SR_ID_Q"] = sr; f.write_text(json.dumps(d, ensure_ascii=False)); return d
        except Exception:
            pass
    return {}
s = pd.read_csv(ROOT / "data/external/raw/kbo_SK_all.csv", parse_dates=["date"])
parts = s.extra.str.split(r" \| ", expand=True)
s["stadium"] = parts[2].str.strip(); s["note"] = parts[3].str.strip()
s = s[s.stadium == "문학"].copy()
recs = []
for _, r in s.iterrows():
    d = get(r.season, r.game_id) if isinstance(r.game_id, str) and r.note == "-" else {}
    recs.append(dict(date=r.date.date(), weekday="월화수목금토일"[r.date.day_of_week], sched_time=r.time,
        home=r.home, away=r.away, status="played" if d.get("CROWD_CN") else (r.note if r.note != "-" else "unknown"),
        start_time=d.get("START_TM"), end_time=d.get("END_TM"), duration=d.get("USE_TM"),
        attendance=int(str(d["CROWD_CN"]).replace(",", "")) if d.get("CROWD_CN") else None,
        series={0: "regular", 1: "exhibition", 3: "postseason", 4: "postseason", 5: "postseason", 7: "postseason", 9: "exhibition", 6: "exhibition"}.get(d.get("SR_ID_Q"), ""),
        game_id=r.game_id, source="koreabaseball.com GameCenter"))
out = pd.DataFrame(recs)
out.to_csv(ROOT / "data/external/games_ssg_munhak.csv", index=False)
print(out.status.value_counts().to_string()); print(out.groupby([pd.to_datetime(out.date).dt.year, "series"]).size().to_string())
print(out.tail(8).to_string())
