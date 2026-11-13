"""부천FC1995 (K26) home games 2025-2026 with attendance. Source: kleague.com schedule endpoint."""
import json, time, subprocess
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/kleague"; RAW.mkdir(parents=True, exist_ok=True)
rows = []
for year in range(2016, 2027):
    for lg in (1, 2):
        for m in range(1, 13):
            f = RAW / f"{year}_{lg}_{m:02d}.json"
            if not f.exists():
                body = json.dumps({"leagueId": str(lg), "year": str(year), "month": f"{m:02d}", "teamId": "K26", "ticketYn": ""})
                out = subprocess.run(["curl", "-s", "-m", "30", "-X", "POST", "https://www.kleague.com/getScheduleList.do",
                    "-H", "User-Agent: Mozilla/5.0", "-H", "Content-Type: application/json",
                    "-H", "Referer: https://www.kleague.com/schedule.do", "-d", body], capture_output=True, text=True).stdout
                f.write_text(out if out.strip().startswith("{") else "{}"); time.sleep(0.8)
            for g in (json.loads(f.read_text()).get("data") or {}).get("scheduleList") or []:
                if "K26" in (g["homeTeam"], g["awayTeam"]):
                    rows.append(g)
df = pd.DataFrame(rows).drop_duplicates(["year", "leagueId", "gameId"])
df["date"] = pd.to_datetime(df.gameDate, format="%Y.%m.%d")
home = df[df.homeTeam == "K26"]
out = pd.DataFrame(dict(date=home.date.dt.date, weekday=home.weekdayShort, kickoff=home.gameTime,
    competition=home.meetName, home=home.homeTeamName, away=home.awayTeamName, stadium=home.fieldNameFull,
    attendance=home.audienceQty, status=home.gameStatus, source="kleague.com")).sort_values("date")
out.to_csv(ROOT / "data/external/games_bucheonfc.csv", index=False)
print(len(out)); print(out.stadium.value_counts().to_string()); print(out.competition.value_counts().to_string())
print(out[pd.to_datetime(out.date) >= "2026-06-15"].to_string())
