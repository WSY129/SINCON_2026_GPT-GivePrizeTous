"""Hourly weather 2015-01-01..2026-10-31 for Incheon (37.4563,126.7052) and Bucheon (37.5034,126.7660).
Source: Open-Meteo Historical Weather API (ERA5 reanalysis, archive-api.open-meteo.com). No key needed."""
import json, subprocess
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw"
VARS = "temperature_2m,apparent_temperature,precipitation,rain,snowfall,relative_humidity_2m,wind_speed_10m"
frames = []
for name, lat, lon in [("incheon", 37.4563, 126.7052), ("bucheon", 37.5034, 126.7660)]:
    f = RAW / f"weather_{name}.json"
    if not f.exists():
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date=2015-01-01&end_date=2026-10-31&hourly={VARS}&timezone=Asia%2FSeoul")
        f.write_text(subprocess.run(["curl", "-s", "-m", "180", url], capture_output=True, text=True).stdout)
    h = json.loads(f.read_text())["hourly"]
    df = pd.DataFrame(h); df["time"] = pd.to_datetime(df.time); df["area"] = name
    frames.append(df)
w = pd.concat(frames)
w["date"] = w.time.dt.date; w["hour"] = w.time.dt.hour
w.to_csv(ROOT / "data/external/weather_hourly_2015_2026.csv", index=False)
print(w.groupby("area").time.agg(["min", "max", "count"]))
print(w.isna().mean().round(3).to_dict())
