"""시간별 미세먼지(PM10, PM2.5) — 인천·부천.
Source: Open-Meteo Air Quality API (CAMS 유럽 대기 모델 재분석·예보, 키 필요 없음). 관측소 실측(에어코리아)이 아니라 모델 값이에요.
에어코리아 실측은 data.go.kr 인증키가 필요해요.
Output: data/external/airquality_hourly.csv (time, area, pm10, pm2_5, date, hour)
"""
import json, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw"
frames = []
for name, lat, lon in [("incheon", 37.4563, 126.7052), ("bucheon", 37.5034, 126.7660)]:
    f = RAW / f"airquality_{name}.json"
    if not f.exists():
        url = (f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
               f"&hourly=pm10,pm2_5&start_date=2015-01-01&end_date=2026-10-31&timezone=Asia%2FSeoul")
        f.write_text(subprocess.run(["curl", "-s", "-m", "300", url], capture_output=True, text=True).stdout)
    h = json.loads(f.read_text())["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df.time)
    df["area"] = name
    frames.append(df)
a = pd.concat(frames).dropna(subset=["pm10", "pm2_5"], how="all")
a["date"] = a.time.dt.strftime("%Y-%m-%d")
a["hour"] = a.time.dt.hour
a.to_csv(ROOT / "data/external/airquality_hourly.csv", index=False)
print(a.groupby("area").time.agg(["min", "max", "count"]))
