"""시간별 날씨 → 기상특보 '근사' 플래그 (날짜별).
기상청 특보 원자료(data.kma.go.kr)는 로그인해야 받을 수 있어서, 특보 기준을 시간별 관측값에 그대로 적용해 근사해요.
- 호우주의보 근사: 3시간 누적 강수 ≥ 60mm 또는 12시간 누적 ≥ 110mm
- 호우경보 근사:   3시간 누적 ≥ 90mm 또는 12시간 누적 ≥ 180mm
- 폭염주의보 근사: 일 최고 체감온도 ≥ 33℃ 가 2일 이상 연속
- 폭염경보 근사:   일 최고 체감온도 ≥ 35℃ 가 2일 이상 연속
- 강풍 근사:       풍속 ≥ 14 m/s (10m 풍속, km/h 단위면 /3.6)
입력: data/external/weather_hourly_2015_2026.csv (날씨 담당 파일, 열: time, area, precipitation, apparent_temperature, wind_speed_10m)
출력: data/external/weather_alert_proxy_daily.csv
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/external/weather_hourly_2015_2026.csv"
w = pd.read_csv(src, parse_dates=["time"]).sort_values(["area", "time"])
out = []
for area, g in w.groupby("area"):
    g = g.set_index("time")
    p3 = g.precipitation.rolling("3h").sum()
    p12 = g.precipitation.rolling("12h").sum()
    wind = g.wind_speed_10m / 3.6  # Open-Meteo 기본 단위 km/h
    h = pd.DataFrame({"rain_watch": (p3 >= 60) | (p12 >= 110), "rain_warn": (p3 >= 90) | (p12 >= 180),
                      "wind": wind >= 14, "at": g.apparent_temperature})
    d = h.resample("D").agg({"rain_watch": "max", "rain_warn": "max", "wind": "max", "at": "max"})
    hot33, hot35 = d["at"] >= 33, d["at"] >= 35
    d["heat_watch"] = hot33 & (hot33.shift(1, fill_value=False) | hot33.shift(-1, fill_value=False))
    d["heat_warn"] = hot35 & (hot35.shift(1, fill_value=False) | hot35.shift(-1, fill_value=False))
    d = d.rename(columns={"at": "max_apparent_temp"}).reset_index().rename(columns={"time": "date"})
    d["area"] = area
    out.append(d)
res = pd.concat(out)
res["date"] = res["date"].dt.strftime("%Y-%m-%d")
res.to_csv(ROOT / "data/external/weather_alert_proxy_daily.csv", index=False)
print(res.groupby("area")[["rain_watch", "rain_warn", "heat_watch", "heat_warn", "wind"]].sum())
