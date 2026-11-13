"""인천공항 월별 여객 2015-01 ~ 최신 (국제+국내, 출발+도착, 유임+환승).
Source: 항공정보포털(airportal.go.kr) 항공운송통계 > 상세통계 > 공항별 (getDetailedAirTransportStats2Time.do). 키 필요 없음.
일별 자료는 공개 화면에 없어요 (data.go.kr OpenAPI 인증키 필요).
Output: data/external/airport_icn_monthly_2015_2026.csv
"""
import json, re, subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.airportal.go.kr/stats/transport/getDetailedAirTransportStats2Time.do"
rows = []
for y in range(2015, 2027):
    body = json.dumps({"last_yearmonth": f"{y}01", "this_yearmonth": f"{y}12", "di_gubun": "total",
                       "arvl_type": "total", "airport_gubun": "total"})
    out = subprocess.run(["curl", "-s", "-m", "60", "-A", "Mozilla/5.0", "-H", "Content-Type: application/json",
                          "-X", "POST", URL, "-d", body], capture_output=True, text=True).stdout
    for it in json.loads(out)["content"]:
        m = re.match(r"(.+?)\s*(\d{4})년\s*(\d{2})월", it["airport_name"])
        if m:
            rows.append({"airport": m[1].strip(), "ym": f"{m[2]}-{m[3]}", "passengers": int(it["pass"] or 0),
                         "flights": int(it["flight_cnt"] or 0)})
d = pd.DataFrame(rows)
print(d.airport.unique()[:20])
icn = d[d.airport.str.contains("인천")].drop_duplicates(["airport", "ym"]).sort_values("ym")
icn["source"] = "https://www.airportal.go.kr/stats/transport/chartDetail2.do"
icn.to_csv(ROOT / "data/external/airport_icn_monthly_2015_2026.csv", index=False)
print(icn.tail(14).to_string())
