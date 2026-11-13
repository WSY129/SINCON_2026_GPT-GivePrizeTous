"""서울시 일별 승하차(OA-12914) 원본 → 우리 역(7호선 부천·인천 구간)과 환승 상대역만 추린 일별 표.
Output: data/external/seoul_daily_2015_2026h1.csv
  date, line, station_name, station_id(대회 역이면 번호, 환승 상대역이면 빈칸), role(target/partner), partner_of, boardings, alightings
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/seoul_daily"
meta = pd.read_csv(ROOT / "data/metaData.csv")
l7 = meta[meta.canonical_line == 7].set_index("station_name").station_id.to_dict()
# 환승 상대역: (노선에 들어가는 글자, 역명) → 대회 쪽 환승역
PARTNER = {("경인", "부평"): "부평(3120 쪽)", ("경인", "주안"): "주안(2호선)", ("공항철도", "계양"): "계양", ("공항철도", "검암"): "검암",
           ("수인", "원인재"): "원인재", ("7호선", "온수"): "(7호선 서울 경계)"}


def read(f):
    for enc in ("utf-8-sig", "cp949"):
        try:
            d = pd.read_csv(f, encoding=enc, dtype=str, index_col=False, usecols=range(5), engine="python", on_bad_lines=lambda x: x[:5])
            break
        except UnicodeDecodeError:
            continue
    d = d.iloc[:, :5]
    d.columns = ["date", "line", "station_name", "boardings", "alightings"]
    return d


df = pd.concat([read(f) for f in sorted(RAW.glob("CARD_SUBWAY_MONTH_*.csv"))], ignore_index=True)
df["date"] = pd.to_datetime(df.date.str.strip(), format="%Y%m%d")
assert df.date.max() <= pd.Timestamp("2026-06-30"), "금지 기간 자료가 섞였어요"
for c in ("boardings", "alightings"):
    df[c] = pd.to_numeric(df[c]).round().astype("Int64")
df["station_name"] = df.station_name.str.replace(r"\(.*\)", "", regex=True).str.strip()

# 7호선 부천·인천 구간은 2022-01부터 인천교통공사 운영 → 이 자료에는 2021-12-31까지만 제대로 있어요 (그 뒤는 몇 명짜리 잔여 행)
tgt = df[(df.line == "7호선") & df.station_name.isin(l7) & (df.date <= "2021-12-31")].copy()
tgt["station_id"] = tgt.station_name.map(l7).astype("Int64")
tgt["role"], tgt["partner_of"] = "target", ""
parts = []
for (ln, nm), of in PARTNER.items():
    p = df[df.line.str.contains(ln) & (df.station_name == nm)].copy()
    p["station_id"], p["role"], p["partner_of"] = pd.NA, "partner", of
    parts.append(p)
out = pd.concat([tgt] + parts).sort_values(["date", "role", "line", "station_name"])
out["date"] = out.date.dt.strftime("%Y-%m-%d")
out[["date", "line", "station_name", "station_id", "role", "partner_of", "boardings", "alightings"]].to_csv(
    ROOT / "data/external/seoul_daily_2015_2026h1.csv", index=False)
print(len(out))
print(out.groupby(["role", "line", "station_name"]).date.agg(["min", "max", "size"]).to_string())
print("전체 노선:", sorted(df.line.unique()))
print("날짜 수:", df.date.nunique(), df.date.min().date(), df.date.max().date())
allm = pd.date_range("2015-01-01", "2026-06-30")
print("빠진 날:", [d.strftime("%Y-%m-%d") for d in allm.difference(pd.DatetimeIndex(df.date.unique()))][:40])
