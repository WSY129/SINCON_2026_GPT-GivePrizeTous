"""역 ↔ 행정동 연결표 + 역세권 인구(월별).

- 역 좌표: OpenStreetMap(Overpass) 역 노드 평균. 서구청·지식정보단지는 OSM 역 노드가 없어 Nominatim(출구 좌표 평균).
- 행정동 경계: vuski/admdongkor (GitHub). 동이 생기고 나뉜 시점마다 경계가 달라서 **그 달에 맞는 경계 버전**을 써요.
    VERSIONS 표 참고: 동 신설·분할(검단·미추홀 2018-07, 부천 광역동 2019-07·환원 2021-06, 부천 구 부활 2024-01,
    아라1·2동 분할 2025-10, 행정체제 개편 2026-07) 때마다 경계 버전을 바꿔요.
  (원본 35MB짜리는 인천·부천만 잘라 data/external/raw/admdong/sub_*.geojson 으로 남겼어요)
- 역세권: 역 중심 반경 500m / 1km 원. 동 인구는 '동 안에 고르게 산다'고 보고 원과 겹친 면적 비율만큼 나눠요.
- 인구와는 '시군구+동 이름'으로 붙여요 (예전 경계 파일엔 10자리 코드가 없어서). 이름 표기 차이(쉼표·공백)는 맞췄어요.

Output:
  data/external/station_coords.csv
  data/external/station_dong_map.csv              역 × 행정동 (최신 경계 기준, 겹친 비율·거리)
  data/external/station_catchment_pop_monthly.csv 역 × 월 역세권 인구 (500m·1km), coverage_1km
"""
import json, math, re
from pathlib import Path
import pandas as pd
import shapely
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external"
ADM = EXT / "raw/admdong"
VERSIONS = [("2015-01", "20160201"), ("2016-07", "20170418"), ("2018-07", "20180724"), ("2018-11", "20181106"),
            ("2019-01", "20190403"), ("2019-07", "20190908"), ("2019-11", "20191231"), ("2020-10", "20201001"),
            ("2021-06", "20210701"), ("2024-01", "20240101"), ("2025-07", "20250701"), ("2025-10", "20251231"),
            ("2026-01", "20260201"), ("2026-07", "20261001")]
KX, KY = 111320 * math.cos(math.radians(37.5)), 110540  # 경위도 → 미터 (이 범위에서 오차 1% 미만)
to_m = lambda g: shapely.transform(g, lambda c: c * [KX, KY])

# ---------- 1) 역 좌표 ----------
meta = pd.read_csv(ROOT / "data/metaData.csv")
els = json.load(open(EXT / "raw/osm/stations.json"))["elements"]
norm_st = lambda s: re.sub(r"\(.*?\)|역$", "", s or "").strip()
SKIP_OP = {"한국철도공사", "공항철도주식회사", "롯데쇼핑"}
MANUAL = {3210: (37.5445, 126.6768, "nominatim(출구 평균)"), 3135: (37.3768, 126.6463, "nominatim")}
rows = []
for m in meta.itertuples():
    if m.station_id in MANUAL:
        lat, lon, src = MANUAL[m.station_id]
    else:
        c = [e for e in els if norm_st(e["tags"].get("name")) == norm_st(m.station_name)
             and e["tags"].get("operator") not in SKIP_OP]
        want = "light_rail" if m.canonical_line == 2 else "subway"  # 인천시청·석남·부평구청 등 환승역 노선 구분
        st_ = [e for e in c if e["tags"].get("station") == want] or c
        lat, lon = sum(e["lat"] for e in st_) / len(st_), sum(e["lon"] for e in st_) / len(st_)
        src = f"osm({len(st_)} nodes)"
    rows.append({"station_id": m.station_id, "station_name": m.station_name, "line": m.canonical_line,
                 "lat": round(lat, 6), "lon": round(lon, 6), "coord_source": src})
st = pd.DataFrame(rows)
st.to_csv(EXT / "station_coords.csv", index=False)


# ---------- 2) 버전별 역 × 행정동 ----------
def key(adm_nm):
    """'인천광역시 서구 도화2,3동' → ('인천광역시', '서구', '도화2.3동'); 부천은 '부천시원미구'처럼 붙여 씀"""
    p = adm_nm.split()
    return p[0], "".join(p[1:-1]).replace(" ", ""), re.sub(r"[,·]", ".", p[-1])


def station_dong(ver):
    fs = json.load(open(ADM / f"sub_{ver}.geojson"))["features"]
    out = []
    for f in fs:
        g = to_m(shape(f["geometry"]))
        sido, sgg, dong = key(f["properties"]["adm_nm"])
        for s in st.itertuples():
            pt = Point(s.lon * KX, s.lat * KY)
            d = g.distance(pt)
            if d > 1000:
                continue
            b5, b10 = pt.buffer(500, quad_segs=32), pt.buffer(1000, quad_segs=32)
            i5, i10 = g.intersection(b5).area, g.intersection(b10).area
            out.append({"station_id": s.station_id, "station_name": s.station_name,
                        "adm_code": f["properties"].get("adm_cd2", ""), "adm_nm": f["properties"]["adm_nm"],
                        "sido": sido, "sgg": sgg, "dong": dong, "is_home_dong": int(g.contains(pt)), "dist_m": round(d),
                        "share_of_500m": round(i5 / b5.area, 4), "share_of_1km": round(i10 / b10.area, 4),
                        "dong_frac_in_500m": i5 / g.area, "dong_frac_in_1km": i10 / g.area})
    return pd.DataFrame(out)


maps = {v: station_dong(v) for _, v in VERSIONS}
latest = maps[VERSIONS[-1][1]].copy()
for c in ("dong_frac_in_500m", "dong_frac_in_1km"):
    latest[c] = latest[c].round(4)
latest.drop(columns=["sido", "sgg", "dong"]).sort_values(
    ["station_id", "is_home_dong", "share_of_1km"], ascending=[True, False, False]).to_csv(EXT / "station_dong_map.csv", index=False)
print("stations:", len(st), "| home dong:", latest.is_home_dong.sum(), "| rows:", len(latest))

# ---------- 3) 월별 역세권 인구 ----------
pop = pd.read_csv(EXT / "population_dong_monthly_2015_2026.csv", dtype={"adm_code": str})
pop = pop[pop.level == "dong"].copy()
pop["sgg"] = pop.sigungu.fillna("").str.replace(" ", "")
pop["dong"] = pop.dong.str.replace(r"[,·]", ".", regex=True)
yms = sorted(pop.ym.unique())
res = []
for i, (start, ver) in enumerate(VERSIONS):
    end = VERSIONS[i + 1][0] if i + 1 < len(VERSIONS) else "9999-99"
    mm = maps[ver]
    p = pop[(pop.ym >= start) & (pop.ym < end)]
    x = mm.merge(p[["ym", "sido", "sgg", "dong", "pop"]], on=["sido", "sgg", "dong"], how="left")
    # 구 이름만 다른 경우(부천 일반구 폐지·부활 등) 같은 달 동 이름이 하나뿐이면 이름만으로 다시 붙임
    miss = x[x["pop"].isna()].drop(columns=["ym", "pop"]).drop_duplicates()
    if len(miss):
        u = p.groupby(["ym", "sido", "dong"]).filter(lambda g: len(g) == 1)
        y = miss.merge(u[["ym", "sido", "dong", "pop"]], on=["sido", "dong"])
        x = pd.concat([x.dropna(subset=["pop"]), y])
    x = x.dropna(subset=["ym"])
    x["w500"], x["w1k"] = x["pop"] * x.dong_frac_in_500m, x["pop"] * x.dong_frac_in_1km
    tot = mm.groupby("station_id").share_of_1km.sum()
    r = x.dropna(subset=["pop"]).groupby(["station_id", "ym"]).agg(
        pop_500m=("w500", "sum"), pop_1km=("w1k", "sum"), cov=("share_of_1km", "sum")).reset_index()
    r["coverage_1km"] = (r["cov"] / r.station_id.map(tot)).round(3)
    r["boundary_ver"] = ver
    res.append(r.drop(columns="cov"))
res = pd.concat(res).merge(st[["station_id", "station_name"]], on="station_id")
res[["pop_500m", "pop_1km"]] = res[["pop_500m", "pop_1km"]].round()
res = res.sort_values(["station_id", "ym"])
# 경계 버전이 바뀐 달: 이 달 전후로 값이 계단처럼 뛸 수 있어요 (실제 인구 변화가 아니라 동 경계·분할 때문)
res["boundary_change"] = (res.groupby("station_id").boundary_ver.shift() != res.boundary_ver).astype(int)
res.loc[res.ym == res.ym.min(), "boundary_change"] = 0
res = res[["ym", "station_id", "station_name", "pop_500m", "pop_1km", "coverage_1km", "boundary_ver", "boundary_change"]]
res.to_csv(EXT / "station_catchment_pop_monthly.csv", index=False)
print("rows:", len(res), "| coverage<0.95:", (res.coverage_1km < 0.95).sum())
