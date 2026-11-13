"""공공데이터포털 과거판(fetch_datagokr_past.py로 받은 파일)을 정리해 train과 같은 역 번호로 만든다.

출력 (data/external/)
  ictr_daily_hourly_2019_2020.csv   역 × 일 × 시간(6~23) 승하차, 1·2호선, 2019-01-01 ~ 2020-12-31
  ictr_daily_station_2019_2026h1.csv 역 × 일 승하차 (2019·2020, 2022-01 ~ 2026-06; 2021년은 역별 자료 없음)
  ictr_line_daily_hourly_2019_2021.csv 노선 합계 × 일 × 시간, 1·2호선 2019~2021, 7호선 2021-05-22~12-31
집계 기준이 두 가지다 (2026-11-13 대조):
  F = 공사 게시판 월별·2026년 train과 같은 기준.  L = 2025년 train과 같은 기준 (역마다 0~2% 적고, 귤현 약 25% 적음)
  train 자체가 2025년은 L, 2026-01부터 F 다. 원본 파일도 판마다 기준이 다르다 → basis 열.
값 열
  boardings / alightings                     : 6~23시 기준, F(2026 train 기준)
  boardings_2025basis / alightings_2025basis : 6~23시 기준, L(2025 train 기준) ← 1번 파일 boardings 와 같은 기준
  boardings_raw / alightings_raw (일 파일)     : 원본 (하루 전체 시간)
  변환: L↔F 는 기존 ictr_to_train_factor_station_hour.csv 의 역 비율, 하루 전체→6~23시는 2026 상반기 train ÷ F 일별 파일 (역별)
2026-07 이후 날짜는 없다(있으면 멈춤).
사용: .venv/bin/python -I analysis/build_past_daily.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/external/raw/datagokr_past"
EXT = ROOT / "data/external"
META = ROOT / "data/metaData.csv"
TRAIN = ROOT / "data/train.csv"
FACTOR_H = EXT / "ictr_to_train_factor_station_hour.csv"

# 서해구청: 2026-07 행정구역 개편으로 바뀐 이름(같은 역 3210 서구청). "7호선 부평구청": 환승역 표기
ALIAS = {"가정시장": "가정중앙시장", "아시아드": "아시아드경기장", "센트럴파크재외동포청": "센트럴파크", "서해구청": "서구청"}


def base(name):
    b = re.sub(r"\(.*?\)", "", str(name)).replace(" ", "").strip()
    b = re.sub(r"^\d호선", "", b)
    return ALIAS.get(b, b)


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip().replace({"-": "0", "": np.nan}), errors="coerce")


def read(path):
    raw = path.read_bytes()
    try:
        txt = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        txt = raw.decode("cp949")
    from io import StringIO
    df = pd.read_csv(StringIO(txt), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def norm_date(s):
    s = s.astype(str).str.strip().str.replace(".", "-", regex=False)
    ok = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    return s.where(ok)


meta = pd.read_csv(META)
meta["key"] = meta["station_name"].map(base)
IDMAP = {(int(r.canonical_line), r.key): r.station_id for r in meta.itertuples()}


def sid(line, name):
    return IDMAP.get((int(line), base(name)))


# ── 1. 역 × 일 × 시간 (2019·2020, 1·2호선) ───────────────────────────────────
def load_hourly():
    out = []
    for f in sorted((RAW / "15004329").glob("*역별시간대별*.csv")):
        df = read(f)
        df["date"] = norm_date(df["통행일자"])
        df = df[df["date"].notna()]
        hcols = [c for c in df.columns if re.match(r"^\d{2}~\d{2}$", c)]
        long = df.melt(id_vars=["호선", "date", "역명", "구분"], value_vars=hcols, var_name="slot", value_name="n")
        long["hour"] = long["slot"].str[:2].astype(int) % 24
        long["n"] = num(long["n"]).fillna(0)
        long["src"] = f.name
        out.append(long)
    h = pd.concat(out)
    h["line"] = h["호선"].astype(int)
    w = h.pivot_table(index=["date", "line", "역명", "hour"], columns="구분", values="n", aggfunc="sum").reset_index()
    w = w.rename(columns={"역명": "station_name_raw", "승차": "boardings_raw", "하차": "alightings_raw"})
    w["station_id"] = [sid(l, n) for l, n in zip(w["line"], w["station_name_raw"])]
    return w


# ── 2. 역 × 일 (여러 파일) ───────────────────────────────────────────────
def load_station_daily():
    recs = []
    line_of = {"15060369": 1, "15060370": 2, "15102674": 7}
    for did, line in line_of.items():
        for f in sorted((RAW / did).glob("*.csv")):
            df = read(f)
            dcol = next(c for c in df.columns if c in ("일자", "운행일자", "통행일자"))
            if "역명" not in df.columns:  # 노선 합계 파일 (2019~2021)
                continue
            d = pd.DataFrame({"date": norm_date(df[dcol]), "line": line, "station_name_raw": df["역명"].str.strip(),
                              "boardings_raw": num(df["승차인원"]), "alightings_raw": num(df["하차인원"]), "src": f.name})
            recs.append(d)
    for f in sorted((RAW / "15004329").glob("인천교통공사_역별일별*.csv")):
        df = read(f)
        if "구분" in df.columns:  # 승차/하차가 행으로 나뉜 형식
            vcol = "합계" if "합계" in df.columns else "이용인원"
            df["v"] = num(df[vcol])
            p = df.pivot_table(index=["호선", "통행일자", "역명"], columns="구분", values="v", aggfunc="sum").reset_index()
            d = pd.DataFrame({"date": norm_date(p["통행일자"]), "line": p["호선"].astype(int), "station_name_raw": p["역명"].str.strip(),
                              "boardings_raw": p.get("승차"), "alightings_raw": p.get("하차", np.nan), "src": f.name})
        else:
            d = pd.DataFrame({"date": norm_date(df["통행일자"]), "line": df["호선"].astype(int), "station_name_raw": df["역명"].str.strip(),
                              "boardings_raw": num(df["승차인원"]), "alightings_raw": num(df["하차인원"]), "src": f.name})
        recs.append(d)
    d = pd.concat(recs, ignore_index=True)
    d = d[d["date"].notna()]
    d["station_id"] = [sid(l, n) for l, n in zip(d["line"], d["station_name_raw"])]
    return d


# ── 3. 노선 합계 × 일 × 시간 (2019~2021) ───────────────────────────────────
def load_line_hourly():
    out = []
    for did, line in {"15060369": 1, "15060370": 2, "15102674": 7}.items():
        for f in sorted((RAW / did).glob("*.csv")):
            df = read(f)
            if "역명" in df.columns:
                continue
            df["date"] = norm_date(df["일자"])
            df = df[df["date"].notna()]
            hcols = [c for c in df.columns if re.match(r"^\d{2}시$", c)]
            long = df.melt(id_vars=["date", "구분"], value_vars=hcols, var_name="slot", value_name="n")
            long["hour"] = long["slot"].str[:2].astype(int) % 24
            long["n"] = num(long["n"]).fillna(0)
            long["line"] = line
            out.append(long)
    h = pd.concat(out)
    w = h.pivot_table(index=["date", "line", "hour"], columns="구분", values="n", aggfunc="sum").reset_index()
    return w.rename(columns={"승차": "boardings_raw", "하차": "alightings_raw"})


def basis_of(src, date):
    """집계 기준: F = 공사 게시판·2026년 train과 같은 기준, L = 2025년 train과 같은 (조금 적게 센) 기준.
    2026-11-13 대조 결과 (analysis/portal_scan_20261113 참고):
      역별시간대별 2019·2020, 역별일별 20220630·20230630판, 모든 20260630판 → F
      노선별 일별 20230531·20240630·20250630판, 역별일별 20240630·20250630판 → L"""
    if "역별시간대별" in src or "20260630" in src:
        return "F"
    if "역별일별" in src and ("20220630" in src or "20230630" in src):
        return "F"
    return "L"


def main():
    train = pd.read_csv(TRAIN)
    fac = pd.read_csv(FACTOR_H)  # train2025 ÷ 공사 = L ÷ F (역 × 시간, 역 전체)
    lf = fac.groupby("station_id")[["factor_boardings_station", "factor_alightings_station"]].first()
    lf = (1 / lf).rename(columns={"factor_boardings_station": "LtoF_b", "factor_alightings_station": "LtoF_a"})

    # ── 1. 역 × 일 × 시간 (2019·2020, F 기준 = 공사 게시판과 칸마다 같음) ──
    hourly = load_hourly()
    miss = hourly[hourly["station_id"].isna()]["station_name_raw"].unique()
    assert len(miss) == 0, miss
    hourly["station_id"] = hourly["station_id"].astype(int)
    hh = hourly[hourly["hour"].between(6, 23)].merge(
        fac[["station_id", "hour", "factor_boardings", "factor_alightings"]], on=["station_id", "hour"], how="left")
    hh[["factor_boardings", "factor_alightings"]] = hh[["factor_boardings", "factor_alightings"]].fillna(1.0)
    hh["boardings"], hh["alightings"] = hh["boardings_raw"], hh["alightings_raw"]
    hh["boardings_2025basis"] = (hh["boardings_raw"] * hh["factor_boardings"]).round(1)
    hh["alightings_2025basis"] = (hh["alightings_raw"] * hh["factor_alightings"]).round(1)
    hh["basis"] = "F"
    hh = hh[["date", "line", "station_id", "station_name_raw", "hour", "boardings", "alightings",
             "boardings_2025basis", "alightings_2025basis", "basis"]].sort_values(["date", "station_id", "hour"])
    assert hh["date"].max() < "2026-07-01"
    hh.to_csv(EXT / "ictr_daily_hourly_2019_2020.csv", index=False, encoding="utf-8-sig")
    print("ictr_daily_hourly_2019_2020.csv", hh.shape)

    # ── 2. 역 × 일 ───────────────────────────────────────────────────────
    daily = load_station_daily()
    unm = daily[daily["station_id"].isna()].groupby(["line", "station_name_raw"]).size()
    assert len(unm) == 0, unm
    daily["station_id"] = daily["station_id"].astype(int)
    daily["basis"] = [basis_of(s, d) for s, d in zip(daily["src"], daily["date"])]
    # 같은 역·날짜가 여러 파일에 있으면: 승차·하차가 다 있는 것 > F 기준 > 노선별 파일
    daily["prio"] = daily["alightings_raw"].isna() * 4 + (daily["basis"] == "L") * 2 + (~daily["src"].str.contains("호선 일별")) * 1
    daily = daily.sort_values(["date", "station_id", "prio"]).drop_duplicates(["date", "station_id"], keep="first")

    # 하루 전체 → 6~23시 비율 (역별): 2026-01~06 train ÷ 같은 기간 F 일별 파일 (둘 다 F 기준)
    tr_day = train.groupby(["date", "station_id"])[["boardings", "alightings"]].sum().reset_index()
    ov = tr_day[tr_day["date"] >= "2026-01-01"].merge(daily[daily["basis"] == "F"], on=["date", "station_id"])
    share = ov.groupby("station_id").agg(b=("boardings", "sum"), br=("boardings_raw", "sum"), a=("alightings", "sum"), ar=("alightings_raw", "sum"))
    share = pd.DataFrame({"share6to23_b": share.b / share.br, "share6to23_a": share.a / share.ar})

    daily = daily.merge(share, left_on="station_id", right_index=True, how="left").merge(lf, left_on="station_id", right_index=True, how="left")
    isL = daily["basis"] == "L"
    for k, s_, f_ in (("boardings", "share6to23_b", "LtoF_b"), ("alightings", "share6to23_a", "LtoF_a")):
        v6 = daily[f"{k}_raw"] * daily[s_]
        daily[k] = np.where(isL, v6 * daily[f_], v6).round(1)
        daily[f"{k}_2025basis"] = np.where(isL, v6, v6 / daily[f_]).round(1)

    # 2019·2020: 시간대 파일이 있으니 6~23시를 정확히 더한다
    d19 = hh.groupby(["date", "line", "station_id", "station_name_raw"])[["boardings", "alightings", "boardings_2025basis", "alightings_2025basis"]].sum().reset_index()
    raw19 = hourly.groupby(["date", "station_id"])[["boardings_raw", "alightings_raw"]].sum().reset_index()
    d19 = d19.merge(raw19, on=["date", "station_id"])
    d19["basis"], d19["src"] = "F", "15004329 역별시간대별 2019·2020 (6~23시 합)"
    cols = ["date", "line", "station_id", "station_name_raw", "boardings", "alightings", "boardings_2025basis",
            "alightings_2025basis", "boardings_raw", "alightings_raw", "basis", "src"]
    allday = pd.concat([d19[cols], daily[cols]], ignore_index=True).sort_values(["date", "station_id"])
    assert allday["date"].max() < "2026-07-01"
    allday.to_csv(EXT / "ictr_daily_station_2019_2026h1.csv", index=False, encoding="utf-8-sig")
    fac_out = share.join(lf, how="outer").round(5).reset_index()
    fac_out.to_csv(EXT / "ictr_daily_basis_factor_station.csv", index=False, encoding="utf-8-sig")
    print("ictr_daily_station_2019_2026h1.csv", allday.shape)

    # ── 3. 노선 합계 × 일 × 시간 ──────────────────────────────────────────
    lh = load_line_hourly().sort_values(["date", "line", "hour"])
    assert lh["date"].max() < "2026-07-01"
    lh.to_csv(EXT / "ictr_line_daily_hourly_2019_2021.csv", index=False, encoding="utf-8-sig")
    print("ictr_line_daily_hourly_2019_2021.csv", lh.shape)


if __name__ == "__main__":
    main()
