"""인천교통공사 '역 시간대별 통행량' 월별 엑셀 111개 → 긴 형식 CSV 하나.

입력: data/external/raw/ictr_hourly/{YYYY-MM}_{msg_seq}.xls(x)  (fetch_ictr_hourly.py 로 받음)
출력: data/external/ictr_hourly_monthly_2015_2024.csv
  data_month, line, station_name_raw, station_id, hour, boardings, alightings  ← train 기준으로 보정한 값
  boardings_raw, alightings_raw ← 공사 원본 값, factor_* ← 곱한 배율, quality

- 값은 '그 달 전체 합계'다 (일별 아님).
- hour 는 원본 'NN시' 의 NN. train.csv 와 같게 6~23시만 남긴다 (05시·24시이후는 버림).
- 시트가 1개인 초기 파일(2015-10 ~ 2016-12)은 1호선만 있다.
- 노선 계·공사계 행은 뺀다.
- station_id 는 data/metaData.csv 의 역명(괄호 앞부분)과 노선으로 맞춘다. 못 맞추면 빈칸.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/external/raw/ictr_hourly"
OUT = ROOT / "data/external/ictr_hourly_monthly_2015_2024.csv"
OUT_OVERLAP = ROOT / "data/external/ictr_hourly_monthly_2025_2026h1.csv"  # train 기간 검증용
META = ROOT / "data/metaData.csv"
TRAIN = ROOT / "data/train.csv"
FACTOR = ROOT / "data/external/ictr_to_train_factor_station_hour.csv"  # 공사 → train 기준 배율


ALIAS = {"가정시장": "가정중앙시장", "아시아드": "아시아드경기장", "센트럴파크재외동포청": "센트럴파크"}


def base(name):
    b = re.sub(r"\(.*?\)", "", str(name)).replace(" ", "").strip()
    return ALIAS.get(b, b)


def sheet_line(sheet_name, title):
    for s in (sheet_name, title):
        m = re.search(r"([127])\s*호선", str(s))
        if m:
            return int(m.group(1))
    return 1  # 초기 단일 시트 파일은 1호선만 있음


def parse_sheet(df, line, ym):
    # 시간대 헤더 행 찾기 ('05시' 가 있는 행)
    hdr_row = next(i for i in range(len(df)) if any(re.fullmatch(r"\s*05\s*시\s*", str(v)) for v in df.iloc[i]))
    hour_cols = {}
    for j, v in enumerate(df.iloc[hdr_row]):
        m = re.match(r"\s*(\d{1,2})\s*시", str(v))
        if m:
            hour_cols[j] = int(m.group(1))
    recs = []
    station = None
    for i in range(hdr_row + 1, len(df)):
        name, kind = df.iat[i, 0], df.iat[i, 1]
        if pd.notna(name) and str(name).strip():
            station = str(name).strip()
        kind = str(kind).strip()
        if kind not in ("승차", "하차") or station is None:
            continue
        if "계" in station.replace("계양", "").replace("계산", ""):
            continue  # 공사계, 1호선 계 등
        for j, h in hour_cols.items():
            v = df.iat[i, j]
            recs.append((ym, line, station, h, kind, 0 if pd.isna(v) else int(v)))
    return recs


def main():
    recs = []
    for f in sorted(glob.glob(str(RAW / "*.xls*"))):
        ym = Path(f).name[:7]
        xl = pd.ExcelFile(f)
        for sh in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sh, header=None)
            title = df.iat[1, 0] if len(df) > 1 else ""
            recs += parse_sheet(df, sheet_line(sh, title), ym)
    long = pd.DataFrame(recs, columns=["data_month", "line", "station_name_raw", "hour", "kind", "n"])
    wide = (
        long.pivot_table(index=["data_month", "line", "station_name_raw", "hour"], columns="kind", values="n", aggfunc="sum")
        .rename(columns={"승차": "boardings", "하차": "alightings"})
        .reset_index()
    )
    meta = pd.read_csv(META)
    meta["key"] = meta["station_name"].map(base)
    idmap = {(r.canonical_line, r.key): r.station_id for r in meta.itertuples()}
    wide["station_id"] = [idmap.get((l, base(n))) for l, n in zip(wide["line"], wide["station_name_raw"])]
    wide["station_id"] = wide["station_id"].astype("Int64")
    # 대회 데이터와 같은 6~23시만 남긴다 (05시, 24시(이후)는 버림)
    wide = wide[wide["hour"].between(6, 23)]
    wide = wide[["data_month", "line", "station_name_raw", "station_id", "hour", "boardings", "alightings"]]
    wide = wide.sort_values(["data_month", "line", "station_id", "hour"])
    # 품질 표시: 원본 파일 자체가 이상한 달
    # 2015-11: 역 행이 밀려 있음 (인천터미널 40,873 / 동춘=동막 같은 값 등) → 분석에서 빼야 함
    wide["quality"] = "ok"
    wide.loc[wide["data_month"] == "2015-11", "quality"] = "bad_source_rows_shifted"
    assert wide["data_month"].max() <= "2026-06"  # test 기간 금지
    past = wide[wide["data_month"] < "2025-01"].copy()
    over = wide[wide["data_month"] >= "2025-01"]

    # ── train 기준으로 맞추기 ──────────────────────────────────────────────
    # 팀 결정(2026-11-12): 공사 자료와 train 이 다르면 train 이 맞다고 본다.
    # 2025년 train(=공공데이터포털 15159353) 은 공사 게시판 월별 자료보다 역마다 일정 비율로 적다.
    # 그래서 과거(2015~2024) 공사 자료에 2025년 '역 × 시간대' 비율(train ÷ 공사)을 곱해 train 기준으로 바꾼다.
    # 표본이 작은 칸(2025년 공사 합계 < 2,000명)은 역 전체 비율을 쓴다.
    train = pd.read_csv(TRAIN)
    train["data_month"] = train["date"].str[:7]
    tr = train[train["data_month"] < "2026-01"].groupby(["station_id", "hour"])[["boardings", "alightings"]].sum()
    ic = over[over["data_month"] < "2026-01"].groupby(["station_id", "hour"])[["boardings", "alightings"]].sum()
    tr_s, ic_s = tr.groupby(level=0).sum(), ic.groupby(level=0).sum()
    fac = pd.DataFrame(index=ic.index)
    for k in ("boardings", "alightings"):
        cell = tr[k].reindex(ic.index) / ic[k]
        stn = (tr_s[k] / ic_s[k]).reindex(ic.index.get_level_values(0)).to_numpy()
        fac[f"factor_{k}"] = np.where(ic[k] >= 2000, cell, stn)
        fac[f"factor_{k}_station"] = stn
    fac = fac.round(5).reset_index()
    fac.to_csv(FACTOR, index=False, encoding="utf-8-sig")
    print(FACTOR, fac.shape)

    past = past.rename(columns={"boardings": "boardings_raw", "alightings": "alightings_raw"})
    past = past.merge(fac[["station_id", "hour", "factor_boardings", "factor_alightings"]], on=["station_id", "hour"], how="left")
    # 2025년에 없는 역·시간 조합은 없음(과거 역은 모두 2025년에도 있음). 혹시 없으면 1.0
    past[["factor_boardings", "factor_alightings"]] = past[["factor_boardings", "factor_alightings"]].fillna(1.0)
    past["boardings"] = (past["boardings_raw"] * past["factor_boardings"]).round(1)
    past["alightings"] = (past["alightings_raw"] * past["factor_alightings"]).round(1)
    past = past[["data_month", "line", "station_name_raw", "station_id", "hour", "boardings", "alightings",
                 "boardings_raw", "alightings_raw", "factor_boardings", "factor_alightings", "quality"]]
    past.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(OUT, past.shape)
    if len(over):
        over.to_csv(OUT_OVERLAP, index=False, encoding="utf-8-sig")
        print(OUT_OVERLAP, over.shape)


if __name__ == "__main__":
    main()
