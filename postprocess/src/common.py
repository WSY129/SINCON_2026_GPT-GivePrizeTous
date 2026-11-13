"""Shared loading, calendar and metric helpers for SINCON 2026."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# Korean public holidays incl. substitute/temporary/election days.
# Sources (checked 2026-11-11): 관공서의 공휴일에 관한 규정, 인사혁신처 발표,
# news reports on 2026 restoration of 제헌절(7/17) and 노동절(5/1).
HOLIDAYS = {
    # 2025
    "2025-01-01": "신정",
    "2025-01-27": "임시공휴일",
    "2025-01-28": "설연휴", "2025-01-29": "설날", "2025-01-30": "설연휴",
    "2025-03-03": "삼일절 대체",
    "2025-05-05": "어린이날·부처님오신날", "2025-05-06": "대체공휴일",
    "2025-06-03": "대통령선거",
    "2025-06-06": "현충일",
    "2025-08-15": "광복절",
    "2025-10-03": "개천절",
    "2025-10-06": "추석", "2025-10-07": "추석연휴", "2025-10-08": "추석 대체",
    "2025-10-09": "한글날",
    "2025-12-25": "성탄절",
    # 2026
    "2026-01-01": "신정",
    "2026-02-16": "설연휴", "2026-02-17": "설날", "2026-02-18": "설연휴",
    "2026-03-02": "삼일절 대체",
    "2026-05-01": "노동절",
    "2026-05-05": "어린이날",
    "2026-05-25": "부처님오신날 대체",
    "2026-06-03": "지방선거",
    "2026-07-17": "제헌절",
    "2026-08-17": "광복절 대체",
    "2026-09-24": "추석연휴", "2026-09-25": "추석", "2026-09-26": "추석연휴",
    "2026-10-05": "개천절 대체",
    "2026-10-09": "한글날",
    "2026-12-25": "성탄절",
}
# Days that are not official holidays but behave partly like one.
SEMI_HOLIDAYS = {
    "2025-05-01": "근로자의날",
}
LUNAR_HOLIDAY_NAMES = ("설", "추석")


def load():
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    meta = pd.read_csv(DATA_DIR / "metaData.csv")
    return train, test, meta


def calendar(start="2024-12-01", end="2026-11-30"):
    """One row per date with calendar features."""
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    ds = d["date"].dt.strftime("%Y-%m-%d")
    d["dow"] = d["date"].dt.day_of_week
    d["month"] = d["date"].dt.month
    d["doy"] = d["date"].dt.day_of_year
    d["is_holiday"] = ds.isin(HOLIDAYS).astype(int)
    d["is_semi_holiday"] = ds.isin(SEMI_HOLIDAYS).astype(int)
    d["is_lunar_holiday"] = ds.map(
        lambda s: int(any(k in HOLIDAYS.get(s, "") for k in LUNAR_HOLIDAY_NAMES))
    )
    off = (d["dow"] >= 5) | (d["is_holiday"] == 1)
    d["is_off"] = off.astype(int)
    # day type: 0 weekday, 1 saturday, 2 sunday/holiday
    d["daytype"] = np.where(d["is_holiday"] == 1, 2, np.where(d["dow"] == 6, 2, np.where(d["dow"] == 5, 1, 0)))
    d["prev_off"] = off.shift(1, fill_value=False).astype(int)
    d["next_off"] = off.shift(-1, fill_value=False).astype(int)
    # weekday sandwiched between off days ("징검다리")
    d["bridge"] = ((~off) & d["prev_off"].astype(bool) & d["next_off"].astype(bool)).astype(int)
    # length of the off-block this day belongs to (0 for working days)
    block = (off != off.shift()).cumsum()
    d["off_block_len"] = np.where(off, off.groupby(block).transform("sum"), 0)
    return d


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def write_submission(test, pred, name):
    OUT_DIR.mkdir(exist_ok=True)
    sub = pd.DataFrame({"example_id": test["example_id"], "boardings": np.clip(pred, 0, None)})
    assert len(sub) == 157_194 and sub["example_id"].is_unique
    assert np.isfinite(sub["boardings"]).all()
    path = OUT_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    return path
