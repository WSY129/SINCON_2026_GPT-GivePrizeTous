"""South Korean known-in-advance holiday covariates, version 2026-06-30.

Only calendar facts/public declarations issued by the competition cutoff are
encoded. No ridership, weather realization or event outcome is used. Sources,
publication dates and reuse notes: reports/holiday_sources.md.

`is_holiday` means a named national holiday, substitute or election holiday;
ordinary Sundays are represented separately by `is_weekend` / `is_rest_day`.
Pass the forecast origin as `as_of` for historical CV. This hides a temporary
holiday/new designation until the cited public announcement was available.
No feature reveals that a not-yet-announced designation will occur.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

KNOWLEDGE_CUTOFF = "2026-06-30"
COVERAGE_START = "2025-01-01"
COVERAGE_END = "2026-12-31"

HOLIDAY_TYPES = {
    "none": 0, "new_year": 1, "seollal": 2, "independence": 3,
    "children": 4, "buddha": 5, "children_and_buddha": 6, "memorial": 7,
    "liberation": 8, "chuseok": 9, "national_foundation": 10,
    "hangul": 11, "christmas": 12, "temporary": 13,
    "presidential_election": 14, "local_election": 15,
    "labor": 16, "constitution": 17,
}


@dataclass(frozen=True)
class Holiday:
    name: str
    kind: str
    known_since: str
    source: str
    substitute: bool = False


def _h(name, kind, year, substitute=False):
    # Conservative availability of the final annual calendar announcement.
    known = "2024-06-20" if year == 2025 else "2025-06-30"
    return Holiday(name, kind, known, f"annual_{year}", substitute)


HOLIDAYS = {
    "2025-01-01": _h("신정", "new_year", 2025),
    "2025-01-27": Holiday("임시공휴일", "temporary", "2025-01-14", "temporary_20250127"),
    "2025-01-28": _h("설날 전날", "seollal", 2025),
    "2025-01-29": _h("설날", "seollal", 2025),
    "2025-01-30": _h("설날 다음날", "seollal", 2025),
    "2025-03-01": _h("삼일절", "independence", 2025),
    "2025-03-03": _h("삼일절 대체공휴일", "independence", 2025, True),
    "2025-05-05": _h("어린이날·부처님오신날", "children_and_buddha", 2025),
    "2025-05-06": _h("부처님오신날 대체공휴일", "buddha", 2025, True),
    "2025-06-03": Holiday("제21대 대통령선거", "presidential_election", "2025-04-08", "election_20250603"),
    "2025-06-06": _h("현충일", "memorial", 2025),
    "2025-08-15": _h("광복절", "liberation", 2025),
    "2025-10-03": _h("개천절", "national_foundation", 2025),
    "2025-10-05": _h("추석 전날", "chuseok", 2025),
    "2025-10-06": _h("추석", "chuseok", 2025),
    "2025-10-07": _h("추석 다음날", "chuseok", 2025),
    "2025-10-08": _h("추석 대체공휴일", "chuseok", 2025, True),
    "2025-10-09": _h("한글날", "hangul", 2025),
    "2025-12-25": _h("기독탄신일", "christmas", 2025),
    "2026-01-01": _h("신정", "new_year", 2026),
    "2026-02-16": _h("설날 전날", "seollal", 2026),
    "2026-02-17": _h("설날", "seollal", 2026),
    "2026-02-18": _h("설날 다음날", "seollal", 2026),
    "2026-03-01": _h("삼일절", "independence", 2026),
    "2026-03-02": _h("삼일절 대체공휴일", "independence", 2026, True),
    "2026-05-01": Holiday("노동절", "labor", "2026-04-06", "labor_2026"),
    "2026-05-05": _h("어린이날", "children", 2026),
    "2026-05-24": _h("부처님오신날", "buddha", 2026),
    "2026-05-25": _h("부처님오신날 대체공휴일", "buddha", 2026, True),
    "2026-06-03": _h("제9회 전국동시지방선거", "local_election", 2026),
    "2026-06-06": _h("현충일", "memorial", 2026),
    "2026-07-17": Holiday("제헌절", "constitution", "2026-02-03", "constitution_2026"),
    "2026-08-15": _h("광복절", "liberation", 2026),
    "2026-08-17": _h("광복절 대체공휴일", "liberation", 2026, True),
    "2026-09-24": _h("추석 전날", "chuseok", 2026),
    "2026-09-25": _h("추석", "chuseok", 2026),
    "2026-09-26": _h("추석 다음날", "chuseok", 2026),
    # 2026-09-28 is NOT a substitute holiday. Chuseok on Saturday alone does
    # not add a substitute day; the official calendar says Sep 24-27 break.
    "2026-10-03": _h("개천절", "national_foundation", 2026),
    "2026-10-05": _h("개천절 대체공휴일", "national_foundation", 2026, True),
    "2026-10-09": _h("한글날", "hangul", 2026),
    "2026-12-25": _h("기독탄신일", "christmas", 2026),
}


def holiday_records() -> pd.DataFrame:
    """All encoded facts and provenance; not a model feature table."""
    return pd.DataFrame([
        {"date": date, "name": h.name, "kind": h.kind,
         "holiday_type": HOLIDAY_TYPES[h.kind], "known_since": h.known_since,
         "source_id": h.source, "is_substitute": int(h.substitute)}
        for date, h in sorted(HOLIDAYS.items())
    ])


@lru_cache(maxsize=128)
def _known_records(as_of: str):
    return {pd.Timestamp(d): h for d, h in HOLIDAYS.items() if h.known_since <= as_of}


def calendar_features(dates: Iterable, as_of=KNOWLEDGE_CUTOFF) -> pd.DataFrame:
    """Return numeric int8 features in the input row order.

    A Series input preserves its index. Other iterables receive a RangeIndex.
    `as_of` is one forecast origin, not a vector; it must be <= 2026-06-30.
    Scope is 2025-2026. Reject invalid/missing/out-of-range dates rather than
    silently labeling an unsupported year's holidays as ordinary days.
    """
    origin = pd.Timestamp(as_of).normalize()
    if pd.isna(origin) or origin > pd.Timestamp(KNOWLEDGE_CUTOFF):
        raise ValueError(f"as_of must be a valid date <= {KNOWLEDGE_CUTOFF}")
    index = dates.index if isinstance(dates, pd.Series) else None
    d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if d.isna().any():
        raise ValueError("calendar dates must not contain NaT")
    if len(d) and (d.min() < pd.Timestamp(COVERAGE_START) or d.max() > pd.Timestamp(COVERAGE_END)):
        raise ValueError(f"calendar only covers {COVERAGE_START} through {COVERAGE_END}")
    known = _known_records(origin.strftime("%Y-%m-%d"))
    # Unique-date lookup makes repeated station/hour rows inexpensive.
    unique = d.unique()
    lookup = {}
    for day in unique:
        h = known.get(day)
        before = day - pd.Timedelta(days=1)
        after = day + pd.Timedelta(days=1)
        weekend = day.dayofweek >= 5
        rest = weekend or h is not None
        prev_rest = before.dayofweek >= 5 or before in known
        next_rest = after.dayofweek >= 5 or after in known
        kind = h.kind if h else "none"
        lookup[day] = (
            int(h is not None), HOLIDAY_TYPES[kind], int(h.substitute if h else False),
            int(kind in {"temporary", "presidential_election"}),
            int(kind in {"presidential_election", "local_election"}),
            int(kind in {"seollal", "buddha", "children_and_buddha", "chuseok"}),
            int(day.month == 5 and day.day == 1), int(weekend), int(rest),
            int(after in known), int(before in known),
            int(not rest and prev_rest and next_rest),
        )
    columns = [
        "is_holiday", "holiday_type", "is_substitute_holiday", "is_temporary_holiday",
        "is_election_day", "is_lunar_holiday", "is_labor_day", "calendar_is_weekend",
        "is_rest_day", "is_holiday_eve", "is_after_holiday", "is_bridge_day",
    ]
    if len(d):
        values = np.asarray([lookup[x] for x in d], dtype=np.int8)
    else:
        values = np.empty((0, len(columns)), dtype=np.int8)
    return pd.DataFrame(values, columns=columns, index=index)


if __name__ == "__main__":
    print(holiday_records().to_string(index=False))
