"""Holiday declarations must not leak across the forecast-origin boundary."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.holiday_calendar import calendar_features, holiday_records


class HolidayCalendarTests(unittest.TestCase):
    def test_temporary_holiday_announcement_boundary(self):
        dates = ["2025-01-26", "2025-01-27", "2025-01-28"]
        before = calendar_features(dates, as_of="2025-01-13")
        after = calendar_features(dates, as_of="2025-01-14")
        self.assertEqual(before.is_holiday.tolist(), [0, 0, 1])
        self.assertEqual(after.is_holiday.tolist(), [0, 1, 1])
        self.assertEqual(before.loc[0, "is_holiday_eve"], 0)
        self.assertEqual(after.loc[0, "is_holiday_eve"], 1)
        self.assertEqual(before.loc[1, "holiday_type"], 0)

    def test_2025_presidential_election_was_not_known_in_january(self):
        self.assertEqual(calendar_features(["2025-06-03"], "2025-01-31").is_holiday.iloc[0], 0)
        after = calendar_features(["2025-06-03"], "2025-04-08")
        self.assertEqual(after.is_holiday.iloc[0], 1)
        self.assertEqual(after.is_election_day.iloc[0], 1)

    def test_labor_day_and_constitution_day_new_designations(self):
        dates = ["2025-05-01", "2026-05-01", "2025-07-17", "2026-07-17"]
        final = calendar_features(dates)
        self.assertEqual(final.is_holiday.tolist(), [0, 1, 0, 1])
        before = calendar_features(dates, "2026-02-02")
        self.assertEqual(before.is_holiday.tolist(), [0, 0, 0, 0])
        self.assertEqual(before.is_labor_day.tolist(), [1, 1, 0, 0])

    def test_no_false_substitutes(self):
        dates = ["2026-09-26", "2026-09-27", "2026-09-28", "2026-06-08", "2026-10-05"]
        frame = calendar_features(dates)
        self.assertEqual(frame.is_holiday.tolist(), [1, 0, 0, 0, 1])
        self.assertEqual(frame.is_rest_day.tolist(), [1, 1, 0, 0, 1])
        self.assertEqual(frame.is_substitute_holiday.tolist(), [0, 0, 0, 0, 1])

    def test_year_and_test_scope_counts(self):
        records = holiday_records()
        self.assertEqual((records.date.str.startswith("2025")).sum(), 19)
        self.assertEqual((records.date.str.startswith("2026")).sum(), 22)
        test = calendar_features(pd.date_range("2026-07-01", "2026-10-31"))
        self.assertEqual(test.is_holiday.sum(), 9)

    def test_order_index_dtype_and_empty_input(self):
        dates = pd.Series(["2026-10-05", "2025-05-01", "2026-10-05"], index=[9, 4, 9])
        frame = calendar_features(dates)
        self.assertEqual(frame.index.tolist(), [9, 4, 9])
        self.assertEqual(frame.is_holiday.tolist(), [1, 0, 1])
        self.assertTrue(all(t == np.dtype("int8") for t in frame.dtypes))
        self.assertEqual(calendar_features([]).shape, (0, 12))

    def test_unsupported_dates_and_post_cutoff_origin_rejected(self):
        for dates in [[None], ["2027-01-01"], ["2024-12-31"]]:
            with self.assertRaises(ValueError):
                calendar_features(dates)
        with self.assertRaises(ValueError):
            calendar_features(["2026-07-17"], "2026-07-01")


if __name__ == "__main__":
    unittest.main()
