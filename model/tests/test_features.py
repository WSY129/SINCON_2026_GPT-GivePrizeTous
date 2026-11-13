"""Leakage and horizon-boundary regression tests using independent toy data."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import make_features, training_matrix


def fixture(periods=500):
    dates = pd.date_range("2025-01-01", periods=periods)
    records = []
    for station in [101, 202]:
        for hour in [6, 17]:
            for day, date in enumerate(dates):
                value = station + hour + day * 3
                records.append((date, station, hour, value, value * 2))
    history = pd.DataFrame(records, columns=["date", "station_id", "hour", "boardings", "alightings"])
    metadata = pd.DataFrame({
        "station_id": [101, 202], "canonical_line": [1, 2],
        "station_name": ["A", "B"], "first_date": ["2025-01-01"]*2,
        "last_date": ["2026-06-30"]*2,
    })
    return history, metadata


def queries(origin, horizons=(1, 7, 123)):
    return pd.DataFrame([
        (pd.Timestamp(origin) + pd.Timedelta(days=horizon), station, hour)
        for horizon in horizons for station in [101, 202] for hour in [6, 17]
    ], columns=["date", "station_id", "hour"])


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history, cls.metadata = fixture()
        cls.origin = pd.Timestamp("2025-12-31")

    def test_future_target_and_alightings_mutation_cannot_change_features(self):
        query = queries(self.origin)
        expected = make_features(self.history, query, self.metadata, self.origin)
        mutated = self.history.copy()
        future = mutated.date > self.origin
        mutated.loc[future, "boardings"] = 999_999_999
        mutated.loc[future, "alightings"] = -999_999_999
        actual = make_features(mutated, query, self.metadata, self.origin)
        assert_frame_equal(expected, actual, check_exact=True)
        # Supplying hidden labels on query rows must not expose them as features.
        labelled_query = query.assign(boardings=123_456, alightings=654_321)
        assert_frame_equal(expected, make_features(self.history, labelled_query, self.metadata, self.origin), check_exact=True)

    def test_query_reordering_keeps_each_feature_attached_to_its_key(self):
        query = queries(self.origin)
        expected = make_features(self.history, query, self.metadata, self.origin)
        permutation = np.array([8, 2, 11, 0, 5, 9, 3, 7, 1, 10, 4, 6])
        actual = make_features(self.history, query.iloc[permutation], self.metadata, self.origin)
        assert_frame_equal(expected.iloc[permutation].reset_index(drop=True), actual, check_exact=True)

    def test_seasonal7_repeats_latest_observed_weekday_for_all_horizons(self):
        query = queries(self.origin)
        features = make_features(self.history, query, self.metadata, self.origin)
        for index, row in query.iterrows():
            observed = self.history.loc[
                (self.history.station_id == row.station_id)
                & (self.history.hour == row.hour)
                & (self.history.date <= self.origin)
                & (self.history.date.dt.dayofweek == row.date.dayofweek)
            ].sort_values("date").iloc[-1]
            self.assertEqual(features.loc[index, "seasonal7"], observed.boardings)
            self.assertEqual(features.loc[index, "seasonal7_alight"], observed.alightings)
            self.assertEqual(features.loc[index, "horizon"], (row.date - self.origin).days)

    def test_year364_never_uses_lookup_after_origin(self):
        origin = pd.Timestamp('2025-11-26')
        query = queries(origin, horizons=(1, 123, 364, 365, 400))
        features = make_features(self.history, query, self.metadata, origin)
        for index, row in query.iterrows():
            lookup_date = row.date - pd.Timedelta(days=364)
            if lookup_date > origin or lookup_date < self.history.date.min():
                self.assertTrue(np.isnan(features.loc[index, "year364"]))
            else:
                match = self.history.loc[(self.history.date == lookup_date) & (self.history.station_id == row.station_id) & (self.history.hour == row.hour)]
                self.assertEqual(features.loc[index, "year364"], match.boardings.iloc[0])

    def test_metadata_last_date_and_future_first_date_do_not_leak(self):
        query = queries(self.origin)
        expected = make_features(self.history, query, self.metadata, self.origin)
        metadata = self.metadata.assign(first_date="2099-01-01", last_date="1900-01-01", station_name="Arbitrary")
        actual = make_features(self.history, query, metadata, self.origin)
        assert_frame_equal(expected, actual, check_exact=True)
        self.assertNotIn("last_date", actual.columns)
        self.assertNotIn("first_date", actual.columns)

    def test_full_123_day_label_window_cutoff_boundary(self):
        # First anchor is Jan 28. A cutoff exactly 123 days later permits it;
        # one day earlier must not shorten the training label window silently.
        history, metadata = fixture(periods=200)
        anchor = pd.Timestamp("2025-01-28")
        cutoff = anchor + pd.Timedelta(days=123)
        x, y, audit = training_matrix(history, metadata, cutoff, horizon=123, max_rows=10000)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["origin"], "2025-01-28")
        self.assertEqual(pd.Timestamp(audit[0]["label_end"]), cutoff)
        self.assertEqual(x.horizon.min(), 1)
        self.assertEqual(x.horizon.max(), 123)
        self.assertEqual(len(x), 123 * 4)
        self.assertEqual(len(x), len(y))
        self.assertTrue(all(pd.Timestamp(a["label_end"]) <= cutoff for a in audit))
        self.assertTrue(all(pd.Timestamp(a["origin"]) + pd.Timedelta(days=123) <= cutoff for a in audit))
        with self.assertRaisesRegex(ValueError, "Insufficient history"):
            training_matrix(history, metadata, cutoff - pd.Timedelta(days=1), horizon=123)
        # Changing labels beyond the cutoff cannot change either X or y.
        mutated = history.copy()
        mutated.loc[mutated.date > cutoff, ["boardings", "alightings"]] = 999_999
        changed_x, changed_y, changed_audit = training_matrix(mutated, metadata, cutoff, horizon=123, max_rows=10000)
        assert_frame_equal(x, changed_x, check_exact=True)
        np.testing.assert_array_equal(y, changed_y)
        self.assertEqual(audit, changed_audit)


if __name__ == "__main__":
    unittest.main()
