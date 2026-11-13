"""Behavior checks for score conventions, leakage and submission corruption."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import forecast_diagnostics, regression_metrics, residual_autocorrelation, score, seasonal_scales
from validate_submission import validate_submission


class MetricTests(unittest.TestCase):
    def test_score_conventions(self):
        result = regression_metrics([0, 2, 4], [0, 4, 1])
        self.assertAlmostEqual(result["mae"], 5 / 3)
        self.assertAlmostEqual(result["rmse"], math.sqrt(13 / 3))
        self.assertAlmostEqual(result["mbe"], -1 / 3)
        self.assertAlmostEqual(result["smape"], (0 + 200 * 2 / 6 + 200 * 3 / 5) / 3)
        self.assertAlmostEqual(result["rmsle"], math.sqrt(((math.log(5)-math.log(3))**2 + (math.log(2)-math.log(5))**2)/3))
        self.assertEqual(score([0, 0], [0, 0], "smape"), 0)

    def test_invalid_inputs_rejected(self):
        for y, p in [([], []), ([1], [1, 2]), ([1], [np.nan]), ([np.inf], [1])]:
            with self.assertRaises(ValueError):
                regression_metrics(y, p)
        with self.assertRaises(ValueError):
            score([1], [-1], "rmsle")

    def test_official_rmsle_is_root_mean_squared_log1p_error(self):
        truth = np.expm1(np.array([0., 1., 2.]))
        predictions = np.expm1(np.array([1., 3., 5.]))
        expected = math.sqrt((1.0**2 + 2.0**2 + 3.0**2) / 3)
        self.assertAlmostEqual(score(truth, predictions, "rmsle"), expected, places=12)
        self.assertAlmostEqual(regression_metrics(truth, predictions)["rmsle"], expected, places=12)
        self.assertEqual(score([0, 1, 20], [0, 1, 20], "rmsle"), 0)
        self.assertAlmostEqual(score([0], [1], "rmsle"), math.log(2), places=12)

    def test_seasonal_scale_matches_calendar_not_rows(self):
        train = pd.DataFrame({"station_id": [1]*3, "hour": [6]*3, "date": ["2025-01-01", "2025-01-08", "2025-01-16"], "boardings": [10, 12, 99]})
        scales = seasonal_scales(train)
        self.assertEqual(scales.loc[0, "seasonal_pairs"], 1)
        self.assertEqual(scales.loc[0, "mase_scale"], 2)

    def test_zero_scales_undefined_and_coverage_reported(self):
        train = pd.DataFrame({"station_id": [1]*8+[2]*8, "hour": [6]*16, "date": list(pd.date_range("2025-01-01", periods=8))*2, "boardings": list(range(8))+[3]*8})
        valid = pd.DataFrame({"station_id": [1, 2], "hour": [6, 6], "date": ["2025-01-09"]*2, "boardings": [8, 3]})
        result = forecast_diagnostics(train, valid, [15, 4])
        self.assertAlmostEqual(result["summary"]["mase_macro"], 1)
        self.assertAlmostEqual(result["summary"]["mase_row_weighted"], 1)
        self.assertEqual(result["summary"]["mase_row_coverage"], 0.5)
        self.assertEqual(result["summary"]["mase_undefined_series"], 1)
        self.assertEqual(result["per_step"]["horizon"].tolist(), [1])
        with self.assertRaisesRegex(ValueError, "leakage"):
            forecast_diagnostics(train, valid, [15, 4], origin="2025-01-07")

    def test_residual_acf_does_not_cross_groups_or_missing_days(self):
        frame = pd.DataFrame({"station_id": [1]*4 + [2]*4, "hour": [6]*8, "date": list(pd.date_range("2025-01-01", periods=4))*2, "residual": [1,2,3,4,100,90,80,70]})
        acf = residual_autocorrelation(frame, lags=(1, 7))
        self.assertTrue(np.allclose(acf.loc[acf.lag_days.eq(1), "acf"], 1))
        self.assertEqual(acf.loc[acf.lag_days.eq(1), "pairs"].tolist(), [3, 3])
        self.assertEqual(acf.loc[acf.lag_days.eq(7), "pairs"].tolist(), [0, 0])
        self.assertTrue(acf.loc[acf.lag_days.eq(7), "acf"].isna().all())


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.test = pd.DataFrame({"example_id": ["J0", "J1", "J2"], "date": ["2026-07-01", "2026-09-30", "2026-10-01"]})
        self.sample = pd.DataFrame({"example_id": ["J0", "J1"], "boardings": [0, 0]})
        self.sub = pd.DataFrame({"example_id": ["J0", "J1"], "boardings": [1.2, 2.3]})

    def test_template_discrepancy_explicit(self):
        result = validate_submission(self.sub, self.test, self.sample, scope="template")
        self.assertTrue(result["valid"])
        self.assertFalse(result["template_matches_test"])
        self.assertEqual(result["scope_discrepancy"]["test_ids_absent_from_template"], 1)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertFalse(result["matches_official_scope"])
        self.assertIn("diagnostic only", result["warnings"][0])
        self.assertEqual(result["scope_discrepancy"]["missing_period_start"], "2026-10-01")
        with self.assertRaisesRegex(ValueError, "requires 3"):
            validate_submission(self.sub, self.test, self.sample, scope="test")

    def test_invalid_files_rejected(self):
        invalid = [self.sub.iloc[::-1], self.sub.assign(boardings=[np.nan, 2]), self.sub.assign(boardings=[np.inf, 2]), self.sub.assign(boardings=[-1, 2]), self.sub.assign(example_id=["J0", "J0"]), self.sub.assign(extra=0), self.sub.assign(example_id=["J0", "BAD"])]
        for frame in invalid:
            with self.subTest(frame=frame.to_dict()):
                with self.assertRaises(ValueError):
                    validate_submission(frame, self.test, self.sample, scope="template")

    def test_test_scope_and_distribution(self):
        sub = pd.DataFrame({"example_id": self.test.example_id, "boardings": [1, 2, 3]})
        result = validate_submission(sub, self.test, self.sample, scope="test", reference_predictions=[1, 1, 1])
        self.assertEqual(result["rows"], 3)
        self.assertTrue(result["matches_official_scope"])
        self.assertEqual(result["distribution_comparison"]["mean_ratio"], 2)

    def test_default_requires_official_full_test_scope(self):
        with self.assertRaisesRegex(ValueError, "requires 3"):
            validate_submission(self.sub, self.test, self.sample)
        full = pd.DataFrame({"example_id": self.test.example_id, "boardings": [1, 2, 3]})
        result = validate_submission(full, self.test, self.sample)
        self.assertEqual(result["scope"], "test")
        self.assertTrue(result["matches_official_scope"])


if __name__ == "__main__":
    unittest.main()
