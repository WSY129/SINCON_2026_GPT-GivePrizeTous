"""OOF blend constraints, duplicate filtering, and MAE bias correction."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ensemble import fit_ensemble, predict_ensemble
from src.metrics import score


class EnsembleTests(unittest.TestCase):
    def test_single_model_identity_preserves_exact_predictions(self):
        values=np.array([.123456785,13.456789125,1000.123456785],dtype=np.float64)
        plan={'weights':{'only':1.,'unused':0.},'bias':0.,'blend_space':'log1p'}
        np.testing.assert_array_equal(predict_ensemble({'only':values},plan),values)

    def test_primary_constraint_and_no_bias_match_boundary_optimum(self):
        truth=np.expm1(np.array([2.,3.,4.,5.]))
        predictions={'large':np.expm1(np.log1p(truth)+.4),
                     'cat':np.expm1(np.log1p(truth)-.1),
                     'xgb':np.expm1(np.log1p(truth)+.6)}
        policy={'method':'primary_constrained_log_mse','primary_model':'large',
                'primary_min_weight':.5,'bias_correction':False}
        plan,_=fit_ensemble(truth,predictions,policy=policy)
        # Unconstrained optimum can cancel bias with large=.2, cat=.8.
        # Requiring large>=.5 makes [.5,.5,0] the known constrained optimum.
        np.testing.assert_allclose([plan['weights'][n] for n in predictions],[.5,.5,0],atol=1e-8)
        self.assertEqual(plan['bias'],0.)
        self.assertEqual(plan['excluded'],{})
        self.assertAlmostEqual(plan['development_score'],.15,places=10)

    def test_primary_blend_keeps_correlated_models_and_is_order_invariant(self):
        truth=np.expm1(np.array([2.,3.,4.,5.]))
        error=np.array([.1,-.2,.3,-.1])
        predictions={'large':np.expm1(np.log1p(truth)+error),
                     'cat':np.expm1(np.log1p(truth)+error*.9),
                     'xgb':np.expm1(np.log1p(truth)-error*.7)}
        policy={'method':'primary_constrained_log_mse','primary_model':'large',
                'primary_min_weight':.5,'bias_correction':False}
        plan,corr=fit_ensemble(truth,predictions,policy=policy)
        other,_=fit_ensemble(truth,dict(reversed(list(predictions.items()))),policy=policy)
        self.assertGreater(corr.loc['large','cat'],.95)
        self.assertEqual(set(plan['retained']),set(predictions))
        self.assertAlmostEqual(sum(plan['weights'].values()),1.,places=12)
        self.assertGreaterEqual(plan['weights']['large'],.5-1e-12)
        np.testing.assert_allclose(predict_ensemble(predictions,plan),predict_ensemble(predictions,other),rtol=1e-8)

    def test_primary_blend_rejects_invalid_inputs(self):
        policy={'method':'primary_constrained_log_mse','primary_model':'large','primary_min_weight':.5}
        with self.assertRaises(ValueError):fit_ensemble([1.,2.],{'large':np.array([1.,np.nan])},policy=policy)
        with self.assertRaises(ValueError):fit_ensemble([1.,2.],{'large':np.array([-1.,2.])},policy=policy)
        with self.assertRaises(ValueError):fit_ensemble([1.,2.],{'cat':np.array([1.,2.])},policy=policy)
        with self.assertRaises(ValueError):fit_ensemble([1.,2.],{'large':np.array([1.,2.])},metric='mae',policy=policy)

    def test_convex_weights_and_duplicate_residual_filter(self):
        truth = np.array([10., 15., 20., 25., 30., 35., 40., 45.])
        error = np.array([1., -3., 4., -1., 2., -4., 3., -2.])
        predictions = {"model_a": truth + error, "duplicate_a": truth + error, "model_b": truth - error * .8 + 0.4}
        plan, correlation = fit_ensemble(truth, predictions, metric="mae")
        self.assertTrue(all(weight >= 0 for weight in plan["weights"].values()))
        self.assertAlmostEqual(sum(plan["weights"].values()), 1., places=12)
        self.assertLessEqual(len(set(plan["retained"]) & {"model_a", "duplicate_a"}), 1)
        for i, left in enumerate(plan["retained"]):
            for right in plan["retained"][i + 1:]:
                self.assertFalse(correlation.loc[left, right] > .95)
        blended = predict_ensemble(predictions, plan)
        self.assertTrue(np.isfinite(blended).all())
        self.assertTrue((blended >= 0).all())
        self.assertLessEqual(score(truth, blended, 'mae'), min(score(truth, p, 'mae') for p in predictions.values()))

    def test_mae_bias_uses_median_and_improves_objective(self):
        truth = np.array([10., 20., 30., 40., 50.])
        # Residual mean is -0.2 while the median correction is +5.
        predictions = {"only": truth + np.array([-5., -5., -5., -5., 19.])}
        plan, _ = fit_ensemble(truth, predictions, metric="mae")
        self.assertEqual(plan["bias"], 5.)
        result = predict_ensemble(predictions, plan)
        self.assertLess(score(truth, result, 'mae'), score(truth, predictions["only"], 'mae'))
        self.assertAlmostEqual(plan["development_score"], score(truth, result, 'mae'))

    def test_rmse_does_not_apply_mae_median_correction(self):
        truth = np.array([10., 20., 30.])
        predictions = {"only": truth - 5}
        plan, _ = fit_ensemble(truth, predictions, metric="rmse")
        self.assertEqual(plan["bias"], 0.)

    def test_rmsle_blends_and_calibrates_in_log1p_space(self):
        truth_log = np.array([1., 2., 3., 4., 5.])
        truth = np.expm1(truth_log)
        residual_log = np.array([-.2, -.2, -.2, -.2, .6])
        predictions = {"only": np.expm1(truth_log + residual_log)}
        plan, _ = fit_ensemble(truth, predictions)
        self.assertEqual(plan["metric"], "rmsle")
        self.assertEqual(plan["blend_space"], "log1p")
        self.assertEqual(plan["bias_space"], "log1p")
        # RMSLE minimizes squared error in log space, so use mean, not median.
        self.assertAlmostEqual(plan["bias"], .04, places=12)
        expected = np.expm1(truth_log + residual_log + .04)
        actual = predict_ensemble(predictions, plan)
        np.testing.assert_allclose(actual, expected)
        self.assertLess(score(truth, actual, "rmsle"), score(truth, predictions["only"], "rmsle"))

    def test_predict_ensemble_uses_weighted_geometric_mean_of_one_plus_counts(self):
        predictions = {"a": np.array([0., 3., 8.]), "b": np.array([3., 8., 15.])}
        plan = {"weights": {"a": .25, "b": .75}, "bias": .1, "blend_space": "log1p"}
        expected = np.exp(.1) * (1 + predictions["a"])**.25 * (1 + predictions["b"])**.75 - 1
        np.testing.assert_allclose(predict_ensemble(predictions, plan), expected)
        plan["bias"] = -10.
        np.testing.assert_array_equal(predict_ensemble(predictions, plan), np.zeros(3))

    def test_constant_residual_duplicates_are_removed_even_when_correlation_undefined(self):
        truth = np.array([10., 20., 30.])
        predictions = {"a": truth + 5, "duplicate": truth + 5}
        plan, correlation = fit_ensemble(truth, predictions, metric="mae")
        self.assertTrue(np.isnan(correlation.loc["a", "duplicate"]))
        self.assertEqual(len(plan["retained"]), 1)
        self.assertEqual(len(plan["excluded"]), 1)


if __name__ == "__main__":
    unittest.main()
