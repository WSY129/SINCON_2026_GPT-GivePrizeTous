"""Metric, parameter isolation, and common-round selection checks."""
import unittest
import numpy as np
from src.tune_lightgbm import clipped_log_rmsle,choose_common_round,candidate_parameters
from src.models import lightgbm_parameters

class LightGBMTuningTests(unittest.TestCase):
    def test_metric_matches_official_count_rmsle_with_clipping(self):
        y=np.array([0.,1.,8.]);p=np.array([-2.,.5,2.5])
        name,actual,greater=clipped_log_rmsle(np.log1p(y),p)
        expected=np.sqrt(np.mean((np.log1p(np.expm1(np.maximum(p,0)))-np.log1p(y))**2))
        self.assertEqual(name,'rmsle');self.assertFalse(greater)
        self.assertAlmostEqual(actual,expected,places=12)

    def test_round_selection_uses_pooled_squared_error_and_common_prefix(self):
        iteration,score,common,curve=choose_common_round([[0.,.6,0.],[1.,.6]],[1,1])
        # Arithmetic mean would pick iteration 1; pooled squared error picks 2.
        self.assertEqual(iteration,2);self.assertEqual(common,2)
        self.assertAlmostEqual(score,.6)
        self.assertAlmostEqual(curve[0],np.sqrt(.5))

    def test_selection_respects_hard_cap_and_row_counts(self):
        a=np.ones(1001);a[-1]=0
        iteration,_,common,_=choose_common_round([a,a],[2,3],1000)
        self.assertEqual(common,1000);self.assertEqual(iteration,1)
        _,_,_,curve=choose_common_round([[1.,2.],[3.,4.]],[1,3])
        self.assertAlmostEqual(curve[0],np.sqrt(7))

    def test_legacy_parameters_unchanged_and_override_is_explicit(self):
        cfg={'seed':2026,'threads':4}
        self.assertEqual(lightgbm_parameters(cfg)['n_estimators'],220)
        self.assertEqual(lightgbm_parameters(cfg,False)['num_leaves'],31)
        changed=lightgbm_parameters(dict(cfg,lightgbm_params={'n_estimators':437,'reg_lambda':10}))
        self.assertEqual(changed['n_estimators'],437)
        self.assertEqual(changed['reg_lambda'],10)
        with self.assertRaises(ValueError):lightgbm_parameters(dict(cfg,lightgbm_params={'objective':'poisson'}))

    def test_search_is_deterministic_and_includes_original_parameters(self):
        a=candidate_parameters(2026,16)
        self.assertEqual(a,candidate_parameters(2026,16))
        self.assertEqual(len(a),16);self.assertEqual(a[0]['num_leaves'],63)
        self.assertEqual(a[0]['learning_rate'],.055)

if __name__=='__main__':unittest.main()
