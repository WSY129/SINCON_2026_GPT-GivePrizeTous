"""Cutoff, holiday, centering and row alignment checks for postprocessing."""
import unittest
import numpy as np
import pandas as pd
from src.postprocessing import build_context,fit_plan,apply_plan

def spec(kind='none',scale=0.,days=28,weight=.2):
    return {'name':'test','calibration':{'kind':kind,'scale':scale},'anchor':{'days':days,'weight':weight},'cap_days':0,'eligible':True}

class PostprocessingTests(unittest.TestCase):
    def setUp(self):
        dates=pd.date_range('2025-01-01','2025-02-05')
        self.history=pd.DataFrame({'date':dates,'station_id':1,'hour':6,'boardings':100.})
        # Jan 27 is a preannounced temporary holiday, excluded from Monday profile.
        self.history.loc[self.history.date=='2025-01-27','boardings']=10000.
        self.query=pd.DataFrame({'date':pd.to_datetime(['2025-02-03','2025-03-01','2025-02-03']),'station_id':[1,1,99],'hour':6})

    def test_history_is_cutoff_safe_and_query_targets_are_ignored(self):
        original=build_context(self.history,self.query,'2025-01-31')
        changed=self.history.copy();changed.loc[changed.date>'2025-01-31','boardings']=1e8
        query=self.query.assign(boardings=-999999)
        pd.testing.assert_frame_equal(original,build_context(changed,query,'2025-01-31'))

    def test_historical_holidays_excluded_and_missing_anchor_falls_back(self):
        context=build_context(self.history,self.query,'2025-01-31')
        self.assertAlmostEqual(context.anchor_28.iloc[0],np.log1p(100.))
        self.assertTrue(np.isnan(context.anchor_28.iloc[2]))
        pred=np.array([200.,1000.,333.])
        plan=fit_plan(context,pred,pred,spec())
        result=apply_plan(context,pred,plan)
        self.assertAlmostEqual(result[0],np.expm1(.8*np.log1p(200)+.2*np.log1p(100)))
        self.assertEqual(result[1],pred[1]) # March 1 holiday unchanged exactly.
        self.assertAlmostEqual(result[2],pred[2])

    def test_context_and_group_correction_preserve_query_order(self):
        context=build_context(self.history,self.query,'2025-01-31')
        pred=np.array([200.,1000.,333.]);plan=fit_plan(context,pred,pred,spec())
        expected=apply_plan(context,pred,plan)
        order=[2,0,1]
        changed=build_context(self.history,self.query.iloc[order],'2025-01-31')
        np.testing.assert_array_equal(apply_plan(changed,pred[order],plan),expected[order])

    def test_per_fold_centering_does_not_transport_global_bias(self):
        context=pd.DataFrame({'hour':[6,7,6,7],'dow':[0,0,0,0],'is_holiday':False})
        pred=np.expm1(np.full(4,4.))
        truth=np.expm1(np.array([4.2,4.2,3.7,3.7]))
        plan=fit_plan(context,pred,truth,spec('hour',.5,0,0),fold_ids=[1,1,2,2])
        self.assertTrue(all(abs(v)<1e-12 for v in plan['offsets'].values()))

    def test_identity_exact_and_anchor_share_guard(self):
        context=build_context(self.history,self.query,'2025-01-31')
        pred=np.array([.123456785,1000.,333.])
        plan=fit_plan(context,pred,pred,spec(days=0,weight=0))
        np.testing.assert_array_equal(apply_plan(context,pred,plan),pred)
        plan['spec']['anchor']={'days':28,'weight':.5}
        with self.assertRaises(ValueError):apply_plan(context,pred,plan)

if __name__=='__main__':unittest.main()
