import unittest
import numpy as np
import pandas as pd
from .features import protected_window,matched_growth,REGIME_START,weighted_stats
from .data import cap_origins
from .ensemble import fit_weights,buckets,BUCKETS
from .convergence import final_iterations

class Policies(unittest.TestCase):
    def test_all_station_boundary(self):
        dates=pd.date_range('2025-06-01','2025-08-01');center=pd.Timestamp('2025-07-02')
        keep=protected_window(dates,center,pd.Timestamp('2026-06-30'))
        self.assertTrue((dates[keep]>=REGIME_START).all())
    def test_final_growth_zero(self):
        dates=pd.date_range('2025-01-01','2026-06-30');a=np.ones((len(dates),71));mask=np.ones_like(a,dtype=bool)
        delta,n=matched_growth(dates,a,mask,pd.Timestamp('2026-06-30'),dates.dayofweek.to_numpy(),np.zeros(71,bool),True)
        self.assertTrue((delta==0).all());self.assertTrue((n<14).all())
    def test_raw_mean_keeps_spikes(self):
        a=np.array([[1.],[1.],[9.]])
        self.assertAlmostEqual(weighted_stats(a,np.ones_like(a))['mean'][0],11/3)
    def test_three_fixed_horizon_bands(self):
        c=pd.DataFrame({'date':pd.Timestamp('2026-06-30'),'station_id':3107,'oi':range(100),'horizon':range(1,101),'rich':True})
        d=c[['date','station_id','oi']].copy();d['day']=0
        out,_=cap_origins(c,d)
        self.assertLessEqual(len(out),3);self.assertEqual(len(((out.horizon-1)//41).unique()),len(out))
    def test_bucket_constraints(self):
        names=['lightgbm','catboost_cpu','xgboost','tabm_v5','decomp_gbdt','low_demand_specialist','st_itransformer']
        rng=np.random.default_rng(7);y=rng.normal(size=200);p=y[:,None]+rng.normal(size=(200,7));f=np.repeat([1,2,3,4],50)
        for b in ['global']+BUCKETS:
            w=fit_weights(y,p,f,names,b);self.assertAlmostEqual(w.sum(),1)
            if b!='late_or_low':self.assertEqual(w[5],0)
            if b in ['regular','global']:self.assertGreaterEqual(w[:3].sum(),.2-1e-7)
            if b=='new_stations':self.assertLessEqual(w[6],.45+1e-7)
    def test_partition(self):
        x=pd.DataFrame({'station_id':[3107,3110,3127,3130,3131],'window_spike_amp':[1,1,1,0,0],'hour':[22,22,22,22,12],'anchor_v5':[1,1,1,1,5]})
        np.testing.assert_array_equal(buckets(x),[0,1,2,3,4])
    def test_final_count(self):
        self.assertEqual(final_iterations(2234,65),81)
        self.assertEqual(final_iterations(10,100),110)
if __name__=='__main__':unittest.main()
