import unittest
import numpy as np
import pandas as pd
from src.v4.features import make_features,weighted_stats
from src.v4.ensemble import buckets,fit_weights
from src.v4.models import softmax_shares
from src.v4.preprocessing import NNPreprocessorV4

class V4Tests(unittest.TestCase):
    def test_xgboost_reload_restores_backend(self):
        from unittest.mock import Mock,patch
        from types import SimpleNamespace
        from src.v4.pipeline import infer_one
        for count,policy,expected in [(1,'auto','cuda:0'),(2,'auto','cuda:1'),(0,'auto','cpu'),(1,'cpu','cpu')]:
            model=Mock();model.predict_log.return_value=np.array([1.])
            with patch('src.v4.pipeline.joblib.load',side_effect=[{'x':'features'},model]),patch('src.v4.pipeline.devices',return_value={'cuda_count':count}),patch('src.v4.pipeline.np.save'):
                infer_one(SimpleNamespace(cache='cache',dest='dest',model='xgboost'),{'device':policy})
            model.model.set_params.assert_called_once_with(device=expected)
            model.predict_log.assert_called_once_with('features')
    def test_fourier_constant_level(self):
        from src.v4.features import fourier_predict
        h=pd.DataFrame({'date':pd.date_range('2025-01-01',periods=60),'station_id':3110,'hour':12,'eff_dow':0,'clean':True,'board_log':4.5})
        q=h.tail(5).copy();q['date']=q.date+pd.Timedelta(days=123)
        np.testing.assert_allclose(fourier_predict(h,q,'board_log',np.zeros(len(q))),4.5,atol=1e-6)
    def test_future_invariance(self):
        dates=pd.date_range('2025-01-01','2026-04-05');idx=pd.MultiIndex.from_product([dates,[3107,3112],range(6,24)],names=['date','station_id','hour'])
        h=idx.to_frame(index=False);h['boardings']=20+h.hour+(h.date.dt.dayofyear%30);h['alightings']=h.boardings*2
        h=h.loc[(h.station_id!=3107)|(h.date>='2025-06-28')].reset_index(drop=True)
        meta=pd.DataFrame({'station_id':[3107,3112],'canonical_line':[1,1]});o=pd.Timestamp('2026-02-27');q=h.loc[h.date.between('2026-03-01','2026-03-03')]
        a=make_features(h,q,meta,o);changed=h.copy();changed.loc[changed.date>o,['boardings','alightings']]=999999
        b=make_features(changed,q,meta,o)
        np.testing.assert_array_equal(a.to_numpy(),b.to_numpy())
        for col in ['anchor_v4','z_yoy_raw_mean','delta_yoy_shrunk','z_new_station_trend','z_fourier_ridge']:self.assertIn(col,a)
    def test_shares(self):
        s=softmax_shares(np.random.default_rng(0).normal(size=(71,18)));np.testing.assert_allclose(s.sum(1),1,atol=1e-14)
        total=np.arange(71)+100;restored=(total[:,None]+9)*s-.5
        np.testing.assert_allclose(restored.sum(1),total,atol=1e-12)
    def test_partition(self):
        x=pd.DataFrame({'station_id':[3107,3112,3112,3112],'past_spike_amp':[1,1,0,0],'hour':[22,22,22,12],'anchor_v4':[2,2,2,5]})
        np.testing.assert_array_equal(buckets(x),[0,1,2,3])
    def test_simplex(self):
        y=np.arange(10.);p=np.column_stack([y+1,y-1]);w=fit_weights(y,p);self.assertAlmostEqual(w.sum(),1);self.assertTrue((w>=0).all())
    def test_finite_scaler(self):
        x=pd.DataFrame({'year364_clean_log':[1.]*10+[3.]*10+[np.nan]*80,'anchor_board_log':[8.]*100})
        p=NNPreprocessorV4().fit(x);j=p.cont.index('year364_clean_log');self.assertEqual(p.median[j],2);self.assertEqual(p.iqr[j],2)
        z,_=p.transform(x);self.assertTrue(np.isfinite(z).all());self.assertEqual(z[-1,len(p.cont)+j],1)
    def test_weighted_fractional_trim(self):
        s=weighted_stats(np.array([[0.],[2.],[100.]]),np.array([[1.],[8.],[1.]]))
        self.assertAlmostEqual(s['trimmed'][0],2);self.assertEqual(s['n'][0],3)
    def test_partial_origin_boundary(self):
        from src.v4.data import origins
        for cutoff, step in [('2025-12-30',14),('2026-02-27',7),('2026-06-30',14)]:
            os=origins(cutoff,step)
            self.assertEqual(os[-1],pd.Timestamp(cutoff)-pd.Timedelta(days=14))
            self.assertEqual(len(os),len(set(os)))
            self.assertTrue(all(o<pd.Timestamp(cutoff) for o in os))
    def test_graph_adjacency(self):
        # Isolate torch from native tree libraries on macOS.
        import subprocess,sys
        code="from src.v4.spatial import adjacency; import pandas as pd,numpy as np; m=pd.DataFrame({'station_id':[3117,3118,3761],'canonical_line':[1,1,7]}); a=adjacency(m); assert a[0,1]>0 and a[1,2]>0 and a[0,2]==0; np.testing.assert_allclose(a.sum(1),1)"
        subprocess.run([sys.executable,'-c',code],check=True)
    def test_tabm_per_head_loss_and_shape(self):
        import subprocess,sys
        code="from src.v4.tabm import MultiTaskTabM,multitask_loss; import torch; m=MultiTaskTabM(4,[3],k=16,hidden=32,frequencies=8); p=m(torch.zeros(2,4),torch.ones(2,1,dtype=torch.long)); assert p.shape==(2,16,2); p=torch.zeros(2,16,2); p[:,:8,0]=1; p[:,8:,0]=-1; assert abs(multitask_loss(p,torch.zeros(2,2)).item()-1)<1e-6"
        subprocess.run([sys.executable,'-c',code],check=True)

if __name__=='__main__':unittest.main()
