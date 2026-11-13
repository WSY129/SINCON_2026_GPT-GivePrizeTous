import unittest
import pickle
import numpy as np
import pandas as pd
from src.features_v2 import cleaned,calendar,make_features_v2,partial_origins,sampled_pairs
from src.nn_preprocessor import NNPreprocessor
from src.ensemble_v2 import convex_weights,evaluate_ensemble

def fixture(start='2025-01-01',end='2026-06-30',stations=(3110,3111,3107)):
    index=pd.MultiIndex.from_product([pd.date_range(start,end),stations,[6,7]],names=['date','station_id','hour'])
    frame=index.to_frame(index=False);frame['boardings']=10.;frame['alightings']=20.
    meta=pd.DataFrame({'station_id':stations,'canonical_line':1})
    return frame,meta

class FeatureTests(unittest.TestCase):
    def test_partial_and_latest_origins(self):
        origins=partial_origins('2026-06-30')
        self.assertEqual(origins[0],pd.Timestamp('2025-02-11'))
        self.assertEqual(origins[-1],pd.Timestamp('2026-06-16'))
        frame,_=fixture();pairs,origins=sampled_pairs(frame,'2026-06-30',12300)
        dates=frame.loc[pairs.row,'date'].reset_index(drop=True)
        os=pd.Series([origins[i] for i in pairs.origin_index])
        self.assertTrue((dates>os).all());self.assertTrue((dates<=pd.Timestamp('2026-06-30')).all())
        self.assertTrue(((dates-os).dt.days==pairs.horizon).all())
        self.assertEqual(pairs.horizon.nunique(),123)
        self.assertLessEqual(pairs.horizon.value_counts().max()-pairs.horizon.value_counts().min(),1)
        self.assertTrue((pairs.origin_index==len(origins)-1).any())
    def test_cleaning_opening_holiday_break(self):
        frame,_=fixture();h=cleaned(frame,'2025-08-30')
        self.assertFalse(h.loc[h.date=='2025-05-01','clean'].any())
        self.assertFalse(h.loc[h.date=='2025-08-15','clean'].any())
        self.assertFalse(h.loc[h.station_id.eq(3107)&h.date.between('2025-06-28','2025-06-30'),'clean'].any())
        self.assertFalse(h.loc[h.station_id.isin([3110,3111])&(h.date<'2025-07-01'),'clean'].any())
        self.assertTrue(h.loc[h.station_id.eq(3107)&h.date.eq('2025-07-01'),'clean'].all())
    def test_future_invariance_and_holiday_anchor(self):
        frame,meta=fixture();frame.loc[frame.date=='2025-08-15','boardings']=1e8
        query=pd.DataFrame({'date':pd.to_datetime(['2025-10-03','2025-09-01']),'station_id':[3110,3110],'hour':[6,6]})
        a=make_features_v2(frame,query,meta,'2025-08-30')
        frame.loc[frame.date>'2025-08-30',['boardings','alightings']]=1e12
        b=make_features_v2(frame,query,meta,'2025-08-30')
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.eff_dow.iloc[0],6)
        np.testing.assert_allclose(a.anchor_board_log,np.log1p(10),rtol=1e-6)
    def test_as_of_election_not_known_early(self):
        early=calendar(pd.to_datetime(['2025-06-03']),'2025-02-11')
        later=calendar(pd.to_datetime(['2025-06-03']),'2025-05-31')
        self.assertEqual(int(early.is_election_day.iloc[0]),0)
        self.assertEqual(int(later.is_election_day.iloc[0]),1)
    def test_year_holiday_replaced(self):
        frame,meta=fixture();frame.loc[frame.date=='2025-10-06','boardings']=10000
        q=pd.DataFrame({'date':pd.to_datetime(['2026-10-05']),'station_id':[3110],'hour':[6]})
        x=make_features_v2(frame,q,meta,'2026-06-30')
        self.assertAlmostEqual(x.year364_clean_log.iloc[0],np.log1p(10),places=5)
    def test_elasticity_uses_clean_ordinary_regime(self):
        frame,meta=fixture(stations=(3110,3107))
        frame.loc[(frame.station_id==3110)&(frame.date<'2025-07-01'),'boardings']=1e8
        frame.loc[frame.date=='2025-05-01','boardings']=1e9
        query=pd.DataFrame({'date':pd.to_datetime(['2025-09-01','2025-09-01']),'station_id':[3110,3107],'hour':[6,6]})
        x=make_features_v2(frame,query,meta,'2025-08-30')
        np.testing.assert_allclose(x.holiday_elasticity,0,atol=1e-6)

class NeuralAndEnsembleTests(unittest.TestCase):
    def test_xgboost_validation_matches_inference(self):
        import subprocess,sys,textwrap
        code=textwrap.dedent('''
            import numpy as np,pandas as pd
            from src.xgb_compat import InferenceConsistentXGBRegressor
            rng=np.random.default_rng(12)
            tr=pd.DataFrame({'a':rng.normal(size=600),'b':pd.Categorical(rng.integers(0,5,600),categories=range(5))})
            va=pd.DataFrame({'a':rng.normal(1,2,size=180),'b':pd.Categorical(rng.integers(0,5,180),categories=range(5))})
            y=tr.a.to_numpy()*.3+tr.b.astype(int).to_numpy()*.2
            v=va.a.to_numpy()*.3+va.b.astype(int).to_numpy()*.2
            model=InferenceConsistentXGBRegressor(tree_method='hist',enable_categorical=True,n_estimators=25,max_depth=3,n_jobs=1,early_stopping_rounds=5,eval_metric='rmse',device='cpu')
            model.fit(tr,y,eval_set=[(va,v)],verbose=False)
            actual=np.sqrt(np.mean((model.predict(va)-v)**2))
            recorded=model.evals_result()['validation_0']['rmse'][model.best_iteration]
            assert abs(actual-recorded)<1e-6,(actual,recorded)
        ''')
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
    def test_fold_preprocessor_unknown_mask_serialization(self):
        x=pd.DataFrame({'station_id':[1,2,1],'value':[0.,2.,np.nan]})
        p=pickle.loads(pickle.dumps(NNPreprocessor().fit(x)))
        num,cat=p.transform(pd.DataFrame({'station_id':[3,1],'value':[np.nan,1e9]}))
        self.assertEqual(cat[0,0],0);self.assertEqual(num[0,-1],1);self.assertEqual(num[1,0],10)
        self.assertTrue(np.isfinite(num).all())
    def test_tabm_shape_and_non_cancelling_loss(self):
        import torch
        from src.tabm_v2 import MultiTaskTabM,multitask_loss
        torch.set_num_threads(1);model=MultiTaskTabM(6,[5,3],k=8,hidden=16)
        out=model(torch.randn(7,6),torch.zeros(7,2,dtype=torch.long))
        self.assertEqual(tuple(out.shape),(7,8,2))
        p=torch.tensor([[[1.,2.],[-1.,-2.]]],requires_grad=True)
        loss=multitask_loss(p,torch.zeros(1,2));self.assertAlmostEqual(loss.item(),1.8,places=5)
        loss.backward();self.assertGreater(p.grad.abs().sum().item(),0)
    def test_convex_and_lofo(self):
        y=np.linspace(0,2,80);pred=np.column_stack([y+.1,y-.2,y+.3,y-.4]);fold=np.repeat(np.arange(1,5),20)
        w=convex_weights(y,pred);self.assertAlmostEqual(w.sum(),1,places=10);self.assertTrue((w>=0).all())
        self.assertLess(np.mean((pred@w-y)**2),1e-8)
        _,_,report,_=evaluate_ensemble(y,pred,fold,['a','b','c','d'],[.15,.15,.35,.35])
        self.assertEqual(len(report['lofo']),4)
        for row in report['lofo']:self.assertAlmostEqual(sum(row['weights'].values()),1,places=9)

if __name__=='__main__':unittest.main()
