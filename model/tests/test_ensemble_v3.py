import unittest, pickle
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.domain_events import oneoff_mask, add_domain_features, schedules
from src.features_v3 import make_features_v3, cleaned, sampled_pairs
from src.features_v2 import calendar
from src.nn_preprocessor_v3 import NNPreprocessorV3
from src.models_v3 import ResidualTree
from src.ensemble_v3 import evaluate_ensemble
from test_ensemble_v2 import fixture

class V3Tests(unittest.TestCase):
    def test_opening_and_exact_outages(self):
        rows=pd.MultiIndex.from_product([pd.date_range('2025-06-28','2025-06-30'),[3107,3108,3109],range(6,24)],names=['date','station_id','hour']).to_frame(index=False)
        self.assertEqual(int(oneoff_mask(rows).sum()),162)
        rows=pd.DataFrame({'date':pd.to_datetime(['2025-08-13']*3),'station_id':[3112]*3,'hour':[11,12,13]})
        self.assertEqual(oneoff_mask(rows).tolist(),[False,True,True])

    def test_schedule_asof_and_september(self):
        q=pd.DataFrame({'date':pd.to_datetime(['2026-07-03','2026-09-04','2026-07-31']),'station_id':[3127,3127,3139],'hour':[21]*3})
        early=pd.concat([q,calendar(q.date,'2025-12-01')],axis=1)
        late=pd.concat([q,calendar(q.date,'2026-06-30')],axis=1)
        # Synthetic schedule verifies the generic as-of path without loading
        # KBO facts excluded from production for unresolved reuse conditions.
        fixture_schedule={'kbo':[{'known_since':'2025-12-19','coverage_start':'2026-03-28','coverage_end':'2026-09-06','home_dates':['2026-07-03','2026-09-04']}],
                          'festivals':[{'kind':'pentaport','known_since':'2026-05-31','start':'2026-07-31','end':'2026-08-02'}]}
        with patch('src.domain_events.schedules',return_value=fixture_schedule):
            a=add_domain_features(early,'2025-12-01');b=add_domain_features(late,'2026-06-30')
        self.assertEqual(a.munhak_kbo_scheduled.sum(),0);self.assertEqual(a.songdo_pentaport_scheduled.sum(),0)
        self.assertEqual(b.munhak_kbo_scheduled.sum(),2);self.assertEqual(b.songdo_pentaport_scheduled.sum(),1)
        self.assertTrue(b.loc[1,'munhak_schedule_known'])
        self.assertEqual(schedules()['kbo'],[])

    def test_neural_anchor_imputation_and_mask(self):
        x=pd.DataFrame({'station_id':[1,1,2,2],'anchor_board_log':[1.,3.,5.,7.],'year364_clean_log':[np.nan]*4,'yoy_seas_log_delta':[0.]*4})
        p=pickle.loads(pickle.dumps(NNPreprocessorV3().fit(x)))
        a,mask=p.values(x);idx=p.cont.index('year364_clean_log')
        np.testing.assert_allclose(a[:,idx],x.anchor_board_log)
        num,cat=p.transform(x)
        np.testing.assert_allclose(num[:,idx],num[:,p.cont.index('anchor_board_log')])
        self.assertTrue((num[:,len(p.cont)+idx]==1).all());self.assertTrue(np.isfinite(num).all())
        self.assertGreater(p.iqr[idx],0)

    def test_tree_category_separation(self):
        x=pd.DataFrame({'station_id':[3110.,3111.],'station_hour':[1.,2.],'station_eff_dow':[3.,4.],'value':[np.nan,1.]})
        for name in ['lightgbm','xgboost','catboost']:
            model=ResidualTree(name,{});frame=model.frame(x,fit=True)
            self.assertEqual('station_hour' in frame,name=='catboost')
            self.assertTrue(pd.isna(frame.value.iloc[0]))

    def test_unseen_event_is_neutral_in_neural_input(self):
        x=pd.DataFrame({'station_id':[1,1,1], 'event':[0.,0.,0.], 'rare':[0.,0.,1.]})
        p=NNPreprocessorV3().fit(x)
        test=x.copy();test['event']=1
        num,_=p.transform(test)
        self.assertTrue((num[:,p.cont.index('event')]==0).all())
        self.assertFalse(p.constant[p.cont.index('rare')])

    def test_future_invariance_smooth_clean_and_sampling(self):
        h,meta=fixture(stations=(3110,3112,3107))
        h.loc[(h.station_id==3112)&(h.date=='2025-08-13')&(h.hour==7),'boardings']=10 # normal zero values are not globally removed
        q=pd.DataFrame({'date':pd.to_datetime(['2026-08-15','2026-09-01']),'station_id':[3110,3107],'hour':[7,7]})
        a=make_features_v3(h,q,meta,'2026-06-30')
        np.testing.assert_allclose(a.yoy_smooth15_log_delta,0,atol=1e-6)
        q2=pd.DataFrame({'date':pd.to_datetime(['2026-05-01']),'station_id':[3110],'hour':[7]})
        before=make_features_v3(h,q2,meta,'2026-02-27')
        h.loc[h.date>'2026-02-27',['boardings','alightings']]=1e12
        after=make_features_v3(h,q2,meta,'2026-02-27')
        pd.testing.assert_frame_equal(before,after)
        pairs,origins=sampled_pairs(h,'2026-02-27',1230)
        labels=h.loc[pairs.row]
        self.assertFalse(oneoff_mask(labels).any())
        self.assertTrue((labels.date<=pd.Timestamp('2026-02-27')).all())
        self.assertEqual(pairs.horizon.nunique(),123)

    def test_recent_weights_ignore_early_labels(self):
        folds=np.repeat([1,2,3,4],20);y=np.ones(80)
        p=np.column_stack([y+.1,y-.2]);dates=pd.date_range('2025-01-01',periods=80)
        _,_,a,_=evaluate_ensemble(y,p,folds,['a','b'],[.15,.15,.35,.35],dates)
        y[:40]=100
        _,_,b,_=evaluate_ensemble(y,p,folds,['a','b'],[.15,.15,.35,.35],dates)
        np.testing.assert_allclose(list(a['weights'].values()),list(b['weights'].values()))
        self.assertAlmostEqual(sum(b['weights'].values()),1)
        self.assertEqual(a['selection_rows'],40)

if __name__=='__main__':unittest.main()
