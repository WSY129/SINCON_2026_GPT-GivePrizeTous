"""Anchor audit before any v5 model fitting; no validation labels removed."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from ..features import load_data
from ..holiday_calendar import calendar_features
from ..v4.features import make_features as old_features
from .features import make_features
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/ensemble_v5'

def run_audit():
    dest=OUT/'anchor_v5_audit.json'
    if dest.exists():return json.loads(dest.read_text())
    start=time.time();data=load_data(ROOT/'data/raw');h=data['train'];meta=data['metaData'];report={'folds':{}}
    for fold,origin in [(3,'2025-12-30'),(4,'2026-02-27')]:
        o=pd.Timestamp(origin);q=h.loc[h.date.between(o+pd.Timedelta(days=1),o+pd.Timedelta(days=123))].sort_values(['date','station_id','hour']).reset_index(drop=True)
        x=make_features(h,q,meta,o);old=old_features(h,q,meta,o);truth=np.log1p(q.boardings.to_numpy());cal=calendar_features(q.date,as_of=o)
        assert len(q)==157194 and np.isfinite(x.anchor_v5).all()
        segments={'all':np.ones(len(q),bool),'labor_day':((q.date.dt.month==5)&(q.date.dt.day==1)).to_numpy(),'gyeyang_gyulhyeon':q.station_id.isin([3110,3111]).to_numpy(),'holiday_known_at_origin':cal.is_holiday.to_numpy(bool)}
        for sid in [3127,3226,3107]:segments[str(sid)]=(q.station_id==sid).to_numpy()
        rows={}
        for name,mask in segments.items():
            rows[name]={'rows':int(mask.sum())}
            for version,p in [('v4',old.anchor_v4.to_numpy()),('v5',x.anchor_v5.to_numpy())]:
                err=p[mask]-truth[mask];rows[name][version]={'rmsle':float(np.sqrt(np.mean(err**2))) if mask.any() else None,'log_sse':float(np.sum(err**2))}
        report['folds'][str(fold)]=rows
        x.to_pickle(OUT/f'anchor_audit_fold{fold}.pkl')
        print('ANCHOR_AUDIT',fold,rows['all'],flush=True)
    q=data['test'].sort_values(['date','station_id','hour']).reset_index(drop=True);x=make_features(h,q,meta,'2026-06-30')
    assert (x.delta_yoy_shrunk==0).all(),'Final POST-PRE growth leakage'
    assert x.new_station_weekly_slope.fillna(0).between(0,.01500001).all()
    # Mutation invariance: future observations and permanently excluded break history.
    probe=q.iloc[::1000].copy();base=make_features(h,probe,meta,'2026-06-30')
    mutated=h.copy();mask=mutated.station_id.isin([3110,3111])&(mutated.date<'2025-07-01');mutated.loc[mask,['boardings','alightings']]=9999999
    check=make_features(mutated,probe,meta,'2026-06-30');assert np.allclose(base,check,equal_nan=True)
    future=h.iloc[:18].copy();future['date']=pd.Timestamp('2026-07-01');future[['boardings','alightings']]=9999999
    check=make_features(pd.concat([h,future]),probe,meta,'2026-06-30');assert np.allclose(base,check,equal_nan=True)
    report.update(final_delta_all_zero=True,final_rows=len(x),protected_preopening_mutation_invariant=True,future_mutation_invariant=True,seconds=time.time()-start,note='Anchor improvement is measured, not a condition for altering validation labels. All 157194 rows retained per fold.')
    dest.write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':run_audit()
