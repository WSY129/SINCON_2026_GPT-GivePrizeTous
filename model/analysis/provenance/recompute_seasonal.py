"""Independently reproduce the locally created seasonal baseline score.

This does not import the forecasting pipeline or download baseline code.
All features come from observations <= 2026-02-27. Validation labels are used
only in the last scoring step. This is not a Kaggle leaderboard score.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=Path('/Users/hongtaeyang/Desktop/SINCON'))
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'seasonal_score_proof.json')
args=parser.parse_args()
raw=args.project/'data/raw/train.csv'
train=pd.read_csv(raw,parse_dates=['date'])
origin=pd.Timestamp('2026-02-27')
end=pd.Timestamp('2026-06-30')
history=train.loc[train.date<=origin].copy()
valid=train.loc[(train.date>origin)&(train.date<=end)].copy()
history['weekday']=history.date.dt.dayofweek
valid['weekday']=valid.date.dt.dayofweek
keys=['station_id','hour','weekday']
latest=history.sort_values('date').drop_duplicates(keys,keep='last')
lookup=latest[keys+['date','boardings']].rename(columns={'date':'source_date','boardings':'prediction'})
scored=valid.merge(lookup,on=keys,how='left',validate='many_to_one',sort=False)
assert len(scored)==157194 and scored.prediction.notna().all()
assert (scored.source_date<=origin).all()
assert (scored.source_date.dt.dayofweek==scored.date.dt.dayofweek).all()
y=scored.boardings.to_numpy(dtype=np.float64)
p=scored.prediction.to_numpy(dtype=np.float64)
msle=float(np.mean((np.log1p(p)-np.log1p(y))**2))
saved=pd.read_csv(args.project/'outputs/baseline/fold_3/oof.csv.gz',parse_dates=['date'])
merged=scored.merge(saved[['date','station_id','hour','seasonal7']],on=['date','station_id','hour'],validate='one_to_one')
result={'model_origin':'created locally by assistant; not downloaded official baseline','score_kind':'local temporal validation; not Kaggle leaderboard',
        'history_end':str(origin.date()),'validation_start':str(scored.date.min().date()),'validation_end':str(end.date()),
        'validation_rows':len(scored),'validation_days':int(scored.date.nunique()),'stations':int(scored.station_id.nunique()),
        'source_observation_dates':sorted(str(d.date()) for d in pd.to_datetime(scored.source_date.unique())),
        'msle':msle,'rmsle':float(np.sqrt(msle)),'predictions_exactly_match_saved_oof':bool(np.array_equal(merged.prediction,merged.seasonal7)),
        'train_csv_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
assert result['predictions_exactly_match_saved_oof']
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
