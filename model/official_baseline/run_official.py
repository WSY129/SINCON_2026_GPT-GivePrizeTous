"""Execute the organizer's preserved code and independently measure its RMSLE."""
from pathlib import Path
import os
import sys
import json
import hashlib
import runpy
import contextlib
import io
import argparse
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=Path('/Users/hongtaeyang/Desktop/SINCON'))
args=parser.parse_args()
root=Path(__file__).resolve().parent
run=root/'run';reports=root/'reports'
run.mkdir(exist_ok=True);reports.mkdir(exist_ok=True)
data_link=run/'data'
if not data_link.exists():data_link.symlink_to(args.project/'data/raw',target_is_directory=True)
source=root/'source/official_baseline.py'
captured=source.read_text()
checksum=2166136261
for char in captured:checksum=((checksum^ord(char))*16777619)&0xffffffff
assert len(captured)==1788 and checksum==1627979271, 'Preserved source differs from captured organizer code'
stdout=io.StringIO();old=Path.cwd()
try:
    os.chdir(run)
    with contextlib.redirect_stdout(stdout):scope=runpy.run_path(str(source))
finally:os.chdir(old)
(reports/'execution.log').write_text(stdout.getvalue())
print(stdout.getvalue())
valid=scope['valid'];pred=np.asarray(scope['valid_pred'],dtype='float64')
y=valid.boardings.to_numpy(dtype='float64')
exact=float(np.sqrt(np.mean(np.square(np.log1p(pred)-np.log1p(y)))))
assert abs(exact-float(scope['rmsle'](pred,y)))<1e-14
assert round(exact,5)==0.30770
valid_out=valid[['date','station_id','hour','boardings']].copy();valid_out['prediction']=pred
valid_out.to_csv(reports/'validation_predictions.csv.gz',index=False,compression='gzip')
sys.path.insert(0,str(args.project))
from src.validate_submission import validate_submission
validation=validate_submission(run/'submission.csv',args.project/'data/raw/test.csv',args.project/'data/raw/sample_submission.csv',scope='test')
(reports/'submission_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
# Same-cutoff comparison with our simple seasonal copy, not an official baseline claim.
history=scope['train'].loc[scope['train'].date<='2026-02-28'].copy()
history['dow']=pd.to_datetime(history.date).dt.dayofweek
queries=valid.copy();queries['dow']=pd.to_datetime(queries.date).dt.dayofweek
keys=['station_id','hour','dow']
last=history.sort_values('date').drop_duplicates(keys,keep='last')[keys+['boardings']].rename(columns={'boardings':'seasonal_pred'})
joined=queries.merge(last,on=keys,validate='many_to_one',sort=False)
assert len(joined)==len(valid) and joined.seasonal_pred.notna().all()
same_cutoff=float(np.sqrt(np.mean((np.log1p(joined.seasonal_pred)-np.log1p(joined.boardings))**2)))
measure={
 'source_url':'https://www.kaggle.com/competitions/sincon-2026/discussion/757952',
 'author':'jungjunu1 (Competition Host)',
 'post_title':'[운영진] 시작 코드(Baseline) 공유 — 최근 8주 같은 요일·시간대 평균',
 'post_date_kst':'2026-11-11 17:38:47','retrieved_date_kst':'2026-11-11',
 'retrieval_method':'NAVER Whale UI; official discussion code block copied from rendered page; no separate downloadable notebook attachment observed',
 'source_preserved_without_algorithm_or_date_changes':True,
 'browser_capture_verification':{'characters':len(captured),'lines':captured.count('\n'),'fnv1a_utf16':checksum,'matches_original_capture':True},
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'training_window':['2026-01-05','2026-02-28'],'training_window_days_inclusive':55,
 'validation_window':['2026-03-01','2026-06-30'],'validation_days':122,'validation_rows':len(valid),
 'measured_local_rmsle':exact,'measured_local_rmsle_5dp':f'{exact:.5f}',
 'organizer_reported_local_rmsle':0.30770,'matches_organizer_reported_5dp':True,
 'organizer_reported_public_rmsle':0.28264,'public_score_independently_measured':False,'kaggle_submission_performed':False,
 'final_profile_window':['2026-05-04','2026-06-30'],'final_profile_days_inclusive':58,
 'submission_rows':len(scope['submission']),'submission_sha256':hashlib.sha256((run/'submission.csv').read_bytes()).hexdigest(),
 'comparison_same_cutoff_seasonal_copy_rmsle':same_cutoff,
 'earlier_0_291005_definition':'assistant-created seasonal7 at cutoff2026-02-27; validation2026-02-28~06-30; not the official baseline',
 'train_csv_sha256':hashlib.sha256((args.project/'data/raw/train.csv').read_bytes()).hexdigest(),
 'numpy_version':np.__version__,'pandas_version':pd.__version__
}
(reports/'measurement.json').write_text(json.dumps(measure,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(measure,ensure_ascii=False,indent=2))
