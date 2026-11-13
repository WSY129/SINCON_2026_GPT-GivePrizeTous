"""Post-hoc neural feature diagnostics; never edits baseline/deployment outputs."""
from pathlib import Path
import argparse
import json
import sys
import hashlib
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--project',type=Path,default=Path('/Users/hongtaeyang/Desktop/SINCON'))
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args();root=args.project.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root))
from src.features import load_data,make_features,training_matrix
from src.models import NeuralRegressor
from src.metrics import regression_metrics

class ClippedNeuralRegressor(NeuralRegressor):
    def transform(self,x):return np.clip(super().transform(x),-5,5)

cfg=json.loads((root/'configs/default.json').read_text())
data=load_data(root/'data/raw');train=data['train'];meta=data['metaData']
rows=[];drifts=[];losses=[];checks=[]
for fold,origin_str in enumerate(cfg['fold_origins'],1):
    origin=pd.Timestamp(origin_str)
    valid=train.loc[(train.date>origin)&(train.date<=origin+pd.Timedelta(days=123))].copy()
    x,y,anchors=training_matrix(train,meta,origin,123,28,cfg['max_training_rows'],cfg['seed'])
    xv=make_features(train,valid,meta,origin)
    baseline_oof=pd.read_csv(root/f'outputs/baseline/fold_{fold}/oof.csv.gz')
    for variant in ['original','drop_absolute_time','clip_z_5','drop_time_clip_z_5']:
        drop=variant in ['drop_absolute_time','drop_time_clip_z_5']
        xx=x.drop(columns=['date_index','history_days']) if drop else x
        vv=xv.drop(columns=['date_index','history_days']) if drop else xv
        cls=ClippedNeuralRegressor if 'clip' in variant else NeuralRegressor
        model=cls(cfg['seed'],cfg['threads'],'cpu').fit(xx,np.log1p(y))
        pred=np.expm1(np.maximum(0,model.predict(vv))).astype('float64')
        train_pred=np.expm1(np.maximum(0,model.predict(xx))).astype('float64')
        metric=regression_metrics(valid.boardings,pred)
        rows.append({'fold':fold,'variant':variant,'scope':'posthoc_diagnostic_not_new_holdout','train_rmsle':regression_metrics(y,train_pred)['rmsle'],**metric})
        losses.extend({'fold':fold,'variant':variant,'epoch':epoch+1,'training_batch_rmse':loss} for epoch,loss in enumerate(model.losses))
        if variant=='original':
            diff=np.max(np.abs(pred-baseline_oof.mlp.to_numpy()))
            checks.append({'fold':fold,'max_abs_prediction_diff_vs_saved_oof':float(diff)})
            if diff>0.01:raise AssertionError('Original neural reproduction differs materially')
            z=model.transform(vv)[:,:len(model.columns)]
            zt=model.transform(xx)[:,:len(model.columns)]
            for i,col in enumerate(model.columns):
                drifts.append({'fold':fold,'feature':col,'train_min_z':float(np.min(zt[:,i])),'train_max_z':float(np.max(zt[:,i])),'validation_min_z':float(np.min(z[:,i])),'validation_max_z':float(np.max(z[:,i])),'outside_train_range_fraction':float(np.mean((z[:,i]<zt[:,i].min()-1e-6)|(z[:,i]>zt[:,i].max()+1e-6))),'abs_z_gt5_fraction':float(np.mean(np.abs(z[:,i])>5))})
        print(f'fold={fold} {variant}: train={rows[-1]["train_rmsle"]:.6f}, val={metric["rmsle"]:.6f}',flush=True)
    pd.DataFrame(rows).to_csv(out/'neural_ablations.csv',index=False)
    pd.DataFrame(drifts).to_csv(out/'neural_feature_drift.csv',index=False)
    pd.DataFrame(losses).to_csv(out/'neural_training_curves.csv',index=False)
manifest={'project':str(root),'method':'Original seeded MLP refit plus three fixed post-hoc feature diagnostics. Original OOF reproduced. No deployment changes. Fold3 has already been observed and is not a fresh holdout.','original_oof_checks':checks,'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['src/models.py','src/features.py','configs/default.json']}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
