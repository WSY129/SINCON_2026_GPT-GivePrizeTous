"""Inspect trusted local estimators without modifying or retraining them."""
from pathlib import Path
import json
import hashlib
import sys
import joblib
import lightgbm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent

def main():
    result={'library_version':lightgbm.__version__,'models':{},'retrained':False}
    for relative,name in [('outputs/three_tree/model_bundle.joblib','lightgbm_large'),
                          ('outputs/baseline/model_bundle.joblib','lightgbm_small')]:
        path=ROOT/relative
        bundle=joblib.load(path)
        model=bundle['models'][name].estimator
        trees=model.booster_.dump_model()['tree_info']
        result['models'][name]={'bundle_path':relative,'bundle_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                               'parameters':model.get_params(),'n_estimators_actual':model.n_estimators_,
                               'n_iter_actual':model.n_iter_,'num_trees':model.booster_.num_trees(),
                               'best_iteration':model.best_iteration_,'best_score':model.best_score_,
                               'evals_result':model.evals_result_,'feature_count':model.n_features_in_,
                               'training_cutoff':bundle['training_cutoff'],
                               'minimum_tree_leaves':min(t['num_leaves'] for t in trees),
                               'maximum_tree_leaves':max(t['num_leaves'] for t in trees),
                               'total_leaves':sum(t['num_leaves'] for t in trees)}
    (OUT/'saved_model_settings.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
