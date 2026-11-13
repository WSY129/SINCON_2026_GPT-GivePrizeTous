"""Independently check downloaded v3 artifacts and preservation manifest."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ensemble_v3'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(name):
    return json.loads((OUT / name).read_text())

cfg = read('config_resolved.json')
fingerprint = read('run_fingerprint.json')
assert all(sha(ROOT / 'data/raw' / n) == h for n, h in fingerprint['inputs'].items())
packaging_metadata = []
for n, h in fingerprint['source_sha256'].items():
    if n.startswith('._'):
        # macOS tar AppleDouble sidecars are not executable source. The exact
        # archived bytes remain available separately, without polluting src/.
        path = OUT / 'training_source_snapshot/src' / n
        packaging_metadata.append(n)
    else:
        path = ROOT / 'src' / n
    assert sha(path) == h, n
assert sha(ROOT / 'configs/ensemble_v3.json') == fingerprint['config_sha256']
oof = pd.read_csv(OUT / 'oof_predictions.csv.gz')
assert len(oof) == 628776
assert not oof.duplicated(['fold', 'date', 'station_id', 'hour']).any()
assert oof.groupby('fold').size().eq(157194).all()
assert oof.horizon.between(1, 123).all()
values = oof[cfg['models'] + ['ensemble', 'true_log']].to_numpy()
assert np.isfinite(values).all() and (values >= 0).all()
report = read('ensemble_report.json')
weights = np.array([report['weights'][m] for m in cfg['models']])
assert weights.min() >= -1e-12 and abs(weights.sum() - 1) < 1e-9
reconstructed = oof[cfg['models']].to_numpy() @ weights
assert np.max(np.abs(reconstructed - oof.ensemble)) < 1e-8
sq = (oof.true_log - oof.ensemble) ** 2
scores = {
    'oof_rmsle': float(np.sqrt(sq.mean())),
    'fold34_rmsle': float(np.sqrt(sq[oof.fold >= 3].mean())),
    'weighted_rmsle': float(np.sqrt(sum(w * sq[oof.fold == f].mean() for f, w in enumerate(cfg['fold_weights'], 1)))),
}
assert all(abs(v - report['ensemble'][k]) < 1e-9 for k, v in scores.items())
test = pd.read_csv(ROOT / 'data/raw/test.csv')
submission = pd.read_csv(OUT / 'submission_ensemble_v3.csv')
assert len(submission) == len(test) == 157194
assert list(submission) == ['example_id', 'boardings']
assert submission.example_id.equals(test.example_id) and submission.example_id.is_unique
assert np.isfinite(submission.boardings).all() and (submission.boardings >= 0).all()
models = list(OUT.glob('final/*/*/model.joblib'))
assert len(models) == 8
families = {m: np.mean([np.load(OUT / f'final/{m}/{seed}/prediction_log.npy') for seed in cfg['final_seeds']], axis=0) for m in cfg['models']}
expected_log = sum(report['weights'][m] * pred for m, pred in families.items())
max_log_delta = float(np.max(np.abs(np.log1p(submission.boardings) - expected_log)))
# CPU reload / CUDA inference can differ at floating-point tolerance.
assert max_log_delta < 2e-5
preserved = read('preserved_v2_sha256.json')
if 'files' in preserved:
    preserved = preserved['files']
changed = [n for n, h in preserved.items() if not (ROOT / n).exists() or sha(ROOT / n) != h]
assert not changed, changed
preservation = {'checked_files': len(preserved), 'unchanged': True, 'changed_files': changed}
(OUT / 'preservation_check.json').write_text(json.dumps(preservation, indent=2))
result = {
    'passed': True, 'input_source_config_hashes_match': True,
    'packaging_metadata_verified_from_snapshot': len(packaging_metadata),
    'oof_rows': len(oof), 'recent_oof_rows': int((oof.fold >= 3).sum()),
    'final_model_count': len(models), 'recomputed_scores': scores,
    'submission_rows': len(submission), 'ids_in_test_order': True,
    'max_submission_vs_training_log_difference': max_log_delta,
    'preserved_file_count': len(preserved), 'submission_sha256': sha(OUT / 'submission_ensemble_v3.csv'),
}
(OUT / 'artifact_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
