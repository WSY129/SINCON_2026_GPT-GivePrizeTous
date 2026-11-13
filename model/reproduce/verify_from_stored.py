"""Re-runs the last stages into a scratch folder and compares the results with the delivered files (nothing in outputs/ is written).

    python -m reproduce.verify_from_stored [--retrain]

Chain checked:  stored model outputs (v10, v11, v14, v15)  ->  v15 stack  ->  v19 corrections  ->  v20e entries.
With --retrain one LightGBM and one XGBoost fit of v15 (fold 4, seed 2026) are trained again and compared with the stored forecasts.
"""
import hashlib
import json
import shutil
import sys
import warnings
from pathlib import Path
import numpy as np

warnings.simplefilter('ignore')
ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    tmp = ROOT / 'reproduce/_scratch'; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True); out = {}          # inside the project: v19 records its base folder relative to the root
    from src.v15 import run as v15
    v15.OUT = tmp / 'v15'; v15.OUT.mkdir(parents=True); v15.main()
    from src.v19 import run as v19
    v19.OUT = tmp / 'v19'; (v19.OUT / 'external').mkdir(parents=True); shutil.copy(ROOT / 'outputs/ensemble_v19/external/mart_closure_calendar.csv', v19.OUT / 'external'); v19.BASE = tmp / 'v15'; v19.main()
    from src.v20 import extras as v20
    v20.OUT = tmp / 'v20'; v20.OUT.mkdir(parents=True); v20.BASE = tmp / 'v19'; v20.main()
    for name, new, old in (('v15 submission', tmp / 'v15/submission_ensemble_v15.csv', ROOT / 'outputs/ensemble_v15/submission_ensemble_v15.csv'),
                           ('v15 out-of-fold', None, None), ('v19 submission', tmp / 'v19/submission_ensemble_v19.csv', ROOT / 'outputs/ensemble_v19/submission_ensemble_v19.csv'),
                           ('v20e submission', tmp / 'v20/submission_ensemble_v20e.csv', ROOT / 'outputs/ensemble_v20/submission_ensemble_v20e.csv')):
        if new is None:
            a = np.load(tmp / 'v15/oof_v15.npz'); b = np.load(ROOT / 'outputs/ensemble_v15/oof_v15.npz'); out[name] = dict(identical=bool(all(np.array_equal(a[k], b[k]) for k in b.files)))
        else:
            out[name] = dict(identical=sha(new) == sha(old), sha256=sha(new))
    if '--retrain' in sys.argv:
        from src.v11 import train as T
        from src.v15 import train as tr
        from src.v15.panel import Panel15
        D = T.Data(Panel15(), 'f4')
        for model in ('lgb_h', 'xgb_i'):
            pred, curve = tr.fit(D, model, 2026); old = np.load(ROOT / f'outputs/ensemble_v15/models/f4__{model}__2026.npz')['pred']
            out[f'retrain {model} fold 4 seed 2026'] = dict(identical=bool(np.array_equal(pred, old)), max_abs_log_difference=float(np.abs(pred - old).max()), valid_rmsle_last=round(float(curve[-1]), 5))
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(out, indent=1)); (ROOT / 'reproduce/verify_result.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
