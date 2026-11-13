"""Freezes the v21 training panels (folds f1-f4 and the final cutoff) into one npz per cutoff, so training can run on any
machine with numpy + lightgbm + xgboost only (src/v21/podtrain.py).  Writes outputs/ensemble_v21/panel/<cutoff>.npz and
panel_manifest.json (SHA-256, shapes, feature names, cutoffs).  Run: .venv/bin/python -m src.v21.freeze"""
import hashlib, json, time
import numpy as np
from src.v11 import train as T
from .panel import Panel21, COLS21, NAMES21

OUT = T.c8.ROOT / 'outputs/ensemble_v21/panel' if hasattr(T, 'c8') else None


def main():
    from src.v8 import core as c8
    out = c8.ROOT / 'outputs/ensemble_v21/panel'; out.mkdir(parents=True, exist_ok=True)
    panel = Panel21(); man = dict(features=NAMES21, rounds=T.ROUNDS, lr=T.LR, cutoffs=T.CUTOFFS, files={})
    for name in ('f1', 'f2', 'f3', 'f4', 'final'):
        t0 = time.time(); D = T.Data(panel, name)
        Xtr = D.Xtr[:, COLS21].astype(np.float32); Xp = D.Xp.reshape(-1, D.Xp.shape[-1])[:, COLS21].astype(np.float32)
        arrs = dict(Xtr=Xtr, r=D.r.astype(np.float32), w=np.asarray(D.w, np.float32), Xp=Xp, ap=D.ap.astype(np.float32), shape=np.array(D.shape))
        if D.truth is not None:
            arrs['truth'] = D.truth.astype(np.float32)
        p = out / f'{name}.npz'; np.savez(p, **arrs)
        man['files'][name] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), rows_train=int(len(Xtr)), rows_pred=int(len(Xp)), shape=list(map(int, D.shape)),
                                  bytes=p.stat().st_size, seconds=round(time.time() - t0, 1))
        print(name, man['files'][name], flush=True)
    (out / 'panel_manifest.json').write_text(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
