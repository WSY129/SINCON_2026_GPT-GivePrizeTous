#!/usr/bin/env bash
# Order of the stages behind submission_ensemble_v20e.csv.  Run from the project root with the pinned packages
# (reproduce/requirements.txt).  NOT run end to end as one script: stages marked GPU were trained on rented GPU machines
# and their stored outputs are part of this repository (outputs/ensemble_v2, v4, v5, v9/models, v11/models).
# Stages from "v10" down were re-run and compared with the delivered files by reproduce/verify_from_stored.py.
set -euo pipefail
export V11_THREADS=${V11_THREADS:-12}
PY=${PY:-python}

# ---- base generations (GPU for the neural members; hours) -------------------------------------------------------------
$PY -m src.pipeline v2 run   && $PY -m src.pipeline v2 infer          # LightGBM, CatBoost, XGBoost, TabM
$PY -m src.v4.pipeline run   && $PY -m src.v4.pipeline infer          # + decomposition, low-demand, spatio-temporal transformer
$PY -m src.v5.pipeline run   && $PY -m src.v5.pipeline infer
# ---- rule-based layers (CPU, minutes) ------------------------------------------------------------------------------------
$PY -m src.v6.validate && $PY -m src.v6.build                         # blend of v2 / v4 / v5 + stadium and event specialists
$PY -m src.v7.munhak_audit && $PY -m src.v7.gates && $PY -m src.v7.build
$PY -m src.v8.run both
# ---- v9 structural-anchor models (GPU for xgb / cb / neural; LightGBM on CPU) ------------------------------------------
$PY -m src.v9.train cv --models lgb && $PY -m src.v9.train cv --models xgb,cb,tabm,mdn,stg,rkhs
$PY -m src.v9.train finalize && $PY -m src.v9.train final --models lgb,xgb,cb,tabm,mdn,stg && $PY -m src.v9.blend
$PY -m src.v10.run                                                    # verified: identical file
# ---- v11 unique-day panel models (GPU machine, about 5.4 h) -------------------------------------------------------------
$PY -m src.v11.fetch_kbo
$PY -m src.v11.train --cutoffs f1,f2,f3,f4 --models lgb_a,lgb_b,xgb,cb,dec_lgb,dec_xgb,dec_cb,low_tw,low_po,tabm,stg
$PY -m src.v11.train --cutoffs final --seeds 2026,2027 --models lgb_a,lgb_b,xgb,cb,dec_lgb,dec_xgb,dec_cb,low_tw,low_po,tabm,stg
$PY -m src.v11.stack && $PY -m src.v12.run && $PY -m src.v13.run
# ---- retrained tree models (CPU, about 50 minutes on 12 threads) ---------------------------------------------------------
$PY -m src.v14.train --cutoffs f1,f2,f3,f4 --models lgb_c,lgb_d && $PY -m src.v14.train --cutoffs final --seeds 2026,2027 --models lgb_c,lgb_d && $PY -m src.v14.run
$PY -m src.v15.train --cutoffs f1,f2,f3,f4,final --models lgb_h --seeds 2026,2027,2028
$PY -m src.v15.train --cutoffs f1,f2,f3,f4,final --models xgb_h --seeds 2026
$PY -m src.v15.train --cutoffs f1,f2,f3,f4,final --models xgb_i --seeds 2026,2027,2028,2029,2030
# ---- stack and corrections (CPU, about 2 minutes; verified: identical files) -----------------------------------------------
$PY -m src.v15.run && $PY -m src.v16.run && $PY -m src.v17.run && $PY -m src.v19.run
$PY -m src.v20.extras                                                 # writes outputs/ensemble_v20/submission_ensemble_v20e.csv
