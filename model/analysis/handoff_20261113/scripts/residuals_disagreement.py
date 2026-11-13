"""Section 3 of handoff_for_gemini: v19 out-of-fold residuals by station and fold, and the 500 test cells where
the old v10 forecast and the new tree members (LightGBM / XGBoost) disagree most.  Local data only, no leaderboard value.
Run from the project root:  python outputs/handoff_for_gemini/scripts/residuals_disagreement.py"""
import sys; sys.path.insert(0, '.')
import warnings; warnings.simplefilter('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from src.v11 import stack as s11
from src.v15 import run as r15

OUT = Path('outputs/handoff_for_gemini/03_residuals_disagreement')
S = s11.Stack(); folds = S.folds; st = S.panel.stations
names = pd.read_csv('data/raw/metaData.csv').set_index('station_id').station_name
o19 = {k: np.load('outputs/ensemble_v19/oof_v19.npz')[f'fold{k}'].astype(float) for k in folds}
o10 = {k: np.load('outputs/ensemble_v10/oof_v10.npz')[f'fold{k}'].astype(float) for k in folds}

# ---- residual table: station x fold (aligned truth, log1p space; error = forecast - truth) ----
rows = []
tot = sum(((o19[k] - S.truth[k]) ** 2).sum() for k in folds)
for i, sid in enumerate(st):
    r = {'station_id': int(sid), 'station_name': names[int(sid)]}
    sse = 0.0
    for k in folds:
        e = o19[k][:, i] - S.truth[k][:, i]; e10 = o10[k][:, i] - S.truth[k][:, i]
        r[f'rmsle_f{k}'] = round(float(np.sqrt((e ** 2).mean())), 4)
        r[f'bias_f{k}'] = round(float(e.mean()), 4)
        r[f'rmsle_v10_f{k}'] = round(float(np.sqrt((e10 ** 2).mean())), 4)
        sse += (e ** 2).sum()
    r['rmsle_all'] = round(float(np.sqrt(sse / (4 * 123 * 18))), 4)
    r['sse_share_pct'] = round(float(100 * sse / tot), 2)
    E = np.concatenate([(o19[k][:, i] - S.truth[k][:, i]) for k in folds])          # day x hour
    r['worst_hour'] = int(6 + np.argmax((E ** 2).mean(0))); r['worst_hour_rmsle'] = round(float(np.sqrt((E ** 2).mean(0).max())), 4)
    r['bias_sign_same_all_folds'] = bool(len({np.sign(r[f'bias_f{k}']) for k in folds}) == 1 and min(abs(r[f'bias_f{k}']) for k in folds) > 0.02)
    rows.append(r)
res = pd.DataFrame(rows).sort_values('sse_share_pct', ascending=False)
res.to_csv(OUT / 'v19_residuals_by_station_fold.csv', index=False, encoding='utf-8-sig')

# fold x hour and fold x day-kind summaries
fh = []
for k, b in folds.items():
    e = o19[k] - S.truth[k]; kind = np.asarray(b.kind)
    for h in range(18):
        fh.append(dict(fold=k, hour=6 + h, rmsle=round(float(np.sqrt((e[:, :, h] ** 2).mean())), 4), bias=round(float(e[:, :, h].mean()), 4)))
pd.DataFrame(fh).to_csv(OUT / 'v19_residuals_by_fold_hour.csv', index=False)
fk = []
for k, b in folds.items():
    e = o19[k] - S.truth[k]; kind = np.asarray(b.kind)
    for kd in sorted(set(kind)):
        m = kind == kd
        fk.append(dict(fold=k, origin=str(b.origin)[:10], day_kind=kd, days=int(m.sum()), rmsle=round(float(np.sqrt((e[m] ** 2).mean())), 4), bias=round(float(e[m].mean()), 4)))
pd.DataFrame(fk).to_csv(OUT / 'v19_residuals_by_fold_daykind.csv', index=False)
score = {k: round(float(np.sqrt(((o19[k] - S.truth[k]) ** 2).mean())), 5) for k in folds}
score['all'] = round(float(np.sqrt(sum(((o19[k] - S.truth[k]) ** 2).sum() for k in folds) / (4 * 123 * 71 * 18))), 5)
print('v19 aligned OOF', score)

# ---- test disagreement: v10 vs tree members ----
T = S.test; dates = T.dates
def dense(path):
    sub = pd.read_csv(path); flat = np.log1p(sub.boardings.to_numpy())
    return flat[T.row_index].reshape(len(dates), len(st), 18)
v10 = dense('outputs/ensemble_v10/submission_ensemble_v10.csv')
v19 = dense('outputs/ensemble_v19/submission_ensemble_v19.csv'); v19e = dense('outputs/ensemble_v19/submission_ensemble_v19e.csv'); v20e = dense('outputs/ensemble_v20/submission_ensemble_v20e.csv')
mem = {n: r15.load_final(n) for n in ('lgb_h', 'xgb_i', 'xgb_h', 'lgb_d')}
lgb = (mem['lgb_h'] + mem['lgb_d']) / 2; xgb = (mem['xgb_i'] + mem['xgb_h']) / 2; trees = (lgb + xgb) / 2
d = v10 - trees
idx = np.argsort(-np.abs(d).ravel())[:500]
a, s, h = np.unravel_index(idx, d.shape)
ex = T.example_id[T.row_index].reshape(d.shape)
top = pd.DataFrame(dict(rank=np.arange(1, 501), example_id=ex[a, s, h], date=[str(dates[i])[:10] for i in a], weekday=[dates[i].day_name()[:3] for i in a], day_kind=np.asarray(T.kind)[a],
                        station_id=st[s].astype(int), station_name=[names[int(x)] for x in st[s]], hour=h + 6,
                        v10=np.expm1(v10[a, s, h]).round(1), lgb_h=np.expm1(mem['lgb_h'][a, s, h]).round(1), lgb_d=np.expm1(mem['lgb_d'][a, s, h]).round(1),
                        xgb_i=np.expm1(mem['xgb_i'][a, s, h]).round(1), xgb_h=np.expm1(mem['xgb_h'][a, s, h]).round(1),
                        log_v10_minus_trees=d[a, s, h].round(4), log_lgb_minus_xgb=(lgb - xgb)[a, s, h].round(4),
                        v19=np.expm1(v19[a, s, h]).round(1), v19e=np.expm1(v19e[a, s, h]).round(1), v20e=np.expm1(v20e[a, s, h]).round(1),
                        v20e_minus_v19_log=(v20e - v19)[a, s, h].round(4)))
top.to_csv(OUT / 'test_top500_v10_vs_trees.csv', index=False, encoding='utf-8-sig')
# where: by station, date, hour; plus network-wide disagreement levels
ad = np.abs(d)
by_st = pd.DataFrame(dict(station_id=st.astype(int), station_name=[names[int(x)] for x in st], mean_abs_log_diff=ad.mean(axis=(0, 2)).round(4), mean_signed=d.mean(axis=(0, 2)).round(4),
                          cells_over_0_3=(ad > 0.3).sum(axis=(0, 2)), in_top500=np.bincount(s, minlength=len(st)))).sort_values('mean_abs_log_diff', ascending=False)
by_st.to_csv(OUT / 'test_disagreement_by_station.csv', index=False, encoding='utf-8-sig')
by_day = pd.DataFrame(dict(date=[str(x)[:10] for x in dates], weekday=[x.day_name()[:3] for x in dates], day_kind=np.asarray(T.kind), mean_abs_log_diff=ad.mean(axis=(1, 2)).round(4),
                           mean_signed=d.mean(axis=(1, 2)).round(4), in_top500=np.bincount(a, minlength=len(dates)))).sort_values('mean_abs_log_diff', ascending=False)
by_day.to_csv(OUT / 'test_disagreement_by_date.csv', index=False, encoding='utf-8-sig')
by_h = pd.DataFrame(dict(hour=np.arange(6, 24), mean_abs_log_diff=ad.mean(axis=(0, 1)).round(4), mean_signed=d.mean(axis=(0, 1)).round(4), in_top500=np.bincount(h, minlength=18)))
by_h.to_csv(OUT / 'test_disagreement_by_hour.csv', index=False)
# same statistic on the validation folds, with which side was closer to the truth
vf = []
for k in folds:
    tk = {n: r15.load_fold(n, k) for n in mem}; tr = ((tk['lgb_h'] + tk['lgb_d']) / 2 + (tk['xgb_i'] + tk['xgb_h']) / 2) / 2
    dd = o10[k] - tr; big = np.abs(dd) > 0.3; t = S.truth[k]
    vf.append(dict(fold=k, mean_abs_log_diff=round(float(np.abs(dd).mean()), 4), cells_over_0_3=int(big.sum()),
                   rmsle_v10_on_those=round(float(np.sqrt(((o10[k] - t)[big] ** 2).mean())), 4), rmsle_trees_on_those=round(float(np.sqrt(((tr - t)[big] ** 2).mean())), 4),
                   rmsle_v19_on_those=round(float(np.sqrt(((o19[k] - t)[big] ** 2).mean())), 4), share_trees_closer=round(float((np.abs(tr - t) < np.abs(o10[k] - t))[big].mean()), 3)))
pd.DataFrame(vf).to_csv(OUT / 'oof_disagreement_v10_vs_trees_by_fold.csv', index=False)
print('test mean |v10-trees|', round(float(ad.mean()), 4), 'cells >0.3:', int((ad > 0.3).sum()))
print(pd.DataFrame(vf).to_string()); print(by_st.head(10).to_string()); print(top.head(15).to_string())
