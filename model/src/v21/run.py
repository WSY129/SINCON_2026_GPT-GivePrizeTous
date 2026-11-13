"""Builds submission_ensemble_v21.csv = v19 + a correction from the hourly BUS boardings next to event venues (CPU, seconds).

    python -m src.bus.stcis                     (once: bus boardings of 18 stops, 2025-08 .. 2026-10, from https://www.stcis.go.kr)
    python -m src.v21.run

For station s, day d, hour h:  bus excess E = boardings at the stops next to the station - their median on days of the same
type within five weeks.  A cell is flagged when E(h) + lam x E(h+1) >= THR and the boardings are at least RATIO x the
median (bus taps come a few minutes after the subway taps, hence the next hour).  On flagged cells the forecast is moved
towards  target = typical forecast of that station / day type / hour + k_s x (E(h) + lam x E(h+1)),  new = (1 - w) cur + w log1p(target).
k_s is fitted per station on the other folds (date-blocked), a station is used only if the correction lowers its error on
those folds, and lam, THR, w are picked on the other folds as well.  Kept only if the all-fold score improves and no fold
gets worse.  The bus boardings of the target day are a realised fact of that day, used the same way in the folds and in
the test period; no subway ridership of the evaluation period is read.
"""
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from src.v8 import core as c8
from src.v11 import stack as s11
from src.v13 import run as R

OUT = c8.ROOT / 'outputs/ensemble_v21'
BASE = c8.ROOT / 'outputs/ensemble_v19'
BUS = c8.ROOT / 'data/external_bus/stcis'
TOL = 1e-5
GRID = dict(lam=(0.0, 0.3), thr=(20.0, 50.0), ratio=(2.0, 3.0), w=(0.5, 1.0))
DAY_CLIP, DAY_SMOOTH = 0.5, 20.0                                     # day step: clip of the log ratio, counts added to both sums
K_MAX = 30.0


def load_bus(stations):
    frames = [pd.read_csv(p, dtype=dict(date=str)) for p in sorted(BUS.glob('*.csv'))]
    b = pd.concat(frames); b = b[(b.hour >= 6) & (b.hour <= 23)]
    g = b.groupby(['station_id', 'date', 'hour']).ride.sum().reset_index()
    days = pd.date_range(g.date.min(), g.date.max()); di = {str(d.date()): i for i, d in enumerate(days)}; sl = sorted(g.station_id.unique())
    X = np.full((len(sl), len(days), 18), np.nan)
    X[np.searchsorted(sl, g.station_id), g.date.map(di).to_numpy(), g.hour.to_numpy() - 6] = g.ride.to_numpy()
    for i, s in enumerate(sl):                                             # days with no row at all are missing, not zero
        empty = ~np.isfinite(X[i]).any(1); X[i][~empty] = np.nan_to_num(X[i][~empty])
    kinds = np.asarray(c8.day_kinds(days, c8.FINAL)); grp = R.day_group(kinds)
    base = np.full(X.shape, np.nan)
    for i in range(len(sl)):
        for d in range(len(days)):
            m = [e for e in range(max(d - 35, 0), min(d + 36, len(days))) if e != d and grp[e] == grp[d] and np.isfinite(X[i, e]).all()]
            if len(m) >= 3:
                base[i, d] = np.median(X[i, m], 0)
    return dict(stations=[int(s) for s in sl], days=days, di=di, X=X, base=base)


def main():
    S = s11.Stack(); folds = S.folds; test = S.test; st = np.asarray(S.panel.stations).astype(int); sidx = {int(s): i for i, s in enumerate(st)}
    names = pd.read_csv(c8.ROOT / 'data/raw/metaData.csv').set_index('station_id').station_name
    cur = {k: np.load(BASE / 'oof_v19.npz')[f'fold{k}'].astype(float) for k in folds}
    sub19 = pd.read_csv(BASE / 'submission_ensemble_v19.csv'); cur_t = np.log1p(sub19.boardings.to_numpy())[test.row_index].reshape(test.pred.shape)
    blocks = {**folds, 'test': test}; zz = {**cur, 'test': cur_t}; bus = load_bus(st)
    keep = lambda k: {j: np.asarray((folds[j].dates < folds[k].dates.min()) | (folds[j].dates > folds[k].dates.max())) for j in folds if j != k}

    def cells(n, lam, thr, ratio):
        """flagged cells of block n: list per bus station of (day idx, hour idx, Eeff, typical count)"""
        b = blocks[n]; z = zz[n]; grp = R.day_group(b.kind); out = {}
        for bi, s in enumerate(bus['stations']):
            j = sidx[s]; rows = []; flag = np.zeros((len(b.dates), 18), bool); eff = np.zeros((len(b.dates), 18))
            for i, d in enumerate(pd.DatetimeIndex(b.dates)):
                di = bus['di'].get(str(d.date()))
                if di is None or not np.isfinite(bus['X'][bi, di]).all() or not np.isfinite(bus['base'][bi, di]).all():
                    continue
                x, bs = bus['X'][bi, di], bus['base'][bi, di]; e = x - bs; e1 = np.r_[e[1:], 0.0]
                ef = np.maximum(e, 0) + lam * np.maximum(e1, 0); big = (x >= ratio * (bs + 5)) | (np.r_[x[1:], 0] >= ratio * (np.r_[bs[1:], 0] + 5))
                flag[i] = (ef >= thr) & big; eff[i] = ef
            if not flag.any():
                continue
            typ = np.zeros((len(b.dates), 18))
            for g in range(4):
                dm = grp == g
                if not dm.any():
                    continue
                ok = dm & ~flag.any(1); ref = np.median(np.expm1(z[ok, j]), 0) if ok.sum() >= 3 else np.median(np.expm1(z[dm, j]), 0)
                typ[dm] = ref[None, :]
            ii, hh = np.nonzero(flag); out[s] = (ii, hh, eff[ii, hh], typ[ii, hh])
        return out

    def fit_k(parts):
        """parts: list of (block, cells, day mask) -> k per station and the stations whose error falls with it"""
        num, den = {}, {}
        for n, cl, m in parts:
            for s, (ii, hh, ef, ty) in cl.items():
                sel = m[ii]; y = np.expm1(S.truth[n][ii[sel], sidx[s], hh[sel]]) - ty[sel]
                num[s] = num.get(s, 0.0) + float((ef[sel] * y).sum()); den[s] = den.get(s, 0.0) + float((ef[sel] ** 2).sum())
        return {s: float(np.clip(num[s] / den[s], 0, K_MAX)) for s in num if den[s] > 0}

    def apply(n, cl, k, w, use):
        z = zz[n].copy()
        for s, (ii, hh, ef, ty) in cl.items():
            if s in k and s in use:
                j = sidx[s]; z[ii, j, hh] = (1 - w) * z[ii, j, hh] + w * np.log1p(np.maximum(ty + k[s] * ef, 0))
        return z

    def sse(n, z, m=None):
        r = (z - S.truth[n]) ** 2
        return float(r[m].sum() if m is not None else r.sum())

    def station_gain(n, cl, k, w, m):
        g = {}
        for s, (ii, hh, ef, ty) in cl.items():
            if s not in k:
                continue
            sel = m[ii]; j = sidx[s]; t = S.truth[n][ii[sel], j, hh[sel]]; a = zz[n][ii[sel], j, hh[sel]]; b = (1 - w) * a + w * np.log1p(np.maximum(ty[sel] + k[s] * ef[sel], 0))
            g[s] = float(((a - t) ** 2).sum() - ((b - t) ** 2).sum())
        return g

    cache = {}
    def C(n, lam, thr, ratio):
        key = (n, lam, thr, ratio)
        if key not in cache:
            cache[key] = cells(n, lam, thr, ratio)
        return cache[key]

    def choose(train):
        """train: {fold j: day mask}; returns the setting with the lowest training error and its k / stations"""
        best = None
        for lam, thr, ratio, w in itertools.product(*GRID.values()):
            parts = [(j, C(j, lam, thr, ratio), m) for j, m in train.items()]; k = fit_k(parts)
            gain = {}
            for j, cl, m in parts:
                for s, v in station_gain(j, cl, k, w, m).items():
                    gain[s] = gain.get(s, 0.0) + v
            use = {s for s, v in gain.items() if v > 0}; total = sum(gain[s] for s in use)
            if best is None or total > best[0]:
                best = (total, dict(lam=lam, thr=thr, ratio=ratio, w=w), k, use, gain)
        return best

    new = {}; picked = {}
    for kf in folds:
        tr = keep(kf); total, prm, k, use, gain = choose(tr)
        new[kf] = apply(kf, C(kf, prm['lam'], prm['thr'], prm['ratio']), k, prm['w'], use)
        picked[kf] = dict(setting=prm, stations_used={f'{s} {names[s]}': round(k[s], 2) for s in sorted(use)}, training_gain=round(total, 1))
    a, b = S.score(cur), S.score(new); ok = b['all_oof'] < a['all_oof'] - 1e-7 and all(b[f'fold_{k}'] <= a[f'fold_{k}'] + TOL for k in folds)
    print(f"bus     {a['all_oof']:.5f} -> {b['all_oof']:.5f}  " + ' '.join(f"F{k} {b[f'fold_{k}'] - a[f'fold_{k}']:+.6f}" for k in folds) + ('  ADOPTED' if ok else '  rejected'), flush=True)
    for kf, v in picked.items():
        print('  fold', kf, v['setting'], v['stations_used'])
    total, prm, k, use, gain = choose({j: np.ones(len(folds[j].dates), bool) for j in folds})
    cl_t = C('test', prm['lam'], prm['thr'], prm['ratio']); t = apply('test', cl_t, k, prm['w'], use) if ok else cur_t.copy()
    td = pd.DatetimeIndex(test.dates); changed = []
    for s, (ii, hh, ef, ty) in cl_t.items():
        if s in use and ok:
            j = sidx[s]
            for d in sorted(set(ii)):
                hs = hh[ii == d]; changed.append(dict(station=f'{s} {names[s]}', date=str(td[d].date()), hours=[int(h) + 6 for h in hs],
                                                     before=[int(round(np.expm1(cur_t[d, j, h]))) for h in hs], after=[int(round(np.expm1(t[d, j, h]))) for h in hs]))
    out = dict(information='local OOF only; nested and date-blocked on aligned truth; no leaderboard value and no subway ridership of the evaluation period is used',
               protocol='for fold k the bus correction (k per station, stations used, lam / threshold / ratio / weight) is fitted on the other folds without the days that fall inside fold k; kept only if the all-fold score improves and no fold gets worse',
               bus=dict(stops_per_station={str(s): int(len([p for p in BUS.glob('*.csv') if pd.read_csv(p, nrows=1).station_id.iloc[0] == s])) for s in bus['stations']}, days=[str(bus['days'][0].date()), str(bus['days'][-1].date())]),
               step=dict(before=a, after=b, adopted=bool(ok), folds=picked, test_setting=prm, test_k={f'{s} {names[s]}': round(k[s], 2) for s in sorted(use)}, test_cells_changed=changed),
               generations={'v19': dict(aligned=a, raw_truth=S.score(cur, raw=True)), 'v21': dict(aligned=S.score(new if ok else cur), raw_truth=S.score(new if ok else cur, raw=True))})
    # ---- second step: day-level bus anomaly (all stations with a stop; days with a flagged surge are left out) --------------
    cur2 = new if ok else cur; cur2_t = t; flagged = {}
    for n in blocks:
        f = np.zeros((len(blocks[n].dates), len(st)), bool)
        for s, (ii, hh, ef, ty) in C(n, prm['lam'], prm['thr'], prm['ratio']).items():
            f[ii, sidx[s]] = True
        flagged[n] = f

    def day_x(n):
        b = blocks[n]; x = np.zeros((len(b.dates), len(st)))
        for bi, s in enumerate(bus['stations']):
            j = sidx[s]
            for i, d in enumerate(pd.DatetimeIndex(b.dates)):
                di = bus['di'].get(str(d.date()))
                if di is None or not np.isfinite(bus['X'][bi, di]).all() or not np.isfinite(bus['base'][bi, di]).all() or flagged[n][i, j]:
                    continue
                x[i, j] = np.clip(np.log((bus['X'][bi, di].sum() + DAY_SMOOTH) / (bus['base'][bi, di].sum() + DAY_SMOOTH)), -DAY_CLIP, DAY_CLIP)
        return x
    XD = {n: day_x(n) for n in blocks}; zz2 = {**cur2, 'test': cur2_t}
    fitb = lambda parts: float(sum((x[:, :, None] * r).sum() for x, r in parts) / max(sum(((x ** 2).sum() * 18) for x, r in parts), 1e-9))
    new2 = {}; betas = {}
    for kf in folds:
        m = keep(kf); betas[kf] = fitb([(XD[j][m[j]], (S.truth[j] - zz2[j])[m[j]]) for j in m]); new2[kf] = np.maximum(zz2[kf] + betas[kf] * XD[kf][:, :, None], 0)
    a2, b2 = S.score(cur2), S.score(new2); ok2 = b2['all_oof'] < a2['all_oof'] - 1e-7 and all(b2[f'fold_{k}'] <= a2[f'fold_{k}'] + TOL for k in folds)
    print(f"busday  {a2['all_oof']:.5f} -> {b2['all_oof']:.5f}  " + ' '.join(f"F{k} {b2[f'fold_{k}'] - a2[f'fold_{k}']:+.6f}" for k in folds) + ('  ADOPTED' if ok2 else '  rejected'), '| slope nested', {k: round(v, 3) for k, v in betas.items()}, flush=True)
    beta_t = fitb([(XD[j], S.truth[j] - zz2[j]) for j in folds])
    out['step_day'] = dict(before=a2, after=b2, adopted=bool(ok2), slope_nested=betas, slope_test=round(beta_t, 4), clip=DAY_CLIP, stations_with_bus=len(bus['stations']))
    if ok2:
        cur2 = new2; t = np.maximum(cur2_t + beta_t * XD['test'][:, :, None], 0)
    out['generations']['v21'] = dict(aligned=S.score(cur2), raw_truth=S.score(cur2, raw=True))
    final = cur2
    np.savez_compressed(OUT / 'oof_v21.npz', **{f'fold{k}': final[k].astype(np.float32) for k in folds})
    flat = np.empty(t.size); flat[test.row_index] = np.maximum(t, 0).reshape(-1); sub = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
    assert len(sub) == 157194 and np.isfinite(sub.boardings).all() and (sub.boardings >= 0).all() and (sub.example_id.to_numpy() == sub19.example_id.to_numpy()).all()
    path = OUT / 'submission_ensemble_v21.csv'; sub.to_csv(path, index=False, float_format='%.8f'); ref = np.log1p(sub19.boardings.to_numpy())
    out['submission'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(sub)), rows_changed_vs_v19=int((np.abs(flat - ref) > 1e-9).sum()))
    (OUT / 'gates_v21.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print('v21', {kk: round(v, 5) for kk, v in out['generations']['v21']['aligned'].items()}, '| test cells changed', out['submission']['rows_changed_vs_v19'])
    return out


if __name__ == '__main__':
    main()
