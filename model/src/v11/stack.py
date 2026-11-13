"""Stacking, station gates and station specialists of ensemble_v11; builds submission_ensemble_v11.csv.

    python -m src.v11.stack            # needs outputs/ensemble_v11/models (pod run) and outputs/ensemble_v10/oof_v10.npz

Members: the v10 forecast and every v11 model at a checkpoint chosen from the other folds' validation curves.
Weights: ridge towards equal weights on the simplex, fitted on the other folds (nested).  A station keeps the v10
forecast when the stack is worse than v10 on that station in the fitting folds (station gate).  Specialists
(separate models for single stations) replace the stack on their station only if they beat it in the fitting folds.
All fits use aligned truth (see src.v10.run); both aligned and raw scores are reported.  No leaderboard value is used.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.v8 import core as c8

OUT = c8.ROOT / 'outputs/ensemble_v11'
MODELS = OUT / 'models'
FOLDS = {1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4'}
REPORT = [3127, 3107, 3226, 3754, 3122, 3227, 3111, 3201, 3217, 3208, 3108, 3759, 3139, 3209, 3137, 3109, 3136]
RIDGE = [0.0, 0.0005, 0.002, 0.01]


def model_names(seed=2026):
    names = sorted({p.stem.split('__')[1] for p in MODELS.glob(f'f[1-4]__*__{seed}.npz')})
    return [n for n in names if all((MODELS / f'{f}__{n}__{seed}.npz').exists() for f in FOLDS.values())]


def curves(name, seed=2026):
    return {k: json.loads((MODELS / f'{f}__{name}__{seed}.json').read_text())['valid_rmsle'] for k, f in FOLDS.items()}


def strict_index(name, k=None, seed=2026):
    """Checkpoint index for fold k from the other folds' curves; k=None -> for the final model (folds 3 and 4)."""
    c = curves(name, seed)
    use = [3, 4] if k is None else [j for j in c if j != k]
    return int(round(np.mean([int(np.argmin(c[j])) for j in use])))


def load_fold(name, k, seed=2026):
    d = np.load(MODELS / f'{FOLDS[k]}__{name}__{seed}.npz')
    return np.maximum(d['anchor'] + d['pred'][min(strict_index(name, k, seed), d['pred'].shape[0] - 1)], 0).astype(float)


def load_final(name):
    out = []
    for p in sorted(MODELS.glob(f'final__{name}__*.npz')):
        d = np.load(p)
        out.append(np.maximum(d['anchor'] + d['pred'][min(strict_index(name), d['pred'].shape[0] - 1)], 0))
    return np.mean(out, 0).astype(float) if out else None


def ridge_weights(E, lam):
    """Simplex weights minimising mean squared error + lam * |w - uniform|^2 (E: rows x members of errors)."""
    m = E.shape[1]; G = E.T @ E / len(E); u = np.full(m, 1.0 / m)
    f = lambda w: w @ G @ w + lam * ((w - u) ** 2).sum()
    res = minimize(f, u, jac=lambda w: 2 * G @ w + 2 * lam * (w - u), method='SLSQP', bounds=[(0, 1)] * m,
                   constraints=[dict(type='eq', fun=lambda w: w.sum() - 1, jac=lambda w: np.ones(m))], options=dict(maxiter=500, ftol=1e-13))
    w = np.clip(res.x, 0, None)
    return w / w.sum()


class Stack:
    def __init__(self):
        self.panel, self.folds, self.test = c8.load_blocks()
        p = self.panel
        f = pd.read_csv(c8.SHARE / 'ictr_to_train_factor_station_hour.csv'); logf = np.zeros((p.S, p.H))
        logf[np.searchsorted(p.stations, f.station_id.to_numpy()), f.hour.to_numpy() - 6] = np.log(f.factor_boardings.to_numpy())
        self.logf = logf
        self.early = {k: b.origin < '2026-01-01' for k, b in self.folds.items()}
        self.jump = {k: (np.where((b.dates.year == 2026)[:, None, None], logf[None], 0.0) if self.early[k] else np.zeros(b.truth.shape)) for k, b in self.folds.items()}
        self.raw = {k: b.truth.copy() for k, b in self.folds.items()}
        self.truth = {k: self.raw[k] + self.jump[k] for k in self.folds}
        o = np.load(c8.ROOT / 'outputs/ensemble_v10/oof_v10.npz')
        self.names = model_names()
        self.members = ['v10'] + self.names
        self.pred = {k: {'v10': o[f'fold{k}'].astype(float), **{n: load_fold(n, k) for n in self.names}} for k in self.folds}

    def score(self, pred, raw=False):
        """raw=True: the forecasts as they are against the unconverted truth (folds with a 2025 origin then carry the
        aggregation break of their 2026 rows as error); raw=False: against aligned truth."""
        t = self.raw if raw else self.truth
        mse = {k: float(np.mean((pred[k] - t[k]) ** 2)) for k in self.folds}
        n = {k: t[k].size for k in self.folds}
        out = dict(all_oof=float(np.sqrt(sum(mse[k] * n[k] for k in mse) / sum(n.values()))), weighted_4fold=float(np.sqrt(sum(c8.FOLD_WEIGHTS[k] * mse[k] for k in mse))),
                   fold_3_4=float(np.sqrt((mse[3] * n[3] + mse[4] * n[4]) / (n[3] + n[4]))))
        out.update({f'fold_{k}': float(np.sqrt(v)) for k, v in mse.items()})
        return out

    def station_sse(self, pred, folds):
        return sum(((pred[k] - self.truth[k]) ** 2).sum(axis=(0, 2)) for k in folds)

    def station_rmse(self, pred, sid):
        s = int(np.searchsorted(self.panel.stations, sid))
        e = np.concatenate([(pred[k][:, s] - self.truth[k][:, s]).ravel() for k in self.folds])
        return float(np.sqrt(np.mean(e ** 2)))

    def errors(self, folds, members):
        return np.concatenate([np.stack([(self.pred[k][m] - self.truth[k]).ravel() for m in members], 1) for k in folds])

    def blend(self, k, w, members):
        return sum(wi * self.pred[k][m] for wi, m in zip(w, members))

    def fit_apply(self, train_folds, k, lam, members, gate=True):
        w = ridge_weights(self.errors(train_folds, members), lam)
        out = self.blend(k, w, members)
        off = np.zeros(self.panel.S, bool)
        if gate:                                                         # station gate decided on the fitting folds only
            stacked = {j: self.blend(j, w, members) for j in train_folds}
            off = self.station_sse(stacked, train_folds) > self.station_sse({j: self.pred[j]['v10'] for j in train_folds}, train_folds)
            out[:, off] = self.pred[k]['v10'][:, off]
        return out, w, off


# ------------------------------------------------------------------------------------------ specialists
def specialist_forecasts(refresh=False):
    """Fold and final forecasts of every specialist group (trained locally on CPU), cached in specialists.npz."""
    from .panel import Panel11
    from . import special
    path = OUT / 'specialists.npz'
    if path.exists() and not refresh:
        return dict(np.load(path))
    P = Panel11(); out = {}
    for name, origin in list(zip(FOLDS.values(), c8.ORIGINS)) + [('final', c8.FINAL)]:
        d = P.build(origin)
        for g, stations in special.GROUPS.items():
            out[f'{name}__{g}'] = special.fit_group(P, d, stations).astype(np.float32)
    np.savez_compressed(path, **out)
    return out


def main():
    from src.v7 import core as core7
    from src.v8 import actual
    from src.v8.run import block_events
    from src.v10 import games as g10
    from . import special
    S = Stack(); p = S.panel; folds = S.folds; test = S.test
    cfg = json.loads((c8.ROOT / 'configs/ensemble_v11.json').read_text()); cfg10 = g10.config()
    schedule = core7.schedule_v7()
    events = cfg10.get('station_events', []) + cfg.get('station_events', [])
    extra = [(sid, [pd.Timestamp(d) for d in e['dates'] + e['past_dates']]) for e in events for sid in e['stations']]
    for b in list(folds.values()) + [test]:
        b.blocked[:] = False; block_events(b, schedule, extra=extra)
    keep_v10 = {k: np.broadcast_to(b.blocked[:, :, None], b.truth.shape).copy() for k, b in folds.items()}
    # Munhak and Bucheon are decided by station gate / specialists (the new models see attendance and end time),
    # except Munhak's non-league days (concerts, listed live-audience games): v10 has explicit lifts for those.
    m_idx = int(np.searchsorted(p.stations, 3127))
    # Stations younger than 400 days keep v10 (which carries the growth drift).  Their only observations of the test
    # months are the first weeks after opening, so a model with day-of-year terms reads growth as summer seasonality:
    # the new LightGBM puts July 2026 at 3107 0.16 log below the June 2026 actual.  Folds cannot show this.
    young = lambda b: (b.O - p.first < 400) & (p.first > 0)
    for k, b in folds.items():
        keep_v10[k][:, young(b)] = True
    listed = {pd.Timestamp(e['date']) for e in cfg10.get('munhak_events', [])}
    # v10 only has a lift for a concert it knew about at that origin; where it did not, the new models' event flag is used
    munhak_days_at = lambda origin: listed | {pd.Timestamp(e['date']) for e in schedule.venue_events('munhak_main') if str(e['known_since'])[:10] <= str(origin)[:10]}
    for k, b in folds.items():
        keep_v10[k][:, int(np.searchsorted(p.stations, 3754))] = False
        keep_v10[k][:, m_idx] = np.asarray(b.dates.isin(list(munhak_days_at(b.origin))))[:, None]
    out = dict(information='local OOF only; nested weights, gates and specialists fitted on aligned truth; no leaderboard value is used', members=S.members)
    base = {k: S.pred[k]['v10'] for k in folds}
    out['v10'] = dict(aligned=S.score(base), raw_truth=S.score(base, raw=True))
    out['singles'] = {m: dict(aligned=S.score({k: S.pred[k][m] for k in folds}), raw_truth=S.score({k: S.pred[k][m] for k in folds}, raw=True),
                              checkpoint={k: strict_index(m, k) for k in folds} if m != 'v10' else None, valid_curves=curves(m) if m != 'v10' else None) for m in S.members}
    protocols = dict(standard=lambda k: [j for j in folds if j != k], matched=lambda k: {4: [3], 3: [4], 1: [3, 4], 2: [3, 4]}[k])
    trials = {}
    for lam in RIDGE:
        for name, tr in protocols.items():
            pred, weights, gates = {}, {}, {}
            for k in folds:
                z, w, off = S.fit_apply(tr(k), k, lam, S.members)
                z = np.where(keep_v10[k], base[k], z)
                pred[k] = z; weights[k] = dict(zip(S.members, [round(float(x), 3) for x in w])); gates[k] = [int(s) for s in p.stations[off]]
            trials[(lam, name)] = dict(pred=pred, weights=weights, gated=gates, aligned=S.score(pred))
    v10a = out['v10']['aligned']
    ok = lambda a: all(a[f'fold_{k}'] <= v10a[f'fold_{k}'] + 1e-5 for k in (1, 2, 3, 4))
    ranked = sorted(trials.items(), key=lambda kv: (not ok(kv[1]['aligned']), kv[0][1] != 'standard', kv[1]['aligned']['all_oof']))
    (lam, proto), best = ranked[0]
    out['stack_trials'] = {f'lam={l} {n}': dict(aligned=t['aligned'], no_fold_worse_than_v10=ok(t['aligned'])) for (l, n), t in trials.items()}
    out['stack'] = dict(ridge=lam, protocol=proto, weights_nested=best['weights'], stations_kept_on_v10_nested=best['gated'],
                        aligned=best['aligned'], raw_truth=S.score(best['pred'], raw=True))
    stack = best['pred']
    print(f"stack: lam {lam} protocol {proto}  aligned {v10a['all_oof']:.5f} -> {best['aligned']['all_oof']:.5f}  " + ' '.join(f"F{k} {v10a[f'fold_{k}']:.5f}->{best['aligned'][f'fold_{k}']:.5f}" for k in (1, 2, 3, 4)), flush=True)

    # ---- specialists: replace a station only where the specialist beat the stack in the fitting folds --------------
    spec = specialist_forecasts()
    final = {k: stack[k].copy() for k in folds}; decisions = {}
    sse = lambda z, t: ((z - t) ** 2).sum(axis=(0, 2))
    for g, stations in special.GROUPS.items():
        for j, sid in enumerate(stations):
            s = int(np.searchsorted(p.stations, sid)); used = {}
            for k in folds:
                tr = protocols[proto](k)
                a = sum(float(((spec[f'{FOLDS[i]}__{g}'][:, j] - S.truth[i][:, s]) ** 2).sum()) for i in tr)
                b_ = sum(float(((stack[i][:, s] - S.truth[i][:, s]) ** 2).sum()) for i in tr)
                used[k] = bool(a < b_)
                if used[k]:
                    final[k][:, s] = np.where(keep_v10[k][:, s], final[k][:, s], spec[f'{FOLDS[k]}__{g}'][:, j])
            tr_all = [3, 4] if proto == 'matched' else list(folds)
            a = sum(float(((spec[f'{FOLDS[i]}__{g}'][:, j] - S.truth[i][:, s]) ** 2).sum()) for i in tr_all)
            b_ = sum(float(((stack[i][:, s] - S.truth[i][:, s]) ** 2).sum()) for i in tr_all)
            blocked = bool(young(test)[s])                           # young stations keep v10 whatever the specialist does
            if blocked:
                used = {k: False for k in folds}
            decisions[int(sid)] = dict(group=g, used_nested=used, used_for_test=bool(a < b_) and not blocked, kept_on_v10_as_young_station=blocked,
                                       rmsle=dict(v10=S.station_rmse(base, sid), stack=S.station_rmse(stack, sid),
                                                  specialist=S.station_rmse({k: np.where(np.arange(p.S)[None, :, None] == s, 0, 0) + _put(stack[k], s, spec[f'{FOLDS[k]}__{g}'][:, j]) for k in folds}, sid),
                                                  final=S.station_rmse(final, sid)))
    out['specialists'] = decisions
    out['final'] = dict(aligned=S.score(final), raw_truth=S.score(final, raw=True))
    fa = out['final']['aligned']
    print(f"final: aligned {fa['all_oof']:.5f}  " + ' '.join(f"F{k} {fa[f'fold_{k}']:.5f}" for k in (1, 2, 3, 4)) + f" | raw {out['final']['raw_truth']['all_oof']:.5f}", flush=True)
    out['stations'] = {int(sid): dict(v10=S.station_rmse(base, sid), stack=S.station_rmse(stack, sid), final=S.station_rmse(final, sid)) for sid in REPORT}
    rows = [dict(step=n, **{f'aligned_{a}': b for a, b in v['aligned'].items()}, **{f'raw_{a}': b for a, b in v['raw_truth'].items()})
            for n, v in [('v10', out['v10'])] + [(f'단독 {m}', out['singles'][m]) for m in S.names] + [('v11 스태킹', out['stack']), ('v11 최종(전용 모델 포함)', out['final'])]]
    pd.DataFrame(rows).to_csv(OUT / 'VALIDATION_V11.csv', index=False, float_format='%.5f')
    np.savez_compressed(OUT / 'oof_v11.npz', **{f'fold{k}': final[k].astype(np.float32) for k in folds})

    # ---- test forecast -------------------------------------------------------------------------------------------
    tr_all = [3, 4] if proto == 'matched' else list(folds)
    w = ridge_weights(S.errors(tr_all, S.members), lam)
    w = np.where(w < 0.005, 0.0, w); w = w / w.sum()                 # members below half a percent are dropped
    finals = {m: (load_final(m) if wi > 0 else 0.0) for wi, m in zip(w[1:], S.names)}
    if all(v is not None for v in finals.values()):
        v10_test = np.log1p(pd.read_csv(c8.ROOT / 'outputs/ensemble_v10/submission_ensemble_v10.csv').boardings.to_numpy())[test.row_index].reshape(test.pred.shape)
        stacked = {j: S.blend(j, w, S.members) for j in tr_all}
        off = S.station_sse(stacked, tr_all) > S.station_sse({j: base[j] for j in tr_all}, tr_all)
        z = w[0] * v10_test + sum(wi * finals[m] for wi, m in zip(w[1:], S.names))
        z[:, off] = v10_test[:, off]
        keep = np.broadcast_to(test.blocked[:, :, None], z.shape).copy()
        keep[:, int(np.searchsorted(p.stations, 3754))] = False
        keep[:, m_idx] = np.asarray(test.dates.isin(list(munhak_days_at(c8.FINAL))))[:, None]
        keep[:, young(test)] = True
        keep[int(np.flatnonzero(test.dates == pd.Timestamp('2026-09-12'))[0]), int(np.searchsorted(p.stations, 3139))] = True
        z = np.where(keep, v10_test, z)
        for g, stations in special.GROUPS.items():
            for j, sid in enumerate(stations):
                if decisions[int(sid)]['used_for_test']:
                    s = int(np.searchsorted(p.stations, sid)); z[:, s] = np.where(keep[:, s], z[:, s], spec[f'final__{g}'][:, j])
        z, event_log = apply_events(test, z, cfg.get('station_events', []))
        flat = np.empty(z.size); flat[test.row_index] = np.maximum(z, 0).reshape(-1)
        frame = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
        assert len(frame) == 157194 and np.isfinite(frame.boardings).all() and (frame.boardings >= 0).all()
        path = OUT / 'submission_ensemble_v11.csv'; frame.to_csv(path, index=False, float_format='%.8f')
        d = flat - np.log1p(pd.read_csv(c8.ROOT / 'outputs/ensemble_v10/submission_ensemble_v10.csv').boardings.to_numpy())
        out['test'] = dict(weights=dict(zip(S.members, [round(float(x), 3) for x in w])), stations_kept_on_v10=[int(s) for s in p.stations[off]],
                           specialists_used=[sid for sid, v in decisions.items() if v['used_for_test']], new_events=event_log,
                           young_stations_kept_on_v10=[int(x) for x in p.stations[young(test)]],
                           final_checkpoints={m: strict_index(m) for wi, m in zip(w[1:], S.names) if wi > 0}, final_seeds={m: len(list(MODELS.glob(f'final__{m}__*.npz'))) for wi, m in zip(w[1:], S.names) if wi > 0})
        out['submission'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(frame)), mean_boardings=float(frame.boardings.mean()),
                                 log_rmsd_vs_v10=float(np.sqrt(np.mean(d ** 2))), mean_log_diff_vs_v10=float(d.mean()))
        print(json.dumps(out['submission'], indent=1), flush=True)
    (OUT / 'gates_v11.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return out


def apply_events(block, pred, events):
    """Events with an earlier edition in the data: shrink x the lift measured on that edition is added to the forecast.
    mode 'add' keeps only positive lifts, 'signed' also lowers (weeks without classes)."""
    from src.v10 import games as g10
    out = pred.copy(); log = []
    for e in events:
        for sid in e['stations']:
            s = int(np.searchsorted(block.panel.stations, sid))
            for d, past in zip(e['dates'], e['past_dates'] if len(e['past_dates']) == len(e['dates']) else [e['past_dates'][0]] * len(e['dates'])):
                lift, n = g10.lift_profile(block, sid, [past])
                d = pd.Timestamp(d)
                if n == 0 or d not in block.dates:
                    continue
                k = int(np.flatnonzero(block.dates == d)[0])
                add = e.get('shrink', 0.85) * (lift if e.get('mode') == 'signed' else np.clip(lift, 0, None))
                log.append(dict(event=e['name'], station=int(sid), date=str(d.date()), past=past, daily_before=round(float(np.expm1(out[k, s]).sum())),
                                daily_after=round(float(np.expm1(out[k, s] + add).sum())), mean_log_change=round(float(add.mean()), 3)))
                out[k, s] = out[k, s] + add
    return out, log


def _put(z, s, col):
    z = z.copy(); z[:, s] = col
    return z


if __name__ == '__main__':
    main()
