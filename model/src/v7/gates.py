"""Module-by-module gating of ensemble_v7 on local out-of-fold predictions (no leaderboard information).

    python -m src.v7.gates            # writes outputs/ensemble_v7/gates.json and ABLATION_LOFO.csv

Strict numbers are nested: blend weights and every module coefficient applied to a fold are fitted on the
other three folds.  "plain" numbers use the saved in-fold ensemble columns of v2/v4/v5 with weights and
coefficients fitted on all four folds (the convention of the earlier reports).
"""
import json
from functools import partial
import numpy as np
import pandas as pd
from src.v6.schedule import Schedule
from . import core, modules, growth
from .panel import Panel
from .pipeline import apply_specialists, MunhakV7

ORDER = ['eq', 'beer', 'bucheon', 'monday', 'growth', 'gap', 'rain', 'rain0']
LABEL = dict(eq='M3 평형 기준선 모델군 결합', beer='M1 송도맥주축제 일정', bucheon='M5 부천 전년 경기 흔적 제거',
             monday='M5 문학 미발표일 월요일 분리', growth='M4 신규 역 성장 추세', gap='M5 최근 구간 급등 오염 보정',
             rain='M2 강수 보정(계수 적합)', rain0='M2 강수 g-계산(이론 계수: 기후 비 비율 − 기준 구간 비 비율)')
FOOTBALL = dict(despike_kinds=('concert', 'festival', 'football'))


class Context:
    def __init__(self, eq=None):
        self.cfg = core.config(); self.origins = self.cfg['fold_origins']
        self.train = core.load_train()
        self.sched6 = Schedule(core.ROOT / self.cfg['schedule_v6']); self.sched7 = core.schedule_v7(self.cfg)
        self.panel = Panel(core.ROOT, self.sched6)
        self.eq = core.eq_models() if eq is None else eq
        f = core.load_frame(self.cfg, self.eq)
        best = core.load_frame(self.cfg, self.eq, key='best')
        for m in self.eq:
            f[m + '_fit'] = best[m].to_numpy()
        self.frame = f
        self.y = f.true_log.to_numpy(); self.fold = f.fold.to_numpy(); self.h = f.horizon.to_numpy().astype(float)
        self.ordinary = (f.is_holiday.to_numpy() == 0)
        ext = core.OUT / 'external'
        self.rain = modules.Rain(ext / 'weather_incheon_daily.csv', ext / 'climate_incheon_daily_2015_2024.csv')
        self._slopes = modules.growth_columns(self.panel, f, self.origins)
        parts = [modules.gap_feature(self.panel, f[f.fold == k], o) for k, o in enumerate(self.origins, 1)]
        self._gap = np.concatenate(parts)
        g1, g2, self.rain_info = [], [], {}
        for k, o in enumerate(self.origins, 1):
            a, b, info = self.rain.features(self.panel, f[f.fold == k], o)
            g1.append(a); g2.append(b); self.rain_info[str(k)] = info
        self._rain = [np.concatenate(g1), np.concatenate(g2)]
        self._cache = {}; self.params = {}

    # ---- pipeline for a set of enabled modules -----------------------------------------------------
    def base(self, with_eq, nested, saved):
        key = ('base', with_eq, nested, saved)
        if key not in self._cache:
            f = self.frame; sfx = '_fit' if saved else ''
            names = core.VERSIONS + (self.eq if with_eq else [])
            tmp = pd.DataFrame({n: f[n + sfx].to_numpy() for n in names})
            for c in ('true_log', 'fold', 'is_holiday', 'station_id'):
                tmp[c] = f[c].to_numpy()
            full, per_fold = core.blend_columns(tmp, names, 'b')
            self._cache[key] = (tmp['b' if nested else 'b_in'].to_numpy(), full, per_fold)
        return self._cache[key]

    def special(self, base, key, beer, bucheon, monday):
        key = ('sp', key, beer, bucheon, monday)
        if key not in self._cache:
            f = self.frame.assign(_b=base)
            hook = partial(apply_specialists, munhak_cls=MunhakV7 if monday else apply_specialists.__defaults__[1],
                           overlay=FOOTBALL if bucheon else None)
            self._cache[key] = core.specialists(f, '_b', self.train, self.sched7 if beer else self.sched6, self.cfg, hook=hook)
        return self._cache[key]

    def residual_step(self, name, pred, nested):
        res = pred - self.y
        out = pred.copy(); folds = [1, 2, 3, 4]
        splits = [(self.fold != k, self.fold == k) for k in folds] if nested else [(np.ones(len(res), bool), np.ones(len(res), bool))]
        fitted = []
        for tr, te in splits:
            if name == 'growth':                                  # no fitted coefficient (see growth.py)
                s = self._slopes[0]
                out[te] = pred[te] + growth.correction(s[te], self.h[te], growth.LAG, growth.PERSISTENCE)
                use = tr & self.ordinary & (s != 0)
                a, b = growth.fit_ab(s[use], self.h[use], res[use])
                fitted.append({**growth.VARIANTS[0], 'a': growth.LAG, 'b': growth.PERSISTENCE,
                               'diagnostic_least_squares_a': round(a, 2), 'diagnostic_least_squares_b': round(b, 3)})
            elif name == 'gap':
                rows = tr & (self._gap != 0)
                k = float(modules.least_squares([-self._gap], res, rows)[0])
                out[te] = pred[te] - k * self._gap[te]; fitted.append(dict(kappa=round(k, 4)))
            elif name == 'rain':
                rows = tr & self.ordinary & (self.frame.station_id.to_numpy() != modules.MUNHAK)
                k = modules.least_squares(self._rain, res, rows)
                out[te] = pred[te] + k[0] * self._rain[0][te] + k[1] * self._rain[1][te]
                fitted.append(dict(k_window=round(float(k[0]), 3), k_climate=round(float(k[1]), 3)))
            elif name == 'rain0':                                 # g-computation with the coefficients theory gives
                keep = (self.frame.station_id.to_numpy() != modules.MUNHAK)
                out[te] = pred[te] + np.where(keep, self._rain[1] - self._rain[0], 0.0)[te]
                fitted.append(dict(k_window=-1.0, k_climate=1.0))
        return out, fitted

    def run(self, steps, nested=True, saved=False):
        steps = [s for s in ORDER if s in steps]
        key = ('run', tuple(steps), nested, saved)
        if key in self._cache:
            return self._cache[key]
        base, full, per_fold = self.base('eq' in steps, nested, saved)
        pred = self.special(base, ('eq' in steps, nested, saved), 'beer' in steps, 'bucheon' in steps, 'monday' in steps)
        fitted = dict(blend=per_fold if nested else full)
        for name in ('growth', 'gap', 'rain', 'rain0'):
            if name in steps:
                pred, fitted[name] = self.residual_step(name, pred, nested)
        self._cache[key] = (pred, fitted)
        return self._cache[key]

    def score(self, pred):
        return core.metrics(self.frame.assign(_p=pred), '_p')


def decide(before, after, changed_by_fold):
    """Adopt when the strict overall score improves, no fold gets worse, and Fold 2 and Fold 4 both improve.
    Folds where the module changes no row cannot improve; the result then says so."""
    tol = 1e-6
    worse = [f for f in (1, 2, 3, 4) if after[f'fold_{f}'] > before[f'fold_{f}'] + tol]
    better = [f for f in (1, 2, 3, 4) if after[f'fold_{f}'] < before[f'fold_{f}'] - tol]
    untouched = [f for f in (1, 2, 3, 4) if changed_by_fold[f] == 0 or abs(after[f'fold_{f}'] - before[f'fold_{f}']) <= tol]
    overall = after['all_oof'] < before['all_oof'] - tol
    strict = overall and not worse and 2 in better and 4 in better
    partial_rule = overall and not worse and all(f in better or f in untouched for f in (2, 4)) and bool(untouched)
    return dict(overall_improves=bool(overall), folds_better=better, folds_worse=worse, folds_untouched=untouched,
                strict_rule=bool(strict), no_harm_rule=bool(partial_rule and not strict))


def aux_folds(ctx):
    """Fold_Hot26 and Fold_Summer25.  v2/v4/v5 have no forecast at these origins, so the base here is the mean of
    the strict predictions of the equilibrium tree models; the modules are then applied exactly as in the folds."""
    from src.holiday_calendar import calendar_features
    from src.v6.pipeline import apply_specialists as v6_specialists
    out = {}
    folder = core.OUT / 'eq'
    for name, (origin, horizon) in ctx.cfg['aux_folds'].items():
        O = pd.Timestamp(origin)
        truth = ctx.train[(ctx.train.date > O) & (ctx.train.date <= O + pd.Timedelta(days=horizon))].copy()
        truth = truth.sort_values(['date', 'station_id', 'hour']).reset_index(drop=True)
        truth['true_log'] = np.log1p(truth.boardings)
        truth['is_holiday'] = calendar_features(truth.date, as_of=origin).is_holiday.to_numpy()
        di = (truth.date.to_numpy() - np.datetime64(origin)).astype('timedelta64[D]').astype(int) - 1
        si = np.searchsorted(ctx.panel.stations, truth.station_id.to_numpy()); hi = truth.hour.to_numpy() - 6
        y = truth.true_log.to_numpy(); models = {}
        for m in sorted({p.stem.split('__')[1] for p in folder.glob(f'{name}__*.npz')}):
            data = np.load(folder / f'{name}__{m}.npz')
            if 'lofo' in data.files:
                models[m] = data['lofo'][di, si, hi]
        if not models:
            continue
        trees = [m for m in models if m.split('_')[0] in ('lgb', 'xgb', 'cb')] or list(models)
        base = np.mean([models[m] for m in trees], 0)
        hist = ctx.train[ctx.train.date <= O]
        sp, _ = v6_specialists(hist, truth, base, origin, ctx.sched6)
        slope = growth.slope_column(ctx.panel, truth, origin, **growth.VARIANTS[0])
        gr = sp + growth.correction(slope, (di + 1).astype(float), growth.LAG, growth.PERSISTENCE)
        new = truth.station_id.isin(growth.new_stations(ctx.panel)).to_numpy()
        score = lambda p, m=None: core.rmse((p - y) if m is None else (p - y)[m])
        out[name] = dict(origin=origin, days=horizon, rows=int(len(truth)), base_models=trees,
                         models={m: score(p) for m, p in models.items()},
                         eq_tree_mean=score(base), plus_v6_specialists=score(sp), plus_growth=score(gr),
                         new_stations=dict(rows=int(new.sum()), before=score(sp, new), after=score(gr, new),
                                           mean_bias_before=float((sp - y)[new].mean()), mean_bias_after=float((gr - y)[new].mean())),
                         stations={str(sid): dict(eq_tree_mean=score(base, (truth.station_id == sid).to_numpy()),
                                                  final=score(gr, (truth.station_id == sid).to_numpy())) for sid in core.REPORT_STATIONS})
    return out


def growth_proxy(ctx):
    """Mechanism check that needs no base model: a rolling-anchor forecast (0.6 x mean of the last 8 same-class days
    + 0.4 x mean of the last 4) is biased low for the new stations relative to the old ones; does the drift remove it?"""
    P = ctx.panel; out = {}
    new = [int(np.searchsorted(P.stations, s)) for s in growth.new_stations(P)]
    old = np.flatnonzero((P.first == 0) & (P.stations != 3127))
    origins = dict(zip(['f1', 'f2', 'f3', 'f4'], ctx.origins)); origins.update({k: v[0] for k, v in ctx.cfg['aux_folds'].items()})
    for name, origin in origins.items():
        horizon = dict(ctx.cfg['aux_folds']).get(name, [None, 123])[1]
        O = P.day_index(origin); st = P.state(O); cal = P.cal(O); days = np.arange(O + 1, O + horizon + 1)
        slope = growth.slopes(P, origin, **growth.VARIANTS[0])
        if not slope:
            out[name] = dict(origin=origin, applied=False); continue
        anchor = 0.6 * st['mean8'][cal['cls'][days]] + 0.4 * st['m4'][cal['cls'][days]]
        res = anchor - np.moveaxis(P.Z[:, days], 0, 1); keep = cal['frame'][days, 0] == 0
        with np.errstate(all='ignore'):
            rel = np.nanmean(res[keep][:, new], axis=(1, 2)) - np.nanmean(res[keep][:, old], axis=(1, 2))
        sl = list(slope.values())[0]
        corr = growth.correction(np.full(keep.sum(), sl), (days - O)[keep].astype(float), growth.LAG, growth.PERSISTENCE)
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
        out[name] = dict(origin=origin, applied=True, days=int(keep.sum()), common_slope_per_day=sl, relative_bias_before=float(rel.mean()),
                         relative_bias_after=float((rel + corr).mean()), relative_rms_before=rms(rel), relative_rms_after=rms(rel + corr))
    return out


def ablation_table(out):
    rows = []
    hot = out.get('aux_folds', {}).get('hot26', {})

    def line(label, mode, steps, m, plain, adopt, rule, changed):
        # Fold_Hot26 has no v2/v4/v5 forecast: the value is on the equilibrium-tree base (+ v6 specialists),
        # with the growth drift added for rows whose steps include it; blank where the module cannot be run there.
        only = set(steps.split('+')) - {'v6'}
        hot26 = hot.get('plus_growth') if only and only <= {'growth', 'beer'} and 'growth' in only else (
            hot.get('plus_v6_specialists') if only <= {'beer'} else None)
        rows.append(dict(module=label, mode=mode, steps=steps, adopted=adopt, rule=rule, rows_changed=changed,
                         strict_lofo_all=m['all_oof'], plain_oof_all=plain, fold_1=m['fold_1'], fold_2=m['fold_2'], fold_3=m['fold_3'],
                         fold_4=m['fold_4'], fold_hot26_on_eq_tree_base=hot26, weighted_4fold=m['weighted_4fold'], fold_3_4=m['fold_3_4'],
                         **{k: v for k, v in m.items() if k.startswith('st_')}))
    b = out['baseline']
    line('v6 기준선 (중첩 LOFO 가중치)', 'baseline', 'v6', b['strict_nested'], b['plain_saved_columns']['all_oof'], '', '', 0)
    line('v6 기준선 (v6 보고 방식: 전체 fold 가중치)', 'baseline', 'v6', b['v6_convention_weights_on_all_folds'], b['plain_saved_columns']['all_oof'], '', '', 0)
    for key, r in out['single'].items():
        v = r['verdict']; rule = 'strict' if v['strict_rule'] else ('no_harm' if v['no_harm_rule'] else 'fail')
        line(r['module'], 'single', r['steps'], r['strict'], r['plain_all_oof'], key in out['adopted'], rule, r['rows_changed'])
    for r in out['cumulative']:
        line(r['module'], 'cumulative', r['steps'], r['strict'], r['plain_all_oof'], True, '', r['rows_changed'])
    return pd.DataFrame(rows)


def main(adopt=None):
    ctx = Context()
    f = ctx.frame
    out = dict(information='local OOF only; no leaderboard score is read anywhere in src/v7', eq_models=ctx.eq,
               rain=ctx.rain_info, single={}, cumulative=[], rows=[])
    base_pred, base_fit = ctx.run([])
    base = ctx.score(base_pred)
    plain_base = ctx.score(ctx.run([], nested=False, saved=True)[0])
    v6_in = ctx.score(ctx.run([], nested=False)[0])
    out['baseline'] = dict(strict_nested=base, v6_convention_weights_on_all_folds=v6_in, plain_saved_columns=plain_base)
    out['base_versions'] = {v: dict(strict=core.metrics(f, v, []), saved=core.metrics(f, v + '_fit', [])) for v in core.VERSIONS}
    out['eq_scores'] = {m: dict(strict=core.metrics(f, m, []), early_stopped_on_fold=core.metrics(f, m + '_fit', [])) for m in ctx.eq}

    def changed(a, b):
        return {k: int((np.abs(a - b)[ctx.fold == k] > 1e-9).sum()) for k in (1, 2, 3, 4)}

    def record(module, mode, steps, before_pred, before):
        pred, fitted = ctx.run(steps)
        after = ctx.score(pred); ch = changed(pred, before_pred)
        plain = ctx.score(ctx.run(steps, nested=False, saved=True)[0])
        verdict = decide(before, after, ch)
        row = dict(module=LABEL.get(module, module), key=module, mode=mode, steps='+'.join(steps) or 'v6', rows_changed=sum(ch.values()),
                   rows_changed_by_fold=ch, strict=after, plain_all_oof=plain['all_oof'], verdict=verdict,
                   fitted={k: v for k, v in fitted.items() if k != 'blend'} | ({'blend': fitted['blend']} if module == 'eq' else {}))
        return pred, after, row

    for module in ORDER:
        if module == 'eq' and not ctx.eq:
            continue
        _, _, row = record(module, 'single', [module], base_pred, base)
        out['single'][module] = row
        v = row['verdict']
        print(f"[single] {module:8s} all {base['all_oof']:.5f}->{row['strict']['all_oof']:.5f}  "
              + ' '.join(f"F{k} {base[f'fold_{k}']:.5f}->{row['strict'][f'fold_{k}']:.5f}" for k in (1, 2, 3, 4))
              + f"  rows {row['rows_changed']}  strict={v['strict_rule']} no_harm={v['no_harm_rule']}", flush=True)

    decisions = {}
    quota_path = core.OUT / 'munhak_quota.json'
    quota = json.loads(quota_path.read_text()) if quota_path.exists() else None
    overrides = ctx.cfg.get('adoption_overrides', {})
    for module, row in out['single'].items():
        v = row['verdict']
        ok = v['strict_rule'] or v['no_harm_rule']
        reason = ('엄격 규칙 통과: 전체·Fold 2·Fold 4 개선, 악화 fold 없음' if v['strict_rule'] else
                  '변경된 fold만 개선, 악화 fold 없음' if v['no_harm_rule'] else
                  '기각: ' + ('전체 미개선' if not v['overall_improves'] else '') + (f" 악화 fold {v['folds_worse']}" if v['folds_worse'] else '')
                  + ('' if v['folds_worse'] or not v['overall_improves'] else ' Fold 2·Fold 4 동시 개선 아님'))
        if module == 'monday' and ok and quota:
            a = quota['validation_summer25']['v6']['rmsle_unannounced_rows']; b = quota['validation_summer25']['v7_monday_split']['rmsle_unannounced_rows']
            if b >= a:
                ok = False; reason = f'기각: Fold 1은 개선되지만 Fold_Summer25 미발표일 RMSLE가 {a:.4f} -> {b:.4f}로 악화'
        if module in overrides:
            ok = True; reason = '지시서 명시 지시로 채택(검증 fold에 해당 행 없음): ' + overrides[module]
        decisions[module] = dict(adopt=bool(ok), reason=reason)
    out['decisions'] = decisions
    if adopt is None:
        adopt = [m for m in ORDER if decisions.get(m, {}).get('adopt')]
    steps, prev_pred, prev = [], base_pred, base
    for module in [m for m in ORDER if m in adopt]:
        steps = steps + [module]
        pred, after, row = record(module, 'cumulative', steps, prev_pred, prev)
        out['cumulative'].append(row)
        print(f"[cumulative] +{module:8s} all {prev['all_oof']:.5f}->{after['all_oof']:.5f} "
              + ' '.join(f"F{k} {after[f'fold_{k}']:.5f}" for k in (1, 2, 3, 4)), flush=True)
        prev_pred, prev = pred, after
    out['adopted'] = [m for m in ORDER if m in adopt]
    final_pred, final_fit = ctx.run(out['adopted'])
    out['final_strict'] = ctx.score(final_pred)
    out['final_plain'] = ctx.score(ctx.run(out['adopted'], nested=False, saved=True)[0])
    in_pred, in_fit = ctx.run(out['adopted'], nested=False)
    out['final_coefficients_all_folds'] = in_fit
    f.assign(v6_strict=base_pred, v7_strict=final_pred)[['fold', 'date', 'station_id', 'hour', 'true_log', 'is_holiday', 'v6_strict', 'v7_strict']] \
        .to_pickle(core.OUT / 'oof_v6_v7.pkl')
    out['aux_folds'] = aux_folds(ctx)
    out['growth_proxy_check'] = growth_proxy(ctx)
    for name, r in out['aux_folds'].items():
        print(f"[aux] {name}: eq tree mean {r['eq_tree_mean']:.5f} -> +specialists {r['plus_v6_specialists']:.5f} -> +growth {r['plus_growth']:.5f}"
              f" | new stations {r['new_stations']['before']:.4f}->{r['new_stations']['after']:.4f}", flush=True)
    (core.OUT / 'gates.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    ablation_table(out).to_csv(core.OUT / 'ABLATION_LOFO.csv', index=False, float_format='%.5f')
    return out, ctx


if __name__ == '__main__':
    main()
