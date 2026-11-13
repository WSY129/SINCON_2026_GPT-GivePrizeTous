"""Builds submission_ensemble_v12.csv from the stored v11 forecasts (CPU only; no leaderboard value is used).

    python -m src.v12.run

Changes against v11 (each one decided on nested, aligned out-of-fold errors):
  1. checkpoints: neighbouring stored checkpoints are averaged
  2. two-stage stack: weights on all cells, then segment weights (ridge towards stage 1) for the segments that
     improve in every fold; the other segments keep the stage-1 weights
  3. station gate: continuous empirical-Bayes weight between the stack and v10 instead of 0/1
  4. Bucheon: on match days the exit hours (kick-off + 1 h .. + 4 h) are blended with v10's attendance regression;
     the Bucheon tree specialist is dropped
  5. new stations: weather-only residual model without day-of-year terms and without the first 60 days after opening
  6. one-off evening event at the grand park (2026-10-10) with a conservative attendance share
"""
import hashlib
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.v7 import core as core7
from src.v8 import core as c8, actual
from src.v8.run import block_events
from src.v10 import games as g10
from src.v11 import stack as s11, special
from src.v11.panel import Panel11, FI
from . import meta

OUT = c8.ROOT / 'outputs/ensemble_v12'
LAM2 = 0.002
REPORT = s11.REPORT
NEW_FEATURES = ['sid', 'hour', 'dow', 'cat', 'hol_eff', 'rain_h', 'rain_prev2', 'rain_next1', 'rain_mm', 'wet', 'temp_h', 'app_h', 'tmax', 'tdev', 'hot', 'heatwave',
                'comfort', 'rad_h', 'rad_day', 'raddev', 'sun_hours', 'cloud_day', 'wind_max', 'outing']
SETTLE = 60
MIN_TRAIN_DAYS = 45  # days of training rows a new station needs after the settling period
EB_STRENGTH = 0.25   # prior weight of the station shrinkage, in units of the median station's evidence
BAND = 5000          # crowd above which the attendance regression gets its own weight at Bucheon (fixed before fitting)


def new_station_model(wx, origin):
    """Weather/holiday residual for the stations opened inside the data window: no day-of-year terms, no rows from
    the first 60 days, and the growth trend removed from the training residuals.  Returns {station index: [day, H]}."""
    d = wx.build(origin); O = d['O']; cols = [FI[f] for f in NEW_FEATURES]
    idx = np.flatnonzero(wx.p.first > 0); Xs, ys, Xp = [], [], {}
    for s in idx:
        first = int(wx.p.first[s]); lo = first + SETTLE
        if O - lo < MIN_TRAIN_DAYS:
            continue
        r = d['r'][s, lo:O + 1]; age = d['age'][lo:O + 1].astype(float)
        daily = np.nanmean(r, 1); ok = ~np.isnan(daily)
        b, a = np.polyfit(age[ok], daily[ok], 1)                        # growth seen from the cutoff: residual ~ a + b * age
        det = r - (a + b * age)[:, None]
        good = ~np.isnan(det)
        Xs.append(d['X'][s, lo:O + 1][good][:, cols]); ys.append(det[good]); Xp[int(s)] = d['X'][s, O + 1:][:, :, cols]
    if not Xs:
        return {}
    X = np.concatenate(Xs); y = np.concatenate(ys); out = {}
    models = [lgb.train(dict(objective='regression', learning_rate=0.03, num_leaves=7, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                             lambda_l2=10.0, verbose=-1, num_threads=8, seed=seed), lgb.Dataset(X, y), 300) for seed in (1, 2, 3)]
    for s, xp in Xp.items():
        f = np.mean([m.predict(xp.reshape(-1, len(cols))) for m in models], 0).reshape(xp.shape[:2])
        out[s] = np.maximum(d['anchor'][s, O + 1:] + f, 0)
    return out


def centre(z, base, dates):
    """Removes the monthly mean of (weather model - baseline) per station: the level and its seasonal drift stay with
    the baseline, only the day-to-day and hour-to-hour weather / holiday effects of the model are kept."""
    months = np.asarray(dates.to_period('M').astype(str)); out = {}
    for s, a in z.items():
        d = a - base[:, s]
        for m in np.unique(months):
            d[months == m] -= d[months == m].mean()
        out[s] = base[:, s] + d
    return out


def main():
    S = s11.Stack(); p = S.panel; folds = S.folds; test = S.test
    wx = Panel11(); schedule = core7.schedule_v7()
    cfg10 = g10.config(); cfg11 = json.loads((c8.ROOT / 'configs/ensemble_v11.json').read_text()); cfg = json.loads((c8.ROOT / 'configs/ensemble_v12.json').read_text())
    names = S.names; members = ['v10'] + names
    P = {k: {'v10': S.pred[k]['v10'], **{n: meta.load_fold(n, k) for n in names}} for k in folds}
    base = {k: P[k]['v10'] for k in folds}
    seg = {k: meta.segments(b, base[k], wx) for k, b in folds.items()}
    events = cfg10.get('station_events', []) + cfg11.get('station_events', [])
    extra = [(sid, [pd.Timestamp(d) for d in e['dates'] + e['past_dates']]) for e in events for sid in e['stations']]
    for b in list(folds.values()) + [test]:
        b.blocked[:] = False; block_events(b, schedule, extra=extra)
    m_idx = int(np.searchsorted(p.stations, 3127)); b_idx = int(np.searchsorted(p.stations, 3754)); park = int(np.searchsorted(p.stations, 3226))
    listed = {pd.Timestamp(e['date']) for e in cfg10.get('munhak_events', [])}
    munhak_days = lambda origin: listed | {pd.Timestamp(e['date']) for e in schedule.venue_events('munhak_main') if str(e['known_since'])[:10] <= str(origin)[:10]}
    young = lambda b: (b.O - p.first < 400) & (p.first > 0)

    def keep_mask(b):
        """cells that keep the v10 value: listed event station-days (v10 has explicit lifts), not Bucheon."""
        keep = np.broadcast_to(b.blocked[:, :, None], b.pred.shape).copy()
        keep[:, b_idx] = False
        keep[:, m_idx] = np.asarray(b.dates.isin(list(munhak_days(b.origin))))[:, None]
        return keep
    keep = {k: keep_mask(b) for k, b in folds.items()}
    E = lambda ks: np.concatenate([np.stack([(P[k][m] - S.truth[k]).ravel() for m in members], 1) for k in ks])
    SG = lambda ks: np.concatenate([seg[k].ravel() for k in ks])
    out = dict(information='local OOF only; nested fits on aligned truth; no leaderboard value is used', members=members, segments=meta.SEGMENTS)
    o11 = np.load(c8.ROOT / 'outputs/ensemble_v11/oof_v11.npz'); v11 = {k: o11[f'fold{k}'].astype(float) for k in folds}
    o8 = np.load(c8.ROOT / 'outputs/ensemble_v8/oof_v8.npz'); o9 = np.load(c8.ROOT / 'outputs/ensemble_v9/oof_v9_main.npz')
    earlier = {'v7': {k: o8[f'v7_fold{k}'].astype(float) for k in folds}, 'v8 보험용(상반기 마감)': {k: o8[f'precutoff_fold{k}'].astype(float) for k in folds},
               'v8': {k: o8[f'main_fold{k}'].astype(float) for k in folds}, 'v9': {k: o9[f'fold{k}'].astype(float) for k in folds}, 'v10': base, 'v11': v11}
    out['generations'] = {name: dict(aligned=S.score(z), raw_truth=S.score(z, raw=True)) for name, z in earlier.items()}

    # ---- two-stage weights; a segment gets its own weights only if that improves it in every fold -------------------
    one, two, wfold = {}, {}, {}
    gain = {i: [] for i in meta.SEGMENTS}
    for k in folds:
        tr = [j for j in folds if j != k]
        w1, ws = meta.fit_two_stage(E(tr), SG(tr), LAM2); wfold[k] = (w1, ws)
        mem = [P[k][m] for m in members]
        one[k] = meta.apply_two_stage(mem, seg[k], {i: w1 for i in meta.SEGMENTS}); two[k] = meta.apply_two_stage(mem, seg[k], ws)
        for i in meta.SEGMENTS:
            m = seg[k] == i
            gain[i].append(float((((two[k] - S.truth[k]) ** 2) - ((one[k] - S.truth[k]) ** 2))[m].sum()))
    own = [i for i in meta.SEGMENTS if all(g < 0 for g in gain[i])]
    P0 = {k: {'v10': base[k], **{n: meta.load_fold(n, k, smoothed=False) for n in names}} for k in folds}; plain = {}
    for k in folds:
        tr = [j for j in folds if j != k]
        w0 = meta.simplex_ridge(np.concatenate([np.stack([(P0[j][m] - S.truth[j]).ravel() for m in members], 1) for j in tr]))
        plain[k] = sum(w * P0[k][m] for w, m in zip(w0, members))
    out['checkpoint_average'] = dict(kernel=list(meta.KERNEL), single_checkpoint=S.score(plain), averaged=S.score(one),
                                     best_single_model={n: dict(single=S.score({k: P0[k][n] for k in folds})['all_oof'], averaged=S.score({k: P[k][n] for k in folds})['all_oof']) for n in ('lgb_b', 'xgb', 'cb')})
    stack = {k: np.where(np.isin(seg[k], own), two[k], one[k]) for k in folds}
    out['two_stage'] = dict(ridge_to_stage1=LAM2, sse_change_by_segment_and_fold={meta.SEGMENTS[i]: [round(g, 1) for g in gain[i]] for i in gain},
                            segments_with_own_weights=[meta.SEGMENTS[i] for i in own],
                            one_weight_set=S.score(one), all_segments=S.score(two), gated=S.score(stack),
                            weights_nested={k: dict(stage1=dict(zip(members, [round(float(x), 3) for x in wfold[k][0]])),
                                                    **{meta.SEGMENTS[i]: dict(zip(members, [round(float(x), 3) for x in wfold[k][1][i]])) for i in own}) for k in folds})
    print('two-stage: one set', round(out['two_stage']['one_weight_set']['all_oof'], 5), '| all segments', round(out['two_stage']['all_segments']['all_oof'], 5),
          '| gated', round(out['two_stage']['gated']['all_oof'], 5), '| own weights:', out['two_stage']['segments_with_own_weights'], flush=True)

    # ---- continuous station weights (empirical Bayes) instead of the 0/1 gate ---------------------------------------
    eb, alphas = {}, {}
    for k in folds:
        tr = [j for j in folds if j != k]
        a = meta.eb_station_weights([stack[j] for j in tr], [base[j] for j in tr], [S.truth[j] for j in tr], EB_STRENGTH); alphas[k] = a
        eb[k] = base[k] + a[None, :, None] * (stack[k] - base[k])
    binary = {}
    for k in folds:
        tr = [j for j in folds if j != k]
        off = S.station_sse({j: stack[j] for j in tr}, tr) > S.station_sse({j: base[j] for j in tr}, tr)
        z = stack[k].copy(); z[:, off] = base[k][:, off]; binary[k] = z
    out['station_weights'] = dict(no_gate=S.score(stack), binary_gate=S.score(binary), empirical_bayes=S.score(eb))
    use_eb = S.score(eb)['all_oof'] <= S.score(binary)['all_oof']
    gated = eb if use_eb else binary
    # event station-days: v11 kept the v10 value there (weight 1).  v12 fits that weight on the other folds, separately for Munhak and the rest
    grp = {k: np.where(keep[k], np.where(np.arange(len(p.stations))[None, :, None] == m_idx, 1, 2), 0) for k in folds}
    def keep_w(ks, gid):
        num = sum(float(((S.truth[j] - gated[j]) * (base[j] - gated[j]))[grp[j] == gid].sum()) for j in ks)
        den = sum(float(((base[j] - gated[j]) ** 2)[grp[j] == gid].sum()) for j in ks)
        return float(np.clip(num / max(den, 1e-9), 0, 1))
    wk = {k: {gid: keep_w([j for j in folds if j != k], gid) for gid in (1, 2)} for k in folds}
    part = lambda z, gid: {k: round(float(((z[k] - S.truth[k]) ** 2)[grp[k] == gid].sum()), 2) for k in folds}
    soft = {k: gated[k] + np.where(grp[k] == 1, wk[k][1], np.where(grp[k] == 2, wk[k][2], 0.0)) * (base[k] - gated[k]) for k in folds}
    hard = {k: np.where(keep[k], base[k], gated[k]) for k in folds}
    out['event_days'] = dict(cells={k: [int((grp[k] == 1).sum()), int((grp[k] == 2).sum())] for k in folds}, weight_on_v10_nested=wk,
                             sse_munhak=dict(stack=part(gated, 1), v10=part(base, 1), fitted=part(soft, 1)), sse_other=dict(stack=part(gated, 2), v10=part(base, 2), fitted=part(soft, 2)),
                             all_v10=S.score(hard), fitted_weight=S.score(soft), no_rule=S.score(gated))
    by_station = {}
    for k in folds:
        dd = (((base[k] - S.truth[k]) ** 2) - ((gated[k] - S.truth[k]) ** 2)) * (grp[k] == 2)
        for s_ in np.flatnonzero(dd.sum(axis=(0, 2)) != 0):
            by_station.setdefault(int(p.stations[s_]), {})[k] = round(float(dd[:, s_].sum()), 2)
    out['event_days']['sse_v10_minus_stack_by_station'] = by_station
    # The fitted weight is NOT used for the test.  Inside the folds v10 has no lift on these days (the past editions are
    # the fold targets themselves), while for the test period v10 carries lifts taken from those past editions, so the
    # fold comparison is biased against v10 on exactly these cells.  The v11 rule (v10 on listed event days) is kept and
    # the fold scores below include it.
    out['event_days']['used'] = 'all_v10'
    print('event days: all v10', round(S.score(hard)['all_oof'], 5), '| fitted weight', round(S.score(soft)['all_oof'], 5), '| no rule', round(S.score(gated)['all_oof'], 5), flush=True)
    cur = hard
    out['station_weights']['used'] = 'empirical_bayes' if use_eb else 'binary_gate'
    out['station_weights']['alpha_nested_range'] = {k: [round(float(alphas[k].min()), 3), round(float(np.median(alphas[k])), 3)] for k in folds}
    ladder, ladder_raw = {}, {}
    def step(name, z):
        ladder[name] = S.score(z); ladder_raw[name] = S.score(z, raw=True)
    step('가중치 한 벌(지점 평균 포함)', one); step('기상 구간 전용 가중치', stack); step('역별 연속 수축', eb); step('행사일 v10 유지 규칙', cur)
    print('station weights: none', round(S.score(stack)['all_oof'], 5), 'binary', round(S.score(binary)['all_oof'], 5), 'EB', round(S.score(eb)['all_oof'], 5), flush=True)

    # ---- parks specialist (as v11) -----------------------------------------------------------------------------------
    spec = dict(np.load(c8.ROOT / 'outputs/ensemble_v11/specialists.npz'))
    park_use = {}
    for k in folds:
        tr = [j for j in folds if j != k]
        a = sum(float(((spec[f'{s11.FOLDS[j]}__parks'][:, 0] - S.truth[j][:, park]) ** 2).sum()) for j in tr); b_ = sum(float(((cur[j][:, park] - S.truth[j][:, park]) ** 2).sum()) for j in tr)
        park_use[k] = bool(a < b_)
    cur2 = {k: cur[k].copy() for k in folds}
    for k in folds:
        if park_use[k]:
            cur2[k][:, park] = np.where(keep[k][:, park], cur[k][:, park], spec[f'{s11.FOLDS[k]}__parks'][:, 0])
    park_test = sum(float(((spec[f'{s11.FOLDS[j]}__parks'][:, 0] - S.truth[j][:, park]) ** 2).sum()) for j in folds) < sum(float(((cur[j][:, park] - S.truth[j][:, park]) ** 2).sum()) for j in folds)
    out['parks_specialist'] = dict(used_nested=park_use, used_for_test=bool(park_test), before=S.station_rmse(cur, 3226), after=S.station_rmse(cur2, 3226))
    cur = cur2
    step('인천대공원 전용 모델(v11과 동일)', cur)

    # ---- Bucheon: specialist only outside the exit hours; exit hours blended with v10's attendance regression by crowd band
    matches = g10.bucheon_matches(cfg10)

    def exit_cells(b):
        """(mask, attendance) [day, H] of the exit hours (kick-off + 1 h .. + 4 h) of the home matches inside a block."""
        m = np.zeros((len(b.dates), p.H), bool); att = np.zeros((len(b.dates), p.H))
        for r in matches[matches.date.isin(b.dates)].itertuples():
            d = int(np.flatnonzero(b.dates == r.date)[0]); h0 = int((r.end - 115) // 60)
            hs = [h - 6 for h in range(h0 + 1, h0 + 5) if 6 <= h <= 23]
            m[d, hs] = True; att[d, hs] = r.attendance
        return m, att
    ex, att = {}, {}
    for k, b in folds.items():
        ex[k], att[k] = exit_cells(b)
    tb = lambda j: S.truth[j][:, b_idx]
    sp = lambda j: spec[f'{s11.FOLDS[j]}__bucheon'][:, 0].astype(float)
    ls = lambda num, den: float(np.clip(num / max(den, 1e-9), 0, 1))
    sse = lambda z, ks, mask: {j: round(float((((z[j] if isinstance(z, dict) else z(j)) - tb(j)) ** 2)[mask(j)].sum()), 3) for j in ks}
    # (a) outside the exit hours: convex weight on the station's own tree model
    src = {k: cur[k][:, b_idx].copy() for k in folds}
    w_spec = lambda ks: ls(sum(float(((tb(j) - src[j]) * (sp(j) - src[j]))[~ex[j]].sum()) for j in ks), sum(float(((sp(j) - src[j]) ** 2)[~ex[j]].sum()) for j in ks))
    wsn = {k: w_spec([j for j in folds if j != k]) for k in folds}
    quiet = {k: np.where(ex[k], src[k], src[k] + wsn[k] * (sp(k) - src[k])) for k in folds}
    q0 = sse(src, folds, lambda j: ~ex[j]); q1 = sse(quiet, folds, lambda j: ~ex[j])
    adopt_q = sum(q1.values()) < sum(q0.values()) and q1[3] + q1[4] < q0[3] + q0[4]
    ws_test = w_spec(list(folds)) if adopt_q else 0.0
    mid = quiet if adopt_q else src
    # (b) exit hours: weight on v10 fitted separately below and above the crowd band
    vb = lambda j: base[j][:, b_idx]
    def band_weights(ks):
        w = []
        for high in (False, True):
            sel = lambda j: ex[j] & ((att[j] >= BAND) == high)
            w.append(ls(sum(float(((tb(j) - mid[j]) * (vb(j) - mid[j]))[sel(j)].sum()) for j in ks), sum(float(((vb(j) - mid[j]) ** 2)[sel(j)].sum()) for j in ks)))
        return w
    wbn = {k: band_weights([j for j in folds if j != k]) for k in folds}
    blend = {k: np.where(ex[k], mid[k] + np.where(att[k] >= BAND, wbn[k][1], wbn[k][0]) * (vb(k) - mid[k]), mid[k]) for k in folds}
    e0 = sse(mid, folds, lambda j: ex[j]); e1 = sse(blend, folds, lambda j: ex[j]); ev = sse(lambda j: vb(j), folds, lambda j: ex[j])
    adopt_b = sum(e1.values()) < sum(e0.values()) and e1[3] + e1[4] < e0[3] + e0[4]
    wb_test = band_weights(list(folds)) if adopt_b else [0.0, 0.0]
    cur3 = {k: cur[k].copy() for k in folds}
    for k in folds:
        cur3[k][:, b_idx] = blend[k] if adopt_b else mid[k]
    hi = lambda z, j: (z[j] - tb(j))[ex[j] & (att[j] >= BAND)]
    out['bucheon'] = dict(crowd_band=BAND,
                          outside_exit_hours=dict(weight_on_station_model_nested=wsn, weight_test=ws_test, sse_before=q0, sse_after=q1, adopted=bool(adopt_q)),
                          exit_hours=dict(weight_on_v10_nested_low_high=wbn, weight_test_low_high=wb_test, sse_v10=ev, sse_stack=e0, sse_blended=e1, adopted=bool(adopt_b),
                                          mean_log_error_high_crowd={'v10': {j: round(float((vb(j) - tb(j))[ex[j] & (att[j] >= BAND)].mean()), 3) for j in folds},
                                                                     'stack': {j: round(float(hi(mid, j).mean()), 3) for j in folds},
                                                                     'blended': {j: round(float(hi(blend, j).mean()), 3) for j in folds}}),
                          station=dict(v10=S.station_rmse(base, 3754), v11=S.station_rmse(v11, 3754), before=S.station_rmse(cur, 3754), after=S.station_rmse(cur3, 3754)))
    cur = cur3
    step('부천 교정(비퇴장 시간 전용 모델 + 퇴장 시간 관중 회귀 혼합)', cur)
    print('bucheon outside exit hours: w', {k: round(v, 2) for k, v in wsn.items()}, 'sse', q0, '->', q1, 'adopt', adopt_q, flush=True)
    print('bucheon exit hours: w(low, high)', {k: [round(x, 2) for x in v] for k, v in wbn.items()}, 'sse v10', ev, 'stack', e0, 'blend', e1, 'adopt', adopt_b, 'test', wb_test, flush=True)

    # ---- new stations: weather residual model, validated on Folds 3 and 4 ---------------------------------------------
    new = {k: new_station_model(wx, b.origin) for k, b in folds.items()}
    new_test = new_station_model(wx, c8.FINAL)
    plain_new = new; new = {k: centre(plain_new[k], base[k], folds[k].dates) for k in folds}
    yidx = [int(s) for s in np.flatnonzero(p.first > 0)]
    def new_w(tr, s):
        num = sum(float(((S.truth[j][:, s] - base[j][:, s]) * (new[j][s] - base[j][:, s])).sum()) for j in tr if s in new[j])
        den = sum(float(((new[j][s] - base[j][:, s]) ** 2).sum()) for j in tr if s in new[j])
        return float(np.clip(num / max(den, 1e-9), 0, 1))
    wn = {s: {4: new_w([3], s), 3: new_w([4], s), 2: new_w([3, 4], s), 1: 0.0} for s in yidx}           # later two folds validate each other
    mix = lambda k, s, w: base[k][:, s] + w * (new[k][s] - base[k][:, s]) if s in new[k] else base[k][:, s]
    er = lambda z, k, s: c8.rmse(z - S.truth[k][:, s])
    ok = {s: all(er(mix(k, s, wn[s][k]), k, s) < er(base[k][:, s], k, s) - 1e-5 for k in (3, 4)) for s in yidx}
    w_new_test = {s: (new_w([3, 4], s) if ok[s] else 0.0) for s in yidx}
    final = {}
    for k, b in folds.items():
        z = cur[k].copy()
        for s in np.flatnonzero(young(b)):
            s = int(s)
            z[:, s] = np.where(keep[k][:, s], base[k][:, s], mix(k, s, wn[s][k]) if ok.get(s) else base[k][:, s])
        final[k] = z
    cells = lambda z, k: np.concatenate([(z[k][:, s] - S.truth[k][:, s]).ravel() for s in yidx])
    def uncentred(s):                                                        # the same protocol without removing the monthly mean (reported only)
        w = lambda tr: float(np.clip(sum(float(((S.truth[j][:, s] - base[j][:, s]) * (plain_new[j][s] - base[j][:, s])).sum()) for j in tr)
                                     / max(sum(float(((plain_new[j][s] - base[j][:, s]) ** 2).sum()) for j in tr), 1e-9), 0, 1))
        ws_ = {4: w([3]), 3: w([4]), 2: w([3, 4])}
        return {k: round(er(base[k][:, s] + ws_[k] * (plain_new[k][s] - base[k][:, s]), k, s), 4) for k in (2, 3, 4) if s in plain_new[k]}
    out['new_stations'] = dict(features=NEW_FEATURES, excluded_first_days=SETTLE, monthly_mean_removed=True,
                               blended_without_centring={int(p.stations[s]): uncentred(s) for s in yidx},
                               per_station={int(p.stations[s]): dict(weight_nested=wn[s], weight_test=w_new_test[s], adopted=bool(ok[s]),
                                                                     v10={k: round(er(base[k][:, s], k, s), 4) for k in (2, 3, 4)},
                                                                     weather_model={k: round(er(new[k][s], k, s), 4) for k in (2, 3, 4) if s in new[k]},
                                                                     blended={k: round(er(mix(k, s, wn[s][k]), k, s), 4) for k in (2, 3, 4)}) for s in yidx},
                               rmsle_new_station_cells={k: dict(v10=round(c8.rmse(cells(base, k)), 4), v12=round(c8.rmse(cells(final, k)), 4)) for k in (2, 3, 4)})
    print('new stations:', {sid: (v['adopted'], round(v['weight_test'], 2), v['v10'], v['blended']) for sid, v in out['new_stations']['per_station'].items()}, flush=True)
    step('신설 역 규칙(개통 400일 미만은 v10, 채택 역은 기상 잔차 혼합) = v12', final)
    out['ladder'] = ladder; out['ladder_raw'] = ladder_raw
    out['generations']['v12'] = dict(aligned=S.score(final), raw_truth=S.score(final, raw=True))
    fa = out['generations']['v12']['aligned']; va = out['generations']['v11']['aligned']
    out['no_fold_worse_than_v11'] = bool(all(fa[f'fold_{k}'] <= va[f'fold_{k}'] + 1e-5 for k in (1, 2, 3, 4)))
    out['folds_worse_than_v11'] = {k: round(fa[f'fold_{k}'] - va[f'fold_{k}'], 6) for k in (1, 2, 3, 4) if fa[f'fold_{k}'] > va[f'fold_{k}'] + 1e-5}
    print(f"v12: aligned {va['all_oof']:.5f} -> {fa['all_oof']:.5f}  " + ' '.join(f"F{k} {va[f'fold_{k}']:.5f}->{fa[f'fold_{k}']:.5f}" for k in (1, 2, 3, 4))
          + f" | raw {out['generations']['v11']['raw_truth']['all_oof']:.5f} -> {out['generations']['v12']['raw_truth']['all_oof']:.5f}", flush=True)
    out['stations'] = {int(sid): {name.split(' ')[0] + ('p' if '보험' in name else ''): S.station_rmse(z, sid) for name, z in {**earlier, 'v12': final}.items()} for sid in REPORT}
    np.savez_compressed(OUT / 'oof_v12.npz', **{f'fold{k}': final[k].astype(np.float32) for k in folds})

    # ---- test forecast -----------------------------------------------------------------------------------------------
    v10_test = np.log1p(pd.read_csv(c8.ROOT / 'outputs/ensemble_v10/submission_ensemble_v10.csv').boardings.to_numpy())[test.row_index].reshape(test.pred.shape)
    v11_test = np.log1p(pd.read_csv(c8.ROOT / 'outputs/ensemble_v11/submission_ensemble_v11.csv').boardings.to_numpy())[test.row_index].reshape(test.pred.shape)
    all_folds = list(folds)
    w1, ws = meta.fit_two_stage(E(all_folds), SG(all_folds), LAM2)
    weights = {i: (ws[i] if i in own else w1) for i in meta.SEGMENTS}
    need = [n for j, n in enumerate(names) if max(w[j + 1] for w in weights.values()) >= 0.005]
    finals = {n: (meta.load_final(n) if n in need else np.zeros_like(v10_test)) for n in names}
    weights = {i: np.where(np.array([True] + [n in need for n in names]), w, 0.0) for i, w in weights.items()}
    weights = {i: w / w.sum() for i, w in weights.items()}
    seg_t = meta.segments(test, v10_test, wx)
    z = meta.apply_two_stage([v10_test] + [finals[n] for n in names], seg_t, weights)
    if use_eb:
        a = meta.eb_station_weights([stack[j] for j in all_folds], [base[j] for j in all_folds], [S.truth[j] for j in all_folds], EB_STRENGTH)
        z = v10_test + a[None, :, None] * (z - v10_test)
    else:
        off = S.station_sse(stack, all_folds) > S.station_sse(base, all_folds); a = np.where(off, 0.0, 1.0); z[:, off] = v10_test[:, off]
    kt = keep_mask(test)
    kt[int(np.flatnonzero(test.dates == pd.Timestamp('2026-09-12'))[0]), int(np.searchsorted(p.stations, 3139))] = True
    event_days = [dict(station=int(p.stations[j]), date=str(test.dates[d])[:10], v10=int(np.expm1(v10_test[d, j]).sum()), stack=int(np.expm1(z[d, j]).sum()))
                  for d, j in zip(*np.nonzero(kt.any(axis=2))) if j != m_idx]
    z = np.where(kt, v10_test, z)
    if park_test:
        z[:, park] = np.where(kt[:, park], z[:, park], spec['final__parks'][:, 0])
    new_test = centre(new_test, v10_test, test.dates)
    et, at = exit_cells(test)
    zb = z[:, b_idx].copy()
    zb = np.where(et, zb, zb + ws_test * (spec['final__bucheon'][:, 0] - zb))
    zb = np.where(et, zb + np.where(at >= BAND, wb_test[1], wb_test[0]) * (v10_test[:, b_idx] - zb), zb)
    bucheon_test = [dict(date=str(r.date)[:10], attendance=int(r.attendance), hours=[int(h) + 6 for h in np.flatnonzero(et[d])],
                         v10=[round(float(x)) for x in np.expm1(v10_test[d, b_idx][et[d]])], v11=[round(float(x)) for x in np.expm1(v11_test[d, b_idx][et[d]])],
                         v12=[round(float(x)) for x in np.expm1(zb[d][et[d]])])
                    for r in matches[matches.date.isin(test.dates)].sort_values('date').itertuples() for d in [int(np.flatnonzero(test.dates == r.date)[0])]]
    z[:, b_idx] = zb
    for s in np.flatnonzero(young(test)):
        s = int(s)
        zz = v10_test[:, s] + (w_new_test.get(s, 0.0) * (new_test[s] - v10_test[:, s]) if s in new_test else 0.0)
        z[:, s] = np.where(kt[:, s], v10_test[:, s], zz)
    z, event_log = s11.apply_events(test, z, cfg11.get('station_events', []))
    # one-off evening event at the grand park: attendance x conservative transit share x exit-flow shape after the end
    one_off = []
    for e in cfg.get('one_off_events', []):
        s = int(np.searchsorted(p.stations, e['station'])); d = int(np.flatnonzero(test.dates == pd.Timestamp(e['date']))[0])
        shape = np.array(e.get('flow_shape') or [0.0])
        end = actual.minutes(e['end']); flow = np.zeros(18)
        for j, share in enumerate(shape):                              # share of the leaving crowd in each 30-minute step after the end
            h = int((end + 30 * j) // 60) - 6
            if 0 <= h < 18:
                flow[h] += share
        add = e['attendance'] * e['transit_share'] * flow
        before = np.expm1(z[d, s]).copy(); z[d, s] = np.log1p(before + add)
        one_off.append(dict(event=e['name'], station=e['station'], date=e['date'], attendance=e['attendance'], transit_share=e['transit_share'],
                            boardings_17_23_before=[round(float(x)) for x in before[11:]], boardings_17_23_after=[round(float(x)) for x in np.expm1(z[d, s])[11:]]))
    flat = np.empty(z.size); flat[test.row_index] = np.maximum(z, 0).reshape(-1)
    frame = pd.DataFrame({'example_id': test.example_id, 'boardings': np.expm1(flat)})
    assert len(frame) == 157194 and np.isfinite(frame.boardings).all() and (frame.boardings >= 0).all()
    path = OUT / 'submission_ensemble_v12.csv'; frame.to_csv(path, index=False, float_format='%.8f')
    ref = np.log1p(pd.read_csv(c8.ROOT / 'outputs/ensemble_v11/submission_ensemble_v11.csv').boardings.to_numpy())
    out['test'] = dict(stage1_weights=dict(zip(members, [round(float(x), 3) for x in weights[4]])),
                       segment_weights={meta.SEGMENTS[i]: dict(zip(members, [round(float(x), 3) for x in weights[i]])) for i in own},
                       station_alpha={int(sid): round(float(x), 3) for sid, x in zip(p.stations, a) if x < 0.95}, models_used=need,
                       new_station_weight={int(p.stations[s]): round(w, 3) for s, w in w_new_test.items()}, bucheon_matches=bucheon_test,
                       parks_specialist=bool(park_test), events=event_log, one_off=one_off, event_days_kept_on_v10=event_days)
    out['submission'] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=int(len(frame)), mean_boardings=float(frame.boardings.mean()),
                             log_rmsd_vs_v11=float(np.sqrt(np.mean((flat - ref) ** 2))), mean_log_diff_vs_v11=float((flat - ref).mean()),
                             rows_changed_vs_v11=int((np.abs(flat - ref) > 1e-9).sum()))
    (OUT / 'gates_v12.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(out['submission'], indent=1), flush=True)
    return out


if __name__ == '__main__':
    main()
