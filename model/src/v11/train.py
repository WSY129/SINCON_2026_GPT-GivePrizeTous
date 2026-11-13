"""Train the v11 model families on the unique-day panel (GPU pod; `--smoke` runs a tiny CPU check).

    python -m src.v11.train --cutoffs f1,f2,f3,f4 --models lgb_a,lgb_b,xgb,cb,dec_lgb,dec_xgb,dec_cb,low_tw,low_po,tabm,stg
    python -m src.v11.train --cutoffs final --seeds 2026,2027 --models ...

No early stopping: every tree model is trained to the last round of a fixed grid and its forecast is stored at
every checkpoint, together with the validation RMSLE of each checkpoint.  The checkpoint used for a fold is chosen
later from the *other* folds' curves (src.v11.stack), so nothing is tuned on the fold it is scored on.

Families
  direct   lgb_a (31 leaves), lgb_b (127 leaves), xgb (CUDA, depth 8), cb (GPU, symmetric depth 8) on the hourly target
  dec_*    daily station total (weather, curve, attendance ...) + hourly share, both as residuals to the anchor
  low_*    Tweedie / Poisson count models with the anchor as offset for low-demand cells (anchor < 40 boardings)
  tabm     multitask TabM: boardings and alightings residuals from one trunk (K = 16 members)
  stg      multitask station-graph + hour-convolution network on whole days
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from .panel import Panel11, FEATURES, FI, CATEGORICAL, CAT_SIZE, flat_train, pred_rows

CUTOFFS = {'f1': '2025-08-30', 'f2': '2025-10-30', 'f3': '2025-12-30', 'f4': '2026-02-27', 'final': '2026-06-30'}
ROUNDS = [100, 200, 400, 700, 1000, 1500, 2200, 3000]
LR = 0.03
LOW = 40.0
THREADS = int(os.environ.get('V11_THREADS', '16'))
SMOKE = False
DAY_FEATS = ['sid', 'dow', 'cat', 'line', 'age', 'f1s', 'f1c', 'f2s', 'f2c', 'f3s', 'f3c', 'growth', 'rain_mm', 'wet', 'wet_ly', 'tmax', 'tdev', 'hot', 'heatwave',
             'comfort', 'rad_day', 'raddev', 'sun_hours', 'cloud_day', 'wind_max', 'outing', 'game_att', 'margin', 'rainout', 'event', 'commute']
NN_FLAGS = ['yoy', 'game_end_rel', 'margin', 'wet_ly', 'temp_h']


def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rounds():
    return [20, 40] if SMOKE else ROUNDS


class Data:
    def __init__(self, panel, name):
        self.name = name; self.d = d = panel.build(CUTOFFS[name])
        self.Xtr, self.r, self.ra, self.w, self.ok = flat_train(d)
        self.Xp, self.ap = pred_rows(d)                                    # [day, S, H, F], [day, S, H]
        self.shape = self.ap.shape
        O = d['O']; days = np.arange(O + 1, O + 1 + self.shape[0])
        truth = None
        if name != 'final':
            z = np.moveaxis(panel.p.Z[:, days], 0, 1).astype(float)
            if not d['basis2026']:                                           # fold scored on the 2025 basis (aligned truth)
                z = z + np.where((days >= panel.cut)[:, None, None], panel.logf[None], 0.0)
            truth = z
        self.truth = truth
        self.anchor_tr = np.moveaxis(d['anchor'][:, :O + 1], 0, 0)[self.ok]

    def score(self, r_hat):
        return None if self.truth is None else rmse(np.maximum(self.ap + r_hat, 0), self.truth)


# ---------------------------------------------------------------------------------------------- trees
def tree_fit(kind, X, y, w, seed, objective=None, init=None, leaves=31):
    n = rounds()[-1]
    if kind == 'lgb':
        import lightgbm as lgb
        params = dict(objective=objective or 'regression', learning_rate=LR, num_leaves=leaves, min_data_in_leaf=100 if leaves <= 31 else 200,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=THREADS, seed=seed, verbose=-1)
        if objective == 'tweedie':
            params['tweedie_variance_power'] = 1.2
        m = lgb.train(params, lgb.Dataset(X, y, weight=w, init_score=init), n)
        return lambda Z, k: m.predict(Z, num_iteration=k, num_threads=THREADS, raw_score=True)
    if kind == 'xgb':
        import xgboost as xgb
        m = xgb.train(dict(tree_method='hist', device=device(), max_depth=8, eta=LR, subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                           reg_lambda=5.0, objective='reg:squarederror', seed=seed, nthread=THREADS), xgb.DMatrix(X, y, weight=w, nthread=THREADS), n)
        return lambda Z, k: m.predict(xgb.DMatrix(Z, nthread=THREADS), iteration_range=(0, k))
    from catboost import CatBoostRegressor, Pool
    gpu = device() == 'cuda'
    m = CatBoostRegressor(task_type='GPU' if gpu else 'CPU', depth=8, learning_rate=LR, iterations=n, l2_leaf_reg=5.0, loss_function='RMSE', random_seed=seed,
                          verbose=0, thread_count=THREADS, allow_writing_files=False, **({'gpu_ram_part': 0.3} if gpu else {}))
    m.fit(Pool(X, y, weight=w))
    return lambda Z, k: m.predict(Z, ntree_end=k, thread_count=THREADS)


def checkpoints(D, predict, Xflat):
    out = np.stack([predict(Xflat, k).reshape(D.shape) for k in rounds()]).astype(np.float32)
    return out, [D.score(p) for p in out]


def fit_direct(D, name, seed):
    kind = name.split('_')[0]
    predict = tree_fit(kind, D.Xtr, D.r, D.w, seed, leaves=127 if name == 'lgb_b' else 31)
    return checkpoints(D, predict, D.Xp.reshape(-1, len(FEATURES)))


def daily_frame(X4, anchor, r=None):
    """[S, day, H, F] hourly panel -> station-day rows: day-level inputs plus aggregates of the hourly ones."""
    cols = [FI[f] for f in DAY_FEATS]
    base = X4[:, :, 8, :][:, :, cols]
    tot_anchor = np.log(np.expm1(np.clip(anchor, 0, None)).sum(2) + 1.0)
    with np.errstate(all='ignore'):
        agg = np.stack([tot_anchor, X4[..., FI['season']].mean(2), X4[..., FI['hol_eff']].mean(2), np.nanmean(X4[..., FI['yoy']], 2),
                        X4[..., FI['r_sat']].mean(2), X4[..., FI['r_sun']].mean(2), np.nanmax(X4[..., FI['game_end_rel']], 2)], -1)
    Xd = np.concatenate([base, agg], -1)
    rt = None
    if r is not None:
        rt = np.log(np.expm1(np.clip(anchor + r, 0, None)).sum(2) + 1.0) - tot_anchor      # NaN when an hour is missing
    return Xd, rt


def fit_decomp(D, name, seed):
    kind = name.split('_')[1]
    d = D.d; O = d['O']
    Xd, rt = daily_frame(d['X'][:, :O + 1], d['anchor'][:, :O + 1], d['r'][:, :O + 1])
    okd = ~np.isnan(rt)
    wd = np.broadcast_to((0.5 ** (d['age'][:O + 1] / 240.0))[None, :], rt.shape)
    total = tree_fit(kind, Xd[okd], rt[okd], wd[okd].astype(np.float32), seed)
    k_total = min(700, rounds()[-1])
    rt_fit = np.where(okd, rt, 0.0)
    share_target = (d['r'][:, :O + 1] - rt_fit[:, :, None])[D.ok]
    keep = np.broadcast_to(okd[:, :, None], D.ok.shape)[D.ok]
    share = tree_fit(kind, D.Xtr[keep], share_target[keep], D.w[keep], seed + 1)
    Xdp, _ = daily_frame(d['X'][:, O + 1:], d['anchor'][:, O + 1:])
    tp = total(Xdp.reshape(-1, Xdp.shape[-1]), k_total).reshape(Xdp.shape[:2]).T              # [day, S]
    flat = D.Xp.reshape(-1, len(FEATURES))
    out = np.stack([tp[:, :, None] + share(flat, k).reshape(D.shape) for k in rounds()]).astype(np.float32)
    return out, [D.score(p) for p in out]


def fit_low(D, name, seed, fallback):
    """Count model with the anchor as offset on low-demand cells; other cells use the direct LightGBM forecast."""
    objective = 'tweedie' if name == 'low_tw' else 'poisson'
    counts = np.expm1(np.clip(D.anchor_tr + D.r, 0, None))
    sel = np.expm1(np.clip(D.anchor_tr, 0, None)) < LOW
    offset = np.log(np.expm1(np.clip(D.anchor_tr[sel], 0, None)) + 0.5)
    predict = tree_fit('lgb', D.Xtr[sel], counts[sel], D.w[sel], seed, objective=objective, init=offset)
    low = np.expm1(np.clip(D.ap, 0, None)) < LOW
    flat = D.Xp.reshape(-1, len(FEATURES))[low.ravel()]
    off_p = np.log(np.expm1(np.clip(D.ap[low], 0, None)) + 0.5)
    out = []
    for i, k in enumerate(rounds()):
        mu = np.exp(off_p + predict(flat, k))
        z = np.log1p(mu) - 0.5 * mu / (1.0 + mu) ** 2                    # E[log1p Y] for a Poisson-like count with mean mu
        p = fallback[i].copy(); p[low] = z - D.ap[low]
        out.append(p)
    out = np.stack(out).astype(np.float32)
    return out, [D.score(p) for p in out]


# ------------------------------------------------------------------------------------------ neural nets
def nn_arrays(D):
    import torch
    num = [f for f in FEATURES if f not in CATEGORICAL]
    cat = CATEGORICAL
    ni = [FI[f] for f in num]; fi = [FI[f] for f in NN_FLAGS]
    mean = np.nanmean(D.Xtr[:, ni], 0); std = np.maximum(np.nanstd(D.Xtr[:, ni], 0), 1e-3)
    mean = np.where(np.isnan(mean), 0, mean); std = np.where(np.isnan(std), 1, std)

    def prep(X):
        x = np.clip((X[..., ni] - mean) / std, -6, 6)
        x = np.concatenate([np.nan_to_num(x), np.isnan(X[..., fi]).astype(np.float32)], -1).astype(np.float32)
        c = np.stack([np.clip(np.nan_to_num(X[..., FI[f]]), 0, CAT_SIZE[f] - 1) for f in cat], -1).astype(np.int64)
        return x, c
    return prep, len(ni) + len(fi), [CAT_SIZE[f] for f in cat]


def fit_tabm(D, seed):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    dev = device(); torch.manual_seed(seed)
    prep, n_num, cats = nn_arrays(D)
    xn, xc = prep(D.Xtr)
    xn = torch.from_numpy(xn).to(dev); xc = torch.from_numpy(xc).to(dev)
    y = torch.from_numpy(np.stack([D.r, np.clip(D.ra, -3, 3)], 1).astype(np.float32)).to(dev); w = torch.from_numpy(D.w).to(dev)
    K, hidden = 16, 384

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(c, min(8, (c + 1) // 2)) for c in cats])
            dim = n_num + sum(e.embedding_dim for e in self.emb)
            self.scale = nn.Parameter(1 + 0.1 * torch.randn(K, dim))
            self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, hidden), nn.ReLU())
            self.w = nn.Parameter(torch.randn(K, hidden, 2) / hidden ** 0.5); self.b = nn.Parameter(torch.zeros(K, 2))

        def forward(self, xn, xc):
            x = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], -1)
            h = self.trunk(x[:, None, :] * self.scale[None])
            return torch.einsum('bkh,kht->bkt', h, self.w) + self.b                # [B, K, 2 tasks]
    net = Net().to(dev)
    epochs = 2 if SMOKE else 24; batch = 4096; steps = int(np.ceil(len(y) / batch)); total = epochs * steps
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / (2 * steps), 1.0) * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * min(s / total, 1.0)))))
    pn, pc = prep(D.Xp.reshape(-1, len(FEATURES))); pn = torch.from_numpy(pn).to(dev); pc = torch.from_numpy(pc).to(dev)

    def predict():
        net.eval()
        with torch.no_grad():
            out = torch.cat([net(pn[i:i + 65536], pc[i:i + 65536]).mean(1)[:, 0] for i in range(0, len(pn), 65536)])
        return out.reshape(D.shape).float().cpu().numpy()
    log = dict(train_loss=[], valid=[]); swa, n_swa = None, 0
    gen = torch.Generator(device=dev).manual_seed(seed)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(y), device=dev, generator=gen); tot = 0.0
        for i in range(steps):
            idx = perm[i * batch:(i + 1) * batch]
            out = net(xn[idx], xc[idx])
            err = Fn.huber_loss(out, y[idx][:, None, :].expand(-1, K, -1), delta=0.5, reduction='none')
            loss = ((err[..., 0] + 0.3 * err[..., 1]).mean(1) * w[idx]).sum() / w[idx].sum()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step(); tot += float(loss.detach())
        log['train_loss'].append(round(tot / steps, 5)); log['valid'].append(D.score(predict()))
        if ep >= epochs - 5:
            state = {k: v.detach().clone().double() for k, v in net.state_dict().items() if v.is_floating_point()}
            swa = state if swa is None else {k: swa[k] + state[k] for k in swa}; n_swa += 1
    sd = net.state_dict(); sd.update({k: (v / n_swa).to(sd[k].dtype) for k, v in swa.items()}); net.load_state_dict(sd)
    p = predict()
    return p[None].astype(np.float32), [D.score(p)], log


def fit_stg(D, seed, adjacency):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    dev = device(); torch.manual_seed(seed)
    prep, n_num, cats = nn_arrays(D)
    d = D.d; O = d['O']
    Xd = np.moveaxis(d['X'][:, :O + 1], 0, 1)                                     # [day, S, H, F]
    xn, xc = prep(Xd); xn = torch.from_numpy(xn).to(dev); xc = torch.from_numpy(xc).to(dev)
    r = np.moveaxis(d['r'][:, :O + 1], 0, 1); ra = np.moveaxis(d['ra'][:, :O + 1], 0, 1)
    mask = torch.from_numpy((~np.isnan(r)).astype(np.float32)).to(dev)
    y = torch.from_numpy(np.stack([np.nan_to_num(r), np.clip(np.nan_to_num(ra), -3, 3)], -1).astype(np.float32)).to(dev)
    wd = torch.from_numpy((0.5 ** (d['age'][:O + 1] / 240.0)).astype(np.float32)).to(dev)
    A = torch.tensor(adjacency, device=dev); hidden = 128

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(c, min(8, (c + 1) // 2)) for c in cats])
            dim = n_num + sum(e.embedding_dim for e in self.emb)
            self.inp = nn.Linear(dim, hidden)
            self.gc = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)]); self.tc = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1) for _ in range(2)])
            self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
            self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

        def forward(self, xn, xc):
            h = Fn.relu(self.inp(torch.cat([xn] + [e(xc[..., i]) for i, e in enumerate(self.emb)], -1)))
            B, S, H, Dm = h.shape
            for gc, tc, norm in zip(self.gc, self.tc, self.norm):
                g = gc(torch.einsum('ij,bjhd->bihd', A, h))
                t = tc(h.reshape(B * S, H, Dm).transpose(1, 2)).transpose(1, 2).reshape(B, S, H, Dm)
                h = norm(h + Fn.dropout(Fn.relu(g + t), 0.2, self.training))
            return self.out(h)
    net = Net().to(dev)
    n_days = xn.shape[0]; batch = 16; steps = int(np.ceil(n_days / batch)); epochs = 2 if SMOKE else max(40, int(3000 / steps)); total = epochs * steps
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / (3 * steps), 1.0) * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * min(s / total, 1.0)))))
    pn, pc = prep(D.Xp); pn = torch.from_numpy(pn).to(dev); pc = torch.from_numpy(pc).to(dev)

    def predict():
        net.eval()
        with torch.no_grad():
            out = torch.cat([net(pn[i:i + 32], pc[i:i + 32])[..., 0] for i in range(0, pn.shape[0], 32)])
        return out.float().cpu().numpy()
    log = dict(valid=[]); swa, n_swa = None, 0
    gen = torch.Generator(device=dev).manual_seed(seed)
    last = max(epochs // 5, 1)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n_days, device=dev, generator=gen)
        for i in range(steps):
            idx = perm[i * batch:(i + 1) * batch]
            err = Fn.huber_loss(net(xn[idx], xc[idx]), y[idx], delta=0.5, reduction='none')
            wt = mask[idx] * wd[idx][:, None, None]
            loss = ((err[..., 0] + 0.3 * err[..., 1]) * wt).sum() / wt.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            log['valid'].append(D.score(predict()))
        if ep >= epochs - last:
            state = {k: v.detach().clone().double() for k, v in net.state_dict().items() if v.is_floating_point()}
            swa = state if swa is None else {k: swa[k] + state[k] for k in swa}; n_swa += 1
    sd = net.state_dict(); sd.update({k: (v / n_swa).to(sd[k].dtype) for k, v in swa.items()}); net.load_state_dict(sd)
    p = predict()
    return p[None].astype(np.float32), [D.score(p)], dict(log, epochs=epochs)


def adjacency(panel, O):
    T = panel.p.T[:, :O + 1]; diff = T[:, 7:] - T[:, :-7]
    corr = np.ma.corrcoef(np.ma.masked_invalid(diff)).filled(0); np.fill_diagonal(corr, -1)
    A = np.zeros((panel.S, panel.S), np.float32)
    for i in range(panel.S):
        A[i, np.argsort(corr[i])[-4:]] = 1
        for j in (i - 1, i + 1):
            if 0 <= j < panel.S and panel.p.line[i] == panel.p.line[j]:
                A[i, j] = 1
    A = np.maximum(A, A.T) + np.eye(panel.S, dtype=np.float32)
    return A / A.sum(1, keepdims=True)


def main():
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoffs', default='f1,f2,f3,f4')
    ap.add_argument('--models', default='lgb_a,lgb_b,xgb,cb,dec_lgb,dec_xgb,dec_cb,low_tw,low_po,tabm,stg')
    ap.add_argument('--seeds', default='2026')
    ap.add_argument('--out', default='outputs/ensemble_v11/models')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(); SMOKE = args.smoke
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    panel = Panel11()
    for name in args.cutoffs.split(','):
        t0 = time.time(); D = Data(panel, name)
        print(f'[{name}] panel {time.time() - t0:.1f}s train rows {len(D.r)}', flush=True)
        for seed in [int(s) for s in args.seeds.split(',')]:
            cache = {}
            for model in args.models.split(','):
                tag = f'{name}__{model}__{seed}'
                if (out / f'{tag}.npz').exists():
                    continue
                t0 = time.time(); extra = {}
                if model in ('lgb_a', 'lgb_b', 'xgb', 'cb'):
                    pred, curve = fit_direct(D, model, seed)
                    if model == 'lgb_a':
                        cache['lgb_a'] = pred
                elif model.startswith('dec_'):
                    pred, curve = fit_decomp(D, model, seed)
                elif model.startswith('low_'):
                    if 'lgb_a' not in cache:
                        cache['lgb_a'] = np.load(out / f'{name}__lgb_a__{seed}.npz')['pred'] if (out / f'{name}__lgb_a__{seed}.npz').exists() else fit_direct(D, 'lgb_a', seed)[0]
                    pred, curve = fit_low(D, model, seed, cache['lgb_a'])
                elif model == 'tabm':
                    pred, curve, extra = fit_tabm(D, seed)
                else:
                    pred, curve, extra = fit_stg(D, seed, adjacency(panel, D.d['O']))
                np.savez_compressed(out / f'{tag}.npz', pred=pred, anchor=D.ap.astype(np.float32))
                info = dict(model=model, cutoff=name, seed=seed, rounds=rounds() if pred.shape[0] > 1 else None, valid_rmsle=curve, seconds=round(time.time() - t0, 1),
                            n_train=int(len(D.r)), features=len(FEATURES), **extra)
                (out / f'{tag}.json').write_text(json.dumps(info, indent=1, default=float))
                best = None if curve[0] is None else int(np.argmin(curve))
                print(f'[{name}] {model} seed {seed} {info["seconds"]}s valid {[None if c is None else round(c, 4) for c in curve]} best_idx {best}', flush=True)


if __name__ == '__main__':
    main()
