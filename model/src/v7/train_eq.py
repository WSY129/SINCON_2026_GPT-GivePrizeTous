"""Train the v7 equilibrium model family (designed for the RTX 4090 pod; `--smoke` runs a tiny CPU check).

Every model predicts r_eq = log1p(y) - mu_eq from the panel features.  Trees stop on the fold's evaluation
period for monitoring only: the strict (LOFO) prediction of a fold uses the round count averaged over the
*other* folds, and neural nets use the fixed schedule with SWA, so no strict prediction is tuned on its own fold.

  python -m src.v7.train_eq cv --models lgb            # CPU stream
  python -m src.v7.train_eq cv --models xgb,cb,tabm,mdn,stg,rkhs
  python -m src.v7.train_eq finalize                   # strict predictions + summary
  python -m src.v7.train_eq final --models ...         # 2 seeds at origin 2026-06-30
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .panel import Panel, FEATURES, FI, YOY, CATEGORICAL, HORIZON, MIN_HISTORY, load_academic

FOLDS = {'f1': ('2025-08-30', 123), 'f2': ('2025-10-30', 123), 'f3': ('2025-12-30', 123),
         'f4': ('2026-02-27', 123), 'hot26': ('2026-05-15', 46), 'summer25': ('2025-06-30', 123)}
MAIN = ['f1', 'f2', 'f3', 'f4']
FINAL_ORIGIN = '2026-06-30'
DROP = ['month', 'doy_sin', 'doy_cos']            # absolute season is never seen for the evaluation months
FLAGS = ['ly_rel', 'ly_seas', 'long16', 'trend4']
CAT_SIZE = dict(sid=71, line=3, hour=18, dow=7, cls=7, month=13, holiday_type=24, kbo_state=6)
TREE = dict(lr=0.025, rounds=4000, patience=250)
NN = dict(tabm=dict(epochs=30, patience=7), mdn=dict(epochs=25, patience=6), stg=dict(epochs=25, patience=6))
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1]
# Neural-net regularisation after the first run tripped the early-stop alarm (best epoch 2-5, then the nets
# memorised station-specific seasonal drift of the training months): no station embedding, stronger dropout and
# weight decay, lower peak learning rate, Huber loss.
NNREG = dict(lr=6e-4, weight_decay=3e-2, dropout=0.3, huber_delta=0.4)
THREADS = int(os.environ.get('V7_THREADS', '8'))
SMOKE = False


def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def make_data(panel, origin, horizon, seed, pairs=2400, labelled=True):
    O = panel.day_index(origin)
    total = sum(min(HORIZON, O - o) for o in range(O - 7, MIN_HISTORY - 2, -7))
    keep = min(1.0, (60 if SMOKE else pairs) / total)
    Xtr, mutr, ytr, ktr = panel.training_set(origin, keep=keep, seed=seed)
    Xev, muev, yev, kev = panel.eval_set(origin, horizon)
    T = panel.T[:, :O + 1]
    diff = T[:, 7:] - T[:, :-7]
    corr = np.ma.corrcoef(np.ma.masked_invalid(diff)).filled(0)
    np.fill_diagonal(corr, -1)
    A = np.zeros((panel.S, panel.S), np.float32)
    for i in range(panel.S):
        A[i, np.argsort(corr[i])[-4:]] = 1
        for j in (i - 1, i + 1):
            if 0 <= j < panel.S and panel.line[i] == panel.line[j]:
                A[i, j] = 1
    A = np.maximum(A, A.T) + np.eye(panel.S, dtype=np.float32)
    A /= A.sum(1, keepdims=True)
    return dict(Xtr=Xtr, mutr=mutr, ytr=ytr, Xev=Xev, muev=muev, yev=yev, A=A, origin=origin,
                mtr=~np.isnan(ytr) & ~np.isnan(mutr), mev=~np.isnan(yev) & ~np.isnan(muev))


IDENT = ['sid', 'line', 'age', 'mu', 'mu_tot', 'share']       # absolute identity / level of a station


def columns(variant):
    """Feature variants: '' default, 's' adds absolute season, 'r' removes station identity, 'rs' both,
    'noyoy' removes the year-over-year block."""
    drop = set() if 's' in variant and variant != 'noyoy' else set(DROP)
    if variant in ('r', 'rs'):
        drop |= set(IDENT)
    if variant == 'noyoy':
        drop |= set(YOY)
    names = [f for f in FEATURES if f not in drop]
    return names, [FI[f] for f in names]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def z_of(d, r_hat):
    return np.maximum(d['muev'] + r_hat, 0).astype(np.float32)


def curve(values, step):
    return [round(float(v), 5) for v in values[::step]]


# ---------------------------------------------------------------------------------------------- trees
def flat(d, cols):
    F = len(FEATURES)
    Xtr = d['Xtr'].reshape(-1, F)[d['mtr'].ravel()][:, cols]
    rtr = (d['ytr'] - d['mutr'])[d['mtr']]
    Xev = d['Xev'].reshape(-1, F)[:, cols]
    return Xtr, rtr, Xev


def fit_tree(kind, d, cols, seed, rounds=None):
    """rounds=None: early stopping on the evaluation period (CV). Returns (model, info, predict(n) -> r_hat)."""
    Xtr, rtr, Xev = flat(d, cols)
    mev = d['mev'].ravel(); rev = (d['yev'] - d['muev']).ravel()
    cv = rounds is None
    max_rounds = (60 if SMOKE else TREE['rounds']) if cv else rounds
    patience = 20 if SMOKE else TREE['patience']
    shape = d['muev'].shape
    info = dict(n_train=int(len(rtr)))
    if kind == 'lgb':
        import lightgbm as lgb
        params = dict(objective='regression', learning_rate=TREE['lr'], num_leaves=63, min_data_in_leaf=300,
                      feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, max_bin=255,
                      num_threads=THREADS, seed=seed, verbose=-1)
        dtr = lgb.Dataset(Xtr, rtr)
        hist, sets, names, cbs = {}, [dtr], ['training'], []
        if cv:
            sets.append(lgb.Dataset(Xev[mev], rev[mev], reference=dtr)); names.append('valid')
            cbs = [lgb.early_stopping(patience, verbose=False), lgb.record_evaluation(hist)]
        model = lgb.train(params, dtr, max_rounds, valid_sets=sets, valid_names=names, callbacks=cbs,
                          keep_training_booster=True)
        if cv:
            info.update(best=int(model.best_iteration), trained=int(model.current_iteration()),
                        train_curve=curve(hist['training']['l2'], 25), valid_curve=curve(hist['valid']['l2'], 25))
        predict = lambda n=None: model.predict(Xev, num_iteration=n, num_threads=THREADS).reshape(shape)
        save = lambda path: model.save_model(str(path) + '.txt', num_iteration=-1)
    elif kind == 'xgb':
        import xgboost as xgb
        params = dict(tree_method='hist', device=device(), max_depth=8, eta=TREE['lr'], subsample=0.7,
                      colsample_bytree=0.7, min_child_weight=100, reg_lambda=10.0, max_bin=256,
                      objective='reg:squarederror', eval_metric='rmse', seed=seed, nthread=THREADS)
        dtr = xgb.QuantileDMatrix(Xtr, rtr, max_bin=256, nthread=THREADS)
        dall = xgb.DMatrix(Xev, nthread=THREADS)
        res, kw = {}, {}
        if cv:
            dva = xgb.QuantileDMatrix(Xev[mev], rev[mev], ref=dtr, nthread=THREADS)
            kw = dict(evals=[(dtr, 'train'), (dva, 'valid')], early_stopping_rounds=patience, evals_result=res)
        model = xgb.train(params, dtr, max_rounds, verbose_eval=False, **kw)
        if cv:
            info.update(best=int(model.best_iteration) + 1, trained=int(model.num_boosted_rounds()),
                        train_curve=curve(np.square(res['train']['rmse']), 25),
                        valid_curve=curve(np.square(res['valid']['rmse']), 25))
        predict = lambda n=None: model.predict(dall, iteration_range=(0, n or 0)).reshape(shape)
        save = lambda path: model.save_model(str(path) + '.ubj')
    else:
        from catboost import CatBoostRegressor, Pool
        gpu = device() == 'cuda'
        model = CatBoostRegressor(task_type='GPU' if gpu else 'CPU', depth=8, learning_rate=TREE['lr'],
                                  iterations=max_rounds, l2_leaf_reg=5.0, border_count=128, loss_function='RMSE',
                                  od_type='Iter', od_wait=patience, use_best_model=False, random_seed=seed,
                                  verbose=0, thread_count=THREADS, allow_writing_files=False)
        if cv:
            model.fit(Pool(Xtr, rtr), eval_set=Pool(Xev[mev], rev[mev]))
            ev = model.get_evals_result()
            info.update(best=int(model.get_best_iteration()) + 1, trained=int(model.tree_count_),
                        train_curve=curve(np.square(ev['learn']['RMSE']), 25),
                        valid_curve=curve(np.square(ev['validation']['RMSE']), 25))
        else:
            model.fit(Pool(Xtr, rtr))
        predict = lambda n=None: model.predict(Xev, ntree_end=n or 0, thread_count=THREADS).reshape(shape)
        save = lambda path: model.save_model(str(path) + '.cbm')
    if cv:
        info['valid_rmse_best'] = round(float(np.sqrt(min(info['valid_curve']))), 5)
        info['alarm_early_stop'] = bool(info['best'] < 200)
        info['note'] = 'curves are mean squared error of r_eq every 25 rounds; for an L2 objective the per-row gradient L2 norm equals sqrt(train mse)'
    return model, info, predict, save


def load_tree(kind, path):
    if kind == 'lgb':
        import lightgbm as lgb
        m = lgb.Booster(model_file=str(path) + '.txt')
        return lambda X, n: m.predict(X, num_iteration=n, num_threads=THREADS)
    if kind == 'xgb':
        import xgboost as xgb
        m = xgb.Booster(model_file=str(path) + '.ubj')
        m.set_param({'device': device()})
        return lambda X, n: m.predict(xgb.DMatrix(X), iteration_range=(0, n))
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(); m.load_model(str(path) + '.cbm')
    return lambda X, n: m.predict(X, ntree_end=n, thread_count=THREADS)


# ---------------------------------------------------------------------------------------- neural nets
def nn_arrays(d, names):
    """Dense standardised arrays (n, S, H, .) for train and evaluation."""
    import torch
    num = [f for f in names if f not in CATEGORICAL]
    cat = [f for f in CATEGORICAL if f in names]
    flags = [f for f in FLAGS if f in names]
    out = {}
    tr = d['Xtr'][..., [FI[f] for f in num]]
    mean = np.nanmean(tr[d['mtr']], 0); std = np.maximum(np.nanstd(tr[d['mtr']], 0), 1e-3)
    mean = np.where(np.isnan(mean), 0, mean); std = np.where(np.isnan(std), 1, std)
    for key in ('tr', 'ev'):
        X = d['X' + key]
        x = np.clip((X[..., [FI[f] for f in num]] - mean) / std, -6, 6)
        fl = np.isnan(X[..., [FI[f] for f in flags]]).astype(np.float32)
        x = np.concatenate([np.nan_to_num(x, nan=0.0), fl], -1).astype(np.float32)
        c = np.stack([np.nan_to_num(X[..., FI[f]], nan=0) + (1 if f == 'kbo_state' else 0) for f in cat], -1)
        c = np.clip(c, 0, [CAT_SIZE[f] - 1 for f in cat]).astype(np.int64)
        out['num_' + key] = torch.from_numpy(x).to(device()); out['cat_' + key] = torch.from_numpy(c).to(device())
    r = np.nan_to_num(d['ytr'] - d['mutr'], nan=0.0).astype(np.float32)
    out.update(r=torch.from_numpy(r).to(device()), mask=torch.from_numpy(d['mtr']).to(device()),
               mu_ev=torch.from_numpy(d['muev']).to(device()), n_num=x.shape[-1],
               cat_sizes=[CAT_SIZE[f] for f in cat])
    return out


def build_net(kind, a, A):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class Embed(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(c, min(8, (c + 1) // 2)) for c in a['cat_sizes']])
            self.dim = a['n_num'] + sum(e.embedding_dim for e in self.emb)

        def forward(self, xn, xc):
            return torch.cat([xn] + [e(xc[..., i]) for i, e in enumerate(self.emb)], -1)

    class TabM(nn.Module):                      # BatchEnsemble-style: K members share the trunk
        def __init__(self, k=16, hidden=384):
            super().__init__()
            self.e = Embed(); self.k = k
            self.scale = nn.Parameter(1 + 0.1 * torch.randn(k, self.e.dim))
            self.trunk = nn.Sequential(nn.Linear(self.e.dim, hidden), nn.ReLU(), nn.Dropout(NNREG['dropout']),
                                       nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(NNREG['dropout']),
                                       nn.Linear(hidden, hidden), nn.ReLU())
            self.w = nn.Parameter(torch.randn(k, hidden) / hidden ** 0.5); self.b = nn.Parameter(torch.zeros(k))

        def members(self, xn, xc):
            h = self.trunk(self.e(xn, xc)[:, None, :] * self.scale[None])
            return (h * self.w[None]).sum(-1) + self.b

        def loss(self, xn, xc, r):
            return Fn.huber_loss(self.members(xn, xc), r[:, None].expand(-1, self.k), delta=NNREG['huber_delta'])

        def predict(self, xn, xc, mu):
            return self.members(xn, xc).mean(1)

    class MDN(nn.Module):                       # 3-component Gaussian mixture on r_eq
        def __init__(self, hidden=384, comps=3):
            super().__init__()
            self.e = Embed()
            self.trunk = nn.Sequential(nn.Linear(self.e.dim, hidden), nn.ReLU(), nn.Dropout(NNREG['dropout']),
                                       nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(NNREG['dropout']),
                                       nn.Linear(hidden, hidden), nn.ReLU())
            self.head = nn.Linear(hidden, 3 * comps)
            with torch.no_grad():
                self.head.bias[comps:2 * comps] = torch.tensor([-0.3, 0.0, 0.3])
            x, w = np.polynomial.hermite.hermgauss(15)
            self.register_buffer('gx', torch.tensor(x, dtype=torch.float32))
            self.register_buffer('gw', torch.tensor(w / np.sqrt(np.pi), dtype=torch.float32))

        def parts(self, xn, xc):
            logit, mean, raw = self.head(self.trunk(self.e(xn, xc))).chunk(3, -1)
            return Fn.log_softmax(logit, -1), mean, 0.05 + Fn.softplus(raw)        # sigma >= 0.05

        def loss(self, xn, xc, r):
            logpi, mean, sigma = self.parts(xn, xc)
            logn = -0.5 * ((r[:, None] - mean) / sigma) ** 2 - torch.log(sigma) - 0.9189385
            nll = -torch.logsumexp(logpi + logn, -1).mean()
            entropy = -(logpi.exp() * logpi).sum(-1).mean()
            return nll - 0.01 * entropy

        def predict(self, xn, xc, mu):          # E[max(0, Z)] by 15-point Gauss-Hermite, returned as r_hat
            logpi, mean, sigma = self.parts(xn, xc)
            z = mu[:, None, None] + mean[:, :, None] + 1.4142135 * sigma[:, :, None] * self.gx
            return (logpi.exp() * (Fn.relu(z) * self.gw).sum(-1)).sum(-1) - mu

        def stats(self, xn, xc):
            logpi, mean, sigma = self.parts(xn, xc)
            return logpi.exp().mean(0), mean.mean(0), sigma.mean(0), sigma.min()

    class STG(nn.Module):                       # station-graph convolution + convolution along the hours
        def __init__(self, hidden=128, blocks=2):
            super().__init__()
            self.e = Embed()
            self.inp = nn.Linear(self.e.dim, hidden)
            self.gc = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(blocks)])
            self.tc = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1) for _ in range(blocks)])
            self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(blocks)])
            self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.register_buffer('A', torch.tensor(A))

        def forward(self, xn, xc):
            h = Fn.relu(self.inp(self.e(xn, xc)))                                  # (B, S, H, d)
            B, S, H, D = h.shape
            for gc, tc, norm in zip(self.gc, self.tc, self.norm):
                g = gc(torch.einsum('ij,bjhd->bihd', self.A, h))
                t = tc(h.reshape(B * S, H, D).transpose(1, 2)).transpose(1, 2).reshape(B, S, H, D)
                h = norm(h + Fn.dropout(Fn.relu(g + t), NNREG['dropout'], self.training))
            return self.out(h).squeeze(-1)

    torch.manual_seed(a['seed'])
    return {'tabm': TabM, 'mdn': MDN, 'stg': STG}[kind]().to(device())


def fit_nn(kind, d, names, seed, cv=True):
    import torch
    import torch.nn.functional as Fn
    cfg = NN[kind]
    names = [f for f in names if f != 'sid']
    epochs = 2 if SMOKE else cfg['epochs']
    a = nn_arrays(d, names); a['seed'] = seed
    net = build_net(kind, a, d['A'])
    dense = kind == 'stg'
    gen = torch.Generator(device=device()).manual_seed(seed)
    if dense:
        items = torch.arange(a['num_tr'].shape[0], device=device()); batch = 16
        xn, xc, r, mask = a['num_tr'], a['cat_tr'], a['r'], a['mask']
    else:
        m = a['mask'].reshape(-1)
        xn = a['num_tr'].reshape(-1, a['n_num'])[m]; xc = a['cat_tr'].reshape(-1, a['cat_tr'].shape[-1])[m]
        r = a['r'].reshape(-1)[m]; items = torch.arange(len(r), device=device()); batch = 4096
    steps = int(np.ceil(len(items) / batch)); total = epochs * steps; warm = 3 * steps
    opt = torch.optim.AdamW(net.parameters(), lr=NNREG['lr'], weight_decay=NNREG['weight_decay'])
    lr_at = lambda s: min((s + 1) / warm, 1.0) * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * min(s / total, 1.0))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    shape = d['muev'].shape

    def predict():
        net.eval()
        with torch.no_grad():
            if dense:
                out = torch.cat([net(a['num_ev'][i:i + 32], a['cat_ev'][i:i + 32])
                                 for i in range(0, shape[0], 32)])
            else:
                en = a['num_ev'].reshape(-1, a['n_num']); ec = a['cat_ev'].reshape(-1, a['cat_ev'].shape[-1])
                mu = a['mu_ev'].reshape(-1)
                out = torch.cat([net.predict(en[i:i + 65536], ec[i:i + 65536], mu[i:i + 65536])
                                 for i in range(0, len(en), 65536)])
        return out.reshape(shape).float().cpu().numpy()

    log = dict(train_loss=[], grad_norm=[], valid_rmsle=[])
    best, best_pred, best_epoch, swa, swa_n = np.inf, None, 0, None, 0
    for epoch in range(epochs):
        net.train()
        perm = items[torch.randperm(len(items), device=device(), generator=gen)]
        tot, gsum = 0.0, 0.0
        for i in range(steps):
            idx = perm[i * batch:(i + 1) * batch]
            if dense:
                err = Fn.huber_loss(net(xn[idx], xc[idx]), r[idx], delta=NNREG['huber_delta'], reduction='none')
                loss = (err * mask[idx]).sum() / mask[idx].sum().clamp(min=1)
            else:
                loss = net.loss(xn[idx], xc[idx], r[idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            gsum += float(torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0))
            opt.step(); sched.step(); tot += float(loss)
        log['train_loss'].append(round(tot / steps, 5)); log['grad_norm'].append(round(gsum / steps, 5))
        if epoch >= epochs - 5:                                      # SWA over the last five epochs
            state = {k: v.detach().clone().double() for k, v in net.state_dict().items() if v.is_floating_point()}
            swa = state if swa is None else {k: swa[k] + state[k] for k in swa}; swa_n += 1
        if cv:
            pred = predict()
            score = rmse(z_of(d, pred)[d['mev']], d['yev'][d['mev']])
            log['valid_rmsle'].append(round(score, 5))
            if score < best:
                best, best_pred, best_epoch = score, pred, epoch + 1
    state = net.state_dict()
    state.update({k: (v / swa_n).to(state[k].dtype) for k, v in swa.items()})
    net.load_state_dict(state)
    swa_pred = predict()
    info = dict(log, epochs=epochs, n_train=int(len(items)), swa_epochs=swa_n, regularisation=NNREG)
    if cv:
        since = [e for e in range(best_epoch, epochs) if e - best_epoch >= cfg['patience']]
        info.update(best_epoch=best_epoch, valid_rmsle_best=round(best, 5),
                    valid_rmsle_swa=round(rmse(z_of(d, swa_pred)[d['mev']], d['yev'][d['mev']]), 5),
                    patience_stop_epoch=(since[0] + 1) if since else None,
                    alarm_early_stop=bool(best_epoch < 5))
    if kind == 'mdn':
        with torch.no_grad():
            en = a['num_ev'].reshape(-1, a['n_num'])[:200000]; ec = a['cat_ev'].reshape(-1, a['cat_ev'].shape[-1])[:200000]
            pi, mean, sigma, smin = net.stats(en, ec)
        info['mdn'] = dict(mix_weight=[round(float(v), 4) for v in pi], mean=[round(float(v), 4) for v in mean],
                           sigma=[round(float(v), 4) for v in sigma], sigma_min=round(float(smin), 4),
                           sigma_floor=0.05, entropy_weight=0.01, quadrature_points=15)
    del a
    torch.cuda.empty_cache() if device() == 'cuda' else None
    return info, best_pred, swa_pred


def fit_rkhs(d, names, seed, lambdas=LAMBDAS, landmarks=4096):
    """Nystrom kernel ridge with a Laplacian + RBF kernel on 4096 landmarks, solved by Cholesky."""
    import torch
    a = nn_arrays(d, [f for f in names if f not in ('sid', 'line')] )
    extra = lambda key: torch.cat([a['num_' + key], a['cat_' + key].float() / 6.0], -1)
    m = a['mask'].reshape(-1)
    X = extra('tr'); X = X.reshape(-1, X.shape[-1])[m]; r = a['r'].reshape(-1)[m].double()
    E = extra('ev'); E = E.reshape(-1, E.shape[-1])
    gen = torch.Generator(device=device()).manual_seed(seed)
    n_land = 256 if SMOKE else landmarks
    L = X[torch.randperm(len(X), device=device(), generator=gen)[:n_land]]
    g1 = 1.0 / torch.cdist(L[:1024], L[:1024], p=1).median()
    g2 = 1.0 / torch.cdist(L[:1024], L[:1024]).median() ** 2
    kern = lambda B: 0.5 * torch.exp(-g1 * torch.cdist(B, L, p=1)) + 0.5 * torch.exp(-g2 * torch.cdist(B, L) ** 2)
    G = torch.zeros(n_land, n_land, dtype=torch.float64, device=device())
    b = torch.zeros(n_land, dtype=torch.float64, device=device())
    for i in range(0, len(X), 8192):
        K = kern(X[i:i + 8192]).double(); G += K.T @ K; b += K.T @ r[i:i + 8192]
    Kmm = kern(L).double(); eye = torch.eye(n_land, dtype=torch.float64, device=device())
    scale = torch.trace(G) / torch.trace(Kmm)
    preds, info = {}, dict(n_train=int(len(X)), landmarks=n_land, gamma_laplace=float(g1), gamma_rbf=float(g2), valid_rmsle={})
    for lam in lambdas:
        chol = torch.linalg.cholesky(G + lam * scale * Kmm + 1e-8 * scale * eye)
        alpha = torch.cholesky_solve(b[:, None], chol)[:, 0]
        out = torch.cat([kern(E[i:i + 8192]).double() @ alpha for i in range(0, len(E), 8192)])
        preds[lam] = out.reshape(d['muev'].shape).float().cpu().numpy()
        if d['mev'].any():
            info['valid_rmsle'][str(lam)] = round(rmse(z_of(d, preds[lam])[d['mev']], d['yev'][d['mev']]), 5)
    return info, preds


# ------------------------------------------------------------------------------------------- driver
def get_panel(args):
    from src.v6.schedule import Schedule
    rain = None
    if args.rain:
        rain = pd.read_csv(args.rain, parse_dates=['date']).set_index('date')['wet']
    return Panel(args.root, Schedule(), load_academic(Path(args.root) / 'configs/ensemble_v7.json'), rain)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def run_cv(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    panel = get_panel(args)
    for fold in args.folds.split(','):
        origin, horizon = FOLDS[fold]
        t0 = time.time(); d = make_data(panel, origin, horizon, args.seed)
        print(f'[{fold}] data {time.time() - t0:.1f}s train rows {int(d["mtr"].sum())}', flush=True)
        for name in args.models.split(','):
            kind, _, variant = name.partition('_')
            names, cols = columns(variant)
            tag = f'{fold}__{name}{args.tag}'
            if (out / f'{tag}.json').exists() and not args.force:
                continue
            t0 = time.time(); arrays = {}
            if kind in ('lgb', 'xgb', 'cb'):
                model, info, predict, save = fit_tree(kind, d, cols, args.seed)
                save(out / tag); arrays['best'] = z_of(d, predict(info['best']))
            elif kind == 'rkhs':
                info, preds = fit_rkhs(d, names, args.seed)
                arrays = {f'lam_{lam}': z_of(d, p) for lam, p in preds.items()}
            else:
                info, best_pred, swa_pred = fit_nn(kind, d, names, args.seed)
                arrays = dict(best=z_of(d, best_pred), lofo=z_of(d, swa_pred))
            info.update(model=name, fold=fold, seed=args.seed, seconds=round(time.time() - t0, 1), kind=kind,
                        variant=variant, features=len(names))
            np.savez_compressed(out / f'{tag}.npz', **arrays); dump(out / f'{tag}.json', info)
            show = {k: info[k] for k in ('best', 'trained', 'valid_rmse_best', 'best_epoch', 'valid_rmsle_best',
                                         'valid_rmsle_swa', 'valid_rmsle', 'alarm_early_stop') if k in info}
            if 'valid_rmsle' in show and isinstance(show['valid_rmsle'], list):
                show['valid_rmsle'] = show['valid_rmsle'][-1]
            print(f'[{fold}] {name}{args.tag} {info["seconds"]}s {show}', flush=True)


def finalize(args):
    """Strict predictions: tree rounds / ridge penalty for a fold come from the other main folds."""
    out = Path(args.out); panel = get_panel(args); F = len(FEATURES)
    infos = {p.stem: json.loads(p.read_text()) for p in out.glob('*__*.json') if not p.stem.startswith('final')}
    models = sorted({k.split('__')[1] for k in infos})
    summary = {}
    for name in models:
        kind = name.split('_')[0]
        folds = [f for f in FOLDS if f'{f}__{name}' in infos]
        variant = infos[f'{folds[0]}__{name}']['variant']
        _, cols = columns(variant)
        entry = {}
        for fold in folds:
            info = infos[f'{fold}__{name}']
            others = [f for f in MAIN if f != fold and f'{f}__{name}' in infos]
            data = dict(np.load(out / f'{fold}__{name}.npz'))
            if kind in ('lgb', 'xgb', 'cb') and others:
                n = int(round(np.mean([infos[f'{f}__{name}']['best'] for f in others])))
                origin, horizon = FOLDS[fold]
                X, mu, y, _ = panel.eval_set(origin, horizon)
                r_hat = load_tree(kind, out / f'{fold}__{name}')(X.reshape(-1, F)[:, cols], min(n, info['trained']))
                data['lofo'] = np.maximum(mu + r_hat.reshape(mu.shape), 0).astype(np.float32)
                entry[fold] = dict(best=info['best'], strict_rounds=min(n, info['trained']))
            elif kind == 'rkhs' and others:
                mean = {lam: np.mean([infos[f'{f}__{name}']['valid_rmsle'][str(lam)] for f in others]) for lam in LAMBDAS}
                lam = min(mean, key=mean.get)
                data['lofo'] = data[f'lam_{lam}']; data['best'] = data[f'lam_{min(LAMBDAS, key=lambda v: info["valid_rmsle"][str(v)])}']
                entry[fold] = dict(strict_lambda=lam)
            elif kind not in ('lgb', 'xgb', 'cb', 'rkhs'):
                entry[fold] = dict(best_epoch=info['best_epoch'])
            np.savez_compressed(out / f'{fold}__{name}.npz', **data)
        main = [f for f in MAIN if f'{f}__{name}' in infos]
        if kind in ('lgb', 'xgb', 'cb'):
            late = [infos[f'{f}__{name}']['best'] for f in ('f3', 'f4') if f'{f}__{name}' in infos]
            entry['final_rounds'] = int(round(1.1 * np.mean(late))) if late else None
        if kind == 'rkhs' and main:
            mean = {lam: np.mean([infos[f'{f}__{name}']['valid_rmsle'][str(lam)] for f in main]) for lam in LAMBDAS}
            entry['final_lambda'] = min(mean, key=mean.get)
        summary[name] = entry
    dump(out / 'cv_summary.json', summary)
    print(json.dumps(summary, indent=1))


def run_final(args):
    out = Path(args.out); panel = get_panel(args)
    summary = json.loads((out / 'cv_summary.json').read_text())
    for seed in (2026, 2027):
        d = make_data(panel, FINAL_ORIGIN, HORIZON, seed)
        for name in args.models.split(','):
            kind, _, variant = name.partition('_')
            names, cols = columns(variant)
            tag = f'final__{name}__{seed}'
            if (out / f'{tag}.npz').exists() and not args.force:
                continue
            t0 = time.time()
            if kind in ('lgb', 'xgb', 'cb'):
                rounds = 30 if SMOKE else summary[name]['final_rounds']
                model, info, predict, save = fit_tree(kind, d, cols, seed, rounds=rounds)
                pred = predict(None); info['rounds'] = rounds; save(out / tag)
            elif kind == 'rkhs':
                lam = summary[name]['final_lambda']
                info, preds = fit_rkhs(d, names, seed, lambdas=[lam]); pred = preds[lam]; info['lambda'] = lam
            else:
                info, _, pred = fit_nn(kind, d, names, seed, cv=False)
            info.update(model=name, seed=seed, seconds=round(time.time() - t0, 1))
            np.savez_compressed(out / f'{tag}.npz', pred=z_of(d, pred)); dump(out / f'{tag}.json', info)
            print(f'[final {seed}] {name} {info["seconds"]}s mean z {float(z_of(d, pred).mean()):.4f}', flush=True)


def main():
    global SMOKE
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['cv', 'finalize', 'final'])
    p.add_argument('--models', default='lgb,xgb,cb,tabm,mdn,stg,rkhs')
    p.add_argument('--folds', default=','.join(FOLDS))
    p.add_argument('--out', default='outputs/ensemble_v7/eq')
    p.add_argument('--root', default='.')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--tag', default='')
    p.add_argument('--rain', default='')
    p.add_argument('--force', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    SMOKE = args.smoke
    {'cv': run_cv, 'finalize': finalize, 'final': run_final}[args.stage](args)


if __name__ == '__main__':
    main()
