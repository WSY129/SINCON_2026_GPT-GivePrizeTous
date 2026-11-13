"""Graph-mixed historical patches with attention and an 18-hour direct head.

A custom graph/temporal Transformer; not an upstream iTransformer reproduction.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from .preprocessing import NNPreprocessorV4
from ..features_v3 import TRANSFERS


def adjacency(meta):
    ids=sorted(meta.station_id.tolist());mapping={s:i for i,s in enumerate(ids)};a=np.eye(len(ids),dtype='float32')
    lines=meta.set_index('station_id').canonical_line
    for s in ids:
        for t in [s-1,s+1,TRANSFERS.get(s)]:
            if t in mapping and (t==TRANSFERS.get(s) or lines[s]==lines[t]):a[mapping[s],mapping[t]]=1
    return a/a.sum(1,keepdims=True)


def contexts(train,meta,os):
    ids=sorted(meta.station_id.tolist());a=adjacency(meta);out=[]
    cells=pd.MultiIndex.from_product([ids,range(6,24)],names=['station_id','hour'])
    for o in os:
        dates=pd.date_range(pd.Timestamp(o)-pd.Timedelta(days=55),pd.Timestamp(o))
        history=train.loc[train.date<=pd.Timestamp(o)]
        channels=[]
        for col in ['boardings','alightings']:
            v=history.pivot(index='date',columns=['station_id','hour'],values=col).reindex(index=dates,columns=cells).to_numpy().reshape(56,len(ids),18)
            channels.extend([np.nan_to_num(np.log1p(v))/8,np.isfinite(v).astype(float)])
        z=np.stack(channels,axis=-1).astype('float32') # 56,S,18,4
        mixed=np.einsum('ij,tjhc->tihc',a,z)
        # 8 nonoverlapping weekly patches; include own and graph-neighbor profiles.
        patch=np.concatenate([z,mixed],axis=-1).transpose(1,0,2,3).reshape(len(ids),8,7*18*8)
        out.append(patch)
    return np.stack(out).astype('float32'),ids

class SpatialNet(nn.Module):
    def __init__(self,nfeatures,hidden=96):
        super().__init__();self.patch=nn.Linear(7*18*8,hidden);self.pos=nn.Parameter(torch.randn(8,hidden)*.01)
        layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=.15,activation='gelu',batch_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.future=nn.Sequential(nn.Linear(nfeatures,hidden),nn.LayerNorm(hidden),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.SiLU(),nn.Dropout(.15),nn.Linear(hidden,1))
    def forward(self,past,future):
        history=self.temporal(self.patch(past)+self.pos).mean(1)
        encoded=self.future(future)
        return self.head(torch.cat([encoded,history[:,None,:].expand(-1,18,-1)],-1)).squeeze(-1)

class SpatialRegressor:
    def __init__(self,cfg,seed=2026,iterations=None,hardware=None):self.cfg=cfg;self.seed=seed;self.iterations=iterations
    def device_name(self):
        if self.cfg['device']=='cpu':return 'cpu'
        return 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    def prepare(self,x,keys,fit=False):
        if fit:self.pre=NNPreprocessorV4().fit(x)
        num,cat=self.pre.transform(x)
        # Encoded categories scaled to bounded IDs; future numeric embeddings remain causal.
        z=np.concatenate([num,cat/np.maximum(1,np.array(self.pre.cardinalities))],axis=1).astype('float32').reshape(-1,18,num.shape[1]+cat.shape[1])
        days=keys.iloc[::18];origin=days.origin.map(self.origin_map).to_numpy();sid=days.station_id.map(self.station_map).to_numpy()
        return z,origin,sid
    def fit(self,x,y,xvalid,yvalid,keys,validkeys,train,meta):
        torch.manual_seed(self.seed);np.random.seed(self.seed);torch.set_num_threads(8)
        self.device=self.device_name();self.origins=sorted(pd.to_datetime(pd.concat([keys.origin,validkeys.origin])).unique())
        ctx,ids=contexts(train,meta,self.origins);self.context=ctx;self.origin_map={pd.Timestamp(v):i for i,v in enumerate(self.origins)};self.station_map={s:i for i,s in enumerate(ids)}
        prepared=self.prepare(x,keys,True);valid=self.prepare(xvalid,validkeys)
        self.model=SpatialNet(prepared[0].shape[-1],self.cfg['st_itransformer']['hidden']).to(self.device)
        p=self.cfg['st_itransformer'];opt=torch.optim.AdamW(self.model.parameters(),lr=p['learning_rate'],weight_decay=.0003)
        epochs=self.iterations or p['epochs'];scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=1e-5)
        amp=self.device.startswith('cuda');scaler=torch.amp.GradScaler('cuda',enabled=amp)
        targets=y[:,0].reshape(-1,18);batch=p['batch_size'];rng=np.random.default_rng(self.seed);best=float('inf');bad=0;state=None;self.history=[]
        for epoch in range(1,epochs+1):
            self.model.train();order=rng.permutation(len(targets));total=0
            for start in range(0,len(order),batch):
                ix=order[start:start+batch];future,oi,si=prepared
                past=torch.from_numpy(ctx[oi[ix],si[ix]]).to(self.device);f=torch.from_numpy(future[ix]).to(self.device);t=torch.from_numpy(targets[ix]).to(self.device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):loss=(self.model(past,f)-t).square().mean()
                scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(self.model.parameters(),5);scaler.step(opt);scaler.update();total+=loss.item()*len(ix)
            scheduler.step();val=None
            if yvalid is not None:
                pred=self._predict(valid,xvalid.anchor_board_log.to_numpy());truth=yvalid[:,0]+xvalid.anchor_board_log.to_numpy();val=float(np.sqrt(np.mean((pred-truth)**2)))
                if val<best-1e-7:best=val;bad=0;self.best_iteration=epoch;state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                else:bad+=1
            else:self.best_iteration=epoch
            row={'epoch':epoch,'train_mse':total/len(targets),'valid_rmsle':val};self.history.append(row);print(row,flush=True)
            if yvalid is not None and bad>=p['patience']:break
        if state is not None:self.model.load_state_dict(state)
        return self
    def _predict(self,prepared,anchor):
        future,oi,si=prepared;out=[];self.model.eval()
        with torch.no_grad():
            for start in range(0,len(future),512):
                ix=slice(start,start+512);p=torch.from_numpy(self.context[oi[ix],si[ix]]).to(self.device);f=torch.from_numpy(future[ix]).to(self.device)
                out.append(self.model(p,f).cpu().numpy())
        return np.maximum(0,anchor+np.concatenate(out).ravel())
    def predict_log(self,x,keys):return self._predict(self.prepare(x,keys),x.anchor_board_log.to_numpy())
    def to_cpu(self):self.device='cpu';self.model.cpu();return self
