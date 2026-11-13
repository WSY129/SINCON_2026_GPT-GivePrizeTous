"""Pure PyTorch multitask BatchEnsemble with periodic numerical embeddings.

This is the requested TabM-style architecture, not the upstream TabM package.
"""
import copy
import time
import numpy as np
import torch
from torch import nn
from .preprocessing import NNPreprocessorV5 as NNPreprocessor
from ..models_v2 import seed_everything
from .checkpoints import TopCheckpoints

class PeriodicEmbedding(nn.Module):
    def __init__(self,n,frequencies):
        super().__init__();self.weight=nn.Parameter(torch.randn(n,frequencies)*.1)
    def forward(self,x):
        phase=2*torch.pi*x[:,:,None]*self.weight[None,:,:]
        return torch.cat([phase.sin(),phase.cos()],dim=-1).flatten(1)

class BatchEnsembleBlock(nn.Module):
    def __init__(self,inputs,hidden,k,dropout):
        super().__init__();self.linear=nn.Linear(inputs,hidden,bias=False)
        self.r=nn.Parameter(torch.empty(k,inputs).bernoulli_(.5).mul_(2).sub_(1))
        self.s=nn.Parameter(torch.empty(k,hidden).bernoulli_(.5).mul_(2).sub_(1))
        self.bias=nn.Parameter(torch.zeros(k,hidden));self.norm=nn.LayerNorm(hidden)
        self.activation=nn.SiLU();self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.dropout(self.activation(self.norm(self.linear(x*self.r)*self.s+self.bias)))

class MultiTaskTabM(nn.Module):
    def __init__(self,n_cont,cardinalities,k=8,hidden=256,blocks=3,frequencies=4,dropout=.15):
        super().__init__();self.k=k
        self.raw_cont=n_cont//2
        self.numeric=PeriodicEmbedding(self.raw_cont,frequencies)
        self.embeddings=nn.ModuleList([nn.Embedding(c,min(16,max(2,int(np.ceil(np.sqrt(c))))),padding_idx=0) for c in cardinalities])
        width=self.raw_cont*frequencies*2+self.raw_cont+sum(e.embedding_dim for e in self.embeddings)
        self.blocks=nn.ModuleList([BatchEnsembleBlock(width if i==0 else hidden,hidden,k,dropout) for i in range(blocks)])
        self.head=nn.Linear(hidden,2)
    def forward(self,num,cat):
        x=torch.cat([self.numeric(num[:,:self.raw_cont]),num[:,self.raw_cont:]]+[e(cat[:,i]) for i,e in enumerate(self.embeddings)],dim=1)
        x=x[:,None,:].expand(-1,self.k,-1)
        for block in self.blocks:x=block(x)
        return self.head(x)

def multitask_loss(pred,target,auxiliary_weight=.2,weights=None):
    error=(pred-target[:,None,:]).square()
    # Compute each head's loss before averaging. Opposing heads cannot cancel.
    weights=torch.ones(len(target),device=pred.device) if weights is None else weights
    weighted=(error[:,:,0]+auxiliary_weight*error[:,:,1])*weights[:,None]
    per_head=weighted.sum(dim=0)/weights.sum()
    return per_head.mean()

class TabMRegressor:
    def __init__(self,cfg,seed=2026,iterations=None,hardware=None):
        self.cfg=cfg;self.seed=seed;self.iterations=iterations;self.hardware=hardware or {}
    def device_name(self):
        if self.cfg['device']=='cpu':return 'cpu'
        if torch.cuda.is_available():return 'cuda:0'
        if torch.backends.mps.is_available():return 'mps'
        return 'cpu'
    def fit(self,x,y,xvalid=None,yvalid=None):
        seed_everything(self.seed);torch.manual_seed(self.seed);torch.set_num_threads(self.cfg['threads'])
        if torch.cuda.is_available():torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        self.device=self.device_name();p=self.cfg['tabm'];amp=self.device.startswith('cuda')
        self.preprocessor=NNPreprocessor().fit(x)
        num,cat=self.preprocessor.transform(x);target=torch.from_numpy(y)
        self.arch={key:p[key] for key in ['k','hidden','blocks','frequencies','dropout']}
        self.arch.update(n_cont=num.shape[1],cardinalities=self.preprocessor.cardinalities)
        self.model=MultiTaskTabM(**self.arch).to(self.device)
        num=torch.from_numpy(num);cat=torch.from_numpy(cat)
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
        epochs=self.iterations or p['epochs'];scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e+1)/3 if e<3 else 1e-5/p['learning_rate']+(1-1e-5/p['learning_rate'])*.5*(1+np.cos(np.pi*(e-3)/max(1,epochs-3))))
        scaler=torch.amp.GradScaler('cuda',enabled=amp)
        self.history=[];best=float('inf');bad=0;best_state=None
        checkpoints=TopCheckpoints(self.cfg.get('checkpoint_average',3))
        weights=torch.from_numpy(np.where(x.n_yoy>=2,1.25,1.).astype('float32'))
        valid_prepared=None if xvalid is None else self.preprocessor.transform(xvalid)
        valid_true=None if yvalid is None else yvalid[:,0]+xvalid.anchor_board_log.to_numpy()
        batch=p['batch_size'];rng=np.random.default_rng(self.seed)
        for epoch in range(1,epochs+1):
            start=time.perf_counter();self.model.train();total=0;order=rng.permutation(len(x))
            for offset in range(0,len(x),batch):
                ix=torch.from_numpy(order[offset:offset+batch]);a=num[ix].to(self.device);b=cat[ix].to(self.device);t=target[ix].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda' if amp else 'cpu',dtype=torch.float16,enabled=amp):
                    loss=multitask_loss(self.model(a,b),t,p['auxiliary_weight'],weights[ix].to(self.device))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TabM loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),5.0)
                scaler.step(optimizer);scaler.update();total+=loss.item()*len(ix)
            scheduler.step();val=None
            if xvalid is not None:
                pred=self._predict_prepared(valid_prepared,xvalid.anchor_board_log.to_numpy())
                val=float(np.sqrt(np.mean((pred-valid_true)**2)))
                if val<best-1e-7:
                    best=val;bad=0;self.best_iteration=epoch
                    best_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                else:bad+=1
            else:self.best_iteration=epoch
            checkpoints.add(self.model,epoch,val)
            row={'epoch':epoch,'train_loss':total/len(x),'valid_rmsle':val,'seconds':time.perf_counter()-start,'learning_rate':optimizer.param_groups[0]['lr']}
            self.history.append(row);print(row,flush=True)
            if xvalid is not None and bad>=p['patience']:break
        self.averaged_epochs=checkpoints.apply(self.model)
        self.checkpoint_policy='validation_top3' if xvalid is not None else 'last3_planned_epochs'
        return self
    def _predict_prepared(self,prepared,anchor):
        self.model.eval();num,cat=prepared;out=[];batch=self.cfg['tabm']['batch_size']
        with torch.no_grad():
            for start in range(0,len(num),batch):
                a=torch.from_numpy(num[start:start+batch]).to(self.device);b=torch.from_numpy(cat[start:start+batch]).to(self.device)
                residual=self.model(a,b)[:,:,0].float().cpu().numpy()
                # Clip each member in log space BEFORE the K-member average.
                out.append(np.maximum(0,anchor[start:start+batch,None]+residual).mean(axis=1))
        return np.concatenate(out).astype('float64')
    def predict_log(self,x):return self._predict_prepared(self.preprocessor.transform(x),x.anchor_board_log.to_numpy())
    def to_cpu(self):self.model.to('cpu');self.device='cpu';return self
