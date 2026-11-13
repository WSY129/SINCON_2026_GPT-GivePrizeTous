"""Scratch-trained CPU tree baselines and a small tabular neural baseline."""
import numpy as np
import sys
from .models_v2 import ResidualTree

def lightgbm_parameters(config,large=True):
    """Common parameters, with explicit overrides from a saved tuning plan."""
    params=dict(objective='regression',n_estimators=220 if large else 150,
                learning_rate=.055,num_leaves=63 if large else 31,min_child_samples=80,
                reg_lambda=3,colsample_bytree=.9,subsample=.9,subsample_freq=1,
                random_state=config['seed'],n_jobs=config['threads'],verbosity=-1,linear_tree=False)
    overrides=config.get('lightgbm_params',{})
    allowed={'n_estimators','learning_rate','num_leaves','max_depth','min_child_samples',
             'min_child_weight','reg_lambda','reg_alpha','colsample_bytree','subsample',
             'subsample_freq','min_split_gain'}
    if set(overrides)-allowed:raise ValueError(f'Unsupported LightGBM overrides: {set(overrides)-allowed}')
    params.update(overrides)
    return params

class NeuralRegressor:
    """MLP with training-only scaling and missing masks; not TabM/RevIN."""
    def __init__(self,seed=2026,threads=4,device='cpu',epochs=10):
        self.seed,self.threads,self.device,self.epochs=seed,threads,device,epochs

    def fit(self,x,y):
        import torch
        # Apple OpenMP runtimes used by tree wheels can deadlock with Torch's
        # indexed tensor kernels. One Torch CPU thread avoids this interaction.
        torch.manual_seed(self.seed); torch.set_num_threads(1 if sys.platform=='darwin' else self.threads)
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.columns=[c for c in x.columns if c in ['station_id','hour','dow','canonical_line','horizon','month','date_index','weekend','hour_sin','hour_cos','dow_sin','dow_cos','day_of_year_sin','day_of_year_cos','history_days','cold_start','origin_lag_1','origin_lag_7','origin_lag_28','seasonal7','seasonal7_alight','recent_mean','recent_std','recent_alight','baseline','is_holiday','holiday_type','is_holiday_eve','is_after_holiday']]
        a=x[self.columns].to_numpy(dtype='float32')
        self.fill=np.nanmedian(a,axis=0)
        self.fill=np.nan_to_num(self.fill,nan=0).astype('float32')
        filled=np.where(np.isnan(a),self.fill,a)
        self.mean=filled.mean(axis=0); self.std=filled.std(axis=0).clip(min=1)
        z=self.transform(x)
        # y is log1p(count). Squared error is exactly the square of RMSLE.
        target=y-np.log1p(x.baseline.to_numpy())
        self.net=torch.nn.Sequential(torch.nn.Linear(z.shape[1],96),torch.nn.ReLU(),torch.nn.Linear(96,64),torch.nn.ReLU(),torch.nn.Linear(64,1)).to(self.device)
        optimizer=torch.optim.AdamW(self.net.parameters(),lr=0.001,weight_decay=0.001)
        use_amp=self.device.startswith('cuda')
        scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
        rng=np.random.default_rng(self.seed)
        n=min(len(z),160000)
        idx=rng.choice(len(z),n,replace=False)
        xt=torch.from_numpy(z[idx]); yt=torch.tensor(target[idx],dtype=torch.float32)
        self.losses=[]
        self.net.train()
        for epoch in range(self.epochs):
            perm=torch.randperm(n); losses=[]
            for ids in perm.split(2048):
                xb=xt[ids].to(self.device); yb=yt[ids].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=use_amp):
                    pred=self.net(xb).squeeze(1);loss=torch.nn.functional.mse_loss(pred,yb)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite neural loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.0,error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update()
                losses.append(float(loss.detach().cpu()))
            self.losses.append(float(np.sqrt(np.mean(losses))))
        self.net.to('cpu');self.net.eval()
        return self

    def transform(self,x):
        a=x[self.columns].to_numpy(dtype='float32')
        mask=np.isnan(a)
        z=(np.where(mask,self.fill,a)-self.mean)/self.std
        return np.concatenate([z,mask.astype('float32')],axis=1).astype('float32')

    def predict(self,x):
        import torch
        z=self.transform(x);parts=[]
        self.net.eval()
        with torch.no_grad():
            for start in range(0,len(x),8192):
                parts.append(self.net(torch.from_numpy(z[start:start+8192])).squeeze(1).numpy())
        return np.log1p(x.baseline.to_numpy())+np.concatenate(parts)

class Model:
    def __init__(self,name,config):
        self.name,self.config=name,config

    def _frame(self,x):
        if self.name=='catboost':
            z=x.copy()
            for col in ['station_id','canonical_line','hour','dow']:z[col]=z[col].astype('int32').astype(str)
            return z
        return x

    def fit(self,x,y):
        name=self.name;cfg=self.config
        if name in ['seasonal7','weekday_mean']:return self
        if cfg['training_loss']!='log_mse':
            raise ValueError('SINCON official RMSLE requires this implementation to train log_mse.')
        y=np.log1p(y)
        if name.startswith('lightgbm'):
            from lightgbm import LGBMRegressor
            large=name.endswith('large')
            self.estimator=LGBMRegressor(**lightgbm_parameters(cfg,large))
            self.estimator.fit(x,y,categorical_feature=['station_id','canonical_line','hour','dow'])
        elif name=='catboost':
            from catboost import CatBoostRegressor
            self.estimator=CatBoostRegressor(iterations=220,depth=7,learning_rate=0.07,loss_function='RMSE',l2_leaf_reg=5,random_seed=cfg['seed'],thread_count=cfg['threads'],verbose=False,allow_writing_files=False)
            self.estimator.fit(self._frame(x),y,cat_features=['station_id','canonical_line','hour','dow'])
        elif name=='xgboost':
            from xgboost import XGBRegressor
            self.estimator=XGBRegressor(objective='reg:squarederror',n_estimators=180,max_depth=6,learning_rate=0.06,min_child_weight=30,reg_lambda=5,subsample=0.9,colsample_bytree=0.85,tree_method='hist',n_jobs=cfg['threads'],random_state=cfg['seed'])
            self.estimator.fit(x,y)
        elif name=='mlp':
            self.estimator=NeuralRegressor(cfg['seed'],cfg['threads'],cfg['device']).fit(x,y)
        else:raise ValueError(f'Unknown model {name}')
        return self

    def predict(self,x):
        if self.name in ['seasonal7','weekday_mean']:return x[self.name].to_numpy(dtype='float64')
        pred=np.expm1(np.maximum(0,self.estimator.predict(self._frame(x)))).astype('float64')
        if not np.isfinite(pred).all():raise ValueError(f'{self.name}: nonfinite predictions')
        return pred
