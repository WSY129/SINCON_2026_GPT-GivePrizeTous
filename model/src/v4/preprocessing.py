"""Finite-only scaler statistics, then row-anchor imputation and missing masks."""
import warnings
import numpy as np
from ..nn_preprocessor_v3 import NNPreprocessorV3
from ..features_v3 import CATS

class NNPreprocessorV4(NNPreprocessorV3):
    def fit(self,x):
        self.cats=[c for c in CATS if c in x];self.cont=[c for c in x if c not in self.cats]
        a=x[self.cont].to_numpy(dtype='float32',copy=True);a[~np.isfinite(a)]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            self.median=np.nanmedian(a,axis=0)
            self.iqr=np.nanpercentile(a,75,axis=0)-np.nanpercentile(a,25,axis=0)
            self.constant=np.nanmax(a,axis=0)==np.nanmin(a,axis=0)
        rare=np.isfinite(a).sum(0)<10
        self.median=np.where(rare,0,np.nan_to_num(self.median))
        self.iqr=np.where(rare,1,np.nan_to_num(self.iqr))
        self.constant|=rare
        self.maps={c:{int(v):i+1 for i,v in enumerate(sorted(x[c].dropna().unique()))} for c in self.cats}
        self.cardinalities=[len(self.maps[c])+1 for c in self.cats]
        return self
    def values(self,x):
        a,mask=super().values(x)
        for j,name in enumerate(self.cont):
            if name in ('z_yoy_raw_mean','z_yoy_clean_med','z_yoy_trimmed','z_yoy_q85'):
                a[mask[:,j],j]=x.anchor_board_log.to_numpy()[mask[:,j]]
        return a,mask
    def transform(self,x):
        a,mask=self.values(x)
        a=np.clip((np.where(np.isfinite(a),a,self.median)-self.median)/(self.iqr+1e-5),-10,10)
        a[:,self.constant]=0
        num=np.concatenate([a,mask.astype('float32')],axis=1).astype('float32')
        cat=np.column_stack([x[c].map(self.maps[c]).fillna(0).to_numpy(dtype='int64') for c in self.cats]) if self.cats else np.empty((len(x),0),dtype='int64')
        return num,cat
