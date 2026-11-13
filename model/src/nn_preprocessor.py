"""Fold-fitted robust continuous scaling, missing masks and unknown categories."""
import numpy as np
from .features_v2 import CATS

class NNPreprocessor:
    def fit(self,x):
        self.cats=[c for c in CATS if c in x];self.cont=[c for c in x if c not in self.cats]
        a=x[self.cont].to_numpy(dtype='float32')
        self.median=np.nan_to_num(np.nanmedian(a,axis=0),nan=0)
        self.iqr=np.nan_to_num(np.nanpercentile(a,75,axis=0)-np.nanpercentile(a,25,axis=0),nan=0)
        self.maps={c:{int(v):i+1 for i,v in enumerate(sorted(x[c].dropna().unique()))} for c in self.cats}
        self.cardinalities=[len(self.maps[c])+1 for c in self.cats]
        return self
    def transform(self,x):
        a=x[self.cont].to_numpy(dtype='float32');mask=~np.isfinite(a)
        a=np.clip((np.where(mask,self.median,a)-self.median)/(self.iqr+1e-5),-10,10)
        continuous=np.concatenate([a,mask.astype('float32')],axis=1).astype('float32')
        categorical=np.column_stack([x[c].map(self.maps[c]).fillna(0).to_numpy(dtype='int64') for c in self.cats])
        return continuous,categorical
