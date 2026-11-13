"""Use raw validation data so early stopping measures the inference path.

Training keeps QuantileDMatrix/hist. The installed XGBoost 3.4.1 produced
different predictions with reference-category validation metadata versus
raw DMatrix/inplace prediction in this dataset. ResidualTree.frame fixes
identical categorical levels/codes for training, validation and inference,
so validation needs no additional reference-category recoding. This adapter
changes only validation construction, not training or hyperparameters.
"""
from xgboost import XGBRegressor,DMatrix

class InferenceConsistentXGBRegressor(XGBRegressor):
    def _create_dmatrix(self,ref,**kwargs):
        if ref is not None:
            kwargs['feature_types']=None
            return DMatrix(**kwargs,nthread=self.n_jobs)
        return super()._create_dmatrix(ref=ref,**kwargs)
