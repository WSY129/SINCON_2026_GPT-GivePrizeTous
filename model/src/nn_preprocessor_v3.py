"""Origin-anchor imputation for prior-year log levels, retaining missing masks."""
import numpy as np
from .nn_preprocessor import NNPreprocessor
from .features_v2 import CATS

class NNPreprocessorV3(NNPreprocessor):
    def values(self, x):
        a = x[self.cont].to_numpy(dtype='float32', copy=True)
        mask = ~np.isfinite(a)
        for j, name in enumerate(self.cont):
            # Only level features: delta/missing flags must not receive a log level.
            if name in ('year364_clean_log', 'year364_board_log', 'year364_alight_log'):
                anchor = 'anchor_alight_log' if 'alight' in name else 'anchor_board_log'
                if anchor in x:
                    a[mask[:, j], j] = x[anchor].to_numpy()[mask[:, j]]
        return a, mask

    def fit(self, x):
        self.cats = [c for c in CATS if c in x]
        self.cont = [c for c in x if c not in self.cats]
        a, _ = self.values(x)
        a[~np.isfinite(a)] = np.nan
        self.median = np.nan_to_num(np.nanmedian(a, axis=0), nan=0)
        self.iqr = np.nan_to_num(np.nanpercentile(a, 75, axis=0) - np.nanpercentile(a, 25, axis=0), nan=0)
        # Unseen event flags must not activate arbitrary network weights.
        # Preserve rare but nonconstant columns even when their IQR is zero.
        filled = np.where(np.isfinite(a), a, self.median)
        self.constant = np.ptp(filled, axis=0) == 0
        self.maps = {c: {int(v): i + 1 for i, v in enumerate(sorted(x[c].dropna().unique()))} for c in self.cats}
        self.cardinalities = [len(self.maps[c]) + 1 for c in self.cats]
        return self

    def transform(self, x):
        a, original_mask = self.values(x)
        a = np.clip((np.where(np.isfinite(a), a, self.median) - self.median) / (self.iqr + 1e-5), -10, 10)
        a[:, self.constant] = 0
        num = np.concatenate([a, original_mask.astype('float32')], axis=1).astype('float32')
        cat = np.column_stack([x[c].map(self.maps[c]).fillna(0).to_numpy(dtype='int64') for c in self.cats])
        return num, cat
