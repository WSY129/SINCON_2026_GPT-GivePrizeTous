"""v3 regularized trees; v2 checkpoints and feature schemas remain unchanged."""
from .models_v2 import ResidualTree as ResidualTreeV2, devices

class ResidualTree(ResidualTreeV2):
    def frame(self, x, fit=False):
        if self.name in ('lightgbm', 'xgboost'):
            x = x.drop(columns=['station_hour', 'station_eff_dow'], errors='ignore')
        return super().frame(x, fit=fit)
