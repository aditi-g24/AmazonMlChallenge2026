import os
import sys
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import lightgbm as lgb
from features import FEATURE_NAMES

class EntityMatchingModel:
    def __init__(self, threshold=0.55):
        self.model = None
        self.threshold = threshold

    def train(self, X_train, y_train, X_val=None, y_val=None):
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.08,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'n_jobs': -1
        }
        
        valid_sets = [dtrain]
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=dtrain)
            valid_sets.append(dval)
            
        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=150,
            valid_sets=valid_sets
        )

    def predict_proba(self, X):
        if len(X) == 0:
            return np.array([])
        return self.model.predict(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def save(self, path):
        if self.model:
            self.model.save_model(path)

    def load(self, path):
        self.model = lgb.Booster(model_file=path)
