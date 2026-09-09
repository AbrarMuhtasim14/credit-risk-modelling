"""
Model factory: 3 architectures with FROZEN hyperparameters.

Experimental control: every imbalance strategy for a given model family uses
identical architecture hyperparameters. The ONLY thing that changes between
baseline / cost-sensitive / SMOTE is the imbalance intervention itself, so any
performance difference is attributable to the intervention (not tuning).
"""

from typing import Any

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_SEED

# Frozen architecture hyperparameters (chosen conservatively for 4-core / 8GB box).
# sklearn 1.9: LogisticRegression penalty defaults to l2 ('penalty' arg deprecated);
# n_jobs has no effect on LR since 1.8.
LR_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=2000)
RF_PARAMS = dict(n_estimators=200, max_depth=12, min_samples_leaf=20, min_samples_split=40, max_features="sqrt")
XGB_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", eval_metric="logloss",
)


def make_model(model_name: str, strategy: str, scale_pos_weight: float, seed: int) -> Any:
    """
    Build a model for one cell of the 9-cell matrix.

    Parameters
    ----------
    model_name : 'logistic_regression' | 'random_forest' | 'xgboost'
    strategy   : 'baseline' | 'cost_sensitive' | 'smote'
        (for 'smote' the model itself stays at baseline settings; resampling
        happens on the training fold before fit)
    scale_pos_weight : neg/pos ratio computed on the TRAINING window only.
    seed : per-fold seed for reproducibility.
    """
    cost_sensitive = strategy == "cost_sensitive"

    if model_name == "logistic_regression":
        return LogisticRegression(
            **LR_PARAMS,
            class_weight="balanced" if cost_sensitive else None,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            **RF_PARAMS,
            class_weight="balanced" if cost_sensitive else None,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        return xgb.XGBClassifier(
            **XGB_PARAMS,
            scale_pos_weight=scale_pos_weight if cost_sensitive else 1.0,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")
