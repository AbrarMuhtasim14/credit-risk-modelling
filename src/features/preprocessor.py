"""
Leakage-free preprocessing via scikit-learn ColumnTransformer.

Design rule: the preprocessor is ALWAYS fitted on training data only and then
applied to held-out data. Two usage modes:
  - OOT track:  fit once on 2018-2019, transform 2020.
  - CV track:   re-fit inside every fold (fit on train fold, transform val fold).

Numerical: median imputation + RobustScaler (robust to residual outliers).
Categorical: constant 'Missing' imputation + OneHotEncoder(handle_unknown='ignore').
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config import ID_COL, TARGET_COL, TIME_COL

logger = logging.getLogger(__name__)

META_COLS = [ID_COL, TARGET_COL, TIME_COL, "app_year"]


def split_column_types(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """Classify feature columns into numeric vs categorical."""
    exclude = exclude or META_COLS
    feature_cols = [c for c in df.columns if c not in exclude]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
        n_jobs=None,
    )


def fit_transform(df_train: pd.DataFrame) -> Tuple[np.ndarray, List[str], ColumnTransformer, List[str], List[str]]:
    """
    Fit a fresh preprocessor on df_train and transform it.

    Returns (X_train_float32, feature_names, fitted_preprocessor, numeric_cols, categorical_cols).
    """
    numeric_cols, categorical_cols = split_column_types(df_train)
    pre = build_preprocessor(numeric_cols, categorical_cols)
    arr = pre.fit_transform(df_train[numeric_cols + categorical_cols]).astype(np.float32)
    feature_names = list(pre.get_feature_names_out())
    logger.info(
        "Preprocessor fitted on %d rows: %d numeric + %d categorical -> %d features",
        len(df_train), len(numeric_cols), len(categorical_cols), len(feature_names),
    )
    return arr, feature_names, pre, numeric_cols, categorical_cols


def transform_only(df: pd.DataFrame, pre: ColumnTransformer, numeric_cols: List[str], categorical_cols: List[str]) -> np.ndarray:
    """Apply an already-fitted preprocessor to new data (no refitting)."""
    return pre.transform(df[numeric_cols + categorical_cols]).astype(np.float32)
