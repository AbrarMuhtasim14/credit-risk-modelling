"""Fit champion XGBoost on the full 2018-2019 training window and serialize to models/champion_xgboost.joblib."""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

from src.models.factory import make_model
from src.config import RANDOM_SEED, TARGET_COL

def export_champion():
    train_path = PROCESSED_DIR / "engineered_train_2018_2019.parquet"
    prep_path = MODELS_DIR / "oot_preprocessor.joblib"
    model_path = MODELS_DIR / "champion_xgboost.joblib"

    if model_path.exists():
        logger.info("Champion model already exists at: %s", model_path)
        return

    logger.info("Loading training parquet: %s", train_path)
    df_train = pd.read_parquet(train_path)

    logger.info("Loading preprocessor bundle: %s", prep_path)
    bundle = joblib.load(prep_path)
    pre = bundle["preprocessor"]
    num_cols = bundle["numeric_cols"]
    cat_cols = bundle["categorical_cols"]

    logger.info("Transforming feature matrix...")
    X = pre.transform(df_train[num_cols + cat_cols]).astype(np.float32)
    y = df_train[TARGET_COL].to_numpy(dtype=np.int8)

    spw = float(np.sum(y == 0)) / max(float(np.sum(y == 1)), 1.0)

    logger.info("Fitting champion XGBoost Baseline (frozen dissertation hyperparams)...")
    xgb_model = make_model("xgboost", "baseline", spw, RANDOM_SEED)
    xgb_model.fit(X, y)

    logger.info("Saving champion model to %s...", model_path)
    joblib.dump(xgb_model, model_path, compress=3)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info("Successfully exported champion model (%.2f MB).", size_mb)

if __name__ == "__main__":
    export_champion()
