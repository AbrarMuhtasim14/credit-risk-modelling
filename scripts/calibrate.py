"""Calibration: time one CV fold end-to-end to estimate total runtime."""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from run_experiment import load_and_split
from src.config import RANDOM_SEED, SMOTE_K_NEIGHBORS
from src.features.preprocessor import fit_transform, transform_only
from src.models.factory import make_model
from src.models.metrics import compute_metrics

df_train, df_test = load_and_split()

# Subsample 40k rows for calibration (keeps class balance via stratify)
from sklearn.model_selection import train_test_split
_, df_cal = train_test_split(df_train, test_size=40000, stratify=df_train["TARGET"], random_state=RANDOM_SEED)
df_cal = df_cal.reset_index(drop=True)

tr_pos = np.arange(32000)
va_pos = np.arange(32000, 40000)
df_tr, df_va = df_cal.iloc[tr_pos], df_cal.iloc[va_pos]
y_tr = df_tr["TARGET"].to_numpy(dtype=np.int8)
y_va = df_va["TARGET"].to_numpy(dtype=np.int8)

t0 = time.time()
X_tr, feat_names, pre, num_cols, cat_cols = fit_transform(df_tr)
X_va = transform_only(df_va, pre, num_cols, cat_cols)
logger.info("Preprocess fit+transform (32k rows): %.1fs -> %d features", time.time() - t0, len(feat_names))

spw = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)

t0 = time.time()
smote = SMOTE(sampling_strategy="auto", k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
logger.info("SMOTE on 32k rows: %.1fs (%d -> %d)", time.time() - t0, len(y_tr), len(y_tr_sm))

results = []
for model_name in ["logistic_regression", "random_forest", "xgboost"]:
    for strategy in ["baseline", "cost_sensitive", "smote"]:
        model = make_model(model_name, strategy, spw, RANDOM_SEED)
        t0 = time.time()
        if strategy == "smote":
            model.fit(X_tr_sm, y_tr_sm)
        else:
            model.fit(X_tr, y_tr)
        fit_t = time.time() - t0
        proba = model.predict_proba(X_va)[:, 1]
        m = compute_metrics(y_va, proba)
        results.append((model_name, strategy, fit_t, m["roc_auc"], m["pr_auc"]))
        logger.info("%-20s %-15s fit=%.1fs ROC-AUC=%.4f PR-AUC=%.4f", model_name, strategy, fit_t, m["roc_auc"], m["pr_auc"])
        del model

total_per_fold_40k = sum(r[2] for r in results) + 0
logger.info("Total fit time for 9 cells on 32k train rows: %.1fs", total_per_fold_40k)

# Scale to full 164k train rows (5.12x more rows): fits scale ~linearly for
# LR/RF/XGB-hist; SMOTE scales ~linearly too.
scale = 164000 / 32000
est_fold_full = total_per_fold_40k * scale * 1.15  # 15% overhead margin
est_total_cv = est_fold_full * 15  # 3 repeats x 5 folds
est_oot = total_per_fold_40k * scale * 1.15
logger.info("Estimated per-fold time at full size: %.0fs", est_fold_full)
logger.info("Estimated total CV (15 folds): %.1f min", est_total_cv / 60)
logger.info("Estimated OOT track: %.1f min", est_oot / 60)
logger.info("Estimated grand total (CV + OOT): %.1f min", (est_total_cv + est_oot) / 60)
