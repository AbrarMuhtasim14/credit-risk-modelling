"""
Main experiment runner: leakage-free 9-cell matrix.

Two evaluation tracks, both strictly leakage-free:

1. IN-TIME track: 3x5 repeated stratified K-fold on the 2018-2019 window ONLY.
   - Preprocessor is re-fit INSIDE every fold (train fold only).
   - SMOTE is applied to the training fold only, computed once per fold and
     shared by the three SMOTE cells.
   - The 2020 data is never touched here.

2. OUT-OF-TIME (OOT) track: fit everything on 2018-2019, evaluate once on 2020.
   - Preprocessor fit once on the full training window.
   - Income winsorization cap computed on training years only.

Champion selection: highest mean CV PR-AUC (primary metric for imbalanced data).
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

from src.config import (
    DATA_DIR,
    DEFAULT_COST_RATIO,
    FIXED_THRESHOLD,
    ID_COL,
    INCOME_WINSOR_QUANTILE,
    LOGS_DIR,
    MODELS_DIR,
    N_FOLDS,
    N_REPEATS,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_APPLICATION_TRAIN,
    SMOTE_K_NEIGHBORS,
    TABLES_DIR,
    TARGET_COL,
    TEST_YEARS,
    TIME_COL,
    TRAIN_YEARS,
)
from src.features.engineering import clean_and_engineer
from src.features.preprocessor import fit_transform, transform_only
from src.models.factory import make_model
from src.models.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "experiment.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

MODELS = ["logistic_regression", "random_forest", "xgboost"]
STRATEGIES = ["baseline", "cost_sensitive", "smote"]
DISPLAY = {
    ("logistic_regression", "baseline"): "Logistic Regression (Baseline)",
    ("logistic_regression", "cost_sensitive"): "Logistic Regression (Cost-Sensitive)",
    ("logistic_regression", "smote"): "Logistic Regression (SMOTE)",
    ("random_forest", "baseline"): "Random Forest (Baseline)",
    ("random_forest", "cost_sensitive"): "Random Forest (Cost-Sensitive)",
    ("random_forest", "smote"): "Random Forest (SMOTE)",
    ("xgboost", "baseline"): "XGBoost (Baseline)",
    ("xgboost", "cost_sensitive"): "XGBoost (Cost-Sensitive)",
    ("xgboost", "smote"): "XGBoost (SMOTE)",
}


def load_and_split():
    """Load raw data, engineer features, split temporally. Returns engineered dfs.

    Fast path: if the engineered parquet checkpoints from a previous run exist in
    data/processed/, they are loaded directly (bit-identical data, no raw CSV
    re-processing). Delete the parquets to force re-engineering from the raw CSV.
    """
    train_path = PROCESSED_DIR / "engineered_train_2018_2019.parquet"
    test_path = PROCESSED_DIR / "engineered_test_2020.parquet"
    if train_path.exists() and test_path.exists():
        logger.info("Engineered parquet checkpoints found - loading them directly.")
        df_train = pd.read_parquet(train_path)
        df_test = pd.read_parquet(test_path)
        logger.info(
            "Temporal split: train 2018-2019 = %d rows (%.2f%% default), "
            "test 2020 = %d rows (%.2f%% default)",
            len(df_train), 100 * df_train[TARGET_COL].mean(),
            len(df_test), 100 * df_test[TARGET_COL].mean(),
        )
        return df_train, df_test

    if not RAW_APPLICATION_TRAIN.exists():
        sample_path = DATA_DIR / "sample" / "sample_application_train.csv"
        if sample_path.exists():
            logger.info("Raw data not found at %s. Falling back to sample dataset: %s", RAW_APPLICATION_TRAIN, sample_path)
            raw_source = sample_path
        else:
            raise FileNotFoundError(f"Neither {RAW_APPLICATION_TRAIN} nor {sample_path} found. Run scripts/download_data.py")
    else:
        raw_source = RAW_APPLICATION_TRAIN

    logger.info("Loading data from %s ...", raw_source)
    df = pd.read_csv(raw_source)
    logger.info("Raw shape: %s", df.shape)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df["app_year"] = df[TIME_COL].dt.year

    # Leakage control: income cap fitted on TRAINING YEARS ONLY
    train_mask = df["app_year"].isin(TRAIN_YEARS)
    if train_mask.sum() == 0:
        income_cap = float(df["AMT_INCOME_TOTAL"].quantile(INCOME_WINSOR_QUANTILE))
    else:
        income_cap = float(df.loc[train_mask, "AMT_INCOME_TOTAL"].quantile(INCOME_WINSOR_QUANTILE))
    logger.info("Income cap (99.9th pct, train years only): $%s", f"{income_cap:,.0f}")

    df_eng = clean_and_engineer(df, income_cap=income_cap)

    df_train = df_eng[df_eng["app_year"].isin(TRAIN_YEARS)].copy()
    df_test = df_eng[df_eng["app_year"].isin(TEST_YEARS)].copy()
    logger.info(
        "Temporal split: train 2018-2019 = %d rows (%.2f%% default), "
        "test 2020 = %d rows (%.2f%% default)",
        len(df_train), 100 * df_train[TARGET_COL].mean(),
        len(df_test), 100 * df_test[TARGET_COL].mean(),
    )
    return df_train, df_test


def run_cv_track(df_train: pd.DataFrame, n_repeats: int = N_REPEATS, n_folds: int = N_FOLDS,
                 repeat_offset: int = 0, checkpoint_path: Path = None) -> pd.DataFrame:
    """
    Repeated stratified CV on the 2018-2019 window.
    Preprocessor re-fit per fold; SMOTE inside fold only.

    Resume support (for Colab handover):
    - repeat_offset: repeats 1..repeat_offset are skipped (already checkpointed).
      Seeds remain tied to the GLOBAL repeat number, so resumed runs are
      bit-identical to uninterrupted runs.
    - checkpoint_path: if given, the cumulative fold-results CSV is rewritten
      after every repeat so a disconnected runtime loses at most one repeat.
    """
    y_all = df_train[TARGET_COL].to_numpy(dtype=np.int8)
    n = len(df_train)

    fold_records = []

    for repeat in range(1, n_repeats + 1):
        if repeat <= repeat_offset:
            logger.info("Repeat %d skipped (already checkpointed)", repeat)
            continue
        seed = RANDOM_SEED + (repeat - 1) * 1000
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (tr_pos, va_pos) in enumerate(skf.split(np.zeros(n), y_all), start=1):
            t_fold = time.time()
            df_tr = df_train.iloc[tr_pos]
            df_va = df_train.iloc[va_pos]
            y_tr = y_all[tr_pos]
            y_va = y_all[va_pos]

            # --- Nested preprocessing: fit on train fold only ---------------
            X_tr, feat_names, pre, num_cols, cat_cols = fit_transform(df_tr)
            X_va = transform_only(df_va, pre, num_cols, cat_cols)

            # scale_pos_weight from this fold's training portion only
            spw = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)

            # --- SMOTE once per fold (shared by the 3 smote cells) ----------
            smote = SMOTE(sampling_strategy="auto", k_neighbors=SMOTE_K_NEIGHBORS, random_state=seed)
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
            logger.info(
                "Repeat %d Fold %d: train=%d (SMOTE->%d), val=%d, %d features",
                repeat, fold_idx, len(y_tr), len(y_tr_sm), len(y_va), len(feat_names),
            )

            for model_name in MODELS:
                for strategy in STRATEGIES:
                    cell_seed = seed + fold_idx
                    model = make_model(model_name, strategy, spw, cell_seed)

                    if strategy == "smote":
                        model.fit(X_tr_sm, y_tr_sm)
                    else:
                        model.fit(X_tr, y_tr)

                    proba = model.predict_proba(X_va)[:, 1]
                    m = compute_metrics(y_va, proba, FIXED_THRESHOLD, DEFAULT_COST_RATIO)

                    fold_records.append({
                        "repeat": repeat,
                        "fold": fold_idx,
                        "model_name": model_name,
                        "strategy": strategy,
                        "display_name": DISPLAY[(model_name, strategy)],
                        **m,
                    })
                    del model
                gc.collect()

            del X_tr, X_va, X_tr_sm, y_tr_sm, pre
            gc.collect()
            _memory_guard()
            logger.info(
                "  Repeat %d Fold %d done in %.1fs", repeat, fold_idx, time.time() - t_fold
            )

        if checkpoint_path is not None:
            pd.DataFrame(fold_records).to_csv(checkpoint_path, index=False)
            logger.info("Checkpoint after repeat %d: %d fold records on disk",
                        repeat, len(fold_records))

    return pd.DataFrame(fold_records)


def run_oot_track(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """Fit on full 2018-2019 window, evaluate once on 2020."""
    y_train = df_train[TARGET_COL].to_numpy(dtype=np.int8)
    y_test = df_test[TARGET_COL].to_numpy(dtype=np.int8)

    X_train, feat_names, pre, num_cols, cat_cols = fit_transform(df_train)
    X_test = transform_only(df_test, pre, num_cols, cat_cols)
    spw = float(np.sum(y_train == 0)) / max(float(np.sum(y_train == 1)), 1.0)

    smote = SMOTE(sampling_strategy="auto", k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    logger.info("OOT: train=%d (SMOTE->%d), test=%d", len(y_train), len(y_train_sm), len(y_test))

    records = []
    oot_probas = {}

    for model_name in MODELS:
        for strategy in STRATEGIES:
            label = DISPLAY[(model_name, strategy)]
            t0 = time.time()
            model = make_model(model_name, strategy, spw, RANDOM_SEED)
            if strategy == "smote":
                model.fit(X_train_sm, y_train_sm)
            else:
                model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]
            m = compute_metrics(y_test, proba, FIXED_THRESHOLD, DEFAULT_COST_RATIO)
            records.append({
                "display_name": label,
                "model_name": model_name,
                "strategy": strategy,
                "fit_seconds": round(time.time() - t0, 2),
                **m,
            })
            oot_probas[label] = proba
            logger.info("OOT %-45s ROC-AUC=%.4f PR-AUC=%.4f (%.1fs)", label, m["roc_auc"], m["pr_auc"], time.time() - t0)

    # Persist OOT probabilities + feature names for SHAP/plots
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    proba_df = pd.DataFrame(oot_probas)
    proba_df[ID_COL] = df_test[ID_COL].to_numpy()
    proba_df[TARGET_COL] = y_test
    proba_df.to_parquet(MODELS_DIR / "oot_probas.parquet", index=False)

    with open(MODELS_DIR / "oot_feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feat_names, f)

    joblib.dump({"preprocessor": pre, "numeric_cols": num_cols, "categorical_cols": cat_cols},
                MODELS_DIR / "oot_preprocessor.joblib")

    return pd.DataFrame(records)


def _memory_guard():
    """Halt if RAM pressure gets dangerous on this 8GB box."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        load = int(stat.dwMemoryLoad)
        avail_mb = stat.ullAvailPhys // (1024 * 1024)
        if load > 90:
            logger.warning("MEMORY PRESSURE: %d%% used, %d MB available - forcing GC", load, avail_mb)
            gc.collect()
        if load > 96:
            raise MemoryError(f"System RAM at {load}% ({avail_mb} MB free) - aborting to protect the machine")
    except MemoryError:
        raise
    except Exception:
        pass  # guard is best-effort


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 1 repeat x 2 folds on a 60k subsample of the train window")
    args = parser.parse_args()

    t_start = time.time()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_train, df_test = load_and_split()

    if args.quick:
        from sklearn.model_selection import train_test_split

        if len(df_train) > 60000:
            _, df_train = train_test_split(
                df_train, test_size=60000, stratify=df_train[TARGET_COL], random_state=RANDOM_SEED
            )
            df_train = df_train.reset_index(drop=True)
        logger.info("QUICK MODE: CV on %d-row dataset, 1 repeat x 2 folds", len(df_train))

    # Save engineered splits for downstream SHAP / reproducibility.
    # Never in quick mode: the subsampled df_train would corrupt the full checkpoint.
    if not args.quick:
        df_train.to_parquet(PROCESSED_DIR / "engineered_train_2018_2019.parquet", index=False)
        df_test.to_parquet(PROCESSED_DIR / "engineered_test_2020.parquet", index=False)

    logger.info("=" * 80)
    logger.info("TRACK 1: %dx%d repeated CV on 2018-2019 (in-time)", N_REPEATS, N_FOLDS)
    logger.info("=" * 80)
    if args.quick:
        # Quick mode uses its own output file so it never pollutes the full-run checkpoint
        df_cv = run_cv_track(df_train, n_repeats=1, n_folds=2)
        df_cv.to_csv(TABLES_DIR / "cv_fold_results_quick.csv", index=False)
    else:
        cv_path = TABLES_DIR / "cv_fold_results.csv"
        if cv_path.exists():
            prior = pd.read_csv(cv_path)
            repeat_offset = int(prior["repeat"].max())
            if repeat_offset >= N_REPEATS:
                logger.info("Full CV checkpoint found (%d repeats) - reusing it", repeat_offset)
                df_cv = prior
            else:
                logger.info("Partial CV checkpoint (%d/%d repeats) - resuming",
                            repeat_offset, N_REPEATS)
                df_new = run_cv_track(df_train, repeat_offset=repeat_offset,
                                      checkpoint_path=cv_path)
                df_cv = pd.concat([prior, df_new], ignore_index=True)
        else:
            df_cv = run_cv_track(df_train, checkpoint_path=cv_path)
        df_cv.to_csv(cv_path, index=False)

    if args.quick:
        logger.info("QUICK MODE: skipping OOT track and aggregation.")
        logger.info("Smoke test finished in %.1f minutes.", (time.time() - t_start) / 60)
        return

    logger.info("=" * 80)
    logger.info("TRACK 2: OOT temporal benchmark (train 2018-19 -> test 2020)")
    logger.info("=" * 80)
    df_oot = run_oot_track(df_train, df_test)
    df_oot.to_csv(TABLES_DIR / "oot_results.csv", index=False)

    # Aggregate CV table (mean/std across the 15 observations per config)
    agg = (
        df_cv.groupby("display_name")
        .agg(
            roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
            recall_mean=("recall", "mean"), precision_mean=("precision", "mean"),
            f1_mean=("f1", "mean"), f2_mean=("f2", "mean"),
            cost_mean=("cost_per_applicant", "mean"),
        )
        .reset_index()
        .sort_values("pr_auc_mean", ascending=False)
    )
    agg.to_csv(TABLES_DIR / "cv_summary.csv", index=False)

    elapsed = time.time() - t_start
    logger.info("Experiment finished in %.1f minutes.", elapsed / 60)
    logger.info("Champion by mean CV PR-AUC: %s", agg.iloc[0]["display_name"])


if __name__ == "__main__":
    main()
