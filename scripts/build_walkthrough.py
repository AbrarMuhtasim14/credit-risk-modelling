"""Generate WALKTHROUGH.ipynb — a granular, step-by-step, fully visible tour of the
entire experiment, from raw CSV to SHAP. Designed so a teammate can open the notebook
and SEE every step: each transformation prints its effect, each artifact is shown
being created, and every result table/figure is displayed inline.

The notebook is executed once (jupyter nbconvert --execute) so all outputs are
embedded — a reader sees everything without running anything, and can re-run any
cell to reproduce it.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "WALKTHROUGH.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

# ===================================================================== intro
cells.append(md("""# WALKTHROUGH — The Full Experiment, Step by Step

**Loan Default Prediction (MSc Data Science dissertation) — Home Credit `application_train.csv`**

This notebook re-runs the **entire experiment as small, explicit steps** so that every
stage is visible — nothing is hidden inside a single function call:

| Section | What you will see happen |
|---|---|
| 0 | Environment & project layout |
| 1 | Raw data loading (307,511 rows) |
| 2 | Exploratory data analysis highlights |
| 3 | Cleaning step 1 — the `DAYS_EMPLOYED = 365243` sentinel correction |
| 4 | Cleaning step 2 — income winsorisation (cap fitted on training years only) |
| 5 | Feature engineering & the temporal split (2018-19 train / 2020 test) |
| 6 | Inside ONE cross-validation fold (preprocessor fit, SMOTE, model fit, metrics) |
| 7 | A live mini cross-validation run — watch the fold records become a CSV |
| 8 | The full 3x5 repeated CV results (135 fold records) |
| 9 | Out-of-time (2020) evaluation + figures |
| 10 | Statistical significance battery (16 comparisons) |
| 11 | SHAP explainability (importance, beeswarm, case studies) |
| 12 | Complete artifact inventory — every file this experiment creates |

**Two ways to use this notebook**
1. **Just read it** — it was executed end-to-end, so every cell already shows its output.
2. **Re-run it** — Runtime/Kernel > Restart & Run All. Lightweight steps re-run live
   (~10 min); the expensive full 3x5 CV and SHAP steps load their checkpointed results
   from disk (that IS the reproducibility mechanism — same seeds, same data, same numbers).

The heavy production entry points are `run_experiment.py` and `run_analysis.py`;
this notebook opens them up so you can watch each gear turn.
"""))

# ===================================================================== colab setup
cells.append(md("""## First cell — running on Google Colab?

If you opened this notebook from Google Drive in Colab, the cell below mounts your
Drive, **auto-detects** the project folder, installs the dependencies, and moves
into it. On a local machine it does nothing (no-op), so this notebook runs
identically in both places.

**On Colab:** run this cell, click **Allow** on the Drive popup, wait for the
packages to install (~1 min)."""))
cells.append(code("""import os, sys

try:
    from google.colab import drive
    drive.mount('/content/drive')

    # Auto-detect the project root (folder containing src/ and run_experiment.py)
    from pathlib import Path
    PROJECT_ROOT = None
    for candidate in Path('/content/drive/MyDrive').rglob('run_experiment.py'):
        if (candidate.parent / 'src').is_dir():
            PROJECT_ROOT = candidate.parent
            break
    assert PROJECT_ROOT is not None, "Could not find the project folder on Drive"

    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Ensure expected directories exist (log files are opened at import time)
    for _d in ('logs', 'data/raw', 'data/processed', 'models',
               'reports/tables', 'reports/figures', 'outputs/eda'):
        os.makedirs(PROJECT_ROOT / _d, exist_ok=True)

    print('Colab: project found at', PROJECT_ROOT)
except ImportError:
    print('Local machine detected - skipping Drive mount.')
"""))
cells.append(code("""# Install dependencies (Colab only; no-op locally since packages already present)
try:
    from google.colab import drive
    get_ipython().system('pip install -q "xgboost>=3.4,<3.5" "imbalanced-learn>=0.14,<0.15" "shap>=0.52,<0.53" "pyarrow>=25"')
    print('Dependencies installed.')
except ImportError:
    print('Local machine - skipping install.')
"""))

# ===================================================================== 0. environment
cells.append(md("## 0. Environment & project layout"))
cells.append(code("""import sys
from pathlib import Path

# Ensure runtime directories exist (run_experiment.py writes a log file on import)
Path("logs").mkdir(exist_ok=True)
Path("outputs/eda").mkdir(parents=True, exist_ok=True)

print("Python:", sys.version.split()[0])
for pkg in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "imblearn", "shap",
            "matplotlib", "seaborn", "pyarrow", "joblib"):
    mod = __import__(pkg)
    print(f"{pkg:12s} {mod.__version__}")
"""))
cells.append(code("""# Project layout (this notebook lives in the project root)
from pathlib import Path
ROOT = Path.cwd()
for p in sorted(ROOT.iterdir()):
    if p.name.startswith("."):
        continue
    kind = "DIR " if p.is_dir() else "file"
    print(f"{kind}  {p.name}")
"""))

# ===================================================================== 1. raw data
cells.append(md("""## 1. Loading the raw data

`data/raw/application_train.csv` — the Home Credit Default Risk dataset:
one row per loan application, `TARGET = 1` means the client defaulted."""))
cells.append(code("""import pandas as pd
import numpy as np

RAW_PATH = ROOT / "data" / "raw" / "application_train.csv"
print(f"File size: {RAW_PATH.stat().st_size / 1e6:.0f} MB")

df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.head()
"""))
cells.append(code("""# Column types at a glance
print(df.dtypes.value_counts().to_string())
print()
cat_cols  = df.select_dtypes(include=["object", "str"]).columns.tolist()
flag_cols = [c for c in df.columns if c.startswith("FLAG_")]
print(f"Categorical columns : {len(cat_cols)}")
print(f"Binary FLAG columns : {len(flag_cols)}")
print(f"Numeric columns     : {df.shape[1] - len(cat_cols) - 2}")
"""))

# ===================================================================== 2. EDA
cells.append(md("""## 2. Exploratory Data Analysis — highlights

The full EDA (11 sections: overview, duplicates, imbalance, temporal structure,
missingness, anomalies, distributions, categoricals, correlations, multicollinearity,
leakage candidates) was run with `scripts/eda.py`. Its figures and tables live in
`outputs/eda/`. The findings below are what drove the experiment design."""))
cells.append(code("""from IPython.display import Image, display
EDA = ROOT / "outputs" / "eda"

print("Class imbalance — only ~8% of loans default:")
display(Image(filename=str(EDA / "01_target_distribution.png"), width=430))
"""))
cells.append(code("""print("Temporal structure — application volume and default rate drift over time.")
print("This is WHY the experiment uses a time-based (out-of-time) split instead of a random split:")
display(Image(filename=str(EDA / "02_application_date_trend.png"), width=780))
"""))
cells.append(code("""print("Missing values — top 30 columns (red line = 50%):")
display(Image(filename=str(EDA / "03_missing_top30.png"), width=820))
"""))
cells.append(code("""print("Which features correlate with default? EXT_SOURCE external scores dominate:")
display(Image(filename=str(EDA / "06_correlation_top20.png"), width=560))
"""))
cells.append(code("""print("Multicollinearity found by EDA — near-duplicate feature pairs (|r| > 0.9).")
print("This is WHY the experiment drops 30 redundant columns:")
pairs = pd.read_csv(EDA / "high_corr_pairs.csv")
display(pairs.head(12))
print(f"... {len(pairs)} pairs total with |r| > 0.9")
"""))

# ===================================================================== 3. sentinel
cells.append(md("""## 3. Cleaning step 1 — the `DAYS_EMPLOYED = 365243` sentinel

The EDA found that `DAYS_EMPLOYED` uses the value **365243** (\"1000 years\") as a
sentinel for unemployed/retired applicants. Left in place it is a massive outlier and
creates a fake -0.9998 correlation with `FLAG_EMP_PHONE`. The experiment corrects it:
sentinel rows become missing (imputed per fold later) and get an explicit
`DAYS_EMPLOYED_ANOM` flag column."""))
cells.append(code("""sentinel_mask = df["DAYS_EMPLOYED"] == 365243
print(f"Rows with the sentinel value : {sentinel_mask.sum():,}  ({sentinel_mask.mean():.2%} of all rows)")
print(f"Default rate in sentinel rows: {df.loc[sentinel_mask, 'TARGET'].mean():.3%}")
print(f"Default rate in normal rows  : {df.loc[~sentinel_mask, 'TARGET'].mean():.3%}")
print()
print("BEFORE correction:")
print(df["DAYS_EMPLOYED"].describe().round(1).to_string())
"""))
cells.append(code("""# Apply the correction exactly as src/features/engineering.py does
corrected = df["DAYS_EMPLOYED"].copy()
corrected[sentinel_mask] = np.nan

print("AFTER correction (sentinel -> NaN, imputed per fold during modelling):")
print(corrected.describe().round(1).to_string())
print()
print("The corrected values feed the engineering step in Section 5.")
"""))

# ===================================================================== 4. income cap
cells.append(md("""## 4. Cleaning step 2 — income winsorisation (leakage-controlled)

Extreme incomes are capped at the **99.9th percentile**. Crucially, the cap is fitted
on the **training years (2018-2019) only** — computing it on all years would leak
2020 information into preprocessing."""))
cells.append(code("""df["_year"] = pd.to_datetime(df["application_date"]).dt.year

train_mask = df["_year"].isin([2018, 2019])
income_cap = float(df.loc[train_mask, "AMT_INCOME_TOTAL"].quantile(0.999))
print(f"Income cap (99.9th pct of 2018-2019 only): {income_cap:,.0f}")

n_clipped = int((df["AMT_INCOME_TOTAL"] > income_cap).sum())
print(f"Rows clipped by this cap: {n_clipped}")
print()
print("Top 5 incomes BEFORE capping:")
print(df["AMT_INCOME_TOTAL"].nlargest(5).to_string())
df.drop(columns=["_year"], inplace=True)
"""))

# ===================================================================== 5. engineering + split
cells.append(md("""## 5. Feature engineering & the temporal split

`src/features/engineering.py` applies the sentinel correction, the income cap, drops
the 30 near-duplicate columns found in EDA, and derives ratio/aggregate features.
Then the data is split **by time**: train = 2018-2019, test = 2020.

The engineered splits are saved as **parquet checkpoints** in `data/processed/` —
every downstream step (CV, OOT, SHAP) loads these files instead of re-processing."""))
cells.append(code("""import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# The exact production function (same code the full run used)
from run_experiment import load_and_split

df_train, df_test = load_and_split()
"""))
cells.append(code("""print(f"TRAIN (2018-2019): {df_train.shape[0]:,} rows x {df_train.shape[1]} columns | default rate {df_train['TARGET'].mean():.4%}")
print(f"TEST  (2020)     : {df_test.shape[0]:,} rows x {df_test.shape[1]} columns | default rate {df_test['TARGET'].mean():.4%}")
print()
print("Parquet checkpoints written to data/processed/ :")
for f in sorted((ROOT / "data" / "processed").glob("*.parquet")):
    print(f"  {f.name:42s} {f.stat().st_size / 1e6:6.1f} MB")
"""))
cells.append(code("""# What the engineered feature set looks like
print(f"{df_train.shape[1] - 3} feature columns + ID + TARGET + app_year. First 25:")
print(", ".join(df_train.columns[:25]), "...")
"""))

# ===================================================================== 6. one fold
cells.append(md("""## 6. Inside ONE cross-validation fold

This is the heart of the experiment, opened up. For a single fold we will visibly:
1. split the training window into a train-fold and a validation-fold,
2. fit the preprocessor **on the train-fold only** (no leakage),
3. apply SMOTE to the train-fold,
4. fit one model and score it on the held-out validation-fold.

This is exactly what `run_experiment.py` repeats 135 times (3 repeats x 5 folds x 9 configs)."""))
cells.append(code("""from sklearn.model_selection import StratifiedKFold
from src.features.preprocessor import fit_transform, transform_only
from src.models.factory import make_model
from src.models.metrics import compute_metrics
from src.config import RANDOM_SEED, FIXED_THRESHOLD, DEFAULT_COST_RATIO, SMOTE_K_NEIGHBORS
from imblearn.over_sampling import SMOTE

y_all = df_train["TARGET"].to_numpy(dtype=np.int8)
n = len(df_train)

# Repeat 1, fold 1 (seed = 42 for repeat 1)
seed = RANDOM_SEED
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
tr_pos, va_pos = next(iter(skf.split(np.zeros(n), y_all)))

df_tr, df_va = df_train.iloc[tr_pos], df_train.iloc[va_pos]
y_tr, y_va = y_all[tr_pos], y_all[va_pos]
print(f"Train-fold rows: {len(y_tr):,}  |  Validation-fold rows: {len(y_va):,}")
"""))
cells.append(code("""# Step A: fit preprocessor on the TRAIN FOLD ONLY
X_tr, feat_names, pre, num_cols, cat_cols = fit_transform(df_tr)
X_va = transform_only(df_va, pre, num_cols, cat_cols)
print(f"Encoded feature matrix: {X_tr.shape[1]} features (numeric {len(num_cols)} + one-hot categorical)")
print(f"First 8 feature names: {feat_names[:8]}")
"""))
cells.append(code("""# Step B: SMOTE on the train fold only (never on validation)
spw = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)
smote = SMOTE(sampling_strategy="auto", k_neighbors=SMOTE_K_NEIGHBORS, random_state=seed)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
print(f"Train-fold before SMOTE: {len(y_tr):,} rows  ({y_tr.mean():.2%} default)")
print(f"Train-fold after  SMOTE: {len(y_tr_sm):,} rows  ({y_tr_sm.mean():.2%} default)")
print(f"scale_pos_weight (from train fold): {spw:.2f}")
"""))
cells.append(code("""# Step C: fit ONE model (XGBoost baseline) and score on the held-out fold
import time
t0 = time.time()
model = make_model("xgboost", "baseline", spw, seed + 1)
model.fit(X_tr, y_tr)
proba = model.predict_proba(X_va)[:, 1]
m = compute_metrics(y_va, proba, FIXED_THRESHOLD, DEFAULT_COST_RATIO)
print(f"Fit + predict took {time.time()-t0:.1f}s")
print()
print("Fold-1 metrics for XGBoost (Baseline):")
for k, v in m.items():
    print(f"  {k:22s} {v:.4f}")
"""))

# ===================================================================== 7. live mini CV
cells.append(md("""## 7. A live mini cross-validation — watch the CSV being built

To keep this notebook fast, we run a **small live CV** (1 repeat x 2 folds x 3 configs)
right here, and you can watch the fold records accumulate into the same table structure
the full run produces. (The full 3x5x9 run is loaded from its checkpoint in Section 8.)"""))
cells.append(code("""from run_experiment import MODELS, STRATEGIES, DISPLAY

mini_records = []
mini_models = ["logistic_regression", "xgboost"]          # 2 families
mini_strats = ["baseline", "cost_sensitive"]                # 2 strategies
skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)

for fold_idx, (tr_pos, va_pos) in enumerate(skf2.split(np.zeros(n), y_all), start=1):
    df_tr, df_va = df_train.iloc[tr_pos], df_train.iloc[va_pos]
    y_tr, y_va = y_all[tr_pos], y_all[va_pos]
    X_tr, feat_names, pre, num_cols, cat_cols = fit_transform(df_tr)
    X_va = transform_only(df_va, pre, num_cols, cat_cols)
    spw = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)
    for mn in mini_models:
        for st in mini_strats:
            mdl = make_model(mn, st, spw, RANDOM_SEED + fold_idx)
            mdl.fit(X_tr, y_tr)
            proba = mdl.predict_proba(X_va)[:, 1]
            m = compute_metrics(y_va, proba, FIXED_THRESHOLD, DEFAULT_COST_RATIO)
            rec = {"fold": fold_idx, "display_name": DISPLAY[(mn, st)], **m}
            mini_records.append(rec)
            print(f"  fold {fold_idx} | {DISPLAY[(mn,st)]:34s} ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")

mini_df = pd.DataFrame(mini_records)
print(f"\\nBuilt {len(mini_df)} fold records live.")
"""))
cells.append(code("""# This is the exact table shape that becomes cv_fold_results.csv in the full run
mini_df.round(4)
"""))

# ===================================================================== 8. full CV from checkpoint
cells.append(md("""## 8. The full 3x5 repeated cross-validation (135 fold records)

The production run (`run_experiment.py`) executed **3 repeats x 5 folds x 9 configs =
135 model fits** and checkpointed every fold to `reports/tables/cv_fold_results.csv`.
We load that checkpoint here (same seeds + same data = identical numbers)."""))
cells.append(code("""TABLES = ROOT / "reports" / "tables"
cv = pd.read_csv(TABLES / "cv_fold_results.csv")
print(f"Loaded {len(cv)} fold records from cv_fold_results.csv")
print(f"Columns: {list(cv.columns)}")
cv.head()
"""))
cells.append(code("""# Aggregate: mean +/- std across the 15 observations per configuration
agg = (cv.groupby("display_name")
         .agg(roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean"),
              recall=("recall", "mean"), precision=("precision", "mean"),
              f1=("f1", "mean"), cost=("cost_per_applicant", "mean"))
         .sort_values("pr_auc", ascending=False))
print("Champion by mean PR-AUC is at the top:")
agg.round(4)
"""))

# ===================================================================== 9. OOT
cells.append(md("""## 9. Out-of-time (2020) evaluation

Each of the 9 configs is fit on the **full** 2018-2019 window and scored once on the
untouched 2020 cohort. Results are in `reports/tables/oot_results.csv`; the evaluation
figures are generated by `src/models/plots.py`."""))
cells.append(code("""oot = pd.read_csv(TABLES / "oot_results.csv")
print("Out-of-time results, sorted by PR-AUC:")
oot.sort_values("pr_auc", ascending=False).round(4)
"""))
cells.append(code("""FIG = ROOT / "reports" / "figures"
print("ROC curves (OOT 2020):")
display(Image(filename=str(FIG / "roc_curves_oot.png"), width=720))
print("\\nPrecision-Recall curves (OOT 2020) — the honest metric at 8% base rate:")
display(Image(filename=str(FIG / "pr_curves_oot.png"), width=720))
"""))
cells.append(code("""print("Confusion matrices (OOT 2020):")
display(Image(filename=str(FIG / "confusion_matrices_oot.png"), width=820))
print("\\nCost curves (OOT 2020):")
display(Image(filename=str(FIG / "cost_curves_oot.png"), width=720))
"""))

# ===================================================================== 10. statistics
cells.append(md("""## 10. Statistical significance battery

Paired tests across the 15 CV observations per comparison (16 comparisons total).
Results in `reports/tables/statistical_significance.csv`."""))
cells.append(code("""stats = pd.read_csv(TABLES / "statistical_significance.csv")
n_sig = int(stats["significant_alpha_0.05"].sum())
print(f"Significant at alpha=0.05: {n_sig} / {len(stats)} comparisons")
stats.round(6)
"""))
cells.append(code("""print("Significance heatmap:")
display(Image(filename=str(FIG / "statistical_significance_heatmap.png"), width=760))
"""))

# ===================================================================== 11. SHAP
cells.append(md("""## 11. SHAP explainability

TreeSHAP on the champion XGBoost and LinearSHAP on the logistic baseline, computed on
the 2018-2019 window. Importance tables, beeswarm, dependence panels and three worked
case studies (TP / TN / FP)."""))
cells.append(code("""imp = pd.read_csv(TABLES / "shap_importance_top30.csv")
print("Top 15 features by mean |SHAP| (XGBoost / TreeSHAP):")
imp.head(15).round(4)
"""))
cells.append(code("""print("Global importance (TreeSHAP):")
display(Image(filename=str(FIG / "shap_importance.png"), width=720))
print("\\nBeeswarm (TreeSHAP):")
display(Image(filename=str(FIG / "shap_beeswarm.png"), width=780))
"""))
cells.append(code("""print("Feature dependence panels:")
display(Image(filename=str(FIG / "shap_dependence.png"), width=860))
"""))
cells.append(code("""print("Local case studies (clean custom bar charts drawn from the SHAP values):")
display(Image(filename=str(FIG / "shap_case_studies.png"), width=900))
print("\\nCase-study table (top-5 drivers per applicant):")
display(pd.read_csv(TABLES / "shap_case_studies.csv"))
"""))

# ===================================================================== 12. inventory
cells.append(md("""## 12. Complete artifact inventory

Every file this experiment produces, grouped by folder. This is what a teammate will
find after running `run_experiment.py` + `run_analysis.py` (or the Colab notebooks)."""))
cells.append(code("""def show_tree(base, label):
    files = sorted(base.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"--- {label} ({len(files)} files) ---")
    for f in files:
        rel = f.relative_to(ROOT)
        size = f.stat().st_size
        unit = "MB" if size > 1e6 else "KB"
        val = size / 1e6 if size > 1e6 else size / 1e3
        print(f"  {str(rel):60s} {val:8.1f} {unit}")
    print()

show_tree(ROOT / "data" / "processed", "data/processed  (engineered splits)")
show_tree(ROOT / "reports" / "tables", "reports/tables  (all numeric results)")
show_tree(ROOT / "reports" / "figures", "reports/figures (all figures)")
show_tree(ROOT / "models", "models          (saved artefacts)")
"""))
cells.append(md("""## Done

You have now seen **every step** of the experiment — raw data, EDA, each cleaning
decision, the temporal split, the mechanics of a single CV fold, a live mini CV, the
full 135-fold results, the out-of-time evaluation, the statistical battery, the SHAP
explainability, and the complete list of artifacts.

To reproduce from scratch on a fresh machine, run the two production entry points:

```bash
python run_experiment.py     # cleaning + 3x5 CV + OOT  (~1-3 h)
python run_analysis.py       # statistics + plots + SHAP (~10-20 min)
```

or open `colab/RUN_ALL.ipynb` in Google Colab."""))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"cells": cells,
               "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                            "language_info": {"name": "python"}},
               "nbformat": 4, "nbformat_minor": 5}, f, indent=1)
print(f"Part 1 written: {len(cells)} cells -> {OUT.name}")
