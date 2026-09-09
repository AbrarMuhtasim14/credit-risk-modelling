"""Generate the colab/ handover folder: 6 rich, self-documenting notebooks + README.

Design goals (v2 — full experiment from EDA to SHAP):
- The ENTIRE project folder is uploaded to Google Drive.
- Notebooks mount Drive, locate the project root, and work IN PLACE, so every
  output (figures, parquets, CSVs, logs) is automatically persistent.
- Phase 0 is a COMPLETE exploratory data analysis with inline figures.
- Phases 1-4 run the real modelling/statistics/SHAP pipeline by importing the
  SAME src/ package and run_experiment.py as the local artefact, so results are
  bit-identical; each phase also DISPLAYS its tables and figures inline so the
  work is visible, not hidden behind a function call.
- Each phase checks for its checkpoint and skips/resumes accordingly.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLAB = ROOT / "colab"
COLAB.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "Credit Risk MSc Project", "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


SETUP_CELL = '''# ============================================================
# SETUP: mount Google Drive and locate the project folder
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os

# AUTO-DETECT the project root (folder containing src/ and run_experiment.py).
# If auto-detection fails, set PROJECT_ROOT manually, e.g.:
#   PROJECT_ROOT = Path('/content/drive/MyDrive/credit-risk-project - Copy')
PROJECT_ROOT = None
for candidate in Path('/content/drive/MyDrive').rglob('run_experiment.py'):
    if (candidate.parent / 'src').is_dir():
        PROJECT_ROOT = candidate.parent
        break
assert PROJECT_ROOT is not None, "Could not find the project folder on Drive - set PROJECT_ROOT manually"

os.chdir(PROJECT_ROOT)
import sys
sys.path.insert(0, str(PROJECT_ROOT))
# Ensure expected directories exist (log files are opened at import time)
for _d in ('logs', 'data/raw', 'data/processed', 'models',
           'reports/tables', 'reports/figures', 'outputs/eda'):
    os.makedirs(PROJECT_ROOT / _d, exist_ok=True)
print(f"Project root: {PROJECT_ROOT}")
print("Working directory set. All outputs are saved here (persistent on Drive).")
'''

# ================================================================ Phase 00 — EDA
nb00 = notebook([
    md("""# Phase 0 — Exploratory Data Analysis

**MSc Credit Risk Project — Colab handover (0 of 6)**

A complete EDA of the Home Credit `application_train.csv` dataset. This is the
investigative work that motivated every design decision in the experiment:

1. Dataset overview & dtypes
2. Duplicate detection
3. Target variable & class imbalance
4. Temporal structure (`application_date`) — why we use a time-based split
5. Missing values
6. Known anomalies & data quality (the `DAYS_EMPLOYED = 365243` sentinel)
7. Key numeric feature distributions
8. Categorical features & default rate by category
9. Correlation with the target
10. Multicollinearity (high inter-feature correlation)
11. Potential leakage candidates

All figures are saved to `outputs/eda/` on Drive. **Runtime: ~5 minutes.**
"""),
    code(SETUP_CELL),
    code("""# Verify the raw dataset is present before doing anything
from pathlib import Path
RAW = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
DESC = PROJECT_ROOT / "data" / "raw" / "HomeCredit_columns_description.csv"
assert RAW.exists(), (
    f"Missing {RAW.name}! Upload application_train.csv into the project's data/raw/ "
    f"folder on Drive, then re-run this notebook."
)
print(f"Raw dataset found: {RAW.stat().st_size / 1e6:.0f} MB")
"""),
    code("""import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT = PROJECT_ROOT / "outputs" / "eda"
sns.set_style("whitegrid")

# ---------------------------------------------------------------- 1. OVERVIEW
print("=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)
df = pd.read_csv(RAW, low_memory=False)
desc = pd.read_csv(DESC, encoding="latin-1")
desc_map = dict(zip(desc["Row"], desc["Description"]))

print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
print("\\nDtype counts:")
print(df.dtypes.value_counts().to_string())

cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
flag_cols = [c for c in df.columns if c.startswith("FLAG_")]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ("SK_ID_CURR", "TARGET") and c not in flag_cols]
print(f"\\nCategorical: {len(cat_cols)} | Binary flags: {len(flag_cols)} | Numeric: {len(num_cols)}")
df.head()
"""),
    code("""# ---------------------------------------------------------------- 2. DUPLICATES
print("=" * 70)
print("2. DUPLICATES")
print("=" * 70)
dup_full = df.duplicated().sum()
dup_id = df.duplicated(subset=["SK_ID_CURR"]).sum()
print(f"Exact duplicate rows: {dup_full:,}")
print(f"Duplicate SK_ID_CURR: {dup_id:,}")
"""),
    code("""# ---------------------------------------------------------------- 3. TARGET / IMBALANCE
print("=" * 70)
print("3. TARGET VARIABLE (class imbalance)")
print("=" * 70)
vc = df["TARGET"].value_counts()
print(vc.to_string())
rate = df["TARGET"].mean()
print(f"\\nDefault rate: {rate:.4%}")
print(f"Imbalance ratio (non-default : default): {vc[0]/vc[1]:.2f} : 1")

fig, ax = plt.subplots(figsize=(5, 4))
vc.plot.bar(ax=ax, color=["#4c72b0", "#c44e52"], edgecolor="black")
ax.set_xticklabels(["0 = repaid", "1 = default"], rotation=0)
ax.set_ylabel("Count")
ax.set_title(f"TARGET distribution (default rate {rate:.2%})")
for i, v in enumerate(vc):
    ax.text(i, v + 3000, f"{v:,}", ha="center")
plt.tight_layout(); plt.savefig(OUT / "01_target_distribution.png", dpi=150); plt.show()
"""),
    code("""# ---------------------------------------------------------------- 4. TEMPORAL STRUCTURE
print("=" * 70)
print("4. APPLICATION DATE (temporal structure)")
print("=" * 70)
ad = pd.to_datetime(df["application_date"], errors="coerce")
print(f"Parse failures: {ad.isna().sum():,}")
print(f"Range: {ad.min()}  ->  {ad.max()}")
monthly = df.assign(_m=ad.dt.to_period("M")).groupby("_m")["TARGET"].agg(["count", "mean"])
print("\\nMonthly applications & default rate (first 6 / last 6):")
print(monthly.head(6).round(4).to_string())
print("...")
print(monthly.tail(6).round(4).to_string())

fig, ax1 = plt.subplots(figsize=(11, 4))
x = monthly.index.to_timestamp()
ax1.bar(x, monthly["count"], width=20, color="#4c72b0", alpha=0.7, label="applications")
ax1.set_ylabel("Applications per month")
ax2 = ax1.twinx()
ax2.plot(x, monthly["mean"], color="#c44e52", lw=1.5, label="default rate")
ax2.set_ylabel("Default rate")
ax1.set_title("Application volume and default rate over time")
plt.tight_layout(); plt.savefig(OUT / "02_application_date_trend.png", dpi=150); plt.show()
"""),
    md("""**Why this matters:** applications span 2015-2020 and the default rate *drifts*
over time. A random train/test split would leak future information into the past.
This is the direct evidence for using a **temporal out-of-time split**
(fit 2018-2019, validate on 2020) rather than a random split.
"""),
    code("""# ---------------------------------------------------------------- 5. MISSING VALUES
print("=" * 70)
print("5. MISSING VALUES")
print("=" * 70)
miss = df.isna().mean().sort_values(ascending=False)
miss_n = df.isna().sum()
print(f"Columns with any missing: {(miss > 0).sum()} / {df.shape[1]}")
print(f"Columns >50% missing: {(miss > 0.5).sum()}")
print(f"Columns >30% missing: {(miss > 0.3).sum()}")
print(f"Columns >10% missing: {(miss > 0.1).sum()}")
print("\\nTop 25 by missing %:")
top_miss = pd.DataFrame({"missing_pct": (miss * 100).round(2), "missing_n": miss_n}).head(25)
top_miss["description"] = top_miss.index.map(lambda c: str(desc_map.get(c, ""))[:60])
print(top_miss.to_string())
miss.to_frame("missing_pct").to_csv(OUT / "missing_by_column.csv")

fig, ax = plt.subplots(figsize=(12, 5))
top30 = miss.head(30) * 100
ax.bar(range(len(top30)), top30.values, color="#dd8452")
ax.set_xticks(range(len(top30)))
ax.set_xticklabels(top30.index, rotation=80, fontsize=7)
ax.set_ylabel("Missing %")
ax.set_title("Top 30 columns by missing values")
ax.axhline(50, color="red", ls="--", lw=1)
plt.tight_layout(); plt.savefig(OUT / "03_missing_top30.png", dpi=150); plt.show()

miss_by_t = df.groupby("TARGET").apply(lambda g: g.isna().mean().mean(), include_groups=False)
print(f"\\nAvg cell missingness - default=1: {miss_by_t.get(1, np.nan):.4f} | default=0: {miss_by_t.get(0, np.nan):.4f}")
"""),
    code("""# ---------------------------------------------------------------- 6. ANOMALIES / DATA QUALITY
print("=" * 70)
print("6. KNOWN ANOMALIES & DATA QUALITY")
print("=" * 70)
anom = df["DAYS_EMPLOYED"] == 365243
print(f"DAYS_EMPLOYED == 365243 (sentinel for unemployed/retired): {anom.sum():,} ({anom.mean():.2%})")
print(f"  default rate in sentinel group: {df.loc[anom, 'TARGET'].mean():.3%} vs others: {df.loc[~anom, 'TARGET'].mean():.3%}")

neg_birth = (df["DAYS_BIRTH"] >= 0).sum()
print(f"DAYS_BIRTH >= 0 (invalid): {neg_birth:,}")
age_years = -df["DAYS_BIRTH"] / 365
print(f"Age range derived from DAYS_BIRTH: {age_years.min():.1f} - {age_years.max():.1f} years")

for c in ["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE", "OCCUPATION_TYPE"]:
    xna = (df[c].astype(str) == "XNA").sum()
    if xna:
        print(f"{c}: 'XNA' values = {xna:,}")

print(f"\\nAMT_ANNUITY: min={df['AMT_ANNUITY'].min():,.0f} max={df['AMT_ANNUITY'].max():,.0f} "
      f"(missing {df['AMT_ANNUITY'].isna().mean():.2%})")
print(f"AMT_GOODS_PRICE missing: {df['AMT_GOODS_PRICE'].isna().mean():.2%}")
print(f"AMT_INCOME_TOTAL: min={df['AMT_INCOME_TOTAL'].min():,.0f} max={df['AMT_INCOME_TOTAL'].max():,.0f}")
hi = (df["AMT_INCOME_TOTAL"] > 1e6).sum()
print(f"  incomes > 1,000,000: {hi:,}")
"""),
    md("""**Key finding:** `DAYS_EMPLOYED = 365243` is a sentinel (\"1000 years\") marking
unemployed/retired applicants, present in ~18% of rows. Left uncorrected it acts as a
massive outlier and creates a spurious -0.9998 correlation with `FLAG_EMP_PHONE`.
The experiment corrects this sentinel and winsorises income at the 99.9th percentile.
"""),
    code("""# ---------------------------------------------------------------- 7. NUMERIC DISTRIBUTIONS
print("=" * 70)
print("7. KEY NUMERIC FEATURES")
print("=" * 70)
key_num = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
           "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
           "REGION_POPULATION_RELATIVE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
key_num = [c for c in key_num if c in df.columns]
print(df[key_num].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(2).to_string())

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, c in zip(axes.ravel(), ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                                 "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
    s = df[c].dropna()
    if c.startswith("AMT"):
        s = s.clip(upper=s.quantile(0.99))
    ax.hist(s, bins=60, color="#4c72b0", edgecolor="none")
    ax.set_title(c, fontsize=10)
plt.suptitle("Key numeric feature distributions (AMT clipped at 99th pct)")
plt.tight_layout(); plt.savefig(OUT / "04_numeric_distributions.png", dpi=150); plt.show()

print("\\nEXT_SOURCE columns (external risk scores):")
for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
    m = df[c].isna().mean()
    corr = df[[c, "TARGET"]].corr().iloc[0, 1]
    print(f"  {c}: missing {m:.2%}, corr with TARGET = {corr:.3f}")
"""),
    code("""# ---------------------------------------------------------------- 8. CATEGORICAL FEATURES
print("=" * 70)
print("8. CATEGORICAL FEATURES")
print("=" * 70)
card = pd.DataFrame({
    "n_unique": df[cat_cols].nunique(),
    "missing_pct": (df[cat_cols].isna().mean() * 100).round(2),
}).sort_values("n_unique", ascending=False)
print(card.to_string())

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_cats = ["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS",
             "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "NAME_CONTRACT_TYPE"]
for ax, c in zip(axes.ravel(), plot_cats):
    g = df.groupby(c, dropna=False)["TARGET"].agg(["mean", "count"]).sort_values("mean", ascending=False).head(10)
    ax.bar(range(len(g)), g["mean"] * 100, color="#55a868")
    ax.axhline(rate * 100, color="red", ls="--", lw=1)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(x)[:16] for x in g.index], rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Default rate by {c}", fontsize=9)
    ax.set_ylabel("Default %")
plt.suptitle("Default rate by category (red line = overall average)")
plt.tight_layout(); plt.savefig(OUT / "05_default_rate_by_category.png", dpi=150); plt.show()
"""),
    code("""# ---------------------------------------------------------------- 9. CORRELATION WITH TARGET
print("=" * 70)
print("9. CORRELATION WITH TARGET")
print("=" * 70)
num_df = df[num_cols + flag_cols].copy()
for c in num_df.columns:
    if num_df[c].dtype == object or str(num_df[c].dtype) in ("str", "string"):
        num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
corr_target = num_df.corrwith(df["TARGET"]).abs().sort_values(ascending=False)
print("Top 20 features by |corr| with TARGET:")
top20 = pd.DataFrame({"abs_corr": corr_target.head(20).round(4)})
top20["signed_corr"] = num_df.corrwith(df["TARGET"]).round(4)
top20["description"] = top20.index.map(lambda c: str(desc_map.get(c, ""))[:70])
print(top20.to_string())
top20.to_csv(OUT / "correlation_with_target.csv")

fig, ax = plt.subplots(figsize=(8, 7))
t = corr_target.head(20)
ax.barh(range(len(t))[::-1], t.values, color="#4c72b0")
ax.set_yticks(range(len(t))[::-1])
ax.set_yticklabels(t.index, fontsize=8)
ax.set_xlabel("|Pearson corr| with TARGET")
ax.set_title("Top 20 features correlated with default")
plt.tight_layout(); plt.savefig(OUT / "06_correlation_top20.png", dpi=150); plt.show()
"""),
    code("""# ---------------------------------------------------------------- 10. MULTICOLLINEARITY
print("=" * 70)
print("10. HIGHLY CORRELATED FEATURE PAIRS (multicollinearity candidates)")
print("=" * 70)
sample = num_df.sample(min(30000, len(num_df)), random_state=42)
cm = sample.corr()
pairs = []
for i in range(len(cm.columns)):
    for j in range(i + 1, len(cm.columns)):
        r = cm.iloc[i, j]
        if abs(r) > 0.9:
            pairs.append((cm.columns[i], cm.columns[j], round(r, 3)))
pairs.sort(key=lambda x: -abs(x[2]))
for p in pairs[:20]:
    print(f"  {p[0]} <-> {p[1]}: {p[2]}")
print(f"  ... total pairs |r|>0.9: {len(pairs)}")
pd.DataFrame(pairs, columns=["feat1", "feat2", "corr"]).to_csv(OUT / "high_corr_pairs.csv", index=False)
"""),
    md("""**Key finding:** the building features come in `_AVG`/`_MEDI`/`_MODE` triplets that are
near-duplicates (|r| > 0.97), and `OBS_60`/`DEF_60` social-circle columns duplicate the
30-DPD variants. This is the evidence for **dropping 30 redundant columns** in the
experiment's feature-selection step.
"""),
    code("""# ---------------------------------------------------------------- 11. LEAKAGE CANDIDATES
print("=" * 70)
print("11. POTENTIAL DATA LEAKAGE / SCOPE CANDIDATES (review & document)")
print("=" * 70)
suspects = ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
            "LIVE_CITY_NOT_WORK_CITY", "REG_CITY_NOT_WORK_CITY",
            "DAYS_LAST_PHONE_CHANGE", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3"]
for c in suspects:
    if c in df.columns:
        print(f"  {c}: {str(desc_map.get(c, ''))[:90]}")

print("\\n" + "=" * 70)
print("EDA COMPLETE - figures saved to outputs/eda/ on Drive")
print("=" * 70)
"""),
    md("""## EDA summary — decisions this analysis drove

| EDA finding | Experiment decision |
|---|---|
| Default rate drifts over time | Temporal OOT split (fit 2018-19, test 2020) |
| 8.1% default rate (heavy imbalance) | PR-AUC as primary metric; SMOTE & cost-sensitive strategies |
| `DAYS_EMPLOYED=365243` sentinel (~18%) | Sentinel correction before modelling |
| Extreme income outliers | Winsorise income at 99.9th pct (fit on train years only) |
| 30 near-duplicate column pairs | Drop 30 redundant columns |
| High missingness in EXT_SOURCE | Per-fold imputation, no global statistics |
"""),
])

# ================================================================ Phase 01 — Data Prep
nb01 = notebook([
    md("""# Phase 1 — Data Preparation & Temporal Split

**MSc Credit Risk Project — Colab handover (1 of 6)**

Applies the full cleaning and feature-engineering pipeline (the exact same code
that produced the dissertation results) and creates the temporal split:

- `DAYS_EMPLOYED = 365243` sentinel correction
- Income winsorisation at the 99.9th percentile (fitted on training years only — leakage control)
- Drop 30 near-duplicate columns (found in Phase 0 EDA)
- Derived features (income/credit ratios, document counts, etc.)
- Temporal split: **train = 2018-2019, test = 2020**

**Checkpoint produced:** `data/processed/engineered_train_2018_2019.parquet` +
`engineered_test_2020.parquet`. **Runtime: ~5 minutes** (skipped entirely if the
parquets already exist on Drive).
"""),
    code(SETUP_CELL),
    code("""# Install pinned dependencies (Colab pre-installs most; these ensure version parity
# with the local artefact that produced the dissertation numbers).
!pip install -q "xgboost>=3.4,<3.5" "imbalanced-learn>=0.14,<0.15" "shap>=0.52,<0.53" "pyarrow>=25"
"""),
    code("""# Verify the raw dataset is present
from src.config import RAW_APPLICATION_TRAIN
assert RAW_APPLICATION_TRAIN.exists(), (
    f"Missing {RAW_APPLICATION_TRAIN.name}!\\n"
    f"Upload application_train.csv (~161 MB) into the project's data/raw/ folder on Drive,\\n"
    f"then re-run this notebook."
)
print(f"Raw dataset found: {RAW_APPLICATION_TRAIN} ({RAW_APPLICATION_TRAIN.stat().st_size / 1e6:.0f} MB)")
"""),
    code("""# Load, clean, engineer, split (uses the SAME code as the local artefact).
# If the engineered parquets already exist on Drive, load_and_split() loads them
# directly (bit-identical, no re-processing). Otherwise it processes the raw CSV
# and saves the parquets as checkpoints.
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from run_experiment import load_and_split
from src.config import PROCESSED_DIR

df_train, df_test = load_and_split()
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
if not (PROCESSED_DIR / "engineered_train_2018_2019.parquet").exists():
    df_train.to_parquet(PROCESSED_DIR / "engineered_train_2018_2019.parquet", index=False)
    df_test.to_parquet(PROCESSED_DIR / "engineered_test_2020.parquet", index=False)
    print("Engineered parquets saved to data/processed/ (persistent on Drive).")
"""),
    code("""# ---- Inspect the engineered splits ----
print(f"TRAIN (2018-2019): {df_train.shape[0]:,} rows x {df_train.shape[1]} columns, "
      f"default rate {df_train['TARGET'].mean():.4%}")
print(f"TEST  (2020):      {df_test.shape[0]:,} rows x {df_test.shape[1]} columns, "
      f"default rate {df_test['TARGET'].mean():.4%}")
df_train.head()
"""),
    code("""# ---- Phase 1 verification (expected values from the dissertation) ----
assert len(df_train) == 205_007, f"train rows {len(df_train)} != 205,007"
assert len(df_test) == 102_504, f"test rows {len(df_test)} != 102,504"
assert abs(df_train['TARGET'].mean() - 0.0812) < 0.001
assert abs(df_test['TARGET'].mean() - 0.0798) < 0.001
print("PHASE 1 CHECKPOINT OK: train=205,007 (8.12% default), test=102,504 (7.98% default)")
"""),
])

# ================================================================ Phase 02 — CV
nb02 = notebook([
    md("""# Phase 2 — Cross-Validation (the long phase)

**MSc Credit Risk Project — Colab handover (2 of 6)**

Runs the **3x5 repeated stratified cross-validation** over all 9 configurations
(3 model families x 3 imbalance strategies = **135 model fits**):

| Family | Strategies |
|---|---|
| Logistic Regression | Baseline / SMOTE / Cost-Sensitive |
| Random Forest | Baseline / SMOTE / Cost-Sensitive |
| XGBoost | Baseline / SMOTE / Cost-Sensitive |

Hyperparameters are **frozen** across imbalance strategies so the comparison is fair.
Primary metric: **PR-AUC** (accuracy is misleading at 8% base rate).

**Checkpoint:** `reports/tables/cv_fold_results.csv` is rewritten after **each repeat**.
If the runtime disconnects, re-run this notebook — it detects the partial checkpoint and
resumes from the next repeat (seeds are tied to the global repeat number, so results
stay bit-identical).

**Runtime:** ~30-45 minutes on Colab CPU for all 3 repeats.
"""),
    code(SETUP_CELL),
    code("""import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import pandas as pd
from src.config import TABLES_DIR, N_REPEATS

cv_path = TABLES_DIR / "cv_fold_results.csv"

if cv_path.exists():
    prior = pd.read_csv(cv_path)
    done = int(prior['repeat'].max())
    if done >= N_REPEATS:
        print(f"Full checkpoint found ({done}/{N_REPEATS} repeats) - SKIPPING CV track.")
        df_cv = prior
    else:
        print(f"Partial checkpoint ({done}/{N_REPEATS} repeats) - RESUMING from repeat {done + 1}...")
        from run_experiment import run_cv_track, load_and_split
        df_train, _ = load_and_split()
        df_new = run_cv_track(df_train, repeat_offset=done, checkpoint_path=cv_path)
        df_cv = pd.concat([prior, df_new], ignore_index=True)
        df_cv.to_csv(cv_path, index=False)
else:
    print("No checkpoint - running full CV track (3 repeats x 5 folds x 9 configs = 135 fits)...")
    from run_experiment import run_cv_track, load_and_split
    df_train, _ = load_and_split()
    df_cv = run_cv_track(df_train, checkpoint_path=cv_path)
    df_cv.to_csv(cv_path, index=False)

print(f"CV track complete: {len(df_cv)} fold records (expected 135).")
"""),
    code("""# ---- Display the aggregated CV results (Table 4.1 of the dissertation) ----
agg = (df_cv.groupby('display_name')
       .agg(roc_auc_mean=('roc_auc', 'mean'), roc_auc_std=('roc_auc', 'std'),
            pr_auc_mean=('pr_auc', 'mean'), pr_auc_std=('pr_auc', 'std'),
            recall_mean=('recall', 'mean'), precision_mean=('precision', 'mean'),
            f1_mean=('f1', 'mean'), f2_mean=('f2', 'mean'),
            cost_mean=('cost_per_applicant', 'mean'))
       .sort_values('pr_auc_mean', ascending=False))
agg.to_csv(TABLES_DIR / "cv_summary.csv", index=False)
print("Aggregated CV results (sorted by mean PR-AUC):")
agg.round(4)
"""),
    code("""# ---- Phase 2 verification against dissertation Table 4.1 ----
xgb_base = agg.loc['XGBoost (Baseline)']
assert abs(xgb_base['pr_auc_mean'] - 0.2502) < 0.005, f"PR-AUC {xgb_base['pr_auc_mean']:.4f} != ~0.2502"
assert abs(xgb_base['roc_auc_mean'] - 0.7622) < 0.005
assert len(df_cv) == 135
print("\\nPHASE 2 CHECKPOINT OK: champion XGBoost (Baseline) PR-AUC ~0.2502, ROC-AUC ~0.7622")
print("Note: SMOTE rows should show LOWER PR-AUC than their baselines - SMOTE is harmful here.")
"""),
])

# ================================================================ Phase 03 — OOT + stats
nb03 = notebook([
    md("""# Phase 3 — Out-of-Time Test & Statistical Testing

**MSc Credit Risk Project — Colab handover (3 of 6)**

1. Trains the 9 configurations on the **full** 2018-2019 window and evaluates once on
   the untouched 2020 cohort (the out-of-time benchmark).
2. Runs the **16-comparison statistical battery** (paired t-tests on the 15 paired CV
   observations per comparison, plus the OOT champion check).
3. Produces the evaluation figures: ROC curves, PR curves, confusion matrices,
   cost curves, and the significance heatmap.

**Checkpoints:** `reports/tables/oot_results.csv`, `models/oot_probas.parquet`, figures.
**Runtime:** ~10 minutes.
"""),
    code(SETUP_CELL),
    code("""import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.config import TABLES_DIR

if (TABLES_DIR / "oot_results.csv").exists():
    print("OOT checkpoint found - SKIPPING OOT track.")
else:
    from run_experiment import load_and_split, run_oot_track
    df_train, df_test = load_and_split()
    df_oot = run_oot_track(df_train, df_test)
    df_oot.to_csv(TABLES_DIR / "oot_results.csv", index=False)
    print("OOT track complete.")
"""),
    code("""# ---- Display the OOT results (Table 4.2 of the dissertation) ----
import pandas as pd
df_oot = pd.read_csv(TABLES_DIR / "oot_results.csv")
print("Out-of-time (2020) results, sorted by PR-AUC:")
df_oot.sort_values('pr_auc', ascending=False).round(4)
"""),
    code("""# Statistical battery + evaluation plots (identical to run_analysis.py steps 1-2)
from src.utils.stats_testing import run_all_tests
from src.config import FIGURES_DIR, MODELS_DIR

df_stats = run_all_tests(
    cv_path=TABLES_DIR / "cv_fold_results.csv",
    out_path=TABLES_DIR / "statistical_significance.csv",
    figure_path=FIGURES_DIR / "statistical_significance_heatmap.png",
)
print(f"Significant at alpha=0.05: {int(df_stats['significant_alpha_0.05'].sum())} / {len(df_stats)}")
df_stats.round(6)
"""),
    code("""# ---- Display the evaluation figures inline ----
from IPython.display import Image, display
from src.models.plots import plot_confusion_grid, plot_cost_curves, plot_pr_curves, plot_roc_curves
probas_path = MODELS_DIR / "oot_probas.parquet"
plot_roc_curves(probas_path, FIGURES_DIR / "roc_curves_oot.png")
plot_pr_curves(probas_path, FIGURES_DIR / "pr_curves_oot.png")
plot_confusion_grid(probas_path, FIGURES_DIR / "confusion_matrices_oot.png")
plot_cost_curves(probas_path, FIGURES_DIR / "cost_curves_oot.png")
print("Evaluation figures saved to reports/figures/\\n")

for name in ["roc_curves_oot.png", "pr_curves_oot.png",
             "confusion_matrices_oot.png", "cost_curves_oot.png",
             "statistical_significance_heatmap.png"]:
    display(Image(str(FIGURES_DIR / name), width=700))
"""),
    code("""# ---- Phase 3 verification against dissertation Table 4.2 / 4.3 ----
champ = df_oot[df_oot['display_name'] == 'XGBoost (Baseline)'].iloc[0]
assert abs(champ['roc_auc'] - 0.7669) < 0.002, champ['roc_auc']
assert abs(champ['pr_auc'] - 0.2513) < 0.005, champ['pr_auc']

cs = df_oot[df_oot['display_name'] == 'XGBoost (Cost-Sensitive)'].iloc[0]
assert abs(cs['recall'] - 0.657) < 0.01, cs['recall']

df_stats = pd.read_csv(TABLES_DIR / "statistical_significance.csv")
assert int(df_stats['significant_alpha_0.05'].sum()) == 15
print("PHASE 3 CHECKPOINT OK: OOT ROC-AUC 0.7669, PR-AUC 0.2513, cost-sensitive recall 0.657, 15/16 significant")
"""),
])

# ================================================================ Phase 04 — SHAP
nb04 = notebook([
    md("""# Phase 4 — SHAP Explainability

**MSc Credit Risk Project — Colab handover (4 of 6)**

Fits the champion XGBoost (TreeSHAP) and the logistic baseline (LinearSHAP) on the
full training window, then computes:

1. Feature importance tables (top 30, both models)
2. Beeswarm plot (TreeSHAP)
3. Dependence panels for the top features
4. **Three worked case studies** (a true positive, a true negative, a false positive)
   with the bar-chart attribution panels from Chapter 5 of the dissertation
5. Tree-vs-linear top-30 overlap analysis

**Checkpoints:** `reports/tables/shap_*.csv`, `reports/figures/shap_*.png`,
`models/champion_xgboost.joblib`. **Runtime:** ~5 minutes.
"""),
    code(SETUP_CELL),
    code("""import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.config import TABLES_DIR

if (TABLES_DIR / "shap_importance_top30.csv").exists():
    print("SHAP checkpoint found - SKIPPING SHAP pipeline.")
else:
    from src.explainability.shap_engine import run_shap_pipeline
    run_shap_pipeline()
    print("SHAP pipeline complete.")
"""),
    code("""# ---- Display SHAP results inline ----
import pandas as pd
from IPython.display import Image, display
from src.config import FIGURES_DIR

imp = pd.read_csv(TABLES_DIR / "shap_importance_top30.csv")
print("Top 15 features by mean |SHAP| (XGBoost / TreeSHAP):")
display(imp.head(15).round(4))

for name in ["shap_importance.png", "shap_beeswarm.png",
             "shap_dependence.png", "shap_case_studies.png"]:
    p = FIGURES_DIR / name
    if p.exists():
        display(Image(str(p), width=800))
"""),
    code("""# Save the trained champion model as a shareable artefact (optional handover extra)
import joblib
import numpy as np
from src.config import MODELS_DIR, RANDOM_SEED
from src.models.factory import make_model
from src.explainability.shap_engine import load_training_matrix

X, y, ids, feature_names = load_training_matrix()
spw = float(np.sum(y == 0)) / max(float(np.sum(y == 1)), 1.0)
champion = make_model("xgboost", "baseline", spw, RANDOM_SEED)
champion.fit(X, y)
joblib.dump({"model": champion, "feature_names": feature_names},
            MODELS_DIR / "champion_xgboost.joblib")
print(f"Champion XGBoost saved: {MODELS_DIR / 'champion_xgboost.joblib'}")
"""),
    code("""# ---- Phase 4 verification against dissertation Chapter 5 ----
imp = pd.read_csv(TABLES_DIR / "shap_importance_top30.csv")
assert imp.iloc[0]['feature'] == 'EXT_SOURCES_MEAN'
assert abs(imp.iloc[0]['mean_abs_shap'] - 0.438) < 0.02

cases = pd.read_csv(TABLES_DIR / "shap_case_studies.csv")
assert cases['applicant_id'].unique().tolist() == [187933, 237456, 334854]

with open(TABLES_DIR / "shap_tree_vs_linear_overlap.txt") as f:
    assert "16/30" in f.read()
print("PHASE 4 CHECKPOINT OK: EXT_SOURCES_MEAN top (0.438), cases 187933/237456/334854, overlap 16/30")
"""),
])

# ================================================================ Phase 05 — Collect
nb05 = notebook([
    md("""# Phase 5 — Collect & Download Results

**MSc Credit Risk Project — Colab handover (5 of 6)**

Zips all tables, figures and logs into one file and offers it for download. Use this
to bring the Colab-reproduced results back to your local machine for comparison with
the original artefact.
"""),
    code(SETUP_CELL),
    code("""import shutil
from src.config import PROJECT_ROOT as _PR

zip_base = _PR / "colab_reproduction_results"
shutil.make_archive(str(zip_base), "zip", root_dir=_PR, base_dir="reports")
print(f"Zipped: {zip_base}.zip ({zip_base.with_suffix('.zip').stat().st_size / 1e6:.1f} MB)")

from google.colab import files
files.download(str(zip_base) + ".zip")
"""),
    md("""## Done

All six phases reproduced the dissertation numbers end-to-end — from exploratory
data analysis through cross-validation, out-of-time testing, statistical testing,
and SHAP explainability.

The full result set lives in `reports/` and `outputs/eda/` inside the project folder
on Drive, and the zip you just downloaded contains the same content for local comparison.
"""),
])

# ================================================================ Master run-all
def strip_setup(nb):
    """Return a phase notebook's cells minus its standalone Drive-setup cell."""
    out = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "".join(cell["source"]).strip() == SETUP_CELL.strip():
            continue
        out.append(cell)
    return out


master_cells = [
    md("""# FULL EXPERIMENT — Data Loading → EDA → Modelling → Statistics → SHAP

**MSc Credit Risk Project — single-notebook Colab reproduction**

This one notebook runs the **entire experiment end-to-end**, exactly as it was done
locally. Just open it in Google Colab and choose **Runtime > Run all**.

The pipeline:
1. **Setup** — mount Google Drive, locate the project folder, install dependencies
2. **Phase 0 — EDA** — full exploratory analysis (imbalance, temporal drift, missing
   values, sentinel anomaly, correlations, multicollinearity) with inline figures
3. **Phase 1 — Data prep** — cleaning, feature engineering, temporal split
4. **Phase 2 — Cross-validation** — 3x5 repeated CV over all 9 configs (135 fits)
5. **Phase 3 — OOT + statistics** — out-of-time 2020 test, 16 statistical tests,
   evaluation figures
6. **Phase 4 — SHAP** — TreeSHAP + LinearSHAP, case studies, champion model
7. **Phase 5 — Collect** — zip all results and download

**Total runtime: ~1 hour on Colab CPU.** Every phase checks for its checkpoint on
Drive, so if the runtime disconnects you can simply re-run this notebook and it picks
up where it left off. Each phase ends with an assertion cell that verifies the numbers
against the dissertation.

> Prefer to run one phase at a time? Use the individual `00_eda.ipynb` …
> `05_collect_results.ipynb` notebooks instead — they contain the same cells.
"""),
    code(SETUP_CELL),
]
for _nb in (nb00, nb01, nb02, nb03, nb04, nb05):
    master_cells.extend(strip_setup(_nb))

nb_master = notebook(master_cells)

# ================================================================ Write everything
FILES = {
    "RUN_ALL.ipynb": nb_master,
    "00_eda.ipynb": nb00,
    "01_setup_and_data.ipynb": nb01,
    "02_cross_validation.ipynb": nb02,
    "03_oot_and_statistics.ipynb": nb03,
    "04_shap.ipynb": nb04,
    "05_collect_results.ipynb": nb05,
}

# Remove any stale notebook from the previous 5-notebook layout
# (FROM_SCRATCH.ipynb is generated separately by build_from_scratch.py - keep it)
for stale in COLAB.glob("*.ipynb"):
    if stale.name not in FILES and stale.name != "FROM_SCRATCH.ipynb":
        stale.unlink()

for name, nb in FILES.items():
    path = COLAB / name
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_cells = len(nb["cells"])
    print(f"Wrote {path.name} ({n_cells} cells)")

README = """# Colab Handover — Loan Default Prediction (MSc Project)

Reproduces the **entire experiment from data loading to EDA to final SHAP analysis**
on Google Colab, with checkpoints so a disconnected free-tier runtime never loses
progress.

## One-time setup

1. Upload the **entire project folder** to Google Drive (e.g. `MyDrive/credit-risk-project`).
   The raw dataset `data/raw/application_train.csv` (~152 MB) is already inside it.
2. Open the notebooks in Google Colab (File > Open > Drive tab).
3. Runtime type: **CPU** is sufficient (Runtime > Change runtime type).

## Option A — FROM_SCRATCH.ipynb (recommended: independent reproduction)

Open **`FROM_SCRATCH.ipynb`** and choose **Runtime > Run all**. This is the
**fully independent** reproduction: its second cell deletes every result artifact
from any earlier run, then recomputes **every phase live** — EDA, cleaning/split,
all 135 cross-validation fits, the out-of-time test, the statistical battery, and
SHAP. Nothing is loaded from a previous run. Each phase ends with an assertion
that verifies the freshly computed numbers against the local run's published
results. **~1 hour on Colab CPU.**

If the runtime disconnects, just re-run the notebook: the fresh-start wipe only
happens once per session (a marker file records it), so the phases resume from
their checkpoints instead of restarting. To force a brand-new wipe, set
`FORCE = True` in the fresh-start cell.

## Option B — RUN_ALL.ipynb (checkpoint-aware)

Open **`RUN_ALL.ipynb`** and choose **Runtime > Run all**. Same pipeline, but each
phase checks for its checkpoint first and skips/resumes rather than wiping —
useful if you want to re-run without recomputing finished phases.

## Option C — Phase by phase

Run the six notebooks below **in order** if you prefer to step through the
experiment one stage at a time (same cells as RUN_ALL.ipynb):

| # | Notebook | What it does | Approx. time | Checkpoint |
|---|----------|--------------|--------------|------------|
| 0 | `00_eda.ipynb` | Full exploratory data analysis (11 sections, 6 figures) | ~5 min | `outputs/eda/*` |
| 1 | `01_setup_and_data.ipynb` | Install deps, clean + engineer features, temporal split | ~5 min | `data/processed/*.parquet` |
| 2 | `02_cross_validation.ipynb` | 3x5 repeated CV, all 9 configs (135 fits) | ~30-45 min | `reports/tables/cv_fold_results.csv` (per repeat) |
| 3 | `03_oot_and_statistics.ipynb` | OOT 2020 test, 16 statistical tests, evaluation figures | ~10 min | `reports/tables/oot_results.csv` |
| 4 | `04_shap.ipynb` | TreeSHAP + LinearSHAP, case studies, champion model save | ~5 min | `reports/tables/shap_*.csv`, `models/champion_xgboost.joblib` |
| 5 | `05_collect_results.ipynb` | Zip `reports/` and download | ~1 min | `colab_reproduction_results.zip` |

## If the runtime disconnects

Just re-open the notebook and run it again. Every phase detects its checkpoint
on Drive and either skips (complete) or resumes (phase 2 resumes from the next
repeat). Seeds are tied to the global repeat number, so a resumed run is
bit-identical to an uninterrupted one.

## Verification

Each phase ends with an assertion cell that checks its output against the
dissertation's published numbers (e.g. champion OOT ROC-AUC 0.7669, PR-AUC
0.2513, cost-sensitive recall 0.657, 15/16 comparisons significant, SHAP top
feature EXT_SOURCES_MEAN at 0.438). If all assertions pass, the reproduction
is confirmed.

## Notes

- The notebooks import the SAME `src/` package and `run_experiment.py` as the
  local artefact — no rewritten copies, so results cannot diverge.
- All outputs are written inside the project folder on Drive, so they persist
  automatically across sessions.
- EDA, OOT/statistics and SHAP phases display their figures and tables inline,
  so you can see the results as they are produced.
- Phase 4 additionally saves the trained champion XGBoost as
  `models/champion_xgboost.joblib` for anyone who wants the model object
  without retraining.
"""

(COLAB / "README.md").write_text(README, encoding="utf-8")
print(f"Wrote colab/README.md")
