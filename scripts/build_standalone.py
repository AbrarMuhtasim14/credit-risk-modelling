"""Build STANDALONE.ipynb - ONE fully self-contained Colab notebook.

Design (per user requirement):
- Shareable as a SINGLE .ipynb file. No project folder, no src/ package, no
  run_experiment.py, no prepopulated results - everything is inlined.
- Dataset comes from the user's own Colab session (left-panel upload or Kaggle
  download via kagglehub).
- EVERY phase computes live: EDA -> cleaning/split -> 135-fit CV -> OOT ->
  16 statistical tests -> evaluation plots -> SHAP. Nothing is ever read back
  from disk except the raw CSV the user provides and this session's own
  per-repeat CV checkpoint (resume-after-disconnect only).
- Logic is copied verbatim from the project's src/ modules (logger lines
  replaced with print), so the numbers reproduce the dissertation results
  (same seeds, same frozen hyperparameters).
- Test hooks: env vars MCR_N_REPEATS / MCR_N_FOLDS / MCR_SHAP_SAMPLE allow a
  reduced smoke run without editing cells.
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "STANDALONE.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

# ============================================================================
# TITLE
# ============================================================================
cells.append(md("""# Loan Default Prediction - FULL EXPERIMENT FROM SCRATCH
### MSc Data Science dissertation reproduction - single self-contained notebook

**This file is the entire experiment.** It needs no project folder, no `src/`
package, and no pre-computed results. Every number and figure below is computed
live in this session, with the same code logic and the same seeds as the
original local run, so the results should match the dissertation:

| Phase | What it computes | ~Time |
|---|---|---|
| 0 | Setup, dependencies, dataset acquisition | 2 min |
| 1 | Exploratory data analysis (11 sections) | 5 min |
| 2 | Cleaning, feature engineering, temporal split, preprocessor | 5 min |
| 3 | Cross-validation: 3 repeats x 5 folds x 9 configs = **135 model fits** | 30-45 min |
| 4 | Out-of-time test (train 2018-19 -> test 2020) | 10 min |
| 5 | Statistical significance battery (16 paired comparisons) | 1 min |
| 6 | Evaluation figures (ROC / PR / confusion / cost) | 1 min |
| 7 | SHAP explainability (TreeSHAP + LinearSHAP + case studies) | 5-10 min |
| 8 | Verification against the dissertation numbers + results download | 1 min |

**Total: ~1 hour on Colab CPU.**

### What you need to provide
The Home Credit dataset file **`application_train.csv`** (~152 MB), either:
- **Option A:** upload it with Colab's left panel (folder icon > *Upload to
  session storage*) before running cell 0.2, or
- **Option B:** let cell 0.2 download it from Kaggle (needs Kaggle API
  credentials uploaded as `kaggle.json`, and the competition rules accepted on
  kaggle.com first).

Optional: `HomeCredit_columns_description.csv` (adds column descriptions to the
EDA printouts; everything works without it).

### If the runtime disconnects
Re-run from the top. Cells 0-2 take ~10 minutes. Phase 3 checkpoints itself to
the session disk after every repeat, so a disconnect loses at most one repeat
of cross-validation - and that checkpoint is written by THIS session, never
loaded from any pre-existing result.

### Quick smoke test (optional)
To verify the notebook runs end-to-end in ~15 minutes before the full run, set
in cell 0.3: `N_REPEATS = 1`, `N_FOLDS = 2`, `SHAP_SAMPLE = 500`. The
verification asserts in phase 8 are skipped automatically in that mode.
"""))

# ============================================================================
# 0. SETUP
# ============================================================================
cells.append(md("## 0. Setup"))

cells.append(code("""# 0.1 Install pinned dependencies (version parity with the original local run)
!pip install -q "xgboost>=3.4,<3.5" "imbalanced-learn>=0.14,<0.15" "shap>=0.52,<0.53" "pyarrow>=25" kagglehub

import warnings
warnings.filterwarnings("ignore")

import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
print("Setup done.")
"""))

cells.append(code("""# 0.2 Locate or download the dataset
RAW = Path("application_train.csv")
if not RAW.exists() and Path("/content/application_train.csv").exists():
    RAW = Path("/content/application_train.csv")

if not RAW.exists():
    print("application_train.csv not found in the session - trying Kaggle...")
    try:
        import kagglehub
        import zipfile
        d = kagglehub.competition_download("home-credit-default-risk")
        with zipfile.ZipFile(d) as z:
            z.extract("application_train.csv")
        RAW = Path("application_train.csv")
        print("Downloaded from Kaggle.")
    except Exception as e:
        raise RuntimeError(
            "Dataset not found. Upload application_train.csv via the left panel "
            "(Upload to session storage), or upload Kaggle credentials (kaggle.json) "
            "after accepting the competition rules. Source: "
            "https://www.kaggle.com/competitions/home-credit-default-risk/data"
        ) from e

print(f"Dataset: {RAW} ({RAW.stat().st_size / 1e6:.0f} MB)")

desc_map = {}
DESC = Path("HomeCredit_columns_description.csv")
if DESC.exists():
    desc = pd.read_csv(DESC, encoding="latin-1")
    desc_map = dict(zip(desc["Row"], desc["Description"]))
    print(f"Column descriptions loaded ({len(desc_map)} entries).")
else:
    print("No column-description file found (optional) - continuing without it.")

RESULTS = Path("from_scratch_results")
RESULTS.mkdir(exist_ok=True)
(RESULTS / "figures").mkdir(exist_ok=True)
print(f"All computed artifacts will be saved under ./{RESULTS}/")
"""))

cells.append(code("""# 0.3 Experiment constants (identical to the original run's configuration)
RANDOM_SEED = 42
N_REPEATS = int(os.environ.get("MCR_N_REPEATS", "3"))   # full run: 3
N_FOLDS = int(os.environ.get("MCR_N_FOLDS", "5"))       # full run: 5
SHAP_SAMPLE = int(os.environ.get("MCR_SHAP_SAMPLE", "2000"))

TRAIN_YEARS = [2018, 2019]
TEST_YEARS = [2020]
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
TIME_COL = "application_date"

DAYS_EMPLOYED_SENTINEL = 365243          # "1000 years" placeholder -> pensioner/unemployed
DAYS_EMPLOYED_ANOM_COL = "DAYS_EMPLOYED_ANOM"
INCOME_WINSOR_QUANTILE = 0.999           # cap fitted on training years ONLY (leakage control)

DEFAULT_COST_RATIO = 10.0   # a missed default (FN) costs 10x a false alarm (FP)
FIXED_THRESHOLD = 0.5
SMOTE_K_NEIGHBORS = 5
FIGURE_DPI = 300

# 30 redundant columns identified by the EDA (near-duplicate pairs, |r| > 0.9):
# 28 building _MEDI/_MODE variants + 2 social-circle 60-DPD variants.
BUILDING_DROP = [
    "APARTMENTS_MEDI", "APARTMENTS_MODE",
    "BASEMENTAREA_MEDI", "BASEMENTAREA_MODE",
    "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_MEDI", "YEARS_BUILD_MODE",
    "COMMONAREA_MEDI", "COMMONAREA_MODE",
    "ELEVATORS_MEDI", "ELEVATORS_MODE",
    "ENTRANCES_MEDI", "ENTRANCES_MODE",
    "FLOORSMAX_MEDI", "FLOORSMAX_MODE",
    "FLOORSMIN_MEDI", "FLOORSMIN_MODE",
    "LANDAREA_MEDI", "LANDAREA_MODE",
    "LIVINGAPARTMENTS_MEDI", "LIVINGAPARTMENTS_MODE",
    "LIVINGAREA_MEDI", "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_MEDI", "NONLIVINGAREA_MODE",
]
SOCIAL_CIRCLE_DROP = ["OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"]
COLS_TO_DROP = BUILDING_DROP + SOCIAL_CIRCLE_DROP
EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

FULL_RUN = (N_REPEATS == 3 and N_FOLDS == 5 and SHAP_SAMPLE == 2000)
print(f"Config: {N_REPEATS}x{N_FOLDS} repeated CV, seed {RANDOM_SEED}, SHAP sample {SHAP_SAMPLE}")
print(f"Full-run verification mode: {FULL_RUN}")
"""))


# ============================================================================
# 1. EDA
# ============================================================================
cells.append(md("""## 1. Exploratory Data Analysis

The full 11-section EDA. Every figure below is drawn live from the raw CSV.
This is the investigative work that motivated every design decision in the
experiment."""))

cells.append(code("""# 1.1 Dataset overview
print("=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)
df = pd.read_csv(RAW, low_memory=False)

print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
print("\\nDtype counts:")
print(df.dtypes.value_counts().to_string())

cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
flag_cols = [c for c in df.columns if c.startswith("FLAG_")]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ("SK_ID_CURR", "TARGET") and c not in flag_cols]
print(f"\\nCategorical: {len(cat_cols)} | Binary flags: {len(flag_cols)} | Numeric: {len(num_cols)}")
df.head()
"""))

cells.append(code("""# 1.2 Duplicates
print("=" * 70)
print("2. DUPLICATES")
print("=" * 70)
dup_full = df.duplicated().sum()
dup_id = df.duplicated(subset=["SK_ID_CURR"]).sum()
print(f"Exact duplicate rows: {dup_full:,}")
print(f"Duplicate SK_ID_CURR: {dup_id:,}")
"""))

cells.append(code("""# 1.3 Target variable / class imbalance
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
plt.tight_layout(); plt.savefig(RESULTS / "figures" / "01_target_distribution.png", dpi=150); plt.show()
"""))

cells.append(code("""# 1.4 Temporal structure - WHY the experiment uses a time-based split
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
plt.tight_layout(); plt.savefig(RESULTS / "figures" / "02_application_date_trend.png", dpi=150); plt.show()
"""))

cells.append(md("""**Why this matters:** applications span 2015-2020 and the default rate *drifts*
over time. A random train/test split would leak future information into the past.
This is the direct evidence for using a **temporal out-of-time split**
(fit 2018-2019, validate on 2020) rather than a random split."""))

cells.append(code("""# 1.5 Missing values
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
miss.to_frame("missing_pct").to_csv(RESULTS / "missing_by_column.csv")

fig, ax = plt.subplots(figsize=(12, 5))
top30 = miss.head(30) * 100
ax.bar(range(len(top30)), top30.values, color="#dd8452")
ax.set_xticks(range(len(top30)))
ax.set_xticklabels(top30.index, rotation=80, fontsize=7)
ax.set_ylabel("Missing %")
ax.set_title("Top 30 columns by missing values")
ax.axhline(50, color="red", ls="--", lw=1)
plt.tight_layout(); plt.savefig(RESULTS / "figures" / "03_missing_top30.png", dpi=150); plt.show()

miss_by_t = df.groupby("TARGET").apply(lambda g: g.isna().mean().mean(), include_groups=False)
print(f"\\nAvg cell missingness - default=1: {miss_by_t.get(1, np.nan):.4f} | default=0: {miss_by_t.get(0, np.nan):.4f}")
"""))

cells.append(code("""# 1.6 Known anomalies & data quality
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
"""))

cells.append(md("""**Key finding:** `DAYS_EMPLOYED = 365243` is a sentinel ("1000 years") marking
unemployed/retired applicants, present in ~18% of rows. Left uncorrected it acts as a
massive outlier and creates a spurious -0.9998 correlation with `FLAG_EMP_PHONE`.
The experiment corrects this sentinel and winsorises income at the 99.9th percentile."""))

cells.append(code("""# 1.7 Key numeric feature distributions
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
plt.tight_layout(); plt.savefig(RESULTS / "figures" / "04_numeric_distributions.png", dpi=150); plt.show()

print("\\nEXT_SOURCE columns (external risk scores):")
for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
    m = df[c].isna().mean()
    corr = df[[c, "TARGET"]].corr().iloc[0, 1]
    print(f"  {c}: missing {m:.2%}, corr with TARGET = {corr:.3f}")
"""))

cells.append(code("""# 1.8 Categorical features & default rate by category
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
plt.tight_layout(); plt.savefig(RESULTS / "figures" / "05_default_rate_by_category.png", dpi=150); plt.show()
"""))

cells.append(code("""# 1.9 Correlation with the target
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
top20.to_csv(RESULTS / "correlation_with_target.csv")

fig, ax = plt.subplots(figsize=(8, 7))
t = corr_target.head(20)
ax.barh(range(len(t))[::-1], t.values, color="#4c72b0")
ax.set_yticks(range(len(t))[::-1])
ax.set_yticklabels(t.index, fontsize=8)
ax.set_xlabel("|Pearson corr| with TARGET")
ax.set_title("Top 20 features correlated with default")
plt.tight_layout(); plt.savefig(RESULTS / "figures" / "06_correlation_top20.png", dpi=150); plt.show()
"""))

cells.append(code("""# 1.10 Multicollinearity (high inter-feature correlation)
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
pd.DataFrame(pairs, columns=["feat1", "feat2", "corr"]).to_csv(RESULTS / "high_corr_pairs.csv", index=False)
"""))

cells.append(md("""**Key finding:** the building features come in `_AVG`/`_MEDI`/`_MODE` triplets that are
near-duplicates (|r| > 0.97), and `OBS_60`/`DEF_60` social-circle columns duplicate the
30-DPD variants. This is the evidence for **dropping 30 redundant columns** in the
experiment's feature-selection step."""))

cells.append(code("""# 1.11 Potential leakage candidates
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
print("EDA COMPLETE")
print("=" * 70)
"""))

cells.append(md("""## EDA summary - decisions this analysis drove

| EDA finding | Experiment decision |
|---|---|
| Default rate drifts over time | Temporal OOT split (fit 2018-19, test 2020) |
| 8.1% default rate (heavy imbalance) | PR-AUC as primary metric; SMOTE & cost-sensitive strategies |
| `DAYS_EMPLOYED=365243` sentinel (~18%) | Sentinel correction before modelling |
| Extreme income outliers | Winsorise income at 99.9th pct (fit on train years only) |
| 30 near-duplicate column pairs | Drop 30 redundant columns |
| High missingness in EXT_SOURCE | Per-fold imputation, no global statistics |"""))


# ============================================================================
# 2. CLEANING, FEATURE ENGINEERING, TEMPORAL SPLIT, PREPROCESSOR
# ============================================================================
cells.append(md("""## 2. Cleaning, Feature Engineering & Temporal Split

The exact cleaning/engineering pipeline from the original run:

1. `DAYS_EMPLOYED = 365243` sentinel -> NaN + binary anomaly flag
2. Income winsorisation at the 99.9th percentile, **fitted on training years only** (leakage control)
3. `ORGANIZATION_TYPE` grouped 58 -> 11 sectors
4. `XNA` tokens -> NaN
5. Drop the 30 redundant columns found in the EDA
6. Engineer domain ratios and EXT_SOURCE aggregates
7. Temporal split: train = 2018-2019, test = 2020
8. Build the leakage-free preprocessor (median impute + RobustScaler for numeric;
   constant impute + one-hot for categorical), fitted on the training window only"""))

cells.append(code("""# 2.1 Domain feature engineering (pure function - no global fitting)

ORGANIZATION_SECTOR_RULES = [
    ("Business Entity", ["BUSINESS ENTITY"]),
    ("Industry", ["INDUSTRY"]),
    ("Trade", ["TRADE"]),
    ("Transport", ["TRANSPORT"]),
    ("Government/Public", ["GOVERNMENT", "POLICE", "MILITARY", "SECURITY", "LEGAL SERVICES"]),
    ("Education/Healthcare", ["MEDICINE", "KINDERGARTEN", "SCHOOL", "UNIVERSITY"]),
    ("Financial/RealEstate", ["BANK", "INSURANCE", "REALTOR"]),
    ("Services/Hospitality", ["HOTEL", "RESTAURANT", "SERVICES", "CLEANING", "POSTAL"]),
    ("Infrastructure/Agriculture", ["CONSTRUCTION", "HOUSING", "AGRICULTURE", "ELECTRICITY", "TELECOM"]),
]


def group_organization_type(value) -> str:
    \"\"\"Map one of 58 raw ORGANIZATION_TYPE categories into 11 sectors.\"\"\"
    if pd.isna(value):
        return "Missing"
    text = str(value).strip().upper()
    if text in ("", "XNA", "NAN", "NONE"):
        return "Missing"
    for sector, keywords in ORGANIZATION_SECTOR_RULES:
        if any(k in text for k in keywords):
            return sector
    return "Other"


def clean_and_engineer(df_in: pd.DataFrame, income_cap: float) -> pd.DataFrame:
    \"\"\"Clean anomalies and engineer domain features.

    income_cap: 99.9th percentile of AMT_INCOME_TOTAL computed on the
    TRAINING YEARS ONLY by the caller (leakage control).
    \"\"\"
    df = df_in.copy()

    # -- 1. DAYS_EMPLOYED sentinel correction -------------------------------
    sentinel_mask = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    df[DAYS_EMPLOYED_ANOM_COL] = sentinel_mask.astype(np.int8)
    df.loc[sentinel_mask, "DAYS_EMPLOYED"] = np.nan
    print(f"DAYS_EMPLOYED sentinel corrected: {int(sentinel_mask.sum()):,} rows "
          f"({100.0 * sentinel_mask.mean():.2f}%)")

    # -- 2. Income winsorization (cap fitted on train years by caller) ------
    n_clipped = int((df["AMT_INCOME_TOTAL"] > income_cap).sum())
    df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=income_cap)
    print(f"Income winsorized at ${income_cap:,.0f}: {n_clipped:,} rows clipped")

    # -- 3. ORGANIZATION_TYPE grouping --------------------------------------
    df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].map(group_organization_type)

    # -- 4. XNA token cleanup ------------------------------------------------
    df["CODE_GENDER"] = df["CODE_GENDER"].replace({"XNA": np.nan})

    # -- 5. Drop redundant columns -------------------------------------------
    drop_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_present)
    print(f"Dropped {len(drop_present)} redundant columns")

    # -- 6. Domain feature engineering ---------------------------------------
    eps = 1e-6

    # Financial burden ratios
    df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + eps)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + eps)
    df["CREDIT_GOODS_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"].fillna(1) + eps)

    # Demographics / tenure
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + eps)
    df["PHONE_TO_BIRTH_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / (df["DAYS_BIRTH"] + eps)
    df["REGISTRATION_TO_BIRTH_RATIO"] = df["DAYS_REGISTRATION"] / (df["DAYS_BIRTH"] + eps)
    df["ID_PUBLISH_TO_BIRTH_RATIO"] = df["DAYS_ID_PUBLISH"] / (df["DAYS_BIRTH"] + eps)
    df["CAR_TO_AGE_RATIO"] = df["OWN_CAR_AGE"] / (df["AGE_YEARS"] + eps)

    # External bureau score aggregates (nan-safe)
    ext_present = [c for c in EXT_SOURCE_COLS if c in df.columns]
    ext = df[ext_present].to_numpy(dtype=np.float64)
    valid_counts = np.sum(~np.isnan(ext), axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df["EXT_SOURCES_MEAN"] = np.nanmean(ext, axis=1)
        ext_std = np.nanstd(ext, axis=1)
        ext_std[valid_counts < 2] = np.nan
        df["EXT_SOURCES_STD"] = ext_std
        ext_min = np.nanmin(ext, axis=1)
        ext_max = np.nanmax(ext, axis=1)
        ext_min[valid_counts == 0] = np.nan
        ext_max[valid_counts == 0] = np.nan
        df["EXT_SOURCES_MIN"] = ext_min
        df["EXT_SOURCES_MAX"] = ext_max

    if len(ext_present) == 3:
        df["EXT_SOURCES_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        df["EXT_SOURCE_1_2_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
        df["EXT_SOURCE_2_3_PROD"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        df["EXT_SOURCE_1_3_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]

    # Document flag aggregate
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1).astype(np.int8)

    print(f"Feature engineering complete: {df.shape[1]} columns")
    return df

print("clean_and_engineer() defined.")
"""))

cells.append(code("""# 2.2 Load raw data, fit the income cap on TRAINING YEARS ONLY, engineer, split
t0 = time.time()
df_clean = pd.read_csv(RAW, low_memory=False)
print(f"Raw shape: {df_clean.shape}")

df_clean[TIME_COL] = pd.to_datetime(df_clean[TIME_COL])
df_clean["app_year"] = df_clean[TIME_COL].dt.year

# Leakage control: income cap fitted on TRAINING YEARS ONLY
train_mask = df_clean["app_year"].isin(TRAIN_YEARS)
income_cap = float(df_clean.loc[train_mask, "AMT_INCOME_TOTAL"].quantile(INCOME_WINSOR_QUANTILE))
print(f"Income cap (99.9th pct, train years only): ${income_cap:,.0f}")

df_eng = clean_and_engineer(df_clean, income_cap=income_cap)
del df_clean; gc.collect()

df_train = df_eng[df_eng["app_year"].isin(TRAIN_YEARS)].copy()
df_test = df_eng[df_eng["app_year"].isin(TEST_YEARS)].copy()
del df_eng; gc.collect()

print(f"\\nTemporal split: train 2018-2019 = {len(df_train):,} rows "
      f"({100 * df_train[TARGET_COL].mean():.2f}% default), "
      f"test 2020 = {len(df_test):,} rows ({100 * df_test[TARGET_COL].mean():.2f}% default)")
print(f"Split done in {time.time() - t0:.1f}s")
"""))

cells.append(code("""# 2.3 Leakage-free preprocessor (ColumnTransformer), fitted on training data only
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

META_COLS = [ID_COL, TARGET_COL, TIME_COL, "app_year"]


def split_column_types(df_in, exclude=None):
    \"\"\"Classify feature columns into numeric vs categorical.\"\"\"
    exclude = exclude or META_COLS
    feature_cols = [c for c in df_in.columns if c not in exclude]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_in[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric_cols, categorical_cols):
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


def fit_transform(df_tr):
    \"\"\"Fit a fresh preprocessor on df_tr and transform it.\"\"\"
    numeric_cols, categorical_cols = split_column_types(df_tr)
    pre = build_preprocessor(numeric_cols, categorical_cols)
    arr = pre.fit_transform(df_tr[numeric_cols + categorical_cols]).astype(np.float32)
    feature_names = list(pre.get_feature_names_out())
    print(f"Preprocessor fitted on {len(df_tr):,} rows: {len(numeric_cols)} numeric + "
          f"{len(categorical_cols)} categorical -> {len(feature_names)} features")
    return arr, feature_names, pre, numeric_cols, categorical_cols


def transform_only(df_in, pre, numeric_cols, categorical_cols):
    \"\"\"Apply an already-fitted preprocessor to new data (no refitting).\"\"\"
    return pre.transform(df_in[numeric_cols + categorical_cols]).astype(np.float32)

print("Preprocessor functions defined.")
"""))


# ============================================================================
# 3. CROSS-VALIDATION
# ============================================================================
cells.append(md("""## 3. Cross-Validation (the long phase)

**3x5 repeated stratified cross-validation** over all 9 configurations
(3 model families x 3 imbalance strategies = **135 model fits**):

| Family | Strategies |
|---|---|
| Logistic Regression | Baseline / SMOTE / Cost-Sensitive |
| Random Forest | Baseline / SMOTE / Cost-Sensitive |
| XGBoost | Baseline / SMOTE / Cost-Sensitive |

Hyperparameters are **frozen** across imbalance strategies so the comparison is
fair - the ONLY thing that changes between baseline / cost-sensitive / SMOTE is
the imbalance intervention itself. Primary metric: **PR-AUC** (accuracy is
misleading at an 8% base rate).

Leakage control inside every fold:
- the preprocessor is re-fit on the training fold only,
- SMOTE is applied to the training fold only (computed once per fold, shared by
  the three SMOTE cells),
- `scale_pos_weight` comes from the training fold only.

If the runtime disconnects, re-run from the top: after every repeat the fold
results are checkpointed to the session disk (written by THIS notebook), so at
most one repeat is lost."""))

cells.append(code("""# 3.1 Model factory: 3 architectures with FROZEN hyperparameters
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

# Frozen architecture hyperparameters (chosen conservatively, identical to the
# original run). sklearn 1.9: LR penalty defaults to l2; n_jobs has no effect on LR.
LR_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=2000)
RF_PARAMS = dict(n_estimators=200, max_depth=12, min_samples_leaf=20,
                 min_samples_split=40, max_features="sqrt")
XGB_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", eval_metric="logloss",
)


def make_model(model_name: str, strategy: str, scale_pos_weight: float, seed: int):
    \"\"\"Build a model for one cell of the 9-cell matrix.\"\"\"
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

print("make_model() defined - 3 families x 3 strategies, frozen hyperparameters.")
"""))

cells.append(code("""# 3.2 Evaluation metrics (accuracy deliberately NOT a headline metric)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def pr_auc_exact(y_true, y_prob) -> float:
    \"\"\"True area under the precision-recall curve via trapezoidal integration.

    sklearn's precision_recall_curve returns recall in DECREASING order, so the
    raw trapezoid integral is negative; we take the absolute value.
    \"\"\"
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(abs(np.trapezoid(precision, recall)))


def compute_metrics(y_true, y_prob, threshold: float = 0.5, fn_cost: float = 10.0) -> dict:
    \"\"\"Full metric battery for one set of predictions.\"\"\"
    y_pred = (y_prob >= threshold).astype(np.int8)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = pr_auc_exact(y_true, y_prob)
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))

    # Expected monetary cost per applicant:
    # false negative (missed default) costs fn_cost units, false positive costs 1.
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    cost_per_applicant = float((fn * fn_cost + fp) / max(len(y_true), 1))

    # Best achievable F1 over all thresholds (threshold-free reference)
    best_f1 = 0.0
    best_thr = 0.5
    prec_grid, rec_grid, thr_grid = precision_recall_curve(y_true, y_prob)
    for p, r, t in zip(prec_grid[:-1], rec_grid[:-1], thr_grid):
        if p + r > 0:
            f = 2 * p * r / (p + r)
            if f > best_f1:
                best_f1, best_thr = f, float(t)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "f2": f2,
        "cost_per_applicant": cost_per_applicant,
        "best_f1": best_f1,
        "best_f1_threshold": best_thr,
        "n_false_negatives": fn,
        "n_false_positives": fp,
    }

print("compute_metrics() defined.")
"""))

cells.append(code("""# 3.3 Repeated stratified CV on the 2018-2019 window (in-time track).
# Preprocessor re-fit per fold; SMOTE inside fold only; 2020 never touched.
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

CV_CHECKPOINT = RESULTS / "cv_fold_results_partial.csv"  # written by THIS session only


def run_cv_track(df_tr_window, n_repeats=N_REPEATS, n_folds=N_FOLDS, repeat_offset=0):
    y_all = df_tr_window[TARGET_COL].to_numpy(dtype=np.int8)
    n = len(df_tr_window)
    fold_records = []

    for repeat in range(1, n_repeats + 1):
        if repeat <= repeat_offset:
            print(f"Repeat {repeat} skipped (already checkpointed by this session)")
            continue
        seed = RANDOM_SEED + (repeat - 1) * 1000
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold_idx, (tr_pos, va_pos) in enumerate(skf.split(np.zeros(n), y_all), start=1):
            t_fold = time.time()
            df_tr = df_tr_window.iloc[tr_pos]
            df_va = df_tr_window.iloc[va_pos]
            y_tr = y_all[tr_pos]
            y_va = y_all[va_pos]

            # --- Nested preprocessing: fit on train fold only ---------------
            X_tr, feat_names, pre, num_cols, cat_cols = fit_transform(df_tr)
            X_va = transform_only(df_va, pre, num_cols, cat_cols)

            # scale_pos_weight from this fold's training portion only
            spw = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)

            # --- SMOTE once per fold (shared by the 3 smote cells) ----------
            smote = SMOTE(sampling_strategy="auto", k_neighbors=SMOTE_K_NEIGHBORS,
                          random_state=seed)
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
            print(f"Repeat {repeat} Fold {fold_idx}: train={len(y_tr):,} "
                  f"(SMOTE->{len(y_tr_sm):,}), val={len(y_va):,}, {len(feat_names)} features")

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
            print(f"  Repeat {repeat} Fold {fold_idx} done in {time.time() - t_fold:.1f}s")

        pd.DataFrame(fold_records).to_csv(CV_CHECKPOINT, index=False)
        print(f"Checkpoint after repeat {repeat}: {len(fold_records)} fold records on disk\\n")

    return pd.DataFrame(fold_records)


t_cv = time.time()
if CV_CHECKPOINT.exists():
    prior = pd.read_csv(CV_CHECKPOINT)
    done = int(prior["repeat"].max())
    if done >= N_REPEATS:
        print(f"This session already finished all {N_REPEATS} repeats - reusing its checkpoint.")
        df_cv = prior
    else:
        print(f"Resuming this session's CV from repeat {done + 1}...")
        df_new = run_cv_track(df_train, repeat_offset=done)
        df_cv = pd.concat([prior, df_new], ignore_index=True)
        df_cv.to_csv(CV_CHECKPOINT, index=False)
else:
    print(f"Running full CV track: {N_REPEATS} repeats x {N_FOLDS} folds x 9 configs "
          f"= {N_REPEATS * N_FOLDS * 9} model fits...")
    df_cv = run_cv_track(df_train)

df_cv.to_csv(RESULTS / "cv_fold_results.csv", index=False)
print(f"\\nCV track complete: {len(df_cv)} fold records "
      f"(expected {N_REPEATS * N_FOLDS * 9}) in {(time.time() - t_cv) / 60:.1f} min.")
"""))

cells.append(code("""# 3.4 Aggregated CV results (Table 4.1 of the dissertation)
agg = (df_cv.groupby("display_name")
       .agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
            recall_mean=("recall", "mean"), precision_mean=("precision", "mean"),
            f1_mean=("f1", "mean"), f2_mean=("f2", "mean"),
            cost_mean=("cost_per_applicant", "mean"))
       .reset_index()
       .sort_values("pr_auc_mean", ascending=False))
agg.to_csv(RESULTS / "cv_summary.csv", index=False)
print("Aggregated CV results (sorted by mean PR-AUC):")
agg.round(4)
"""))

cells.append(code("""# 3.5 CV verification against dissertation Table 4.1
if FULL_RUN:
    xgb_base = agg.set_index("display_name").loc["XGBoost (Baseline)"]
    assert abs(xgb_base["pr_auc_mean"] - 0.2502) < 0.005, f"PR-AUC {xgb_base['pr_auc_mean']:.4f} != ~0.2502"
    assert abs(xgb_base["roc_auc_mean"] - 0.7622) < 0.005
    assert len(df_cv) == 135
    print("CV CHECKPOINT OK: champion XGBoost (Baseline) PR-AUC ~0.2502, ROC-AUC ~0.7622")
    print("Note: SMOTE rows should show LOWER PR-AUC than their baselines - SMOTE is harmful here.")
else:
    print("Reduced smoke run - skipping dissertation-number assertions.")
"""))


# ============================================================================
# 4. OUT-OF-TIME TEST
# ============================================================================
cells.append(md("""## 4. Out-of-Time Test (train 2018-19 -> test 2020)

Trains the 9 configurations on the **full** 2018-2019 window and evaluates once
on the untouched 2020 cohort - the honest generalisation benchmark. The
preprocessor is fit once on the full training window; the 2020 rows are never
seen during any fitting step."""))

cells.append(code("""# 4.1 OOT track: fit on full 2018-2019, evaluate once on 2020
import json

t_oot = time.time()
y_train = df_train[TARGET_COL].to_numpy(dtype=np.int8)
y_test = df_test[TARGET_COL].to_numpy(dtype=np.int8)

X_train, feat_names, pre_oot, num_cols, cat_cols = fit_transform(df_train)
X_test = transform_only(df_test, pre_oot, num_cols, cat_cols)
spw = float(np.sum(y_train == 0)) / max(float(np.sum(y_train == 1)), 1.0)

smote = SMOTE(sampling_strategy="auto", k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"OOT: train={len(y_train):,} (SMOTE->{len(y_train_sm):,}), test={len(y_test):,}")

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
        print(f"OOT {label:<45} ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
              f"({time.time() - t0:.1f}s)")

df_oot = pd.DataFrame(records)
df_oot.to_csv(RESULTS / "oot_results.csv", index=False)
print(f"\\nOOT track complete in {(time.time() - t_oot) / 60:.1f} min.")
"""))

cells.append(code("""# 4.2 Display the OOT results (Table 4.2 of the dissertation)
print("Out-of-time (2020) results, sorted by PR-AUC:")
df_oot.sort_values("pr_auc", ascending=False).round(4)
"""))

cells.append(code("""# 4.3 OOT verification against dissertation Table 4.2
if FULL_RUN:
    champ = df_oot[df_oot["display_name"] == "XGBoost (Baseline)"].iloc[0]
    assert abs(champ["roc_auc"] - 0.7669) < 0.002, champ["roc_auc"]
    assert abs(champ["pr_auc"] - 0.2513) < 0.005, champ["pr_auc"]

    cs = df_oot[df_oot["display_name"] == "XGBoost (Cost-Sensitive)"].iloc[0]
    assert abs(cs["recall"] - 0.657) < 0.01, cs["recall"]
    print("OOT CHECKPOINT OK: ROC-AUC 0.7669, PR-AUC 0.2513, cost-sensitive recall 0.657")
else:
    print("Reduced smoke run - skipping dissertation-number assertions.")
"""))

# ============================================================================
# 5. STATISTICAL SIGNIFICANCE
# ============================================================================
cells.append(md("""## 5. Statistical Significance Battery

Paired comparisons over matched CV observations. Because we use 3x5 repeated
stratified K-fold, each configuration yields 15 observations (3 repeats x
5 folds), paired by (repeat, fold) key. Tests per comparison:

- Paired Student's t-test (parametric)
- Wilcoxon signed-rank (non-parametric)
- Cohen's d and Hedges' g effect sizes with 95% CI of the mean difference"""))

cells.append(code("""# 5.1 Paired comparison machinery + the 16-comparison battery

def paired_comparison(df_cv_in, config_a, config_b, metric, category):
    \"\"\"Compare two configurations on matched (repeat, fold) observations.\"\"\"
    df_a = df_cv_in[df_cv_in["display_name"] == config_a].sort_values(["repeat", "fold"])
    df_b = df_cv_in[df_cv_in["display_name"] == config_b].sort_values(["repeat", "fold"])

    if len(df_a) == 0 or len(df_b) == 0:
        raise ValueError(f"Config not found: {config_a} / {config_b}")

    merged = df_a.merge(
        df_b[["repeat", "fold", metric]],
        on=["repeat", "fold"],
        suffixes=("_a", "_b"),
    )
    vals_a = merged[f"{metric}_a"].to_numpy(dtype=np.float64)
    vals_b = merged[f"{metric}_b"].to_numpy(dtype=np.float64)
    diff = vals_a - vals_b
    n = len(diff)

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    t_stat, p_t = stats.ttest_rel(vals_a, vals_b)

    try:
        w_stat, p_w = stats.wilcoxon(vals_a, vals_b)
    except ValueError:
        w_stat, p_w = np.nan, 1.0

    cohen_d = mean_diff / std_diff if std_diff > 1e-12 else 0.0
    j = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0) if n > 2 else 1.0
    hedges_g = cohen_d * j

    t_crit = stats.t.ppf(0.975, df=n - 1)
    se = std_diff / np.sqrt(n)
    ci_lo = mean_diff - t_crit * se
    ci_hi = mean_diff + t_crit * se

    if abs(cohen_d) < 0.2:
        effect = "Negligible"
    elif abs(cohen_d) < 0.5:
        effect = "Small"
    elif abs(cohen_d) < 0.8:
        effect = "Medium"
    else:
        effect = "Large"

    return {
        "category": category,
        "comparison": f"{config_a} vs {config_b}",
        "metric": metric,
        "n_observations": n,
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": float(t_stat),
        "p_value_ttest": float(p_t),
        "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else np.nan,
        "p_value_wilcoxon": float(p_w),
        "cohen_d": cohen_d,
        "hedges_g": hedges_g,
        "ci95_lower": ci_lo,
        "ci95_upper": ci_hi,
        "significant_alpha_0.05": bool(p_t < 0.05),
        "significant_alpha_0.01": bool(p_t < 0.01),
        "effect_size_interpretation": effect,
    }


COMPARISONS = [
    # --- Model family comparisons (baseline vs baseline), ROC-AUC ---
    ("Model Family", "XGBoost (Baseline)", "Logistic Regression (Baseline)", "roc_auc"),
    ("Model Family", "XGBoost (Baseline)", "Random Forest (Baseline)", "roc_auc"),
    ("Model Family", "Logistic Regression (Baseline)", "Random Forest (Baseline)", "roc_auc"),
    # --- Model family comparisons on PR-AUC (primary metric) ---
    ("Model Family", "XGBoost (Baseline)", "Logistic Regression (Baseline)", "pr_auc"),
    ("Model Family", "XGBoost (Baseline)", "Random Forest (Baseline)", "pr_auc"),
    ("Model Family", "Logistic Regression (Baseline)", "Random Forest (Baseline)", "pr_auc"),
    # --- Imbalance strategy comparisons within each model family ---
    ("Imbalance Strategy", "XGBoost (Cost-Sensitive)", "XGBoost (Baseline)", "roc_auc"),
    ("Imbalance Strategy", "XGBoost (SMOTE)", "XGBoost (Baseline)", "roc_auc"),
    ("Imbalance Strategy", "Random Forest (Cost-Sensitive)", "Random Forest (Baseline)", "roc_auc"),
    ("Imbalance Strategy", "Random Forest (SMOTE)", "Random Forest (Baseline)", "roc_auc"),
    ("Imbalance Strategy", "Logistic Regression (Cost-Sensitive)", "Logistic Regression (Baseline)", "roc_auc"),
    ("Imbalance Strategy", "Logistic Regression (SMOTE)", "Logistic Regression (Baseline)", "roc_auc"),
    # --- Recall impact of cost-sensitive learning ---
    ("Recall Impact", "XGBoost (Cost-Sensitive)", "XGBoost (Baseline)", "recall"),
    ("Recall Impact", "Random Forest (Cost-Sensitive)", "Random Forest (Baseline)", "recall"),
    ("Recall Impact", "Logistic Regression (Cost-Sensitive)", "Logistic Regression (Baseline)", "recall"),
    # --- Financial cost impact ---
    ("Financial Cost", "XGBoost (Cost-Sensitive)", "XGBoost (Baseline)", "cost_per_applicant"),
]

records_stats = [paired_comparison(df_cv, a, b, m, cat) for cat, a, b, m in COMPARISONS]
df_stats = pd.DataFrame(records_stats)
df_stats.to_csv(RESULTS / "statistical_significance.csv", index=False)
print(f"Significant at alpha=0.05: {int(df_stats['significant_alpha_0.05'].sum())} / {len(df_stats)}")
df_stats.round(6)
"""))

cells.append(code("""# 5.2 Significance heatmap
df_plot = df_stats.copy()
df_plot["label"] = df_plot["comparison"] + "\\n[" + df_plot["metric"] + "]"

p_vals = df_plot["p_value_ttest"].to_numpy().reshape(-1, 1)
annot = []
for p, d in zip(p_vals.flatten(), df_plot["cohen_d"].to_numpy()):
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    annot.append(f"p={p:.2e}\\nd={d:+.2f} {sig}")

fig, ax = plt.subplots(figsize=(9, 11), dpi=FIGURE_DPI)
sns.heatmap(
    -np.log10(np.clip(p_vals, 1e-300, 1.0)),
    annot=np.array(annot).reshape(-1, 1),
    fmt="", cmap="YlOrRd",
    yticklabels=df_plot["label"].tolist(),
    xticklabels=["-log10(p) paired t-test"],
    cbar_kws={"label": "-log10(p-value)"},
    ax=ax,
)
ax.set_title(
    "Statistical Significance Matrix: Paired Tests over 3x5 Repeated CV (n=15)\\n"
    "*** p<0.001 | ** p<0.01 | * p<0.05 | n.s. not significant",
    fontsize=11, fontweight="bold", pad=12,
)
plt.tight_layout()
fig.savefig(RESULTS / "figures" / "statistical_significance_heatmap.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""# 5.3 Statistics verification
if FULL_RUN:
    assert int(df_stats["significant_alpha_0.05"].sum()) == 15, \\
        int(df_stats["significant_alpha_0.05"].sum())
    print("STATISTICS CHECKPOINT OK: 15/16 comparisons significant at alpha=0.05")
else:
    print("Reduced smoke run - skipping dissertation-number assertions.")
"""))


# ============================================================================
# 6. EVALUATION FIGURES
# ============================================================================
cells.append(md("""## 6. Evaluation Figures

ROC curves, precision-recall curves, confusion matrices and financial cost
curves, all computed from the out-of-time (2020) probabilities produced in
phase 4 - probabilities from models that never saw the 2020 rows."""))

cells.append(code("""# 6.1 Plot helpers (styled per model family / strategy)
from sklearn.metrics import confusion_matrix, roc_curve

STRATEGY_STYLE = {"baseline": "-", "cost_sensitive": "--", "smote": ":"}
MODEL_COLOR = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "XGBoost": "#d62728"}


def _style(label: str):
    color = "#1f77b4"
    for fam, c in MODEL_COLOR.items():
        if label.startswith(fam):
            color = c
            break
    ls = "-"
    if "Cost-Sensitive" in label:
        ls = "--"
    elif "SMOTE" in label:
        ls = ":"
    return color, ls


labels_oot = list(oot_probas.keys())
y_oot = y_test
"""))

cells.append(code("""# 6.2 ROC curves (OOT 2020)
from sklearn.metrics import auc

fig, ax = plt.subplots(figsize=(8, 7), dpi=FIGURE_DPI)
for label in labels_oot:
    fpr, tpr, _ = roc_curve(y_oot, oot_probas[label])
    a = auc(fpr, tpr)
    color, ls = _style(label)
    ax.plot(fpr, tpr, ls, color=color, linewidth=1.8, label=f"{label} (AUC={a:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("False Positive Rate", fontweight="bold")
ax.set_ylabel("True Positive Rate", fontweight="bold")
ax.set_title(f"ROC Curves - Out-Of-Time Test (2020, N={len(y_oot):,})", fontweight="bold")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "figures" / "roc_curves_oot.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""# 6.3 Precision-Recall curves (OOT 2020)
base_rate = float(np.mean(y_oot))

fig, ax = plt.subplots(figsize=(8, 7), dpi=FIGURE_DPI)
for label in labels_oot:
    precision, recall, _ = precision_recall_curve(y_oot, oot_probas[label])
    a = auc(recall, precision)
    color, ls = _style(label)
    ax.plot(recall, precision, ls, color=color, linewidth=1.8, label=f"{label} (AP={a:.4f})")
ax.axhline(base_rate, color="gray", linestyle=":", linewidth=1, label=f"Base rate ({base_rate:.3f})")
ax.set_xlabel("Recall", fontweight="bold")
ax.set_ylabel("Precision", fontweight="bold")
ax.set_title("Precision-Recall Curves - Out-Of-Time Test (2020)", fontweight="bold")
ax.legend(fontsize=7.5, loc="lower left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "figures" / "pr_curves_oot.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""# 6.4 Confusion matrices @ threshold 0.5 (OOT 2020)
fig, axes = plt.subplots(3, 3, figsize=(14, 12), dpi=FIGURE_DPI)
for ax, label in zip(axes.ravel(), labels_oot):
    pred = (oot_probas[label] >= FIXED_THRESHOLD).astype(int)
    cm = confusion_matrix(y_oot, pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"Confusion Matrices @ threshold={FIXED_THRESHOLD} - OOT Test (2020)",
             fontweight="bold", y=1.0)
plt.tight_layout()
plt.savefig(RESULTS / "figures" / "confusion_matrices_oot.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""# 6.5 Financial cost vs decision threshold (OOT 2020)
thresholds = np.linspace(0.05, 0.95, 91)

fig, ax = plt.subplots(figsize=(9, 6.5), dpi=FIGURE_DPI)
for label in labels_oot:
    proba = oot_probas[label]
    costs = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        fn = int(np.sum((y_oot == 1) & (pred == 0)))
        fp = int(np.sum((y_oot == 0) & (pred == 1)))
        costs.append((fn * DEFAULT_COST_RATIO + fp) / len(y_oot))
    color, ls = _style(label)
    ax.plot(thresholds, costs, ls, color=color, linewidth=1.6, label=label)
ax.set_xlabel("Decision Threshold", fontweight="bold")
ax.set_ylabel(f"Expected Cost per Applicant (FN cost = {DEFAULT_COST_RATIO:.0f}x FP)", fontweight="bold")
ax.set_title("Financial Cost vs Threshold - OOT Test (2020)", fontweight="bold")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "figures" / "cost_curves_oot.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()
"""))


# ============================================================================
# 7. SHAP EXPLAINABILITY
# ============================================================================
cells.append(md("""## 7. SHAP Explainability

Two explainers, honestly implemented:
1. **TreeSHAP** on the champion XGBoost (exact for tree ensembles).
2. **LinearSHAP** on the Logistic Regression baseline (linear-vs-tree comparison).

Attributions are computed on a stratified sample of N applicants drawn from the
2018-2019 training window (the models are re-fit on that window, so no OOT data
is involved in explanation). Outputs: global importance (top 30), beeswarm,
dependence panels, and three local case studies (TP / TN / FP)."""))

cells.append(code("""# 7.1 Domain names for readable SHAP labels
DOMAIN_NAMES = {
    "EXT_SOURCES_MEAN": "External Bureau Scores (Mean)",
    "EXT_SOURCES_STD": "External Bureau Scores (Std Dev)",
    "EXT_SOURCES_MIN": "External Bureau Scores (Min)",
    "EXT_SOURCES_MAX": "External Bureau Scores (Max)",
    "EXT_SOURCE_1": "Bureau Score 1",
    "EXT_SOURCE_2": "Bureau Score 2",
    "EXT_SOURCE_3": "Bureau Score 3",
    "EXT_SOURCES_PROD": "Bureau Score Product (1x2x3)",
    "EXT_SOURCE_2_3_PROD": "Bureau Interaction (2x3)",
    "EXT_SOURCE_1_2_PROD": "Bureau Interaction (1x2)",
    "EXT_SOURCE_1_3_PROD": "Bureau Interaction (1x3)",
    "PAYMENT_RATE": "Payment Rate (Annuity/Credit)",
    "CREDIT_INCOME_PERCENT": "Debt-to-Income Ratio",
    "ANNUITY_INCOME_PERCENT": "Debt Service Ratio (Annuity/Income)",
    "GOODS_CREDIT_RATIO": "Goods-to-Credit Ratio",
    "CREDIT_GOODS_DIFF": "Credit Minus Goods Price",
    "INCOME_PER_PERSON": "Income Per Household Member",
    "AMT_INCOME_TOTAL": "Total Income",
    "AMT_CREDIT": "Total Credit Amount",
    "AMT_ANNUITY": "Loan Annuity",
    "AMT_GOODS_PRICE": "Goods Price",
    "DAYS_BIRTH": "Age (Days)",
    "AGE_YEARS": "Age (Years)",
    "DAYS_EMPLOYED": "Employment Length (Days)",
    "EMPLOYED_YEARS": "Employment Length (Years)",
    "DAYS_EMPLOYED_PERCENT": "Employment % of Adult Life",
    "DAYS_EMPLOYED_ANOM": "Pensioner/Unemployed Flag",
    "DAYS_LAST_PHONE_CHANGE": "Days Since Phone Change",
    "DAYS_REGISTRATION": "Days Since Registration",
    "DAYS_ID_PUBLISH": "Days Since ID Publish",
    "PHONE_TO_BIRTH_RATIO": "Phone Change / Age Ratio",
    "REGISTRATION_TO_BIRTH_RATIO": "Registration / Age Ratio",
    "ID_PUBLISH_TO_BIRTH_RATIO": "ID Publish / Age Ratio",
    "CAR_TO_AGE_RATIO": "Car Age / Age Ratio",
    "OWN_CAR_AGE": "Car Age",
    "DOCUMENT_COUNT": "Verification Documents Count",
    "REGION_RATING_CLIENT": "Region Risk Rating",
    "REGION_RATING_CLIENT_W_CITY": "Region Risk Rating (City)",
    "CNT_FAM_MEMBERS": "Family Members",
    "CNT_CHILDREN": "Children Count",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Defaults in Social Circle (30 DPD)",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Observations in Social Circle (30 DPD)",
}


def display_name(feature: str) -> str:
    return DOMAIN_NAMES.get(feature, feature)


print("DOMAIN_NAMES loaded.")
"""))

cells.append(code("""# 7.2 Stratified SHAP sample from the 2018-2019 training window
from sklearn.model_selection import train_test_split

ids_train = df_train[ID_COL].to_numpy()
_, X_s, _, y_s, _, id_s = train_test_split(
    X_train, y_train, ids_train, test_size=SHAP_SAMPLE,
    stratify=y_train, random_state=RANDOM_SEED,
)
print(f"Stratified SHAP sample: {len(X_s):,} rows ({100 * y_s.mean():.1f}% default)")
"""))

cells.append(code("""# 7.3 TreeSHAP on the champion XGBoost (baseline hyperparameters)
import shap

t_shap = time.time()
print("Fitting champion XGBoost on full 2018-2019 window for SHAP...")
xgb_model = make_model("xgboost", "baseline", spw, RANDOM_SEED)
xgb_model.fit(X_train, y_train)

print(f"Computing TreeSHAP values on {len(X_s):,} samples...")
tree_explainer = shap.TreeExplainer(xgb_model)
shap_tree = tree_explainer.shap_values(X_s)
base_tree = float(np.asarray(tree_explainer.expected_value).ravel()[0])

mean_abs_tree = np.mean(np.abs(shap_tree), axis=0)
order = np.argsort(mean_abs_tree)[::-1]
df_imp = pd.DataFrame({
    "rank": np.arange(1, len(feat_names) + 1),
    "feature": [feat_names[i] for i in order],
    "display_name": [display_name(feat_names[i]) for i in order],
    "mean_abs_shap": mean_abs_tree[order],
})
df_imp.head(30).to_csv(RESULTS / "shap_importance_top30.csv", index=False)
print(f"Top-5 TreeSHAP features: {df_imp.head(5)['feature'].tolist()}")
print(f"TreeSHAP done in {time.time() - t_shap:.1f}s")
"""))

cells.append(code("""# 7.4 Global importance bar chart (TreeSHAP)
top = df_imp.head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(11, 9), dpi=FIGURE_DPI)
ax.barh(top["display_name"], top["mean_abs_shap"], color="#1f77b4", alpha=0.85)
ax.set_xlabel("Mean |SHAP value| (impact on default log-odds)", fontweight="bold")
ax.set_title("Global Feature Importance (TreeSHAP, XGBoost Champion)\\n"
             f"Stratified sample N={SHAP_SAMPLE:,}", fontweight="bold")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(RESULTS / "figures" / "shap_importance.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

print("Top 15 features by mean |SHAP| (XGBoost / TreeSHAP):")
df_imp.head(15).round(4)
"""))

cells.append(code("""# 7.5 Beeswarm summary plot
explanation = shap.Explanation(
    values=shap_tree,
    base_values=np.full(len(X_s), base_tree),
    data=X_s,
    feature_names=[display_name(f) for f in feat_names],
)
plt.figure(figsize=(12, 9), dpi=FIGURE_DPI)
shap.plots.beeswarm(explanation, max_display=20, show=False)
plt.title("SHAP Beeswarm: feature value vs impact on default risk", fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS / "figures" / "shap_beeswarm.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""# 7.6 Dependence panels (EXT_SOURCES_MEAN, PAYMENT_RATE, CREDIT_INCOME_PERCENT)
targets = ["EXT_SOURCES_MEAN", "PAYMENT_RATE", "CREDIT_INCOME_PERCENT"]
idx = {f: feat_names.index(f) for f in targets if f in feat_names}

fig, axes = plt.subplots(1, len(idx), figsize=(6 * len(idx), 5), dpi=FIGURE_DPI)
if len(idx) == 1:
    axes = [axes]
for ax, (feat, i) in zip(axes, idx.items()):
    ax.scatter(X_s[:, i], shap_tree[:, i], alpha=0.4, s=15, color="#1976d2")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(display_name(feat), fontweight="bold")
    ax.set_ylabel("SHAP value", fontweight="bold")
    ax.set_title(f"Dependence: {feat}", fontweight="bold")
    ax.grid(alpha=0.3)
fig.suptitle("SHAP Dependence Panels (XGBoost, OOT-clean training window)", fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS / "figures" / "shap_dependence.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()
"""))

cells.append(code("""# 7.7 Three local case studies (TP / TN / FP) with attribution panels

def _draw_case_panel(ax, title, prob, actual, base_value, shap_row, feature_names, top_k=8):
    \"\"\"Clean horizontal-bar panel of one applicant's top SHAP contributions.\"\"\"
    order = np.argsort(np.abs(shap_row))[::-1][:top_k][::-1]
    names = [display_name(feature_names[i]) for i in order]
    vals = shap_row[order]
    colors = ["#c0392b" if v > 0 else "#2166ac" for v in vals]

    y = np.arange(len(order))
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.68)

    xmax = np.abs(vals).max()
    for yi, v in zip(y, vals):
        ax.text(v + (0.06 * xmax if v >= 0 else -0.06 * xmax), yi,
                f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=11, fontweight="bold")

    ax.axvline(0, color="#444444", linewidth=1.2)
    ax.plot([base_value], [-0.75], marker="D", color="#555555", markersize=9,
            clip_on=False, zorder=5)
    ax.text(base_value, -1.15, f"base value = {base_value:.2f}",
            ha="center", fontsize=10, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_ylim(-1.8, len(order) - 0.3)
    pad = xmax * 0.28
    ax.set_xlim(min(vals.min(), base_value) - pad, max(vals.max(), base_value) + pad)
    ax.set_xlabel("SHAP impact on default log-odds", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    outcome = "DEFAULTED" if actual == 1 else "REPAID"
    ax.set_title(f"{title}\\nP(default) = {prob:.3f}   |   actual outcome: {outcome}",
                 fontsize=13, fontweight="bold", pad=10, loc="left")


proba_s = xgb_model.predict_proba(X_s)[:, 1]

tp = np.where((y_s == 1) & (proba_s >= 0.5))[0]
tn = np.where((y_s == 0) & (proba_s < 0.1))[0]
fp = np.where((y_s == 0) & (proba_s >= 0.45) & (proba_s <= 0.65))[0]

idx_tp = tp[np.argmax(proba_s[tp])] if len(tp) else int(np.argmax(proba_s))
idx_tn = tn[np.argmin(proba_s[tn])] if len(tn) else int(np.argmin(proba_s))
idx_fp = fp[np.argmax(proba_s[fp])] if len(fp) else int(np.where(y_s == 0)[0][0])

cases = [
    (idx_tp, f"Case 1: True Positive - applicant {int(id_s[idx_tp])} (defaulter correctly flagged)"),
    (idx_tn, f"Case 2: True Negative - applicant {int(id_s[idx_tn])} (solvent borrower approved)"),
    (idx_fp, f"Case 3: False Positive - applicant {int(id_s[idx_fp])} (borderline false alarm)"),
]

fig, axes = plt.subplots(3, 1, figsize=(10.5, 14), dpi=FIGURE_DPI,
                         constrained_layout=True)
records_cases = []
for ax, (idx, title) in zip(axes, cases):
    _draw_case_panel(ax, title, float(proba_s[idx]), int(y_s[idx]),
                     base_tree, shap_tree[idx], feat_names)

    top5 = np.argsort(np.abs(shap_tree[idx]))[::-1][:5]
    for rank, fi in enumerate(top5, start=1):
        records_cases.append({
            "case": title.split(":")[0],
            "applicant_id": int(id_s[idx]),
            "predicted_default_prob": float(proba_s[idx]),
            "actual_default": int(y_s[idx]),
            "factor_rank": rank,
            "feature": feat_names[fi],
            "display_name": display_name(feat_names[fi]),
            "feature_value": float(X_s[idx, fi]),
            "shap_impact": float(shap_tree[idx, fi]),
        })

from matplotlib.patches import Patch
handles = [
    Patch(color="#c0392b", label="pushes prediction toward default"),
    Patch(color="#2166ac", label="pushes prediction toward repayment"),
    plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="#555555",
               markersize=9, label="model base value"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, -0.005))

plt.savefig(RESULTS / "figures" / "shap_case_studies.png", dpi=FIGURE_DPI,
            bbox_inches="tight", facecolor="white")
plt.show()

df_cases = pd.DataFrame(records_cases)
df_cases.to_csv(RESULTS / "shap_case_studies.csv", index=False)
print("Case studies:")
df_cases[df_cases["factor_rank"] == 1][["case", "applicant_id", "predicted_default_prob", "actual_default"]]
"""))

cells.append(code("""# 7.8 LinearSHAP on the Logistic Regression baseline (comparison)
import scipy.sparse as sp

print("Fitting Logistic Regression for LinearSHAP comparison...")
lr_model = make_model("logistic_regression", "baseline", spw, RANDOM_SEED)
lr_model.fit(X_train, y_train)

background = sp.csr_matrix(X_train[: min(2000, len(X_train))])
linear_explainer = shap.LinearExplainer(lr_model, background)
shap_linear = linear_explainer.shap_values(X_s)

mean_abs_lin = np.mean(np.abs(shap_linear), axis=0)
order_lin = np.argsort(mean_abs_lin)[::-1]
df_imp_lin = pd.DataFrame({
    "rank": np.arange(1, len(feat_names) + 1),
    "feature": [feat_names[i] for i in order_lin],
    "display_name": [display_name(feat_names[i]) for i in order_lin],
    "mean_abs_shap": mean_abs_lin[order_lin],
})
df_imp_lin.head(30).to_csv(RESULTS / "shap_importance_top30_logistic.csv", index=False)

top_lin = df_imp_lin.head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(11, 9), dpi=FIGURE_DPI)
ax.barh(top_lin["display_name"], top_lin["mean_abs_shap"], color="#1f77b4", alpha=0.85)
ax.set_xlabel("Mean |SHAP value| (impact on default log-odds)", fontweight="bold")
ax.set_title("Global Feature Importance (LinearSHAP, Logistic Regression)\\n"
             f"Stratified sample N={SHAP_SAMPLE:,}", fontweight="bold")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(RESULTS / "figures" / "shap_importance_logistic.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# Rank agreement between tree and linear explanations (top-30 overlap)
top30_tree = set(df_imp.head(30)["feature"])
top30_lin = set(df_imp_lin.head(30)["feature"])
overlap = len(top30_tree & top30_lin)
print(f"Top-30 importance overlap (tree vs linear): {overlap}/30")
with open(RESULTS / "shap_tree_vs_linear_overlap.txt", "w", encoding="utf-8") as f:
    f.write(f"Top-30 feature overlap between TreeSHAP (XGBoost) and LinearSHAP (LR): {overlap}/30\\n")
"""))

cells.append(code("""# 7.9 SHAP verification against dissertation Chapter 5
if FULL_RUN:
    assert df_imp.iloc[0]["feature"] == "EXT_SOURCES_MEAN"
    assert abs(df_imp.iloc[0]["mean_abs_shap"] - 0.438) < 0.02
    assert df_cases["applicant_id"].unique().tolist() == [187933, 237456, 334854]
    assert "16/30" in open(RESULTS / "shap_tree_vs_linear_overlap.txt").read()
    print("SHAP CHECKPOINT OK: EXT_SOURCES_MEAN top (0.438), cases 187933/237456/334854, overlap 16/30")
else:
    print("Reduced smoke run - skipping dissertation-number assertions.")
"""))

# ============================================================================
# 8. VERIFICATION & DOWNLOAD
# ============================================================================
cells.append(md("""## 8. Verification Summary & Results Download

Everything above was computed live in this session. This final section
collects the headline numbers, checks them against the dissertation, and
offers the full result set for download."""))

cells.append(code("""# 8.1 Headline results summary
summary_rows = []
if FULL_RUN:
    champ_cv = agg.iloc[0]
    champ_oot = df_oot[df_oot["display_name"] == "XGBoost (Baseline)"].iloc[0]
    cs_oot = df_oot[df_oot["display_name"] == "XGBoost (Cost-Sensitive)"].iloc[0]
    summary_rows = [
        ("Train window (2018-19) rows", f"{len(df_train):,}", "205,007"),
        ("Test window (2020) rows", f"{len(df_test):,}", "102,504"),
        ("Train default rate", f"{df_train[TARGET_COL].mean():.2%}", "8.12%"),
        ("Test default rate", f"{df_test[TARGET_COL].mean():.2%}", "7.98%"),
        ("CV champion (mean PR-AUC)", f"{champ_cv['display_name']} ({champ_cv['pr_auc_mean']:.4f})",
         "XGBoost (Baseline) (~0.2502)"),
        ("OOT champion ROC-AUC", f"{champ_oot['roc_auc']:.4f}", "0.7669"),
        ("OOT champion PR-AUC", f"{champ_oot['pr_auc']:.4f}", "0.2513"),
        ("OOT cost-sensitive recall", f"{cs_oot['recall']:.3f}", "~0.657"),
        ("Significant comparisons (alpha=0.05)", f"{int(df_stats['significant_alpha_0.05'].sum())}/16", "15/16"),
        ("Top SHAP feature", f"{df_imp.iloc[0]['feature']} ({df_imp.iloc[0]['mean_abs_shap']:.3f})",
         "EXT_SOURCES_MEAN (0.438)"),
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["quantity", "this run", "dissertation"])
    print(df_summary.to_string(index=False))
else:
    print("Reduced smoke run - summary of what was computed:")
    print(f"  CV fold records: {len(df_cv)} | OOT configs: {len(df_oot)} | "
          f"statistical comparisons: {len(df_stats)}")
"""))

cells.append(code("""# 8.2 Zip all computed artifacts and download
import shutil

shutil.make_archive("from_scratch_results", "zip", root_dir=".", base_dir="from_scratch_results")
print(f"Zipped: from_scratch_results.zip "
      f"({Path('from_scratch_results.zip').stat().st_size / 1e6:.1f} MB)")

try:
    from google.colab import files
    files.download("from_scratch_results.zip")
    print("Download started.")
except ImportError:
    print("Not running on Colab - the zip is in the working directory.")
"""))

cells.append(md("""## Done

All eight phases reproduced the experiment end-to-end - from exploratory data
analysis through cleaning, cross-validation, out-of-time testing, statistical
testing, evaluation figures, and SHAP explainability.

Every number in this notebook was computed live in this session from the raw
`application_train.csv` you provided. Nothing was loaded from a pre-existing
result file. If the full-run assertions in phases 3-7 all passed, the
reproduction matches the dissertation."""))

# ============================================================================
# WRITE NOTEBOOK
# ============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "Loan Default Prediction - Full Experiment From Scratch",
                  "provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Wrote {OUT.name}: {len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown)")
