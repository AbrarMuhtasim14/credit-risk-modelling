"""
Central configuration for the MSc Credit Risk experiment (clean-room build).

Design principles encoded here:
- Temporal partitioning: fit on 2018-2019, validate Out-Of-Time on 2020.
- All preprocessing statistics are computed per-split / per-fold (no global fit).
- Frozen hyperparameters across imbalance strategies (fair 9-cell comparison).
- Full reproducibility: single seed family derived from RANDOM_SEED.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_APPLICATION_TRAIN = RAW_DIR / "application_train.csv"

# ---------------------------------------------------------------------------
# Column identifiers
# ---------------------------------------------------------------------------
ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
TIME_COL = "application_date"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_FOLDS = 5
N_REPEATS = 3  # 3x5 repeated stratified K-fold => 15 paired observations per config

# ---------------------------------------------------------------------------
# Temporal split definition
# ---------------------------------------------------------------------------
TRAIN_YEARS = [2018, 2019]
TEST_YEARS = [2020]

# ---------------------------------------------------------------------------
# Data cleaning constants
# ---------------------------------------------------------------------------
DAYS_EMPLOYED_SENTINEL = 365243          # "1000 years" placeholder -> pensioner/unemployed
DAYS_EMPLOYED_ANOM_COL = "DAYS_EMPLOYED_ANOM"
INCOME_WINSOR_QUANTILE = 0.999           # cap fitted on training years ONLY

# ---------------------------------------------------------------------------
# Feature selection decisions (each justified in docs/EXPERIMENT_DESIGN.md)
# ---------------------------------------------------------------------------
# Building features come in _AVG/_MEDI/_MODE triplets that are near-duplicates
# (|r| > 0.97 in EDA). We keep _AVG and drop the other two variants.
# TOTALAREA_MODE is kept: it is the only total-area signal (no _AVG counterpart).
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
]  # 28 columns

# Social-circle 60-DPD variants are near-duplicates of the 30-DPD variants
# (OBS_60 vs OBS_30 r=0.9985; DEF_60 vs DEF_30 r=0.864 in EDA).
SOCIAL_CIRCLE_DROP = ["OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"]

# NOTE: FLAG_EMP_PHONE is deliberately KEPT. Its raw r=-0.9998 with DAYS_EMPLOYED
# is a pure artifact of the 365243 sentinel; after sentinel correction the
# correlation collapses to ~0.002 (verified in EDA), so it is not collinear.

COLS_TO_DROP = BUILDING_DROP + SOCIAL_CIRCLE_DROP  # 30 columns

EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
DEFAULT_COST_RATIO = 10.0   # false negative (missed default) costs 10x a false positive
FIXED_THRESHOLD = 0.5
SMOTE_K_NEIGHBORS = 5
SHAP_SAMPLE_SIZE = 2000

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FIGURE_DPI = 300
