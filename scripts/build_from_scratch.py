"""Build FROM_SCRATCH.ipynb — ONE Colab notebook that reproduces the ENTIRE
experiment from scratch.

Key difference from WALKTHROUGH.ipynb (which displayed checkpointed local results):
this notebook DELETES all prior result artifacts in its second cell, then runs every
phase live. Every table and figure it displays was computed by this notebook itself.

Cells are reused verbatim from build_colab_folder.py (the battle-tested phase
notebooks), so results cannot diverge from the local artefact.
"""

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import build_colab_folder as B  # noqa: E402  (import rewrites colab/ notebooks; harmless)

OUT = B.COLAB / "FROM_SCRATCH.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


FRESH_START_CELL = code("""# ============================================================
# FRESH START: delete every result artifact from any earlier run
# so this notebook computes everything from scratch.
# (The raw dataset in data/raw/ is NEVER touched.)
#
# Resume-safe: a marker file records that the wipe already happened
# in this from-scratch session, so if Colab disconnects and you
# re-run, the phases resume from their checkpoints instead of
# wiping and restarting. To force a brand-new from-scratch run,
# set FORCE = True below.
# ============================================================
from pathlib import Path

FORCE = False   # set True to wipe and restart even after a disconnect

marker = PROJECT_ROOT / ".fresh_start_done"
if marker.exists() and not FORCE:
    print("Fresh start already done in this session - keeping checkpoints so the run can resume.")
else:
    victims = []
    for pattern in [
        "data/processed/*.parquet",
        "reports/tables/*.csv", "reports/tables/*.txt",
        "reports/figures/*.png",
        "models/*",
        "outputs/eda/*",
    ]:
        for p in PROJECT_ROOT.glob(pattern):
            p.unlink()
            victims.append(str(p.relative_to(PROJECT_ROOT)))
    marker.write_text("from-scratch session started", encoding="utf-8")
    print(f"Cleared {len(victims)} prior result files:")
    for v in victims:
        print("  -", v)
    print("\\nThis notebook will now recompute every phase from the raw CSV.")
""")

FOLD_DEMO_CELL = code("""# ---- Bonus: the mechanics of ONE cross-validation fold, made visible ----
# The full CV phase below repeats exactly this procedure 135 times
# (3 repeats x 5 folds x 9 configurations).
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from src.features.preprocessor import fit_transform, transform_only
from src.models.factory import make_model
from src.models.metrics import compute_metrics
from src.config import RANDOM_SEED, FIXED_THRESHOLD, DEFAULT_COST_RATIO, SMOTE_K_NEIGHBORS

y_all = df_train["TARGET"].to_numpy(dtype=np.int8)
n = len(df_train)

seed = RANDOM_SEED  # repeat 1 seed (production uses RANDOM_SEED + (repeat-1)*1000)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
tr_pos, va_pos = next(iter(skf.split(np.zeros(n), y_all)))
df_tr, df_va = df_train.iloc[tr_pos], df_train.iloc[va_pos]
y_tr, y_va = y_all[tr_pos], y_all[va_pos]
print(f"Fold 1 of repeat 1: train-fold {len(y_tr):,} rows | validation-fold {len(y_va):,} rows")

# 1. Preprocessor fitted on the TRAIN FOLD ONLY (no leakage)
X_tr, feat_names, pre, num_cols, cat_cols = fit_transform(df_tr)
X_va = transform_only(df_va, pre, num_cols, cat_cols)
print(f"Encoded feature matrix: {X_tr.shape[1]} features")

# 2. SMOTE on the train fold only
smote = SMOTE(sampling_strategy="auto", k_neighbors=SMOTE_K_NEIGHBORS, random_state=seed)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
spw = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)
print(f"SMOTE: {len(y_tr):,} -> {len(y_tr_sm):,} rows | scale_pos_weight = {spw:.2f}")

# 3. Fit one model (XGBoost baseline) and score on the held-out fold
t0 = time.time()
model = make_model("xgboost", "baseline", spw, seed + 1)
model.fit(X_tr, y_tr)
proba = model.predict_proba(X_va)[:, 1]
m = compute_metrics(y_va, proba, FIXED_THRESHOLD, DEFAULT_COST_RATIO)
print(f"\\nFold-1 metrics for XGBoost (Baseline)  [{time.time()-t0:.1f}s]:")
for k, v in m.items():
    print(f"  {k:22s} {v:.4f}")
""")

cells = [
    md("""# FULL EXPERIMENT FROM SCRATCH — Loan Default Prediction (MSc Project)

**One notebook. Every phase computed live. Nothing loaded from a previous run.**

This notebook reproduces the entire experiment end-to-end on Google Colab:

| Phase | What it computes | Approx. time |
|---|---|---|
| Setup | Mount Drive, locate project, install dependencies | ~1 min |
| Fresh start | Deletes all prior result artifacts | instant |
| 0 — EDA | Full exploratory analysis, 6 figures | ~5 min |
| 1 — Data prep | Cleaning, feature engineering, temporal split | ~5 min |
| 2 — Cross-validation | 3x5 repeated CV, 9 configs = **135 model fits** | ~30-45 min |
| 3 — OOT + statistics | Out-of-time 2020 test, 16 statistical tests, figures | ~10 min |
| 4 — SHAP | TreeSHAP + LinearSHAP, case studies, champion model | ~5 min |
| 5 — Collect | Zip all freshly computed results + download | ~1 min |

**Total: ~1 hour on Colab CPU.**

Every table and figure shown below is computed by this notebook itself. At the end
of each phase, an assertion checks the fresh numbers against the local run's
published results (e.g. champion OOT ROC-AUC 0.7669, PR-AUC 0.2513) — if all
assertions pass, the from-scratch reproduction matches the local artefact.

**If the runtime disconnects:** re-run the notebook. The fresh-start cell only wipes
at the beginning of a session; the phases checkpoint their progress on Drive, so a
re-run resumes instead of starting over.
"""),
    code(B.SETUP_CELL),
    code("""# Install pinned dependencies (Colab pre-installs most; these ensure version parity
# with the local artefact that produced the dissertation numbers).
!pip install -q "xgboost>=3.4,<3.5" "imbalanced-learn>=0.14,<0.15" "shap>=0.52,<0.53" "pyarrow>=25"
"""),
    FRESH_START_CELL,
]

# Chain the battle-tested phase cells (minus their individual setup cells)
for _nb in (B.nb00, B.nb01, B.nb02, B.nb03, B.nb04, B.nb05):
    cells.extend(B.strip_setup(_nb))

# Insert the fold-mechanics demo right after phase 1's verification cell,
# i.e. just before phase 2's title cell.
insert_at = None
for i, c in enumerate(cells):
    if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("# Phase 2"):
        insert_at = i
        break
assert insert_at is not None, "Phase 2 header not found"
cells.insert(insert_at, md("""## Bonus: inside ONE cross-validation fold

Before the full 135-fit CV runs, here is a single fold opened up so you can see
exactly what each fit does: split -> fit preprocessor on the train fold only ->
SMOTE on the train fold only -> fit one model -> score on the held-out fold."""))
cells.insert(insert_at + 1, FOLD_DEMO_CELL)

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT.name} ({len(cells)} cells)")
