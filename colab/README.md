# Colab Handover — Loan Default Prediction (MSc Project)

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
