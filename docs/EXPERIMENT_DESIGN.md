# Experiment Design — Loan Default Prediction
### Class Imbalance Handling & Model Explainability (MSc Project)

**Dataset:** Home Credit Default Risk — `application_train.csv` (307,511 rows × 123 cols)
**Prepared by:** Mavis (senior ML engineering review) · 2026-09-01
**Status:** PROPOSED — pending student/supervisor sign-off on §3 (split strategy)

---

## 1. Background & Problem Framing

The project must answer one focused question, per the approved proposal:

> **How does class imbalance handling affect the performance and interpretation of
> machine learning models for loan default prediction?**

Dataset facts that shape the design (verified via EDA, see `outputs/eda/`):

| Fact | Value | Design consequence |
|---|---|---|
| Default rate | 8.07% (24,825 / 307,511), ratio 11.39:1 | Accuracy is banned as a headline metric; PR-AUC leads |
| Time span | Jan 2018 → Dec 2020, 1,096 unique dates | Temporal split is viable; monthly default rate stable at 7.7–8.3% (no drift) |
| Missingness | 67 cols affected; 41 cols >50%; defaulters have more missing cells (26.9% vs 24.2%) | Missingness is signal → missing-indicator features |
| Strongest predictors | EXT_SOURCE_1/2/3 (corr −0.155 to −0.179); EXT_SOURCE_1 is 56% missing | Impute + indicator, never drop |
| Redundancy | 62 pairs \|r\|>0.9 (mostly `_AVG/_MEDI/_MODE` triplets); FLAG_EMP_PHONE ↔ DAYS_EMPLOYED r=−0.9998 raw but 0.002 after sentinel cleaning | Prune duplicates; justify drops with post-cleaning stats |
| Anomalies | DAYS_EMPLOYED=365243 sentinel (18% of rows, that group defaults at 5.4% vs 8.7%); AMT_INCOME_TOTAL max 117M | Sentinel → NaN + flag; winsorize income at 99.9th pct (=900,000) |
| Duplicates | 0 | No dedup step needed |

**What this experiment is NOT:** not a Kaggle leaderboard chase, not a production
lending system, not a new-algorithm contribution. It is a controlled, reproducible
comparison of three imbalance interventions under identical conditions.

---

## 2. Research Questions & Hypotheses

**RQ1.** Does imbalance handling improve default detection (recall, F2) and at what
cost to precision and ranking ability (ROC-AUC, PR-AUC)?

**RQ2.** Which intervention — cost-sensitive learning or SMOTE — gives the best
balance between detection, ranking, and practical suitability?

**RQ3.** Does imbalance handling change *which features* the best model appears to
rely on (SHAP stability across strategies)?

Pre-registered hypotheses (state these in the methodology chapter *before* results):

- **H1:** Cost-sensitive weighting and SMOTE will raise recall substantially at the
  default 0.5 threshold, at the cost of precision.
- **H2:** SMOTE will underperform cost-sensitive weighting on ranking metrics
  (ROC-AUC, PR-AUC) because k-NN interpolation in ~190 dimensions creates noisy
  synthetic points (curse of dimensionality).
- **H3:** Ranking ability (ROC-AUC) is approximately invariant to the intervention;
  the interventions mainly move the decision threshold, not the score ordering.
- **H4:** Top-10 SHAP feature rankings are broadly stable across strategies, with
  EXT_SOURCE features dominant in all conditions.

H2 and H3 are the intellectually interesting ones — they predict *why* the results
come out as they do, which is exactly what distinction-grade discussion requires.

---

## 3. Data Partitioning Strategy (the most important decision)

### Recommended: temporal holdout + nested stratified CV

```
Full data (307,511 rows, Jan 2018 – Dec 2020)
│
├── TRAIN/DEV pool: Jan 2018 – Jun 2020  (~262k rows, ~85%)
│   └── 5-fold Stratified CV (seed=42) inside this pool ONLY
│       ├── used for: model selection, hyperparameter tuning,
│       │   the 9-cell imbalance comparison
│       └── preprocessing fitted INSIDE each fold (see §4)
│
└── HELD-OUT TEST: Jul 2020 – Dec 2020  (~45k rows, ~15%)
    └── touched exactly ONCE per final configuration
        → the single source of headline numbers in the dissertation
```

**Why temporal, not random:**
1. Credit models are deployed against *future* applicants. A random split lets the
   model "memorize" applicants from the same month it is tested on — optimistic bias.
2. The EDA shows the default rate is stable across all 36 months (7.7–8.3%), so a
   temporal split costs almost nothing in distribution shift while buying realism.
3. It directly answers the leakage objective in the proposal ("identifying features
   that may create data leakage") — the split itself is a leakage control.

**Sensitivity check:** repeat the 9-cell comparison once with a stratified random
70/15/15 split. If conclusions agree (they should, given no drift), that's a strong
robustness argument for the discussion chapter. If they disagree, that's an even
better discussion.

**Rules:**
- Test set is used ONCE per configuration for final reporting. No iterating on it.
- All CV, tuning, and SMOTE happen inside the train/dev pool.
- Stratification by TARGET everywhere (folds preserve the 8.07% rate).
- Fixed seeds: split=42, CV folds=42, models=42 (documented in `config.py`).

---

## 4. Preprocessing Pipeline (leakage-free, nested)

Built as a single scikit-learn `Pipeline` so every transform is fit on training
data only — this fixes the one genuine flaw in the existing `F:\credit-risk-project`
codebase (which fits the ColumnTransformer on all rows before CV).

### Step order (all inside the CV fold / final-fit):

1. **Anomaly remediation** (deterministic, no fitting — safe anywhere)
   - `DAYS_EMPLOYED == 365243` → NaN; add binary `DAYS_EMPLOYED_ANOM`
   - `CODE_GENDER == 'XNA'` → NaN
   - `FLAG_OWN_CAR/FLAG_OWN_REALTY` 'Y'/'N' → 1/0

2. **Winsorization** (fit on train fold)
   - `AMT_INCOME_TOTAL` clipped at train-fold 99.9th percentile

3. **Feature engineering** (deterministic transforms)
   - `CREDIT_INCOME_RATIO` = AMT_CREDIT / AMT_INCOME_TOTAL
   - `ANNUITY_INCOME_RATIO` = AMT_ANNUITY / AMT_INCOME_TOTAL
   - `CREDIT_GOODS_RATIO` = AMT_CREDIT / AMT_GOODS_PRICE
   - `CREDIT_TERM` = AMT_CREDIT / AMT_ANNUITY
   - `AGE_YEARS` = −DAYS_BIRTH / 365
   - `EMPLOYED_YEARS` = −DAYS_EMPLOYED / 365 (post-sentinel)
   - `EXT_SOURCE_MEAN` = nanmean(EXT_SOURCE_1/2/3)  ← computed BEFORE imputation
   - `BUREAU_ENQ_TOTAL` = sum of the 6 AMT_REQ_CREDIT_BUREAU_* columns
   - (~8–10 ratios total; keep the list exact and documented)

4. **Feature selection** (static, decided from EDA — not fit on target)
   - Drop `SK_ID_CURR`, `application_date` (used only for the split),
     `WEEKDAY_APPR_PROCESS_START`, `NAME_TYPE_SUITE`
   - From each building `_AVG/_MEDI/_MODE` triplet keep only `_AVG` (29 drops)
   - Drop `OBS_60_CNT_SOCIAL_CIRCLE` (r=0.9985 with OBS_30)
   - Drop `FLAG_EMP_PHONE` — justification: redundant with `DAYS_EMPLOYED_ANOM`
     after sentinel cleaning (r=0.002 with cleaned DAYS_EMPLOYED, but the 55,386
     FLAG=0 rows are exactly the sentinel rows). ⚠ Do NOT cite the raw −0.9998
     correlation in the dissertation — it only exists pre-cleaning.
   - `ORGANIZATION_TYPE`: frequency-group to top-10 + "OTHER"
   - Result: ~85–95 numeric + 16 categorical features

5. **Missing indicators** (high-value, cheap)
   - Binary `_was_missing` flags for: `EXT_SOURCE_1`, `EXT_SOURCE_3`,
     `AMT_ANNUITY`, `OCCUPATION_TYPE` (31% missing), and `BUREAU_ENQ_TOTAL`
     (13.5% missing). Missingness differs by class — this captures it.

6. **Numeric pipeline:** median imputation → RobustScaler
   (median: financials are right-skewed; RobustScaler: outlier-robust IQR scaling
   for the logistic regression baseline)

7. **Categorical pipeline:** constant 'Missing' imputation →
   OneHotEncoder(handle_unknown='ignore')
   ('Missing' as its own category: not providing occupation is itself a risk signal)

**Expected final dimension:** ~190–200 features.

---

## 5. Models & Hyperparameter Policy

| Model | Role | Imbalance knobs |
|---|---|---|
| Logistic Regression | Transparent baseline (max_iter=1000, solver lbfgs) | `class_weight='balanced'` |
| Random Forest (n_estimators=400) | Bagging ensemble | `class_weight='balanced'` |
| XGBoost | Gradient boosting (expected winner) | `scale_pos_weight = 11.39` |

**Critical control — hyperparameters are FIXED across imbalance strategies.**
Tune once (small RandomizedSearchCV, ~30 iterations, on the train/dev pool using
the BASELINE strategy only), then freeze. If you re-tune per strategy, you can no
longer attribute performance differences to the imbalance intervention — the whole
point of the experiment. State this control explicitly in the methodology.

---

## 6. The Imbalance Intervention (the experimental manipulation)

Three conditions, applied identically across all three models = **9-cell matrix**:

| Condition | Mechanism | Where applied |
|---|---|---|
| **C1 Baseline** | Natural 8.07% distribution, unweighted | — |
| **C2 Cost-sensitive** | Inverse-frequency class weights in the loss function | Training fold only |
| **C3 SMOTE** | k-NN synthetic oversampling to 1:1 (k=5, imbalanced-learn) | Training fold only — NEVER on validation/test |

SMOTE rules (non-negotiable):
- Applied after preprocessing transform, inside each training fold
- Validation fold always stays at the natural 8.07% distribution
- Documented with a code comment + methodology paragraph

### Threshold policy (second experimental factor, often missed)

At an 8% base rate, the naive 0.5 threshold yields ~2% recall (verified in the
existing runs). Every cell is therefore reported at FOUR threshold policies:

1. Fixed 0.5 (naive reference)
2. F1-optimal (chosen on CV validation folds)
3. **F2-optimal** (recall-weighted — the credit-risk-appropriate choice)
4. Cost-optimal under an explicit cost matrix:
   missed default costs 5× a wrongful rejection (justify in dissertation)

Thresholds are selected on CV folds and applied to the test set — never chosen
using test-set information.

---

## 7. Evaluation Protocol

### Metric hierarchy (pre-declared)

| Tier | Metrics | Purpose |
|---|---|---|
| Primary | **PR-AUC** | Most informative under severe imbalance |
| Secondary | ROC-AUC, F1, F2, precision, recall | Ranking + detection |
| Threshold-fixed | Confusion matrix, precision/recall at each of the 4 thresholds | Operational view |
| Calibration | Brier score, reliability diagram | Class weighting distorts probabilities — worth one paragraph |
| Reported but de-emphasized | Accuracy | Shown once to demonstrate why it misleads (91.9% for a do-nothing model) |

### Statistical testing (handbook Appendix F requires real statistics)

- **3 × 5-fold repeated CV** (seeds 42, 43, 44) on the train/dev pool → 15 paired
  fold observations per comparison
- Paired comparisons (baseline vs cost-sensitive vs SMOTE, per model):
  **Wilcoxon signed-rank test** + **Nadeau & Bengio (2003) corrected paired t-test**
- Report p-values and effect sizes (mean Δ ROC-AUC, Δ PR-AUC, Δ recall with 95% CIs)
- This upgrades "SMOTE was worse" into "SMOTE was significantly worse (p < 0.001,
  corrected)" — the difference between description and inference.

---

## 8. Explainability Phase (SHAP)

Applied after the winner is selected:

1. **SHAP summary (beeswarm)** on the best model, test-set sample (~5,000 rows)
2. **SHAP dependence plots** for EXT_SOURCE_1/2/3 and the top financial ratios
3. **Cross-strategy stability analysis** (answers RQ3, novel angle):
   compute top-15 SHAP importances for XGBoost under all three strategies;
   report rank correlation (Spearman) between the importance vectors.
   If SMOTE reshuffles importance, that's a genuine finding about explainability
   under resampling — discussion-chapter gold.
4. **Two case studies** (one defaulter, one repayer) with waterfall plots
5. Critical discussion: do the SHAP stories align with credit-risk intuition
   and the literature (Lessmann et al. 2015; Bussmann et al. 2021)?

---

## 9. Run Inventory & Compute Budget

| Run | Count | Notes |
|---|---|---|
| 9-cell matrix × 3×5-fold CV | 135 fits | Core experiment |
| Hyperparameter search (baseline only) | ~30 × 5 fits | One-time |
| Final refit + test evaluation | 9 fits | One per cell |
| Sensitivity split (random) | 9 cells × 5 folds | Robustness check |
| SHAP | 3 models × 5k samples | TreeExplainer, fast |

Estimated total runtime on a laptop: **2–4 hours** (XGBoost dominates; SMOTE adds
~2–3 min per fold at this dimensionality). Entirely feasible.

---

## 10. Reproducibility Controls

- Single `config.py`: all seeds, paths, hyperparameters, cost matrix
- `requirements.txt` with pinned versions (pandas, scikit-learn, imbalanced-learn,
  xgboost, shap, matplotlib, seaborn)
- One entry point: `python run_experiment.py` regenerates every table and figure
- Every results table written to `reports/tables/*.csv`, every figure to
  `reports/figures/*.png` (300 DPI)
- Runtime log saved alongside results
- Directory layout:

```
credit-risk-project/
├── data/raw/                  # original CSV, never modified
├── data/processed/            # cached splits only (no SMOTE ever cached)
├── src/{config,data,features,models,explainability,utils}/
├── notebooks/                 # EDA + narrative walkthrough
├── reports/{tables,figures}/  # all experimental outputs
└── run_experiment.py
```

---

## 11. Mapping to Proposal Objectives & Dissertation

| Proposal objective | Covered by |
|---|---|
| Lit review | Ch. 2 (credit scoring, imbalance, XAI) |
| Dataset selection & justification | §1 + EDA chapter |
| EDA | `scripts/eda.py` + EDA chapter (done ✅) |
| Pre-processing incl. leakage check | §4 + methodology chapter |
| Reproducible pipeline | §10 + artefact |
| 3-model comparison | §5, §9 |
| Imbalance handling before/after | §6, §7 — the core results chapter |
| Imbalanced-appropriate metrics | §7 |
| SHAP on best model | §8 |
| Critical comparison | Discussion chapter: H1–H4 verdicts + split sensitivity + calibration |

Dissertation structure: Intro → Lit Review → Methodology (this document, in prose)
→ EDA → Experimental Results → Explainability → Critical Discussion (incl. ethics,
fairness limits, limitations) → Conclusions. ~14,400 words, third person, Harvard
referencing (handbook §7).

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| SMOTE runtime blows up with repeats | k=5 default, float32, run overnight if needed |
| XGBoost dominates so strongly that the comparison feels one-sided | That IS the finding — frame it against Lessmann et al. benchmark literature |
| Small fold count → weak stats | 3×5 repeats (15 observations) + corrected tests |
| Examiner recomputes a claimed statistic | Every number traceable to a CSV in `reports/tables/` |
| Deadline pressure (Sep 10 if full-time track) | Sensitivity split + calibration are droppable extras; core matrix + SHAP are the minimum viable dissertation |

---

## 13. Acceptance Criteria (definition of done)

- [ ] 9-cell matrix complete with all metrics at 4 threshold policies
- [ ] Statistical tests with p-values for all pairwise strategy comparisons
- [ ] Test-set numbers produced from a single untouched holdout
- [ ] SHAP analysis incl. cross-strategy stability
- [ ] `run_experiment.py` reproduces every reported number end-to-end
- [ ] Methodology chapter describes the design BEFORE results are written up

---

## Decisions needing sign-off

1. **Temporal holdout as primary evaluation** (recommended) vs CV-only (what the
   existing Gemini codebase does). Temporal is more defensible; CV-only is faster
   and already built.
2. **Rebuild in this workspace** (`F:\credit-risk-project - Copy`) implementing this
   design cleanly vs patching the existing `F:\credit-risk-project` code. Rebuild is
   recommended — the existing code's global preprocessing fit contradicts §4.
3. Drop the extras (sensitivity split, calibration) if the deadline is tight.
