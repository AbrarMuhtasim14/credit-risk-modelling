# Experiment Results Report — Clean-Room Build

**Date:** 2026-09-01 · **Dataset:** Home Credit `application_train.csv` (307,511 rows × 123 cols)
**Runtime:** 123.6 min (CV) + 12 min (OOT) + 1.5 min (analysis) on i3-1115G4 / 8 GB RAM
**Seed:** 42 (repeat seeds 42 / 1042 / 2042)

---

## 1. Methodology (leakage-free by construction)

| Stage | Decision |
|---|---|
| Temporal split | Train 2018–2019 (N=205,007, 8.12% default) → Test 2020 (N=102,504, 7.98% default) |
| Income winsorization | Cap $900,000 = 99.9th percentile computed on **training years only** (278 rows clipped) |
| Sentinel fix | DAYS_EMPLOYED=365243 → NaN + `DAYS_EMPLOYED_ANOM` flag (55,374 rows, 18.01%) |
| Collinearity pruning | 30 columns dropped (28 building _MEDI/_MODE + OBS_60/DEF_60 social circle). FLAG_EMP_PHONE **kept** (raw r=−0.9998 is a sentinel artifact; post-cleaning r≈0.002) |
| Feature space | 97 numeric + 16 categorical → 196 one-hot features |
| CV track | **3×5 repeated stratified K-fold on 2018–2019 only** (2020 never touched); preprocessor re-fit inside every fold; SMOTE applied to training fold only, computed once per fold and shared by the 3 SMOTE cells |
| OOT track | Single fit on full 2018–2019 window, evaluated once on 2020 |
| Hyperparameters | Frozen across imbalance strategies (LR: C=1.0 lbfgs; RF: 200 trees, depth 12; XGB: 200 trees, lr=0.05, depth 6, hist) |
| Metrics | PR-AUC (primary), ROC-AUC, recall, precision, F1, F2, cost/applicant (FN=10× FP) |
| Statistics | Paired t-test + Wilcoxon signed-rank + Cohen's d + Hedges' g + 95% CI over n=15 matched observations per comparison |

## 2. In-Time Results — 3×5 Repeated CV on 2018–2019

| Configuration | ROC-AUC | PR-AUC | Recall@0.5 | Precision@0.5 | F2 | Cost/App |
|---|---|---|---|---|---|---|
| **XGBoost (Baseline)** | **0.7622 ± 0.0046** | **0.2502 ± 0.0051** | 0.020 | 0.568 | 0.025 | 0.796 |
| XGBoost (Cost-Sensitive) | 0.7600 ± 0.0045 | 0.2470 ± 0.0050 | **0.643** | 0.178 | **0.423** | **0.530** |
| Random Forest (Cost-Sensitive) | 0.7474 ± 0.0052 | 0.2307 ± 0.0062 | 0.619 | 0.173 | 0.409 | 0.549 |
| Random Forest (Baseline) | 0.7455 ± 0.0050 | 0.2300 ± 0.0067 | 0.001 | 0.634 | 0.001 | 0.811 |
| Logistic Regression (Baseline) | 0.7475 ± 0.0048 | 0.2276 ± 0.0056 | 0.008 | 0.523 | 0.011 | 0.806 |
| Logistic Regression (Cost-Sensitive) | 0.7476 ± 0.0046 | 0.2260 ± 0.0054 | 0.680 | 0.161 | 0.413 | 0.548 |
| XGBoost (SMOTE) | 0.7438 ± 0.0048 | 0.2240 ± 0.0058 | 0.016 | 0.510 | 0.019 | 0.800 |
| Logistic Regression (SMOTE) | 0.7410 ± 0.0046 | 0.2198 ± 0.0046 | 0.663 | 0.160 | 0.407 | 0.556 |
| Random Forest (SMOTE) | 0.7135 ± 0.0051 | 0.1750 ± 0.0037 | 0.218 | 0.222 | 0.219 | 0.697 |

## 3. Out-Of-Time Results — Train 2018–19, Test 2020

| Configuration | ROC-AUC | PR-AUC | Recall@0.5 | Precision@0.5 |
|---|---|---|---|---|
| **XGBoost (Baseline)** | **0.7669** | **0.2513** | 0.022 | 0.573 |
| XGBoost (Cost-Sensitive) | 0.7661 | 0.2497 | 0.657 | 0.176 |
| Random Forest (Cost-Sensitive) | 0.7531 | 0.2338 | 0.637 | 0.173 |
| Logistic Regression (Cost-Sensitive) | 0.7522 | 0.2298 | 0.685 | 0.159 |
| Logistic Regression (Baseline) | 0.7521 | 0.2320 | 0.008 | 0.519 |
| Random Forest (Baseline) | 0.7506 | 0.2313 | 0.002 | 0.750 |
| XGBoost (SMOTE) | 0.7483 | 0.2278 | 0.014 | 0.542 |
| Logistic Regression (SMOTE) | 0.7459 | 0.2256 | 0.669 | 0.159 |
| Random Forest (SMOTE) | 0.7197 | 0.1752 | 0.222 | 0.217 |

**Generalization gap:** XGBoost Baseline CV→OOT ROC-AUC 0.7622 → 0.7669 (+0.005). No temporal degradation — the 2020 cohort was slightly easier to rank.

## 4. Statistical Significance (paired tests, n=15 matched observations)

15 of 16 comparisons significant at α=0.05; 13 at α=0.01.

| Comparison (ROC-AUC) | Δ mean | p (t-test) | p (Wilcoxon) | Cohen's d |
|---|---|---|---|---|
| XGBoost vs Logistic Regression (baseline) | +0.0147 | 1.0e-13 | 6.1e-05 | 7.27 (Large) |
| XGBoost vs Random Forest (baseline) | +0.0166 | 1.3e-17 | 6.1e-05 | 13.86 (Large) |
| LR vs RF (baseline) | +0.0020 | 6.4e-05 | 6.1e-05 | 1.45 (Large) |
| XGBoost SMOTE vs XGBoost Baseline | −0.0184 | 1.6e-13 | 6.1e-05 | −7.03 (Large) |
| RF SMOTE vs RF Baseline | −0.0320 | 2.2e-18 | 6.1e-05 | −15.76 (Large) |
| LR SMOTE vs LR Baseline | −0.0065 | 7.4e-12 | 6.1e-05 | −5.31 (Large) |
| RF Cost-Sensitive vs RF Baseline | +0.0018 | 2.4e-07 | 6.1e-05 | 2.39 (Large) |
| XGBoost Cost-Sensitive vs XGBoost Baseline | −0.0022 | 3.2e-05 | 1.8e-04 | −1.55 (Large) |
| LR Cost-Sensitive vs LR Baseline | +0.00005 | 0.689 (n.s.) | 0.978 (n.s.) | 0.11 (Negligible) |

Recall impact of cost-sensitive learning (all three models): Δrecall ≈ +0.62 to +0.67, p < 1e-27, Cohen's d > 70.
Financial cost: XGBoost cost-sensitive reduces expected cost/applicant by 0.266 (p=1.5e-23).

## 5. SHAP Explainability (TreeSHAP, N=2,000 stratified sample)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | EXT_SOURCES_MEAN (bureau score average) | 0.438 |
| 2 | PAYMENT_RATE (annuity/credit) | 0.128 |
| 3 | GOODS_CREDIT_RATIO | 0.079 |
| 4 | EXT_SOURCE_2_3_PROD (bureau interaction) | 0.077 |
| 5 | EXT_SOURCES_MAX | 0.073 |

TreeSHAP vs LinearSHAP top-30 overlap: **16/30** — the linear model agrees on the dominant bureau-score signal but misses tree-captured interactions.
Three local case studies (TP / TN / FP) exported as waterfall plots + factor tables (GDPR Art. 22 adverse-action style).

## 6. Conclusions

1. **Champion: XGBoost (Baseline)** — best CV PR-AUC (0.2502) and best OOT ROC-AUC (0.7669).
2. **Cost-sensitive learning is the practical deployment choice**: recall jumps from ~2% to ~64–66% at negligible ranking cost (−0.002 ROC-AUC for XGBoost), and expected cost/applicant drops by 33%.
3. **SMOTE degrades every model family** (statistically significant, large effect sizes). Worst for Random Forest (−0.032 ROC-AUC). K-NN interpolation in a 196-dim space with 8% minority mass manufactures noise rather than signal.
4. **Logistic Regression is competitive** (within 0.015 ROC-AUC of XGBoost) — supports reporting it as an interpretable baseline and for the GDPR explanation pathway.
5. **No temporal degradation**: OOT performance matches or exceeds in-time CV, supporting deployment robustness on future cohorts.

## 7. Reproducibility Manifest

| Artifact | Path |
|---|---|
| Per-fold CV results (135 rows) | `reports/tables/cv_fold_results.csv` |
| CV summary | `reports/tables/cv_summary.csv` |
| OOT results | `reports/tables/oot_results.csv` |
| Statistical tests (16 comparisons) | `reports/tables/statistical_significance.csv` |
| OOT probability archive | `models/oot_probas.parquet` |
| Fitted OOT preprocessor | `models/oot_preprocessor.joblib` |
| Engineered splits | `data/processed/engineered_train_2018_2019.parquet`, `engineered_test_2020.parquet` |
| Figures (8) | `reports/figures/*.png` (300 DPI) |
| SHAP tables (4) | `reports/tables/shap_*.csv` |
| Run logs | `logs/experiment.log`, `logs/analysis.log` |

**To reproduce:** `py run_experiment.py` then `py run_analysis.py` (≈ 2.2 hours on this hardware).
