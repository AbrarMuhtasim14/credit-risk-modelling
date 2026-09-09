"""
Evaluation metrics for imbalanced credit risk classification.

Accuracy is deliberately NOT a headline metric: at ~8% base rate a trivial
all-negative classifier scores ~92% accuracy. Primary: PR-AUC. Secondary: ROC-AUC.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def pr_auc_exact(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    True area under the precision-recall curve via trapezoidal integration.

    Note: sklearn's precision_recall_curve returns recall in DECREASING order
    (last point is recall=0, precision=1), so the raw trapezoid integral is
    negative; we take the absolute value.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(abs(np.trapezoid(precision, recall)))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    fn_cost: float = 10.0,
) -> Dict[str, float]:
    """Full metric battery for one set of predictions."""
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
