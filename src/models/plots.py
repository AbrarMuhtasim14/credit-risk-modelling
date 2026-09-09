"""
Publication-quality evaluation plots, generated from the OOT (2020) predictions.

Using OOT predictions for the comparison curves is the most defensible choice:
these probabilities come from models that never saw the 2020 rows.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGY_STYLE = {"baseline": "-", "cost_sensitive": "--", "smote": ":"}
MODEL_COLOR = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "XGBoost": "#d62728"}


def _style(label: str):
    for fam, color in MODEL_COLOR.items():
        if label.startswith(fam):
            break
    for strat, ls in STRATEGY_STYLE.items():
        if strat in label.lower().replace("-", "_") or (
            strat == "baseline" and "Cost" not in label and "SMOTE" not in label
        ):
            break
    return color, ls


def load_oot_probas(probas_path, threshold: float = 0.5):
    df = pd.read_parquet(probas_path)
    from src.config import ID_COL, TARGET_COL
    y = df[TARGET_COL].to_numpy()
    cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    return df, y, cols


def plot_roc_curves(probas_path, save_path, dpi: int = 300) -> None:
    from sklearn.metrics import auc
    df, y, cols = load_oot_probas(probas_path)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    for label in cols:
        fpr, tpr, _ = roc_curve(y, df[label].to_numpy())
        a = auc(fpr, tpr)
        color, ls = _style(label)
        ax.plot(fpr, tpr, ls, color=color, linewidth=1.8, label=f"{label} (AUC={a:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("ROC Curves - Out-Of-Time Test (2020, N=102,504)", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curves to %s", save_path)


def plot_pr_curves(probas_path, save_path, dpi: int = 300) -> None:
    from sklearn.metrics import auc
    df, y, cols = load_oot_probas(probas_path)
    base_rate = float(np.mean(y))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    for label in cols:
        precision, recall, _ = precision_recall_curve(y, df[label].to_numpy())
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
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PR curves to %s", save_path)


def plot_confusion_grid(probas_path, save_path, threshold: float = 0.5, dpi: int = 300) -> None:
    df, y, cols = load_oot_probas(probas_path)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), dpi=dpi)
    for ax, label in zip(axes.ravel(), cols):
        pred = (df[label].to_numpy() >= threshold).astype(int)
        cm = confusion_matrix(y, pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Confusion Matrices @ threshold={threshold} - OOT Test (2020)", fontweight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix grid to %s", save_path)


def plot_cost_curves(probas_path, save_path, fn_cost: float = 10.0, dpi: int = 300) -> None:
    """Expected cost per applicant as a function of the decision threshold."""
    df, y, cols = load_oot_probas(probas_path)
    thresholds = np.linspace(0.05, 0.95, 91)

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=dpi)
    for label in cols:
        proba = df[label].to_numpy()
        costs = []
        for t in thresholds:
            pred = (proba >= t).astype(int)
            fn = int(np.sum((y == 1) & (pred == 0)))
            fp = int(np.sum((y == 0) & (pred == 1)))
            costs.append((fn * fn_cost + fp) / len(y))
        color, ls = _style(label)
        ax.plot(thresholds, costs, ls, color=color, linewidth=1.6, label=label)
    ax.set_xlabel("Decision Threshold", fontweight="bold")
    ax.set_ylabel(f"Expected Cost per Applicant (FN cost = {fn_cost:.0f}x FP)", fontweight="bold")
    ax.set_title("Financial Cost vs Threshold - OOT Test (2020)", fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cost curves to %s", save_path)
