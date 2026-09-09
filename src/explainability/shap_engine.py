"""
SHAP explainability engine.

Two explainers, honestly implemented:
1. TreeSHAP on the champion XGBoost (exact for tree ensembles).
2. LinearSHAP on the Logistic Regression baseline (linear-vs-tree comparison).

Attributions are computed on a stratified sample of N=2,000 applicants drawn
from the 2018-2019 training window (the models are re-fit on that window, so
no OOT data is involved in explanation).

Outputs:
- Global mean |SHAP| importance table (top 30) + bar chart
- Beeswarm summary plot
- Dependence panels (EXT_SOURCES_MEAN, PAYMENT_RATE, CREDIT_INCOME_PERCENT)
- Three local case studies (TP / TN / FP) as waterfall plots + summary table
"""

import gc
import logging
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_COST_RATIO,
    FIGURE_DPI,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    SHAP_SAMPLE_SIZE,
    TABLES_DIR,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def load_training_matrix():
    """Load engineered 2018-2019 window + fitted OOT preprocessor, transform."""
    df_train = pd.read_parquet(PROCESSED_DIR / "engineered_train_2018_2019.parquet")
    bundle = joblib.load(MODELS_DIR / "oot_preprocessor.joblib")
    pre, num_cols, cat_cols = bundle["preprocessor"], bundle["numeric_cols"], bundle["categorical_cols"]
    X = pre.transform(df_train[num_cols + cat_cols]).astype(np.float32)
    y = df_train[TARGET_COL].to_numpy(dtype=np.int8)
    ids = df_train["SK_ID_CURR"].to_numpy()
    with open(MODELS_DIR / "oot_feature_names.json", "r", encoding="utf-8") as f:
        import json
        feature_names = json.load(f)
    logger.info("SHAP data prepared: X=%s, default rate %.2f%%", X.shape, 100 * y.mean())
    return X, y, ids, feature_names


def stratified_sample(X, y, ids, n=SHAP_SAMPLE_SIZE, seed=RANDOM_SEED):
    from sklearn.model_selection import train_test_split
    _, X_s, _, y_s, _, id_s = train_test_split(
        X, y, ids, test_size=n, stratify=y, random_state=seed
    )
    logger.info("Stratified SHAP sample: %d rows (%.1f%% default)", n, 100 * y_s.mean())
    return X_s, y_s, id_s


def run_shap_pipeline() -> None:
    import shap
    from src.models.factory import make_model

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR = TABLES_DIR.parent / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, y, ids, feature_names = load_training_matrix()
    X_s, y_s, id_s = stratified_sample(X, y, ids)

    spw = float(np.sum(y == 0)) / max(float(np.sum(y == 1)), 1.0)

    # ------------------------------------------------------------------
    # 1. TreeSHAP on champion XGBoost (baseline hyperparameters)
    # ------------------------------------------------------------------
    logger.info("Fitting champion XGBoost on full 2018-2019 window for SHAP...")
    xgb_model = make_model("xgboost", "baseline", spw, RANDOM_SEED)
    xgb_model.fit(X, y)

    logger.info("Computing TreeSHAP values on %d samples...", len(X_s))
    tree_explainer = shap.TreeExplainer(xgb_model)
    shap_tree = tree_explainer.shap_values(X_s)
    base_tree = float(np.asarray(tree_explainer.expected_value).ravel()[0])

    mean_abs_tree = np.mean(np.abs(shap_tree), axis=0)
    order = np.argsort(mean_abs_tree)[::-1]
    df_imp = pd.DataFrame({
        "rank": np.arange(1, len(feature_names) + 1),
        "feature": [feature_names[i] for i in order],
        "display_name": [display_name(feature_names[i]) for i in order],
        "mean_abs_shap": mean_abs_tree[order],
    })
    df_imp.head(30).to_csv(TABLES_DIR / "shap_importance_top30.csv", index=False)
    logger.info("Top-5 TreeSHAP features: %s", df_imp.head(5)["feature"].tolist())

    _plot_importance_bar(df_imp, FIGURES_DIR / "shap_importance.png")
    _plot_beeswarm(shap_tree, X_s, feature_names, base_tree, FIGURES_DIR / "shap_beeswarm.png")
    _plot_dependence(shap_tree, X_s, feature_names, FIGURES_DIR)
    _plot_case_studies(xgb_model, shap_tree, base_tree, X_s, y_s, id_s, feature_names,
                       FIGURES_DIR / "shap_case_studies.png", TABLES_DIR / "shap_case_studies.csv")

    del xgb_model, tree_explainer, shap_tree
    gc.collect()

    # ------------------------------------------------------------------
    # 2. LinearSHAP on Logistic Regression baseline (comparison)
    # ------------------------------------------------------------------
    logger.info("Fitting Logistic Regression for LinearSHAP comparison...")
    lr_model = make_model("logistic_regression", "baseline", spw, RANDOM_SEED)
    lr_model.fit(X, y)

    import scipy.sparse as sp
    background = sp.csr_matrix(X[: min(2000, len(X))])
    linear_explainer = shap.LinearExplainer(lr_model, background)
    shap_linear = linear_explainer.shap_values(X_s)

    mean_abs_lin = np.mean(np.abs(shap_linear), axis=0)
    order_lin = np.argsort(mean_abs_lin)[::-1]
    df_imp_lin = pd.DataFrame({
        "rank": np.arange(1, len(feature_names) + 1),
        "feature": [feature_names[i] for i in order_lin],
        "display_name": [display_name(feature_names[i]) for i in order_lin],
        "mean_abs_shap": mean_abs_lin[order_lin],
    })
    df_imp_lin.head(30).to_csv(TABLES_DIR / "shap_importance_top30_logistic.csv", index=False)

    _plot_importance_bar(df_imp_lin, FIGURES_DIR / "shap_importance_logistic.png",
                         title="Global Feature Importance (LinearSHAP, Logistic Regression)")

    # Rank agreement between tree and linear explanations (top-30 overlap)
    top30_tree = set(df_imp.head(30)["feature"])
    top30_lin = set(df_imp_lin.head(30)["feature"])
    overlap = len(top30_tree & top30_lin)
    logger.info("Top-30 importance overlap (tree vs linear): %d / 30", overlap)
    with open(TABLES_DIR / "shap_tree_vs_linear_overlap.txt", "w", encoding="utf-8") as f:
        f.write(f"Top-30 feature overlap between TreeSHAP (XGBoost) and LinearSHAP (LR): {overlap}/30\n")

    logger.info("SHAP pipeline complete.")


def _plot_importance_bar(df_imp: pd.DataFrame, save_path, top_k: int = 25,
                         title: str = "Global Feature Importance (TreeSHAP, XGBoost Champion)") -> None:
    top = df_imp.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 9), dpi=FIGURE_DPI)
    ax.barh(top["display_name"], top["mean_abs_shap"], color="#1f77b4", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value| (impact on default log-odds)", fontweight="bold")
    ax.set_title(f"{title}\nStratified sample N={SHAP_SAMPLE_SIZE:,}", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_beeswarm(shap_values, X_sample, feature_names, base_value, save_path) -> None:
    import shap
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(X_sample), base_value),
        data=X_sample,
        feature_names=[display_name(f) for f in feature_names],
    )
    plt.figure(figsize=(12, 9), dpi=FIGURE_DPI)
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("SHAP Beeswarm: feature value vs impact on default risk", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close("all")


def _plot_dependence(shap_values, X_sample, feature_names, figures_dir) -> None:
    targets = ["EXT_SOURCES_MEAN", "PAYMENT_RATE", "CREDIT_INCOME_PERCENT"]
    idx = {f: feature_names.index(f) for f in targets if f in feature_names}

    fig, axes = plt.subplots(1, len(idx), figsize=(6 * len(idx), 5), dpi=FIGURE_DPI)
    if len(idx) == 1:
        axes = [axes]
    for ax, (feat, i) in zip(axes, idx.items()):
        ax.scatter(X_sample[:, i], shap_values[:, i], alpha=0.4, s=15, color="#1976d2")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(display_name(feat), fontweight="bold")
        ax.set_ylabel("SHAP value", fontweight="bold")
        ax.set_title(f"Dependence: {feat}", fontweight="bold")
        ax.grid(alpha=0.3)
    fig.suptitle("SHAP Dependence Panels (XGBoost, OOT-clean training window)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(figures_dir / "shap_dependence.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _draw_case_panel(ax, title, prob, actual, base_value, shap_row, feature_names, top_k=8):
    """Clean horizontal-bar panel of one applicant's top SHAP contributions."""
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
    ax.set_title(f"{title}\nP(default) = {prob:.3f}   |   actual outcome: {outcome}",
                 fontsize=13, fontweight="bold", pad=10, loc="left")


def _plot_case_studies(model, shap_values, base_value, X_s, y_s, id_s, feature_names,
                       fig_path, table_path) -> None:
    proba = model.predict_proba(X_s)[:, 1]

    tp = np.where((y_s == 1) & (proba >= 0.5))[0]
    tn = np.where((y_s == 0) & (proba < 0.1))[0]
    fp = np.where((y_s == 0) & (proba >= 0.45) & (proba <= 0.65))[0]

    idx_tp = tp[np.argmax(proba[tp])] if len(tp) else int(np.argmax(proba))
    idx_tn = tn[np.argmin(proba[tn])] if len(tn) else int(np.argmin(proba))
    idx_fp = fp[np.argmax(proba[fp])] if len(fp) else int(np.where(y_s == 0)[0][0])

    cases = [
        (idx_tp, f"Case 1: True Positive — applicant {int(id_s[idx_tp])} (defaulter correctly flagged)"),
        (idx_tn, f"Case 2: True Negative — applicant {int(id_s[idx_tn])} (solvent borrower approved)"),
        (idx_fp, f"Case 3: False Positive — applicant {int(id_s[idx_fp])} (borderline false alarm)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 14), dpi=FIGURE_DPI,
                             constrained_layout=True)
    records = []
    for ax, (idx, title) in zip(axes, cases):
        _draw_case_panel(ax, title, float(proba[idx]), int(y_s[idx]),
                         base_value, shap_values[idx], feature_names)

        top5 = np.argsort(np.abs(shap_values[idx]))[::-1][:5]
        for rank, fi in enumerate(top5, start=1):
            records.append({
                "case": title.split(":")[0],
                "applicant_id": int(id_s[idx]),
                "predicted_default_prob": float(proba[idx]),
                "actual_default": int(y_s[idx]),
                "factor_rank": rank,
                "feature": feature_names[fi],
                "display_name": display_name(feature_names[fi]),
                "feature_value": float(X_s[idx, fi]),
                "shap_impact": float(shap_values[idx, fi]),
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

    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(records).to_csv(table_path, index=False)
    logger.info("Case studies saved: %s, %s", fig_path, table_path)
