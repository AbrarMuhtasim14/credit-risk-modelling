"""
Statistical significance testing across the 9-cell matrix.

Paired comparisons over matched CV observations. Because we use 3x5 repeated
stratified K-fold, each configuration yields 15 observations (3 repeats x
5 folds), paired by (repeat, fold) key. This gives the Wilcoxon signed-rank
test enough resolution to reach significance (unlike n=5, where the minimum
two-sided p-value is 0.0625).

Tests per comparison:
- Paired Student's t-test (parametric)
- Wilcoxon signed-rank (non-parametric)
- Cohen's d and Hedges' g effect sizes with 95% CI of the mean difference
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


def paired_comparison(
    df_cv: pd.DataFrame,
    config_a: str,
    config_b: str,
    metric: str,
    category: str,
) -> Dict:
    """Compare two configurations on matched (repeat, fold) observations."""
    df_a = df_cv[df_cv["display_name"] == config_a].sort_values(["repeat", "fold"])
    df_b = df_cv[df_cv["display_name"] == config_b].sort_values(["repeat", "fold"])

    if len(df_a) == 0 or len(df_b) == 0:
        raise ValueError(f"Config not found: {config_a} / {config_b}")

    # Align on identical (repeat, fold) keys
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
        # all differences zero -> cannot rank
        w_stat, p_w = np.nan, 1.0

    # Effect sizes
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


def run_all_tests(cv_path, out_path, figure_path=None) -> pd.DataFrame:
    """Run the full battery of paired comparisons and save results."""
    from src.config import FIGURE_DPI

    df_cv = pd.read_csv(cv_path)

    comparisons = [
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

    records = [paired_comparison(df_cv, a, b, m, cat) for cat, a, b, m in comparisons]
    df_out = pd.DataFrame(records)
    df_out.to_csv(out_path, index=False)
    logger.info("Saved statistical tests to %s (%d comparisons)", out_path, len(df_out))

    if figure_path is not None:
        _plot_heatmap(df_out, figure_path)

    return df_out


def _plot_heatmap(df_stats: pd.DataFrame, save_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.config import FIGURE_DPI

    df_plot = df_stats.copy()
    df_plot["label"] = df_plot["comparison"] + "\n[" + df_plot["metric"] + "]"

    p_vals = df_plot["p_value_ttest"].to_numpy().reshape(-1, 1)
    annot = []
    for p, d in zip(p_vals.flatten(), df_plot["cohen_d"].to_numpy()):
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        annot.append(f"p={p:.2e}\nd={d:+.2f} {sig}")

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
        "Statistical Significance Matrix: Paired Tests over 3x5 Repeated CV (n=15)\n"
        "*** p<0.001 | ** p<0.01 | * p<0.05 | n.s. not significant",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved significance heatmap to %s", save_path)
