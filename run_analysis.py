"""
Downstream analysis: statistical tests, evaluation plots, SHAP explainability.
Run AFTER run_experiment.py completes.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, LOGS_DIR, MODELS_DIR, TABLES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "analysis.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Statistical significance battery (3x5 repeated CV -> n=15 pairs)
    logger.info("=" * 70)
    logger.info("STEP 1: Statistical hypothesis testing")
    logger.info("=" * 70)
    from src.utils.stats_testing import run_all_tests
    df_stats = run_all_tests(
        cv_path=TABLES_DIR / "cv_fold_results.csv",
        out_path=TABLES_DIR / "statistical_significance.csv",
        figure_path=FIGURES_DIR / "statistical_significance_heatmap.png",
    )
    logger.info("Significant at alpha=0.05: %d / %d comparisons",
                int(df_stats["significant_alpha_0.05"].sum()), len(df_stats))

    # 2. Evaluation plots from OOT predictions
    logger.info("=" * 70)
    logger.info("STEP 2: Evaluation plots (OOT 2020 predictions)")
    logger.info("=" * 70)
    from src.models.plots import plot_confusion_grid, plot_cost_curves, plot_pr_curves, plot_roc_curves
    probas_path = MODELS_DIR / "oot_probas.parquet"
    plot_roc_curves(probas_path, FIGURES_DIR / "roc_curves_oot.png")
    plot_pr_curves(probas_path, FIGURES_DIR / "pr_curves_oot.png")
    plot_confusion_grid(probas_path, FIGURES_DIR / "confusion_matrices_oot.png")
    plot_cost_curves(probas_path, FIGURES_DIR / "cost_curves_oot.png")

    # 3. SHAP explainability (TreeSHAP + LinearSHAP)
    logger.info("=" * 70)
    logger.info("STEP 3: SHAP explainability")
    logger.info("=" * 70)
    from src.explainability.shap_engine import run_shap_pipeline
    run_shap_pipeline()

    logger.info("Analysis pipeline complete.")


if __name__ == "__main__":
    main()
