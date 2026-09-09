"""Unit and regression tests for Credit Risk Modelling pipeline (compatible with unittest and pytest)."""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
from src.config import (
    DAYS_EMPLOYED_ANOM_COL,
    DAYS_EMPLOYED_SENTINEL,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COL,
)
from src.features.engineering import clean_and_engineer


class TestCreditRiskPipeline(unittest.TestCase):

    def test_sample_dataset_integrity(self):
        """Verify that sample dataset matches required schema and has valid target distribution."""
        sample_path = ROOT / "data" / "sample" / "sample_application_train.csv"
        self.assertTrue(sample_path.exists(), f"Sample dataset missing at {sample_path}")

        df = pd.read_csv(sample_path)
        self.assertEqual(len(df), 1000, f"Expected 1,000 rows, got {len(df)}")
        self.assertIn(TARGET_COL, df.columns, f"Target column '{TARGET_COL}' missing")
        self.assertIn("SK_ID_CURR", df.columns, "ID column missing")
        self.assertIn("application_date", df.columns, "Application date missing")

        # Default rate should be ~8% (between 6% and 10%)
        default_rate = df[TARGET_COL].mean()
        self.assertTrue(0.06 <= default_rate <= 0.10, f"Unexpected default rate: {default_rate:.4f}")

    def test_sentinel_and_income_winsorization(self):
        """Verify DAYS_EMPLOYED anomaly handling and income winsorization logic."""
        dummy = pd.DataFrame({
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "DAYS_EMPLOYED": [DAYS_EMPLOYED_SENTINEL, -1500],
            "AMT_INCOME_TOTAL": [1_500_000.0, 100_000.0],
            "AMT_CREDIT": [200_000.0, 50_000.0],
            "AMT_ANNUITY": [15_000.0, 4_000.0],
            "AMT_GOODS_PRICE": [180_000.0, 45_000.0],
            "DAYS_BIRTH": [-12000, -15000],
            "DAYS_REGISTRATION": [-2000, -3000],
            "DAYS_ID_PUBLISH": [-1000, -2000],
            "DAYS_LAST_PHONE_CHANGE": [-50, -100],
            "OWN_CAR_AGE": [np.nan, 5],
            "CNT_FAM_MEMBERS": [1, 2],
            "FLAG_DOCUMENT_3": [1, 1],
            "ORGANIZATION_TYPE": ["Business Entity Type 3", "XNA"],
            "CODE_GENDER": ["M", "F"],
            "EXT_SOURCE_1": [0.5, np.nan],
            "EXT_SOURCE_2": [0.6, 0.4],
            "EXT_SOURCE_3": [0.7, 0.3],
        })

        cleaned = clean_and_engineer(dummy, income_cap=900_000.0)

        # 1. Sentinel check
        self.assertEqual(cleaned.loc[0, DAYS_EMPLOYED_ANOM_COL], 1)
        self.assertTrue(pd.isna(cleaned.loc[0, "DAYS_EMPLOYED"]))
        self.assertEqual(cleaned.loc[1, DAYS_EMPLOYED_ANOM_COL], 0)
        self.assertEqual(cleaned.loc[1, "DAYS_EMPLOYED"], -1500)

        # 2. Income winsorization check
        self.assertEqual(cleaned.loc[0, "AMT_INCOME_TOTAL"], 900_000.0)
        self.assertEqual(cleaned.loc[1, "AMT_INCOME_TOTAL"], 100_000.0)

        # 3. Domain ratio checks
        self.assertIn("CREDIT_INCOME_PERCENT", cleaned.columns)
        self.assertIn("PAYMENT_RATE", cleaned.columns)

    def test_champion_model_inference(self):
        """Verify that serialized champion XGBoost model and preprocessor load and produce valid probabilities."""
        prep_path = MODELS_DIR / "oot_preprocessor.joblib"
        model_path = MODELS_DIR / "champion_xgboost.joblib"

        if not prep_path.exists() or not model_path.exists():
            self.skipTest("Model artifacts not found, skipping inference test.")

        bundle = joblib.load(prep_path)
        model = joblib.load(model_path)

        sample_path = ROOT / "data" / "sample" / "sample_application_train.csv"
        df_sample = pd.read_csv(sample_path, nrows=10)
        cleaned = clean_and_engineer(df_sample, income_cap=900_000.0)

        num_cols = bundle["numeric_cols"]
        cat_cols = bundle["categorical_cols"]
        pre = bundle["preprocessor"]

        X = pre.transform(cleaned[num_cols + cat_cols]).astype(np.float32)
        self.assertEqual(X.shape[0], 10)
        self.assertIn(X.shape[1], [194, 196])

        probas = model.predict_proba(X)
        self.assertEqual(probas.shape, (10, 2))
        self.assertTrue(np.all((probas >= 0.0) & (probas <= 1.0)))
        self.assertTrue(np.allclose(probas.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
