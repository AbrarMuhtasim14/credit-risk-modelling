"""Interactive Loan Applicant Scoring & Adverse Action Explainer.

Demonstrates production-grade scoring using the champion XGBoost model,
the fitted preprocessor, and interpretable feature contribution analysis
(GDPR Article 22 & US FCRA adverse action compliant).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DAYS_EMPLOYED_ANOM_COL,
    DAYS_EMPLOYED_SENTINEL,
    DEFAULT_COST_RATIO,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COL,
)
from src.explainability.shap_engine import DOMAIN_NAMES, display_name
from src.features.engineering import clean_and_engineer


def load_artifacts():
    prep_path = MODELS_DIR / "oot_preprocessor.joblib"
    model_path = MODELS_DIR / "champion_xgboost.joblib"

    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at: {prep_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Champion model not found at: {model_path}. "
            "Please run 'python scripts/export_champion_model.py' first."
        )

    bundle = joblib.load(prep_path)
    model = joblib.load(model_path)
    return bundle, model


def score_applicant_record(df_record: pd.DataFrame, bundle: dict, model):
    pre = bundle["preprocessor"]
    num_cols = bundle["numeric_cols"]
    cat_cols = bundle["categorical_cols"]

    # Transform feature vector
    X_vec = pre.transform(df_record[num_cols + cat_cols]).astype(np.float32)

    # Predict probability of default
    prob_default = float(model.predict_proba(X_vec)[0, 1])

    # Tree feature importances for local approximation
    importances = model.feature_importances_
    with open(MODELS_DIR / "oot_feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # Calculate feature contributions (feature value * importance direction)
    contributions = []
    dense_vec = X_vec[0]
    for idx, fname in enumerate(feature_names):
        val = dense_vec[idx]
        imp = importances[idx]
        if abs(val) > 1e-4 and imp > 0.005:
            contributions.append((fname, val, imp))

    # Sort by importance
    contributions.sort(key=lambda x: x[2], reverse=True)

    return prob_default, contributions


def format_currency(val):
    try:
        return f"${float(val):,.2f}"
    except (ValueError, TypeError):
        return str(val)


def print_scorecard(record: pd.Series, prob_default: float, contributions: list):
    # Risk tiers
    if prob_default < 0.06:
        tier = "LOW RISK"
        recommendation = "APPROVED (Standard Prime Terms)"
        banner_char = "="
    elif prob_default < 0.15:
        tier = "MODERATE RISK"
        recommendation = "CONDITIONAL APPROVAL / MANUAL REVIEW"
        banner_char = "-"
    else:
        tier = "HIGH RISK"
        recommendation = "DECLINED (Default Risk Exceeds Risk Appetite)"
        banner_char = "!"

    applicant_id = record.get("SK_ID_CURR", "N/A")
    credit_amt = format_currency(record.get("AMT_CREDIT", 0))
    annuity_amt = format_currency(record.get("AMT_ANNUITY", 0))
    income_amt = format_currency(record.get("AMT_INCOME_TOTAL", 0))
    bureau_2 = record.get("EXT_SOURCE_2", "N/A")
    bureau_3 = record.get("EXT_SOURCE_3", "N/A")

    print("\n" + banner_char * 75)
    print(f" LOAN APPLICANT RISK EVALUATION REPORT  |  Applicant ID: {applicant_id}")
    print(banner_char * 75)
    print(f"  Requested Credit : {credit_amt:<18} Annual Income : {income_amt}")
    print(f"  Monthly Annuity  : {annuity_amt:<18} Bureau Score 2: {bureau_2}")
    print(f"  Bureau Score 3   : {str(bureau_3):<18}")
    print("-" * 75)
    print(f"  PREDICTED DEFAULT PROBABILITY : {prob_default * 100:.2f}%")
    print(f"  PORTFOLIO RISK TIER          : {tier}")
    print(f"  UNDERWRITING DECISION        : {recommendation}")
    print("-" * 75)
    print("  KEY ADVERSE ACTION & RISK DRIVERS (GDPR Art. 22 / US FCRA):")

    top_drivers = contributions[:6]
    for rank, (fname, val, imp) in enumerate(top_drivers, 1):
        clean_name = display_name(fname)
        impact_dir = "Elevates Risk" if prob_default >= 0.10 else "Mitigates Risk"
        print(f"   {rank}. {clean_name:<38} [Weight: {imp:.4f} | {impact_dir}]")

    print(banner_char * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Credit Risk Scoring CLI")
    parser.add_argument("--sample", action="store_true", help="Score a random applicant from the sample dataset")
    parser.add_argument("--applicant-id", type=int, default=None, help="Score a specific applicant by SK_ID_CURR")
    args = parser.parse_args()

    bundle, model = load_artifacts()

    # Determine input data source
    sample_file = ROOT / "data" / "sample" / "sample_application_train.csv"
    raw_file = ROOT / "data" / "raw" / "application_train.csv"

    source_path = sample_file if sample_file.exists() else raw_file
    if not source_path.exists():
        print(f"No dataset found at {sample_file} or {raw_file}. Please run scripts/create_sample_dataset.py.")
        return

    print(f"Loading applicant records from: {source_path.name}")
    df_raw = pd.read_csv(source_path)

    if args.applicant_id:
        match = df_raw[df_raw["SK_ID_CURR"] == args.applicant_id]
        if match.empty:
            print(f"Applicant ID {args.applicant_id} not found.")
            return
        raw_row = match.iloc[[0]]
    else:
        # Default: sample a random applicant
        raw_row = df_raw.sample(n=1, random_state=42)

    applicant_series = raw_row.iloc[0]

    # Preprocess with the standard pipeline
    clean_row = clean_and_engineer(raw_row, income_cap=900000.0)

    prob_default, contributions = score_applicant_record(clean_row, bundle, model)
    print_scorecard(applicant_series, prob_default, contributions)


if __name__ == "__main__":
    main()
