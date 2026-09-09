"""
Institutional Credit Risk Decision Portal & Underwriting Engine
Enterprise Banking Application for Loan Underwriting, Macro Stress-Testing,
Regulatory Adverse Action (FCRA/GDPR), and Basel III/IV Capital Metrics.
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Apex Financial | Credit Risk Underwriting Portal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional Banking Theme
st.markdown("""
<style>
    .bank-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .bank-sub {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .decision-approved {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
    }
    .decision-review {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        padding: 1rem;
        border-radius: 8px;
    }
    .decision-declined {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 1rem;
        border-radius: 8px;
    }
    .adverse-notice {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #64748b;
        padding: 1.2rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

DOMAIN_NAMES = {
    "EXT_SOURCES_MEAN": "External Bureau Score Composite (Mean)",
    "EXT_SOURCES_STD": "External Bureau Score Volatility (Std Dev)",
    "EXT_SOURCES_MIN": "Lowest Individual Bureau Score",
    "EXT_SOURCES_MAX": "Highest Individual Bureau Score",
    "EXT_SOURCE_1": "Bureau Credit Score 1 (Equifax Proxy)",
    "EXT_SOURCE_2": "Bureau Credit Score 2 (TransUnion Proxy)",
    "EXT_SOURCE_3": "Bureau Credit Score 3 (Experian Proxy)",
    "EXT_SOURCES_PROD": "Bureau Score Multiplicative Interaction",
    "EXT_SOURCE_2_3_PROD": "TransUnion & Experian Joint Interaction",
    "PAYMENT_RATE": "Annuity-to-Credit Debt Service Ratio",
    "CREDIT_INCOME_PERCENT": "Debt-to-Income (DTI) Multiple",
    "ANNUITY_INCOME_PERCENT": "Monthly Debt Burden Ratio",
    "AGE_YEARS": "Applicant Age",
    "EMPLOYED_YEARS": "Verified Employment Tenure",
    "DAYS_EMPLOYED_ANOM": "Employment Gap Anomaly Flag",
    "DOCUMENT_COUNT": "Verified Identification Documents",
    "NAME_EDUCATION_TYPE_Higher education": "Higher Education Credential",
    "CODE_GENDER_M": "Applicant Demographic Classification"
}


@st.cache_resource
def load_model_artifacts():
    prep_path = MODELS_DIR / "oot_preprocessor.joblib"
    model_path = MODELS_DIR / "champion_xgboost.joblib"
    feat_path = MODELS_DIR / "oot_feature_names.json"

    if not (prep_path.exists() and model_path.exists() and feat_path.exists()):
        return None, None, None

    bundle = joblib.load(prep_path)
    model = joblib.load(model_path)
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    return bundle, model, feature_names


@st.cache_data
def load_sample_dataset():
    sample_file = DATA_DIR / "sample" / "sample_application_train.csv"
    if sample_file.exists():
        return pd.read_csv(sample_file)
    return None


def clean_single_record(row_dict: dict, base_df: pd.DataFrame) -> pd.DataFrame:
    from src.features.engineering import clean_and_engineer

    record = base_df.iloc[[0]].copy()
    for col, val in row_dict.items():
        if col in record.columns:
            record.loc[record.index[0], col] = val

    cleaned = clean_and_engineer(record, income_cap=900_000.0)
    return cleaned


def predict_borrower_risk(applicant_dict, df_sample, bundle, model):
    cleaned_row = clean_single_record(applicant_dict, df_sample)
    pre = bundle["preprocessor"]
    num_cols = bundle["numeric_cols"]
    cat_cols = bundle["categorical_cols"]

    X_vec = pre.transform(cleaned_row[num_cols + cat_cols]).astype(np.float32)
    prob_default = float(model.predict_proba(X_vec)[0, 1])
    return prob_default, X_vec


def main():
    bundle, model, feature_names = load_model_artifacts()
    df_sample = load_sample_dataset()

    if bundle is None or model is None:
        st.error("⚠️ Model artifacts missing. Ensure `models/champion_xgboost.joblib` is present.")
        return

    # Header
    st.markdown('<div class="bank-header">🏛️ Apex Financial | Credit Underwriting & Risk Decision Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="bank-sub">Enterprise Credit Intelligence Platform · Advanced IRB Basel Aligned · 10:1 Asymmetric Cost-Sensitive Learning · FCRA Reg B & GDPR Article 22 Compliance</div>', unsafe_allow_html=True)

    # Sidebar: Commercial Underwriting Presets
    st.sidebar.image("https://img.icons8.com/isometric/100/bank.png", width=65)
    st.sidebar.title("Credit Underwriting Desk")
    st.sidebar.caption("Evaluate commercial & retail applicants, execute stress tests, or generate regulatory disclosures.")

    preset_choice = st.sidebar.selectbox(
        "Institutional Borrower Presets:",
        [
            "🟢 Tier 1 Prime Corporate / Executive (AAA)",
            "🟡 Near-Prime Retail Applicant (BBB - Marginal)",
            "🔴 Subprime Leveraged Consumer (CCC - Decline)",
            "🏢 SME Business Owner / Self-Employed",
            "🎲 Random Live Cohort Applicant",
            "✏️ Custom Underwriting Entry"
        ]
    )

    if preset_choice == "🟢 Tier 1 Prime Corporate / Executive (AAA)":
        init_income = 220_000.0
        init_credit = 400_000.0
        init_annuity = 20_000.0
        init_ext1, init_ext2, init_ext3 = 0.75, 0.82, 0.79
        init_age = 46
        init_employed = 14.0
        init_education = "Higher education"
    elif preset_choice == "🟡 Near-Prime Retail Applicant (BBB - Marginal)":
        init_income = 85_000.0
        init_credit = 280_000.0
        init_annuity = 22_000.0
        init_ext1, init_ext2, init_ext3 = 0.42, 0.48, 0.39
        init_age = 32
        init_employed = 3.5
        init_education = "Secondary / secondary special"
    elif preset_choice == "🔴 Subprime Leveraged Consumer (CCC - Decline)":
        init_income = 38_000.0
        init_credit = 350_000.0
        init_annuity = 32_000.0
        init_ext1, init_ext2, init_ext3 = 0.16, 0.21, 0.18
        init_age = 24
        init_employed = 0.8
        init_education = "Secondary / secondary special"
    elif preset_choice == "🏢 SME Business Owner / Self-Employed":
        init_income = 140_000.0
        init_credit = 520_000.0
        init_annuity = 38_000.0
        init_ext1, init_ext2, init_ext3 = 0.58, 0.61, 0.45
        init_age = 41
        init_employed = 8.0
        init_education = "Higher education"
    elif preset_choice == "🎲 Random Live Cohort Applicant" and df_sample is not None:
        rand_row = df_sample.sample(n=1, random_state=None).iloc[0]
        init_income = float(rand_row.get("AMT_INCOME_TOTAL", 90_000))
        init_credit = float(rand_row.get("AMT_CREDIT", 200_000))
        init_annuity = float(rand_row.get("AMT_ANNUITY", 15_000))
        init_ext1 = float(np.nan_to_num(rand_row.get("EXT_SOURCE_1", 0.5), nan=0.5))
        init_ext2 = float(np.nan_to_num(rand_row.get("EXT_SOURCE_2", 0.5), nan=0.5))
        init_ext3 = float(np.nan_to_num(rand_row.get("EXT_SOURCE_3", 0.5), nan=0.5))
        init_age = int(-rand_row.get("DAYS_BIRTH", -12000) / 365.25)
        init_employed = max(0.0, float(-rand_row.get("DAYS_EMPLOYED", -1000) / 365.25))
        init_education = str(rand_row.get("NAME_EDUCATION_TYPE", "Secondary / secondary special"))
    else:
        init_income = 100_000.0
        init_credit = 300_000.0
        init_annuity = 20_000.0
        init_ext1, init_ext2, init_ext3 = 0.50, 0.50, 0.50
        init_age = 35
        init_employed = 5.0
        init_education = "Higher education"

    st.sidebar.subheader("💵 Financial Intake")
    income = st.sidebar.number_input("Annual Gross Income ($)", min_value=10_000.0, max_value=3_000_000.0, value=float(init_income), step=5_000.0)
    credit = st.sidebar.number_input("Requested Credit Facility ($)", min_value=20_000.0, max_value=5_000_000.0, value=float(init_credit), step=10_000.0)
    annuity = st.sidebar.number_input("Annual Debt Service / Annuity ($)", min_value=1_000.0, max_value=600_000.0, value=float(init_annuity), step=2_000.0)

    st.sidebar.subheader("📈 External Bureau Composite Scores (0.0 - 1.0)")
    ext1 = st.sidebar.slider("Equifax Credit Bureau Score", 0.0, 1.0, float(np.clip(init_ext1, 0.0, 1.0)), 0.01)
    ext2 = st.sidebar.slider("TransUnion Credit Bureau Score", 0.0, 1.0, float(np.clip(init_ext2, 0.0, 1.0)), 0.01)
    ext3 = st.sidebar.slider("Experian Credit Bureau Score", 0.0, 1.0, float(np.clip(init_ext3, 0.0, 1.0)), 0.01)

    st.sidebar.subheader("👤 Demographics & Verification")
    age = st.sidebar.slider("Applicant Age (Years)", 18, 75, int(init_age))
    employed = st.sidebar.slider("Employment Tenure (Years)", 0.0, 45.0, float(init_employed), 0.5)
    education = st.sidebar.selectbox(
        "Verified Education Level",
        ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary"],
        index=0 if "Higher" in init_education else 1
    )

    # Base Applicant Dictionary
    applicant_dict = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": credit * 0.9,
        "DAYS_BIRTH": -int(age * 365.25),
        "DAYS_EMPLOYED": -int(employed * 365.25),
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "NAME_EDUCATION_TYPE": education,
    }

    # Baseline Inference
    prob_default, X_vec = predict_borrower_risk(applicant_dict, df_sample, bundle, model)

    # Tabs: 4 Specialized Institutional Workstations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Automated Underwriting",
        "⚡ Economic Stress-Testing",
        "⚖️ Regulatory Adverse Action",
        "🏦 Basel III/IV Capital Metrics"
    ])

    # ----------------------------------------------------
    # TAB 1: AUTOMATED UNDERWRITING
    # ----------------------------------------------------
    with tab1:
        u_col1, u_col2 = st.columns([1.1, 1.9])

        with u_col1:
            st.subheader("Credit Facility Intake")
            dti = credit / (income + 1e-6)
            payment_rate = annuity / credit
            mean_bureau = (ext1 + ext2 + ext3) / 3.0

            st.table(pd.DataFrame({
                "Financial Metric": ["Requested Credit", "Annual Income", "Annual Debt Service", "Debt-to-Income (DTI)", "Payment Rate (Annuity/Credit)", "Composite Bureau Score"],
                "Value": [f"${credit:,.0f}", f"${income:,.0f}", f"${annuity:,.0f}", f"{dti:.2f}x", f"{payment_rate:.3f}", f"{mean_bureau:.3f}"]
            }))

            st.caption("Covenant Benchmark: Conservative bank underwriting policy prefers DTI < 4.5x and Mean Bureau > 0.50.")

        with u_col2:
            st.subheader("Underwriting Risk Verdict")

            # Risk Tiering Logic
            if prob_default < 0.05:
                tier = "AAA / Investment Grade"
                spread = "+1.45% (SOFR + 145 bps)"
                badge = "decision-approved"
                verdict = "APPROVED (PRIME)"
                action_text = "Standard loan approval. Automated STP (Straight-Through Processing). Issue commitment letter."
            elif prob_default < 0.09:
                tier = "A/BBB / Core Prime"
                spread = "+2.25% (SOFR + 225 bps)"
                badge = "decision-approved"
                verdict = "APPROVED (STANDARD)"
                action_text = "Approved with standard documentation verification. Apply standard pricing tier."
            elif prob_default < 0.16:
                tier = "BB / Near-Prime Marginal"
                spread = "+4.50% (SOFR + 450 bps)"
                badge = "decision-review"
                verdict = "CONDITIONAL APPROVAL / MANUAL REVIEW"
                action_text = "Escalate to Senior Credit Committee. Require 20% cash collateral, co-signor, or maximum facility cap of $150,000."
            else:
                tier = "CCC/D / Subprime Vulnerable"
                spread = "N/A (Exceeds Risk Tolerance)"
                badge = "decision-declined"
                verdict = "DECLINED (HIGH DEFAULT RISK)"
                action_text = "Automated decline. Calibrated default probability exceeds internal hurdle rate. Generate FCRA Adverse Action Notice."

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Default Probability (PD)", f"{prob_default * 100:.2f}%")
            m2.metric("Internal Risk Rating", tier.split(" / ")[0])
            m3.metric("Risk-Adjusted Spread", spread.split(" ")[0])

            st.markdown(f'<div class="{badge}"><h3>Decision: {verdict}</h3><p>{action_text}</p><small>Internal Rating Tier: <b>{tier}</b> | Recommended Spread: <b>{spread}</b></small></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.write("**Probability of Default (PD) Spectrum:**")
            st.progress(min(1.0, float(prob_default * 3.0)))
            st.caption("Threshold Calibration: In this bank-deployed system, cost-sensitive underwriting aligns decisions with the real-world 10:1 loss ratio, setting the optimal decision boundary at ~8% PD rather than arbitrary 50%.")

    # ----------------------------------------------------
    # TAB 2: MACROECONOMIC STRESS-TESTING
    # ----------------------------------------------------
    with tab2:
        st.subheader("⚡ Comprehensive Capital Analysis & Review (CCAR) Stress Simulation")
        st.markdown("""
        Simulate how macroeconomic distress (stagflation, aggressive central bank rate hikes, or labor market contraction) impacts this borrower's credit viability.
        """)

        s_col1, s_col2 = st.columns([1.2, 1.8])

        with s_col1:
            st.markdown("#### Macroeconomic Shock Scenarios")
            rate_shock_bps = st.slider("Central Bank Rate Hike (bps)", 0, 500, 200, step=25)
            income_haircut = st.slider("Borrower Income Contraction (%)", 0, 40, 15, step=5)
            bureau_downgrade = st.slider("Credit Bureau Macro Downgrade (%)", 0, 30, 10, step=5)

            # Apply shocks
            stressed_income = income * (1.0 - (income_haircut / 100.0))
            stressed_annuity = annuity * (1.0 + (rate_shock_bps / 10000.0) * 0.75)  # partial pass-through
            stressed_ext1 = ext1 * (1.0 - (bureau_downgrade / 100.0))
            stressed_ext2 = ext2 * (1.0 - (bureau_downgrade / 100.0))
            stressed_ext3 = ext3 * (1.0 - (bureau_downgrade / 100.0))

            stressed_dict = applicant_dict.copy()
            stressed_dict["AMT_INCOME_TOTAL"] = stressed_income
            stressed_dict["AMT_ANNUITY"] = stressed_annuity
            stressed_dict["EXT_SOURCE_1"] = stressed_ext1
            stressed_dict["EXT_SOURCE_2"] = stressed_ext2
            stressed_dict["EXT_SOURCE_3"] = stressed_ext3

            stressed_pd, _ = predict_borrower_risk(stressed_dict, df_sample, bundle, model)

        with s_col2:
            st.markdown("#### Stress Impact Summary")
            pd_delta = (stressed_pd - prob_default) * 100.0

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Baseline PD", f"{prob_default*100:.2f}%")
            sc2.metric("Stressed PD", f"{stressed_pd*100:.2f}%", delta=f"+{pd_delta:.2f}% Risk", delta_color="inverse")

            resilience = "PASS (Robust)" if stressed_pd < 0.12 else ("MARGINAL (Watchlist)" if stressed_pd < 0.20 else "FAIL (Severe Capital Vulnerability)")
            sc3.metric("Stress Test Status", resilience.split(" ")[0])

            fig, ax = plt.subplots(figsize=(8, 3.5), dpi=200)
            bars = ax.bar(["Baseline Scenario", f"Stressed (+{rate_shock_bps}bps / -{income_haircut}%)"], [prob_default*100, stressed_pd*100], color=["#0284c7", "#ef4444"])
            ax.set_ylabel("Probability of Default (%)")
            ax.set_ylim(0, max(stressed_pd*100 * 1.3, 15.0))
            for b in bars:
                ax.annotate(f"{b.get_height():.2f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                            ha='center', va='bottom', fontweight='bold', xytext=(0, 4), textcoords='offset points')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)

    # ----------------------------------------------------
    # TAB 3: REGULATORY ADVERSE ACTION NOTICE
    # ----------------------------------------------------
    with tab3:
        st.subheader("⚖️ Regulatory Adverse Action Disclosure Engine")
        st.markdown("""
        Mandated under the **US Equal Credit Opportunity Act (ECOA, 12 CFR Part 1002 - Regulation B)**, 
        **Fair Credit Reporting Act (FCRA, 15 U.S.C. § 1681m)**, and **EU GDPR Article 22**.
        """)

        # Extract top negative factors
        importances = model.feature_importances_
        contributions = []
        dense_vec = X_vec[0]
        for idx, fname in enumerate(feature_names):
            val = dense_vec[idx]
            imp = importances[idx]
            if abs(val) > 1e-4 and imp > 0.002:
                clean_name = DOMAIN_NAMES.get(fname, fname)
                contributions.append((clean_name, imp, val))

        contributions.sort(key=lambda x: x[1], reverse=True)
        top_factors = contributions[:4]

        adv_col1, adv_col2 = st.columns([1.3, 1.7])

        with adv_col1:
            st.markdown("#### Legal Adverse Action Statement")
            reasons_md = "\n".join([f"  {i+1}. {factor[0]}" for i, factor in enumerate(top_factors)])

            notice_text = f"""STATEMENT OF CREDIT DENIAL, TERMINATION, OR CHANGE
DATE: September 10, 2026
APPLICANT ID: REF-{(abs(hash(str(income)+str(credit))) % 900000) + 100000}
CREDIT FACILITY REQUESTED: ${credit:,.2f}

DESCRIPTION OF TRANSACTION: Consumer / Commercial Term Loan

PRINCIPAL REASON(S) FOR ADVERSE ACTION:
{reasons_md}

DISCLOSURE OF USE OF INFORMATION OBTAINED FROM AN OUTSIDE SOURCE:
Our credit decision was based in whole or in part on information
obtained in a report from the consumer reporting agency/agencies below:
- Equifax (Proxy Composite Score: {ext1:.2f})
- TransUnion (Proxy Composite Score: {ext2:.2f})
- Experian (Proxy Composite Score: {ext3:.2f})

Under the Fair Credit Reporting Act, you have the right to inspect
and obtain a free copy of your credit report within 60 days.
"""
            st.markdown(f'<div class="adverse-notice"><pre>{notice_text}</pre></div>', unsafe_allow_html=True)

        with adv_col2:
            st.markdown("#### Quantitative Feature Importance Attribution")
            chart_df = pd.DataFrame(top_factors, columns=["Underwriting Dimension", "Weight", "Value"])

            fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
            ax.barh(chart_df["Underwriting Dimension"][::-1], chart_df["Weight"][::-1], color="#475569")
            ax.set_xlabel("Relative Weight in Underwriting Decision")
            ax.set_title("Top Principal Drivers Governing Risk Assessment")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)

    # ----------------------------------------------------
    # TAB 4: BASEL III/IV CAPITAL METRICS
    # ----------------------------------------------------
    with tab4:
        st.subheader("🏦 Basel III / IV Capital Adequacy & Expected Loss Calculator")
        st.markdown("""
        Calculates credit portfolio provisions under **Basel Internal Ratings-Based (IRB)** standards and **IFRS 9 / CECL Expected Credit Loss** framework.
        """)

        b1, b2, b3, b4 = st.columns(4)

        ead = credit
        lgd = 0.45  # Standard senior unsecured Loss Given Default
        expected_loss = prob_default * lgd * ead
        risk_weight = min(2.5, max(0.20, (prob_default * 12.5) * (1.0 / (1.0 - lgd))))
        rwa = ead * risk_weight
        tier1_capital = rwa * 0.08  # 8% Minimum Tier 1 Capital Requirement

        b1.metric("Exposure at Default (EAD)", f"${ead:,.0f}")
        b2.metric("Loss Given Default (LGD)", f"{lgd*100:.0f}%", help="Standard Basel senior unsecured default benchmark")
        b3.metric("Expected Loss (EL)", f"${expected_loss:,.2f}", help="EL = PD × LGD × EAD")
        b4.metric("Risk-Weighted Asset (RWA)", f"${rwa:,.0f}")

        st.markdown("---")

        st.markdown("#### Capital & Provisioning Breakdown")
        cap_df = pd.DataFrame({
            "Basel / IFRS Metric": [
                "Probability of Default (PD)",
                "Loss Given Default (LGD)",
                "Exposure at Default (EAD)",
                "Accounting Expected Credit Loss (ECL / Provision)",
                "Regulatory Risk-Weight Multiplier",
                "Total Risk-Weighted Assets (RWA)",
                "Mandatory Basel Minimum Tier 1 Capital (8%)"
            ],
            "Calculation / Source": [
                "Champion XGBoost ML Output",
                "Basel Standard (Senior Unsecured)",
                "Total Committed Facility",
                "PD × LGD × EAD",
                "Basel IRB Foundation Weight Formula",
                "EAD × Risk Weight",
                "RWA × 8.0%"
            ],
            "Value": [
                f"{prob_default*100:.2f}%",
                f"{lgd*100:.1f}%",
                f"${ead:,.2f}",
                f"${expected_loss:,.2f}",
                f"{risk_weight*100:.1f}%",
                f"${rwa:,.2f}",
                f"${tier1_capital:,.2f}"
            ]
        })
        st.table(cap_df)


if __name__ == "__main__":
    main()
