# Notebooks Directory

This directory contains interactive Jupyter notebooks for exploring data, training models, and inspecting explainability results.

## Available Notebooks

| Notebook | Purpose | Environment |
|---|---|---|
| **[`credit_risk_standalone_pipeline.ipynb`](credit_risk_standalone_pipeline.ipynb)** | **Complete Self-Contained Pipeline** (Zero dependencies beyond Kaggle dataset; runs end-to-end in ~1 hour on CPU) | Local Jupyter / Google Colab |
| **`colab/RUN_ALL.ipynb`** | Checkpoint-aware full experiment runner on Google Colab | Google Colab |
| **`colab/00_eda.ipynb`** | Deep exploratory data analysis with 6 distribution figures and correlation tables | Local / Colab |
| **`colab/01_setup_and_data.ipynb`** | Feature engineering and strict temporal train/test split | Local / Colab |
| **`colab/02_cross_validation.ipynb`** | 3×5 repeated stratified cross-validation (135 model fits across 9 cells) | Local / Colab |
| **`colab/03_oot_and_statistics.ipynb`** | Out-of-time (2020) validation and paired statistical hypothesis testing | Local / Colab |
| **`colab/04_shap.ipynb`** | TreeSHAP vs LinearSHAP global importance and local adverse action case studies | Local / Colab |

## Running on Google Colab

To run in Google Colab:
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `notebooks/credit_risk_standalone_pipeline.ipynb`.
3. Provide your Kaggle API key or upload `application_train.csv` when prompted.
4. Select **Runtime > Run all**.
