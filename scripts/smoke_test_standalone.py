"""Smoke-test STANDALONE.ipynb locally: copy dataset, set reduced env vars, execute via nbconvert.

Runs the full notebook pipeline with a minimal CV (1 repeat x 2 folds) and small
SHAP sample to verify every cell executes without errors. The reduced-mode guard
skips the dissertation-number assertions automatically.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

# 1. Copy dataset to root (notebook looks for application_train.csv at cwd)
if not Path("application_train.csv").exists():
    shutil.copy2("data/raw/application_train.csv", "application_train.csv")
    print("Copied application_train.csv to root.")
else:
    print("application_train.csv already at root.")

# 2. Clean any stale results dir
res = Path("from_scratch_results")
if res.exists():
    shutil.rmtree(res)
    print("Removed stale from_scratch_results/")

# 3. Set reduced-mode env vars
os.environ["MCR_N_REPEATS"] = "1"
os.environ["MCR_N_FOLDS"] = "2"
os.environ["MCR_SHAP_SAMPLE"] = "500"
print(f"Smoke test config: {os.environ['MCR_N_REPEATS']}x{os.environ['MCR_N_FOLDS']} CV, "
      f"SHAP sample {os.environ['MCR_SHAP_SAMPLE']}")

# 4. Execute via nbconvert
print("\nExecuting STANDALONE.ipynb via nbconvert...\n")
result = subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
     "--output", "STANDALONE_executed.ipynb", "STANDALONE.ipynb"],
    capture_output=True, text=True, timeout=3600,
    cwd=str(ROOT),
)
print("Return code:", result.returncode)
if result.stdout:
    print("STDOUT (last 500):", result.stdout[-500:])
if result.stderr:
    print("STDERR (last 2000):", result.stderr[-2000:])

if result.returncode != 0:
    print("\nFAILED - checking for error details in output notebook...")
    if Path("STANDALONE_executed.ipynb").exists():
        import json
        nb = json.load(open("STANDALONE_executed.ipynb", encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "error":
                        print(f"\nCell {i}: {output.get('ename', '?')}: {output.get('evalue', '?')}")
                        for line in output.get("traceback", [])[-5:]:
                            print(f"  {line}")
    sys.exit(1)

# 5. Verify outputs
print("\n=== Verification ===")
checks = [
    ("cv_fold_results.csv", "CV fold results"),
    ("cv_summary.csv", "CV summary"),
    ("oot_results.csv", "OOT results"),
    ("statistical_significance.csv", "Statistical significance"),
    ("shap_importance_top30.csv", "SHAP importance"),
    ("shap_importance_top30_logistic.csv", "LinearSHAP importance"),
    ("shap_case_studies.csv", "SHAP case studies"),
    ("shap_tree_vs_linear_overlap.txt", "Tree vs linear overlap"),
]
figs = [
    "01_target_distribution.png", "02_application_date_trend.png",
    "03_missing_top30.png", "04_numeric_distributions.png",
    "05_default_rate_by_category.png", "06_correlation_top20.png",
    "roc_curves_oot.png", "pr_curves_oot.png", "confusion_matrices_oot.png",
    "cost_curves_oot.png", "statistical_significance_heatmap.png",
    "shap_importance.png", "shap_beeswarm.png", "shap_dependence.png",
    "shap_case_studies.png", "shap_importance_logistic.png",
]
all_ok = True
for fname, label in checks:
    p = Path("from_scratch_results") / fname
    exists = p.exists()
    all_ok = all_ok and exists
    print(f"  {'OK' if exists else 'MISSING'}  {label}: {fname}")

for fname in figs:
    p = Path("from_scratch_results") / "figures" / fname
    exists = p.exists()
    all_ok = all_ok and exists
    print(f"  {'OK' if exists else 'MISSING'}  figures/{fname}")

# Check CV record count
import pandas as pd
df_cv = pd.read_csv("from_scratch_results/cv_fold_results.csv")
expected = 1 * 2 * 9  # smoke mode: 1 repeat x 2 folds x 9 configs
print(f"\nCV records: {len(df_cv)} (expected {expected})")
assert len(df_cv) == expected, f"Wrong CV record count: {len(df_cv)}"

# Check no errors in executed notebook
import json
nb = json.load(open("STANDALONE_executed.ipynb", encoding="utf-8"))
errors = 0
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors += 1
print(f"Errors in executed notebook: {errors}")

print(f"\n{'ALL CHECKS PASSED' if all_ok and errors == 0 else 'SOME CHECKS FAILED'}")
