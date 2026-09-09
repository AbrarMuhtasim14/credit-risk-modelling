"""Validate STANDALONE.ipynb: no forbidden references, valid nbformat."""
import json
import os

import nbformat

with open("STANDALONE.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

src_all = []
for c in nb["cells"]:
    if c["cell_type"] == "code":
        src_all.append("".join(c["source"]))
allsrc = "\n".join(src_all)

forbidden = ["from src.", "import src.", "from run_experiment", "import run_experiment",
             "TABLES_DIR", "FIGURES_DIR", "MODELS_DIR", "PROCESSED_DIR",
             "RAW_APPLICATION_TRAIN", "read_parquet", "read_csv(TABLES", "SKIPPING",
             "project_root", "PROJECT_ROOT", "drive.mount", "google.colab import drive"]
print("=== Forbidden-reference scan ===")
ok = True
for f in forbidden:
    hits = allsrc.count(f)
    flag = "FAIL" if hits > 0 else "ok"
    if hits > 0:
        ok = False
    print(f"  {flag:4}  {f}: {hits}")

print()
print("=== Notebook stats ===")
print("cells:", len(nb["cells"]))
print("code:", sum(1 for c in nb["cells"] if c["cell_type"] == "code"))
print("markdown:", sum(1 for c in nb["cells"] if c["cell_type"] == "markdown"))
print(f"size: {os.path.getsize('STANDALONE.ipynb') / 1024:.0f} KB")

nbformat.validate(nbformat.read("STANDALONE.ipynb", as_version=4))
print("nbformat validation: OK")
print("Overall:", "PASS" if ok else "FAIL")
