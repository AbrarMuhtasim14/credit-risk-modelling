"""Rebuild WALKTHROUGH_lite.ipynb from the executed full version and validate both."""
import nbformat
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "WALKTHROUGH.ipynb"
LITE = ROOT / "WALKTHROUGH_lite.ipynb"

nb = nbformat.read(str(FULL), as_version=4)
nbformat.validate(nb)
n_err = sum(1 for c in nb.cells if c.cell_type == "code"
            and any(o.get("output_type") == "error" for o in c.get("outputs", [])))
print(f"Full version: {len(nb.cells)} cells, errors={n_err}, size={FULL.stat().st_size/1e6:.1f} MB")

stripped = 0
for c in nb.cells:
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        data = o.get("data", {})
        if "image/png" in data:
            del data["image/png"]
            stripped += 1
            data["text/plain"] = ["[figure embedded in full version - see reports/figures or outputs/eda]"]

nbformat.write(nb, str(LITE))
print(f"Lite version: stripped {stripped} PNGs, size={LITE.stat().st_size/1024:.0f} KB")
