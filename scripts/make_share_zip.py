"""Create a single shareable zip of the whole project (dataset included)."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "credit-risk-project-share.zip"
TOP = "credit-risk-project"  # top-level folder inside the archive

SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".log"}

count = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(ROOT).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if p.suffix in SKIP_SUFFIXES:
            continue
        zf.write(p, f"{TOP}/{p.relative_to(ROOT)}")
        count += 1

size_mb = OUT.stat().st_size / 1e6
print(f"Zipped {count} files -> {OUT.name} = {size_mb:.1f} MB")
