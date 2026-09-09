"""Download helper for the Home Credit Default Risk dataset.

This script fetches the official Home Credit Default Risk application_train.csv
from Kaggle or guides the user through direct download options.
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
TARGET_FILE = RAW_DIR / "application_train.csv"
KAGGLE_COMPETITION = "home-credit-default-risk"


def check_existing() -> bool:
    if TARGET_FILE.exists():
        size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        print(f"Dataset already exists at: {TARGET_FILE} ({size_mb:.1f} MB)")
        return True
    return False


def download_via_kaggle_cli():
    """Attempt download using the kaggle CLI."""
    print("Attempting download via Kaggle CLI...")
    cmd = f"kaggle competitions download -c {KAGGLE_COMPETITION} -f application_train.csv -p {RAW_DIR}"
    ret = os.system(cmd)
    if ret == 0:
        # Check if downloaded as zip
        zip_path = RAW_DIR / "application_train.csv.zip"
        if zip_path.exists():
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(RAW_DIR)
            zip_path.unlink()
            print("Extraction complete.")
        return check_existing()
    return False


def print_manual_instructions():
    print("=" * 70)
    print("DATASET ACQUISITION INSTRUCTIONS")
    print("=" * 70)
    print(f"The raw dataset ({TARGET_FILE.name}, ~152 MB) is required for full pipeline runs.")
    print("\nOption A: Kaggle CLI (Recommended)")
    print("1. Set up your Kaggle API key (~/.kaggle/kaggle.json)")
    print("2. Accept competition rules at: https://www.kaggle.com/c/home-credit-default-risk/rules")
    print("3. Run:")
    print(f"   kaggle competitions download -c {KAGGLE_COMPETITION} -f application_train.csv -p data/raw/")
    print("   (Unzip the resulting file into data/raw/)\n")
    print("Option B: Browser Download")
    print("1. Visit: https://www.kaggle.com/c/home-credit-default-risk/data")
    print("2. Download 'application_train.csv'")
    print(f"3. Place the file inside: {RAW_DIR.resolve()}/\n")
    print("Option C: Fast Quickstart (Zero Download)")
    print("Use the included 1,000-row sample dataset located at:")
    print("   data/sample/sample_application_train.csv")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Download Home Credit Default Risk dataset")
    parser.add_argument("--force", action="store_true", help="Force redownload even if file exists")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.force and check_existing():
        return

    success = download_via_kaggle_cli()
    if not success:
        print("\nKaggle CLI download was not successful or credentials not configured.")
        print_manual_instructions()


if __name__ == "__main__":
    main()
