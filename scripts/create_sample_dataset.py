"""Generate a lightweight 1,000-row sample dataset with identical schema and distribution."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "application_train.csv"
SAMPLE_DIR = ROOT / "data" / "sample"
SAMPLE_PATH = SAMPLE_DIR / "sample_application_train.csv"

def create_sample(n_samples: int = 1000, random_seed: int = 42):
    if not RAW_PATH.exists():
        print(f"Raw data file not found at {RAW_PATH}")
        return

    print(f"Reading raw data from {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH, nrows=50000)
    
    # 8.07% default rate = ~81 defaulters out of 1000
    n_defaulters = int(n_samples * 0.081)
    n_non_defaulters = n_samples - n_defaulters

    defaulters = df[df["TARGET"] == 1].sample(n=n_defaulters, random_state=random_seed)
    non_defaulters = df[df["TARGET"] == 0].sample(n=n_non_defaulters, random_state=random_seed)

    sample_df = pd.concat([defaulters, non_defaulters]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(SAMPLE_PATH, index=False)
    print(f"Created sample dataset: {SAMPLE_PATH}")
    print(f"Rows: {len(sample_df)}, Columns: {sample_df.shape[1]}")
    print(f"Default rate: {sample_df['TARGET'].mean():.4f}")
    print(f"File size: {SAMPLE_PATH.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    create_sample()
