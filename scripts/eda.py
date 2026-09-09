"""
Exploratory Data Analysis — Home Credit Default Risk (application_train)
Project: Loan Default Prediction Using Machine Learning (MSc)
Outputs: console report + outputs/eda/*.png + outputs/eda/*.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(r"F:\credit-risk-project - Copy")
RAW = ROOT / "data" / "raw" / "application_train.csv"
DESC = ROOT / "data" / "raw" / "HomeCredit_columns_description.csv"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

lines = []
def log(s=""):
    print(s)
    lines.append(str(s))

# ---------------------------------------------------------------- load
log("Loading data...")
df = pd.read_csv(RAW, low_memory=False)
desc = pd.read_csv(DESC, encoding="latin-1")
desc_map = dict(zip(desc["Row"], desc["Description"]))

log("=" * 70)
log("1. DATASET OVERVIEW")
log("=" * 70)
log(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
dt = df.dtypes.value_counts()
log("Dtype counts:\n" + dt.to_string())

cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
flag_cols = [c for c in df.columns if c.startswith("FLAG_")]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ("SK_ID_CURR", "TARGET") and c not in flag_cols]
log(f"\nCategorical: {len(cat_cols)} | Binary flags: {len(flag_cols)} | Numeric: {len(num_cols)}")

# ---------------------------------------------------------------- duplicates
log("\n" + "=" * 70)
log("2. DUPLICATES")
log("=" * 70)
dup_full = df.duplicated().sum()
dup_id = df.duplicated(subset=["SK_ID_CURR"]).sum()
log(f"Exact duplicate rows: {dup_full:,}")
log(f"Duplicate SK_ID_CURR: {dup_id:,}")

# ---------------------------------------------------------------- target
log("\n" + "=" * 70)
log("3. TARGET VARIABLE (class imbalance)")
log("=" * 70)
vc = df["TARGET"].value_counts()
log(vc.to_string())
rate = df["TARGET"].mean()
log(f"\nDefault rate: {rate:.4%}")
log(f"Imbalance ratio (non-default : default): {vc[0]/vc[1]:.2f} : 1")

fig, ax = plt.subplots(figsize=(5, 4))
vc.plot.bar(ax=ax, color=["#4c72b0", "#c44e52"], edgecolor="black")
ax.set_xticklabels(["0 = repaid", "1 = default"], rotation=0)
ax.set_ylabel("Count")
ax.set_title(f"TARGET distribution (default rate {rate:.2%})")
for i, v in enumerate(vc):
    ax.text(i, v + 3000, f"{v:,}", ha="center")
plt.tight_layout(); plt.savefig(OUT / "01_target_distribution.png"); plt.close()

# ---------------------------------------------------------------- application_date
log("\n" + "=" * 70)
log("4. APPLICATION DATE (temporal structure)")
log("=" * 70)
if "application_date" in df.columns:
    ad = pd.to_datetime(df["application_date"], errors="coerce")
    log(f"Parse failures: {ad.isna().sum():,}")
    log(f"Range: {ad.min()}  ->  {ad.max()}")
    monthly = df.assign(_m=ad.dt.to_period("M")).groupby("_m")["TARGET"].agg(["count", "mean"])
    log("\nMonthly applications & default rate (first 6 / last 6):")
    log(monthly.head(6).round(4).to_string())
    log("...")
    log(monthly.tail(6).round(4).to_string())

    fig, ax1 = plt.subplots(figsize=(11, 4))
    x = monthly.index.to_timestamp()
    ax1.bar(x, monthly["count"], width=20, color="#4c72b0", alpha=0.7, label="applications")
    ax1.set_ylabel("Applications per month")
    ax2 = ax1.twinx()
    ax2.plot(x, monthly["mean"], color="#c44e52", lw=1.5, label="default rate")
    ax2.set_ylabel("Default rate")
    ax1.set_title("Application volume and default rate over time")
    plt.tight_layout(); plt.savefig(OUT / "02_application_date_trend.png"); plt.close()
else:
    log("No application_date column.")

# ---------------------------------------------------------------- missing
log("\n" + "=" * 70)
log("5. MISSING VALUES")
log("=" * 70)
miss = df.isna().mean().sort_values(ascending=False)
miss_n = df.isna().sum()
log(f"Columns with any missing: {(miss > 0).sum()} / {df.shape[1]}")
log(f"Columns >50% missing: {(miss > 0.5).sum()}")
log(f"Columns >30% missing: {(miss > 0.3).sum()}")
log(f"Columns >10% missing: {(miss > 0.1).sum()}")
log("\nTop 25 by missing %:")
top_miss = pd.DataFrame({"missing_pct": (miss * 100).round(2),
                         "missing_n": miss_n}).head(25)
top_miss["description"] = top_miss.index.map(lambda c: str(desc_map.get(c, ""))[:60])
log(top_miss.to_string())

miss.to_frame("missing_pct").to_csv(OUT / "missing_by_column.csv")

fig, ax = plt.subplots(figsize=(12, 5))
top30 = miss.head(30) * 100
ax.bar(range(len(top30)), top30.values, color="#dd8452")
ax.set_xticks(range(len(top30)))
ax.set_xticklabels(top30.index, rotation=80, fontsize=7)
ax.set_ylabel("Missing %")
ax.set_title("Top 30 columns by missing values")
ax.axhline(50, color="red", ls="--", lw=1)
plt.tight_layout(); plt.savefig(OUT / "03_missing_top30.png"); plt.close()

# missingness by target
miss_by_t = df.groupby("TARGET").apply(lambda g: g.isna().mean().mean(), include_groups=False)
log(f"\nAvg cell missingness — default=1: {miss_by_t.get(1, np.nan):.4f} | default=0: {miss_by_t.get(0, np.nan):.4f}")

# ---------------------------------------------------------------- anomalies
log("\n" + "=" * 70)
log("6. KNOWN ANOMALIES & DATA QUALITY")
log("=" * 70)
anom = df["DAYS_EMPLOYED"] == 365243
log(f"DAYS_EMPLOYED == 365243 (sentinel for unemployed/retired): {anom.sum():,} ({anom.mean():.2%})")
log(f"  default rate in sentinel group: {df.loc[anom, 'TARGET'].mean():.3%} vs others: {df.loc[~anom, 'TARGET'].mean():.3%}")

neg_birth = (df["DAYS_BIRTH"] >= 0).sum()
log(f"DAYS_BIRTH >= 0 (invalid): {neg_birth:,}")
age_years = -df["DAYS_BIRTH"] / 365
log(f"Age range derived from DAYS_BIRTH: {age_years.min():.1f} – {age_years.max():.1f} years")

for c in ["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE", "OCCUPATION_TYPE"]:
    xna = (df[c].astype(str) == "XNA").sum()
    if xna:
        log(f"{c}: 'XNA' values = {xna:,}")

log(f"\nAMT_ANNUITY: min={df['AMT_ANNUITY'].min():,.0f} max={df['AMT_ANNUITY'].max():,.0f} "
    f"(missing {df['AMT_ANNUITY'].isna().mean():.2%})")
log(f"AMT_GOODS_PRICE missing: {df['AMT_GOODS_PRICE'].isna().mean():.2%}")
log(f"AMT_INCOME_TOTAL: min={df['AMT_INCOME_TOTAL'].min():,.0f} max={df['AMT_INCOME_TOTAL'].max():,.0f}")
hi = (df["AMT_INCOME_TOTAL"] > 1e6).sum()
log(f"  incomes > 1,000,000: {hi:,}")

# ---------------------------------------------------------------- numeric distributions
log("\n" + "=" * 70)
log("7. KEY NUMERIC FEATURES")
log("=" * 70)
key_num = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
           "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
           "REGION_POPULATION_RELATIVE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
key_num = [c for c in key_num if c in df.columns]
log(df[key_num].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    .round(2).to_string())

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, c in zip(axes.ravel(), ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                                 "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
    s = df[c].dropna()
    if c.startswith("AMT"):
        s = s.clip(upper=s.quantile(0.99))
    ax.hist(s, bins=60, color="#4c72b0", edgecolor="none")
    ax.set_title(c, fontsize=10)
plt.suptitle("Key numeric feature distributions (AMT clipped at 99th pct)")
plt.tight_layout(); plt.savefig(OUT / "04_numeric_distributions.png"); plt.close()

# EXT_SOURCE missing + target relationship
log("\nEXT_SOURCE columns (external risk scores):")
for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
    if c in df.columns:
        m = df[c].isna().mean()
        corr = df[[c, "TARGET"]].corr().iloc[0, 1]
        log(f"  {c}: missing {m:.2%}, corr with TARGET = {corr:.3f}")

# ---------------------------------------------------------------- categoricals
log("\n" + "=" * 70)
log("8. CATEGORICAL FEATURES")
log("=" * 70)
card = pd.DataFrame({
    "n_unique": df[cat_cols].nunique(),
    "missing_pct": (df[cat_cols].isna().mean() * 100).round(2),
}).sort_values("n_unique", ascending=False)
log(card.to_string())

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_cats = ["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS",
             "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "NAME_CONTRACT_TYPE"]
for ax, c in zip(axes.ravel(), plot_cats):
    g = df.groupby(c, dropna=False)["TARGET"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    g = g.head(10)
    ax.bar(range(len(g)), g["mean"] * 100, color="#55a868")
    ax.axhline(rate * 100, color="red", ls="--", lw=1)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(x)[:16] for x in g.index], rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Default rate by {c}", fontsize=9)
    ax.set_ylabel("Default %")
plt.suptitle("Default rate by category (red line = overall average)")
plt.tight_layout(); plt.savefig(OUT / "05_default_rate_by_category.png"); plt.close()

# ---------------------------------------------------------------- correlation
log("\n" + "=" * 70)
log("9. CORRELATION WITH TARGET")
log("=" * 70)
# coerce defensively (pandas 3.0 may keep stray non-numeric values in numeric-looking cols)
num_df = df[num_cols + flag_cols]
bad_cols = {}
for c in num_df.columns:
    if num_df[c].dtype == object or str(num_df[c].dtype) in ("str", "string"):
        bad_cols[c] = num_df[c].map(lambda v: v if not isinstance(v, (int, float)) else None).dropna().unique()[:5]
        num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
if bad_cols:
    log("Columns coerced (contained non-numeric values):")
    for c, vals in bad_cols.items():
        log(f"  {c}: stray values = {list(vals)}")
corr_target = (num_df.corrwith(df["TARGET"]).abs().sort_values(ascending=False))
log("Top 20 features by |corr| with TARGET:")
top20 = pd.DataFrame({"abs_corr": corr_target.head(20).round(4)})
top20["signed_corr"] = num_df.corrwith(df["TARGET"]).round(4)
top20["description"] = top20.index.map(lambda c: str(desc_map.get(c, ""))[:70])
log(top20.to_string())
top20.to_csv(OUT / "correlation_with_target.csv")

fig, ax = plt.subplots(figsize=(8, 7))
t = corr_target.head(20)
ax.barh(range(len(t))[::-1], t.values, color="#4c72b0")
ax.set_yticks(range(len(t))[::-1])
ax.set_yticklabels(t.index, fontsize=8)
ax.set_xlabel("|Pearson corr| with TARGET")
ax.set_title("Top 20 features correlated with default")
plt.tight_layout(); plt.savefig(OUT / "06_correlation_top20.png"); plt.close()

# high inter-feature correlation (multicollinearity candidates)
log("\nHighly correlated feature pairs (|r| > 0.9, numeric only, sample of 30k rows):")
sample = num_df.sample(min(30000, len(num_df)), random_state=42)
cm = sample.corr()
pairs = []
for i in range(len(cm.columns)):
    for j in range(i + 1, len(cm.columns)):
        r = cm.iloc[i, j]
        if abs(r) > 0.9:
            pairs.append((cm.columns[i], cm.columns[j], round(r, 3)))
pairs.sort(key=lambda x: -abs(x[2]))
for p in pairs[:20]:
    log(f"  {p[0]} <-> {p[1]}: {p[2]}")
log(f"  ... total pairs |r|>0.9: {len(pairs)}")
pd.DataFrame(pairs, columns=["feat1", "feat2", "corr"]).to_csv(OUT / "high_corr_pairs.csv", index=False)

# ---------------------------------------------------------------- leakage candidates
log("\n" + "=" * 70)
log("10. POTENTIAL DATA LEAKAGE / SCOPE CANDIDATES (review & document)")
log("=" * 70)
suspects = ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
            "LIVE_CITY_NOT_WORK_CITY", "REG_CITY_NOT_WORK_CITY",
            "DAYS_LAST_PHONE_CHANGE", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3"]
for c in suspects:
    if c in df.columns:
        log(f"  {c}: {str(desc_map.get(c, ''))[:90]}")

log("\n" + "=" * 70)
log("EDA COMPLETE")
log("=" * 70)

with open(OUT / "eda_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nSaved report + figures to {OUT}")
