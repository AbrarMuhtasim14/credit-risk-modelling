"""Generate EDA-phase figures for the educational writeup (DOCX)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import RAW_APPLICATION_TRAIN, FIGURE_DPI

OUT = ROOT / "reports" / "figures" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

print("Loading raw data...")
df = pd.read_csv(RAW_APPLICATION_TRAIN)
df["application_date"] = pd.to_datetime(df["application_date"])
df["app_year"] = df["application_date"].dt.year
print("Shape:", df.shape)

# ---------------------------------------------------------------- 1. Target balance
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=FIGURE_DPI)
counts = df["TARGET"].value_counts().sort_index()
pct = 100 * counts / len(df)
bars = ax.bar(["TARGET = 0\n(repaid)", "TARGET = 1\n(defaulted)"], counts,
              color=["#4c9f70", "#d9534f"], width=0.5)
for b, c, p in zip(bars, counts, pct):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3000,
            f"{c:,}\n({p:.1f}%)", ha="center", fontweight="bold")
ax.set_ylabel("Number of loan applications")
ax.set_title("The Target Variable: only ~8% of borrowers default", fontweight="bold")
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
fig.savefig(OUT / "eda_target_balance.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print("1/7 target balance done")

# ---------------------------------------------------------------- 2. DAYS_EMPLOYED sentinel
de = df["DAYS_EMPLOYED"]
n_sent = int((de == 365243).sum())
de_clean = de[de != 365243].dropna() / -365.25
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=FIGURE_DPI)
ax.hist(de_clean, bins=80, color="#5b8dd9", edgecolor="white", linewidth=0.3)
ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.annotate(f"SENTINEL: {n_sent:,} rows ({100*n_sent/len(df):.1f}%)\nDAYS_EMPLOYED = 365,243\n(= exactly 1,000 years!)",
            xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
            fontsize=10, fontweight="bold", color="#c0392b",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fdecea", ec="#c0392b"))
ax.set_xlabel("Years in current employment (converted from negative days)")
ax.set_ylabel("Number of applicants")
ax.set_title("The 1,000-Year Employment Anomaly", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "eda_days_employed_sentinel.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print("2/7 sentinel done")

# ---------------------------------------------------------------- 3. Income winsorization
inc = df["AMT_INCOME_TOTAL"]
cap = float(inc.quantile(0.999))
n_clip = int((inc > cap).sum())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=FIGURE_DPI)
ax = axes[0]
ax.hist(inc / 1000, bins=120, range=(0, 1500), color="#8e7cc3", edgecolor="white", linewidth=0.3)
ax.axvline(cap / 1000, color="#c0392b", lw=2, ls="--")
ax.text(cap / 1000 + 15, ax.get_ylim()[1] * 0.85,
        f"99.9th percentile\ncap = ${cap:,.0f}\n({n_clip} rows above)",
        color="#c0392b", fontweight="bold", fontsize=9)
ax.set_xlabel("Annual income ($ thousands)")
ax.set_title("BEFORE: a few extreme incomes stretch the axis", fontweight="bold")
ax.set_ylabel("Applicants")
inc_c = inc.clip(upper=cap)
ax = axes[1]
ax.hist(inc_c / 1000, bins=120, range=(0, 1500), color="#4c9f70", edgecolor="white", linewidth=0.3)
ax.axvline(cap / 1000, color="#c0392b", lw=2, ls="--")
ax.set_xlabel("Annual income ($ thousands)")
ax.set_title("AFTER winsorizing: same 99.9% of people, tamed tail", fontweight="bold")
ax.set_ylabel("Applicants")
plt.tight_layout()
fig.savefig(OUT / "eda_income_winsor.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"3/7 income done (cap=${cap:,.0f}, clipped={n_clip})")

# ---------------------------------------------------------------- 4. EXT_SOURCE scores by target
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=FIGURE_DPI)
for ax, col in zip(axes, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
    for tgt, color, lbl in [(0, "#4c9f70", "Repaid (0)"), (1, "#d9534f", "Defaulted (1)")]:
        vals = df.loc[df["TARGET"] == tgt, col].dropna()
        ax.hist(vals, bins=60, density=True, alpha=0.55, color=color, label=lbl)
    ax.set_title(col, fontweight="bold")
    ax.set_xlabel("Bureau score")
    axes[0].set_ylabel("Density")
axes[0].legend(fontsize=8)
fig.suptitle("External credit-bureau scores: defaulters sit clearly lower", fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "eda_ext_sources.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print("4/7 ext sources done")

# ---------------------------------------------------------------- 5. Building triplet collinearity
triplets = ["APARTMENTS", "BASEMENTAREA", "COMMONAREA", "ELEVATORS",
            "ENTRANCES", "LIVINGAREA"]
cols = [f"{t}_{s}" for t in triplets for s in ("AVG", "MEDI", "MODE")]
cols = [c for c in cols if c in df.columns]
corr = df[cols].corr()
fig, ax = plt.subplots(figsize=(9, 7.5), dpi=FIGURE_DPI)
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=90, fontsize=7)
ax.set_yticklabels(cols, fontsize=7)
for i in range(len(cols)):
    for j in range(len(cols)):
        if abs(corr.values[i, j]) > 0.95:
            ax.plot(j, i, "s", ms=3, color="black", alpha=0.35)
ax.set_title("Correlation matrix of building-information triplets\n"
             "(black squares: |r| > 0.95 = near-perfect duplicates)", fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
plt.tight_layout()
fig.savefig(OUT / "eda_building_corr.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print("5/7 building corr done")

# ---------------------------------------------------------------- 6. FLAG_EMP_PHONE trap
raw_corr = df["FLAG_EMP_PHONE"].astype(float).corr(df["DAYS_EMPLOYED"].astype(float))
de_fixed = df["DAYS_EMPLOYED"].replace(365243, np.nan)
fixed_corr = df["FLAG_EMP_PHONE"].astype(float).corr(de_fixed)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=FIGURE_DPI)
sub = df.sample(25000, random_state=42)
ax = axes[0]
for flag, color, lbl in [(0, "#d9534f", "FLAG_EMP_PHONE = 0"), (1, "#4c9f70", "FLAG_EMP_PHONE = 1")]:
    s = sub[sub["FLAG_EMP_PHONE"] == flag]["DAYS_EMPLOYED"] / -365.25
    ax.hist(s, bins=60, alpha=0.6, color=color, label=lbl)
ax.set_title(f"BEFORE fix: r = {raw_corr:.4f}\n(the 1,000-year spike fakes the correlation)", fontweight="bold")
ax.set_xlabel("Years employed")
ax.legend(fontsize=8)
ax = axes[1]
bars = ax.bar(["Raw data\n(sentinel present)", "After sentinel\ncorrection"],
              [raw_corr, fixed_corr], color=["#d9534f", "#4c9f70"], width=0.45)
for b, v in zip(bars, [raw_corr, fixed_corr]):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.09),
            f"r = {v:.4f}", ha="center", fontweight="bold")
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylim(-1.1, 0.35)
ax.set_ylabel("Correlation with DAYS_EMPLOYED")
ax.set_title("The correlation was a data artifact", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "eda_flag_emp_phone.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"6/7 flag_emp_phone done (raw r={raw_corr:.4f}, fixed r={fixed_corr:.4f})")

# ---------------------------------------------------------------- 7. Temporal split
yearly = df.groupby("app_year").agg(n=("TARGET", "size"), dr=("TARGET", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=FIGURE_DPI)
colors = ["#4c9f70" if y in (2018, 2019) else "#d9534f" for y in yearly["app_year"]]
ax.bar(yearly["app_year"].astype(str), yearly["n"], color=colors, width=0.55)
for _, r in yearly.iterrows():
    ax.text(r["app_year"], r["n"] + 1500, f"n={r['n']:,}\ndefault {100*r['dr']:.1f}%",
            ha="center", fontsize=8.5, fontweight="bold")
ax.axvline(1.5, color="black", lw=1.5, ls="--")
ax.text(0.42, ax.get_ylim()[1] * 0.55, "TRAIN\n2018-2019", ha="center",
        fontweight="bold", color="#2e6b4f", fontsize=11, rotation=90)
ax.text(1.78, ax.get_ylim()[1] * 0.55, "TEST\n2020", ha="center",
        fontweight="bold", color="#a94442", fontsize=11, rotation=90)
ax.set_xlabel("Application year")
ax.set_ylabel("Number of applications")
ax.set_title("Temporal split: train on the past, test on the future", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "eda_temporal_split.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print("7/7 temporal split done")
print("ALL EDA FIGURES SAVED TO", OUT)
