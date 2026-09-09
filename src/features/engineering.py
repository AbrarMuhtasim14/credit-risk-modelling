"""
Domain feature engineering and data cleaning (pure functions, no global fitting).

Every statistic used here (income cap) is passed in from the caller, which
computes it on the training window only. This module never sees the test set's
distribution.

Steps:
1. DAYS_EMPLOYED sentinel (365243) -> NaN + binary anomaly flag.
2. Income winsorization at a caller-supplied cap (99.9th pct of train years).
3. ORGANIZATION_TYPE grouped 58 -> 11 sectors.
4. XNA tokens in CODE_GENDER / ORGANIZATION_TYPE -> NaN.
5. Drop 30 redundant columns (28 building _MEDI/_MODE + 2 social-circle 60-DPD).
6. Engineer domain ratios and EXT_SOURCE aggregates.
"""

import logging

import numpy as np
import pandas as pd

from src.config import (
    COLS_TO_DROP,
    DAYS_EMPLOYED_ANOM_COL,
    DAYS_EMPLOYED_SENTINEL,
    EXT_SOURCE_COLS,
)

logger = logging.getLogger(__name__)

ORGANIZATION_SECTOR_RULES = [
    ("Business Entity", ["BUSINESS ENTITY"]),
    ("Industry", ["INDUSTRY"]),
    ("Trade", ["TRADE"]),
    ("Transport", ["TRANSPORT"]),
    ("Government/Public", ["GOVERNMENT", "POLICE", "MILITARY", "SECURITY", "LEGAL SERVICES"]),
    ("Education/Healthcare", ["MEDICINE", "KINDERGARTEN", "SCHOOL", "UNIVERSITY"]),
    ("Financial/RealEstate", ["BANK", "INSURANCE", "REALTOR"]),
    ("Services/Hospitality", ["HOTEL", "RESTAURANT", "SERVICES", "CLEANING", "POSTAL"]),
    ("Infrastructure/Agriculture", ["CONSTRUCTION", "HOUSING", "AGRICULTURE", "ELECTRICITY", "TELECOM"]),
]


def group_organization_type(value) -> str:
    """Map one of 58 raw ORGANIZATION_TYPE categories into 11 sectors."""
    if pd.isna(value):
        return "Missing"
    text = str(value).strip().upper()
    if text in ("", "XNA", "NAN", "NONE"):
        return "Missing"
    for sector, keywords in ORGANIZATION_SECTOR_RULES:
        if any(k in text for k in keywords):
            return sector
    return "Other"


def clean_and_engineer(df: pd.DataFrame, income_cap: float) -> pd.DataFrame:
    """
    Clean anomalies and engineer domain features.

    Parameters
    ----------
    df : raw application dataframe (all years, before splitting).
    income_cap : 99.9th percentile of AMT_INCOME_TOTAL computed on the
        TRAINING YEARS ONLY by the caller (leakage control).
    """
    df = df.copy()

    # -- 1. DAYS_EMPLOYED sentinel correction -------------------------------
    sentinel_mask = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    df[DAYS_EMPLOYED_ANOM_COL] = sentinel_mask.astype(np.int8)
    df.loc[sentinel_mask, "DAYS_EMPLOYED"] = np.nan
    logger.info(
        "DAYS_EMPLOYED sentinel corrected: %d rows (%.2f%%)",
        int(sentinel_mask.sum()), 100.0 * sentinel_mask.mean(),
    )

    # -- 2. Income winsorization (cap fitted on train years by caller) ------
    n_clipped = int((df["AMT_INCOME_TOTAL"] > income_cap).sum())
    df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=income_cap)
    logger.info("Income winsorized at $%s: %d rows clipped", f"{income_cap:,.0f}", n_clipped)

    # -- 3. ORGANIZATION_TYPE grouping --------------------------------------
    df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].map(group_organization_type)

    # -- 4. XNA token cleanup ------------------------------------------------
    df["CODE_GENDER"] = df["CODE_GENDER"].replace({"XNA": np.nan})

    # -- 5. Drop redundant columns -------------------------------------------
    drop_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_present)
    logger.info("Dropped %d redundant columns", len(drop_present))

    # -- 6. Domain feature engineering ---------------------------------------
    eps = 1e-6

    # Financial burden ratios
    df["CREDIT_INCOME_PERCENT"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["ANNUITY_INCOME_PERCENT"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + eps)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + eps)
    df["CREDIT_GOODS_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"].fillna(1) + eps)

    # Demographics / tenure
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + eps)
    df["PHONE_TO_BIRTH_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / (df["DAYS_BIRTH"] + eps)
    df["REGISTRATION_TO_BIRTH_RATIO"] = df["DAYS_REGISTRATION"] / (df["DAYS_BIRTH"] + eps)
    df["ID_PUBLISH_TO_BIRTH_RATIO"] = df["DAYS_ID_PUBLISH"] / (df["DAYS_BIRTH"] + eps)
    df["CAR_TO_AGE_RATIO"] = df["OWN_CAR_AGE"] / (df["AGE_YEARS"] + eps)

    # External bureau score aggregates (nan-safe)
    ext_present = [c for c in EXT_SOURCE_COLS if c in df.columns]
    ext = df[ext_present].to_numpy(dtype=np.float64)
    valid_counts = np.sum(~np.isnan(ext), axis=1)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df["EXT_SOURCES_MEAN"] = np.nanmean(ext, axis=1)
        ext_std = np.nanstd(ext, axis=1)
        ext_std[valid_counts < 2] = np.nan
        df["EXT_SOURCES_STD"] = ext_std
        ext_min = np.nanmin(ext, axis=1)
        ext_max = np.nanmax(ext, axis=1)
        ext_min[valid_counts == 0] = np.nan
        ext_max[valid_counts == 0] = np.nan
        df["EXT_SOURCES_MIN"] = ext_min
        df["EXT_SOURCES_MAX"] = ext_max

    if len(ext_present) == 3:
        df["EXT_SOURCES_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        df["EXT_SOURCE_1_2_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
        df["EXT_SOURCE_2_3_PROD"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        df["EXT_SOURCE_1_3_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]

    # Document flag aggregate
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1).astype(np.int8)

    logger.info("Feature engineering complete: %d columns", df.shape[1])
    return df
