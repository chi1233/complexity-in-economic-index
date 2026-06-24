"""
features.py
-----------
Feature engineering: complexity bins, time-savings ratio, composite score.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple

BIN_LABELS = ["low", "medium", "high"]

# Primitive keyword lists
SOFTWARE_KW = ["software", "code", "debug", "programming", "develop", "script", "database"]
WRITING_KW  = ["writ", "edit", "draft", "document", "report", "proofread", "copyedit"]


def filter_primitive(df: pd.DataFrame, keywords: list) -> pd.DataFrame:
    """Return rows whose cluster_name matches any keyword (case-insensitive)."""
    mask = df["cluster_name"].str.lower().str.contains("|".join(keywords), na=False)
    result = df[mask].copy()
    assert len(result) >= 10, f"Too few rows ({len(result)}) for reliable analysis."
    return result


def assign_complexity_bins(
    df: pd.DataFrame,
    col: str = "human_time_mean",
    n_bins: int = 3,
    labels: list = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Assign tertile-based complexity bins within a (pre-filtered) dataframe.
    Returns (df_with_bins, cutpoints_dict).
    """
    labels = labels or BIN_LABELS
    df = df.copy()
    assert col in df.columns, f"Column '{col}' not found."
    df["complexity_bin"] = pd.qcut(
        df[col], q=n_bins, labels=labels, duplicates="drop"
    )
    quantiles = [i / n_bins for i in range(n_bins + 1)]
    cutpoints = df[col].quantile(quantiles).to_dict()
    return df, cutpoints


def compute_time_savings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implied time-savings ratio = 1 - (ai_time_mean_hrs / human_time_mean).
    Clips to [0, 1]. Requires ai_time_mean (minutes) and human_time_mean (hours).
    """
    df = df.copy()
    if "ai_time_mean" not in df.columns:
        df["time_savings_ratio"] = np.nan
        return df
    ai_hrs = df["ai_time_mean"] / 60.0
    df["time_savings_ratio"] = (
        1 - ai_hrs / df["human_time_mean"].clip(lower=1e-3)
    ).clip(0, 1)
    return df


def build_complexity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite z-score: 0.5*human_time + 0.3*edu_years + 0.2*opus_indicator.
    Uses available columns; gracefully skips missing ones.
    """
    df = df.copy()
    weights, z_cols = [], []

    if "human_time_mean" in df.columns:
        df["z_human_time"] = stats.zscore(df["human_time_mean"].fillna(df["human_time_mean"].median()))
        z_cols.append(("z_human_time", 0.5))

    if "edu_years_mean" in df.columns:
        df["z_edu"] = stats.zscore(df["edu_years_mean"].fillna(df["edu_years_mean"].median()))
        z_cols.append(("z_edu", 0.3))

    if "autonomy_mean" in df.columns:
        df["z_autonomy"] = stats.zscore(df["autonomy_mean"].fillna(df["autonomy_mean"].median()))
        z_cols.append(("z_autonomy", 0.2))

    if z_cols:
        total_w = sum(w for _, w in z_cols)
        df["complexity_score"] = sum(df[col] * (w / total_w) for col, w in z_cols)
    else:
        df["complexity_score"] = np.nan

    return df
