"""
features.py
-----------
Feature engineering: complexity bins, time-savings ratio, composite score.
"""
import numpy as np
import pandas as pd
from scipy import stats

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
    labels: list | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assign tertile-based complexity bins.

    Uses the rank of `col` to avoid pandas.qcut label/edge mismatches when
    many values are identical. Returns (df_with_bins, cutpoints_dict).
    """
    labels = labels or BIN_LABELS
    df = df.copy()
    assert col in df.columns, f"Column '{col}' not found."

    # Rank-based qcut for stability
    ranks = df[col].rank(method="average")
    cats = pd.qcut(ranks, q=n_bins, duplicates="drop")
    categories = list(cats.cat.categories)
    actual_bins = len(categories)
    if actual_bins < 2:
        raise ValueError("Not enough variation in human_time_mean to form bins.")
    if actual_bins > len(labels):
        raise ValueError("Not enough labels for the number of bins.")

    # Map pandas interval categories -> human-readable labels
    mapping = {cat: labels[i] for i, cat in enumerate(categories)}
    df["complexity_bin"] = cats.cat.rename_categories(mapping)

    # Still compute nominal cutpoints for reporting (based on original values)
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
    Composite z-score: 0.5*human_time + 0.3*edu_years + 0.2*autonomy.
    Uses available columns; gracefully skips missing ones.
    """
    df = df.copy()
    z_cols: list[tuple[str, float]] = []

    if "human_time_mean" in df.columns:
        df["z_human_time"] = stats.zscore(
            df["human_time_mean"].fillna(df["human_time_mean"].median())
        )
        z_cols.append(("z_human_time", 0.5))

    if "edu_years_mean" in df.columns:
        df["z_edu"] = stats.zscore(
            df["edu_years_mean"].fillna(df["edu_years_mean"].median())
        )
        z_cols.append(("z_edu", 0.3))

    if "autonomy_mean" in df.columns:
        df["z_autonomy"] = stats.zscore(
            df["autonomy_mean"].fillna(df["autonomy_mean"].median())
        )
        z_cols.append(("z_autonomy", 0.2))

    if z_cols:
        total_w = sum(w for _, w in z_cols)
        df["complexity_score"] = sum(
            df[col] * (w / total_w) for col, w in z_cols
        )
    else:
        df["complexity_score"] = np.nan

    return df
