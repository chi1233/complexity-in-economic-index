"""
features.py
-----------
Feature engineering: complexity bins, time-savings ratio, composite score.
"""
import numpy as np
import pandas as pd

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

    Robust to low variation: if we truly can't form multiple bins, fall back
    to a single "medium" bin so the rest of the pipeline still runs.
    Returns (df_with_bins, cutpoints_dict).
    """
    labels = labels or BIN_LABELS
    df = df.copy()
    assert col in df.columns, f"Column '{col}' not found."

    # If almost no variation, fall back to a single bin
    non_na = df[col].dropna()
    if non_na.nunique() < 3 or len(non_na) < 3:
        df["complexity_bin"] = "medium"
        quantiles = [0.0, 0.5, 1.0]
        cutpoints = df[col].quantile(quantiles).to_dict()
        return df, cutpoints

    # Rank-based qcut for stability
    ranks = df[col].rank(method="average")
    try:
        cats = pd.qcut(ranks, q=n_bins, duplicates="drop")
        categories = list(cats.cat.categories)
    except ValueError:
        # As an ultra-conservative fallback, assign all to medium
        df["complexity_bin"] = "medium"
        quantiles = [0.0, 0.5, 1.0]
        cutpoints = df[col].quantile(quantiles).to_dict()
        return df, cutpoints

    # Map pandas interval categories -> human-readable labels (up to available)
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


def _safe_z(col: pd.Series) -> pd.Series | None:
    """Simple z-score implementation that avoids SciPy edge cases."""
    col = col.astype(float).dropna()
    if len(col) < 2:
        return None
    std = col.std(ddof=0)
    if std == 0 or np.isnan(std):
        return None
    mean = col.mean()
    return (col - mean) / std


def build_complexity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple complexity_score based on human_time_mean only.

    This avoids SciPy's zscore pitfalls when columns are all-NaN or non-numeric.
    """
    df = df.copy()
    df["complexity_score"] = np.nan
    if "human_time_mean" not in df.columns:
        return df

    z = _safe_z(df["human_time_mean"])
    if z is not None:
        df.loc[z.index, "complexity_score"] = z
    return df
