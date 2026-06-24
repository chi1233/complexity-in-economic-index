"""
tests/test_features.py
----------------------
Unit tests for feature engineering functions.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import assign_complexity_bins, compute_time_savings, build_complexity_score


def make_panel(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "cluster_name":    [f"task_{i}" for i in range(n)],
        "human_time_mean": rng.exponential(3, n),
        "success_pct":     rng.uniform(0.3, 1.0, n),
        "ai_time_mean":    rng.uniform(5, 60, n),   # minutes
        "autonomy_mean":   rng.uniform(2, 5, n),
        "edu_years_mean":  rng.uniform(10, 16, n),
    })


def test_bin_assignment_no_nulls():
    panel, _ = assign_complexity_bins(make_panel())
    assert panel["complexity_bin"].isna().sum() == 0


def test_bin_assignment_three_labels():
    panel, _ = assign_complexity_bins(make_panel())
    assert set(panel["complexity_bin"].cat.categories) == {"low", "medium", "high"}


def test_time_savings_bounds():
    panel = compute_time_savings(make_panel())
    assert panel["time_savings_ratio"].between(0, 1).all()


def test_complexity_score_finite():
    panel = build_complexity_score(make_panel())
    assert panel["complexity_score"].notna().all()
    assert np.isfinite(panel["complexity_score"]).all()


def test_productivity_revision_direction():
    """G_weighted <= G_naive when high-complexity bins have lower success."""
    from src.analysis import compute_productivity_revision
    rng = np.random.default_rng(0)
    n = 300
    human_time = rng.exponential(3, n)
    # success declines with task duration: ensures bias > 0
    success = np.clip(0.9 - 0.1 * human_time + rng.normal(0, 0.05, n), 0.1, 1.0)
    panel = pd.DataFrame({"cluster_name": [f"t{i}" for i in range(n)],
                          "human_time_mean": human_time, "success_pct": success,
                          "ai_time_mean": rng.uniform(5, 30, n)})
    panel, _ = assign_complexity_bins(panel)
    panel    = compute_time_savings(panel)
    r = compute_productivity_revision(panel)
    assert r["G_weighted"] <= r["G_naive"] + 1e-9  # weighted ≤ naive
