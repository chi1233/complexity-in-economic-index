"""
analysis.py
-----------
Core analytical functions: bin summaries, productivity revision.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TABLES_DIR = Path("outputs/tables")
BIN_ORDER = ["low", "medium", "high"]


def summarize_by_bin(
    df: pd.DataFrame,
    group_col: str = "complexity_bin",
    metrics: list | None = None,
) -> pd.DataFrame:
    """
    For each complexity bin compute mean, 95% CI, and N for each metric.
    Returns tidy DataFrame: [bin, metric, mean, ci_lo, ci_hi, n].
    """
    if metrics is None:
        metrics = ["success_pct", "time_savings_ratio", "autonomy_mean", "edu_years_mean"]

    records = []
    for bin_val, grp in df.groupby(group_col, observed=True):
        for metric in metrics:
            if metric not in grp.columns:
                continue
            vals = grp[metric].dropna()
            n = len(vals)
            if n < 2:
                continue
            mean = vals.mean()
            se = vals.std() / np.sqrt(n)
            records.append({
                "bin":    bin_val,
                "metric": metric,
                "mean":   mean,
                "ci_lo":  mean - 1.96 * se,
                "ci_hi":  mean + 1.96 * se,
                "n":      n,
            })
    return pd.DataFrame(records)


def compute_productivity_revision(
    df: pd.DataFrame,
    group_col: str = "complexity_bin",
) -> dict:
    """
    Compare naive aggregate productivity uplift vs complexity-weighted version.

    G_naive    = mean(success_pct) * mean(time_savings_ratio)
    G_weighted = sum_k [ w_k * s_k * delta_k ]  where k = complexity bin

    Returns dict with scalar estimates and per-bin table.
    """
    df = df.dropna(subset=["success_pct", "time_savings_ratio"])
    if len(df) < 10:
        return {"error": "Insufficient data."}

    bins = (
        df.groupby(group_col, observed=True)
        .agg(
            w=(group_col, "count"),
            s=("success_pct", "mean"),
            delta=("time_savings_ratio", "mean"),
        )
    )
    bins["w"] = bins["w"] / bins["w"].sum()

    G_naive    = df["success_pct"].mean() * df["time_savings_ratio"].mean()
    G_weighted = (bins["w"] * bins["s"] * bins["delta"]).sum()
    bias       = G_naive - G_weighted
    bias_pct   = bias / G_naive * 100 if G_naive > 0 else np.nan

    return {
        "G_naive":    G_naive,
        "G_weighted": G_weighted,
        "bias":       bias,
        "bias_pct":   bias_pct,
        "bin_table":  bins,
    }


def compute_primitive_revisions(panel: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """Productivity revision for the full panel and per-primitive subsets.

    Runs compute_productivity_revision on All Tasks plus the software-dev and
    writing/editing keyword subsets (from src.features). Returns:
        results   list of dicts ready for plots.fig4_productivity_revision
        table     tidy DataFrame [group, n, G_naive, G_weighted, bias, bias_pct]
    Groups with insufficient data are skipped (and noted in the table).
    """
    from src.features import filter_primitive, SOFTWARE_KW, WRITING_KW  # noqa: PLC0415

    def _subset(keywords):
        mask = panel["cluster_name"].str.lower().str.contains(
            "|".join(keywords), na=False
        )
        # filter_primitive asserts >= 10 rows; only call when that holds.
        return filter_primitive(panel, keywords) if mask.sum() >= 10 else pd.DataFrame()

    groups = [
        ("All Tasks",   panel),
        ("Software Dev", _subset(SOFTWARE_KW)),
        ("Writing",      _subset(WRITING_KW)),
    ]

    results, records = [], []
    for label, df_sub in groups:
        if len(df_sub) < 10:
            records.append({"group": label, "n": len(df_sub), "note": "insufficient data"})
            continue
        rev = compute_productivity_revision(df_sub)
        if "error" in rev:
            records.append({"group": label, "n": len(df_sub), "note": rev["error"]})
            continue
        results.append({
            "label":      label,
            "G_naive":    rev["G_naive"],
            "G_weighted": rev["G_weighted"],
            "bias_pct":   rev["bias_pct"],
        })
        records.append({
            "group":      label,
            "n":          len(df_sub),
            "G_naive":    rev["G_naive"],
            "G_weighted": rev["G_weighted"],
            "bias":       rev["bias"],
            "bias_pct":   rev["bias_pct"],
        })
    return results, pd.DataFrame(records)


def _quantiles_by_bin(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "complexity_bin",
    quantiles: tuple = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> pd.DataFrame:
    """Per-bin quantiles of a metric. Returns tidy [bin, n, q10, q25, ...]."""
    records = []
    for bin_val, grp in df.groupby(group_col, observed=True):
        vals = grp[metric].dropna() if metric in grp.columns else pd.Series(dtype=float)
        if len(vals) < 2:
            continue
        row = {"bin": bin_val, "n": len(vals)}
        for q in quantiles:
            row[f"q{int(q * 100):02d}"] = vals.quantile(q)
        records.append(row)
    out = pd.DataFrame(records)
    if not out.empty:
        order = {b: i for i, b in enumerate(BIN_ORDER)}
        out = out.sort_values("bin", key=lambda s: s.map(order)).reset_index(drop=True)
    return out


def save_tables(panel: pd.DataFrame, tables_dir: Path = TABLES_DIR) -> dict:
    """Compute and save the analysis tables to CSV.

    Returns a dict of {name: path} for the tables written.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    # Table 1: summary by complexity bin (mean/CI/N per metric).
    t1 = summarize_by_bin(panel)
    p1 = tables_dir / "table1_summary_by_bin.csv"
    t1.to_csv(p1, index=False)
    written["summary_by_bin"] = p1

    # Table 2: productivity revision (per-bin weights + scalar estimates).
    rev = compute_productivity_revision(panel)
    p2 = tables_dir / "table2_productivity_revision.csv"
    if "error" in rev:
        pd.DataFrame([{"note": rev["error"]}]).to_csv(p2, index=False)
    else:
        t2 = rev["bin_table"].reset_index()
        for k in ("G_naive", "G_weighted", "bias", "bias_pct"):
            t2[k] = rev[k]
        t2.to_csv(p2, index=False)
    written["productivity_revision"] = p2

    # Table 3: success-rate quantiles by bin.
    t3 = _quantiles_by_bin(panel, "success_pct")
    p3 = tables_dir / "table3_success_quantiles_by_bin.csv"
    t3.to_csv(p3, index=False)
    written["success_quantiles_by_bin"] = p3

    # Table 4: autonomy quantiles by bin.
    t4 = _quantiles_by_bin(panel, "autonomy_mean")
    p4 = tables_dir / "table4_autonomy_quantiles_by_bin.csv"
    t4.to_csv(p4, index=False)
    written["autonomy_quantiles_by_bin"] = p4

    # Table 5: productivity revision by primitive (all / software / writing).
    _, t5 = compute_primitive_revisions(panel)
    p5 = tables_dir / "table5_primitive_revision.csv"
    t5.to_csv(p5, index=False)
    written["primitive_revision"] = p5

    for _, path in written.items():
        print(f"[analysis]   saved {path}")
    return written


def run_full_summary(panel: pd.DataFrame) -> None:
    """Print the key summary statistics used in the tech note."""
    print("=== Full Panel Summary ===")
    summary = panel.groupby("complexity_bin", observed=True).agg(
        n=("cluster_name", "count"),
        success_mean=("success_pct", "mean"),
        time_savings_mean=("time_savings_ratio", "mean"),
        autonomy_mean=("autonomy_mean", "mean"),
        human_time_mean=("human_time_mean", "mean"),
    ).round(3)
    print(summary)

    cuts = (
        panel.groupby("complexity_bin", observed=True)["human_time_mean"]
        .agg(["min", "max"])
        .round(2)
    )
    print("\n=== Bin Cutpoints (hours) ===")
    print(cuts)

    print("\n=== Productivity Revision ===")
    from src.features import filter_primitive, SOFTWARE_KW, WRITING_KW  # noqa: PLC0415
    sw = (
        filter_primitive(panel, SOFTWARE_KW)
        if panel["cluster_name"].str.lower().str.contains("|".join(SOFTWARE_KW), na=False).any()
        else pd.DataFrame()
    )
    wr = (
        filter_primitive(panel, WRITING_KW)
        if panel["cluster_name"].str.lower().str.contains("|".join(WRITING_KW), na=False).any()
        else pd.DataFrame()
    )

    for label, df_sub in [("All tasks", panel), ("Software Dev", sw), ("Writing", wr)]:
        if len(df_sub) < 10:
            print(f"{label}: insufficient data")
            continue
        r = compute_productivity_revision(df_sub)
        if "error" in r:
            print(f"{label}: {r['error']}")
            continue
        print(
            f"{label}: N={len(df_sub):,}  G_naive={r['G_naive']:.4f}  "
            f"G_weighted={r['G_weighted']:.4f}  bias={r['bias_pct']:.2f}%"
        )
