"""
analysis.py
-----------
Core analytical functions: bin summaries, productivity revision.
"""
import numpy as np
import pandas as pd


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
