"""
plots.py
--------
Figure generation for the complexity stratification analysis.
All figures saved to outputs/figures/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

PALETTE   = {"low": "#4e9fc5", "medium": "#f5a623", "high": "#d0021b"}
BIN_ORDER = ["low", "medium", "high"]
OUTDIR    = Path("outputs/figures")


def _setup():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False})


def _bin_value(summary_df: pd.DataFrame, metric: str, bin_name: str, col: str):
    """Safely pull a single summary value for (metric, bin); NaN if absent."""
    sel = summary_df[(summary_df["metric"] == metric) & (summary_df["bin"] == bin_name)]
    if sel.empty:
        return np.nan
    return sel[col].values[0]


def _has_metric(summary_df: pd.DataFrame, metric: str) -> bool:
    """True if at least one bin has a finite mean for this metric."""
    vals = [_bin_value(summary_df, metric, b, "mean") for b in BIN_ORDER]
    return any(np.isfinite(v) for v in vals)


def fig1_success_by_bin(summary_df: pd.DataFrame, outpath: Path = None) -> None:
    """Bar chart: success rate by complexity bin (all tasks)."""
    _setup()
    if not _has_metric(summary_df, "success_pct"):
        print("[plots] fig1 skipped: no success_pct data available.")
        return
    means = [_bin_value(summary_df, "success_pct", b, "mean")  for b in BIN_ORDER]
    ci_lo = [_bin_value(summary_df, "success_pct", b, "ci_lo") for b in BIN_ORDER]
    ci_hi = [_bin_value(summary_df, "success_pct", b, "ci_hi") for b in BIN_ORDER]
    means_a = np.array(means, dtype=float)
    yerr  = [means_a - np.array(ci_lo, dtype=float),
             np.array(ci_hi, dtype=float) - means_a]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(BIN_ORDER, means, color=[PALETTE[b] for b in BIN_ORDER],
           width=0.5, edgecolor="white", zorder=3)
    ax.errorbar(range(3), means, yerr=yerr, fmt="none",
                color="#333", capsize=5, linewidth=1.5, zorder=4)
    for i, m in enumerate(means):
        if np.isfinite(m):
            ax.text(i, m + 0.015, f"{m:.1%}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Success Rate"); ax.set_xlabel("Complexity Bin")
    ax.set_title("Figure 1 — Success Rate Falls with Task Complexity",
                 fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(outpath or OUTDIR / "fig1_success_by_bin.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()


def fig2_success_vs_time(panel: pd.DataFrame, outpath: Path = None) -> None:
    """Scatter: cluster-level success rate vs. mean human task duration."""
    _setup()
    valid = panel.dropna(subset=["human_time_mean", "success_pct"])
    if len(valid) < 2:
        print("[plots] fig2 skipped: no success_pct data available.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for b in BIN_ORDER:
        sub = panel[panel["complexity_bin"] == b]
        ax.scatter(sub["human_time_mean"], sub["success_pct"],
                   color=PALETTE[b], alpha=0.55, s=18, label=b.capitalize())
    # OLS trend line
    m, c = np.polyfit(valid["human_time_mean"], valid["success_pct"], 1)
    xs = np.linspace(valid["human_time_mean"].min(), valid["human_time_mean"].quantile(0.97), 200)
    ax.plot(xs, m * xs + c, "k--", linewidth=1.5, alpha=0.7, label=f"OLS trend")
    ax.set_xlabel("Mean Human Task Duration (hours)")
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Figure 2 — Success Rate vs. Task Complexity\n(each point = one O*NET task cluster)",
                 fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath or OUTDIR / "fig2_success_vs_time.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()


def fig3_autonomy_by_bin(summary_df: pd.DataFrame, outpath: Path = None) -> None:
    """Bar chart: mean autonomy score by complexity bin."""
    _setup()
    if not _has_metric(summary_df, "autonomy_mean"):
        print("[plots] fig3 skipped: no autonomy_mean data available.")
        return
    means = [_bin_value(summary_df, "autonomy_mean", b, "mean")  for b in BIN_ORDER]
    ci_lo = [_bin_value(summary_df, "autonomy_mean", b, "ci_lo") for b in BIN_ORDER]
    ci_hi = [_bin_value(summary_df, "autonomy_mean", b, "ci_hi") for b in BIN_ORDER]
    means_a = np.array(means, dtype=float)
    yerr  = [means_a - np.array(ci_lo, dtype=float),
             np.array(ci_hi, dtype=float) - means_a]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(BIN_ORDER, means, color=[PALETTE[b] for b in BIN_ORDER],
           width=0.5, edgecolor="white", zorder=3)
    ax.errorbar(range(3), means, yerr=yerr, fmt="none",
                color="#333", capsize=5, linewidth=1.5, zorder=4)
    for i, m in enumerate(means):
        if np.isfinite(m):
            ax.text(i, m + 0.03, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    finite_means = [m for m in means if np.isfinite(m)]
    overall = sum(finite_means) / len(finite_means)
    ax.axhline(overall, color="#888", linestyle=":", linewidth=1.2,
               label=f"Overall mean ({overall:.2f})")
    ax.set_ylabel("Mean AI Autonomy Score (1–5)"); ax.set_xlabel("Complexity Bin")
    ax.set_title("Figure 3 — AI Autonomy Increases with Task Complexity",
                 fontweight="bold", pad=10)
    ax.set_ylim(0, 5); ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath or OUTDIR / "fig3_autonomy_by_bin.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()


def fig4_productivity_revision(results: list, outpath: Path = None) -> None:
    """
    results: list of dicts with keys: label, G_naive, G_weighted, bias_pct
    """
    _setup()
    labels  = [r["label"] for r in results]
    naive   = [r["G_naive"]    * 100 for r in results]
    weighted = [r["G_weighted"] * 100 for r in results]
    x, w = np.arange(len(results)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w/2, naive,    w, color="#aac4d4", edgecolor="white", label="Naïve (G_naive)")
    ax.bar(x + w/2, weighted, w, color="#d4490e", edgecolor="white", label="Complexity-weighted (G_w)")

    for i, r in enumerate(results):
        sign = "−" if r["bias_pct"] > 0 else "+"
        ax.annotate(
            f"{sign}{abs(r['bias_pct']):.1f}%",
            xy=(i + w/2, r["G_weighted"] * 100),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=9.5, fontweight="bold",
            color="#c00" if r["bias_pct"] > 0 else "#080",
        )

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Effective Productivity Uplift (%)")
    ax.set_title("Figure 4 — Complexity-Weighted vs. Naïve Productivity Estimate",
                 fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    plt.tight_layout()
    fig.savefig(outpath or OUTDIR / "fig4_productivity_revision.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()


def fig5_complexity_distribution(panel: pd.DataFrame, outpath: Path = None) -> None:
    """Violin plot: distribution of human_time_mean within each complexity bin."""
    _setup()
    data, positions, colors, present = [], [], [], []
    for i, b in enumerate(BIN_ORDER):
        vals = panel.loc[panel["complexity_bin"] == b, "human_time_mean"].dropna()
        if len(vals) >= 2:
            data.append(vals.values)
            positions.append(i)
            colors.append(PALETTE[b])
            present.append(b)
    if not data:
        print("[plots] fig5 skipped: no human_time_mean data available.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(data, positions=positions, showmedians=True,
                          showextrema=False, widths=0.7)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.75)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#333")
        parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(len(BIN_ORDER)))
    ax.set_xticklabels([b.capitalize() for b in BIN_ORDER])
    ax.set_xlabel("Complexity Bin")
    ax.set_ylabel("Mean Human Task Duration (hours)")
    ax.set_title("Figure 5 — Task Duration Distribution by Complexity Bin",
                 fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(outpath or OUTDIR / "fig5_complexity_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
