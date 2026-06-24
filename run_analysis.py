"""
run_analysis.py
---------------
End-to-end pipeline: download data → build panel → compute bins →
generate all figures → print summary statistics.

Usage:
    python run_analysis.py

Output:
    outputs/figures/fig1_success_by_bin.png
    outputs/figures/fig2_success_vs_time.png
    outputs/figures/fig3_autonomy_by_bin.png
    outputs/figures/fig4_productivity_revision.png
    outputs/tables/task_panel.csv
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loading import load_raw, build_task_panel
from src.features    import (
    assign_complexity_bins, compute_time_savings,
    build_complexity_score, filter_primitive,
    SOFTWARE_KW, WRITING_KW,
)
from src.analysis import summarize_by_bin, compute_productivity_revision
from src.plots    import (
    fig1_success_by_bin, fig2_success_vs_time,
    fig3_autonomy_by_bin, fig4_productivity_revision,
)


def main():
    # ── 1. Load & build panel ────────────────────────────────────────────────
    print("[1/5] Loading data...")
    df_raw = load_raw(data_dir="data")
    panel  = build_task_panel(df_raw)
    panel  = panel.dropna(subset=["human_time_mean", "success_pct"])
    panel, cutpoints = assign_complexity_bins(panel)
    panel  = compute_time_savings(panel)
    panel  = build_complexity_score(panel)
    print(f"      Panel: {len(panel):,} O*NET task clusters")
    print(f"      Bin cutpoints (hours): {cutpoints}")

    # ── 2. Primitive subsets ─────────────────────────────────────────────────
    sw = filter_primitive(panel, SOFTWARE_KW)
    wr = filter_primitive(panel, WRITING_KW)
    print(f"      Software Dev: {len(sw):,} clusters | Writing: {len(wr):,} clusters")

    # ── 3. Summary statistics ────────────────────────────────────────────────
    print("\n[2/5] Summary statistics...")
    summary_all = summarize_by_bin(panel)
    summary_sw  = summarize_by_bin(sw)
    summary_wr  = summarize_by_bin(wr)

    print("\n=== Full Panel ===")
    print(panel.groupby("complexity_bin", observed=True).agg(
        n=("cluster_name", "count"),
        success_mean=("success_pct", "mean"),
        time_savings_mean=("time_savings_ratio", "mean"),
        autonomy_mean=("autonomy_mean", "mean"),
        human_time_mean=("human_time_mean", "mean"),
    ).round(3))

    # ── 4. Productivity revision ─────────────────────────────────────────────
    print("\n[3/5] Productivity revision...")
    rev_all = compute_productivity_revision(panel)
    rev_sw  = compute_productivity_revision(sw)
    rev_wr  = compute_productivity_revision(wr)

    print("\n=== Productivity Revision ===")
    for label, r in [("All tasks", rev_all), ("Software Dev", rev_sw), ("Writing", rev_wr)]:
        print(f"{label}: G_naive={r['G_naive']:.4f}  G_weighted={r['G_weighted']:.4f}  "
              f"bias={r['bias_pct']:.2f}%")

    # ── 5. Figures ───────────────────────────────────────────────────────────
    print("\n[4/5] Generating figures...")
    fig1_success_by_bin(summary_all)
    fig2_success_vs_time(panel)
    fig3_autonomy_by_bin(summary_all)
    fig4_productivity_revision([
        {"label": "All Tasks",       **{k: rev_all[k] for k in ["G_naive","G_weighted","bias_pct"]}},
        {"label": "Software Dev",    **{k: rev_sw[k]  for k in ["G_naive","G_weighted","bias_pct"]}},
        {"label": "Writing & Editing", **{k: rev_wr[k] for k in ["G_naive","G_weighted","bias_pct"]}},
    ])
    print("      Figures saved to outputs/figures/")

    # ── 6. Save panel CSV ────────────────────────────────────────────────────
    print("\n[5/5] Saving panel CSV...")
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    panel.to_csv("outputs/tables/task_panel.csv", index=False)
    print("      Saved outputs/tables/task_panel.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
