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
from pathlib import Path

from src.data_loading import load_raw, build_task_panel
from src.features    import (
    assign_complexity_bins, compute_time_savings,
    build_complexity_score,
)
from src.analysis    import (
    summarize_by_bin, compute_productivity_revision,
    run_full_summary,
)
from src.plots       import (
    fig1_success_by_bin, fig2_success_vs_time,
    fig3_autonomy_by_bin, fig4_productivity_revision,
)


def main() -> None:
    # ── 1. Load & build panel ────────────────────────────────────────────────
    print("[1/5] Loading data...")
    df_raw = load_raw(data_dir="data")
    panel  = build_task_panel(df_raw)

    # At this point, success_pct / autonomy_mean / ai_time_mean may be NA
    # depending on which facets have been wired up. We allow them to be
    # missing and focus the current artifact on complexity and success.
    panel, cutpoints = assign_complexity_bins(panel)
    panel  = compute_time_savings(panel)
    panel  = build_complexity_score(panel)

    print(f"      Panel: {len(panel):,} O*NET task clusters")
    print(f"      Bin cutpoints (hours): {cutpoints}")

    # ── 2. Summary statistics ────────────────────────────────────────────────
    print("\n[2/5] Summary statistics...")
    summary_all = summarize_by_bin(panel)
    print(summary_all.head())

    # ── 3. Productivity revision ─────────────────────────────────────────────
    print("\n[3/5] Productivity revision...")
    rev_all = compute_productivity_revision(panel)
    if "error" in rev_all:
        print(f"      Skipping productivity revision: {rev_all['error']}")
    else:
        print(
            f"All tasks: G_naive={rev_all['G_naive']:.4f}  "
            f"G_weighted={rev_all['G_weighted']:.4f}  "
            f"bias={rev_all['bias_pct']:.2f}%"
        )

    # ── 4. Figures ───────────────────────────────────────────────────────────
    print("\n[4/5] Generating figures...")
    fig1_success_by_bin(summary_all)
    fig2_success_vs_time(panel)
    fig3_autonomy_by_bin(summary_all)

    if "error" not in rev_all:
        fig4_productivity_revision([
            {
                "label": "All Tasks",
                **{k: rev_all[k] for k in ["G_naive", "G_weighted", "bias_pct"]},
            }
        ])
    print("      Figures saved to outputs/figures/")

    # ── 5. Save panel CSV ────────────────────────────────────────────────────
    print("\n[5/5] Saving panel CSV...")
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    panel.to_csv("outputs/tables/task_panel.csv", index=False)
    print("      Saved outputs/tables/task_panel.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
