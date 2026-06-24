# Within-Task Complexity in the Anthropic Economic Index
### Implications for AI Productivity Estimates

**Chinonso Anyanwu** | Schwarzman Scholar, Tsinghua University  
*Working paper — June 2026*

---

## Overview

The Anthropic Economic Index reports headline productivity estimates by averaging task success rates and time savings across O*NET task clusters. This project shows that within-primitive complexity heterogeneity is large enough to bias those estimates upward — a finding with direct implications for how AI productivity gains are measured and governed.

**Core finding:** Treating all tasks within an economic primitive as homogeneous overestimates effective productivity uplift. Complexity-weighting the estimate — using tertile bins of mean human task duration — produces a systematically lower figure, particularly for software development tasks.

---

## Key Results

| Group | G_naive | G_weighted | Bias |
|---|---|---|---|
| All tasks | see outputs | see outputs | computed on run |
| Software Development | see outputs | see outputs | computed on run |
| Writing & Editing | see outputs | see outputs | computed on run |

*Run `python run_analysis.py` to populate with real values.*

---

## Figures

| Figure | Description |
|---|---|
| `fig1_success_by_bin.png` | Success rate by complexity bin (all tasks) |
| `fig2_success_vs_time.png` | Cluster-level success rate vs. human task duration |
| `fig3_autonomy_by_bin.png` | AI autonomy score by complexity bin |
| `fig4_productivity_revision.png` | Naïve vs. complexity-weighted productivity estimate |

---

## Setup / Quick Start

```bash
git clone https://github.com/chi1233/complexity-in-economic-index
cd complexity-in-economic-index
pip install -r requirements.txt
python run_analysis.py
```

`run_analysis.py` will:
1. Download the AEI dataset directly from Hugging Face (~90MB)
2. Build the O*NET task cluster panel
3. Assign complexity bins (tertiles of mean human task duration)
4. Compute productivity revision statistics when `success_pct` and `time_savings_ratio` are available
5. Generate all four figures to `outputs/figures/`
6. Save the task panel to `outputs/tables/task_panel.csv`

Raw data and generated outputs are ignored by Git; they are created locally when you run the pipeline.

---

## Data

**Source:** [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex), `release_2026_03_24`, Claude.ai split (Feb 5–12, 2026).  
**License:** CC-BY 4.0 (data), MIT (code in this repo).  
Raw data is not committed. Run `python run_analysis.py` to download.

---

## Repo Structure

```
complexity-in-economic-index/
├── run_analysis.py        # end-to-end pipeline
├── requirements.txt
├── src/
│   ├── data_loading.py    # HF download + panel construction
│   ├── features.py        # complexity bins, time savings, composite score
│   ├── analysis.py        # bin summaries, productivity revision
│   └── plots.py           # figures 1–4
├── tests/
│   └── test_features.py   # pytest unit tests
└── outputs/
    └── figures/           # generated PNGs (not tracked)
```

---

## Methodology

### Complexity Bins
Task clusters are assigned to low / medium / high complexity bins using tertiles of `human_time_mean` (mean estimated human hours to complete the task without AI). Bin cutpoints are computed within the full panel and reported in run output.

### Productivity Revision

**Naïve estimate:**  
G_naive = mean(success_pct) × mean(time_savings_ratio)

**Complexity-weighted estimate:**  
G_weighted = Σ_k [ w_k · s_k · Δ_k ]

where k indexes complexity bins, w_k is the share of tasks in bin k, s_k is the mean success rate, and Δ_k is the mean time-savings ratio.

The **complexity-aggregation bias** = (G_naive − G_weighted) / G_naive.

### Connection to Anthropic's Published Work
- Replicates and extends the Jan 2026 finding that success rates fall with task duration
- Builds directly on the March 2026 Economic Primitives framework
- Quantifies the measurement bias implied by primitive-level averaging

---

## Technical Note

The main write-up for this project is in [`tech_note/complexity-note.md`](tech_note/complexity-note.md). It provides a 2–3 page summary of the motivation, data, methods, and implications of the within-task complexity analysis, with clear notes on which metrics are fully wired today and which are planned extensions.

---

## Tests

```bash
pip install pytest
pytest tests/
```

---

## Citation

> Anyanwu, C. (2026). *Within-task complexity in the Anthropic Economic Index: preliminary stratification and implications for productivity estimates.* Working paper.
