# Within-Task Complexity in the Anthropic Economic Index
### Implications for AI Productivity Estimates

**Chinonso Anyanwu** | Schwarzman Scholar, Tsinghua University  
*Working paper - June 2026*

---

## Overview

The Anthropic Economic Index reports headline productivity estimates by averaging task success rates and time savings across O*NET task clusters. This project shows that within-primitive complexity heterogeneity is large enough to bias those estimates upward - a finding with direct implications for how AI productivity gains are measured and governed.

**Core finding:** Treating all tasks within an economic primitive as homogeneous overestimates effective productivity uplift. Complexity-weighting the estimate - using a composite score over turn length, tool calls, clarification exchanges, and multi-step dependencies - produces a systematically lower figure. The bias is modest on this release (0.23%) because rising time savings partially offset falling success rates as complexity increases; the correction grows nonlinearly as deployment concentrates on complex professional workflows.

**Safety implication:** The same complexity gradient that drives productivity upside creates deployer pressure to release AI on tasks where reliability is lowest. Section 4.3 of the tech note argues that complexity-stratified performance disclosure is a feasible governance instrument to make this trade-off legible to regulators and workers.

---

## Key Results

Success rate falls monotonically with task complexity across all 3,259 O*NET task clusters (Claude.ai split, Feb 5-12 2026):

| Complexity bin | Mean success rate | 95% CI | N |
|---|---|---|---|
| Low | 77.6% | [76.6%, 78.6%] | 908 |
| Medium | 73.3% | [72.5%, 74.1%] | 912 |
| High | 68.4% | [67.7%, 69.1%] | 919 |

Conditional time-savings ratios run in the opposite direction (84.5% / 90.9% / 94.3%), creating the opposing-gradient structure that bounds the aggregate correction.

Complexity-weighted vs. naive productivity uplift:

| Group | G_naive | G_weighted | Bias (pp) | Bias % |
|---|---|---|---|---|
| All tasks | 0.6560 | 0.6545 | 0.0015 | 0.23% |
| Software Development | 0.6515 | 0.6508 | 0.0006 | 0.10% |
| Writing & Editing | 0.6522 | 0.6514 | 0.0008 | 0.13% |

*Numbers regenerate on every `python run_analysis.py`; the per-primitive breakdown is written to `outputs/tables/table5_primitive_revision.csv`.*

---

## Figures

| Figure | Description |
|---|---|
| `fig1_success_by_bin.png` | Success rate by complexity bin (all tasks) |
| `fig2_success_vs_time.png` | Cluster-level success rate vs. human task duration |
| `fig3_autonomy_by_bin.png` | AI autonomy score by complexity bin |
| `fig4_productivity_revision.png` | Naive vs. complexity-weighted uplift (all / software / writing) |
| `fig5_complexity_distribution.png` | Distribution of human task duration within each complexity bin |

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
2. Build the O*NET task cluster panel, merging five `onet_task::*` facets (human time, success, autonomy, human-with-AI time, education years)
3. Assign complexity bins (terciles of composite score: turn length, tool calls, clarification exchanges, multi-step dependencies)
4. Compute productivity revision statistics for all tasks and per-primitive subsets
5. Generate all five figures to `outputs/figures/`
6. Save the task panel and five analysis tables to `outputs/tables/`

Raw data and generated outputs are ignored by Git; they are created locally when you run the pipeline.

---

## Data

**Source:** [Anthropic Economic Index](https://huggingface.co/datasets/Anthropic/EconomicIndex), `release_2026_03_24`, Claude.ai split (Feb 5-12, 2026).  
**License:** CC-BY 4.0 (data), MIT (code in this repo).  
Raw data is not committed. Run `python run_analysis.py` to download.

---

## Repo Structure

```
complexity-in-economic-index/
├── run_analysis.py        # end-to-end pipeline
├── requirements.txt
├── src/
│   ├── data_loading.py    # HF download + 5-facet panel construction
│   ├── features.py        # complexity bins, time savings, composite score
│   ├── analysis.py        # bin summaries, productivity revision, table writers
│   └── plots.py           # figures 1-5
├── tests/
│   └── test_features.py   # pytest unit tests
├── tech_note/
│   └── complexity-note.md # full technical write-up (6 sections, ~3,000 words)
└── outputs/
    ├── figures/           # generated PNGs (not tracked)
    └── tables/            # generated CSVs (not tracked)
```

---

## Methodology

### Complexity Bins
Task clusters are assigned to low / medium / high complexity bins using terciles of a composite score:

```
C_i = 0.35 * turn_length_norm + 0.30 * tool_calls_norm + 0.20 * clarification_flag + 0.15 * multistep_flag
```

Weights reflect relative contributions to task completion variance estimated in a pilot regression. Complexity is scored from interaction structure, not outcome, to avoid circularity in success-rate comparisons.

### Productivity Revision

**Naive estimate:**  
G_naive = mean(success_pct) x mean(time_savings_ratio)

**Complexity-weighted estimate:**  
G_weighted = sum_b [ w_b * s_b * delta_b ]

where b indexes complexity bins, w_b is the share of tasks in bin b, s_b is the mean success rate, and delta_b is the mean time-savings ratio.

The **complexity-aggregation bias** = (G_naive - G_weighted) / G_naive.

### Connection to Anthropic's Published Work
- Replicates and extends the Jan 2026 finding that success rates fall with task duration
- Builds directly on the March 2026 Economic Primitives framework
- Quantifies the measurement bias implied by primitive-level averaging
- Section 4.3 connects the complexity-reliability trade-off to AI deployment governance

---

## Technical Note

The full write-up is in [`tech_note/complexity-note.md`](tech_note/complexity-note.md). It covers motivation, data, methods, all results tables, and implications for productivity estimation, labor market impact, and AI safety governance.

---

## Tests

```bash
pip install pytest
pytest tests/
```

---

## Citation

> Anyanwu, C. (2026). *Within-task complexity in the Anthropic Economic Index: preliminary stratification and implications for productivity estimates.* Working paper. https://github.com/chi1233/complexity-in-economic-index
