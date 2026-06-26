# Within-Task Complexity in the Anthropic Economic Index

## 1. Introduction

Anthropic's Economic Index provides a large-scale measurement of how AI systems like Claude reshape work by tracking success rates, time savings, and autonomy across millions of real user interactions. The core unit of analysis in the Index is the economic primitive: an O*NET task cluster such as "modify and debug software" or "write, edit, and revise documents".

The headline productivity estimates reported in Anthropic's public work average over all tasks within a primitive. For example, a single software development primitive may mix quick one-off bug fixes with multi-hour refactoring work. If success rates and time savings vary systematically with task complexity inside each primitive, then primitive-level averages can misstate the effective productivity uplift and the reliability of AI assistance at the frontier of difficulty.

This note documents a preliminary stratification of within-primitive complexity using the publicly released Anthropic Economic Index dataset. The goal is not to produce a final, production-ready estimate, but to demonstrate a clear and extensible measurement pipeline that can be refined as additional facets (success, autonomy, human-with-AI time) are wired into the task panel.

---

## 2. Data and Methods

### 2.1 Data Source

- **Dataset:** Anthropic Economic Index, `release_2026_03_24`, Claude.ai split (Feb 5–12, 2026).
- **Unit of observation:** O*NET task clusters (e.g., "modify and debug software programs", "draft correspondence and reports"). After merging facets, the panel contains 3,259 task clusters.
- **Variables wired into the panel** (all at `geo_id == 'GLOBAL'`):
  - `human_time_mean` — mean estimated human hours to complete the task without AI (`onet_task::human_only_time`).
  - `success_pct` — task success rate, taken as the share of `::yes` outcomes from the `onet_task::task_success` categorical distribution.
  - `autonomy_mean` — mean AI autonomy score, 1–5 (`onet_task::ai_autonomy`).
  - `ai_time_mean` — mean human-with-AI time in minutes (`onet_task::human_with_ai_time`).
  - `edu_years_mean` — mean human education years (`onet_task::human_education_years`).

The facets do not share a common cluster-name suffix: most use `::value`, but `task_success` encodes a `::yes`/`::no`/`::not_classified` distribution. The panel therefore joins all facets on the suffix-stripped task description and reads the success rate from the `::yes` rows.

The current implementation constructs a task-cluster-level panel with one row per O*NET task cluster and the following columns:

- `cluster_name`: O*NET task description.
- `human_time_mean`: mean human-only task duration in hours.
- `success_pct`: task success rate (0–1).
- `autonomy_mean`: mean AI autonomy score (1–5).
- `ai_time_mean`: mean human-with-AI task time (minutes).
- `edu_years_mean`: mean human education years.
- `complexity_bin`: categorical bin (low/medium/high) based on tertiles of `human_time_mean`.
- `complexity_score`: a normalized z-score of `human_time_mean`.
- `time_savings_ratio`: implied time savings, `1 - (ai_time_mean/60) / human_time_mean`, clipped to [0, 1].

### 2.2 Complexity Stratification

The current pipeline defines complexity using `human_time_mean`, Anthropic's own estimate of the time required for a human to complete the task without AI. For each task cluster, we compute:

- **Complexity bins:**
  - Low, medium, and high complexity bins are defined as tertiles of `human_time_mean` across all task clusters.
  - When variation is limited (e.g., in small test panels), the code falls back to a single "medium" bin to ensure the pipeline remains runnable.

- **Complexity score:**
  - A simple standardized score based on `human_time_mean`:
    - \( z_i = (t_i - \bar{t}) / \sigma_t \), where \(t_i\) is the mean human-only time for task cluster \(i\).
  - This `complexity_score` is intended as a building block for future models (e.g., regressions of success on complexity or interaction terms with user tenure) once more facets are wired into the panel.

The analysis script `run_analysis.py` orchestrates the following steps:

1. Download the AEI CSV (if needed) and cache it under `data/`.
2. Build the task panel using `build_task_panel` in `src/data_loading.py`.
3. Assign complexity bins and compute `complexity_score` using `src/features.py`.
4. Compute bin-level summaries using `src.analysis.summarize_by_bin`.
5. Compute complexity-weighted productivity revisions for the full panel and per-primitive subsets via `src.analysis.compute_productivity_revision` and `compute_primitive_revisions`.
6. Generate all five figures and save the task panel plus five analysis tables under `outputs/`.

---

## 3. Results

All five `onet_task::*` facets are wired into the panel, so the figures and the productivity revision report actual values. The headline pattern is a monotonic decline in success rate as task complexity rises.

### 3.1 Figure 1 — Success Rate by Complexity Bin

Mean task success rate falls across complexity bins:

| Complexity bin | Mean success rate (95% CI) | Mean human time (hrs) | N |
|---|---|---|---|
| Low | 77.6% (76.6–78.6%) | < 1.85 | 908 |
| Medium | 73.3% (72.5–74.1%) | 1.85–3.39 | 912 |
| High | 68.4% (67.7–69.1%) | > 3.39 | 919 |

The ~9-point gap between low- and high-complexity tasks is consistent with Anthropic's own reports that success rates decline with task duration. Confidence intervals are non-overlapping, so the gradient is well-identified at this sample size.

### 3.2 Figure 2 — Success vs. Human Task Duration

Figure 2 is a scatter plot of cluster-level success rates against `human_time_mean`, with points colored by complexity bin and an overlaid OLS trend line. The negative slope makes the relationship between task duration and success rate visually salient and shows that the longest-duration clusters contribute most to any productivity bias.

### 3.3 Figure 3 — Autonomy by Complexity Bin

Mean AI autonomy score rises modestly with complexity (low 3.29 → medium 3.36 → high 3.49 on the 1–5 scale). Users grant Claude somewhat more autonomy on more complex tasks even as success rates fall — a combination worth flagging from an oversight perspective.

### 3.4 Figure 4 — Complexity-Weighted vs. Naïve Productivity Estimate

The final figure compares a naïve productivity estimate with a complexity-weighted alternative:

- **Naïve uplift:**
  - \( G_{\text{naive}} = \mathbb{E}[\text{success\_pct}] \times \mathbb{E}[\text{time\_savings\_ratio}] \).
- **Complexity-weighted uplift:**
  - \( G_{\text{weighted}} = \sum_k w_k \cdot s_k \cdot \Delta_k \), where \(k\) indexes complexity bins, \(w_k\) is the share of tasks in bin \(k\), \(s_k\) is the bin-specific mean success rate, and \(\Delta_k\) is the bin-specific mean time-savings ratio.

For the full panel, \( G_{\text{naive}} = 0.656 \) and \( G_{\text{weighted}} = 0.654 \), a bias of +0.23%. The bias is small because the time-savings ratio rises with complexity (low 0.84 → high 0.94), partially offsetting the declining success rate. The keyword-defined primitive subsets show similarly modest biases — software development +0.10% (\(G_{\text{naive}}=0.652\)) and writing/editing +0.13% (\(G_{\text{naive}}=0.652\)) — and appear alongside the full-panel bars in Figure 4. The full breakdown is written to `outputs/tables/table5_primitive_revision.csv`.

### 3.5 Figure 5 — Task Duration Distribution by Complexity Bin

Figure 5 is a violin plot of `human_time_mean` within each complexity bin, illustrating how the tertile cutpoints (≈1.85 hrs and ≈3.39 hrs) partition the duration distribution and how much within-bin spread remains — context for interpreting the bin-level means above.

---

## 4. Implications for Productivity Estimates

The core conceptual point of this project is that the distribution of task complexity within each economic primitive matters for how we interpret aggregate productivity statistics. If high-complexity clusters are harder for Claude to complete successfully and yield smaller time savings when they do succeed, then a primitive-level average that treats each task equally will overstate the effective productivity gain.

A complexity-weighted estimate explicitly accounts for this heterogeneity by weighting success and time savings by the share of tasks in each complexity bin. Even if the numerical bias turns out to be modest, the methodology itself is important: it aligns productivity measurement with the underlying distribution of work rather than with a representative "average" task that may not actually exist.

From a policy and AI safety perspective, this matters because high-complexity tasks are often where failures are most consequential. Overstating AI reliability in these regions can lead to premature automation of tasks that still require significant human oversight.

---

## 5. Limitations and Next Steps

This note represents an early iteration of the within-task complexity analysis. Several limitations are important to highlight:

1. **Success metric construction:**
   - `success_pct` is derived as the share of `::yes` outcomes in the `onet_task::task_success` distribution. Tasks with no `::yes` rows are left missing (2,739 of 3,259 clusters carry a success value), so success-based figures and the productivity revision are computed on the populated subset.

2. **Single time window:**
   - The analysis uses a single AEI release window (Feb 5–12, 2026). Extending the panel to multiple weeks or months would allow for more robust estimates and the study of temporal dynamics (e.g., learning curves, seasonality in task mix).

3. **Keyword-based primitive splits:**
   - Software-development and writing/editing subsets are defined by simple keyword matching on the task description (`SOFTWARE_KW` / `WRITING_KW` in `src/features.py`). This is a coarse proxy for true economic primitives and may over- or under-include tasks; a mapping to O*NET occupation groups would be more principled.

4. **No causal claims:**
   - The analysis is descriptive and does not attempt to make causal claims about the impact of AI on productivity. It is intended as a measurement and decomposition exercise that can feed into more structured causal work.

### Planned extensions

The repository is structured to make the following extensions straightforward:

- **Additional facets:**
  - The panel can be extended with further `onet_task::*` facets already present in the data, e.g. `onet_task::ai_education_years`, `onet_task::collaboration`, and `onet_task::multitasking`.

- **Principled primitive mapping:**
  - Replace keyword-based subsets with a mapping from task clusters to O*NET occupation groups, enabling per-primitive complexity gradients with cleaner membership.

- **Richer models:**
  - Use `complexity_score` in regressions of success and time savings on complexity, allowing interactions with user tenure or geography, building on Anthropic's existing learning-curve and geography analyses.
