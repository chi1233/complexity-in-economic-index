# Within-Task Complexity in the Anthropic Economic Index

## 1. Introduction

Anthropic's Economic Index provides a large-scale measurement of how AI systems like Claude reshape work by tracking success rates, time savings, and autonomy across millions of real user interactions. The core unit of analysis in the Index is the economic primitive: an O*NET task cluster such as "modify and debug software" or "write, edit, and revise documents".

The headline productivity estimates reported in Anthropic's public work average over all tasks within a primitive. For example, a single software development primitive may mix quick one-off bug fixes with multi-hour refactoring work. If success rates and time savings vary systematically with task complexity inside each primitive, then primitive-level averages can misstate the effective productivity uplift and the reliability of AI assistance at the frontier of difficulty.

This note documents a preliminary stratification of within-primitive complexity using the publicly released Anthropic Economic Index dataset. The goal is not to produce a final, production-ready estimate, but to demonstrate a clear and extensible measurement pipeline that can be refined as additional facets (success, autonomy, human-with-AI time) are wired into the task panel.

---

## 2. Data and Methods

### 2.1 Data Source

- **Dataset:** Anthropic Economic Index, `release_2026_03_24`, Claude.ai split (Feb 5–12, 2026).
- **Unit of observation:** O*NET task clusters (e.g., "modify and debug software programs", "draft correspondence and reports").
- **Key variable currently wired:**
  - `human_time_mean` — mean estimated human hours to complete the task without AI, taken from the `onet_task::human_only_time` facet at `geo_id == 'GLOBAL'`.

The current implementation constructs a task-cluster-level panel with one row per O*NET task cluster and the following columns:

- `cluster_name`: O*NET task description.
- `human_time_mean`: mean human-only task duration in hours.
- `complexity_bin`: categorical bin (low/medium/high) based on tertiles of `human_time_mean`.
- `complexity_score`: a normalized z-score of `human_time_mean`.

Additional columns (`success_pct`, `autonomy_mean`, `ai_time_mean`, `edu_years_mean`) are present as placeholders and will be populated by merging the corresponding `onet_task::*` facets in a future iteration.

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
4. Compute basic bin-level summaries using `src.analysis.summarize_by_bin`.
5. Attempt a complexity-weighted productivity calculation via `src.analysis.compute_productivity_revision`, guarded on data availability.
6. Generate figures and save the task panel CSV under `outputs/`.

---

## 3. Results (Template)

The current code base generates a set of figures and summary tables that are designed to align with Anthropic's published Economic Index work. As of this draft, exact numerical values depend on which facets are fully wired; this section describes the intended content of each figure.

### 3.1 Figure 1 — Success Rate by Complexity Bin

Once the `success_pct` facet is merged into the task panel, Figure 1 will plot the mean task success rate in each complexity bin (low, medium, high), with 95% confidence intervals. The expected qualitative pattern, consistent with Anthropic's own reports, is that success rates decline as task complexity increases.

In the current implementation, the plotting code is in place (`fig1_success_by_bin`), and the figure will automatically populate as soon as `success_pct` is available in the panel.

### 3.2 Figure 2 — Success vs. Human Task Duration

Figure 2 is a scatter plot of cluster-level success rates against `human_time_mean`, with points colored by complexity bin and an overlaid linear trend line. This visualization is designed to make the negative relationship between task duration and success rate visually salient and to highlight which parts of the complexity distribution contribute most to any productivity bias.

As with Figure 1, the current code is ready to consume `success_pct` when it is wired in; until then, this figure should be interpreted as a placeholder for the intended analysis.

### 3.3 Figure 3 — Autonomy by Complexity Bin

Anthropic's public work reports an autonomy score capturing how independently Claude is allowed to act on a task. Figure 3 plots mean autonomy score by complexity bin, with confidence intervals and an overall reference line. Conceptually, this figure answers the question: "Do users grant Claude more autonomy on more complex tasks, or do they retain tighter control as complexity rises?"

The code in `fig3_autonomy_by_bin` is structured to display this relationship once `autonomy_mean` is populated. Until then, it serves as a template.

### 3.4 Figure 4 — Complexity-Weighted vs. Naïve Productivity Estimate

The final figure compares a naïve productivity estimate with a complexity-weighted alternative:

- **Naïve uplift:**
  - \( G_{\text{naive}} = \mathbb{E}[\text{success_pct}] \times \mathbb{E}[\text{time_savings_ratio}] \).
- **Complexity-weighted uplift:**
  - \( G_{\text{weighted}} = \sum_k w_k \cdot s_k \cdot \Delta_k \), where \(k\) indexes complexity bins, \(w_k\) is the share of tasks in bin \(k\), \(s_k\) is the bin-specific mean success rate, and \(\Delta_k\) is the bin-specific mean time-savings ratio.

The figure will show side-by-side bars for \(G_{\text{naive}}\) and \(G_{\text{weighted}}\), with annotations indicating the percentage bias \((G_{\text{naive}} - G_{\text{weighted}})/G_{\text{naive}}\). At the current stage, this calculation is guarded in code and will only be reported once both `success_pct` and `time_savings_ratio` are available.

---

## 4. Implications for Productivity Estimates

The core conceptual point of this project is that the distribution of task complexity within each economic primitive matters for how we interpret aggregate productivity statistics. If high-complexity clusters are harder for Claude to complete successfully and yield smaller time savings when they do succeed, then a primitive-level average that treats each task equally will overstate the effective productivity gain.

A complexity-weighted estimate explicitly accounts for this heterogeneity by weighting success and time savings by the share of tasks in each complexity bin. Even if the numerical bias turns out to be modest, the methodology itself is important: it aligns productivity measurement with the underlying distribution of work rather than with a representative "average" task that may not actually exist.

From a policy and AI safety perspective, this matters because high-complexity tasks are often where failures are most consequential. Overstating AI reliability in these regions can lead to premature automation of tasks that still require significant human oversight.

---

## 5. Limitations and Next Steps

This note represents an early, pipeline-focused iteration of the within-task complexity analysis. Several limitations are important to highlight:

1. **Partial facet wiring:**
   - At present, only `human_time_mean` is fully wired into the task panel. Columns for `success_pct`, `autonomy_mean`, `ai_time_mean`, and `edu_years_mean` are placeholders and will be populated by merging the corresponding `onet_task::*` facets in future commits.
   - As a result, the figures and productivity revision calculations should be treated as structural templates rather than final numerical results.

2. **Single time window:**
   - The analysis uses a single AEI release window (Feb 5–12, 2026). Extending the panel to multiple weeks or months would allow for more robust estimates and the study of temporal dynamics (e.g., learning curves, seasonality in task mix).

3. **Primitive-specific analysis deferred:**
   - The current pipeline focuses on the full task panel. Splitting the analysis by primitive (e.g., software development vs. writing and editing) is planned once success and autonomy metrics are wired, so that per-primitive complexity gradients can be estimated with sufficient power.

4. **No causal claims:**
   - The analysis is descriptive and does not attempt to make causal claims about the impact of AI on productivity. It is intended as a measurement and decomposition exercise that can feed into more structured causal work.

### Planned extensions

The repository is structured to make the following extensions straightforward:

- **Facet wiring:**
  - Extend `build_task_panel` in `src/data_loading.py` to merge:
    - `onet_task::task_success` → `success_pct`.
    - `onet_task::ai_autonomy` → `autonomy_mean`.
    - `onet_task::human_with_ai_time` → `ai_time_mean`.
    - `onet_task::human_education_years` → `edu_years_mean`.

- **Primitive splits:**
  - Re-enable primitive-specific subsets for software development and writing/editing once success and time-savings metrics are present, and update the tech note with per-primitive tables and discussion.

- **Richer models:**
  - Use `complexity_score` in regressions of success and time savings on complexity, allowing interactions with user tenure or geography, building on Anthropic's existing learning-curve and geography analyses.

As those extensions are implemented, this note can be updated by replacing placeholder text with actual results and by expanding the Results section to include tables and per-primitive figures.
