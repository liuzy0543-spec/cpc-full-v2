# `dp-summarize-deep-features`

`dp-summarize-deep-features` is the step that turns DeepProfiler feature tables into a compact result bundle for human review.

It gives a first-pass view of whether the learned features separate the samples at all.

## Purpose

Use this skill when you want:

- a compact summary of the deep-feature result instead of only parquet tables
- a quick check that the learned features separate the demo wells at all
- a simple figure for first-pass inspection

## Main Outcome

After this skill finishes, the deep-feature branch has a readable summary instead of only high-dimensional tables.

This is a first-pass review step, not a final biological conclusion step.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- the single-cell and well-level deep-feature tables from [dp-collect-deep-features](dp_collect_deep_features.md)
- an optional output directory

In the demo setup, the summary is based on four cells, two wells, and 672 learned feature columns.

## Outputs

- `profile_summary.json`
  A compact machine-readable summary of the deep-feature result.
- `well_metadata_summary.csv`
  A per-well summary of the retained metadata.
- `top_variable_features.csv`
  The learned features that varied the most across wells.
- `pca_coordinates.csv`
  PCA coordinates for each well.
- `pca_plot.png`
  A quick PCA figure.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on the recorded DeepProfiler summary outputs already produced for the repository demo:

- demo record: `demo_record_2026_04_25_gpu_final`
- source summary bundle: `08_dp_summary`

### Request

```text
I already have the deep-feature tables and want a quick summary of what the demo wells look like in learned-feature space.
```

## Structured Trace

```text
user_input:
I already have the deep-feature tables and want a quick summary of what the demo wells look like in learned-feature space.

planner_step_1:
intent:
Select the public DeepProfiler summary step that turns the collected feature tables into a readable result bundle.
tool_calls:
- read("integrations/openclaw/autodl/workspace/skills/cellpaint-pipeline/SKILL.md")
- exec("cd /root/pipeline/CellPainting-Claw && /root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills describe --skill dp-summarize-deep-features")
observed_result:
- the routing guidance identified `dp-summarize-deep-features` as the public step for summarizing the collected deep-feature tables
- the skill description confirmed that this step writes `profile_summary.json`, `top_variable_features.csv`, `pca_coordinates.csv`, and `pca_plot.png`

planner_step_2:
intent:
Inspect the completed summary bundle from the recorded demo run.
tool_calls:
- read("demo/workspace/outputs/demo_record_2026_04_25_gpu_final/08_dp_summary/profile_summary.json")
- exec("find /root/pipeline/CellPainting-Claw/demo/workspace/outputs/demo_record_2026_04_25_gpu_final/08_dp_summary -maxdepth 2 -type f | sort")
observed_result:
- cell_count = 4
- well_count = 2
- feature_count = 672
- top_feature_count = 50
- pca_explained_variance_ratio = [1.0, 1.6090012673630894e-33]

final_answer_snapshot:
The demo deep-feature summary shows a simple two-well separation along one dominant PCA axis, which is useful as a sanity check but too small for a biological claim.
```

## Representative Figure

This figure comes from the recorded deep-feature summary bundle for the repository demo.

![PCA view of the recorded DeepProfiler summary bundle](../_static/deepprofiler/pca_plot.png)

Caption: PCA plot from `dp-summarize-deep-features` for the demo DeepProfiler run. With only two wells and four cells, this figure is useful as a sanity check, not as a phenotype conclusion.

## Observed Result

A real recorded run for this skill wrote:

- `profile_summary.json`
- `well_metadata_summary.csv`
- `top_variable_features.csv`
- `pca_coordinates.csv`
- `pca_plot.png`
- `pipeline_skill_manifest.json`

The recorded output summary showed:

- `4` cells
- `2` wells
- `672` learned feature columns
- `50` reported top variable features
- `PC1 = 1.0`
- `PC2 = 1.6090012673630894e-33`

## Recorded Reply

```text
This summary bundle is the first compact view of the learned DeepProfiler result.

Main result:
- the demo wells separate along one dominant learned-feature axis
- the summary reports the most variable learned features and a simple PCA view

What this means biologically:
- the model captured a measurable difference between the two demo wells in morphology-embedding space
- the individual learned features are not directly human-interpretable in the same way as classical measurements such as area or intensity
- with only two wells and four cells, this result should be treated as a demo-scale first pass rather than a phenotype claim
```

## Next Skills

- [dp-collect-deep-features](dp_collect_deep_features.md)
