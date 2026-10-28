# `dp-collect-deep-features`

`dp-collect-deep-features` is the step that turns raw DeepProfiler feature files into analysis-ready tables.

It is where the learned per-cell embeddings become something a researcher can inspect and compare more directly.

## Purpose

Use this skill when you want:

- one table with one learned feature vector per cell
- one aggregated deep-feature profile per well
- the first analysis-ready outputs from the DeepProfiler branch

## Main Outcome

After this skill finishes, the deep features are no longer only stored field by field.

Instead, they are organized into one cell-level table and one well-level table that can be compared and summarized downstream.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- a completed DeepProfiler project from [dp-run-deep-feature-model](dp_run_deep_feature_model.md)
- the raw DeepProfiler feature files produced by that run
- an optional output directory

In the demo setup, the collected outputs summarize four cells across two wells.

## Outputs

- `deepprofiler_single_cell.parquet`
  The single-cell deep-feature table. Each row is one profiled cell.
- `deepprofiler_well_aggregated.parquet`
  The well-level deep-feature table. Each row is one aggregated well profile.
- `deepprofiler_feature_manifest.json`
  The summary of the collected feature bundle.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on the recorded DeepProfiler feature collection outputs already produced for the repository demo:

- demo record: `demo_record_2026_04_25_gpu_final`
- source feature bundle: `07_dp_features`

### Request

```text
I already ran the DeepProfiler model and now want the learned features collected into analysis-ready single-cell and well-level tables.
```

## Structured Trace

```text
user_input:
I already ran the DeepProfiler model and now want the learned features collected into analysis-ready single-cell and well-level tables.

planner_step_1:
intent:
Select the public DeepProfiler collection step that converts raw feature files into tables.
tool_calls:
- read("integrations/openclaw/autodl/workspace/skills/cellpaint-pipeline/SKILL.md")
- exec("cd /root/pipeline/CellPainting-Claw && /root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills describe --skill dp-collect-deep-features")
observed_result:
- the routing guidance identified `dp-collect-deep-features` as the public step for collecting raw DeepProfiler outputs into tables
- the skill description confirmed that this step writes single-cell and well-level deep-feature tables

planner_step_2:
intent:
Inspect the completed collected-output bundle from the demo record.
tool_calls:
- exec("find /root/pipeline/CellPainting-Claw/demo/workspace/outputs/demo_record_2026_04_25_gpu_final/07_dp_features -maxdepth 3 -type f | sort")
- exec("python ... read deepprofiler_single_cell.parquet and deepprofiler_well_aggregated.parquet, print shapes and metadata counts")
observed_result:
- single_cell_shape = 4 rows x 718 columns
- well_shape = 2 rows x 677 columns
- wells = A01, A02
- plate = BR00000001

final_answer_snapshot:
The collected DeepProfiler outputs provide one learned feature vector per cell and one aggregated deep-feature profile per well.
```

## Observed Result

A real recorded run for this skill wrote:

- `deepprofiler_single_cell.parquet`
- `deepprofiler_well_aggregated.parquet`
- `deepprofiler_feature_manifest.json`
- `pipeline_skill_manifest.json`

The recorded output summary showed:

- `4` single cells
- `2` wells
- `718` columns in the single-cell table
- `677` columns in the well-level table
- `672` learned feature columns in the summarized output manifest

## Recorded Reply

```text
This step converted the raw DeepProfiler feature files into analysis-ready tables.

Main result:
- one deep-feature row per segmented cell
- one aggregated deep-feature profile per well

Produced outputs:
- `deepprofiler_single_cell.parquet`
- `deepprofiler_well_aggregated.parquet`
- `deepprofiler_feature_manifest.json`

What this means biologically:
- each cell now has a learned morphology vector that combines information across the five Cell Painting channels
- the well-level table summarizes those cell states at the sample level
- this is the first point where the deep-feature branch becomes easy to compare across wells or use in downstream analysis
```

## Next Skills

- [dp-summarize-deep-features](dp_summarize_deep_features.md)
