# `cyto-summarize-classical-profiles`

`cyto-summarize-classical-profiles` turns classical profile tables into a readable result bundle.

It is the step that makes the final classical profiles easier to inspect by highlighting the strongest separating features and a simple PCA view.

## Purpose

Use this skill when you want:

- a compact human-readable summary instead of only parquet tables
- a quick view of which features separate wells
- a simple low-dimensional view of the final profile differences

## Main Outcome

After this skill finishes, the classical profile table is accompanied by summary files that tell you which wells differ and which features drive that difference.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- a classical profile table to summarize
- an optional output directory

In the recorded demo run below, the summarized input table was the aggregated classical profile table from [cyto-aggregate-profiles](cyto_aggregate_profiles.md).

## Outputs

- `profile_summary.json`
  A compact machine-readable summary of the classical profiling result.
- `well_metadata_summary.csv`
  A per-well summary of the retained metadata.
- `top_variable_features.csv`
  The features that varied the most across wells.
- `pca_coordinates.csv`
  PCA coordinates for each well.
- `pca_plot.png`
  A quick PCA figure.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw turn against the repository demo config:

- record id: `cytosummary-local-v3`
- recorded on `2026-05-06 13:55 UTC`
- model: `vibe/gpt-5-mini`

### Request

```text
I want a quick biological summary of the final classical profiles in this demo and which features separate the wells most clearly.
```

## Structured Trace

```text
user_input:
I want a quick biological summary of the final classical profiles in this demo and which features separate the wells most clearly.

planner_step_1:
intent:
Run the classical-profile summary step on the demo aggregated profile table.
tool_calls:
- exec("cd /root/pipeline/CellPainting-Claw && /root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills run --config /root/pipeline/CellPainting-Claw/configs/project_config.demo.json --skill cyto-summarize-classical-profiles --feature-selected-path /root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_classical_profiles/pycytominer/aggregated.parquet --manifest-path /root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_classical_profiles/pipeline_skill_manifest.json --output-dir /root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_classical_profile_summary")
- process.poll("salty-wharf")
observed_result:
- skill_key = cyto-summarize-classical-profiles
- row_count = 2
- feature_count = 11
- top_feature_count = 11
- PC1 explained essentially all variance

planner_step_2:
intent:
Inspect the summary files to report the standout features and the PCA interpretation.
tool_calls:
- read("profile_summary.json")
- read("top_variable_features.csv")
- read("well_metadata_summary.csv")
- read("pca_coordinates.csv")
observed_result:
- the summary covered 2 wells and 11 measured features
- the strongest feature differences were DNA, AGP, and RNA intensity
- PCA separated `A01` and `A02` almost entirely along PC1

final_answer_snapshot:
The final classical summary showed that the two demo wells separate along one dominant profile axis driven mainly by intensity features, with smaller Y-position differences.
```

## Observed Result

A real recorded run for this skill wrote:

- `profile_summary.json`
- `well_metadata_summary.csv`
- `top_variable_features.csv`
- `pca_coordinates.csv`
- `pca_plot.png`
- `pipeline_skill_manifest.json`

The recorded output summary showed:

- `2` wells
- `11` measured features
- top variable features:
  - `Cells_Intensity_MeanIntensity_OrigDNA`
  - `Cells_Intensity_MeanIntensity_OrigAGP`
  - `Cells_Intensity_MeanIntensity_OrigRNA`
  - `Cells_AreaShape_BoundingBoxMaximum_Y`
  - `Cells_AreaShape_Center_Y`
  - `Cells_AreaShape_BoundingBoxMinimum_Y`
- `PC1` explained essentially all variance

## Recorded Reply

```text
Here’s a quick biological summary of the final classical well-level profiles in this demo.

What the final profiles represent
- 2 wells: `BR00000001 / A01` and `BR00000001 / A02`
- each row is a well-level phenotype built from 2 measured cells
- there are 11 measured features plus metadata

Quick biological interpretation
- the two wells separate mainly by cell intensity features, not by strong size/shape changes
- the clearest differences are DNA, AGP, and RNA intensity
- small Y-position differences are also present, but those are more likely field-layout effects in this toy demo

Which features separate the wells most clearly
- `Cells_Intensity_MeanIntensity_OrigDNA` — variance 5000.0
- `Cells_Intensity_MeanIntensity_OrigAGP` — variance 3200.0
- `Cells_Intensity_MeanIntensity_OrigRNA` — variance 2450.0
- `Cells_AreaShape_BoundingBoxMaximum_Y` — variance 2.0
- `Cells_AreaShape_Center_Y` — variance 2.0
- `Cells_AreaShape_BoundingBoxMinimum_Y` — variance 2.0

PCA summary
- `A01`: PC1 ≈ -72.99
- `A02`: PC1 ≈ 72.99
- PC1 explains 100% of the variance in this tiny demo
- PC2 is effectively negligible

Bottom line
- the two wells have similar gross size/shape
- they differ mainly in average staining intensity across DNA, AGP, and RNA channels
- the dominant separating phenotype here is signal-state difference, not a dramatic morphology shift
```

## Next Skills

- [cyto-select-profile-features](cyto_select_profile_features.md)
