# `cyto-aggregate-profiles`

`cyto-aggregate-profiles` turns a single-cell table into the first well-level classical profile table.

It is the step that turns many measured cells from the same well into one summarized well-level profile.

## Purpose

Use this skill when you want:

- one profile row per well instead of one row per cell
- the first pycytominer-style table for classical Cell Painting analysis
- a more stable biological summary before annotation, normalization, and feature selection

## Main Outcome

After this skill finishes, the data are no longer organized cell by cell.

Instead, you have a well-level profile table that summarizes the measured cells in each well.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- a single-cell table from [cp-build-single-cell-table](cp_build_single_cell_table.md), or an equivalent table provided by the user
- an optional output directory

## Outputs

- `aggregated.parquet`
  The aggregated classical profile table. Each row is one well-level profile.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw turn against the repository demo config:

- record id: `cytoaggregate-local-v3`
- recorded on `2026-05-06 13:47 UTC`
- model: `vibe/gpt-5-mini`

### Request

```text
I want to turn this single-cell Cell Painting table into a well-level classical profile and understand what one row in that profile represents biologically.
```

## Structured Trace

```text
user_input:
I want to turn this single-cell Cell Painting table into a well-level classical profile and understand what one row in that profile represents biologically.

planner_step_1:
intent:
Run the well-level aggregation step on the demo single-cell table.
tool_calls:
- exec("cd /root/pipeline/CellPainting-Claw && /root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills run --config /root/pipeline/CellPainting-Claw/configs/project_config.demo.json --skill cyto-aggregate-profiles --single-cell-path /root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_single_cell/single_cell.csv.gz --output-dir /root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_classical_profiles")
- process.poll("brisk-gulf")
observed_result:
- skill_key = cyto-aggregate-profiles
- aggregated_path = demo/workspace/outputs/agent_demo_classical_profiles/pycytominer/aggregated.parquet
- row_count = 2
- column_count = 14

planner_step_2:
intent:
Read the aggregated table so the biological meaning of one row can be explained concretely.
tool_calls:
- exec("python ... read aggregated.parquet, print shape, columns, and preview rows")
- process.poll("plaid-valley")
observed_result:
- shape = 2 rows x 14 columns
- the two rows were `BR00000001, A01` and `BR00000001, A02`
- each row summarized one well
- the feature columns contained aggregated morphology and intensity measurements

final_answer_snapshot:
The result is a well-level classical profile table with one summarized row per well.
```

## Observed Result

A real recorded run for this skill wrote:

- `pycytominer/aggregated.parquet`
- `pipeline_skill_manifest.json`

The recorded output summary showed:

- `2` well-level rows
- `14` columns
- wells `A01` and `A02`

The table contained:

- plate and well identifiers
- cell counts per well
- aggregated cell morphology measurements
- aggregated channel-intensity measurements

## Recorded Reply

```text
I turned the single-cell Cell Painting table into a well-level classical profile using the stable step:

- `cyto-aggregate-profiles`

Output
- `/root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_classical_profiles/pycytominer/aggregated.parquet`
- 2 rows
- 14 columns

What one row represents
- one row = one well
- the two demo rows are `BR00000001, A01` and `BR00000001, A02`
- each row is a well-level summary of all single cells measured in that well

What that means biologically
- this is no longer a single-cell table
- each row is the aggregate phenotype of a well
- the row combines the cell-level measurements in that well into one representative profile vector

Example contents
- `Metadata_Object_Count` for how many cells contributed
- aggregated morphology features
- aggregated intensity features such as DNA, AGP, and RNA signal

In this demo:
- 2 single cells contributed to each well profile

Bottom line
- single-cell table: one row = one cell
- classical profile table: one row = one well
```

## Next Skills

- [cyto-annotate-profiles](cyto_annotate_profiles.md)
- [cyto-normalize-profiles](cyto_normalize_profiles.md)
