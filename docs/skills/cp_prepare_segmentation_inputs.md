# `cp-prepare-segmentation-inputs`

`cp-prepare-segmentation-inputs` resolves which image fields will enter segmentation.

It is the planning step before any CellProfiler segmentation run begins.

## Purpose

Use this skill when you want:

- confirm which fields will be sent into segmentation
- inspect the segmentation input table before a longer run
- hand a clean field list to [cp-extract-segmentation-artifacts](cp_extract_segmentation_artifacts.md)

## Main Outcome

After this skill finishes, you know exactly which image fields will go into segmentation.

This is a planning step. It does not produce masks, outlines, or single-cell crops yet.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- the configured raw-image and illumination roots
- optional plate, well, or site filters
- an optional output directory

## Outputs

- `load_data_for_segmentation.csv`
  The field-level segmentation input table.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw turn against the repository demo config:

- session id: `segprepare-rerun-v2`
- recorded on `2026-05-06`
- model: `vibe/gpt-5-mini`

### Request

```text
Before I segment the demo Cell Painting images, I want to confirm exactly which wells and image fields will be included in this run. Please use the demo project config, prepare the segmentation input table, write it under demo/workspace/outputs/agent_demo_segmentation/01_prepare_inputs_rerun_v2, and summarize the wells and fields that will be processed.
```

## Structured Trace

```text
user_input:
Before I segment the demo Cell Painting images, I want to confirm exactly which wells and image fields will be included in this run. Please use the demo project config, prepare the segmentation input table, write it under demo/workspace/outputs/agent_demo_segmentation/01_prepare_inputs_rerun_v2, and summarize the wells and fields that will be processed.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply identified the segmentation-input step used
- the reply reported the output path of the load-data table
- the reply summarized the number of wells and fields included
- the reply listed the exact well/site combinations

final_answer_snapshot:
The demo segmentation run includes two image fields: BR00000001 / A01 / site 1 and BR00000001 / A02 / site 1.
```

## Observed Result

The recorded run wrote:

- `load_data_for_segmentation.csv`
- `pipeline_skill_manifest.json`

The resolved segmentation scope covered:

- `1` plate
- `2` wells
- `2` image fields

The included fields were:

- `BR00000001 / A01 / site 1`
- `BR00000001 / A02 / site 1`

## Recorded Reply

```text
Done — I used the stable step **`cp-prepare-segmentation-inputs`** with the demo project config and wrote the segmentation input table under:

- `demo/workspace/outputs/agent_demo_segmentation/01_prepare_inputs_rerun_v2`

### Output file
- `demo/workspace/outputs/agent_demo_segmentation/01_prepare_inputs_rerun_v2/load_data_for_segmentation.csv`

### What will be processed
This prepared segmentation run includes:

- **1 plate**
- **2 wells**
- **2 image fields (rows in the load-data table)**
```

## Next Skills

- [cp-extract-segmentation-artifacts](cp_extract_segmentation_artifacts.md)
- [cp-generate-segmentation-previews](cp_generate_segmentation_previews.md)
