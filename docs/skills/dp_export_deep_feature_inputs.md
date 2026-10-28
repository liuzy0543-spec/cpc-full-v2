# `dp-export-deep-feature-inputs`

`dp-export-deep-feature-inputs` is the step that prepares a completed segmentation result for DeepProfiler.

It tells DeepProfiler which multichannel Cell Painting image belongs to each field and where the segmented cells are located inside that field.

## Purpose

Use this skill when you want:

- to carry a segmentation result into the deep-feature branch
- one DeepProfiler-ready image record for each field
- one coordinate table for the segmented cells in each field
- a clean handoff from segmentation into deep-feature extraction

## Main Outcome

This skill does not calculate learned features yet.

Its job is to tell DeepProfiler two things:

- which multichannel image belongs to each field
- where each segmented cell is located in that field

In other words, this is the step that turns a segmentation result into model-readable image fields plus cell positions.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- the completed segmentation image table
- the completed nuclei table
- the segmentation load-data table that records the Cell Painting channel paths
- an optional output directory

In the demo setup, these inputs come directly from the completed segmentation result.

## Outputs

The main user-facing results are:

- one DeepProfiler-ready image record for each field
- one coordinate table for the segmented cells in each field
- a compact summary of how many fields and cells were prepared

## Agent Demo

This page is based on a real local OpenClaw run:

- turn timestamp: `2026-05-06 18:41 GMT+8`
- model: `vibe/gpt-5-mini`

### Request

```text
I already have a completed segmentation result and now want to move into the DeepProfiler branch. Please prepare the demo image fields and single-cell locations that DeepProfiler will need, then explain what was prepared biologically.
```

## Structured Trace

```text
user_input:
I already have a completed segmentation result and now want to move into the DeepProfiler branch. Please prepare the demo image fields and single-cell locations that DeepProfiler will need, then explain what was prepared biologically.

planner_step_1:
intent:
Load the documented Cell Painting routing guidance before choosing a DeepProfiler skill.
tool_calls:
- read("integrations/openclaw/autodl/workspace/skills/cellpaint-pipeline/SKILL.md")
observed_result:
- the workspace skill file loaded successfully
- routing guidance pointed to `dp-export-deep-feature-inputs` for requests that stop at the DeepProfiler export bundle

planner_step_2:
intent:
Confirm the documented skill contract before running it.
tool_calls:
- exec("cd $REPO_ROOT && cellpainting-skills describe --skill dp-export-deep-feature-inputs")
observed_result:
- the skill description confirmed that this step writes `manifest.json`, `images/field_metadata.csv`, and `locations/`

planner_step_3:
intent:
Run the documented export skill against the demo config and requested output root.
tool_calls:
- exec("cd $REPO_ROOT && /root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills run --config $REPO_ROOT/configs/project_config.demo.json --skill dp-export-deep-feature-inputs --output-dir $REPO_ROOT/demo/workspace/outputs/agent_demo_dp_inputs")
observed_result:
- skill_key = dp-export-deep-feature-inputs
- field_count = 2
- location_file_count = 2
- total_nuclei = 4
- output_root = /root/autodl-tmp/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_dp_inputs

planner_step_4:
intent:
Inspect the exported field manifest and cell-location files so the biological meaning can be explained back to the user.
tool_calls:
- exec("read field_metadata.csv preview")
- exec("read A01 and A02 location CSV previews")
- exec("read manifest.json")
observed_result:
- the export contained two fields:
  - BR00000001 / A01 / site 1
  - BR00000001 / A02 / site 1
- each field linked the five Cell Painting channels: DNA, RNA, ER, AGP, and Mito
- A01/site_1.csv listed 2 nuclei
- A02/site_1.csv listed 2 nuclei

final_answer_snapshot:
The DeepProfiler export completed successfully and prepared 2 image fields plus per-field cell locations for 4 total segmented cells.
```

## Observed Result

The recorded OpenClaw turn exported a complete DeepProfiler input bundle for:

- `2` image fields
- `2` location files
- `4` total segmented cells

The two exported fields were:

- plate `BR00000001`, well `A01`, site `1`
- plate `BR00000001`, well `A02`, site `1`

## Recorded Reply

```text
I exported the DeepProfiler input bundle successfully.

Summary of what was produced:
- fields exported: `2`
- nuclei location files: `2`
- total nuclei listed: `4`

What this means biologically:
- each image field is now paired with the five Cell Painting channels that describe the cell state in that field
- each segmented cell now has a recorded location that DeepProfiler can use to center a single-cell view
- the segmentation result has therefore been translated into a form that the deep-feature model can read directly

Example cell locations from this export:
- `A01/site_1.csv`:
  - cell 1 at `(22, 20)`
  - cell 2 at `(40, 42)`
- `A02/site_1.csv`:
  - cell 1 at `(24, 22)`
  - cell 2 at `(38, 44)`
```

## Next Skills

- [dp-build-deep-feature-project](dp_build_deep_feature_project.md)
