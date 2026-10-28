# `crop-export-single-cell-crops`

`crop-export-single-cell-crops` exports per-cell crop packages from a completed segmentation result.

It is the step that turns a finished segmentation result into one crop per cell.

## Purpose

Use this skill when you want:

- masked or unmasked single-cell crops
- per-cell image stacks that can be reviewed directly
- per-cell image inputs for deep-feature analysis

## Main Outcome

After this skill finishes, the segmentation result is no longer only a set of masks and tables.

You also have one crop package per segmented cell, with image stacks, masks, and a manifest that links each crop back to its original object.

## Inputs

This skill reads:

- a segmentation workflow root from [cp-extract-segmentation-artifacts](cp_extract_segmentation_artifacts.md)
- a crop mode such as `masked` or `unmasked`
- crop-export settings from the project config or explicit user options
- an optional output directory

## Outputs

- `image_stacks/*.tiff`
  Multi-channel per-cell crops.
- `cell_masks/*.tiff`
  Whole-cell masks for each crop.
- `nuclei_masks/*.tiff`
  Nucleus masks for each crop.
- `single_cell_manifest.csv`
  The manifest that ties all crop files back to the original segmented objects.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw inspection of a completed crop-export result:

- session id: `crop-inspect-v2`
- recorded on `2026-05-06`
- model: `vibe/gpt-5-mini`

### Request

```text
I already have a completed demo segmentation result and exported masked single-cell crops from it. Please inspect the crop result under demo/workspace/outputs/agent_demo_crops_check_v2 and tell me what per-cell outputs are available there for review or deep-feature analysis.
```

## Structured Trace

```text
user_input:
I already have a completed demo segmentation result and exported masked single-cell crops from it. Please inspect the crop result under demo/workspace/outputs/agent_demo_crops_check_v2 and tell me what per-cell outputs are available there for review or deep-feature analysis.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply reported the crop-export folder structure
- the reply summarized the number of cropped cells in the manifest
- the reply explained the purpose of image stacks, cell masks, nuclei masks, and the manifest

final_answer_snapshot:
The completed crop export contains 4 masked single-cell examples, each with an image stack, a cell mask, a nucleus mask, and one manifest row.
```

## Observed Result

The inspected completed crop export under `agent_demo_crops_check_v2` contained:

- `masked/image_stacks/`
- `masked/cell_masks/`
- `masked/nuclei_masks/`
- `masked/single_cell_manifest.csv`

The manifest showed:

- `4` cropped cells total
- plate `BR00000001`
- wells `A01` and `A02`
- site `1`

## Recorded Reply

```text
I inspected the completed masked crop export under:

- `demo/workspace/outputs/agent_demo_crops_check_v2`

## What per-cell outputs are available there

This export contains a **masked single-cell crop set** produced from the completed demo segmentation result.

### Main output folders
Under:

- `demo/workspace/outputs/agent_demo_crops_check_v2/masked/`

you have:

- `image_stacks/`
- `cell_masks/`
- `nuclei_masks/`
- `single_cell_manifest.csv`
```

## Next Skills

- [dp-export-deep-feature-inputs](dp_export_deep_feature_inputs.md)
