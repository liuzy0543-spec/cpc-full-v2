# `cp-generate-segmentation-previews`

`cp-generate-segmentation-previews` writes lightweight field-level preview PNGs.

It is the fast visual check step before you move on to heavier downstream work.

## Purpose

Use this skill when you want:

- quick PNG previews for the selected segmentation fields
- a fast visual check before moving on to crops or feature extraction
- a lightweight inspection step after [cp-prepare-segmentation-inputs](cp_prepare_segmentation_inputs.md)

## Main Outcome

After this skill finishes, you have quick preview images that let you confirm the selected fields look usable.

This is a QC step. It does not create segmentation masks or single-cell crops.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- prepared segmentation inputs or a valid segmentation workflow root
- the source channel paths needed to render the preview RGB images
- an optional output directory

## Outputs

- `sample_previews_png/*.png`
  Field-level preview images for quick QC.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw inspection of a completed preview result:

- session id: `segpreview-inspect-v1`
- recorded on `2026-05-06`
- model: `vibe/gpt-5-mini`

### Request

```text
I already have the demo preview images for this segmentation workflow. Please inspect the result under demo/workspace/outputs/agent_demo_segmentation/03_generate_previews_v7 and explain what these preview images let me check before moving on.
```

## Structured Trace

```text
user_input:
I already have the demo preview images for this segmentation workflow. Please inspect the result under demo/workspace/outputs/agent_demo_segmentation/03_generate_previews_v7 and explain what these preview images let me check before moving on.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply reported the two preview PNG files present in the directory
- the reply explained what these preview images can and cannot validate
- the reply framed them as a quick pre-segmentation visual sanity check

final_answer_snapshot:
The preview result contains two field-level images that let you check whether the selected fields look visually usable before moving on.
```

## Demo Image

![A01 sample preview](../_static/agent_demo_segmentation/preview_A01.png)

Caption: Field-level preview image for demo field `BR00000001 / A01 / site 1`, produced by `cp-generate-segmentation-previews`.

## Observed Result

The inspected preview result contained:

- `sample_previews_png/BR00000001_A01_s1_sample.png`
- `sample_previews_png/BR00000001_A02_s1_sample.png`

So the completed preview set contained:

- `2` field-level preview images
- one image for `A01`
- one image for `A02`

## Recorded Reply

```text
I inspected the existing preview result under:

- `demo/workspace/outputs/agent_demo_segmentation/03_generate_previews_v7`

### What preview outputs are there
This preview set contains:

- `sample_previews_png/BR00000001_A01_s1_sample.png`
- `sample_previews_png/BR00000001_A02_s1_sample.png`
```

## Next Skills

- [cp-extract-segmentation-artifacts](cp_extract_segmentation_artifacts.md)
- [crop-export-single-cell-crops](crop_export_single_cell_crops.md)
