# `cp-extract-segmentation-artifacts`

`cp-extract-segmentation-artifacts` produces the main segmentation output set.

It writes the labels, outlines, and object tables that downstream single-cell steps reuse.

## Purpose

Use this skill when you want:

- segmented nuclei
- segmented whole cells
- review images to inspect whether the boundaries look reasonable
- reusable object definitions for later crop and feature-extraction steps

## Main Outcome

This skill produces the standard segmentation artifact bundle:

- nuclei and whole-cell label images
- outline overlays for review
- CellProfiler object tables
- workflow files that document how the segmentation was run

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- the segmentation input table written by [cp-prepare-segmentation-inputs](cp_prepare_segmentation_inputs.md), or default segmentation inputs resolved from the config
- the selected segmentation `.cppipe`
- the raw Cell Painting images and illumination-correction files from the project config
- an optional output directory

## Outputs

The main user-facing results are:

- label images for nuclei and whole cells
- outline overlays for visual QC
- CellProfiler tables such as `Image.csv`, `Cells.csv`, `Nuclei.csv`, and `Cytoplasm.csv`
- workflow files such as the segmentation `.cppipe`, summary, and manifest

## Agent Demo

This page is based on a real local OpenClaw inspection of a completed demo segmentation result:

- session id: `segartifact-inspect-v1`
- recorded on `2026-05-06`
- model: `vibe/gpt-5-mini`

### Request

```text
I already have a completed demo segmentation result. Please inspect the result under demo/workspace/outputs/agent_demo_segmentation/02_extract_artifacts_v6 and tell me what outputs were produced there and what they mean biologically when I review a Cell Painting segmentation result.
```

## Structured Trace

```text
user_input:
I already have a completed demo segmentation result. Please inspect the result under demo/workspace/outputs/agent_demo_segmentation/02_extract_artifacts_v6 and tell me what outputs were produced there and what they mean biologically when I review a Cell Painting segmentation result.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply identified the segmentation artifact step that produced the directory
- the reply listed the main workflow files, CellProfiler tables, label images, and outline overlays
- the reply explained what nuclei labels, cell labels, outlines, and object tables mean biologically

final_answer_snapshot:
The completed result contains the standard segmentation artifact set: labels, outlines, object tables, and workflow files for two demo fields.
```

## Representative Figure

![Representative segmentation output for one real Cell Painting image](../_static/agent_demo_segmentation/real_fov_segmentation_overview.png)

Caption: Representative output from `cp-extract-segmentation-artifacts`, showing the original field, a selected crop, the nucleus boundaries, and the whole-cell boundaries.

## Observed Result

The inspected completed result under `02_extract_artifacts_v6` contained:

- workflow files such as:
  - `load_data_for_segmentation.csv`
  - `segmentation_workflow_config.json`
  - `CPJUMP1_analysis_mask_export.cppipe`
  - `pipeline_skill_manifest.json`
  - `segmentation_summary.json`
- CellProfiler tables:
  - `Image.csv`
  - `Cells.csv`
  - `Nuclei.csv`
  - `Cytoplasm.csv`
  - `Experiment.csv`
- review images:
  - `labels/*cell_labels.tiff`
  - `labels/*nuclei_labels.tiff`
  - `outlines/*cell_outlines.png`
  - `outlines/*nuclei_outlines.png`

The completed run covered:

- `2` image fields
- wells `A01` and `A02`
- `4` cell objects total
- `4` nuclei objects total
- `4` cytoplasm objects total

## Recorded Reply

```text
I inspected the existing completed segmentation result under:

- `demo/workspace/outputs/agent_demo_segmentation/02_extract_artifacts_v6`

This is a completed run from the stable segmentation skill:

- **`cp-extract-segmentation-artifacts`**
```

## Next Skills

- [cp-generate-segmentation-previews](cp_generate_segmentation_previews.md)
- [crop-export-single-cell-crops](crop_export_single_cell_crops.md)
- [dp-export-deep-feature-inputs](dp_export_deep_feature_inputs.md)
