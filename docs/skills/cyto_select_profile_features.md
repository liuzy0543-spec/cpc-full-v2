# `cyto-select-profile-features`

`cyto-select-profile-features` reduces normalized classical profiles to the retained feature set.

It is the step that keeps the final profile features judged useful enough for downstream comparison.

## Purpose

Use this skill when you want:

- a smaller final classical profile table
- fewer redundant or weakly informative features
- the retained feature set for downstream comparison

## Main Outcome

After this skill finishes, the normalized profile table has been reduced to a smaller feature set.

The metadata stay in place, but only the retained profile features move forward.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- a normalized profile table from [cyto-normalize-profiles](cyto_normalize_profiles.md), or an equivalent table provided by the user
- an optional output directory

## Outputs

- `feature_selected.parquet`
  The feature-selected classical profile table.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw turn against the repository demo config:

- session id: `cytoselect-rerun-v3`
- recorded on `2026-05-07`
- model: `vibe/gpt-5-mini`

### Request

```text
I already have normalized Cell Painting profiles for this demo and want the final retained feature set for downstream comparison. Please use the demo project config, select the final profile features, write the result under demo/workspace/outputs/agent_demo_cyto_select_rerun_v3, and tell me what kind of features were kept.
```

## Structured Trace

```text
user_input:
I already have normalized Cell Painting profiles for this demo and want the final retained feature set for downstream comparison. Please use the demo project config, select the final profile features, write the result under demo/workspace/outputs/agent_demo_cyto_select_rerun_v3, and tell me what kind of features were kept.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply identified the feature-selection step used
- the reply reported the output path of the selected table
- the reply listed the retained feature names
- the reply grouped the retained features into geometry-related and channel-intensity features

final_answer_snapshot:
The final retained profile set kept 6 metadata columns and 6 selected profile features, mainly intensity features plus a small number of geometry features.
```

## Observed Result

A real recorded run for this skill wrote:

- `pycytominer/feature_selected.parquet`
- `pipeline_skill_manifest.json`

The recorded output showed:

- `2` rows
- `12` total columns
- `6` metadata columns
- `6` retained profile features

The retained profile features were:

- `Cells_AreaShape_BoundingBoxMinimum_Y`
- `Cells_AreaShape_BoundingBoxMaximum_Y`
- `Cells_AreaShape_Center_Y`
- `Cells_Intensity_MeanIntensity_OrigDNA`
- `Cells_Intensity_MeanIntensity_OrigAGP`
- `Cells_Intensity_MeanIntensity_OrigRNA`

## Recorded Reply

```text
Done — I used the stable feature-selection step **`cyto-select-profile-features`** with the demo project config and wrote the result under:

- `demo/workspace/outputs/agent_demo_cyto_select_rerun_v3`

### Output
Final selected profile table:
- `/root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_cyto_select_rerun_v3/pycytominer/feature_selected.parquet`
```

## Next Skills

- [cyto-summarize-classical-profiles](cyto_summarize_classical_profiles.md)
