# `cyto-normalize-profiles`

`cyto-normalize-profiles` normalizes an annotated classical profile table.

It is the step that puts profile features onto a comparable scale before feature selection and downstream comparison.

## Purpose

Use this skill when you want:

- wells compared on the same feature scale
- raw profile magnitudes converted into relative deviations
- the standard normalized input for feature selection

## Main Outcome

After this skill finishes, the wells and metadata stay the same.

What changes is the feature matrix: varying profile features are centered and scaled so downstream comparisons are easier to interpret.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- an annotated profile table from [cyto-annotate-profiles](cyto_annotate_profiles.md), or an equivalent table provided by the user
- an optional output directory

## Outputs

- `normalized.parquet`
  The normalized classical profile table.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw turn against the repository demo config:

- session id: `cytonormalize-rerun-v3`
- recorded on `2026-05-07`
- model: `vibe/gpt-5-mini`

### Request

```text
I already have annotated Cell Painting profiles for this demo and want to make the wells comparable before downstream analysis. Please use the demo project config, normalize the profiles, write the result under demo/workspace/outputs/agent_demo_cyto_normalize_rerun_v3, and explain what changed in the profile values.
```

## Structured Trace

```text
user_input:
I already have annotated Cell Painting profiles for this demo and want to make the wells comparable before downstream analysis. Please use the demo project config, normalize the profiles, write the result under demo/workspace/outputs/agent_demo_cyto_normalize_rerun_v3, and explain what changed in the profile values.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply identified the normalization step used
- the reply reported the output path of the normalized table
- the reply compared representative raw values with normalized values
- the reply explained which features became 0 because they did not vary

final_answer_snapshot:
Normalization preserved the metadata but converted varying profile features into comparable scaled values.
```

## Observed Result

A real recorded run for this skill wrote:

- `pycytominer/normalized.parquet`
- `pipeline_skill_manifest.json`

The recorded output showed:

- `2` well-level rows
- `17` columns
- metadata preserved
- varying feature columns transformed from raw measurement values into centered/scaled values

Representative changes included:

- `Cells_Intensity_MeanIntensity_OrigDNA`
  from `1122.5` and `1222.5` to `-0.67449` and `0.67449`
- `Cells_Intensity_MeanIntensity_OrigAGP`
  from `1010.0` and `1090.0` to `-0.67449` and `0.67449`
- `Cells_Intensity_MeanIntensity_OrigRNA`
  from `947.0` and `1017.0` to `-0.67449` and `0.67449`

## Recorded Reply

```text
Done — I used the stable normalization step **`cyto-normalize-profiles`** with the demo project config and wrote the result under:

- `demo/workspace/outputs/agent_demo_cyto_normalize_rerun_v3`

### Output
Normalized profile table:
- `/root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_cyto_normalize_rerun_v3/pycytominer/normalized.parquet`
```

## Next Skills

- [cyto-select-profile-features](cyto_select_profile_features.md)
