# `cyto-annotate-profiles`

`cyto-annotate-profiles` adds experimental metadata to an aggregated classical profile table.

It is the step that turns a numeric well-level profile into a table that can be interpreted biologically.

## Purpose

Use this skill when you want:

- treatment labels attached to aggregated profile rows
- control labels attached before normalization or downstream comparison
- batch information preserved for later interpretation

## Main Outcome

After this skill finishes, the aggregated profile rows are no longer just numeric summaries.

They also carry the metadata needed to say what each well represents in the experiment.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- an aggregated profile table from [cyto-aggregate-profiles](cyto_aggregate_profiles.md), or an equivalent table provided by the user
- an optional output directory

## Outputs

- `annotated.parquet`
  The metadata-annotated classical profile table.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw turn against the repository demo config:

- session id: `cytoannotate-rerun-v2`
- recorded on `2026-05-06`
- model: `vibe/gpt-5-mini`

### Request

```text
I already have the aggregated Cell Painting profiles for this demo, but I need them to be biologically interpretable. Please use the demo project config, add the experimental metadata, write the result under demo/workspace/outputs/agent_demo_cyto_annotate_rerun_v2, and tell me what metadata were attached to each well.
```

## Structured Trace

```text
user_input:
I already have the aggregated Cell Painting profiles for this demo, but I need them to be biologically interpretable. Please use the demo project config, add the experimental metadata, write the result under demo/workspace/outputs/agent_demo_cyto_annotate_rerun_v2, and tell me what metadata were attached to each well.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply identified the stable annotation step used
- the reply reported the output path of the annotated table
- the reply listed the metadata columns attached to each well

final_answer_snapshot:
The annotation step attached plate, well, object-count, treatment, control-type, and batch metadata to the two demo wells.
```

## Observed Result

A real recorded run for this skill wrote:

- `pycytominer/annotated.parquet`
- `pipeline_skill_manifest.json`

The recorded output showed:

- `2` well-level rows
- metadata columns for:
  - plate
  - well
  - object count
  - treatment
  - control type
  - batch

For the demo wells:

- `A01` was labeled `DEMO_A01`, `negative_control`, `demo_batch_01`
- `A02` was labeled `DEMO_A02`, `negative_control`, `demo_batch_01`

## Recorded Reply

```text
Done — I used the stable annotation step **`cyto-annotate-profiles`** with the demo project config and wrote the annotated result under:

- `demo/workspace/outputs/agent_demo_cyto_annotate_rerun_v2`

### Output
Annotated profile table:
- `/root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_cyto_annotate_rerun_v2/pycytominer/annotated.parquet`

### What metadata were attached
The annotation added these metadata columns to each well:

- `Metadata_Plate`
- `Metadata_Well`
- `Metadata_Object_Count`
- `Metadata_Treatment`
- `Metadata_ControlType`
- `Metadata_Batch`
```

## Next Skills

- [cyto-normalize-profiles](cyto_normalize_profiles.md)
