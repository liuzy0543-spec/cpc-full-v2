# `data-plan-download`

`data-plan-download` turns a download request into a concrete plan before any files are fetched.

It tells you which dataset, source, and gallery prefix a request resolves to.

## Purpose

Use this skill when you want:

- confirm exactly what a download request would target
- check the target prefix before downloading any files
- save a reusable plan for the next download step

## Main Outcome

This skill does not download data.

Instead, it writes a plan that says which source the request points to and how many download steps would be executed.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- a requested dataset and source
- an optional output directory

In the recorded demo below, the request uses dataset `cpg0016-jump` and source `source_4`.

## Outputs

- `download_plan.json`
  The saved plan. It records the resolved dataset, source, prefix, and planned download steps.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw turn against the repository demo config:

- session id: `dataplan-rerun-v3`
- recorded on `2026-05-06`
- model: `vibe/gpt-5-mini`

### Request

```text
Before I download the demo Cell Painting data, I want to check exactly what source this request would target. Please use the demo project config, plan the download for dataset cpg0016-jump and source_4, save the result under demo/workspace/outputs/agent_demo_data_plan_rerun_v3, and tell me which gallery prefix this plan points to and how many download steps it contains.
```

## Structured Trace

```text
user_input:
Before I download the demo Cell Painting data, I want to check exactly what source this request would target. Please use the demo project config, plan the download for dataset cpg0016-jump and source_4, save the result under demo/workspace/outputs/agent_demo_data_plan_rerun_v3, and tell me which gallery prefix this plan points to and how many download steps it contains.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply reported that the plan was written under the requested output directory
- the reply identified the resolved gallery prefix
- the reply reported the number of download steps in the saved plan

final_answer_snapshot:
The request resolves to gallery prefix cpg0016-jump/source_4/ and produces one download step.
```

## Observed Result

The recorded run resolved the request to:

- dataset `cpg0016-jump`
- source `source_4`
- gallery prefix `cpg0016-jump/source_4/`

The generated plan contained:

- `1` download step

This means the request is concrete and ready to hand to the next download step.

## Recorded Reply

```text
Done — I planned the download using the demo project config and saved it under:

- `demo/workspace/outputs/agent_demo_data_plan_rerun_v3`

### What source this plan targets
The plan points to the Cell Painting Gallery prefix:

- **`cpg0016-jump/source_4/`**

### How many download steps it contains
- **1 download step**

### Reporting summary
- **Entrypoint run:** `data-plan-download`
- **Config used:** `/root/pipeline/CellPainting-Claw/configs/project_config.demo.json`
- **Output directory written:** `/root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_data_plan_rerun_v3`

The concrete saved plan file is:

- `/root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_data_plan_rerun_v3/download_plan.json`
```

## Next Skills

- [data-download](data_download.md)
