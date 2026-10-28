# `data-download`

`data-download` fetches a selected Cell Painting dataset slice into a local folder.

It is the step that turns a download request into local files that later profiling or segmentation steps can read.

## Purpose

Use this skill when you want:

- local Cell Painting input files on disk
- a small subset that can be used immediately in the next step
- a clear record of which files were fetched

## Main Outcome

This skill writes the requested Cell Painting files into a local download folder.

In the recorded demo run below, it downloaded the two metadata tables that the later demo steps use:

- `load_data.csv`
- `load_data_with_illum.csv`

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- a data request, or a saved plan from [data-plan-download](data_plan_download.md)
- an optional output directory

In the demo run below, the request targeted dataset `cpg0016-jump`, source `source_4`, and a small metadata subset for `BR00117035`.

## Outputs

- `downloads/load_data.csv`
  The field-level metadata table used by downstream demo steps.
- `downloads/load_data_with_illum.csv`
  The same metadata table with illumination-related path columns.
- `download_plan.json`
  The saved request that was executed for this download.
- `downloads/download_manifest.json`
  The file-level record of what was downloaded.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw run against the repository demo config:

- session id: `datadownload-local-v1`
- turn timestamp: `2026-05-05 13:40 UTC`
- model: `vibe/gpt-5-mini`

### Request

```text
I want a tiny local demo subset of the Cell Painting inputs so I can test the next step. Please download the BR00117035 metadata subset from the demo dataset and then tell me what became available locally.
```

## Structured Trace

```text
user_input:
I want a tiny local demo subset of the Cell Painting inputs so I can test the next step. Please download the BR00117035 metadata subset from the demo dataset and then tell me what became available locally.

planner_step_1:
intent:
Load the documented Cell Painting skill routing before choosing a data-access execution skill.
tool_calls:
- read("integrations/openclaw/autodl/workspace/skills/cellpaint-pipeline/SKILL.md")
observed_result:
- the workspace skill file loaded successfully
- the routing guidance identified `data-download` as the public skill for fetching local input files

planner_step_2:
intent:
Confirm the current skill catalog and inspect the documented description for the selected skill.
tool_calls:
- exec("cd /root/pipeline/CellPainting-Claw && /root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills list && echo '---DESCRIBE---' && /root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills describe --skill data-download")
observed_result:
- the public skill catalog listed `data-download`
- the skill description confirmed that this step downloads one configured dataset slice into a local cache directory

planner_step_3:
intent:
Locate the documented demo subset and run the actual metadata download into the named demo output directory.
tool_calls:
- exec("cd /root/pipeline/CellPainting-Claw && /root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills run --config /root/pipeline/CellPainting-Claw/configs/project_config.demo.json --skill data-download --dataset-id cpg0016-jump --source-id source_4 --subprefix workspace/load_data_csv/2021_04_26_Batch1/BR00117035 --output-dir /root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_data_download")
- process.poll("nova-crest")
observed_result:
- skill_key = data-download
- resolved prefix = cpg0016-jump/source_4/workspace/load_data_csv/2021_04_26_Batch1/BR00117035/
- matched files = 2
- downloaded files = 2
- the download completed successfully

planner_step_4:
intent:
Inspect the written output tree and report exactly which files were fetched.
tool_calls:
- exec("cd /root/pipeline/CellPainting-Claw && find demo/workspace/outputs/agent_demo_data_download -maxdepth 3 -type f | sort")
observed_result:
- download_plan.json
- downloads/download_manifest.json
- downloads/load_data.csv
- downloads/load_data_with_illum.csv
- pipeline_skill_manifest.json

final_answer_snapshot:
The demo metadata subset downloaded successfully and made two local metadata tables available for the next step.
```

## Observed Result

The recorded run downloaded one small Cell Painting metadata subset from:

- dataset `cpg0016-jump`
- source `source_4`
- gallery prefix `cpg0016-jump/source_4/workspace/load_data_csv/2021_04_26_Batch1/BR00117035/`

The saved run record showed:

- matched files: `2`
- downloaded files: `2`

The fetched files were:

- `load_data.csv`
- `load_data_with_illum.csv`

## Recorded Reply

```text
I downloaded the requested demo metadata subset successfully.

Source used:
- dataset: `cpg0016-jump`
- source: `source_4`
- gallery prefix: `cpg0016-jump/source_4/workspace/load_data_csv/2021_04_26_Batch1/BR00117035/`

What is now available locally:
- `load_data.csv`
- `load_data_with_illum.csv`

Summary:
- matched files: `2`
- downloaded files: `2`

This was a metadata-only download. In practical terms, you now have the field-level image metadata needed to start the next profiling or segmentation step.
```

## Next Skills

- [cp-extract-measurements](cp_extract_measurements.md)
- [cp-extract-segmentation-artifacts](cp_extract_segmentation_artifacts.md)
