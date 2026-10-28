---
name: cellpaint-pipeline
description: Use the validated Cell Painting pipeline library through the current CellPainting-Skills catalog and supporting CLI surfaces.
---

# Cell Painting Pipeline

Use this skill when the user wants to discover, run, or summarize the standardized Cell Painting workflow library.

## Goal

Translate natural-language requests into the current documented skills rather than custom shell chains or older compatibility aliases.

## Preferred CLI

```bash
/root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills
```

If the executable path changes, fall back to:

```bash
cd $REPO_ROOT
PYTHONPATH=$REPO_ROOT/src $PYTHON_BIN -m cellpainting_skills
```

## Preferred order

1. Discovery

```bash
/root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills list
/root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills describe --skill <skill-key>
```

2. Skill execution

```bash
cd $REPO_ROOT
/root/autodl-tmp/miniconda3_envs/lyx_env/bin/cellpainting-skills run \
  --config $REPO_ROOT/configs/project_config.example.json \
  --skill <skill-key> \
  --output-dir <output-dir>
```

Fallback execution form if the executable path changes:

```bash
cd $REPO_ROOT
PYTHONPATH=$REPO_ROOT/src $PYTHON_BIN -m cellpainting_skills run \
  --config $REPO_ROOT/configs/project_config.example.json \
  --skill <skill-key> \
  --output-dir <output-dir>
```

3. Use `cellpainting_claw` only when the user explicitly needs a lower-level or broader interface

```bash
$PYTHON_BIN -m cellpainting_claw <command> ...
```

## Current skill examples

Segmentation:

- `cp-prepare-segmentation-inputs`
- `cp-extract-segmentation-artifacts`
- `cp-generate-segmentation-previews`

Classical profiling:

- `cp-extract-measurements`
- `cp-build-single-cell-table`
- `cyto-aggregate-profiles`
- `cyto-annotate-profiles`
- `cyto-normalize-profiles`
- `cyto-select-profile-features`
- `cyto-summarize-classical-profiles`

Deep features:

- `dp-export-deep-feature-inputs`
- `dp-build-deep-feature-project`
- `dp-run-deep-feature-model`
- `dp-collect-deep-features`
- `dp-summarize-deep-features`

Data access:

- `data-inspect-availability`
- `data-plan-download`
- `data-download`

## Routing guidance

Prefer the fine-grained public skill that directly matches the user's task.

Examples:

- if the user wants the segmentation load-data table, choose `cp-prepare-segmentation-inputs`
- if the user wants masks, labels, outlines, or segmentation tables, choose `cp-extract-segmentation-artifacts`
- if the user wants quick visual checks, choose `cp-generate-segmentation-previews`
- if the user asks which wells, sites, fields, or image rows are included in a segmentation run, choose `cp-prepare-segmentation-inputs` and run it instead of manually probing the repository tree

For segmentation-scope questions, do not try to infer the answer by listing ad hoc directories such as guessed `demo/wells` or `demo/image_fields` paths. The supported way to answer that question is to execute `cp-prepare-segmentation-inputs` and report its row, well, and site counts together with the generated load-data table.

Do not default to older high-level compatibility names such as `run-segmentation`, `run-classical-profiling`, or `run-deepprofiler` unless the user explicitly asks for those legacy workflow aliases.

## Reporting expectations

For each run, report:

- exact skill key used
- exact command used
- config path
- output directory
- the main files or directories written
- the next likely skill when the task naturally leads to another step

## Working directory rule

Run CellPainting skill commands from `$REPO_ROOT`.

This matters because the demo and example configs resolve backend data paths relative to the repository layout. If the current shell is inside the OpenClaw workspace, prepend `cd $REPO_ROOT` before running the skill command.

## Exec rule

Use the local runtime defaults for command execution.

Do not force `host`, `security`, or `ask` fields in `exec` tool calls unless the user explicitly asks for a different execution mode.

In this AutoDL runtime, the stable path is:

- run commands from `$REPO_ROOT`
- call the validated `cellpainting-skills` executable directly
- use normal `exec` plus `process poll` when a run takes longer

Avoid ad hoc inline Python wrappers for basic skill discovery when the documented CLI already provides the needed result.

## Optional MCP route

If the host is configured to use MCP, start the server with:

```bash
$REPO_ROOT/integrations/openclaw/start_cellpaint_mcp_http.sh
```

Then prefer MCP discovery only to decide which current skill to run. The actual task should still be framed around the same fine-grained public skill catalog.

## Config guidance

Treat the project config as the source of truth for:

- backend roots
- workspace and output roots
- selected `.cppipe` templates
- runtime settings

Do not invent paths or template names when the config can provide them.

## Demo request guidance

If the user explicitly asks about the repository demo or demo Cell Painting data:

- prefer `configs/project_config.demo.json`
- prefer writing demo outputs under `demo/workspace/outputs/agent_demo_segmentation/`
- keep the request natural in user-facing reporting, but keep the execution anchored to the demo config and demo output tree
