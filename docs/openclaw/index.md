# OpenClaw

OpenClaw is the **natural-language entry point** for CellPainting-Claw.

Use it when you want to describe a task in plain language and let an agent route that request onto the documented skills.

## Start Here

For the first concrete agent-facing example, start with:

- [Segmentation Demo](segmentation_demo.md)

That page shows one archived agent-style trace for three segmentation skills, together with the recorded result figures.

## Role Of OpenClaw

In normal use, OpenClaw should do one thing well:

1. receive a natural-language request
2. map that request onto the right skill
3. call the toolkit through MCP
4. return the result through the agent session

So OpenClaw should be understood as a **natural-language front end for the existing skills**, not as a separate workflow.

## OpenClaw Fit

Use OpenClaw when you want:

- natural-language task execution
- a natural-language interface for the project
- a chat-style way to run documented skills

Use the CLI or Python directly when you want:

- the most explicit and reproducible interface
- shell scripts, notebooks, or direct library calls
- easier debugging of configuration or backend problems

## Recommended Setup Order

OpenClaw should **not be the first thing you debug**.

A reliable order is:

1. confirm that the core `cellpainting-claw` CLI works
2. confirm that the MCP server starts cleanly
3. only then add the OpenClaw interface on top

That order separates toolkit problems from OpenClaw-side problems.

## Minimal OpenClaw Flow

This page shows how to use the same public skills through natural-language requests.

Unlike [Quick Start](../quick_start/index.ipynb), this page is not a recorded provider-free demo. OpenClaw requires a configured model provider, so the steps below describe the supported usage path once that provider setup is in place.

### Provider Setup

Choose the setup directory that matches your host:

- `integrations/openclaw/autodl/` for AutoDL-like hosts without nested Docker
- `integrations/openclaw/docker/` for standard Linux hosts with Docker support

In that directory:

```bash
cp provider.env.example provider.env
# edit provider.env
./configure_openai_compatible_provider.sh
```

This step writes the provider configuration that OpenClaw will use for model access.

### Gateway And TUI

Start the gateway first:

```bash
./run_openclaw_gateway.sh
```

Then open the terminal client in a second shell:

```bash
./run_openclaw_tui.sh
```

The gateway keeps the OpenClaw side and the local CellPainting-Claw MCP server connected. The TUI is where you type the request.

### Example Request 1: Segmentation

Prompt:

```text
Run segmentation on configs/project_config.demo.json and write the results to demo/workspace/outputs/openclaw_demo_segmentation.
```

OpenClaw should route this request to:

- `cp-extract-segmentation-artifacts`

Expected outputs:

- `load_data_for_segmentation.csv`
- `CPJUMP1_analysis_mask_export.cppipe`
- `cellprofiler_masks/Image.csv`
- `cellprofiler_masks/Cells.csv`
- `cellprofiler_masks/Nuclei.csv`
- `cellprofiler_masks/labels/`
- `cellprofiler_masks/outlines/`
- `segmentation_summary.json`

### Example Request 2: Preview Generation

Prompt:

```text
Generate segmentation preview PNGs from demo/workspace/outputs/openclaw_demo_segmentation and write them to demo/workspace/outputs/openclaw_demo_previews.
```

OpenClaw should route this request to:

- `cp-generate-segmentation-previews`

Expected outputs:

- `sample_previews_png/`
- `pipeline_skill_manifest.json`

### What This Demonstrates

The OpenClaw path is not a separate workflow. It is the same public skill catalog used through a natural-language interface.

In practice:

- the user states the task in plain language
- OpenClaw maps that request onto a documented skill
- the skill runs through the same toolkit underneath
- the output files are the same kind of files you would get from direct CLI use

## Crop Export Example

Prompt:

```text
Export masked single-cell crops from outputs/demo_segmentation and write them under outputs/demo_crops.
```

Request interpretation:

- the task objective is crop export from an existing segmentation workflow root
- the request should route to `crop-export-single-cell-crops`
- the expected output is a directory of masked single-cell crops plus a crop manifest

## Result Summary Example

Prompt:

```text
Summarize the DeepProfiler outputs from outputs/demo_deepprofiler and write a readable report bundle under outputs/demo_deepprofiler_summary.
```

Request interpretation:

- the task objective is result summarization rather than model execution
- the request should route to `dp-summarize-deep-features`
- the expected output is a summary bundle with metadata tables, top variable features, PCA coordinates, and a PCA plot

## Good OpenClaw Requests

A useful request usually states:

- the task objective
- the config file
- the output location when relevant

The important point is that the request should describe the **task you want done**, not the internal implementation details.

## Setup Paths

For current OpenClaw releases, prefer the TUI path rather than the ACP client path.

## AutoDL Path

```bash
cd integrations/openclaw/autodl
cp provider.env.example provider.env
# edit provider.env
./configure_openai_compatible_provider.sh
./run_openclaw_gateway.sh
./run_openclaw_tui.sh
```

## Docker Path

```bash
cd integrations/openclaw/docker
cp .env.example .env
# edit .env
./compose_up.sh
```

Then start the terminal client in a second shell:

```bash
cd integrations/openclaw/docker
./compose_acp_client.sh
```

## Boundaries

OpenClaw is useful, but it does **not** replace the core toolkit.

Important boundaries are:

- toolkit correctness still depends on the underlying config, data paths, and installed backends
- MCP is the connection path, not the toolkit itself
- provider configuration and gateway state are deployment concerns outside the core pipeline library

## Troubleshooting Order

When OpenClaw fails, a useful debugging order is:

1. run the core CLI directly
2. start the MCP server and confirm it stays healthy
3. start the OpenClaw interface
4. only then debug provider configuration or prompt-level behavior

Common failure classes are:

- provider authentication or base URL configuration
- gateway not running
- MCP server not reachable
- valid agent connection but invalid toolkit config or missing backend dependencies

## Related Pages

- [Skills](../skills/index.md)
- [CLI](../cli/index.md)
- [Quick Start](../quick_start/index.ipynb)

```{toctree}
:maxdepth: 1
:hidden:

segmentation_demo
```
