# NanoBot Quickstart

This directory contains a ready-to-adapt NanoBot handoff for `CellPainting-Claw`.

## What is already validated

- The library exports MCP-friendly tools through `cellpainting_claw` and `cellpaint_pipeline.mcp_tools`
- The library exposes a real MCP server through `cellpainting_claw.run_mcp_server` and `cellpaint_pipeline.mcp_server`
- `stdio` startup was validated
- `streamable-http` startup was validated on `127.0.0.1:8768`
- A direct HTTP probe to `/mcp` returned `406 Not Acceptable`, which confirms the route is live and expects an MCP-compatible client
- NanoBot `0.0.60` was installed locally and validated against the directory-based config under `local_config/`
- NanoBot HTTP startup was validated on `http://127.0.0.1:8090`

## Current boundary

Natural-language task execution still requires a model API key in the environment, for example:

```bash
export OPENAI_API_KEY=...
```

Without that key, NanoBot can start and load the MCP config, but it cannot complete actual model-driven conversations.

## Start the MCP server

Use the active CellPainting-Claw interpreter directly:

```bash
cd $REPO_ROOT
$PYTHON_BIN -m cellpainting_claw serve-mcp --transport streamable-http --host 127.0.0.1 --port 8768 --path /mcp
```

## NanoBot config

Use either `nanobot.yaml.example` as a minimal template or run the ready-made directory config in `local_config/`.

For `nanobot 0.0.60`, a simple YAML file works for basic agent metadata, but rich agent instructions belong in markdown agent files under a config directory.

- `http://127.0.0.1:8768/mcp`

## Recommended first tool calls from NanoBot

1. `list_mcp_tools`
2. `show_public_api_contract`
3. `list_pipeline_skills`
4. `run_pipeline_skill`

## Good first tasks

- Plan gallery data access
- Run the profiling workflow
- Run the full workflow
- Run the DeepProfiler full branch

## Good example prompts for NanoBot

- `List the available Cell Painting MCP tools and summarize what each one is for.`
- `Use the Cell Painting MCP server to list pipeline skills.`
- `Run the plan-data-access skill with the example project config.`
- `Run the DeepProfiler branch from the standardized pipeline interface.`

## Ready-made local files

- `nanobot.yaml.example`: minimal single-file NanoBot config compatible with `nanobot 0.0.60`
- `nanobot.local.http.yaml`: concrete single-file MCP target config for this workspace
- `local_config/`: the recommended directory-based NanoBot config with markdown agent instructions
- `start_cellpaint_mcp_http.sh`: starts the local MCP HTTP server
- `run_nanobot_local.sh`: launches NanoBot against the local config

Typical sequence:

```bash
cd $REPO_ROOT/integrations/nanobot
./start_cellpaint_mcp_http.sh
```

In another shell:

```bash
export OPENAI_API_KEY=...
cd $REPO_ROOT/integrations/nanobot
./run_nanobot_local.sh
```
