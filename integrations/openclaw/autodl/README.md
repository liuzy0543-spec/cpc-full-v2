# OpenClaw AutoDL Runtime

This track is intended for AutoDL-like platforms where the machine is already containerized and nested Docker is not available.

## Goals

- preserve natural-language workflow automation through OpenClaw and MCP
- keep OpenClaw state and workspace under a controlled directory inside this repository
- avoid storing provider keys in repository-managed config files
- keep the runtime reproducible enough for image-based distribution on similar platforms

## Directory Layout

- `openclaw.autodl.json`
  Clean repository template without provider credentials
- `provider.env.example`
  Example env file for local provider configuration
- `configure_openai_compatible_provider.sh`
  Builds the runtime config under ignored state
- `run_openclaw_gateway.sh`
  Starts the OpenClaw gateway and the local CellPaint MCP server
- `run_openclaw_tui.sh`
  Starts the terminal UI
- `start_openclaw_stack_bg.sh`
  Starts the gateway in the background
- `check_openclaw_stack.sh`
  Checks gateway and MCP ports
- `stop_openclaw_stack.sh`
  Stops the background gateway
- `project_config.autodl.json`
  Project config pointing at the local pipeline tree

## Security Model

This is a controlled host-local runtime, not a strong container boundary.

Current controls include:

- `OPENCLAW_HOME` under `integrations/openclaw/autodl/state`
- workspace under `integrations/openclaw/autodl/workspace`
- `repoRoot` pinned to the local `CellPainting-Claw` checkout
- browser and canvas tools denied
- elevated tooling disabled

The platform instance boundary still matters more than OpenClaw itself.

## Setup

Install OpenClaw if needed:

```bash
cd $REPO_ROOT/integrations/openclaw/autodl
./install_openclaw_cli.sh
```

Create a local provider env file:

```bash
cd $REPO_ROOT/integrations/openclaw/autodl
cp provider.env.example provider.env
```

Edit `provider.env`, then generate the runtime config:

```bash
./configure_openai_compatible_provider.sh
```

The generated runtime config is written to:

- `integrations/openclaw/autodl/state/openclaw.runtime.json`

## Start the Stack

Foreground gateway:

```bash
./run_openclaw_gateway.sh
```

Background gateway:

```bash
./start_openclaw_stack_bg.sh
./check_openclaw_stack.sh
```

Launch the terminal UI:

```bash
./run_openclaw_tui.sh
```

Default endpoints:

- OpenClaw gateway: `http://127.0.0.1:18789/`
- CellPaint MCP: `http://127.0.0.1:8768/mcp`

## Direct Pipeline Execution

If you want to run the same controlled project config without natural-language routing:

```bash
cd $REPO_ROOT
PYTHONPATH=src $PYTHON_BIN -m cellpainting_claw run-pipeline-skill \
  --config $REPO_ROOT/integrations/openclaw/autodl/project_config.autodl.json \
  --skill run-segmentation
```

## Release Guidance

On AutoDL-like platforms, the practical release unit is usually the saved platform image rather than a Docker image built inside the instance.

Keep these out of version control:

- `provider.env`
- `state/`
- any logs containing provider metadata

## Client Note

For this OpenClaw version, prefer `run_openclaw_tui.sh`. The ACP client path is kept only as a compatibility wrapper.
