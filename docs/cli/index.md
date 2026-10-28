# Command-Line Interface

CellPainting-Claw exposes **two public CLI surfaces**:

- `cellpainting-skills`
- `cellpainting-claw`

For most users, the right starting point is **`cellpainting-skills`**.

## Choosing A CLI

| User goal | Start with | Rationale |
| --- | --- | --- |
| run one clear named task such as segmentation, pycytominer, or DeepProfiler | `cellpainting-skills` | this is the shortest path from a user goal to a documented task |
| inspect configuration, use data-access helpers, or expose MCP tools | `cellpainting-claw` | this is the direct toolkit CLI |

## `cellpainting-skills`

`cellpainting-skills` is the named-task CLI.

Its three main commands are:

| Command | Role |
| --- | --- |
| `list` | show the public skill catalog |
| `describe` | show what one skill does |
| `run` | execute one named skill |

## First Session

```bash
CONFIG=configs/project_config.demo.json

cellpainting-skills list
cellpainting-skills describe --skill cp-extract-segmentation-artifacts
cellpainting-skills run \
  --config "$CONFIG" \
  --skill cp-extract-segmentation-artifacts \
  --output-dir outputs/demo_segmentation
```

This is the main CLI path that new users should understand first:

- inspect the available skills
- inspect one skill
- run one named task

The public CLI skill catalog currently covers:

- data access inspection, planning, and download
- CellProfiler measurement extraction and single-cell table construction
- pycytominer aggregation, annotation, normalization, feature selection, and summarization
- segmentation preparation, artifact export, preview generation, and crop export
- DeepProfiler input export, project building, model execution, feature collection, and summarization

## `cellpainting-claw`

Use `cellpainting-claw` when you need commands that are more specific than a named skill.

The most useful command groups for normal users are:

| Command group | Scope | Example commands |
| --- | --- | --- |
| configuration inspection | inspect config state and other advanced runtime selections | `show-config`, `show-cppipe-selection`, `validate-cppipe-config` |
| data access | inspect data sources, build plans, and execute downloads | `summarize-data-access`, `plan-data-access`, `execute-download-plan` |
| direct toolkit runs | run toolkit-side commands when you want lower-level control than a public skill | `run-segmentation`, `run-profiling`, `run-deepprofiler-pipeline` |
| MCP bridge | expose or inspect the MCP interface used by agent runtimes | `serve-mcp`, `list-mcp-tools`, `run-mcp-tool` |

## Data Access Commands

```bash
CONFIG=configs/project_config.demo.json

cellpainting-claw summarize-data-access \
  --config "$CONFIG"

cellpainting-claw plan-data-access \
  --config "$CONFIG"
```

## MCP Bridge

```bash
cellpainting-claw serve-mcp --transport stdio
```

This is the bridge interface used by OpenClaw and other MCP-capable clients.

## Lower-Level Commands

The toolkit CLI also contains additional lower-level or compatibility commands. Those remain available, but they are **not the main starting point for most users**, so this page focuses on the command paths that match the public skill model.

This is also where advanced inspection commands remain available when needed.

## Related Pages

- [Skills](../skills/index.md)
- [OpenClaw](../openclaw/index.md)
- [Quick Start](../quick_start/index.ipynb)
