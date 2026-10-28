---
orphan: true
---

# Public Entrypoints

The main Python toolkit surface is exposed through `cellpainting_claw`.

This page is intentionally narrow. It is a lightweight internal reference for the small set of Python entrypoints that a user actually needs to choose between.

## Start With The Right Level

A practical rule is:

1. use a **skill** first
2. use a **direct helper family** when you want lower-level control
3. use the **broad combined entry** only when you intentionally want that larger run shape

## `ProjectConfig`

`ProjectConfig` is the configuration object used across the toolkit.

Use it when you need:

- one resolved config object for Python calls
- access to nested data-access and CellProfiler blocks
- a stable starting point for Python-side execution

Minimal example:

```python
import cellpainting_claw as cp

config = cp.ProjectConfig.from_json("configs/project_config.demo.json")
```

## Start Here: `run_pipeline_skill`

`run_pipeline_skill` is the **default Python entrypoint** for most users.

Use it when:

- you want one stable named task
- you want Python code to match the public skill catalog
- you want a cleaner interface than assembling lower-level options yourself

Example:

```python
import cellpainting_claw as cp

config = cp.ProjectConfig.from_json("configs/project_config.demo.json")
result = cp.run_pipeline_skill(config, "cp-extract-segmentation-artifacts")
print(result.ok)
```

## Lower-Level Helper Families

The main package also exposes helper families for users who want more direct control.

### `.cppipe` helpers

Use these when you want to inspect or validate the selected CellProfiler pipeline from Python.

Key helpers include:

- `available_cppipe_templates`
- `get_cppipe_template`
- `resolve_cppipe_selection`
- `validate_cppipe_configuration`

### Data-access helpers

Use these when the main question is about **inputs**, not yet about profiling or segmentation outputs.

Key helpers include:

- `build_data_request`
- `build_download_plan`
- `execute_download_plan`
- `summarize_data_access`

### Tool-family runners

Use these when you intentionally want direct access to one toolkit family rather than the skill layer.

Examples include:

- `run_profiling_suite`
- `run_segmentation_suite`
- `run_deepprofiler_pipeline`

## `run_end_to_end_pipeline`

`run_end_to_end_pipeline` is the broad combined Python entry.

Use it only when:

- you intentionally want one larger combined run
- you are building a higher-level compatibility layer
- you know you do not want to start from modular skills or smaller helper families

## Choosing Correctly

A practical summary is:

- use `run_pipeline_skill` first
- use helper families for direct control
- use `run_end_to_end_pipeline` only for intentional broad combined runs

## Related Pages

- [Skills](../skills/index.md)
- [CLI](../cli/index.md)
- [OpenClaw](../openclaw/index.md)
