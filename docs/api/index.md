---
orphan: true
---

# API

```{note}
This page is kept as an advanced reference.
The main public documentation path for CellPainting-Claw is centered on
[Introduction](../introduction/index.md),
[Skills](../skills/index.md),
[CLI](../cli/index.md), and
[OpenClaw](../openclaw/index.md).
```

This section documents the **Python toolkit surface** of CellPainting-Claw.

It does **not** duplicate the Skills or CLI pages. Instead, it explains the small set of Python entrypoints that matter when you want to call the toolkit directly from code.

## Section Purpose

Use the API section when you want to answer questions such as:

- which Python entrypoint should I call first?
- when should I use a skill instead of a lower-level helper?
- how do I inspect the selected CellProfiler `.cppipe` from Python?
- when do I need a broader combined entrypoint?

## Keep The Main Distinction Clear

CellPainting-Claw exposes two public Python packages.

### `cellpainting_claw`

Use `cellpainting_claw` when you want:

- the **main Python toolkit surface**
- config loading
- `.cppipe` inspection and validation
- direct access to data-access and lower-level helper families
- direct Python control over toolkit behavior

### `cellpainting_skills`

Use `cellpainting_skills` when you want:

- the **task layer**
- stable skill names
- simple Python calls such as `cp-extract-segmentation-artifacts`
- a cleaner bridge from automation or agent requests into validated toolkit actions

In most cases, users should understand the [Skills](../skills/index.md) page first and then come here only when they need direct Python entrypoints.

## The Small Python Entry Model

For most users, the Python-side entry model is small.

| Entry style | Use it when | Typical example |
| --- | --- | --- |
| skill | you want a stable named task | `run_pipeline_skill(config, "cp-extract-segmentation-artifacts")` |
| direct helper family | you want lower-level control | `.cppipe` helpers, data-access helpers, suite runners |
| broad combined entry | you intentionally want one larger combined run | `run_end_to_end_pipeline(...)` |

## Section Scope

This section stays intentionally short.

It focuses on:

- `ProjectConfig`
- the main `.cppipe` helper family
- the main Python entrypoints
- how to choose between a skill call and a lower-level helper call

## Relationship To Other Sections

If you want:

- the full task catalog, continue to [Skills](../skills/index.md)
- shell-facing command groups, continue to [CLI](../cli/index.md)
- natural-language and agent-mediated use, continue to [OpenClaw](../openclaw/index.md)

```{toctree}
:maxdepth: 1

public_entrypoints
```
