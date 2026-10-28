# Introduction

CellPainting-Claw was built to solve two practical problems in Cell Painting work: the tooling is often fragmented across many packages and scripts, and it is still awkward to expose that work through agents. This project brings those pieces together behind one documented skill catalog that both people and agents can use.

The main idea is simple: **humans should be able to call the documented skills directly, and agents should be able to call the same skills through natural language**.

## Supported Capabilities

CellPainting-Claw covers several connected parts of practical Cell Painting work.

| Capability | Packages or tools | Primary role |
| --- | --- | --- |
| Data access | `boto3`, `quilt3`, `cpgdata` | finding datasets and downloading local inputs |
| Measurement extraction | `CellProfiler` | profiling tables, segmentation masks, outlines, and object measurements |
| Classical profile generation | `pycytominer` | normalization, feature selection, and profile generation from single-cell tables |
| Deep feature extraction | `DeepProfiler` | learned features from segmentation-guided single-cell crops |

These capabilities are presented in the same order that many users encounter them in practice: **get data, extract measurements, build classical profiles, and optionally build deep features**.

## Interfaces

For most users, CellPainting-Claw should be understood through **two main usage paths**.

### Direct Skills

The main human-facing starting point is `cellpainting_skills`.

Use it when you want to:

- call the documented skills directly from Python or from the CLI
- run clear tasks such as segmentation, classical profiling, or DeepProfiler-related tasks
- work from stable task names instead of assembling lower-level helpers yourself

In short, this is the right starting point when you want to **use the documented skills directly**.

### OpenClaw

OpenClaw is the **natural-language entry point**.

Use it when you want to:

- describe a task in natural language
- let an agent route that request to the right documented skill
- use the same skill catalog through a chat-style interface

In short, this is the right starting point when you want to **use the same skills through an agent**.

### Advanced Toolkit Access

The lower-level `cellpainting_claw` package remains available for advanced direct toolkit access, including direct configuration inspection, lower-level toolkit commands, and MCP serving. It is still part of the project, but it is not the main starting point for most users.

## Public Skills

The public skill catalog is organized around concrete outputs rather than one fixed end-to-end route.

Each skill is meant to do **one practical piece of work** and leave behind a result that later skills, scripts, or agent requests can reuse.

| Skill family | Public skills | Main outputs |
| --- | --- | --- |
| Data access | `data-inspect-availability`, `data-plan-download`, `data-download` | availability summaries, download plans, and local dataset downloads |
| Profiling | `cp-extract-measurements`, `cp-build-single-cell-table`, `cyto-aggregate-profiles`, `cyto-annotate-profiles`, `cyto-normalize-profiles`, `cyto-select-profile-features`, `cyto-summarize-classical-profiles` | CellProfiler tables, single-cell tables, classical profiles, and readable profile summaries |
| Segmentation | `cp-prepare-segmentation-inputs`, `cp-extract-segmentation-artifacts`, `cp-generate-segmentation-previews`, `crop-export-single-cell-crops` | segmentation input tables, masks, previews, and crop exports |
| Deep features | `dp-export-deep-feature-inputs`, `dp-build-deep-feature-project`, `dp-run-deep-feature-model`, `dp-collect-deep-features`, `dp-summarize-deep-features` | DeepProfiler-ready inputs, runnable projects, collected feature tables, and readable deep-feature summaries |

## Boundaries

CellPainting-Claw is a **toolkit and interface project**.

It does **not** replace:

- CellProfiler
- pycytominer
- DeepProfiler

Its role is to make those tools easier to configure, run, and expose through a stable public interface.

## Next Steps

After this page, the most useful next pages are:

- [Skills](../skills/index.md)
- [Quick Start](../quick_start/index.ipynb)
- [CLI](../cli/index.md)
- [OpenClaw](../openclaw/index.md)
