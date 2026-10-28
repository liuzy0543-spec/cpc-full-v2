# `cp-extract-measurements`

`cp-extract-measurements` is the CellProfiler measurement step for classical Cell Painting profiling.

It produces the standard measurement tables that later classical profiling steps use.

## Purpose

Use this skill when you want:

- the standard CellProfiler tables before profile aggregation
- per-image metadata together with per-cell and per-nucleus measurements
- the measurement stage that feeds the single-cell table step

## Main Outcome

After this skill finishes, the run has standard CellProfiler measurement tables at the image, cell, and nucleus levels.

In the public demo setup, the packaged repository reuses bundled demo CellProfiler outputs instead of launching a fresh profiling backend run.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- the profiling backend selected by that config
- the profiling `.cppipe` selected by the config
- an optional output directory

## Outputs

- `Image.csv`
  The image-level metadata table. Each row represents one field or site.
- `Cells.csv`
  The whole-cell measurement table. Each row represents one segmented cell.
- `Nuclei.csv`
  The nucleus measurement table. Each row represents one segmented nucleus.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

In this bundled demo output, `Cytoplasm.csv` is not present.

## Agent Demo

This page is based on a real local OpenClaw turn against the repository demo config:

- record id: `cpmeasure-local-v5`
- recorded on `2026-05-06 13:25 UTC`
- model: `vibe/gpt-5-mini`

### Request

```text
I want to inspect the raw CellProfiler measurement tables in this demo and understand what biological level each table represents.
```

## Structured Trace

```text
user_input:
I want to inspect the raw CellProfiler measurement tables in this demo and understand what biological level each table represents.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply referenced the bundled demo CellProfiler outputs under `demo/backend/profiling_backend/outputs/cellprofiler`
- the reply reported `Image.csv`, `Cells.csv`, and `Nuclei.csv`
- the reply explicitly noted that `Cytoplasm.csv` was not present in this demo bundle

final_answer_snapshot:
The raw CellProfiler tables in this demo are Image.csv, Cells.csv, and Nuclei.csv, and each one corresponds to a different biological level in the profiling workflow.
```

## Observed Result

The recorded run exposed these standard CellProfiler tables:

- `Image.csv`
  `2` image rows and `4` columns
- `Cells.csv`
  `4` cell rows and `13` columns
- `Nuclei.csv`
  `4` nucleus rows and `5` columns

In this public demo bundle:

- `Cytoplasm.csv` was not present
- the skill reused bundled CellProfiler outputs instead of launching a fresh backend script

## Recorded Reply

```text
I inspected the raw CellProfiler measurement outputs available in this demo.

Available raw measurement tables
- `/root/pipeline/CellPainting-Claw/demo/backend/profiling_backend/outputs/cellprofiler/Image.csv`
- `/root/pipeline/CellPainting-Claw/demo/backend/profiling_backend/outputs/cellprofiler/Cells.csv`
- `/root/pipeline/CellPainting-Claw/demo/backend/profiling_backend/outputs/cellprofiler/Nuclei.csv`

A `Cytoplasm.csv` table was not available in this bundled demo output.

What biological level each table represents

`Image.csv` — image / field level
- 2 rows
- one row per imaged field / site
- carries plate, well, and site context

`Cells.csv` — whole-cell single-cell level
- 4 rows
- one row per segmented cell
- contains cell coordinates, area, and channel intensities

`Nuclei.csv` — nucleus single-cell compartment level
- 4 rows
- one row per segmented nucleus
- contains nucleus coordinates and nuclear area

How to think about them together
- `Image.csv` → field / site level
- `Cells.csv` → whole-cell single-cell level
- `Nuclei.csv` → nucleus single-cell level

These are still the raw pre-aggregation measurement tables before anything is merged into a single-cell table or summarized into well-level profiles.
```

## Next Skills

- [cp-build-single-cell-table](cp_build_single_cell_table.md)
