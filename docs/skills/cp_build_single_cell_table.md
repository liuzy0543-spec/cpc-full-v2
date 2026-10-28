# `cp-build-single-cell-table`

`cp-build-single-cell-table` turns CellProfiler measurement tables into one single-cell table.

It is the step that takes separate CellProfiler outputs and rewrites them into one table where each row represents one cell.

## Purpose

Use this skill when you want:

- one row per cell instead of separate CellProfiler tables
- a single table that downstream classical profiling steps can use directly
- the bridge between CellProfiler outputs and pycytominer-style profiling

## Main Outcome

After this skill finishes, the measurement outputs are no longer split across separate tables.

Instead, you have one merged single-cell table that downstream classical profiling steps can read directly.

## Inputs

This skill reads:

- a project config such as `configs/project_config.demo.json`
- the standard CellProfiler measurement tables
- an optional output directory

In the recorded demo run below, the merge used the `Cells` object table.

## Outputs

- `single_cell.csv.gz`
  One compressed table where each row is one cell and each column is a measurement or identifier.
- `pipeline_skill_manifest.json`
  The run record for this skill invocation.

## Agent Demo

This page is based on a real local OpenClaw turn against the repository demo config:

- record id: `cpsinglecell-local-v5`
- recorded on `2026-05-06 13:41 UTC`
- model: `vibe/gpt-5-mini`

### Request

```text
I want to organize these Cell Painting measurements into a single-cell table and understand what one row in that table actually represents.
```

## Structured Trace

```text
user_input:
I want to organize these Cell Painting measurements into a single-cell table and understand what one row in that table actually represents.

recorded_turn:
- the saved OpenClaw turn answered this request directly
- the reply identified `cp-build-single-cell-table` as the step used
- the reply reported `single_cell.csv.gz` with 4 rows and 16 columns
- the reply explained that each row represents one segmented cell

final_answer_snapshot:
The resulting table is one row per segmented cell and is the analysis-friendly handoff table for downstream classical profiling steps.
```

## Observed Result

A real recorded run for this skill wrote:

- `single_cell.csv.gz`
- `pipeline_skill_manifest.json`

The recorded output summary showed:

- `4` single-cell rows
- `16` columns
- object table used for the merge: `Cells`

The table contained:

- metadata columns such as plate, well, and site
- object-link columns that preserve CellProfiler identity
- cell morphology features
- channel-intensity features

## Recorded Reply

```text
I organized the demo Cell Painting measurements into a single-cell table using the stable step:

- `cp-build-single-cell-table`

Output table
- `/root/pipeline/CellPainting-Claw/demo/workspace/outputs/agent_demo_single_cell/single_cell.csv.gz`
- 4 rows
- 16 columns

What one row actually represents
- one cell object from `Cells.csv`
- joined with image-level metadata from `Image.csv`
- identified by plate / well / site / image
- carrying that cell’s morphology and intensity measurements

So the table is:
- not one row per well
- not one row per image
- yes one row per single cell

What kinds of information are in each row
- provenance columns such as `ImageNumber`, `Metadata_Plate`, `Metadata_Well`, `Metadata_Site`
- cell identity columns such as `Metadata_ObjectNumber`, `Cells_Parent_Nuclei`
- morphology columns such as `Cells_AreaShape_Center_X`, `Cells_AreaShape_Center_Y`, `Cells_AreaShape_Area`
- intensity columns such as `Cells_Intensity_MeanIntensity_OrigDNA`, `Cells_Intensity_MeanIntensity_OrigAGP`, `Cells_Intensity_MeanIntensity_OrigRNA`

Biologically, one row is a single-cell phenotype record.

In this demo:
- 2 cells from well `A01`, site `1`
- 2 cells from well `A02`, site `1`

Bottom line
- one row = one cell
- columns = that cell’s metadata + measurements
```

## Next Skills

- [cyto-aggregate-profiles](cyto_aggregate_profiles.md)
