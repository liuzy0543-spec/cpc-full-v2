# CellPainting-Skills

`cellpainting_skills` is the public task package of CellPainting-Claw.

Its purpose is to give both people and agents the same stable names for concrete Cell Painting tasks. Each skill is designed to produce a usable output that can stand on its own or feed the next task.

## Public Model

A public skill in this project is:

- one named task
- one main result
- one documented interface that can be used from the CLI, Python, or OpenClaw

The main public catalog below focuses only on the current recommended skill names. Advanced and legacy aliases still exist in the codebase for compatibility, but they are not the primary documentation path.

## Main Catalog

The tables below are written from the user point of view:

- **main input** is the artifact or setting the skill starts from
- **main result** is the main output the skill is expected to leave behind
- **main tools** shows which package or component does most of the work

### Data Access

| Skill key | Main input | Main result | Main tools |
| --- | --- | --- | --- |
| [`data-inspect-availability`](data_inspect_availability.md) | a project config with data-access settings | an availability summary for the configured data sources | `boto3`, `quilt3`, `cpgdata` |
| [`data-plan-download`](data_plan_download.md) | a project config plus a requested dataset or prefix | a saved download plan without downloading files yet | `boto3`, `quilt3`, `cpgdata` |
| [`data-download`](data_download.md) | a saved download plan or direct data request | a local download cache plus execution records | `boto3`, `quilt3`, `cpgdata` |

### Profiling

| Skill key | Main input | Main result | Main tools |
| --- | --- | --- | --- |
| [`cp-extract-measurements`](cp_extract_measurements.md) | raw Cell Painting images plus profiling configuration | CellProfiler measurement tables such as `Image.csv`, `Cells.csv`, `Cytoplasm.csv`, and `Nuclei.csv` | `CellProfiler` |
| [`cp-build-single-cell-table`](cp_build_single_cell_table.md) | CellProfiler measurement tables | one merged single-cell table for downstream analysis | `cellpaint_pipeline.profiling_native` |
| [`cyto-aggregate-profiles`](cyto_aggregate_profiles.md) | a single-cell table | an aggregated classical profile table | `pycytominer` |
| [`cyto-annotate-profiles`](cyto_annotate_profiles.md) | an aggregated profile table | a metadata-annotated profile table | `pycytominer` |
| [`cyto-normalize-profiles`](cyto_normalize_profiles.md) | an annotated profile table | a normalized profile table | `pycytominer` |
| [`cyto-select-profile-features`](cyto_select_profile_features.md) | a normalized profile table | a feature-selected profile table | `pycytominer` |
| [`cyto-summarize-classical-profiles`](cyto_summarize_classical_profiles.md) | one or more classical profile tables | a readable summary bundle with metadata and PCA outputs | `pandas`, `numpy` |

### Segmentation

| Skill key | Main input | Main result | Main tools |
| --- | --- | --- | --- |
| [`cp-prepare-segmentation-inputs`](cp_prepare_segmentation_inputs.md) | image metadata plus segmentation settings | the load-data table used by segmentation | `cellpaint_pipeline.segmentation_native` |
| [`cp-extract-segmentation-artifacts`](cp_extract_segmentation_artifacts.md) | a segmentation load-data table plus the selected segmentation `.cppipe` | masks, labels, outlines, and segmentation measurement tables | `CellProfiler` |
| [`cp-generate-segmentation-previews`](cp_generate_segmentation_previews.md) | segmentation inputs or a segmentation workflow root | preview PNGs for quick review | `Pillow`, `numpy` |
| [`crop-export-single-cell-crops`](crop_export_single_cell_crops.md) | a completed segmentation workflow root | masked or unmasked single-cell crop stacks plus a crop manifest | `cellpaint_pipeline.segmentation_native` |

### Deep Features

| Skill key | Main input | Main result | Main tools |
| --- | --- | --- | --- |
| [`dp-export-deep-feature-inputs`](dp_export_deep_feature_inputs.md) | segmentation outputs or crop exports | DeepProfiler-ready metadata and location files | DeepProfiler export helpers |
| [`dp-build-deep-feature-project`](dp_build_deep_feature_project.md) | a prepared deep-feature input bundle | a runnable DeepProfiler project directory | DeepProfiler project helpers |
| [`dp-run-deep-feature-model`](dp_run_deep_feature_model.md) | a prepared DeepProfiler project | raw DeepProfiler feature files | `DeepProfiler` |
| [`dp-collect-deep-features`](dp_collect_deep_features.md) | a completed DeepProfiler project or run directory | single-cell and aggregated deep-feature tables | `pandas`, `pyarrow` |
| [`dp-summarize-deep-features`](dp_summarize_deep_features.md) | collected deep-feature tables | a readable summary bundle with metadata and PCA outputs | `pandas`, `numpy` |

## Skill Pages

```{toctree}
:maxdepth: 1

data_inspect_availability
data_plan_download
data_download
cp_extract_measurements
cp_build_single_cell_table
cyto_aggregate_profiles
cyto_annotate_profiles
cyto_normalize_profiles
cyto_select_profile_features
cyto_summarize_classical_profiles
cp_prepare_segmentation_inputs
cp_extract_segmentation_artifacts
cp_generate_segmentation_previews
crop_export_single_cell_crops
dp_export_deep_feature_inputs
dp_build_deep_feature_project
dp_run_deep_feature_model
dp_collect_deep_features
dp_summarize_deep_features
```

## Related Pages

- [Quick Start](../quick_start/index.ipynb)
- [CLI](../cli/index.md)
- [OpenClaw](../openclaw/index.md)
