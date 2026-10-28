from __future__ import annotations

"""Registry mapping every executable skill key to its runner."""

from typing import Callable

from cellpaint_pipeline.skills.context import SkillRuntimeContext
from cellpaint_pipeline.skills.definitions import PipelineSkillResult
from cellpaint_pipeline.skills.runners import (
    _run_build_deepprofiler_project,
    _run_cellprofiler_profiling,
    _run_collect_deepprofiler_features,
    _run_cyto_aggregate_profiles,
    _run_cyto_annotate_profiles,
    _run_cyto_normalize_profiles,
    _run_cyto_select_profile_features,
    _run_data_plan_download,
    _run_deepprofiler,
    _run_deepprofiler_profile,
    _run_download_cellpainting_data,
    _run_export_deepprofiler_inputs,
    _run_export_masked_single_cell_crops,
    _run_export_single_cell_crops,
    _run_export_single_cell_measurements,
    _run_export_unmasked_single_cell_crops,
    _run_extract_segmentation_artifacts,
    _run_generate_sample_previews,
    _run_inspect_cellpainting_data,
    _run_prepare_deepprofiler_project,
    _run_prepare_segmentation_inputs,
    _run_pycytominer,
    _run_segmentation_masks,
    _run_summarize_classical_profiles,
    _run_summarize_deepprofiler_profiles,
)

SkillRunner = Callable[[SkillRuntimeContext], PipelineSkillResult]

SKILL_RUNNERS: dict[str, SkillRunner] = {
    'data-inspect-availability': _run_inspect_cellpainting_data,
    'data-plan-download': _run_data_plan_download,
    'data-download': _run_download_cellpainting_data,
    'cp-extract-measurements': _run_cellprofiler_profiling,
    'cp-build-single-cell-table': _run_export_single_cell_measurements,
    'cyto-aggregate-profiles': _run_cyto_aggregate_profiles,
    'cyto-annotate-profiles': _run_cyto_annotate_profiles,
    'cyto-normalize-profiles': _run_cyto_normalize_profiles,
    'cyto-select-profile-features': _run_cyto_select_profile_features,
    'cyto-summarize-classical-profiles': _run_summarize_classical_profiles,
    'cp-prepare-segmentation-inputs': _run_prepare_segmentation_inputs,
    'cp-extract-segmentation-artifacts': _run_extract_segmentation_artifacts,
    'cp-generate-segmentation-previews': _run_generate_sample_previews,
    'crop-export-single-cell-crops': _run_export_single_cell_crops,
    'dp-export-deep-feature-inputs': _run_export_deepprofiler_inputs,
    'dp-build-deep-feature-project': _run_build_deepprofiler_project,
    'dp-run-deep-feature-model': _run_deepprofiler_profile,
    'dp-collect-deep-features': _run_collect_deepprofiler_features,
    'dp-summarize-deep-features': _run_summarize_deepprofiler_profiles,
    'inspect-cellpainting-data': _run_inspect_cellpainting_data,
    'download-cellpainting-data': _run_download_cellpainting_data,
    'run-cellprofiler-profiling': _run_cellprofiler_profiling,
    'export-single-cell-measurements': _run_export_single_cell_measurements,
    'run-pycytominer': _run_pycytominer,
    'summarize-classical-profiles': _run_summarize_classical_profiles,
    'run-segmentation-masks': _run_segmentation_masks,
    'generate-sample-previews': _run_generate_sample_previews,
    'export-single-cell-crops': _run_export_single_cell_crops,
    'prepare-deepprofiler-project': _run_prepare_deepprofiler_project,
    'run-deepprofiler': _run_deepprofiler,
    'summarize-deepprofiler-profiles': _run_summarize_deepprofiler_profiles,
    'export-masked-single-cell-crops': _run_export_masked_single_cell_crops,
    'export-unmasked-single-cell-crops': _run_export_unmasked_single_cell_crops,
    'export-deepprofiler-inputs': _run_export_deepprofiler_inputs,
    'build-deepprofiler-project': _run_build_deepprofiler_project,
    'run-deepprofiler-profile': _run_deepprofiler_profile,
    'collect-deepprofiler-features': _run_collect_deepprofiler_features,
}
