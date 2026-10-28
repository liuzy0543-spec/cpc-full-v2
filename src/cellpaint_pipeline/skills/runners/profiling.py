"""Classical profiling skill runners (CellProfiler tables + pycytominer).

DEPENDENCY LOOKUP
-----------------
The native implementations are resolved through the package facade
(:mod:`cellpaint_pipeline.skills`) at call time rather than bound at import
time.  That keeps a long-standing testing seam working: patching
``cellpaint_pipeline.skills.run_pycytominer_feature_select_native`` still
replaces the function this runner calls, exactly as it did when every runner
lived in a single module.
"""
from __future__ import annotations

import json
from typing import Any

from cellpaint_pipeline.runner import ExecutionResult
from cellpaint_pipeline.skills.context import SkillRuntimeContext
from cellpaint_pipeline.skills.definitions import PipelineSkillResult
from cellpaint_pipeline.skills.finalize import (
    _native,
    _execution_result_to_dict,
    _finalize_skill_result,
    _json_ready,
)
__all__ = [
    '_run_cellprofiler_profiling',
    '_run_cyto_aggregate_profiles',
    '_run_cyto_annotate_profiles',
    '_run_cyto_normalize_profiles',
    '_run_cyto_select_profile_features',
    '_run_export_single_cell_measurements',
    '_run_pycytominer',
    '_run_pycytominer_stage',
    '_run_summarize_classical_profiles',
]


def _run_cellprofiler_profiling(context: SkillRuntimeContext) -> PipelineSkillResult:
    backend_payload = context.config.load_profiling_backend_payload()
    paths_payload = backend_payload['paths']
    image_table_path = context.config.resolve_profiling_backend_path(paths_payload['image_table_csv'])
    cells_table_path = context.config.resolve_profiling_backend_path(paths_payload['cells_table_csv'])
    cytoplasm_table_path = context.config.resolve_profiling_backend_path(paths_payload['cytoplasm_table_csv'])
    nuclei_table_path = context.config.resolve_profiling_backend_path(paths_payload['nuclei_table_csv'])
    script_path = context.config.profiling_backend_root / 'scripts' / '07_run_official_cellprofiler.py'

    try:
        execution = _native('run_profiling_script')(context.config, 'run-official-cellprofiler')
        details = _execution_result_to_dict(execution)
        # Record which implementation produced the tables on *both* paths, so a
        # downstream reader never has to infer it from the presence of an error.
        details['mode'] = 'cellprofiler'
        ok = execution.returncode == 0
        log_path = execution.log_path
    except FileNotFoundError as exc:
        bundled_tables = [image_table_path, cells_table_path, nuclei_table_path]
        if not all(path.exists() for path in bundled_tables):
            raise
        details = {
            'mode': 'bundled-demo-outputs',
            'reason': (
                'Profiling backend script is not packaged in this public demo checkout. '
                'Reusing the bundled CellProfiler measurement tables instead.'
            ),
            'missing_script_path': str(script_path),
            'bundled_outputs': {
                'image_table_path': str(image_table_path),
                'cells_table_path': str(cells_table_path),
                'cytoplasm_table_path': str(cytoplasm_table_path) if cytoplasm_table_path.exists() else None,
                'nuclei_table_path': str(nuclei_table_path),
            },
            'error': str(exc),
        }
        ok = True
        log_path = None
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.workflows.profiling',
        primary_outputs={
            'image_table_path': image_table_path,
            'cells_table_path': cells_table_path,
            'cytoplasm_table_path': cytoplasm_table_path if cytoplasm_table_path.exists() else None,
            'nuclei_table_path': nuclei_table_path,
            'log_path': log_path,
        },
        details=details,
        ok=ok,
    )


def _run_pycytominer_stage(context: SkillRuntimeContext, *, stage: str) -> PipelineSkillResult:
    stage_root = context.run_root / 'pycytominer'
    if stage == 'aggregate':
        result = _native('run_pycytominer_aggregate_native')(
            context.config,
            output_path=stage_root / 'aggregated.parquet',
            single_cell_path=context.single_cell_path,
        )
        primary_outputs = {'aggregated_path': result.output_path}
    elif stage == 'annotate':
        aggregated_path = context.aggregated_path
        if aggregated_path is None:
            aggregated_path = _native('run_pycytominer_aggregate_native')(
                context.config,
                output_path=stage_root / 'aggregated.parquet',
                single_cell_path=context.single_cell_path,
            ).output_path
        result = _native('run_pycytominer_annotate_native')(
            context.config,
            output_path=stage_root / 'annotated.parquet',
            aggregated_path=aggregated_path,
        )
        primary_outputs = {'annotated_path': result.output_path}
    elif stage == 'normalize':
        annotated_path = context.annotated_path
        if annotated_path is None:
            aggregated_path = context.aggregated_path
            if aggregated_path is None:
                aggregated_path = _native('run_pycytominer_aggregate_native')(
                    context.config,
                    output_path=stage_root / 'aggregated.parquet',
                    single_cell_path=context.single_cell_path,
                ).output_path
            annotated_path = _native('run_pycytominer_annotate_native')(
                context.config,
                output_path=stage_root / 'annotated.parquet',
                aggregated_path=aggregated_path,
            ).output_path
        result = _native('run_pycytominer_normalize_native')(
            context.config,
            output_path=stage_root / 'normalized.parquet',
            annotated_path=annotated_path,
        )
        primary_outputs = {'normalized_path': result.output_path}
    elif stage == 'select':
        normalized_path = context.normalized_path
        if normalized_path is None:
            annotated_path = context.annotated_path
            if annotated_path is None:
                aggregated_path = context.aggregated_path
                if aggregated_path is None:
                    aggregated_path = _native('run_pycytominer_aggregate_native')(
                        context.config,
                        output_path=stage_root / 'aggregated.parquet',
                        single_cell_path=context.single_cell_path,
                    ).output_path
                annotated_path = _native('run_pycytominer_annotate_native')(
                    context.config,
                    output_path=stage_root / 'annotated.parquet',
                    aggregated_path=aggregated_path,
                ).output_path
            normalized_path = _native('run_pycytominer_normalize_native')(
                context.config,
                output_path=stage_root / 'normalized.parquet',
                annotated_path=annotated_path,
            ).output_path
        result = _native('run_pycytominer_feature_select_native')(
            context.config,
            output_path=stage_root / 'feature_selected.parquet',
            normalized_path=normalized_path,
        )
        primary_outputs = {'feature_selected_path': result.output_path}
    else:
        raise ValueError(f'Unsupported pycytominer stage: {stage}')
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.profiling_native',
        primary_outputs=primary_outputs,
        details=_json_ready(result),
        ok=True,
    )


def _run_export_single_cell_measurements(context: SkillRuntimeContext) -> PipelineSkillResult:
    export_result = _native('export_cellprofiler_to_singlecell_native')(
        context.config,
        object_table=context.object_table,
        image_table_path=context.image_csv_path,
        object_table_path=context.object_table_path,
        output_path=context.run_root / 'single_cell.csv.gz',
    )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.profiling_native',
        primary_outputs={'single_cell_path': export_result.output_path},
        details=_json_ready(export_result),
        ok=True,
    )


def _run_cyto_aggregate_profiles(context: SkillRuntimeContext) -> PipelineSkillResult:
    return _run_pycytominer_stage(context, stage='aggregate')


def _run_cyto_annotate_profiles(context: SkillRuntimeContext) -> PipelineSkillResult:
    return _run_pycytominer_stage(context, stage='annotate')


def _run_cyto_normalize_profiles(context: SkillRuntimeContext) -> PipelineSkillResult:
    return _run_pycytominer_stage(context, stage='normalize')


def _run_cyto_select_profile_features(context: SkillRuntimeContext) -> PipelineSkillResult:
    return _run_pycytominer_stage(context, stage='select')


def _run_pycytominer(context: SkillRuntimeContext) -> PipelineSkillResult:
    result = _native('run_pycytominer_native')(
        context.config,
        output_dir=context.run_root / 'pycytominer',
        single_cell_path=context.single_cell_path,
    )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.profiling_native',
        primary_outputs={
            'aggregated_path': result.aggregated_path,
            'annotated_path': result.annotated_path,
            'normalized_path': result.normalized_path,
            'feature_selected_path': result.feature_selected_path,
        },
        details=_json_ready(result),
        ok=True,
    )


def _run_summarize_classical_profiles(context: SkillRuntimeContext) -> PipelineSkillResult:
    result = _native('summarize_classical_profiles')(
        context.config,
        output_dir=context.run_root,
        feature_selected_path=context.feature_selected_path,
        manifest_path=context.manifest_path,
    )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.profile_summaries',
        primary_outputs={
            'summary_path': result.summary_path,
            'well_metadata_summary_path': result.well_metadata_summary_path,
            'top_variable_features_path': result.top_variable_features_path,
            'pca_coordinates_path': result.pca_coordinates_path,
            'pca_plot_path': result.pca_plot_path,
        },
        details=_json_ready(result),
        ok=True,
    )
