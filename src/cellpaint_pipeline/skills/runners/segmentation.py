from __future__ import annotations

"""Segmentation skill runners."""

import json
from pathlib import Path

from cellpaint_pipeline.runner import ExecutionResult
from cellpaint_pipeline.skills.context import SkillRuntimeContext
from cellpaint_pipeline.skills.definitions import PipelineSkillResult
from cellpaint_pipeline.skills.finalize import (
    _native,
    _build_isolated_segmentation_skill_config,
    _execution_result_to_dict,
    _finalize_skill_result,
    _json_ready,
    _resolve_segmentation_source_config,
)
from cellpaint_pipeline.skills.outputs import (
    SEGMENTATION_ARTIFACT_OUTPUTS,
    SEGMENTATION_MASK_OUTPUTS,
    SINGLE_CELL_CROP_OUTPUTS,
    build_primary_outputs,
)


def _run_prepare_segmentation_inputs(context: SkillRuntimeContext) -> PipelineSkillResult:
    result = _native('prepare_segmentation_load_data_native')(
        context.config,
        output_path=context.run_root / 'load_data_for_segmentation.csv',
    )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.segmentation_native',
        primary_outputs={'load_data_path': result.output_path},
        details=_json_ready(result),
        ok=True,
    )


def _run_segmentation_backend(workflow_config, context: SkillRuntimeContext):
    """Run the CellProfiler backend, or the in-process segmenter when it is absent.

    The public demo ships the CellProfiler pipeline definition but not a
    CellProfiler install.  Rather than failing the skill, fall back to the native
    segmenter, which writes the same artifact contract; the chosen mode is
    reported in the skill details so the difference is never silent.
    """
    if _native('cellprofiler_executable')() is None:
        native_result = _native('run_mask_export_native')(
            workflow_config,
            output_dir=context.run_root / 'cellprofiler_masks',
        )
        execution = ExecutionResult(
            label='native-mask-export',
            command=[],
            cwd=context.run_root,
            log_path=native_result.log_path,
            returncode=0,
        )
        return execution, native_result.mode
    try:
        execution = _native('run_segmentation_script')(
            workflow_config,
            'run-mask-export',
            ['--config', str(workflow_config.segmentation_backend_config), '--reuse-load-data', '--reuse-pipeline'],
        )
        return execution, 'cellprofiler'
    except FileNotFoundError:
        native_result = _native('run_mask_export_native')(
            workflow_config,
            output_dir=context.run_root / 'cellprofiler_masks',
        )
        execution = ExecutionResult(
            label='native-mask-export',
            command=[],
            cwd=context.run_root,
            log_path=native_result.log_path,
            returncode=0,
        )
        return execution, native_result.mode


def _run_extract_segmentation_artifacts(context: SkillRuntimeContext) -> PipelineSkillResult:
    load_data_path = context.run_root / 'load_data_for_segmentation.csv'
    pipeline_path = context.run_root / 'CPJUMP1_analysis_mask_export.cppipe'
    load_data_result = _native('prepare_segmentation_load_data_native')(context.config, output_path=load_data_path)
    pipeline_result = _native('build_mask_export_pipeline_native')(context.config, output_path=pipeline_path)
    workflow_config = _build_isolated_segmentation_skill_config(
        context.config,
        workflow_root=context.run_root,
        load_data_path=load_data_path,
        pipeline_path=pipeline_path,
    )
    execution, export_mode = _run_segmentation_backend(workflow_config, context)
    mask_output_dir = context.run_root / 'cellprofiler_masks'
    summary_path = context.run_root / 'segmentation_summary.json'
    summary_payload = {
        'implementation': 'cellpaint_pipeline.skills.cp-extract-segmentation-artifacts',
        'load_data_path': str(load_data_result.output_path),
        'pipeline_path': str(pipeline_result.output_path),
        'cellprofiler_output_dir': str(mask_output_dir),
        'image_table_path': str(mask_output_dir / 'Image.csv'),
        'cells_table_path': str(mask_output_dir / 'Cells.csv'),
        'nuclei_table_path': str(mask_output_dir / 'Nuclei.csv'),
        'labels_dir': str(mask_output_dir / 'labels'),
        'outlines_dir': str(mask_output_dir / 'outlines'),
        'execution': _execution_result_to_dict(execution),
            'export_mode': export_mode,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.workflows.segmentation',
        primary_outputs=build_primary_outputs(
            SEGMENTATION_ARTIFACT_OUTPUTS,
            {
                'load_data_path': load_data_result.output_path,
                'pipeline_path': pipeline_result.output_path,
                'cellprofiler_output_dir': mask_output_dir,
                'image_table_path': mask_output_dir / 'Image.csv',
                'cells_table_path': mask_output_dir / 'Cells.csv',
                'cytoplasm_table_path': mask_output_dir / 'Cytoplasm.csv',
                'nuclei_table_path': mask_output_dir / 'Nuclei.csv',
                'experiment_table_path': mask_output_dir / 'Experiment.csv',
                'labels_dir': mask_output_dir / 'labels',
                'outlines_dir': mask_output_dir / 'outlines',
                'illumination_runtime_dir': mask_output_dir / 'illumination_runtime',
                'load_data_absolute_path': mask_output_dir / 'load_data_for_segmentation.absolute.csv',
                'workflow_config_path': context.run_root / 'segmentation_workflow_config.json',
                'summary_path': summary_path,
                'log_path': execution.log_path,
            },
        ),
        details={
            'load_data': _json_ready(load_data_result),
            'pipeline': _json_ready(pipeline_result),
            'execution': _execution_result_to_dict(execution),
            'export_mode': export_mode,
            'summary': summary_payload,
        },
        ok=execution.returncode == 0,
    )


def _run_segmentation_masks(context: SkillRuntimeContext) -> PipelineSkillResult:
    load_data_path = context.run_root / 'load_data_for_segmentation.csv'
    pipeline_path = context.run_root / 'CPJUMP1_analysis_mask_export.cppipe'
    load_data_result = _native('prepare_segmentation_load_data_native')(context.config, output_path=load_data_path)
    pipeline_result = _native('build_mask_export_pipeline_native')(context.config, output_path=pipeline_path)
    workflow_config = _build_isolated_segmentation_skill_config(
        context.config,
        workflow_root=context.run_root,
        load_data_path=load_data_path,
        pipeline_path=pipeline_path,
    )
    execution, export_mode = _run_segmentation_backend(workflow_config, context)
    sample_previews_result = _native('generate_sample_previews_native')(workflow_config, overwrite=True)
    mask_output_dir = context.run_root / 'cellprofiler_masks'
    summary_path = context.run_root / 'segmentation_summary.json'
    summary_payload = {
        'implementation': 'cellpaint_pipeline.skills.run-segmentation-masks',
        'load_data_path': str(load_data_result.output_path),
        'pipeline_path': str(pipeline_result.output_path),
        'cellprofiler_output_dir': str(mask_output_dir),
        'image_table_path': str(mask_output_dir / 'Image.csv'),
        'cells_table_path': str(mask_output_dir / 'Cells.csv'),
        'nuclei_table_path': str(mask_output_dir / 'Nuclei.csv'),
        'labels_dir': str(mask_output_dir / 'labels'),
        'outlines_dir': str(mask_output_dir / 'outlines'),
        'sample_previews_dir': str(sample_previews_result.output_dir),
        'sample_preview_count': sample_previews_result.generated_count,
        'execution': _execution_result_to_dict(execution),
            'export_mode': export_mode,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.workflows.segmentation',
        primary_outputs=build_primary_outputs(
            SEGMENTATION_MASK_OUTPUTS,
            {
                'load_data_path': load_data_result.output_path,
                'pipeline_path': pipeline_result.output_path,
                'cellprofiler_output_dir': mask_output_dir,
                'image_table_path': mask_output_dir / 'Image.csv',
                'cells_table_path': mask_output_dir / 'Cells.csv',
                'cytoplasm_table_path': mask_output_dir / 'Cytoplasm.csv',
                'nuclei_table_path': mask_output_dir / 'Nuclei.csv',
                'experiment_table_path': mask_output_dir / 'Experiment.csv',
                'labels_dir': mask_output_dir / 'labels',
                'outlines_dir': mask_output_dir / 'outlines',
                'illumination_runtime_dir': mask_output_dir / 'illumination_runtime',
                'load_data_absolute_path': mask_output_dir / 'load_data_for_segmentation.absolute.csv',
                'sample_previews_dir': sample_previews_result.output_dir,
                'workflow_config_path': context.run_root / 'segmentation_workflow_config.json',
                'summary_path': summary_path,
                'log_path': execution.log_path,
            },
        ),
        details={
            'load_data': _json_ready(load_data_result),
            'pipeline': _json_ready(pipeline_result),
            'execution': _execution_result_to_dict(execution),
            'export_mode': export_mode,
            'sample_previews': _json_ready(sample_previews_result),
            'summary': summary_payload,
        },
        ok=execution.returncode == 0,
    )


def _run_generate_sample_previews(context: SkillRuntimeContext) -> PipelineSkillResult:
    source_config = _resolve_segmentation_source_config(context)
    result = _native('generate_sample_previews_native')(source_config, output_dir=context.run_root / 'sample_previews_png', overwrite=context.overwrite)
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.segmentation_native',
        primary_outputs={'sample_previews_dir': result.output_dir},
        details=_json_ready(result),
        ok=True,
    )


def _run_export_single_cell_crops(context: SkillRuntimeContext) -> PipelineSkillResult:
    mode = (context.crop_mode or 'masked').strip().lower()
    if mode not in {'masked', 'unmasked'}:
        raise ValueError(f'Unsupported crop_mode {context.crop_mode!r}. Expected masked or unmasked.')
    return _run_single_cell_crop_skill(context, mode=mode)


def _run_single_cell_crop_skill(context: SkillRuntimeContext, *, mode: str) -> PipelineSkillResult:
    source_config = _resolve_segmentation_source_config(context)
    crops_root = context.run_root / mode
    result = _native('extract_single_cell_crops_native')(
        source_config,
        mode=mode,
        output_dir=crops_root,
        manifest_path=crops_root / 'single_cell_manifest.csv',
        workers=context.workers,
    )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.segmentation_native',
        primary_outputs=build_primary_outputs(
            SINGLE_CELL_CROP_OUTPUTS,
            {
                'crops_dir': result.crops_dir,
                'image_stacks_dir': Path(result.crops_dir) / 'image_stacks',
                'cell_masks_dir': Path(result.crops_dir) / 'cell_masks',
                'nuclei_masks_dir': Path(result.crops_dir) / 'nuclei_masks',
                'manifest_path': result.manifest_path,
                'source_config_path': context.run_root / 'segmentation_source_config.json',
            },
        ),
        details=_json_ready(result),
        ok=True,
    )


def _run_export_masked_single_cell_crops(context: SkillRuntimeContext) -> PipelineSkillResult:
    return _run_single_cell_crop_skill(context, mode='masked')


def _run_export_unmasked_single_cell_crops(context: SkillRuntimeContext) -> PipelineSkillResult:
    return _run_single_cell_crop_skill(context, mode='unmasked')
