from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, replace
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from cellpaint_pipeline.adapters.deepprofiler import export_deepprofiler_input
from cellpaint_pipeline.adapters.deepprofiler import infer_deepprofiler_sources_from_workflow_root
from cellpaint_pipeline.adapters.deepprofiler_features import collect_deepprofiler_features
from cellpaint_pipeline.adapters.deepprofiler_project import build_deepprofiler_project
from cellpaint_pipeline.adapters.deepprofiler_project import run_deepprofiler_profile

# The workflow catalogue lives in ``cellpaint_pipeline.capabilities`` so that
# lower layers (the validation report) can read it without importing this
# module.  Re-exported here because ``WORKFLOW_KEYS`` and
# ``available_workflows`` have always been importable from this path.
from cellpaint_pipeline.capabilities import WORKFLOW_KEYS as _WORKFLOW_KEYS
from cellpaint_pipeline.capabilities import available_workflows as _available_workflows
from cellpaint_pipeline.evaluation import run_native_evaluation
from cellpaint_pipeline.profiling_native import run_pycytominer_native
from cellpaint_pipeline.runner import CommandExecutionError, ExecutionResult
from cellpaint_pipeline.segmentation_native import summarize_segmentation_outputs, write_segmentation_summary
from cellpaint_pipeline.workflows.profiling import run_profiling_native, run_profiling_task
from cellpaint_pipeline.workflows.segmentation import run_segmentation_native, run_segmentation_script, run_segmentation_task
from typing import TYPE_CHECKING
from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class WorkflowResult:
    workflow_key: str
    steps: list[dict[str, str | int | bool | None]]
    export_root: Path | None = None
    manifest_path: Path | None = None


@dataclass(frozen=True)
class MaskExportBundle:
    workflow_config: ProjectConfig
    steps: list[dict[str, str | int | bool | None]]
    summary: object
    summary_path: Path


class WorkflowExecutionError(RuntimeError):
    """Raised when a named workflow cannot complete successfully."""

    def __init__(
        self,
        workflow_key: str,
        workflow_root: Path,
        *,
        step_label: str | None = None,
        reason: str,
        hint: str | None = None,
        details: list[str] | None = None,
    ) -> None:
        self.workflow_key = workflow_key
        self.workflow_root = workflow_root
        self.step_label = step_label
        self.reason = reason
        self.hint = hint
        self.details = list(details or [])
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        lines = [
            f"Workflow '{self.workflow_key}' failed.",
            f"Workflow root: {self.workflow_root}",
        ]
        if self.step_label:
            lines.append(f"Step: {self.step_label}")
        lines.append(f"Reason: {self.reason}")
        if self.hint:
            lines.append(f"Hint: {self.hint}")
        if self.details:
            lines.append('Details:')
            lines.extend(f"- {detail}" for detail in self.details)
        return '\n'.join(lines)


WORKFLOW_KEYS = _WORKFLOW_KEYS


def available_workflows() -> list[str]:
    return _available_workflows()


WorkflowHandler = Callable[
    [ProjectConfig, str],
    tuple[list[dict[str, str | int | bool | None]], Path | None],
]


def _run_full_post_mvp_with_script_eval(
    config: ProjectConfig,
    workflow_key: str,
    *,
    workflow_root: Path,
    extra_args: list[str] | None,
) -> tuple[list[dict[str, str | int | bool | None]], Path | None]:
    profiling_result = run_profiling_task(config, 'full-post-mvp', extra_args)
    evaluation_result = run_profiling_task(config, 'evaluation-only')
    steps = [
        _external_step_record(profiling_result),
        _external_step_record(evaluation_result),
    ]
    return steps, None
def _run_full_post_mvp_with_native_eval(
    config: ProjectConfig,
    workflow_key: str,
    *,
    workflow_root: Path,
    extra_args: list[str] | None,
) -> tuple[list[dict[str, str | int | bool | None]], Path | None]:
    profiling_result = run_profiling_task(config, 'full-post-mvp', extra_args)
    evaluation_result = run_native_evaluation(config, output_dir=workflow_root / 'evaluation')
    steps = [
        _external_step_record(profiling_result),
        {
            'label': 'native_evaluation',
            'mode': 'native',
            'returncode': 0,
            'output_dir': str(evaluation_result.output_dir),
            'n_wells': evaluation_result.n_wells,
        },
    ]
    return steps, None
def _run_post_cellprofiler_native_profiling_with_native_eval(
    config: ProjectConfig,
    workflow_key: str,
    *,
    workflow_root: Path,
    extra_args: list[str] | None,
) -> tuple[list[dict[str, str | int | bool | None]], Path | None]:
    single_cell_path = workflow_root / 'single_cell.csv.gz'
    pycytominer_output_dir = workflow_root / 'pycytominer'
    evaluation_output_dir = workflow_root / 'evaluation'

    validation_result = run_profiling_native(config, 'validate-inputs')
    if not validation_result.ok:
        raise WorkflowExecutionError(
            workflow_key,
            workflow_root,
            step_label='validate_inputs',
            reason='Profiling input validation failed.',
            hint='Fix the reported raw image, manifest, or plate map problems before rerunning this workflow.',
            details=[
                f'Manifest path: {validation_result.manifest_path}',
                f'Plate map path: {validation_result.plate_map_path}',
                *_summarize_problem_details(validation_result.problems),
            ],
        )

    single_cell_result = run_profiling_native(
        config,
        'export-cellprofiler-to-singlecell',
        output_path=single_cell_path,
    )
    pycytominer_result = run_pycytominer_native(
        config,
        output_dir=pycytominer_output_dir,
        single_cell_path=single_cell_path,
    )
    evaluation_result = run_native_evaluation(
        config,
        output_dir=evaluation_output_dir,
        feature_selected_path=pycytominer_result.feature_selected_path,
        annotated_path=pycytominer_result.annotated_path,
    )
    steps = [
        {
            'label': 'validate_inputs',
            'mode': 'native',
            'returncode': 0,
            'ok': validation_result.ok,
            'problem_count': len(validation_result.problems),
            'manifest_path': str(validation_result.manifest_path),
            'plate_map_path': str(validation_result.plate_map_path),
        },
        {
            'label': 'export_cellprofiler_to_singlecell',
            'mode': 'native',
            'returncode': 0,
            'output_path': str(single_cell_result.output_path),
            'row_count': single_cell_result.row_count,
            'column_count': single_cell_result.column_count,
            'shard_count': single_cell_result.shard_count,
        },
        {
            'label': 'run_pycytominer',
            'mode': 'native',
            'returncode': 0,
            'output_dir': str(pycytominer_output_dir),
            'feature_selected_path': str(pycytominer_result.feature_selected_path),
            'feature_selected_row_count': pycytominer_result.feature_selected_row_count,
            'feature_selected_column_count': pycytominer_result.feature_selected_column_count,
        },
        {
            'label': 'native_evaluation',
            'mode': 'native',
            'returncode': 0,
            'output_dir': str(evaluation_result.output_dir),
            'n_wells': evaluation_result.n_wells,
            'n_feature_columns': evaluation_result.n_feature_columns,
        },
    ]
    return steps, None
def _run_post_cellprofiler_native_segmentation_suite(
    config: ProjectConfig,
    workflow_key: str,
    *,
    workflow_root: Path,
    extra_args: list[str] | None,
) -> tuple[list[dict[str, str | int | bool | None]], Path | None]:
    segmentation_root = workflow_root / 'segmentation_native'
    masked_root = segmentation_root / 'masked'
    unmasked_root = segmentation_root / 'unmasked'

    load_data_result = run_segmentation_native(
        config,
        'prepare-load-data',
        output_path=workflow_root / 'load_data_for_segmentation.csv',
    )
    pipeline_result = run_segmentation_native(
        config,
        'build-mask-export-pipeline',
        output_path=workflow_root / 'CPJUMP1_analysis_mask_export.cppipe',
    )
    masked_crops_result = run_segmentation_native(
        config,
        'extract-single-cell-crops',
        mode='masked',
        output_path=masked_root,
    )
    unmasked_crops_result = run_segmentation_native(
        config,
        'extract-single-cell-crops',
        mode='unmasked',
        output_path=unmasked_root,
    )
    sample_previews_result = run_segmentation_native(
        config,
        'generate-sample-previews',
        output_path=segmentation_root / 'sample_previews_png',
        overwrite=True,
    )
    masked_previews_result = run_segmentation_native(
        config,
        'generate-png-previews',
        mode='masked',
        output_path=masked_root / 'previews_png',
        manifest_path=masked_crops_result.manifest_path,
    )
    unmasked_previews_result = run_segmentation_native(
        config,
        'generate-png-previews',
        mode='unmasked',
        output_path=unmasked_root / 'previews_png',
        manifest_path=unmasked_crops_result.manifest_path,
    )

    segmentation_summary_path = workflow_root / 'segmentation_suite_summary.json'
    segmentation_summary = {
        'implementation': 'native',
        'workflow_key': workflow_key,
        'load_data_path': str(load_data_result.output_path),
        'load_data_rows': load_data_result.row_count,
        'mask_export_pipeline_path': str(pipeline_result.output_path),
        'source_cppipe_path': str(pipeline_result.source_cppipe_path),
        'selected_via': pipeline_result.selected_via,
        'execution_mode': pipeline_result.execution_mode,
        'masked_manifest_path': str(masked_crops_result.manifest_path),
        'masked_crop_count': masked_crops_result.crop_count,
        'unmasked_manifest_path': str(unmasked_crops_result.manifest_path),
        'unmasked_crop_count': unmasked_crops_result.crop_count,
        'sample_previews_dir': str(sample_previews_result.output_dir),
        'sample_preview_count': sample_previews_result.field_count,
        'masked_previews_dir': str(masked_previews_result.output_dir),
        'masked_preview_count': masked_previews_result.preview_count,
        'unmasked_previews_dir': str(unmasked_previews_result.output_dir),
        'unmasked_preview_count': unmasked_previews_result.preview_count,
    }
    segmentation_summary['ok'] = bool(
        masked_crops_result.crop_count == unmasked_crops_result.crop_count
        and masked_previews_result.preview_count == masked_crops_result.crop_count
        and unmasked_previews_result.preview_count == unmasked_crops_result.crop_count
        and sample_previews_result.field_count > 0
    )
    segmentation_summary_path.write_text(json.dumps(segmentation_summary, indent=2) + chr(10), encoding='utf-8')

    steps = [
        {
            'label': 'prepare_load_data',
            'mode': 'native',
            'returncode': 0,
            'output_path': str(load_data_result.output_path),
            'row_count': load_data_result.row_count,
        },
        {
            'label': 'build_mask_export_pipeline',
            'mode': 'native',
            'returncode': 0,
            'output_path': str(pipeline_result.output_path),
            'module_count': pipeline_result.module_count,
            'source_cppipe_path': str(pipeline_result.source_cppipe_path),
            'selected_via': pipeline_result.selected_via,
            'execution_mode': pipeline_result.execution_mode,
        },
        {
            'label': 'extract_single_cell_crops_masked',
            'mode': 'native',
            'returncode': 0,
            'manifest_path': str(masked_crops_result.manifest_path),
            'crop_count': masked_crops_result.crop_count,
        },
        {
            'label': 'extract_single_cell_crops_unmasked',
            'mode': 'native',
            'returncode': 0,
            'manifest_path': str(unmasked_crops_result.manifest_path),
            'crop_count': unmasked_crops_result.crop_count,
        },
        {
            'label': 'generate_sample_previews',
            'mode': 'native',
            'returncode': 0,
            'output_dir': str(sample_previews_result.output_dir),
            'field_count': sample_previews_result.field_count,
        },
        {
            'label': 'generate_png_previews_masked',
            'mode': 'native',
            'returncode': 0,
            'output_dir': str(masked_previews_result.output_dir),
            'preview_count': masked_previews_result.preview_count,
        },
        {
            'label': 'generate_png_previews_unmasked',
            'mode': 'native',
            'returncode': 0,
            'output_dir': str(unmasked_previews_result.output_dir),
            'preview_count': unmasked_previews_result.preview_count,
        },
        {
            'label': 'segmentation_suite_summary',
            'mode': 'native',
            'returncode': 0,
            'ok': segmentation_summary['ok'],
            'summary_path': str(segmentation_summary_path),
        },
    ]
    return steps, None
def _run_mask_export_script_with_native_postprocessing(
    config: ProjectConfig,
    workflow_key: str,
    *,
    workflow_root: Path,
    extra_args: list[str] | None,
) -> tuple[list[dict[str, str | int | bool | None]], Path | None]:
    bundle = _run_mask_export_postprocessing_bundle(
        config,
        workflow_root=workflow_root,
        extra_args=extra_args,
    )
    return bundle.steps, None
def _run_segmentation_and_deepprofiler_export(
    config: ProjectConfig,
    workflow_key: str,
    *,
    workflow_root: Path,
    extra_args: list[str] | None,
) -> tuple[list[dict[str, str | int | bool | None]], Path | None]:
    bundle = _run_mask_export_postprocessing_bundle(
        config,
        workflow_root=workflow_root,
        extra_args=extra_args,
    )
    source_paths = infer_deepprofiler_sources_from_workflow_root(workflow_root)
    _require_workflow_paths(
        source_paths,
        workflow_key=workflow_key,
        workflow_root=workflow_root,
        step_label='export_deepprofiler_input',
        reason='DeepProfiler export inputs are missing from the workflow root.',
        hint='Confirm that the preceding mask-export workflow completed and wrote CellProfiler tables into the workflow-local output directory.',
    )
    export_result = export_deepprofiler_input(
        bundle.workflow_config,
        output_dir=workflow_root / 'deepprofiler_export',
        image_csv_path=source_paths['image_csv_path'],
        nuclei_csv_path=source_paths['nuclei_csv_path'],
        load_data_csv_path=source_paths['load_data_csv_path'],
        source_label='workflow-local-mask-export',
    )
    steps = [
        *bundle.steps,
        {
            'label': 'export_deepprofiler_input',
            'mode': 'native',
            'returncode': 0,
            'export_root': str(export_result.export_root),
            'manifest_path': str(export_result.manifest_path),
            'field_count': export_result.field_count,
            'location_file_count': export_result.location_file_count,
            'total_nuclei': export_result.total_nuclei,
        },
    ]
    return steps, export_result.export_root
def _run_segmentation_and_deepprofiler_full_stack(
    config: ProjectConfig,
    workflow_key: str,
    *,
    workflow_root: Path,
    extra_args: list[str] | None,
) -> tuple[list[dict[str, str | int | bool | None]], Path | None]:
    bundle = _run_mask_export_postprocessing_bundle(
    config,
    workflow_root=workflow_root,
    extra_args=extra_args,
    )
    source_paths = infer_deepprofiler_sources_from_workflow_root(workflow_root)
    _require_workflow_paths(
    source_paths,
    workflow_key=workflow_key,
    workflow_root=workflow_root,
    step_label='export_deepprofiler_input',
    reason='DeepProfiler export inputs are missing from the workflow root.',
    hint='Confirm that the preceding mask-export workflow completed and wrote CellProfiler tables into the workflow-local output directory.',
    )
    export_result = export_deepprofiler_input(
    bundle.workflow_config,
    output_dir=workflow_root / 'deepprofiler_export',
    image_csv_path=source_paths['image_csv_path'],
    nuclei_csv_path=source_paths['nuclei_csv_path'],
    load_data_csv_path=source_paths['load_data_csv_path'],
    source_label='workflow-local-mask-export',
    )
    project_result = build_deepprofiler_project(
    config,
    output_dir=workflow_root / 'deepprofiler_project',
    export_root=export_result.export_root,
    )
    profile_result = run_deepprofiler_profile(
    config,
    project_root=project_result.project_root,
    experiment_name=project_result.experiment_name,
    )
    collection_result = collect_deepprofiler_features(
    config,
    project_root=project_result.project_root,
    output_dir=workflow_root / 'deepprofiler_tables',
    experiment_name=project_result.experiment_name,
    )
    steps = [
    *bundle.steps,
    {
        'label': 'export_deepprofiler_input',
        'mode': 'native',
        'returncode': 0,
        'export_root': str(export_result.export_root),
        'manifest_path': str(export_result.manifest_path),
        'field_count': export_result.field_count,
        'location_file_count': export_result.location_file_count,
        'total_nuclei': export_result.total_nuclei,
    },
    {
        'label': 'build_deepprofiler_project',
        'mode': 'native',
        'returncode': 0,
        'project_root': str(project_result.project_root),
        'manifest_path': str(project_result.manifest_path),
        'field_count': project_result.field_count,
        'experiment_name': project_result.experiment_name,
    },
    {
        'label': 'run_deepprofiler_profile',
        'mode': 'native',
        'returncode': profile_result.returncode,
        'feature_dir': str(profile_result.feature_dir),
        'log_path': str(profile_result.log_path) if profile_result.log_path else None,
        'experiment_name': profile_result.experiment_name,
    },
    {
        'label': 'collect_deepprofiler_features',
        'mode': 'native',
        'returncode': 0,
        'output_dir': str(collection_result.output_dir),
        'manifest_path': str(collection_result.manifest_path),
        'field_file_count': collection_result.field_file_count,
        'cell_count': collection_result.cell_count,
        'feature_count': collection_result.feature_count,
        'well_count': collection_result.well_count,
    },
    ]
    return steps, collection_result.output_dir


WORKFLOW_HANDLERS: dict[str, WorkflowHandler] = {
    'full-post-mvp-with-script-eval': _run_full_post_mvp_with_script_eval,
    'full-post-mvp-with-native-eval': _run_full_post_mvp_with_native_eval,
    'post-cellprofiler-native-profiling-with-native-eval': _run_post_cellprofiler_native_profiling_with_native_eval,
    'post-cellprofiler-native-segmentation-suite': _run_post_cellprofiler_native_segmentation_suite,
    'mask-export-script-with-native-postprocessing': _run_mask_export_script_with_native_postprocessing,
    'segmentation-and-deepprofiler-export': _run_segmentation_and_deepprofiler_export,
    'segmentation-and-deepprofiler-full-stack': _run_segmentation_and_deepprofiler_full_stack,
}


def run_workflow(
    config: ProjectConfig,
    workflow_key: str,
    *,
    extra_args: list[str] | None = None,
    export_output_dir: Path | None = None,
) -> WorkflowResult:
    """Run one packaged workflow.

    Each key is served by a registered handler, so adding a workflow means
    adding a function plus one registry entry instead of editing a growing
    conditional.  The handler returns the step records and the optional
    export root; manifest writing stays in :func:`_finalize_workflow`.
    """
    handler = WORKFLOW_HANDLERS.get(workflow_key)
    if handler is None:
        raise KeyError(f"Unknown workflow: {workflow_key}. Available: {', '.join(available_workflows())}")

    workflow_root = _resolve_workflow_root(config, workflow_key, export_output_dir)
    workflow_root.mkdir(parents=True, exist_ok=True)

    try:
        steps, export_root = handler(
            config,
            workflow_key,
            workflow_root=workflow_root,
            extra_args=extra_args,
        )
        return _finalize_workflow(config, workflow_key, workflow_root, steps, export_root=export_root)
    except WorkflowExecutionError:
        raise
    except Exception as exc:
        raise _wrap_workflow_exception(workflow_key, workflow_root, exc) from exc
def _run_mask_export_postprocessing_bundle(
    config: ProjectConfig,
    *,
    workflow_root: Path,
    extra_args: list[str] | None,
) -> MaskExportBundle:
    load_data_path = workflow_root / 'load_data_for_segmentation.csv'
    pipeline_path = workflow_root / 'CPJUMP1_analysis_mask_export.cppipe'

    load_data_result = run_segmentation_native(
        config,
        'prepare-load-data',
        output_path=load_data_path,
    )
    pipeline_result = run_segmentation_native(
        config,
        'build-mask-export-pipeline',
        output_path=pipeline_path,
    )
    workflow_config = _build_isolated_segmentation_workflow_config(
        config,
        workflow_root=workflow_root,
        load_data_path=load_data_path,
        pipeline_path=pipeline_path,
    )

    mask_export_args = [
        '--config',
        str(workflow_config.segmentation_backend_config),
        '--reuse-load-data',
        '--reuse-pipeline',
    ]
    if extra_args:
        mask_export_args.extend(extra_args)
    mask_export_result = run_segmentation_script(
        workflow_config,
        'run-mask-export',
        mask_export_args,
    )

    masked_crops_result = run_segmentation_native(workflow_config, 'extract-single-cell-crops', mode='masked')
    unmasked_crops_result = run_segmentation_native(workflow_config, 'extract-single-cell-crops', mode='unmasked')
    sample_previews_result = run_segmentation_native(workflow_config, 'generate-sample-previews', overwrite=True)
    masked_previews_result = run_segmentation_native(workflow_config, 'generate-png-previews', mode='masked')
    unmasked_previews_result = run_segmentation_native(workflow_config, 'generate-png-previews', mode='unmasked')
    summary = summarize_segmentation_outputs(workflow_config)
    summary_path = write_segmentation_summary(summary, workflow_root / 'segmentation_summary.json')

    steps = [
        {
            'label': 'prepare_load_data',
            'mode': 'native',
            'returncode': 0,
            'output_path': str(load_data_result.output_path),
            'row_count': load_data_result.row_count,
        },
        {
            'label': 'build_mask_export_pipeline',
            'mode': 'native',
            'returncode': 0,
            'output_path': str(pipeline_result.output_path),
            'module_count': pipeline_result.module_count,
            'source_cppipe_path': str(pipeline_result.source_cppipe_path),
            'selected_via': pipeline_result.selected_via,
            'execution_mode': pipeline_result.execution_mode,
        },
        _external_step_record(mask_export_result),
        {
            'label': 'extract_single_cell_crops_masked',
            'mode': 'native',
            'returncode': 0,
            'manifest_path': str(masked_crops_result.manifest_path),
            'crop_count': masked_crops_result.crop_count,
        },
        {
            'label': 'extract_single_cell_crops_unmasked',
            'mode': 'native',
            'returncode': 0,
            'manifest_path': str(unmasked_crops_result.manifest_path),
            'crop_count': unmasked_crops_result.crop_count,
        },
        {
            'label': 'generate_sample_previews',
            'mode': 'native',
            'returncode': 0,
            'output_dir': str(sample_previews_result.output_dir),
            'field_count': sample_previews_result.field_count,
        },
        {
            'label': 'generate_png_previews_masked',
            'mode': 'native',
            'returncode': 0,
            'output_dir': str(masked_previews_result.output_dir),
            'preview_count': masked_previews_result.preview_count,
        },
        {
            'label': 'generate_png_previews_unmasked',
            'mode': 'native',
            'returncode': 0,
            'output_dir': str(unmasked_previews_result.output_dir),
            'preview_count': unmasked_previews_result.preview_count,
        },
        {
            'label': 'segmentation_summary',
            'mode': 'native',
            'returncode': 0,
            'ok': summary.ok,
            'masked_manifest_rows': summary.masked_manifest_rows,
            'unmasked_manifest_rows': summary.unmasked_manifest_rows,
            'sample_preview_count': summary.sample_preview_count,
            'summary_path': str(summary_path),
        },
    ]
    return MaskExportBundle(
        workflow_config=workflow_config,
        steps=steps,
        summary=summary,
        summary_path=summary_path,
    )


def _summarize_problem_details(problems: list[str], *, limit: int = 5) -> list[str]:
    summarized = list(problems[:limit])
    if len(problems) > limit:
        summarized.append(f'... plus {len(problems) - limit} more problem(s).')
    return summarized


def _require_workflow_paths(
    paths: dict[str, Path],
    *,
    workflow_key: str,
    workflow_root: Path,
    step_label: str,
    reason: str,
    hint: str,
) -> None:
    missing = [f'{name}: {path}' for name, path in paths.items() if not path.exists()]
    if missing:
        raise WorkflowExecutionError(
            workflow_key,
            workflow_root,
            step_label=step_label,
            reason=reason,
            hint=hint,
            details=missing,
        )


def _wrap_workflow_exception(
    workflow_key: str,
    workflow_root: Path,
    exc: Exception,
) -> WorkflowExecutionError:
    if isinstance(exc, CommandExecutionError):
        details = [f'Command: {shlex.join(exc.command)}']
        if exc.cwd is not None:
            details.append(f'CWD: {exc.cwd}')
        if exc.log_path is not None:
            details.append(f'Log: {exc.log_path}')
        if exc.output_tail:
            tail_preview = ' | '.join(exc.output_tail[-5:])
            details.append(f'Output tail preview: {tail_preview}')
        return WorkflowExecutionError(
            workflow_key,
            workflow_root,
            step_label=exc.label,
            reason='A subprocess-backed workflow step failed.',
            hint='Inspect the log/output details above, fix the backend script or environment issue, and rerun the workflow.',
            details=details,
        )
    if isinstance(exc, FileNotFoundError):
        return WorkflowExecutionError(
            workflow_key,
            workflow_root,
            reason='A required workflow file or directory is missing.',
            hint='Check backend config paths and confirm that prerequisite steps have written their expected outputs.',
            details=[str(exc)],
        )
    if isinstance(exc, ValueError):
        return WorkflowExecutionError(
            workflow_key,
            workflow_root,
            reason='Workflow validation failed.',
            hint='Review the workflow configuration and intermediate outputs referenced below.',
            details=[str(exc)],
        )
    return WorkflowExecutionError(
        workflow_key,
        workflow_root,
        reason=f'Unhandled {type(exc).__name__}: {exc}',
        hint='Inspect the exception details and rerun after correcting the underlying issue.',
    )


def _resolve_workflow_root(config: ProjectConfig, workflow_key: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return config.default_output_root / 'workflows' / workflow_key


def _build_isolated_segmentation_workflow_config(
    config: ProjectConfig,
    *,
    workflow_root: Path,
    load_data_path: Path,
    pipeline_path: Path,
) -> ProjectConfig:
    payload = config.load_segmentation_backend_payload()
    payload['project_name'] = f"{config.project_name}_workflow_mask_export"
    payload['paths']['load_data_csv'] = str(load_data_path)
    payload['paths']['mask_export_pipeline'] = str(pipeline_path)
    payload['paths']['cellprofiler_output_dir'] = str(workflow_root / 'cellprofiler_masks')
    payload['paths']['sample_previews_dir'] = str(workflow_root / 'segmentation_native' / 'sample_previews_png')
    payload['paths']['masked_crops_dir'] = str(workflow_root / 'segmentation_native' / 'masked')
    payload['paths']['masked_manifest_csv'] = str(workflow_root / 'segmentation_native' / 'masked' / 'single_cell_manifest.csv')
    payload['paths']['unmasked_crops_dir'] = str(workflow_root / 'segmentation_native' / 'unmasked')
    payload['paths']['unmasked_manifest_csv'] = str(workflow_root / 'segmentation_native' / 'unmasked' / 'single_cell_manifest.csv')
    runtime_payload = dict(config.mask_export_runtime)
    runtime_payload['work_root'] = str(workflow_root / '.mask_export_sharded_work')
    payload['mask_export_runtime'] = runtime_payload

    runtime_config_path = workflow_root / 'segmentation_workflow_config.json'
    runtime_config_path.write_text(json.dumps(payload, indent=2) + chr(10), encoding='utf-8')
    return replace(config, segmentation_backend_config=runtime_config_path)


def _finalize_workflow(
    config: ProjectConfig,
    workflow_key: str,
    workflow_root: Path,
    steps: list[dict[str, str | int | bool | None]],
    *,
    export_root: Path | None,
) -> WorkflowResult:
    manifest_path = workflow_root / 'workflow_manifest.json'
    manifest = {
        'implementation': 'cellpaint_pipeline.workflow',
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'project_name': config.project_name,
        'workflow_key': workflow_key,
        'workflow_root': str(workflow_root),
        'export_root': str(export_root) if export_root else None,
        'steps': steps,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + chr(10), encoding='utf-8')
    return WorkflowResult(
        workflow_key=workflow_key,
        steps=steps,
        export_root=export_root,
        manifest_path=manifest_path,
    )


def _external_step_record(result: ExecutionResult) -> dict[str, str | int | None]:
    return {
        'label': result.label,
        'mode': 'script',
        'returncode': result.returncode,
        'log_path': str(result.log_path) if result.log_path else None,
    }
