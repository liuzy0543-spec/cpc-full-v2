from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellpaint_pipeline import __version__
from cellpaint_pipeline.data_access import (
    DataAccessSummaryResult,
    DataDownloadExecutionResult,
    DataDownloadPlan,
    DataRequest,
    build_data_request,
    build_download_plan,
    data_access_summary_to_dict,
    data_download_execution_result_to_dict,
    execute_download_plan,
    summarize_data_access,
    write_download_plan,
)
from cellpaint_pipeline.delivery import (
    available_profiling_suites,
    available_segmentation_suites,
    run_profiling_suite,
    run_segmentation_suite,
)
from cellpaint_pipeline.reporting import build_validation_report_payload
from cellpaint_pipeline.config import ProjectConfig


DEEPPROFILER_MODES = ('off', 'export', 'full')


@dataclass(frozen=True)
class EndToEndPipelineResult:
    output_dir: Path
    manifest_path: Path
    run_report_path: Path
    data_access_summary_path: Path | None
    download_plan_path: Path | None
    download_execution_path: Path | None
    profiling_output_dir: Path | None
    profiling_manifest_path: Path | None
    segmentation_output_dir: Path | None
    segmentation_manifest_path: Path | None
    validation_report_path: Path | None
    profiling_suite: str | None
    segmentation_suite: str | None
    deepprofiler_mode: str
    stage_count: int
    ok: bool


def available_deepprofiler_modes() -> list[str]:
    return list(DEEPPROFILER_MODES)


def resolve_segmentation_suite(
    segmentation_suite: str,
    *,
    deepprofiler_mode: str = 'off',
) -> str:
    normalized_mode = deepprofiler_mode.strip().lower()
    if normalized_mode not in DEEPPROFILER_MODES:
        raise ValueError(f'Unsupported deepprofiler_mode: {deepprofiler_mode}')
    if normalized_mode == 'off':
        return segmentation_suite
    if normalized_mode == 'export':
        return 'deepprofiler-export'
    return 'deepprofiler-full'


def run_end_to_end_pipeline(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    data_request: DataRequest | None = None,
    download_plan: DataDownloadPlan | None = None,
    include_data_access_summary: bool = False,
    plan_data_download: bool = False,
    execute_data_download_step: bool = False,
    data_summary_max_keys: int = 1000,
    profiling_suite: str = 'native',
    segmentation_suite: str = 'mask-export',
    run_profiling: bool = True,
    run_segmentation: bool = True,
    include_validation_report: bool = True,
    deepprofiler_mode: str = 'off',
) -> EndToEndPipelineResult:
    normalized_deepprofiler_mode = deepprofiler_mode.strip().lower()
    if normalized_deepprofiler_mode not in DEEPPROFILER_MODES:
        available = ', '.join(DEEPPROFILER_MODES)
        raise ValueError(f'deepprofiler_mode must be one of: {available}')

    if run_profiling and profiling_suite not in available_profiling_suites():
        available = ', '.join(available_profiling_suites())
        raise KeyError(f'Unknown profiling suite: {profiling_suite}. Available: {available}')

    resolved_segmentation_suite = resolve_segmentation_suite(
        segmentation_suite,
        deepprofiler_mode=normalized_deepprofiler_mode,
    )
    if run_segmentation and resolved_segmentation_suite not in available_segmentation_suites():
        available = ', '.join(available_segmentation_suites())
        raise KeyError(f'Unknown segmentation suite: {resolved_segmentation_suite}. Available: {available}')

    if normalized_deepprofiler_mode != 'off' and not run_segmentation:
        raise ValueError('run_segmentation must be True when deepprofiler_mode is export or full.')

    run_root = output_dir.resolve() if output_dir is not None else (config.default_output_root / 'deliveries' / 'end_to_end_pipeline')
    run_root.mkdir(parents=True, exist_ok=True)

    stage_records: list[dict[str, Any]] = []

    summary_result: DataAccessSummaryResult | None = None
    data_access_summary_path: Path | None = None
    if include_data_access_summary or _should_build_summary_for_planning(data_request, download_plan, plan_data_download, execute_data_download_step):
        request_for_summary = download_plan.request if download_plan is not None else _ensure_data_request(data_request)
        summary_result = summarize_data_access(
            config,
            dataset_id=request_for_summary.dataset_id,
            gallery_bucket=request_for_summary.bucket,
            gallery_max_keys=data_summary_max_keys,
            include_gallery=True,
            include_quilt=include_data_access_summary,
            include_cpgdata=include_data_access_summary,
        )
        if include_data_access_summary:
            data_access_summary_path = run_root / 'data_access_summary.json'
            data_access_summary_path.write_text(
                json.dumps(data_access_summary_to_dict(summary_result), indent=2, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
        stage_records.append({
            'stage': 'data_access_summary',
            'ok': summary_result.ok,
            'path': str(data_access_summary_path) if data_access_summary_path else None,
            'resolved_dataset_id': summary_result.resolved_dataset_id,
            'gallery_dataset_count': len(summary_result.dataset_listing.entries) if summary_result.dataset_listing else 0,
            'gallery_source_count': len(summary_result.source_listing.entries) if summary_result.source_listing else 0,
        })

    planned_download: DataDownloadPlan | None = None
    download_plan_path: Path | None = None
    if download_plan is not None:
        planned_download = download_plan
    elif plan_data_download or execute_data_download_step:
        planned_download = build_download_plan(
            config,
            _ensure_data_request(data_request),
            summary=summary_result,
            summary_max_keys=data_summary_max_keys,
            validate_with_summary=True,
        )

    if planned_download is not None:
        download_plan_path = write_download_plan(planned_download, run_root / 'download_plan.json')
        stage_records.append({
            'stage': 'download_plan',
            'ok': planned_download.ok,
            'path': str(download_plan_path),
            'step_count': len(planned_download.steps),
            'resolved_prefix': planned_download.resolved_prefix,
            'summary_used': planned_download.summary_used,
            'error_count': len(planned_download.errors),
        })

    download_execution_result: DataDownloadExecutionResult | None = None
    download_execution_path: Path | None = None
    if execute_data_download_step:
        if planned_download is None:
            raise ValueError('execute_data_download_step requires a built or supplied download plan.')
        if not planned_download.ok:
            raise ValueError('Cannot execute an invalid download plan.')
        download_execution_result = execute_download_plan(config, planned_download)
        download_execution_path = run_root / 'download_execution.json'
        download_execution_path.write_text(
            json.dumps(data_download_execution_result_to_dict(download_execution_result), indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        stage_records.append({
            'stage': 'download_execution',
            'ok': download_execution_result.ok,
            'path': str(download_execution_path),
            'step_count': len(download_execution_result.step_results),
        })

    profiling_output_dir: Path | None = None
    profiling_manifest_path: Path | None = None
    if run_profiling:
        profiling_result = run_profiling_suite(
            config,
            profiling_suite,
            output_dir=run_root / 'profiling',
        )
        profiling_output_dir = profiling_result.output_dir
        profiling_manifest_path = profiling_result.manifest_path
        stage_records.append({
            'stage': 'profiling',
            'ok': True,
            'suite_key': profiling_result.suite_key,
            'workflow_key': profiling_result.workflow_key,
            'output_dir': str(profiling_result.output_dir),
            'manifest_path': str(profiling_result.manifest_path) if profiling_result.manifest_path else None,
            'step_count': profiling_result.step_count,
        })

    segmentation_output_dir: Path | None = None
    segmentation_manifest_path: Path | None = None
    if run_segmentation:
        segmentation_result = run_segmentation_suite(
            config,
            resolved_segmentation_suite,
            output_dir=run_root / 'segmentation',
        )
        segmentation_output_dir = segmentation_result.output_dir
        segmentation_manifest_path = segmentation_result.manifest_path
        stage_records.append({
            'stage': 'segmentation',
            'ok': True,
            'suite_key': segmentation_result.suite_key,
            'workflow_key': segmentation_result.workflow_key,
            'output_dir': str(segmentation_result.output_dir),
            'manifest_path': str(segmentation_result.manifest_path) if segmentation_result.manifest_path else None,
            'step_count': segmentation_result.step_count,
        })

    validation_report_path: Path | None = None
    if include_validation_report:
        validation_payload = build_validation_report_payload(config)
        validation_report_path = run_root / 'validation_report.json'
        validation_report_path.write_text(
            json.dumps(validation_payload, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        stage_records.append({
            'stage': 'validation_report',
            'ok': bool(validation_payload.get('ok')),
            'path': str(validation_report_path),
            'workflow_manifest_count': validation_payload.get('workflow_manifest_count'),
            'artifact_counts': validation_payload.get('artifact_counts'),
        })

    result = EndToEndPipelineResult(
        output_dir=run_root,
        manifest_path=run_root / 'end_to_end_pipeline_manifest.json',
        run_report_path=run_root / 'run_report.md',
        data_access_summary_path=data_access_summary_path,
        download_plan_path=download_plan_path,
        download_execution_path=download_execution_path,
        profiling_output_dir=profiling_output_dir,
        profiling_manifest_path=profiling_manifest_path,
        segmentation_output_dir=segmentation_output_dir,
        segmentation_manifest_path=segmentation_manifest_path,
        validation_report_path=validation_report_path,
        profiling_suite=profiling_suite if run_profiling else None,
        segmentation_suite=resolved_segmentation_suite if run_segmentation else None,
        deepprofiler_mode=normalized_deepprofiler_mode,
        stage_count=len(stage_records),
        ok=_infer_overall_ok(
            planned_download=planned_download,
            download_execution_result=download_execution_result,
        ),
    )
    result.manifest_path.write_text(
        json.dumps(end_to_end_pipeline_result_to_dict(result, stage_records=stage_records), indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    result.run_report_path.write_text(
        _build_run_report_text(config, result, stage_records),
        encoding='utf-8',
    )
    return result


def end_to_end_pipeline_result_to_dict(
    result: EndToEndPipelineResult,
    *,
    stage_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        'implementation': 'cellpaint_pipeline.orchestration',
        'generated_utc': _utc_now(),
        'package_version': __version__,
        'output_dir': str(result.output_dir),
        'manifest_path': str(result.manifest_path),
        'run_report_path': str(result.run_report_path),
        'data_access_summary_path': str(result.data_access_summary_path) if result.data_access_summary_path else None,
        'download_plan_path': str(result.download_plan_path) if result.download_plan_path else None,
        'download_execution_path': str(result.download_execution_path) if result.download_execution_path else None,
        'profiling_output_dir': str(result.profiling_output_dir) if result.profiling_output_dir else None,
        'profiling_manifest_path': str(result.profiling_manifest_path) if result.profiling_manifest_path else None,
        'segmentation_output_dir': str(result.segmentation_output_dir) if result.segmentation_output_dir else None,
        'segmentation_manifest_path': str(result.segmentation_manifest_path) if result.segmentation_manifest_path else None,
        'validation_report_path': str(result.validation_report_path) if result.validation_report_path else None,
        'profiling_suite': result.profiling_suite,
        'segmentation_suite': result.segmentation_suite,
        'deepprofiler_mode': result.deepprofiler_mode,
        'stage_count': result.stage_count,
        'ok': result.ok,
    }
    if stage_records is not None:
        payload['stages'] = stage_records
    return payload


def _should_build_summary_for_planning(
    data_request: DataRequest | None,
    download_plan: DataDownloadPlan | None,
    plan_data_download: bool,
    execute_data_download_step: bool,
) -> bool:
    if download_plan is not None:
        return False
    if not (plan_data_download or execute_data_download_step):
        return False
    request = _ensure_data_request(data_request)
    return request.mode == 'gallery-source'


def _ensure_data_request(data_request: DataRequest | None) -> DataRequest:
    if data_request is not None:
        return data_request
    return build_data_request(mode='gallery-source')


def _infer_overall_ok(
    *,
    planned_download: DataDownloadPlan | None,
    download_execution_result: DataDownloadExecutionResult | None,
) -> bool:
    if planned_download is not None and not planned_download.ok:
        return False
    if download_execution_result is not None and not download_execution_result.ok:
        return False
    return True


def _build_run_report_text(
    config: ProjectConfig,
    result: EndToEndPipelineResult,
    stage_records: list[dict[str, Any]],
) -> str:
    lines = [
        '# End-to-End Pipeline Run Report',
        '',
        f'- Project: {config.project_name}',
        f'- Generated UTC: {_utc_now()}',
        f'- Package version: {__version__}',
        f'- Output directory: {result.output_dir}',
        f'- Overall ok: {result.ok}',
        f'- DeepProfiler mode: {result.deepprofiler_mode}',
        f'- Profiling suite: {result.profiling_suite}',
        f'- Segmentation suite: {result.segmentation_suite}',
        '',
        '## Key Outputs',
        '',
        f'- Manifest: {result.manifest_path}',
        f'- Run report: {result.run_report_path}',
        f'- Data access summary: {result.data_access_summary_path}',
        f'- Download plan: {result.download_plan_path}',
        f'- Download execution: {result.download_execution_path}',
        f'- Profiling manifest: {result.profiling_manifest_path}',
        f'- Segmentation manifest: {result.segmentation_manifest_path}',
        f'- Validation report: {result.validation_report_path}',
        '',
        '## Stages',
        '',
    ]
    for index, stage in enumerate(stage_records, start=1):
        stage_name = stage.get('stage', f'stage_{index}')
        lines.append(f'{index}. {stage_name}')
        for key, value in stage.items():
            if key == 'stage':
                continue
            lines.append(f'   - {key}: {value}')
    lines.append('')
    return '\n'.join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
