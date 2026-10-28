from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellpaint_pipeline.data_access.gallery import (
    GalleryDownloadResult,
    build_gallery_source_prefix,
    download_gallery_prefix,
    download_gallery_source,
    gallery_download_result_to_dict,
)
from cellpaint_pipeline.data_access.summary import DataAccessSummaryResult, summarize_data_access


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class DataRequest:
    mode: str
    dataset_id: str | None
    source_id: str | None
    prefix: str | None
    subprefix: str
    bucket: str | None
    include_substrings: tuple[str, ...]
    exclude_substrings: tuple[str, ...]
    max_files: int | None
    overwrite: bool
    dry_run: bool
    output_dir: Path | None
    manifest_path: Path | None


@dataclass(frozen=True)
class DataDownloadStep:
    step_key: str
    adapter: str
    mode: str
    bucket: str
    prefix: str
    dataset_id: str | None
    source_id: str | None
    subprefix: str
    include_substrings: tuple[str, ...]
    exclude_substrings: tuple[str, ...]
    max_files: int | None
    overwrite: bool
    dry_run: bool
    output_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class DataDownloadPlan:
    request: DataRequest
    resolved_dataset_id: str | None
    resolved_source_id: str | None
    resolved_prefix: str
    steps: list[DataDownloadStep]
    notes: list[str]
    errors: list[str]
    summary_used: bool
    ok: bool


@dataclass(frozen=True)
class DataDownloadStepResult:
    step: DataDownloadStep
    download_result: GalleryDownloadResult
    ok: bool


@dataclass(frozen=True)
class DataDownloadExecutionResult:
    plan: DataDownloadPlan
    step_results: list[DataDownloadStepResult]
    ok: bool


def build_data_request(
    *,
    mode: str = 'gallery-source',
    dataset_id: str | None = None,
    source_id: str | None = None,
    prefix: str | None = None,
    subprefix: str = '',
    bucket: str | None = None,
    include_substrings: list[str] | tuple[str, ...] | None = None,
    exclude_substrings: list[str] | tuple[str, ...] | None = None,
    max_files: int | None = None,
    overwrite: bool = False,
    dry_run: bool = True,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> DataRequest:
    normalized_mode = mode.strip()
    if normalized_mode not in {'gallery-source', 'gallery-prefix'}:
        raise ValueError('mode must be one of: gallery-source, gallery-prefix')
    return DataRequest(
        mode=normalized_mode,
        dataset_id=_clean_optional_string(dataset_id),
        source_id=_clean_optional_string(source_id),
        prefix=_clean_optional_string(prefix),
        subprefix=subprefix.strip(),
        bucket=_clean_optional_string(bucket),
        include_substrings=tuple(str(item) for item in (include_substrings or ())),
        exclude_substrings=tuple(str(item) for item in (exclude_substrings or ())),
        max_files=max_files,
        overwrite=overwrite,
        dry_run=dry_run,
        output_dir=_resolve_optional_path(output_dir),
        manifest_path=_resolve_optional_path(manifest_path),
    )


def build_download_plan(
    config: ProjectConfig,
    request: DataRequest,
    *,
    summary: DataAccessSummaryResult | None = None,
    summary_max_keys: int = 1000,
    validate_with_summary: bool = True,
) -> DataDownloadPlan:
    notes: list[str] = []
    errors: list[str] = []
    resolved_dataset_id: str | None = None
    resolved_source_id: str | None = None
    resolved_prefix = ''
    steps: list[DataDownloadStep] = []
    summary_used = False

    if request.mode == 'gallery-source':
        resolved_dataset_id = request.dataset_id or config.data_access.default_dataset_id
        resolved_source_id = request.source_id or config.data_access.default_source_id
        if not resolved_dataset_id:
            errors.append('dataset_id is required for gallery-source requests when no default is configured.')
        if not resolved_source_id:
            errors.append('source_id is required for gallery-source requests when no default is configured.')
        if not errors:
            resolved_prefix = build_gallery_source_prefix(resolved_dataset_id, resolved_source_id)
            cleaned_subprefix = request.subprefix.strip().strip('/')
            if cleaned_subprefix:
                resolved_prefix = f'{resolved_prefix}{cleaned_subprefix}/'
            if validate_with_summary:
                summary = summary or summarize_data_access(
                    config,
                    dataset_id=resolved_dataset_id,
                    gallery_bucket=request.bucket,
                    gallery_max_keys=summary_max_keys,
                    include_quilt=False,
                    include_cpgdata=False,
                )
                summary_used = True
                _validate_against_summary(summary, resolved_dataset_id, resolved_source_id, notes, errors)
    elif request.mode == 'gallery-prefix':
        if not request.prefix:
            errors.append('prefix is required for gallery-prefix requests.')
        else:
            resolved_prefix = request.prefix.strip()
            if not resolved_prefix.endswith('/'):
                resolved_prefix = f'{resolved_prefix}/'
            notes.append('Prefix-mode request bypasses dataset/source validation.')
    else:
        errors.append(f'Unsupported request mode: {request.mode}')

    resolved_bucket = request.bucket or config.data_access.gallery_bucket
    if resolved_prefix and not errors:
        resolved_output_dir = request.output_dir or (config.data_access.data_cache_root / _default_download_dirname(resolved_prefix))
        resolved_manifest_path = request.manifest_path or (resolved_output_dir / 'download_manifest.json')
        steps.append(
            DataDownloadStep(
                step_key='download-1',
                adapter='gallery',
                mode=request.mode,
                bucket=resolved_bucket,
                prefix=resolved_prefix,
                dataset_id=resolved_dataset_id,
                source_id=resolved_source_id,
                subprefix=request.subprefix.strip(),
                include_substrings=request.include_substrings,
                exclude_substrings=request.exclude_substrings,
                max_files=request.max_files,
                overwrite=request.overwrite,
                dry_run=request.dry_run,
                output_dir=resolved_output_dir,
                manifest_path=resolved_manifest_path,
            )
        )

    return DataDownloadPlan(
        request=request,
        resolved_dataset_id=resolved_dataset_id,
        resolved_source_id=resolved_source_id,
        resolved_prefix=resolved_prefix,
        steps=steps,
        notes=notes,
        errors=errors,
        summary_used=summary_used,
        ok=not errors,
    )


def execute_download_plan(config: ProjectConfig, plan: DataDownloadPlan) -> DataDownloadExecutionResult:
    if not plan.ok:
        raise ValueError('Cannot execute an invalid download plan.')

    step_results: list[DataDownloadStepResult] = []
    for step in plan.steps:
        if step.mode == 'gallery-source':
            download_result = download_gallery_source(
                config,
                dataset_id=step.dataset_id,
                source_id=step.source_id,
                subprefix=step.subprefix,
                bucket=step.bucket,
                output_dir=step.output_dir,
                manifest_path=step.manifest_path,
                include_substrings=step.include_substrings,
                exclude_substrings=step.exclude_substrings,
                max_files=step.max_files,
                overwrite=step.overwrite,
                dry_run=step.dry_run,
            )
        elif step.mode == 'gallery-prefix':
            download_result = download_gallery_prefix(
                config,
                prefix=step.prefix,
                bucket=step.bucket,
                output_dir=step.output_dir,
                manifest_path=step.manifest_path,
                include_substrings=step.include_substrings,
                exclude_substrings=step.exclude_substrings,
                max_files=step.max_files,
                overwrite=step.overwrite,
                dry_run=step.dry_run,
            )
        else:
            raise ValueError(f'Unsupported step mode: {step.mode}')
        step_results.append(
            DataDownloadStepResult(
                step=step,
                download_result=download_result,
                ok=True,
            )
        )

    return DataDownloadExecutionResult(
        plan=plan,
        step_results=step_results,
        ok=all(item.ok for item in step_results),
    )


def write_download_plan(plan: DataDownloadPlan, output_path: Path) -> Path:
    resolved_output_path = output_path.expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        json.dumps(data_download_plan_to_dict(plan), indent=2, ensure_ascii=False) + chr(10),
        encoding='utf-8',
    )
    return resolved_output_path


def load_download_plan(path: Path) -> DataDownloadPlan:
    payload = json.loads(path.expanduser().resolve().read_text(encoding='utf-8'))
    return data_download_plan_from_dict(payload)


def data_request_to_dict(request: DataRequest) -> dict[str, Any]:
    return {
        'mode': request.mode,
        'dataset_id': request.dataset_id,
        'source_id': request.source_id,
        'prefix': request.prefix,
        'subprefix': request.subprefix,
        'bucket': request.bucket,
        'include_substrings': list(request.include_substrings),
        'exclude_substrings': list(request.exclude_substrings),
        'max_files': request.max_files,
        'overwrite': request.overwrite,
        'dry_run': request.dry_run,
        'output_dir': str(request.output_dir) if request.output_dir else None,
        'manifest_path': str(request.manifest_path) if request.manifest_path else None,
    }


def data_request_from_dict(payload: dict[str, Any]) -> DataRequest:
    return build_data_request(
        mode=str(payload['mode']),
        dataset_id=payload.get('dataset_id'),
        source_id=payload.get('source_id'),
        prefix=payload.get('prefix'),
        subprefix=str(payload.get('subprefix', '')),
        bucket=payload.get('bucket'),
        include_substrings=tuple(payload.get('include_substrings', [])),
        exclude_substrings=tuple(payload.get('exclude_substrings', [])),
        max_files=payload.get('max_files'),
        overwrite=bool(payload.get('overwrite', False)),
        dry_run=bool(payload.get('dry_run', True)),
        output_dir=_resolve_optional_path(payload.get('output_dir')),
        manifest_path=_resolve_optional_path(payload.get('manifest_path')),
    )


def data_download_step_to_dict(step: DataDownloadStep) -> dict[str, Any]:
    return {
        'step_key': step.step_key,
        'adapter': step.adapter,
        'mode': step.mode,
        'bucket': step.bucket,
        'prefix': step.prefix,
        'dataset_id': step.dataset_id,
        'source_id': step.source_id,
        'subprefix': step.subprefix,
        'include_substrings': list(step.include_substrings),
        'exclude_substrings': list(step.exclude_substrings),
        'max_files': step.max_files,
        'overwrite': step.overwrite,
        'dry_run': step.dry_run,
        'output_dir': str(step.output_dir),
        'manifest_path': str(step.manifest_path),
    }


def data_download_plan_to_dict(plan: DataDownloadPlan) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.planning',
        'request': data_request_to_dict(plan.request),
        'resolved_dataset_id': plan.resolved_dataset_id,
        'resolved_source_id': plan.resolved_source_id,
        'resolved_prefix': plan.resolved_prefix,
        'steps': [data_download_step_to_dict(step) for step in plan.steps],
        'step_count': len(plan.steps),
        'notes': plan.notes,
        'errors': plan.errors,
        'summary_used': plan.summary_used,
        'ok': plan.ok,
    }


def data_download_plan_from_dict(payload: dict[str, Any]) -> DataDownloadPlan:
    request = data_request_from_dict(dict(payload['request']))
    steps = [
        DataDownloadStep(
            step_key=str(item['step_key']),
            adapter=str(item['adapter']),
            mode=str(item['mode']),
            bucket=str(item['bucket']),
            prefix=str(item['prefix']),
            dataset_id=item.get('dataset_id'),
            source_id=item.get('source_id'),
            subprefix=str(item.get('subprefix', '')),
            include_substrings=tuple(str(token) for token in item.get('include_substrings', [])),
            exclude_substrings=tuple(str(token) for token in item.get('exclude_substrings', [])),
            max_files=item.get('max_files'),
            overwrite=bool(item.get('overwrite', False)),
            dry_run=bool(item.get('dry_run', True)),
            output_dir=Path(item['output_dir']).expanduser().resolve(),
            manifest_path=Path(item['manifest_path']).expanduser().resolve(),
        )
        for item in payload.get('steps', [])
    ]
    return DataDownloadPlan(
        request=request,
        resolved_dataset_id=payload.get('resolved_dataset_id'),
        resolved_source_id=payload.get('resolved_source_id'),
        resolved_prefix=str(payload.get('resolved_prefix', '')),
        steps=steps,
        notes=[str(item) for item in payload.get('notes', [])],
        errors=[str(item) for item in payload.get('errors', [])],
        summary_used=bool(payload.get('summary_used', False)),
        ok=bool(payload.get('ok', False)),
    )


def data_download_step_result_to_dict(result: DataDownloadStepResult) -> dict[str, Any]:
    return {
        'step': data_download_step_to_dict(result.step),
        'download_result': gallery_download_result_to_dict(result.download_result),
        'ok': result.ok,
    }


def data_download_execution_result_to_dict(result: DataDownloadExecutionResult) -> dict[str, Any]:
    return {
        'implementation': 'cellpaint_pipeline.data_access.planning',
        'plan': data_download_plan_to_dict(result.plan),
        'step_results': [data_download_step_result_to_dict(item) for item in result.step_results],
        'step_result_count': len(result.step_results),
        'ok': result.ok,
    }


def _validate_against_summary(
    summary: DataAccessSummaryResult,
    dataset_id: str,
    source_id: str,
    notes: list[str],
    errors: list[str],
) -> None:
    dataset_listing = summary.dataset_listing
    if dataset_listing is not None and dataset_id not in dataset_listing.entries:
        message = f"dataset_id {dataset_id!r} was not present in the gallery dataset summary."
        if dataset_listing.is_truncated:
            notes.append(message + ' Listing was truncated, so this may be incomplete.')
        else:
            errors.append(message)
    source_listing = summary.source_listing
    if source_listing is not None and summary.resolved_dataset_id == dataset_id:
        if source_id not in source_listing.entries:
            message = f"source_id {source_id!r} was not present in the gallery source summary for dataset {dataset_id!r}."
            if source_listing.is_truncated:
                notes.append(message + ' Listing was truncated, so this may be incomplete.')
            else:
                errors.append(message)
    elif source_listing is None:
        notes.append('No gallery source listing was available for plan validation.')


def _clean_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _default_download_dirname(prefix: str) -> str:
    return prefix.strip('/').replace('/', '__') or '__root__'
