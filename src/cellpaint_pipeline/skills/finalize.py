from __future__ import annotations

"""Shared result assembly for every skill runner."""

import json
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellpaint_pipeline.runner import ExecutionResult
from cellpaint_pipeline.skills.context import SkillRuntimeContext
from cellpaint_pipeline.skills.definitions import (
    PipelineSkillResult,
    pipeline_skill_definition_to_dict,
)


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


def _finalize_skill_result(
    context: SkillRuntimeContext,
    *,
    implementation: str,
    primary_outputs: dict[str, Path | None],
    details: dict[str, Any],
    ok: bool,
) -> PipelineSkillResult:
    manifest_path = context.run_root / 'pipeline_skill_manifest.json'
    manifest_payload = {
        'implementation': implementation,
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'skill': pipeline_skill_definition_to_dict(context.definition),
        'output_dir': str(context.run_root),
        'primary_outputs': {key: str(value) if value is not None else None for key, value in primary_outputs.items()},
        'details': details,
        'ok': ok,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return PipelineSkillResult(
        skill_key=context.definition.key,
        category=context.definition.category,
        implementation=implementation,
        output_dir=context.run_root,
        manifest_path=manifest_path,
        primary_outputs=primary_outputs,
        details=details,
        ok=ok,
    )


def _resolve_segmentation_source_config(context: SkillRuntimeContext) -> ProjectConfig:
    if context.workflow_root is None:
        return context.config
    payload = context.config.load_segmentation_backend_payload()
    payload['paths']['load_data_csv'] = str(context.workflow_root / 'load_data_for_segmentation.csv')
    payload['paths']['cellprofiler_output_dir'] = str(context.workflow_root / 'cellprofiler_masks')
    runtime_config_path = context.run_root / 'segmentation_source_config.json'
    runtime_config_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return replace(context.config, segmentation_backend_config=runtime_config_path)


def _build_isolated_segmentation_skill_config(
    config: ProjectConfig,
    *,
    workflow_root: Path,
    load_data_path: Path,
    pipeline_path: Path,
) -> ProjectConfig:
    payload = config.load_segmentation_backend_payload()
    payload['project_name'] = f'{config.project_name}_skill_mask_export'
    payload['paths']['load_data_csv'] = str(load_data_path)
    payload['paths']['mask_export_pipeline'] = str(pipeline_path)
    payload['paths']['cellprofiler_output_dir'] = str(workflow_root / 'cellprofiler_masks')
    payload['paths']['sample_previews_dir'] = str(workflow_root / 'sample_previews_png')
    payload['paths']['masked_crops_dir'] = str(workflow_root / 'masked')
    payload['paths']['masked_manifest_csv'] = str(workflow_root / 'masked' / 'single_cell_manifest.csv')
    payload['paths']['unmasked_crops_dir'] = str(workflow_root / 'unmasked')
    payload['paths']['unmasked_manifest_csv'] = str(workflow_root / 'unmasked' / 'single_cell_manifest.csv')
    runtime_payload = dict(config.mask_export_runtime)
    runtime_payload['work_root'] = str(workflow_root / '.mask_export_sharded_work')
    payload['mask_export_runtime'] = runtime_payload
    runtime_config_path = workflow_root / 'segmentation_workflow_config.json'
    runtime_config_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return replace(config, segmentation_backend_config=runtime_config_path)


def _execution_result_to_dict(result: ExecutionResult) -> dict[str, Any]:
    return {
        'label': result.label,
        'command': list(result.command),
        'cwd': str(result.cwd) if result.cwd is not None else None,
        'log_path': str(result.log_path) if result.log_path is not None else None,
        'returncode': result.returncode,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _native(name: str):
    """Return the native implementation name from the package facade.

    Runners call their native implementations through the facade rather than
    importing them directly, so that patch('cellpaint_pipeline.skills.<name>')
    keeps isolating a runner exactly as it did before the split.  The facade
    is resolved at call time so this module stays importable while the
    package itself is still initialising.
    """
    from cellpaint_pipeline import skills as _skills

    return getattr(_skills, name)
