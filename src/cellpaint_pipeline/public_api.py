from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellpaint_pipeline.registry import (
    ENTRYPOINT_REQUIRES_CONFIG as _ENTRYPOINT_REQUIRES_CONFIG,
)
from cellpaint_pipeline.registry import (
    PATHLIKE_KEYWORDS as _PATHLIKE_KEYWORDS,
)
from cellpaint_pipeline.registry import (
    PLAN_KEYWORDS,
    REQUEST_KEYWORDS,
    is_result_serialised_inline,
    requires_config,
    result_serialiser,
    target_module,
)


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


class PublicApiContractError(ValueError):
    """Raised when a stable public API entrypoint is called with an invalid high-level contract."""


@dataclass(frozen=True)
class PublicApiEntry:
    name: str
    layer: str
    category: str
    stability: str
    description: str
    recommended_for: str
    returns: str
    cli_command: str | None = None


PUBLIC_API_ENTRYPOINTS: dict[str, PublicApiEntry] = {
    'summarize_data_access': PublicApiEntry(
        name='summarize_data_access',
        layer='data-access',
        category='discovery',
        stability='public',
        description='Inspect gallery, Quilt, and cpgdata availability through one summary object.',
        recommended_for='Dataset discovery, environment checks, and planning before any transfer or workflow execution.',
        returns='DataAccessSummary',
        cli_command='summarize-data-access',
    ),
    'build_data_request': PublicApiEntry(
        name='build_data_request',
        layer='data-access',
        category='planning',
        stability='public',
        description='Create a serializable request object describing what gallery data should be fetched.',
        recommended_for='Programmatic construction of gallery-source or gallery-prefix requests.',
        returns='DataRequest',
        cli_command=None,
    ),
    'build_download_plan': PublicApiEntry(
        name='build_download_plan',
        layer='data-access',
        category='planning',
        stability='public',
        description='Resolve a DataRequest into an explicit download plan.',
        recommended_for='Creating reusable, inspectable transfer plans before execution.',
        returns='DataDownloadPlan',
        cli_command='plan-data-access',
    ),
    'execute_download_plan': PublicApiEntry(
        name='execute_download_plan',
        layer='data-access',
        category='execution',
        stability='public',
        description='Execute a previously generated download plan.',
        recommended_for='Controlled download execution after a plan has been reviewed or persisted.',
        returns='DataDownloadExecutionResult',
        cli_command='execute-download-plan',
    ),
    'run_profiling_suite': PublicApiEntry(
        name='run_profiling_suite',
        layer='delivery',
        category='workflow-suite',
        stability='public',
        description='Run a packaged profiling suite through a stable alias.',
        recommended_for='Direct profiling-only execution when you do not need full orchestration.',
        returns='SuiteRunResult',
        cli_command='run-profiling-suite',
    ),
    'run_segmentation_suite': PublicApiEntry(
        name='run_segmentation_suite',
        layer='delivery',
        category='workflow-suite',
        stability='public',
        description='Run a packaged segmentation suite through a stable alias.',
        recommended_for='Direct segmentation-only execution when you do not need full orchestration.',
        returns='SuiteRunResult',
        cli_command='run-segmentation-suite',
    ),
    'run_end_to_end_pipeline': PublicApiEntry(
        name='run_end_to_end_pipeline',
        layer='orchestration',
        category='top-level',
        stability='public',
        description='Top-level orchestrator spanning data access, profiling, segmentation, DeepProfiler routing, and validation.',
        recommended_for='The default library entrypoint when you want one standard Cell Painting pipeline run.',
        returns='EndToEndPipelineResult',
        cli_command='run-end-to-end-pipeline',
    ),
    'run_pipeline_preset': PublicApiEntry(
        name='run_pipeline_preset',
        layer='preset',
        category='named-configuration',
        stability='public',
        description='Execute a named preset that bundles a common orchestration parameter combination.',
        recommended_for='Stable notebook or service calls where task shape is known but raw orchestration arguments are noisy.',
        returns='EndToEndPipelineResult',
        cli_command='run-pipeline-preset',
    ),
    'run_pipeline_skill': PublicApiEntry(
        name='run_pipeline_skill',
        layer='skill',
        category='task-entrypoint',
        stability='public',
        description='Execute a named small-granularity skill that produces one concrete Cell Painting output.',
        recommended_for='Humans or agents that want stable task names such as cp-extract-segmentation-artifacts, crop-export-single-cell-crops, or dp-run-deep-feature-model.',
        returns='PipelineSkillResult',
        cli_command='run-pipeline-skill',
    ),
    'run_deepprofiler_pipeline': PublicApiEntry(
        name='run_deepprofiler_pipeline',
        layer='deepprofiler',
        category='top-level',
        stability='public',
        description='Run the standardized DeepProfiler export, project-build, profile, and feature-collection chain.',
        recommended_for='Entering the DeepProfiler branch from an existing segmentation workflow root or CSV source trio.',
        returns='DeepProfilerPipelineResult',
        cli_command='run-deepprofiler-pipeline',
    ),
}

#: Entrypoints that refuse to run without a :class:`ProjectConfig`.  The set
#: itself lives in :mod:`cellpaint_pipeline.registry` so the public API layer
#: and the MCP layer cannot drift apart; this name is kept because it is part
#: of the published module surface.
PUBLIC_API_REQUIRES_CONFIG = _ENTRYPOINT_REQUIRES_CONFIG

#: Keyword arguments that are resolved to absolute paths before dispatch.
PATHLIKE_KWARGS = _PATHLIKE_KEYWORDS

PUBLIC_API_OUTPUT_CONTRACTS: dict[str, dict[str, Any]] = {
    'summarize_data_access': {
        'output_kind': 'structured-object',
        'primary_fields': ['ok', 'resolved_dataset_id', 'dataset_listing', 'source_listing', 'errors'],
        'primary_artifacts': [],
        'supporting_fields': ['quilt_package_listing', 'cpgdata_prefix_listing'],
        'notes': 'Returns an in-memory discovery summary. No manifest is written unless a higher-level caller persists it.',
    },
    'build_data_request': {
        'output_kind': 'structured-object',
        'primary_fields': ['mode', 'dry_run', 'dataset_id', 'source_id', 'prefix'],
        'primary_artifacts': [],
        'supporting_fields': ['bucket', 'output_dir', 'manifest_path'],
        'notes': 'Returns a serializable request object that can be reused for planning.',
    },
    'build_download_plan': {
        'output_kind': 'structured-object',
        'primary_fields': ['ok', 'resolved_prefix', 'steps', 'errors'],
        'primary_artifacts': [
            {
                'path_field': 'steps[].manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Per-step download manifest destination resolved during planning.',
            },
        ],
        'supporting_fields': ['summary_used', 'request'],
        'notes': 'The plan is returned in memory. Persist it separately if you want a reusable JSON plan file.',
    },
    'execute_download_plan': {
        'output_kind': 'run-artifacts',
        'primary_fields': ['ok', 'plan', 'step_results'],
        'primary_artifacts': [
            {
                'path_field': 'step_results[].download_result.manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Download manifest written by each executed gallery download step.',
            },
            {
                'path_field': 'step_results[].download_result.output_dir',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Destination directory populated by each executed download step.',
            },
        ],
        'supporting_fields': ['step_results[].download_result.downloaded_count'],
        'notes': 'Execution writes one manifest per download step; higher orchestration layers may also write a combined execution summary.',
    },
    'run_profiling_suite': {
        'output_kind': 'run-artifacts',
        'primary_fields': ['suite_key', 'workflow_key', 'output_dir', 'manifest_path', 'step_count'],
        'primary_artifacts': [
            {
                'path_field': 'output_dir',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Profiling suite run root.',
            },
            {
                'path_field': 'manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Workflow manifest describing the profiling suite run.',
            },
        ],
        'supporting_fields': [],
        'notes': 'This contract covers profiling-only delivery runs, not the lower-level workflow internals inside the suite.',
    },
    'run_segmentation_suite': {
        'output_kind': 'run-artifacts',
        'primary_fields': ['suite_key', 'workflow_key', 'output_dir', 'manifest_path', 'step_count'],
        'primary_artifacts': [
            {
                'path_field': 'output_dir',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Segmentation suite run root.',
            },
            {
                'path_field': 'manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Workflow manifest describing the segmentation suite run.',
            },
        ],
        'supporting_fields': [],
        'notes': 'Formal outputs include masks, single-cell crops, or DeepProfiler export artifacts depending on the suite key.',
    },
    'run_end_to_end_pipeline': {
        'output_kind': 'run-artifacts',
        'primary_fields': ['ok', 'output_dir', 'manifest_path', 'run_report_path'],
        'primary_artifacts': [
            {
                'path_field': 'output_dir',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Top-level delivery root for the standard Cell Painting run.',
            },
            {
                'path_field': 'manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Machine-readable manifest for the whole run.',
            },
            {
                'path_field': 'run_report_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Human-readable run summary for reporting and manual inspection.',
            },
            {
                'path_field': 'profiling_manifest_path',
                'required': False,
                'artifact_class': 'formal-output',
                'role': 'Profiling suite manifest when profiling is enabled.',
            },
            {
                'path_field': 'segmentation_manifest_path',
                'required': False,
                'artifact_class': 'formal-output',
                'role': 'Segmentation suite manifest when segmentation is enabled.',
            },
            {
                'path_field': 'validation_report_path',
                'required': False,
                'artifact_class': 'formal-output',
                'role': 'Validation summary JSON when validation reporting is enabled.',
            },
        ],
        'supporting_fields': ['download_plan_path', 'download_execution_path', 'profiling_output_dir', 'segmentation_output_dir', 'stage_count'],
        'notes': 'Preview PNGs, sample images, and other branch-specific helper files may appear under nested output roots, but the manifest and run report are the canonical top-level artifacts.',
    },
    'run_pipeline_preset': {
        'output_kind': 'run-artifacts',
        'primary_fields': ['ok', 'output_dir', 'manifest_path', 'run_report_path'],
        'primary_artifacts': [
            {
                'path_field': 'output_dir',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Top-level delivery root produced by the selected preset.',
            },
            {
                'path_field': 'manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Machine-readable manifest for the preset-backed run.',
            },
            {
                'path_field': 'run_report_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Human-readable report for the preset-backed run.',
            },
        ],
        'supporting_fields': ['profiling_manifest_path', 'segmentation_manifest_path', 'validation_report_path'],
        'notes': 'Preset execution reuses the standard end-to-end output contract.',
    },
    'run_pipeline_skill': {
        'output_kind': 'run-artifacts',
        'primary_fields': ['ok', 'skill_key', 'output_dir', 'manifest_path', 'primary_outputs'],
        'primary_artifacts': [
            {
                'path_field': 'output_dir',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Per-skill run directory that stores the skill manifest and any skill-local artifacts.',
            },
            {
                'path_field': 'manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Machine-readable manifest describing the selected skill execution.',
            },
        ],
        'supporting_fields': ['category', 'implementation', 'details'],
        'notes': 'Each skill returns a skill-specific manifest plus concrete primary output paths instead of reusing the full end-to-end pipeline contract.',
    },
    'run_deepprofiler_pipeline': {
        'output_kind': 'run-artifacts',
        'primary_fields': ['ok', 'output_dir', 'manifest_path', 'collection_manifest_path'],
        'primary_artifacts': [
            {
                'path_field': 'output_dir',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Top-level DeepProfiler pipeline run root.',
            },
            {
                'path_field': 'manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Machine-readable manifest for the full DeepProfiler branch run.',
            },
            {
                'path_field': 'export_manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Manifest describing the DeepProfiler export inputs.',
            },
            {
                'path_field': 'project_manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Manifest describing the generated DeepProfiler project.',
            },
            {
                'path_field': 'collection_manifest_path',
                'required': True,
                'artifact_class': 'formal-output',
                'role': 'Feature-collection manifest describing tables and counts.',
            },
        ],
        'supporting_fields': ['export_root', 'project_root', 'feature_dir', 'collection_output_dir', 'field_count', 'cell_count', 'feature_count', 'well_count'],
        'notes': 'The collected feature tables and manifests are the canonical DeepProfiler outputs; intermediate exports and project files are retained as supporting artifacts.',
    },
}


def recommended_public_api_pathways() -> list[dict[str, Any]]:
    ordered_names = [
        'run_end_to_end_pipeline',
        'run_pipeline_skill',
        'run_pipeline_preset',
        'run_deepprofiler_pipeline',
    ]
    pathway_notes = {
        'run_end_to_end_pipeline': {
            'audience': 'default-human-and-service-entrypoint',
            'why': 'Prefer this when you want one standard Cell Painting run without choosing lower-level workflow bundles yourself.',
        },
        'run_pipeline_skill': {
            'audience': 'task-oriented-automation',
            'why': 'Prefer this for agents or automation layers that should call named tasks instead of assembling raw orchestration parameters.',
        },
        'run_pipeline_preset': {
            'audience': 'named-configuration-callers',
            'why': 'Prefer this when the workflow shape is known and you want a stable preset key instead of repeated parameter bundles.',
        },
        'run_deepprofiler_pipeline': {
            'audience': 'deepprofiler-direct-branch',
            'why': 'Prefer this only when you are intentionally entering the DeepProfiler branch from existing segmentation or CSV inputs.',
        },
    }
    recommendations: list[dict[str, Any]] = []
    for priority, name in enumerate(ordered_names, start=1):
        entry = get_public_api_entrypoint(name)
        payload = public_api_entrypoint_to_dict(entry)
        payload['priority'] = priority
        payload['audience'] = pathway_notes[name]['audience']
        payload['why'] = pathway_notes[name]['why']
        recommendations.append(payload)
    return recommendations


def available_public_api_entrypoints() -> list[str]:
    return list(PUBLIC_API_ENTRYPOINTS)


def available_public_api_output_contracts() -> list[str]:
    return list(PUBLIC_API_OUTPUT_CONTRACTS)


def get_public_api_output_contract(name: str) -> dict[str, Any]:
    if name not in PUBLIC_API_OUTPUT_CONTRACTS:
        available = ', '.join(available_public_api_output_contracts())
        raise PublicApiContractError(f'Unknown public API output contract: {name}. Available: {available}')
    payload = {'name': name}
    payload.update(PUBLIC_API_OUTPUT_CONTRACTS[name])
    return payload


def public_api_output_contract_summary() -> list[dict[str, Any]]:
    return [get_public_api_output_contract(name) for name in available_public_api_output_contracts()]


def get_public_api_entrypoint(name: str) -> PublicApiEntry:
    if name not in PUBLIC_API_ENTRYPOINTS:
        available = ', '.join(available_public_api_entrypoints())
        raise PublicApiContractError(f'Unknown public API entrypoint: {name}. Available: {available}')
    return PUBLIC_API_ENTRYPOINTS[name]


def public_api_entrypoint_to_dict(entry: PublicApiEntry) -> dict[str, Any]:
    return {
        'name': entry.name,
        'layer': entry.layer,
        'category': entry.category,
        'stability': entry.stability,
        'description': entry.description,
        'recommended_for': entry.recommended_for,
        'returns': entry.returns,
        'cli_command': entry.cli_command,
    }


def public_api_contract_summary() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for name in available_public_api_entrypoints():
        entry = get_public_api_entrypoint(name)
        grouped.setdefault(entry.layer, []).append(public_api_entrypoint_to_dict(entry))
    return grouped


def run_public_api_entrypoint(
    name: str,
    *,
    config: ProjectConfig | None = None,
    **kwargs: Any,
) -> Any:
    get_public_api_entrypoint(name)
    function = _resolve_public_api_function(name)
    resolved_kwargs = _normalize_public_api_kwargs(kwargs)
    if config is not None and not isinstance(config, ProjectConfig):
        raise PublicApiContractError(
            f'Public API entrypoint {name} expected config to be a ProjectConfig, got {type(config).__name__}.'
        )
    if requires_config(name):
        if config is None:
            raise PublicApiContractError(
                f'Public API entrypoint {name} requires a ProjectConfig. '
                f'Load one with ProjectConfig.from_json(...) first.'
            )
        return function(config, **resolved_kwargs)
    return function(**resolved_kwargs)


def run_public_api_entrypoint_to_dict(
    name: str,
    *,
    config: ProjectConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    result = run_public_api_entrypoint(name, config=config, **kwargs)
    payload = _public_api_result_to_dict(name, result)
    payload['entrypoint'] = name
    payload['contract'] = public_api_entrypoint_to_dict(get_public_api_entrypoint(name))
    payload['output_contract'] = get_public_api_output_contract(name)
    return payload


def _normalize_public_api_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(kwargs)
    for key in PATHLIKE_KWARGS:
        value = resolved.get(key)
        if isinstance(value, str) and value.strip():
            resolved[key] = Path(value).expanduser().resolve()
    for key in REQUEST_KEYWORDS:
        value = resolved.get(key)
        if isinstance(value, dict):
            from cellpaint_pipeline.data_access import build_data_request

            resolved[key] = build_data_request(**value)
    for key in PLAN_KEYWORDS:
        value = resolved.get(key)
        if isinstance(value, (str, Path)):
            from cellpaint_pipeline.data_access import load_download_plan

            plan_path = Path(value).expanduser().resolve()
            if not plan_path.exists():
                raise PublicApiContractError(f'Download plan path does not exist: {plan_path}')
            resolved[key] = load_download_plan(plan_path)
    if 'extra_args' in resolved and resolved['extra_args'] is not None:
        resolved['extra_args'] = list(resolved['extra_args'])
    return resolved


def _resolve_public_api_function(name: str) -> Any:
    module_name = target_module(name)
    if module_name is None:
        raise PublicApiContractError(f'No callable registered for public API entrypoint: {name}')
    module = import_module(module_name)
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise PublicApiContractError(f'No callable registered for public API entrypoint: {name}') from exc


def _public_api_result_to_dict(name: str, result: Any) -> dict[str, Any]:
    serialiser = result_serialiser(name)
    if serialiser is not None:
        return serialiser(result)
    if is_result_serialised_inline(name):
        return {
            'implementation': 'cellpaint_pipeline.delivery',
            'suite_key': result.suite_key,
            'workflow_key': result.workflow_key,
            'output_dir': str(result.output_dir),
            'manifest_path': str(result.manifest_path) if result.manifest_path else None,
            'step_count': result.step_count,
        }
    raise PublicApiContractError(f'No serializer registered for public API entrypoint: {name}')
