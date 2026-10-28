from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from cellpaint_pipeline.config import ConfigContractError


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


CppipeKind = Literal['profiling', 'segmentation']
CppipeExecutionMode = Literal['analysis', 'illumination', 'derive-mask-export', 'ready-mask-export']


@dataclass(frozen=True)
class CppipeTemplateDefinition:
    key: str
    kind: CppipeKind
    description: str
    backend: Literal['profiling', 'segmentation']
    config_path_key: str
    execution_mode: CppipeExecutionMode
    recommended_for: str


@dataclass(frozen=True)
class ResolvedCppipeSelection:
    kind: CppipeKind
    selected_via: Literal['template', 'custom']
    template_key: str | None
    cppipe_path: Path
    execution_mode: CppipeExecutionMode
    exists: bool
    description: str


@dataclass(frozen=True)
class CppipeValidationResult:
    ok: bool
    selections: tuple[ResolvedCppipeSelection, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


CPPPIPE_TEMPLATE_DEFINITIONS: dict[str, CppipeTemplateDefinition] = {
    'profiling-analysis': CppipeTemplateDefinition(
        key='profiling-analysis',
        kind='profiling',
        description='Bundled CellProfiler analysis pipeline used by the profiling backend.',
        backend='profiling',
        config_path_key='cellprofiler_pipeline_analysis',
        execution_mode='analysis',
        recommended_for='The standard profiling-side CellProfiler analysis pipeline asset.',
    ),
    'profiling-illumination': CppipeTemplateDefinition(
        key='profiling-illumination',
        kind='profiling',
        description='Bundled CellProfiler illumination-correction pipeline used by the profiling backend.',
        backend='profiling',
        config_path_key='cellprofiler_pipeline_illumination',
        execution_mode='illumination',
        recommended_for='Profiling-side illumination correction and related backend inspection.',
    ),
    'segmentation-base': CppipeTemplateDefinition(
        key='segmentation-base',
        kind='segmentation',
        description='Bundled base analysis pipeline used as the source when generating the mask-export pipeline.',
        backend='segmentation',
        config_path_key='base_pipeline',
        execution_mode='derive-mask-export',
        recommended_for='The default segmentation path that derives a mask-export-ready pipeline at runtime.',
    ),
    'segmentation-mask-export': CppipeTemplateDefinition(
        key='segmentation-mask-export',
        kind='segmentation',
        description='Bundled mask-export-ready segmentation pipeline used directly without runtime patching.',
        backend='segmentation',
        config_path_key='mask_export_pipeline',
        execution_mode='ready-mask-export',
        recommended_for='A ready-to-run segmentation pipeline that already includes mask export modules.',
    ),
}


def available_cppipe_templates(kind: CppipeKind | None = None) -> list[str]:
    keys = [
        key
        for key, definition in CPPPIPE_TEMPLATE_DEFINITIONS.items()
        if kind is None or definition.kind == kind
    ]
    return sorted(keys)


def get_cppipe_template(template_key: str) -> CppipeTemplateDefinition:
    if template_key not in CPPPIPE_TEMPLATE_DEFINITIONS:
        available = ', '.join(available_cppipe_templates())
        raise KeyError(f'Unknown .cppipe template: {template_key}. Available: {available}')
    return CPPPIPE_TEMPLATE_DEFINITIONS[template_key]


def cppipe_template_definition_to_dict(
    definition: CppipeTemplateDefinition,
    *,
    config: ProjectConfig | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'key': definition.key,
        'kind': definition.kind,
        'description': definition.description,
        'backend': definition.backend,
        'config_path_key': definition.config_path_key,
        'execution_mode': definition.execution_mode,
        'recommended_for': definition.recommended_for,
    }
    if config is not None:
        try:
            payload['resolved_path'] = str(resolve_cppipe_template_path(config, definition.key))
        except Exception as exc:  # pragma: no cover - defensive only
            payload['resolved_path_error'] = str(exc)
    return payload


def resolved_cppipe_selection_to_dict(selection: ResolvedCppipeSelection) -> dict[str, Any]:
    return {
        'kind': selection.kind,
        'selected_via': selection.selected_via,
        'template_key': selection.template_key,
        'cppipe_path': str(selection.cppipe_path),
        'execution_mode': selection.execution_mode,
        'exists': selection.exists,
        'description': selection.description,
    }


def cppipe_validation_result_to_dict(result: CppipeValidationResult) -> dict[str, Any]:
    return {
        'ok': result.ok,
        'selections': [resolved_cppipe_selection_to_dict(item) for item in result.selections],
        'errors': list(result.errors),
        'warnings': list(result.warnings),
    }


def resolve_cppipe_template_path(config: ProjectConfig, template_key: str) -> Path:
    definition = get_cppipe_template(template_key)
    if definition.backend == 'profiling':
        payload = config.load_profiling_backend_payload()
        relative_path = payload['paths'][definition.config_path_key]
        return config.resolve_profiling_backend_path(relative_path)
    payload = config.load_segmentation_backend_payload()
    relative_path = payload['paths'][definition.config_path_key]
    return config.resolve_segmentation_backend_path(relative_path)


def resolve_cppipe_selection(config: ProjectConfig, kind: CppipeKind) -> ResolvedCppipeSelection:
    if kind == 'profiling':
        custom_path = config.cellprofiler.custom_profiling_cppipe_path
        template_key = config.cellprofiler.profiling_template
        custom_mode: CppipeExecutionMode = 'analysis'
        custom_description = 'Custom profiling .cppipe override path provided through project config.'
    else:
        custom_path = config.cellprofiler.custom_segmentation_cppipe_path
        template_key = config.cellprofiler.segmentation_template
        custom_mode = 'ready-mask-export'
        custom_description = (
            'Custom segmentation .cppipe override path provided through project config. '
            'Phase-1 behavior treats this as a ready-to-run mask-export pipeline.'
        )

    if custom_path is not None:
        return ResolvedCppipeSelection(
            kind=kind,
            selected_via='custom',
            template_key=template_key,
            cppipe_path=custom_path,
            execution_mode=custom_mode,
            exists=custom_path.exists(),
            description=custom_description,
        )

    definition = get_cppipe_template(template_key)
    if definition.kind != kind:
        raise ConfigContractError(
            f"Configured {kind} template '{template_key}' is not a {kind} .cppipe template."
        )
    resolved_path = resolve_cppipe_template_path(config, template_key)
    return ResolvedCppipeSelection(
        kind=kind,
        selected_via='template',
        template_key=template_key,
        cppipe_path=resolved_path,
        execution_mode=definition.execution_mode,
        exists=resolved_path.exists(),
        description=definition.description,
    )


def validate_cppipe_configuration(config: ProjectConfig) -> CppipeValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    selections: list[ResolvedCppipeSelection] = []

    for kind in ('profiling', 'segmentation'):
        try:
            selection = resolve_cppipe_selection(config, kind)
        except Exception as exc:
            errors.append(f'{kind}: {exc}')
            continue

        selections.append(selection)
        if selection.cppipe_path.suffix.lower() != '.cppipe':
            errors.append(
                f"{kind}: selected path does not end with .cppipe: {selection.cppipe_path}"
            )
        if not selection.exists:
            errors.append(f'{kind}: selected .cppipe does not exist: {selection.cppipe_path}')
        if selection.selected_via == 'custom' and kind == 'segmentation':
            warnings.append(
                'segmentation: custom overrides are treated as ready-to-run mask-export pipelines in phase 1.'
            )

    return CppipeValidationResult(
        ok=len(errors) == 0,
        selections=tuple(selections),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
