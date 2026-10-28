from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellpaint_pipeline.data_access import build_data_request, load_download_plan
from cellpaint_pipeline.orchestration import end_to_end_pipeline_result_to_dict
from cellpaint_pipeline.presets import (
    available_pipeline_presets,
    get_pipeline_preset_definition,
    pipeline_preset_definition_to_dict,
    run_pipeline_preset,
)
from cellpaint_pipeline.public_api import (
    available_public_api_entrypoints,
    get_public_api_entrypoint,
    public_api_contract_summary,
    public_api_entrypoint_to_dict,
    run_public_api_entrypoint_to_dict,
)
from cellpaint_pipeline.registry import PLAN_KEYWORDS, REQUEST_KEYWORDS, pathlike_keywords
from cellpaint_pipeline.skills import (
    available_pipeline_skills,
    get_pipeline_skill_definition,
    pipeline_skill_definition_to_dict,
    pipeline_skill_result_to_dict,
    run_pipeline_skill,
)
from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class McpToolDefinition:
    name: str
    kind: str
    target_type: str
    target_name: str | None
    requires_config: bool
    description: str
    recommended_for: str
    input_schema: dict[str, Any]
    returns: str
    cli_command: str | None = None


MCP_TOOLS: dict[str, McpToolDefinition] = {
    'list_public_api_entrypoints': McpToolDefinition(
        name='list_public_api_entrypoints',
        kind='discovery',
        target_type='public-api-catalog',
        target_name=None,
        requires_config=False,
        description='List the recommended stable public API entrypoints that the library exposes.',
        recommended_for='Agent discovery before deciding whether to route through a public entrypoint, preset, or skill.',
        input_schema={'type': 'object', 'properties': {}},
        returns='list[PublicApiEntryDict]',
        cli_command='list-mcp-tools',
    ),
    'show_public_api_contract': McpToolDefinition(
        name='show_public_api_contract',
        kind='discovery',
        target_type='public-api-contract',
        target_name=None,
        requires_config=False,
        description='Return the grouped public API contract summary.',
        recommended_for='Agent reasoning about which stable public entrypoint best matches a user request.',
        input_schema={'type': 'object', 'properties': {}},
        returns='dict[layer, list[PublicApiEntryDict]]',
        cli_command='show-mcp-tool-catalog',
    ),
    'run_public_api_entrypoint': McpToolDefinition(
        name='run_public_api_entrypoint',
        kind='execution',
        target_type='public-api-dispatch',
        target_name='run_public_api_entrypoint',
        requires_config=False,
        description='Run one stable public API entrypoint through the unified dispatcher.',
        recommended_for='General automation when the agent already knows which stable public API entrypoint should execute.',
        input_schema={
            'type': 'object',
            'required': ['entrypoint'],
            'properties': {
                'entrypoint': {'type': 'string', 'enum': available_public_api_entrypoints()},
                'params': {'type': 'object', 'default': {}},
                'config': {'type': 'string'},
            },
        },
        returns='dict',
        cli_command='run-mcp-tool',
    ),
    'list_pipeline_skills': McpToolDefinition(
        name='list_pipeline_skills',
        kind='discovery',
        target_type='skill-catalog',
        target_name=None,
        requires_config=False,
        description='List the modular pipeline skills that give agents and users stable named tools.',
        recommended_for='Agent discovery of modular task names such as cp-extract-segmentation-artifacts, crop-export-single-cell-crops, or dp-run-deep-feature-model.',
        input_schema={'type': 'object', 'properties': {}},
        returns='list[PipelineSkillDefinitionDict]',
        cli_command='list-mcp-tools',
    ),
    'run_pipeline_skill': McpToolDefinition(
        name='run_pipeline_skill',
        kind='execution',
        target_type='skill',
        target_name='run_pipeline_skill',
        requires_config=True,
        description='Run one named modular skill.',
        recommended_for='Agent execution when the task aligns with an existing stable skill key.',
        input_schema={
            'type': 'object',
            'required': ['skill_key'],
            'properties': {
                'skill_key': {'type': 'string', 'enum': available_pipeline_skills()},
                'output_dir': {'type': 'string'},
                'workflow_root': {'type': 'string'},
                'export_root': {'type': 'string'},
                'project_root': {'type': 'string'},
                'image_csv_path': {'type': 'string'},
                'nuclei_csv_path': {'type': 'string'},
                'load_data_csv_path': {'type': 'string'},
                'manifest_path': {'type': 'string'},
                'object_table_path': {'type': 'string'},
                'single_cell_path': {'type': 'string'},
                'aggregated_path': {'type': 'string'},
                'annotated_path': {'type': 'string'},
                'normalized_path': {'type': 'string'},
                'feature_selected_path': {'type': 'string'},
                'single_cell_parquet_path': {'type': 'string'},
                'well_aggregated_parquet_path': {'type': 'string'},
                'object_table': {'type': 'string'},
                'crop_mode': {'type': 'string', 'enum': ['masked', 'unmasked']},
                'workers': {'type': 'integer'},
                'chunk_size': {'type': 'integer'},
                'gpu': {'type': 'string'},
                'experiment_name': {'type': 'string'},
                'config_filename': {'type': 'string'},
                'metadata_filename': {'type': 'string'},
                'overwrite': {'type': 'boolean'},
                'data_request': {'type': 'object'},
                'download_plan': {'type': 'string'},
            },
        },
        returns='PipelineSkillResultDict',
        cli_command='run-mcp-tool',
    ),
    'list_pipeline_presets': McpToolDefinition(
        name='list_pipeline_presets',
        kind='discovery',
        target_type='preset-catalog',
        target_name=None,
        requires_config=False,
        description='List the named presets that bundle common parameter combinations.',
        recommended_for='Agent discovery of reusable combined task shapes before applying task-level skill names.',
        input_schema={'type': 'object', 'properties': {}},
        returns='list[PipelinePresetDefinitionDict]',
        cli_command='list-mcp-tools',
    ),
    'run_pipeline_preset': McpToolDefinition(
        name='run_pipeline_preset',
        kind='execution',
        target_type='preset',
        target_name='run_pipeline_preset',
        requires_config=True,
        description='Run one named preset.',
        recommended_for='Agent execution when the task shape is known but the request does not need a higher-level skill alias.',
        input_schema={
            'type': 'object',
            'required': ['preset_key'],
            'properties': {
                'preset_key': {'type': 'string', 'enum': available_pipeline_presets()},
                'output_dir': {'type': 'string'},
                'profiling_suite': {'type': 'string'},
                'segmentation_suite': {'type': 'string'},
                'deepprofiler_mode': {'type': 'string'},
                'include_validation_report': {'type': 'boolean'},
                'include_data_access_summary': {'type': 'boolean'},
                'plan_data_download': {'type': 'boolean'},
                'execute_data_download_step': {'type': 'boolean'},
                'data_request': {'type': 'object'},
                'download_plan': {'type': 'string'},
            },
        },
        returns='EndToEndPipelineResultDict',
        cli_command='run-mcp-tool',
    ),
}


def available_mcp_tools() -> list[str]:
    return list(MCP_TOOLS)


def get_mcp_tool_definition(name: str) -> McpToolDefinition:
    if name not in MCP_TOOLS:
        available = ', '.join(available_mcp_tools())
        raise KeyError(f'Unknown MCP tool: {name}. Available: {available}')
    return MCP_TOOLS[name]


def mcp_tool_definition_to_dict(definition: McpToolDefinition) -> dict[str, Any]:
    return {
        'name': definition.name,
        'kind': definition.kind,
        'target_type': definition.target_type,
        'target_name': definition.target_name,
        'requires_config': definition.requires_config,
        'description': definition.description,
        'recommended_for': definition.recommended_for,
        'input_schema': definition.input_schema,
        'returns': definition.returns,
        'cli_command': definition.cli_command,
    }


def mcp_tool_catalog() -> list[dict[str, Any]]:
    return [
        mcp_tool_definition_to_dict(get_mcp_tool_definition(name))
        for name in available_mcp_tools()
    ]


def run_mcp_tool(
    name: str,
    *,
    config: ProjectConfig | None = None,
    **kwargs: Any,
) -> Any:
    definition = get_mcp_tool_definition(name)
    resolved_kwargs = _normalize_mcp_kwargs(kwargs)
    if definition.requires_config and config is None:
        raise ValueError(f'MCP tool {name} requires a ProjectConfig.')

    if name == 'list_public_api_entrypoints':
        return [
            public_api_entrypoint_to_dict(get_public_api_entrypoint(entrypoint))
            for entrypoint in available_public_api_entrypoints()
        ]
    if name == 'show_public_api_contract':
        return public_api_contract_summary()
    if name == 'run_public_api_entrypoint':
        entrypoint = resolved_kwargs.pop('entrypoint', None)
        params = resolved_kwargs.pop('params', {})
        if resolved_kwargs:
            unexpected = ', '.join(sorted(resolved_kwargs))
            raise ValueError(f'Unexpected arguments for run_public_api_entrypoint MCP tool: {unexpected}')
        if not entrypoint:
            raise ValueError('run_public_api_entrypoint MCP tool requires an entrypoint value.')
        if not isinstance(params, dict):
            raise ValueError('run_public_api_entrypoint MCP tool expects params to be a dictionary.')
        return run_public_api_entrypoint_to_dict(entrypoint, config=config, **params)
    if name == 'list_pipeline_skills':
        return [
            pipeline_skill_definition_to_dict(get_pipeline_skill_definition(skill_key))
            for skill_key in available_pipeline_skills()
        ]
    if name == 'run_pipeline_skill':
        skill_key = _pop_named_key(resolved_kwargs, 'skill_key', aliases=('skill',))
        if not skill_key:
            raise ValueError('run_pipeline_skill MCP tool requires a skill_key value.')
        return run_pipeline_skill(config, skill_key, **resolved_kwargs)
    if name == 'list_pipeline_presets':
        return [
            pipeline_preset_definition_to_dict(get_pipeline_preset_definition(preset_key))
            for preset_key in available_pipeline_presets()
        ]
    if name == 'run_pipeline_preset':
        preset_key = _pop_named_key(resolved_kwargs, 'preset_key', aliases=('preset',))
        if not preset_key:
            raise ValueError('run_pipeline_preset MCP tool requires a preset_key value.')
        return run_pipeline_preset(config, preset_key, **resolved_kwargs)
    raise ValueError(f'No MCP tool runner registered for: {name}')


def run_mcp_tool_to_dict(
    name: str,
    *,
    config: ProjectConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    definition = get_mcp_tool_definition(name)
    result = run_mcp_tool(name, config=config, **kwargs)
    return {
        'tool': name,
        'definition': mcp_tool_definition_to_dict(definition),
        'result': _mcp_tool_result_to_dict(name, result),
    }


def _normalize_mcp_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    # The pathlike keyword set is shared with the public API layer, so both
    # automation surfaces agree on what a filesystem path looks like.  This
    # used to be a separate, shorter list maintained by hand.
    resolved = dict(kwargs)
    for key in pathlike_keywords():
        value = resolved.get(key)
        if isinstance(value, str) and value.strip():
            resolved[key] = Path(value).expanduser().resolve()
    for key in REQUEST_KEYWORDS:
        value = resolved.get(key)
        if isinstance(value, dict):
            resolved[key] = build_data_request(**value)
    for key in PLAN_KEYWORDS:
        value = resolved.get(key)
        if isinstance(value, (str, Path)):
            resolved[key] = load_download_plan(Path(value).expanduser().resolve())
    return resolved


def _pop_named_key(resolved_kwargs: dict[str, Any], primary: str, *, aliases: tuple[str, ...] = ()) -> str | None:
    value = resolved_kwargs.pop(primary, None)
    if value is not None:
        return value
    for alias in aliases:
        if alias in resolved_kwargs:
            return resolved_kwargs.pop(alias)
    return None


def _mcp_tool_result_to_dict(name: str, result: Any) -> Any:
    if name in {
        'list_public_api_entrypoints',
        'show_public_api_contract',
        'run_public_api_entrypoint',
        'list_pipeline_skills',
        'list_pipeline_presets',
    }:
        return result
    if name == 'run_pipeline_skill':
        return pipeline_skill_result_to_dict(result)
    if name == 'run_pipeline_preset':
        return end_to_end_pipeline_result_to_dict(result)
    return result
