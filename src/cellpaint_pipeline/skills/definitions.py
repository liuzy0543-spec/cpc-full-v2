from __future__ import annotations

"""Immutable value objects exchanged across the skill layer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineSkillDefinition:
    key: str
    description: str
    category: str
    input_keys: tuple[str, ...]
    typical_outputs: tuple[str, ...]
    implements_with: tuple[str, ...]
    user_summary: str
    agent_summary: str
    composes_with: tuple[str, ...] = field(default_factory=tuple)
    status: str = 'primary'
    replaced_by: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PipelineSkillResult:
    skill_key: str
    category: str
    implementation: str
    output_dir: Path
    manifest_path: Path
    primary_outputs: dict[str, Path | None]
    details: dict[str, Any]
    ok: bool


def pipeline_skill_definition_to_dict(definition: PipelineSkillDefinition) -> dict[str, Any]:
    return {
        'key': definition.key,
        'description': definition.description,
        'category': definition.category,
        'input_keys': list(definition.input_keys),
        'typical_outputs': list(definition.typical_outputs),
        'implements_with': list(definition.implements_with),
        'user_summary': definition.user_summary,
        'agent_summary': definition.agent_summary,
        'composes_with': list(definition.composes_with),
        'status': definition.status,
        'replaced_by': list(definition.replaced_by),
    }


def pipeline_skill_result_to_dict(result: PipelineSkillResult) -> dict[str, Any]:
    return {
        'skill_key': result.skill_key,
        'category': result.category,
        'implementation': result.implementation,
        'output_dir': str(result.output_dir),
        'manifest_path': str(result.manifest_path),
        'primary_outputs': {key: str(value) if value is not None else None for key, value in result.primary_outputs.items()},
        'details': result.details,
        'ok': result.ok,
    }
