"""Focused public surface for CellPainting-Skills."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PipelineSkillDefinition",
    "PipelineSkillResult",
    "available_pipeline_skills",
    "get_pipeline_skill_definition",
    "pipeline_skill_definition_to_dict",
    "pipeline_skill_result_to_dict",
    "run_pipeline_skill",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module = import_module('cellpaint_pipeline.skills')
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
