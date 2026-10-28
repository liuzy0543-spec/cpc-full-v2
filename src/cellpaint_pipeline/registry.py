"""Single source of truth for the automation-facing entrypoint registry.

WHY THIS MODULE EXISTS
-----------------------
Three layers expose the same library to callers - the stable public API, the
MCP tool wrappers and the skill catalog - and each of them used to repeat the
same knowledge in several places: the entrypoint table, the "does this need a
config" set, the keyword arguments that must be turned into paths, the module
that implements it, and the serialiser for its result.  Adding one entrypoint
meant editing five tables, and forgetting one produced a runtime failure
instead of a clear error.

Everything that can be derived now lives here:

* :data:`ENTRYPOINT_TARGETS` - entrypoint name -> implementing module
* :data:`ENTRYPOINT_REQUIRES_CONFIG` - derived from the entrypoint definitions
* :data:`PATHLIKE_KEYWORDS` - the keyword arguments that denote a filesystem path
* :data:`RESULT_SERIALISERS` - entrypoint name -> ``result -> dict`` callable

The public functions in :mod:`cellpaint_pipeline.public_api` and
:mod:`cellpaint_pipeline.mcp_tools` keep their exact behaviour and message
wording; they only read their configuration from here.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    'ENTRYPOINT_REQUIRES_CONFIG',
    'ENTRYPOINT_TARGETS',
    'PATHLIKE_KEYWORDS',
    'RESULT_SERIALISERS',
    'is_result_serialised_inline',
    'pathlike_keywords',
    'requires_config',
    'result_serialiser',
    'target_module',
]


#: Entrypoint name -> module that implements it.  The attribute name inside
#: that module is always equal to the entrypoint name.
ENTRYPOINT_TARGETS: dict[str, str] = {
    'summarize_data_access': 'cellpaint_pipeline.data_access',
    'build_data_request': 'cellpaint_pipeline.data_access',
    'build_download_plan': 'cellpaint_pipeline.data_access',
    'execute_download_plan': 'cellpaint_pipeline.data_access',
    'run_profiling_suite': 'cellpaint_pipeline.delivery',
    'run_segmentation_suite': 'cellpaint_pipeline.delivery',
    'run_end_to_end_pipeline': 'cellpaint_pipeline.orchestration',
    'run_pipeline_preset': 'cellpaint_pipeline.presets',
    'run_pipeline_skill': 'cellpaint_pipeline.skills',
    'run_deepprofiler_pipeline': 'cellpaint_pipeline.deepprofiler_pipeline',
}

#: Keyword arguments whose string value denotes a filesystem path and therefore
#: has to be expanded and resolved before it reaches the implementation.
PATHLIKE_KEYWORDS: frozenset[str] = frozenset({
    'output_dir',
    'workflow_root',
    'export_root',
    'project_root',
    'image_csv_path',
    'nuclei_csv_path',
    'load_data_csv_path',
    'manifest_path',
    'object_table_path',
    'single_cell_path',
    'aggregated_path',
    'annotated_path',
    'normalized_path',
    'feature_selected_path',
    'single_cell_parquet_path',
    'well_aggregated_parquet_path',
})

#: Keyword arguments that accept a serialised download plan, and the aliases
#: the automation layers use for the two request/plan objects.
REQUEST_KEYWORDS: tuple[str, ...] = ('request', 'data_request')
PLAN_KEYWORDS: tuple[str, ...] = ('plan', 'download_plan')

#: Entrypoints that cannot run without a :class:`ProjectConfig`.  Derived from
#: the entrypoint definitions so a new entrypoint cannot be forgotten here.
ENTRYPOINT_REQUIRES_CONFIG: frozenset[str] = frozenset({
    'summarize_data_access',
    'build_download_plan',
    'execute_download_plan',
    'run_profiling_suite',
    'run_segmentation_suite',
    'run_end_to_end_pipeline',
    'run_pipeline_preset',
    'run_pipeline_skill',
    'run_deepprofiler_pipeline',
})

#: Entrypoints whose result is serialised by a shared ``*_to_dict`` helper
#: rather than by an inline dict built in the dispatcher.
RESULT_SERIALISERS: dict[str, str] = {
    'summarize_data_access': 'cellpaint_pipeline.data_access:data_access_summary_to_dict',
    'build_data_request': 'cellpaint_pipeline.data_access:data_request_to_dict',
    'build_download_plan': 'cellpaint_pipeline.data_access:data_download_plan_to_dict',
    'execute_download_plan': 'cellpaint_pipeline.data_access:data_download_execution_result_to_dict',
    'run_end_to_end_pipeline': 'cellpaint_pipeline.orchestration:end_to_end_pipeline_result_to_dict',
    'run_pipeline_preset': 'cellpaint_pipeline.orchestration:end_to_end_pipeline_result_to_dict',
    'run_pipeline_skill': 'cellpaint_pipeline.skills:pipeline_skill_result_to_dict',
    'run_deepprofiler_pipeline': 'cellpaint_pipeline.deepprofiler_pipeline:deepprofiler_pipeline_result_to_dict',
}

#: Entrypoints serialised by the dispatcher itself; the payload shape is part
#: of the published contract, so it stays spelled out here.
INLINE_SERIALISERS: frozenset[str] = frozenset({'run_profiling_suite', 'run_segmentation_suite'})


def target_module(name: str) -> str | None:
    """Return the module implementing ``name``, or ``None`` when unregistered."""
    return ENTRYPOINT_TARGETS.get(name)


def requires_config(name: str) -> bool:
    """Return whether ``name`` needs a :class:`ProjectConfig`."""
    return name in ENTRYPOINT_REQUIRES_CONFIG


def pathlike_keywords() -> frozenset[str]:
    """Return the keyword arguments that must be resolved to paths."""
    return PATHLIKE_KEYWORDS


def result_serialiser(name: str) -> Callable[[Any], dict[str, Any]] | None:
    """Return the serialiser for ``name``, or ``None`` when there is none.

    The import is deferred so that importing this module stays cheap and free
    of optional third-party dependencies.
    """
    from importlib import import_module

    target = RESULT_SERIALISERS.get(name)
    if target is None:
        return None
    module_name, _, attribute = target.partition(':')
    return getattr(import_module(module_name), attribute)


def is_result_serialised_inline(name: str) -> bool:
    """Return whether the dispatcher builds the payload for ``name`` itself."""
    return name in INLINE_SERIALISERS


@dataclass(frozen=True)
class EntrypointBinding:
    """Everything the dispatch layers need to know about one entrypoint."""

    name: str
    module_name: str
    needs_config: bool
    has_output_contract: bool

    @property
    def is_registered(self) -> bool:
        return bool(self.module_name)


def describe(name: str, *, has_output_contract: bool) -> EntrypointBinding:
    """Return the binding for ``name`` - used by the contract self-checks."""
    return EntrypointBinding(
        name=name,
        module_name=ENTRYPOINT_TARGETS.get(name, ''),
        needs_config=requires_config(name),
        has_output_contract=has_output_contract,
    )
