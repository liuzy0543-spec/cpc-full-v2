"""Command-line interface for the CellPainting-Claw library.

The implementation is split by command domain under
:mod:`cellpaint_pipeline.cli.commands`; this module keeps the historical
import surface (``from cellpaint_pipeline.cli import main, build_parser``)
and re-exports the implementation proxies, which are part of that surface
because the test-suite patches them here.
"""
from __future__ import annotations

# cli.app does `from cellpaint_pipeline.cli import commands`, so importing it here while this module is still executing made cli and cli.app import each other. Upstream has no such cycle.  Resolving these three names on first access removes it and makes `import cellpaint_pipeline.cli` cheaper.
_LAZY_APP = ('COMMAND_HANDLERS', 'build_parser', 'main')


def __getattr__(name):  # PEP 562
    if name in _LAZY_APP:
        from cellpaint_pipeline.cli import app as _app
        return getattr(_app, name)
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY_APP))
# Re-exported because it was importable from the single-module cli and
# tests patch it here; anything resolving a config through the CLI
# facade would otherwise break.
from cellpaint_pipeline.cli.helpers import (
    _normalize_extra_args,
    _resolve_deepprofiler_source_kwargs,
    _maybe_resolve_path,
    _native_result_to_dict,
    _native_result_ok,
    _native_segmentation_result_to_dict,
)
from cellpaint_pipeline.cli.lazy import *  # noqa: F401,F403
from cellpaint_pipeline.cli.lazy import _LazyCallable, _lazy
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig

__all__ = [
    'ProjectConfig',
    'available_cppipe_templates',
    'available_deepprofiler_modes',
    'available_mcp_tools',
    'available_pipeline_presets',
    'available_pipeline_skills',
    'available_profiling_scripts',
    'available_profiling_suites',
    'available_profiling_tasks',
    'available_public_api_entrypoints',
    'available_segmentation_scripts',
    'available_segmentation_suites',
    'available_segmentation_tasks',
    'available_workflows',
    'browse_quilt_package',
    'build_data_access_status',
    'build_data_request',
    'build_deepprofiler_project',
    'build_download_plan',
    'cache_gallery_listing',
    'collect_deepprofiler_features',
    'collect_validation_report',
    'cpgdata_prefix_list_result_to_dict',
    'cpgdata_sync_result_to_dict',
    'cppipe_template_definition_to_dict',
    'cppipe_validation_result_to_dict',
    'data_access_status_to_dict',
    'data_access_summary_to_dict',
    'data_download_execution_result_to_dict',
    'data_download_plan_to_dict',
    'deepprofiler_pipeline_result_to_dict',
    'download_gallery_prefix',
    'download_gallery_source',
    'end_to_end_pipeline_result_to_dict',
    'execute_download_plan',
    'export_deepprofiler_input',
    'gallery_cache_result_to_dict',
    'gallery_catalog_result_to_dict',
    'gallery_download_result_to_dict',
    'gallery_list_result_to_dict',
    'get_cppipe_template',
    'get_pipeline_preset_definition',
    'get_pipeline_skill_definition',
    'get_public_api_entrypoint',
    'infer_deepprofiler_sources_from_workflow_root',
    'list_cpgdata_prefixes',
    'list_gallery_datasets',
    'list_gallery_prefixes',
    'list_gallery_sources',
    'list_quilt_packages',
    'load_download_plan',
    'mcp_tool_catalog',
    'pipeline_preset_definition_to_dict',
    'pipeline_skill_definition_to_dict',
    'pipeline_skill_result_to_dict',
    'public_api_contract_summary',
    'public_api_entrypoint_to_dict',
    'quilt_package_browse_result_to_dict',
    'quilt_package_list_result_to_dict',
    'resolve_cppipe_selection',
    'resolved_cppipe_selection_to_dict',
    'run_deepprofiler_full_stack',
    'run_deepprofiler_pipeline',
    'run_deepprofiler_profile',
    'run_end_to_end_pipeline',
    'run_full_pipeline',
    'run_mcp_server',
    'run_mcp_tool_to_dict',
    'run_native_evaluation',
    'run_pipeline_preset',
    'run_pipeline_skill',
    'run_profiling_native',
    'run_profiling_script',
    'run_profiling_suite',
    'run_profiling_task',
    'run_public_api_entrypoint_to_dict',
    'run_segmentation_bundle',
    'run_segmentation_native',
    'run_segmentation_script',
    'run_segmentation_suite',
    'run_segmentation_task',
    'run_smoke_test',
    'run_workflow',
    'segmentation_summary_to_dict',
    'summarize_data_access',
    'summarize_segmentation_outputs',
    'sync_cpgdata_index',
    'sync_cpgdata_inventory',
    'validate_cppipe_configuration',
    'write_download_plan',
    'write_segmentation_summary',
    'COMMAND_HANDLERS',
    '_LazyCallable',
    '_lazy',
    'build_parser',
    'main',
    '_normalize_extra_args',
    '_resolve_deepprofiler_source_kwargs',
    '_maybe_resolve_path',
    '_native_result_to_dict',
    '_native_result_ok',
    '_native_segmentation_result_to_dict',
]
