"""Deferred access to the implementation modules the CLI drives.

Importing a command module must not drag in pandas, CellProfiler or any
optional dependency, so every implementation symbol is bound to a small
proxy that resolves on first call.  The proxies live here so that the
command modules can import exactly the ones they use.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


class _LazyCallable:
    def __init__(self, module_name: str, attr_name: str) -> None:
        self.module_name = module_name
        self.attr_name = attr_name

    def _resolve(self):
        module = import_module(self.module_name)
        return getattr(module, self.attr_name)

    def __call__(self, *args: Any, **kwargs: Any):
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)


def _lazy(module_name: str, attr_name: str) -> _LazyCallable:
    return _LazyCallable(module_name, attr_name)


export_deepprofiler_input = _lazy('cellpaint_pipeline.adapters.deepprofiler', 'export_deepprofiler_input')
infer_deepprofiler_sources_from_workflow_root = _lazy('cellpaint_pipeline.adapters.deepprofiler', 'infer_deepprofiler_sources_from_workflow_root')
collect_deepprofiler_features = _lazy('cellpaint_pipeline.adapters.deepprofiler_features', 'collect_deepprofiler_features')
build_deepprofiler_project = _lazy('cellpaint_pipeline.adapters.deepprofiler_project', 'build_deepprofiler_project')
run_deepprofiler_profile = _lazy('cellpaint_pipeline.adapters.deepprofiler_project', 'run_deepprofiler_profile')
available_cppipe_templates = _lazy('cellpaint_pipeline.cppipe', 'available_cppipe_templates')
cppipe_template_definition_to_dict = _lazy('cellpaint_pipeline.cppipe', 'cppipe_template_definition_to_dict')
cppipe_validation_result_to_dict = _lazy('cellpaint_pipeline.cppipe', 'cppipe_validation_result_to_dict')
get_cppipe_template = _lazy('cellpaint_pipeline.cppipe', 'get_cppipe_template')
resolve_cppipe_selection = _lazy('cellpaint_pipeline.cppipe', 'resolve_cppipe_selection')
resolved_cppipe_selection_to_dict = _lazy('cellpaint_pipeline.cppipe', 'resolved_cppipe_selection_to_dict')
validate_cppipe_configuration = _lazy('cellpaint_pipeline.cppipe', 'validate_cppipe_configuration')
deepprofiler_pipeline_result_to_dict = _lazy('cellpaint_pipeline.deepprofiler_pipeline', 'deepprofiler_pipeline_result_to_dict')
run_deepprofiler_pipeline = _lazy('cellpaint_pipeline.deepprofiler_pipeline', 'run_deepprofiler_pipeline')
browse_quilt_package = _lazy('cellpaint_pipeline.data_access', 'browse_quilt_package')
build_data_access_status = _lazy('cellpaint_pipeline.data_access', 'build_data_access_status')
build_data_request = _lazy('cellpaint_pipeline.data_access', 'build_data_request')
build_download_plan = _lazy('cellpaint_pipeline.data_access', 'build_download_plan')
cache_gallery_listing = _lazy('cellpaint_pipeline.data_access', 'cache_gallery_listing')
cpgdata_prefix_list_result_to_dict = _lazy('cellpaint_pipeline.data_access', 'cpgdata_prefix_list_result_to_dict')
cpgdata_sync_result_to_dict = _lazy('cellpaint_pipeline.data_access', 'cpgdata_sync_result_to_dict')
data_access_status_to_dict = _lazy('cellpaint_pipeline.data_access', 'data_access_status_to_dict')
data_access_summary_to_dict = _lazy('cellpaint_pipeline.data_access', 'data_access_summary_to_dict')
data_download_execution_result_to_dict = _lazy('cellpaint_pipeline.data_access', 'data_download_execution_result_to_dict')
data_download_plan_to_dict = _lazy('cellpaint_pipeline.data_access', 'data_download_plan_to_dict')
download_gallery_prefix = _lazy('cellpaint_pipeline.data_access', 'download_gallery_prefix')
download_gallery_source = _lazy('cellpaint_pipeline.data_access', 'download_gallery_source')
gallery_cache_result_to_dict = _lazy('cellpaint_pipeline.data_access', 'gallery_cache_result_to_dict')
gallery_catalog_result_to_dict = _lazy('cellpaint_pipeline.data_access', 'gallery_catalog_result_to_dict')
gallery_download_result_to_dict = _lazy('cellpaint_pipeline.data_access', 'gallery_download_result_to_dict')
gallery_list_result_to_dict = _lazy('cellpaint_pipeline.data_access', 'gallery_list_result_to_dict')
list_cpgdata_prefixes = _lazy('cellpaint_pipeline.data_access', 'list_cpgdata_prefixes')
load_download_plan = _lazy('cellpaint_pipeline.data_access', 'load_download_plan')
list_gallery_datasets = _lazy('cellpaint_pipeline.data_access', 'list_gallery_datasets')
list_gallery_prefixes = _lazy('cellpaint_pipeline.data_access', 'list_gallery_prefixes')
list_gallery_sources = _lazy('cellpaint_pipeline.data_access', 'list_gallery_sources')
list_quilt_packages = _lazy('cellpaint_pipeline.data_access', 'list_quilt_packages')
quilt_package_browse_result_to_dict = _lazy('cellpaint_pipeline.data_access', 'quilt_package_browse_result_to_dict')
quilt_package_list_result_to_dict = _lazy('cellpaint_pipeline.data_access', 'quilt_package_list_result_to_dict')
summarize_data_access = _lazy('cellpaint_pipeline.data_access', 'summarize_data_access')
sync_cpgdata_index = _lazy('cellpaint_pipeline.data_access', 'sync_cpgdata_index')
sync_cpgdata_inventory = _lazy('cellpaint_pipeline.data_access', 'sync_cpgdata_inventory')
execute_download_plan = _lazy('cellpaint_pipeline.data_access', 'execute_download_plan')
write_download_plan = _lazy('cellpaint_pipeline.data_access', 'write_download_plan')
available_profiling_suites = _lazy('cellpaint_pipeline.delivery', 'available_profiling_suites')
available_segmentation_suites = _lazy('cellpaint_pipeline.delivery', 'available_segmentation_suites')
run_deepprofiler_full_stack = _lazy('cellpaint_pipeline.delivery', 'run_deepprofiler_full_stack')
run_full_pipeline = _lazy('cellpaint_pipeline.delivery', 'run_full_pipeline')
run_profiling_suite = _lazy('cellpaint_pipeline.delivery', 'run_profiling_suite')
run_segmentation_bundle = _lazy('cellpaint_pipeline.delivery', 'run_segmentation_bundle')
run_segmentation_suite = _lazy('cellpaint_pipeline.delivery', 'run_segmentation_suite')
run_smoke_test = _lazy('cellpaint_pipeline.delivery', 'run_smoke_test')
run_native_evaluation = _lazy('cellpaint_pipeline.evaluation', 'run_native_evaluation')
available_deepprofiler_modes = _lazy('cellpaint_pipeline.orchestration', 'available_deepprofiler_modes')
end_to_end_pipeline_result_to_dict = _lazy('cellpaint_pipeline.orchestration', 'end_to_end_pipeline_result_to_dict')
run_end_to_end_pipeline = _lazy('cellpaint_pipeline.orchestration', 'run_end_to_end_pipeline')
available_mcp_tools = _lazy('cellpaint_pipeline.mcp_tools', 'available_mcp_tools')
mcp_tool_catalog = _lazy('cellpaint_pipeline.mcp_tools', 'mcp_tool_catalog')
run_mcp_tool_to_dict = _lazy('cellpaint_pipeline.mcp_tools', 'run_mcp_tool_to_dict')
run_mcp_server = _lazy('cellpaint_pipeline.mcp_server', 'run_mcp_server')
available_public_api_entrypoints = _lazy('cellpaint_pipeline.public_api', 'available_public_api_entrypoints')
get_public_api_entrypoint = _lazy('cellpaint_pipeline.public_api', 'get_public_api_entrypoint')
public_api_contract_summary = _lazy('cellpaint_pipeline.public_api', 'public_api_contract_summary')
public_api_entrypoint_to_dict = _lazy('cellpaint_pipeline.public_api', 'public_api_entrypoint_to_dict')
run_public_api_entrypoint_to_dict = _lazy('cellpaint_pipeline.public_api', 'run_public_api_entrypoint_to_dict')
available_pipeline_presets = _lazy('cellpaint_pipeline.presets', 'available_pipeline_presets')
get_pipeline_preset_definition = _lazy('cellpaint_pipeline.presets', 'get_pipeline_preset_definition')
pipeline_preset_definition_to_dict = _lazy('cellpaint_pipeline.presets', 'pipeline_preset_definition_to_dict')
run_pipeline_preset = _lazy('cellpaint_pipeline.presets', 'run_pipeline_preset')
available_pipeline_skills = _lazy('cellpaint_pipeline.skills', 'available_pipeline_skills')
get_pipeline_skill_definition = _lazy('cellpaint_pipeline.skills', 'get_pipeline_skill_definition')
pipeline_skill_definition_to_dict = _lazy('cellpaint_pipeline.skills', 'pipeline_skill_definition_to_dict')
pipeline_skill_result_to_dict = _lazy('cellpaint_pipeline.skills', 'pipeline_skill_result_to_dict')
run_pipeline_skill = _lazy('cellpaint_pipeline.skills', 'run_pipeline_skill')
collect_validation_report = _lazy('cellpaint_pipeline.reporting', 'collect_validation_report')
segmentation_summary_to_dict = _lazy('cellpaint_pipeline.segmentation_native', 'segmentation_summary_to_dict')
summarize_segmentation_outputs = _lazy('cellpaint_pipeline.segmentation_native', 'summarize_segmentation_outputs')
write_segmentation_summary = _lazy('cellpaint_pipeline.segmentation_native', 'write_segmentation_summary')
available_workflows = _lazy('cellpaint_pipeline.workflows.orchestration', 'available_workflows')
run_workflow = _lazy('cellpaint_pipeline.workflows.orchestration', 'run_workflow')
available_profiling_scripts = _lazy('cellpaint_pipeline.workflows.profiling', 'available_profiling_scripts')
available_profiling_tasks = _lazy('cellpaint_pipeline.workflows.profiling', 'available_profiling_tasks')
run_profiling_native = _lazy('cellpaint_pipeline.workflows.profiling', 'run_profiling_native')
run_profiling_script = _lazy('cellpaint_pipeline.workflows.profiling', 'run_profiling_script')
run_profiling_task = _lazy('cellpaint_pipeline.workflows.profiling', 'run_profiling_task')
available_segmentation_scripts = _lazy('cellpaint_pipeline.workflows.segmentation', 'available_segmentation_scripts')
available_segmentation_tasks = _lazy('cellpaint_pipeline.workflows.segmentation', 'available_segmentation_tasks')
run_segmentation_native = _lazy('cellpaint_pipeline.workflows.segmentation', 'run_segmentation_native')
run_segmentation_script = _lazy('cellpaint_pipeline.workflows.segmentation', 'run_segmentation_script')
run_segmentation_task = _lazy('cellpaint_pipeline.workflows.segmentation', 'run_segmentation_task')


#: The package facade re-exports these names wholesale, so the list is
#: part of the observable surface.
__all__ = [
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
    '_LazyCallable',
    '_lazy',
]
