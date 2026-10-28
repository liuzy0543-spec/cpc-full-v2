"""The documented skill catalog and the single entry point that runs a skill.

LAYOUT
------
``definitions``  immutable value objects exchanged across the skill layer
``catalog``      the skill catalog itself (primary / advanced / legacy entries)
``inputs``       the parameter object plus the two accepted call shapes
``context``      the runtime context handed to a runner
``registry``     the skill-key -> runner table
``dispatch``     :func:`run_pipeline_skill`, the public entry point
``finalize``     shared manifest/result assembly
``runners``      the runners, grouped by pipeline stage

Everything that used to be importable from ``cellpaint_pipeline.skills`` is
still importable from here, so this module doubles as the compatibility
facade for the pre-split single-file layout.

WHY THE NAMES ARE RESOLVED LAZILY
---------------------------------
Re-exporting eagerly made every leaf load the whole package: importing any
submodule runs this file first, so importing skills.definitions - a module that
only defines two dataclasses - pulled the catalog, the registry and all four
runner modules; 134 modules for a type definition.  That is a layering defect
as much as a startup cost, because the leaves are supposed to be importable on
their own.

Resolving a name on first attribute access (PEP 562) keeps the public surface
and the patch('cellpaint_pipeline.skills.<name>') seam exactly as they were
while letting leaf modules stay leaves.  mock.patch reads the original through
getattr and writes it back with setattr, so a resolved name behaves like an
ordinary module global from then on.
"""
from __future__ import annotations

import importlib

#: Facade name -> the module that defines it.  One entry per name that used to
#: be re-exported here.
_LAZY_SOURCES: dict[str, str] = {}

_SKILLS = 'cellpaint_pipeline.skills.'


def _declare(module: str, *names: str) -> None:
    for name in names:
        _LAZY_SOURCES[name] = module


_declare(_SKILLS + 'catalog',
         'ADVANCED_PIPELINE_SKILLS', 'ALL_PIPELINE_SKILLS',
         'CURRENT_ADVANCED_PIPELINE_SKILLS', 'CURRENT_LEGACY_PIPELINE_SKILLS',
         'CURRENT_PRIMARY_PIPELINE_SKILLS', 'LEGACY_PIPELINE_SKILLS',
         'PRIMARY_PIPELINE_SKILLS', 'available_pipeline_skills',
         'get_pipeline_skill_definition')
_declare(_SKILLS + 'definitions',
         'PipelineSkillDefinition', 'PipelineSkillResult',
         'pipeline_skill_definition_to_dict', 'pipeline_skill_result_to_dict')
_declare(_SKILLS + 'inputs',
         'DEPRECATED_SKILL_PARAMETERS', 'SkillInputs', 'assemble_skill_inputs',
         'skill_inputs_from_mapping', 'skill_inputs_to_mapping')
_declare(_SKILLS + 'context', 'SkillRuntimeContext')
_declare(_SKILLS + 'registry', 'SKILL_RUNNERS')
# ExecutionResult is the value type run_pipeline_skill returns from the CLI
# layer; it was reachable here before the split and is a real API, not a leak.
_declare('cellpaint_pipeline.runner', 'ExecutionResult')
_declare(_SKILLS + 'dispatch', 'run_pipeline_skill')
_declare(_SKILLS + 'finalize',
         '_build_isolated_segmentation_skill_config', '_execution_result_to_dict',
         '_finalize_skill_result', '_json_ready',
         '_resolve_segmentation_source_config')

# The native implementations the runners call are reachable here for two
# reasons: they were importable from cellpaint_pipeline.skills before the
# split, and tests patch them at that location to isolate a runner.  Keeping
# the names available preserves both the surface and that patching seam.
_declare('cellpaint_pipeline.profile_summaries',
         'summarize_classical_profiles', 'summarize_deepprofiler_profiles')
_declare('cellpaint_pipeline.profiling_native',
         'export_cellprofiler_to_singlecell_native',
         'run_pycytominer_aggregate_native', 'run_pycytominer_annotate_native',
         'run_pycytominer_feature_select_native', 'run_pycytominer_native',
         'run_pycytominer_normalize_native')
_declare('cellpaint_pipeline.segmentation_native',
         'build_mask_export_pipeline_native', 'extract_single_cell_crops_native',
         'generate_sample_previews_native',
         'cellprofiler_executable', 'prepare_segmentation_load_data_native',
         'run_mask_export_native')
_declare('cellpaint_pipeline.workflows.profiling', 'run_profiling_script')
_declare('cellpaint_pipeline.workflows.segmentation', 'run_segmentation_script')
_declare(_SKILLS + 'runners.data_access',
         '_resolve_download_request_and_plan', '_run_data_plan_download',
         '_run_download_cellpainting_data', '_run_inspect_cellpainting_data')
_declare(_SKILLS + 'runners.deepprofiler',
         '_run_build_deepprofiler_project', '_run_collect_deepprofiler_features',
         '_run_deepprofiler', '_run_deepprofiler_profile',
         '_run_export_deepprofiler_inputs', '_run_prepare_deepprofiler_project',
         '_run_summarize_deepprofiler_profiles')
_declare(_SKILLS + 'runners.profiling',
         '_run_cellprofiler_profiling', '_run_cyto_aggregate_profiles',
         '_run_cyto_annotate_profiles', '_run_cyto_normalize_profiles',
         '_run_cyto_select_profile_features', '_run_export_single_cell_measurements',
         '_run_pycytominer', '_run_pycytominer_stage',
         '_run_summarize_classical_profiles')
_declare(_SKILLS + 'runners.segmentation',
         '_run_export_masked_single_cell_crops', '_run_export_single_cell_crops',
         '_run_export_unmasked_single_cell_crops',
         '_run_extract_segmentation_artifacts', '_run_generate_sample_previews',
         '_run_prepare_segmentation_inputs', '_run_segmentation_masks',
         '_run_single_cell_crop_skill')

__all__ = [
    'ADVANCED_PIPELINE_SKILLS',
    'ALL_PIPELINE_SKILLS',
    'CURRENT_ADVANCED_PIPELINE_SKILLS',
    'CURRENT_LEGACY_PIPELINE_SKILLS',
    'CURRENT_PRIMARY_PIPELINE_SKILLS',
    'DEPRECATED_SKILL_PARAMETERS',
    'ExecutionResult',
    'LEGACY_PIPELINE_SKILLS',
    'PRIMARY_PIPELINE_SKILLS',
    'PipelineSkillDefinition',
    'PipelineSkillResult',
    'SKILL_RUNNERS',
    'SkillInputs',
    'SkillRuntimeContext',
    'assemble_skill_inputs',
    'available_pipeline_skills',
    'get_pipeline_skill_definition',
    'pipeline_skill_definition_to_dict',
    'pipeline_skill_result_to_dict',
    'run_pipeline_skill',
    'skill_inputs_from_mapping',
    'skill_inputs_to_mapping',
]


def __getattr__(name: str):
    """Resolve a facade name on first use, then cache it as a module global."""
    module_name = _LAZY_SOURCES.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SOURCES))
