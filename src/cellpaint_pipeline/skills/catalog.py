from __future__ import annotations

"""The skill catalog: every documented, advanced and legacy skill entry."""

from dataclasses import replace
from typing import Any

from cellpaint_pipeline.skills.definitions import (
    PipelineSkillDefinition,
    pipeline_skill_definition_to_dict,
)
from cellpaint_pipeline.skills.outputs import (
    DEEPPROFILER_COLLECT_OUTPUTS,
    SEGMENTATION_ARTIFACT_OUTPUTS,
    SEGMENTATION_MASK_OUTPUTS,
    SINGLE_CELL_CROP_OUTPUTS,
    advertised_outputs,
)

__all__ = [
    'ADVANCED_PIPELINE_SKILLS',
    'ALL_PIPELINE_SKILLS',
    'CURRENT_ADVANCED_PIPELINE_SKILLS',
    'CURRENT_LEGACY_PIPELINE_SKILLS',
    'CURRENT_PRIMARY_PIPELINE_SKILLS',
    'LEGACY_PIPELINE_SKILLS',
    'PRIMARY_PIPELINE_SKILLS',
    'available_pipeline_skills',
    'get_pipeline_skill_definition',
]


CURRENT_PRIMARY_PIPELINE_SKILLS: dict[str, PipelineSkillDefinition] = {
    'inspect-cellpainting-data': PipelineSkillDefinition(
        key='inspect-cellpainting-data',
        description='Inspect the configured Cell Painting data sources and return a usable availability summary.',
        category='data-access',
        input_keys=('output_dir',),
        typical_outputs=('data_access_summary.json',),
        implements_with=('cellpaint_pipeline.data_access', 'boto3', 'cpgdata', 'quilt3'),
        user_summary='Use this skill when you want to see what Cell Painting data can be accessed before downloading anything.',
        agent_summary='Choose this skill when the request is to inspect accessible datasets, sources, or access status.',
        composes_with=('download-cellpainting-data',),
    ),
    'download-cellpainting-data': PipelineSkillDefinition(
        key='download-cellpainting-data',
        description='Download one configured Cell Painting dataset slice into a local cache directory.',
        category='data-access',
        input_keys=('data_request', 'download_plan', 'output_dir'),
        typical_outputs=('download_plan.json', 'downloads/', 'download manifests'),
        implements_with=('cellpaint_pipeline.data_access', 'boto3', 'cpgdata', 'quilt3'),
        user_summary='Use this skill when you want local input files for the rest of the toolkit.',
        agent_summary='Choose this skill when the request is to fetch Cell Painting inputs from the configured access layer.',
        composes_with=('run-cellprofiler-profiling', 'run-segmentation-masks'),
    ),
    'run-cellprofiler-profiling': PipelineSkillDefinition(
        key='run-cellprofiler-profiling',
        description='Run the configured CellProfiler profiling pipeline against the active profiling backend.',
        category='profiling',
        input_keys=('output_dir',),
        typical_outputs=('Image.csv', 'Cells.csv', 'Cytoplasm.csv', 'Nuclei.csv', 'CellProfiler log'),
        implements_with=('cellpaint_pipeline.workflows.profiling', 'CellProfiler'),
        user_summary='Use this skill when you want CellProfiler to generate profiling tables from raw images.',
        agent_summary='Choose this skill when the request is to run the configured profiling .cppipe and produce measurement tables.',
        composes_with=('export-single-cell-measurements', 'run-pycytominer'),
    ),
    'export-single-cell-measurements': PipelineSkillDefinition(
        key='export-single-cell-measurements',
        description='Merge CellProfiler image and object tables into a single-cell measurements table.',
        category='profiling',
        input_keys=('image_csv_path', 'object_table_path', 'object_table', 'output_dir'),
        typical_outputs=('single_cell.csv.gz',),
        implements_with=('cellpaint_pipeline.profiling_native',),
        user_summary='Use this skill when you want one single-cell measurements table after CellProfiler has produced the compartment tables.',
        agent_summary='Choose this skill when the request is to turn CellProfiler output tables into one single-cell table.',
        composes_with=('run-pycytominer',),
    ),
    'run-pycytominer': PipelineSkillDefinition(
        key='run-pycytominer',
        description='Run the configured pycytominer aggregation, annotation, normalization, and feature-selection path.',
        category='profiling',
        input_keys=('output_dir',),
        typical_outputs=('aggregated.parquet', 'annotated.parquet', 'normalized.parquet', 'feature_selected.parquet'),
        implements_with=('cellpaint_pipeline.profiling_native', 'pycytominer'),
        user_summary='Use this skill when you want classical Cell Painting profile tables.',
        agent_summary='Choose this skill when the request is about classical profile generation from single-cell measurements.',
        composes_with=('summarize-classical-profiles',),
    ),
    'summarize-classical-profiles': PipelineSkillDefinition(
        key='summarize-classical-profiles',
        description='Summarize classical Cell Painting profile tables into readable metadata, variability, and PCA outputs.',
        category='profiling',
        input_keys=('feature_selected_path', 'manifest_path', 'output_dir'),
        typical_outputs=('profile_summary.json', 'well_metadata_summary.csv', 'top_variable_features.csv', 'pca_coordinates.csv', 'pca_plot.png'),
        implements_with=('cellpaint_pipeline.profile_summaries', 'pandas', 'numpy'),
        user_summary='Use this skill when you want a readable summary of pycytominer outputs instead of raw profile tables only.',
        agent_summary='Choose this skill when the request is to explain or summarize classical profile outputs for a human reader.',
    ),
    'run-segmentation-masks': PipelineSkillDefinition(
        key='run-segmentation-masks',
        description='Run the segmentation CellProfiler pipeline and produce mask tables, labels, outlines, and sample previews.',
        category='segmentation',
        input_keys=('output_dir',),
        typical_outputs=advertised_outputs(SEGMENTATION_MASK_OUTPUTS),
        implements_with=('cellpaint_pipeline.segmentation_native', 'cellpaint_pipeline.workflows.segmentation', 'CellProfiler'),
        user_summary='Use this skill when you want segmentation outputs, including masks, object tables, and quick field previews.',
        agent_summary='Choose this skill when the request is to execute the segmentation branch and produce mask artifacts plus quick previews.',
        composes_with=('export-single-cell-crops', 'prepare-deepprofiler-project'),
    ),
    'export-single-cell-crops': PipelineSkillDefinition(
        key='export-single-cell-crops',
        description='Export single-cell image stacks from a segmentation workflow root in masked or unmasked mode.',
        category='segmentation',
        input_keys=('workflow_root', 'output_dir', 'workers', 'crop_mode'),
        typical_outputs=advertised_outputs(SINGLE_CELL_CROP_OUTPUTS),
        implements_with=('cellpaint_pipeline.segmentation_native',),
        user_summary='Use this skill when you want single-cell crops as a user-facing result and choose masked or unmasked mode explicitly.',
        agent_summary='Choose this skill when the request is to export single-cell crops and the only decision is masked versus unmasked mode.',
        composes_with=('prepare-deepprofiler-project',),
    ),
    'prepare-deepprofiler-project': PipelineSkillDefinition(
        key='prepare-deepprofiler-project',
        description='Prepare a runnable DeepProfiler project from a segmentation workflow root, explicit source tables, or an existing export root.',
        category='deepprofiler',
        input_keys=('workflow_root', 'export_root', 'image_csv_path', 'nuclei_csv_path', 'load_data_csv_path', 'output_dir', 'experiment_name', 'config_filename', 'metadata_filename'),
        typical_outputs=('project_manifest.json', 'inputs/config/', 'inputs/metadata/', 'inputs/locations/'),
        implements_with=('cellpaint_pipeline.adapters.deepprofiler', 'cellpaint_pipeline.adapters.deepprofiler_project', 'DeepProfiler'),
        user_summary='Use this skill when you want a DeepProfiler project directory that is ready to run next.',
        agent_summary='Choose this skill when the request is to stop at a runnable DeepProfiler project rather than executing the model.',
        composes_with=('run-deepprofiler',),
    ),
    'run-deepprofiler': PipelineSkillDefinition(
        key='run-deepprofiler',
        description='Run the DeepProfiler path and return collected single-cell and well-level feature tables.',
        category='deepprofiler',
        input_keys=('project_root', 'workflow_root', 'image_csv_path', 'nuclei_csv_path', 'load_data_csv_path', 'output_dir', 'experiment_name', 'config_filename', 'metadata_filename', 'gpu'),
        typical_outputs=('deepprofiler_single_cell.parquet', 'deepprofiler_well_aggregated.parquet', 'deepprofiler_feature_manifest.json'),
        implements_with=('cellpaint_pipeline.deepprofiler_pipeline', 'DeepProfiler', 'pandas', 'pyarrow'),
        user_summary='Use this skill when you want final DeepProfiler tables rather than only intermediate project or feature directories.',
        agent_summary='Choose this skill when the request is to run DeepProfiler and hand back analysis-ready outputs.',
        composes_with=('summarize-deepprofiler-profiles',),
    ),
    'summarize-deepprofiler-profiles': PipelineSkillDefinition(
        key='summarize-deepprofiler-profiles',
        description='Summarize DeepProfiler single-cell and well-level tables into readable metadata, variability, and PCA outputs.',
        category='deepprofiler',
        input_keys=('single_cell_parquet_path', 'well_aggregated_parquet_path', 'manifest_path', 'output_dir'),
        typical_outputs=('profile_summary.json', 'well_metadata_summary.csv', 'top_variable_features.csv', 'pca_coordinates.csv', 'pca_plot.png'),
        implements_with=('cellpaint_pipeline.profile_summaries', 'pandas', 'numpy'),
        user_summary='Use this skill when you want a readable summary of DeepProfiler outputs instead of raw embedding tables only.',
        agent_summary='Choose this skill when the request is to explain or summarize DeepProfiler outputs for a human reader.',
    ),
}

CURRENT_ADVANCED_PIPELINE_SKILLS: dict[str, PipelineSkillDefinition] = {
    'generate-sample-previews': PipelineSkillDefinition(
        key='generate-sample-previews',
        description='Render field-level RGB preview PNGs from the segmentation source channels.',
        category='segmentation',
        input_keys=('workflow_root', 'output_dir', 'overwrite'),
        typical_outputs=('sample_previews_png/',),
        implements_with=('cellpaint_pipeline.segmentation_native', 'Pillow', 'numpy'),
        user_summary='Direct-control skill for users who want only preview PNGs from a segmentation workflow root.',
        agent_summary='Direct-control skill for explicitly producing preview PNGs without rerunning the full segmentation skill.',
        status='advanced',
    ),
    'export-masked-single-cell-crops': PipelineSkillDefinition(
        key='export-masked-single-cell-crops',
        description='Export masked single-cell image stacks and masks for each segmented cell.',
        category='segmentation',
        input_keys=('workflow_root', 'output_dir', 'workers'),
        typical_outputs=('masked/image_stacks/', 'masked/cell_masks/', 'masked/nuclei_masks/', 'masked/single_cell_manifest.csv'),
        implements_with=('cellpaint_pipeline.segmentation_native',),
        user_summary='Advanced direct-control skill for explicitly requesting masked crop export.',
        agent_summary='Advanced direct-control skill for explicitly requesting masked crop export.',
        composes_with=('export-deepprofiler-inputs',),
        status='advanced',
    ),
    'export-unmasked-single-cell-crops': PipelineSkillDefinition(
        key='export-unmasked-single-cell-crops',
        description='Export unmasked single-cell image stacks and masks for each segmented cell.',
        category='segmentation',
        input_keys=('workflow_root', 'output_dir', 'workers'),
        typical_outputs=('unmasked/image_stacks/', 'unmasked/cell_masks/', 'unmasked/nuclei_masks/', 'unmasked/single_cell_manifest.csv'),
        implements_with=('cellpaint_pipeline.segmentation_native',),
        user_summary='Advanced direct-control skill for explicitly requesting unmasked crop export.',
        agent_summary='Advanced direct-control skill for explicitly requesting unmasked crop export.',
        composes_with=('export-deepprofiler-inputs',),
        status='advanced',
    ),
    'export-deepprofiler-inputs': PipelineSkillDefinition(
        key='export-deepprofiler-inputs',
        description='Write the DeepProfiler field metadata and per-field nuclei location CSV files.',
        category='deepprofiler',
        input_keys=('workflow_root', 'image_csv_path', 'nuclei_csv_path', 'load_data_csv_path', 'output_dir'),
        typical_outputs=('manifest.json', 'images/field_metadata.csv', 'locations/'),
        implements_with=('cellpaint_pipeline.adapters.deepprofiler',),
        user_summary='Advanced direct-control skill for stopping at the DeepProfiler export stage.',
        agent_summary='Advanced direct-control skill for stopping at the DeepProfiler export stage.',
        composes_with=('build-deepprofiler-project',),
        status='advanced',
    ),
    'build-deepprofiler-project': PipelineSkillDefinition(
        key='build-deepprofiler-project',
        description='Build a runnable DeepProfiler project directory from a DeepProfiler export.',
        category='deepprofiler',
        input_keys=('workflow_root', 'export_root', 'output_dir', 'experiment_name', 'config_filename', 'metadata_filename'),
        typical_outputs=('project_manifest.json', 'inputs/config/', 'inputs/metadata/', 'inputs/locations/'),
        implements_with=('cellpaint_pipeline.adapters.deepprofiler_project', 'DeepProfiler'),
        user_summary='Advanced direct-control skill for explicitly building the DeepProfiler project stage.',
        agent_summary='Advanced direct-control skill for explicitly building the DeepProfiler project stage.',
        composes_with=('run-deepprofiler-profile',),
        status='advanced',
    ),
    'run-deepprofiler-profile': PipelineSkillDefinition(
        key='run-deepprofiler-profile',
        description='Run the DeepProfiler profile command against a prepared DeepProfiler project.',
        category='deepprofiler',
        input_keys=('project_root', 'output_dir', 'experiment_name', 'config_filename', 'metadata_filename', 'gpu'),
        typical_outputs=('outputs/<experiment>/features/', 'deepprofiler_profile log'),
        implements_with=('cellpaint_pipeline.adapters.deepprofiler_project', 'DeepProfiler'),
        user_summary='Advanced direct-control skill for generating raw DeepProfiler feature files.',
        agent_summary='Advanced direct-control skill for generating raw DeepProfiler feature files.',
        composes_with=('collect-deepprofiler-features',),
        status='advanced',
    ),
    'collect-deepprofiler-features': PipelineSkillDefinition(
        key='collect-deepprofiler-features',
        description='Collect DeepProfiler .npz outputs into single-cell and well-level tabular feature files.',
        category='deepprofiler',
        input_keys=('project_root', 'output_dir', 'experiment_name'),
        typical_outputs=advertised_outputs(DEEPPROFILER_COLLECT_OUTPUTS),
        implements_with=('cellpaint_pipeline.adapters.deepprofiler_features', 'pandas', 'pyarrow'),
        user_summary='Advanced direct-control skill for collecting already-generated DeepProfiler feature files into tables.',
        agent_summary='Advanced direct-control skill for collecting already-generated DeepProfiler feature files into tables.',
        status='advanced',
    ),
}

CURRENT_LEGACY_PIPELINE_SKILLS: dict[str, PipelineSkillDefinition] = {
    'build-image-manifest': PipelineSkillDefinition(
        key='build-image-manifest',
        description='Former helper-style skill name retained for compatibility discovery only.',
        category='legacy',
        input_keys=(),
        typical_outputs=(),
        implements_with=(),
        user_summary='Legacy helper alias. Prefer run-cellprofiler-profiling.',
        agent_summary='Legacy helper alias. Prefer run-cellprofiler-profiling.',
        status='legacy',
        replaced_by=('run-cellprofiler-profiling',),
    ),
    'validate-profiling-inputs': PipelineSkillDefinition(
        key='validate-profiling-inputs',
        description='Former helper-style skill name retained for compatibility discovery only.',
        category='legacy',
        input_keys=(),
        typical_outputs=(),
        implements_with=(),
        user_summary='Legacy helper alias. Prefer run-cellprofiler-profiling or export-single-cell-measurements depending on the actual task.',
        agent_summary='Legacy helper alias. Prefer run-cellprofiler-profiling or export-single-cell-measurements depending on the actual task.',
        status='legacy',
        replaced_by=('run-cellprofiler-profiling', 'export-single-cell-measurements'),
    ),
    'export-segmentation-load-data': PipelineSkillDefinition(
        key='export-segmentation-load-data',
        description='Former helper-style skill name retained for compatibility discovery only.',
        category='legacy',
        input_keys=(),
        typical_outputs=(),
        implements_with=(),
        user_summary='Legacy helper alias. Prefer run-segmentation-masks.',
        agent_summary='Legacy helper alias. Prefer run-segmentation-masks.',
        status='legacy',
        replaced_by=('run-segmentation-masks',),
    ),
    'plan-data-access': PipelineSkillDefinition(
        key='plan-data-access',
        description='Legacy skill name retained for compatibility discovery only.',
        category='legacy',
        input_keys=(),
        typical_outputs=(),
        implements_with=(),
        user_summary='Legacy alias. Prefer inspect-cellpainting-data or download-cellpainting-data.',
        agent_summary='Legacy alias. Prefer inspect-cellpainting-data or download-cellpainting-data.',
        status='legacy',
        replaced_by=('inspect-cellpainting-data', 'download-cellpainting-data'),
    ),
    'download-data': PipelineSkillDefinition(
        key='download-data',
        description='Legacy skill name retained for compatibility discovery only.',
        category='legacy',
        input_keys=(),
        typical_outputs=(),
        implements_with=(),
        user_summary='Legacy alias. Prefer download-cellpainting-data.',
        agent_summary='Legacy alias. Prefer download-cellpainting-data.',
        status='legacy',
        replaced_by=('download-cellpainting-data',),
    ),
    'run-classical-profiling': PipelineSkillDefinition(
        key='run-classical-profiling',
        description='Legacy skill name retained for compatibility discovery only.',
        category='legacy',
        input_keys=(),
        typical_outputs=(),
        implements_with=(),
        user_summary='Legacy alias. Prefer export-single-cell-measurements plus run-pycytominer.',
        agent_summary='Legacy alias. Prefer export-single-cell-measurements plus run-pycytominer.',
        status='legacy',
        replaced_by=('export-single-cell-measurements', 'run-pycytominer'),
    ),
    'run-segmentation': PipelineSkillDefinition(
        key='run-segmentation',
        description='Legacy skill name retained for compatibility discovery only.',
        category='legacy',
        input_keys=(),
        typical_outputs=(),
        implements_with=(),
        user_summary='Legacy alias. Prefer run-segmentation-masks.',
        agent_summary='Legacy alias. Prefer run-segmentation-masks.',
        status='legacy',
        replaced_by=('run-segmentation-masks',),
    ),
    'prepare-deepprofiler-inputs': PipelineSkillDefinition(
        key='prepare-deepprofiler-inputs',
        description='Legacy skill name retained for compatibility discovery only.',
        category='legacy',
        input_keys=(),
        typical_outputs=(),
        implements_with=(),
        user_summary='Legacy alias. Prefer prepare-deepprofiler-project or export-deepprofiler-inputs depending on how far you want to go.',
        agent_summary='Legacy alias. Prefer prepare-deepprofiler-project or export-deepprofiler-inputs depending on how far you want to go.',
        status='legacy',
        replaced_by=('prepare-deepprofiler-project', 'export-deepprofiler-inputs'),
    ),
}

def _remap_skill_definition(
    definition: PipelineSkillDefinition,
    *,
    key: str | None = None,
    description: str | None = None,
    category: str | None = None,
    input_keys: tuple[str, ...] | None = None,
    typical_outputs: tuple[str, ...] | None = None,
    implements_with: tuple[str, ...] | None = None,
    user_summary: str | None = None,
    agent_summary: str | None = None,
    composes_with: tuple[str, ...] | None = None,
    status: str | None = None,
    replaced_by: tuple[str, ...] | None = None,
) -> PipelineSkillDefinition:
    return replace(
        definition,
        key=key or definition.key,
        description=description or definition.description,
        category=category or definition.category,
        input_keys=input_keys if input_keys is not None else definition.input_keys,
        typical_outputs=typical_outputs if typical_outputs is not None else definition.typical_outputs,
        implements_with=implements_with if implements_with is not None else definition.implements_with,
        user_summary=user_summary or definition.user_summary,
        agent_summary=agent_summary or definition.agent_summary,
        composes_with=composes_with if composes_with is not None else definition.composes_with,
        status=status or definition.status,
        replaced_by=replaced_by if replaced_by is not None else definition.replaced_by,
    )


PRIMARY_PIPELINE_SKILLS: dict[str, PipelineSkillDefinition] = {
    'data-inspect-availability': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['inspect-cellpainting-data'],
        key='data-inspect-availability',
        description='Inspect the configured Cell Painting data sources and return a usable availability summary.',
        user_summary='Use this skill when you want to inspect configured Cell Painting data sources before planning or downloading anything.',
        agent_summary='Choose this skill when the request is to inspect available Cell Painting sources, dataset identifiers, or access status.',
        composes_with=('data-plan-download', 'data-download'),
    ),
    'data-plan-download': PipelineSkillDefinition(
        key='data-plan-download',
        description='Resolve one configured Cell Painting data request into a concrete download plan without executing the download.',
        category='data-access',
        input_keys=('data_request', 'output_dir'),
        typical_outputs=('download_plan.json',),
        implements_with=('cellpaint_pipeline.data_access', 'boto3', 'cpgdata', 'quilt3'),
        user_summary='Use this skill when you want a concrete download plan before fetching files.',
        agent_summary='Choose this skill when the request is to preview or validate what a download would fetch without executing it.',
        composes_with=('data-download',),
    ),
    'data-download': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['download-cellpainting-data'],
        key='data-download',
        description='Download one configured Cell Painting dataset slice into a local cache directory.',
        user_summary='Use this skill when you want local input files for the rest of the toolkit.',
        agent_summary='Choose this skill when the request is to fetch Cell Painting inputs from the configured access layer.',
        composes_with=('cp-extract-measurements', 'cp-extract-segmentation-artifacts'),
    ),
    'cp-extract-measurements': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['run-cellprofiler-profiling'],
        key='cp-extract-measurements',
        description='Run the configured CellProfiler profiling pipeline and write the standard measurement tables.',
        user_summary='Use this skill when you want CellProfiler measurement tables from raw images.',
        agent_summary='Choose this skill when the request is to run the profiling .cppipe and produce CellProfiler tables.',
        composes_with=('cp-build-single-cell-table',),
    ),
    'cp-build-single-cell-table': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['export-single-cell-measurements'],
        key='cp-build-single-cell-table',
        description='Merge CellProfiler image and object tables into a single-cell measurements table.',
        user_summary='Use this skill when you want one single-cell table after CellProfiler has produced the compartment tables.',
        agent_summary='Choose this skill when the request is to turn CellProfiler output tables into one single-cell table.',
        composes_with=('cyto-aggregate-profiles', 'cyto-annotate-profiles', 'cyto-normalize-profiles', 'cyto-select-profile-features'),
    ),
    'cyto-aggregate-profiles': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['run-pycytominer'],
        key='cyto-aggregate-profiles',
        description='Run the pycytominer path and return the aggregated classical profile table.',
        input_keys=('single_cell_path', 'output_dir'),
        typical_outputs=('aggregated.parquet',),
        user_summary='Use this skill when you want the aggregated pycytominer profile table.',
        agent_summary='Choose this skill when the request is to aggregate single-cell measurements into classical profiles.',
        composes_with=('cyto-annotate-profiles',),
    ),
    'cyto-annotate-profiles': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['run-pycytominer'],
        key='cyto-annotate-profiles',
        description='Run the pycytominer path and return the annotated classical profile table.',
        input_keys=('aggregated_path', 'output_dir'),
        typical_outputs=('annotated.parquet',),
        user_summary='Use this skill when you want the metadata-annotated pycytominer profile table.',
        agent_summary='Choose this skill when the request is to attach plate-map or treatment metadata to classical profiles.',
        composes_with=('cyto-normalize-profiles',),
    ),
    'cyto-normalize-profiles': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['run-pycytominer'],
        key='cyto-normalize-profiles',
        description='Run the pycytominer path and return the normalized classical profile table.',
        input_keys=('annotated_path', 'output_dir'),
        typical_outputs=('normalized.parquet',),
        user_summary='Use this skill when you want the normalized pycytominer profile table.',
        agent_summary='Choose this skill when the request is to normalize classical profiles before feature selection.',
        composes_with=('cyto-select-profile-features',),
    ),
    'cyto-select-profile-features': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['run-pycytominer'],
        key='cyto-select-profile-features',
        description='Run the pycytominer path and return the feature-selected classical profile table.',
        input_keys=('normalized_path', 'output_dir'),
        typical_outputs=('feature_selected.parquet',),
        user_summary='Use this skill when you want the final feature-selected classical profile table.',
        agent_summary='Choose this skill when the request is to produce the feature-selected pycytominer output.',
        composes_with=('cyto-summarize-classical-profiles',),
    ),
    'cyto-summarize-classical-profiles': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['summarize-classical-profiles'],
        key='cyto-summarize-classical-profiles',
        user_summary='Use this skill when you want a readable summary of classical profile outputs instead of raw tables only.',
        agent_summary='Choose this skill when the request is to explain or summarize classical pycytominer outputs for a human reader.',
    ),
    'cp-prepare-segmentation-inputs': PipelineSkillDefinition(
        key='cp-prepare-segmentation-inputs',
        description='Prepare the load-data table that the segmentation CellProfiler pipeline will use.',
        category='segmentation',
        input_keys=('output_dir',),
        typical_outputs=('load_data_for_segmentation.csv',),
        implements_with=('cellpaint_pipeline.segmentation_native',),
        user_summary='Use this skill when you want the segmentation input table without running the full segmentation export yet.',
        agent_summary='Choose this skill when the request is to prepare segmentation inputs before mask extraction.',
        composes_with=('cp-extract-segmentation-artifacts', 'cp-generate-segmentation-previews'),
    ),
    'cp-extract-segmentation-artifacts': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['run-segmentation-masks'],
        key='cp-extract-segmentation-artifacts',
        description='Run the segmentation CellProfiler pipeline and produce mask tables, labels, and outlines.',
        typical_outputs=advertised_outputs(SEGMENTATION_ARTIFACT_OUTPUTS),
        user_summary='Use this skill when you want segmentation outputs such as masks, labels, outlines, and object tables.',
        agent_summary='Choose this skill when the request is to execute the segmentation branch and produce mask artifacts.',
        composes_with=('cp-generate-segmentation-previews', 'crop-export-single-cell-crops', 'dp-export-deep-feature-inputs'),
    ),
    'cp-generate-segmentation-previews': _remap_skill_definition(
        CURRENT_ADVANCED_PIPELINE_SKILLS['generate-sample-previews'],
        key='cp-generate-segmentation-previews',
        user_summary='Use this skill when you want only segmentation preview PNGs from prepared inputs or a workflow root.',
        agent_summary='Choose this skill when the request is explicitly for segmentation preview images.',
    ),
    'crop-export-single-cell-crops': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['export-single-cell-crops'],
        key='crop-export-single-cell-crops',
        category='segmentation',
        user_summary='Use this skill when you want masked or unmasked single-cell crop stacks from a segmentation workflow root.',
        agent_summary='Choose this skill when the request is to export single-cell crops and the only decision is masked versus unmasked mode.',
        composes_with=('dp-export-deep-feature-inputs',),
    ),
    'dp-export-deep-feature-inputs': _remap_skill_definition(
        CURRENT_ADVANCED_PIPELINE_SKILLS['export-deepprofiler-inputs'],
        key='dp-export-deep-feature-inputs',
        description='Write the DeepProfiler field metadata and per-field nuclei location CSV files.',
        category='deepprofiler',
        user_summary='Use this skill when you want the DeepProfiler export bundle without building the project yet.',
        agent_summary='Choose this skill when the request is to stop at the DeepProfiler export stage.',
        composes_with=('dp-build-deep-feature-project',),
    ),
    'dp-build-deep-feature-project': _remap_skill_definition(
        CURRENT_ADVANCED_PIPELINE_SKILLS['build-deepprofiler-project'],
        key='dp-build-deep-feature-project',
        description='Build a runnable DeepProfiler project directory from a DeepProfiler export.',
        category='deepprofiler',
        user_summary='Use this skill when you want a runnable DeepProfiler project directory that is ready to execute next.',
        agent_summary='Choose this skill when the request is to build the DeepProfiler project stage without running the model yet.',
        composes_with=('dp-run-deep-feature-model',),
    ),
    'dp-run-deep-feature-model': _remap_skill_definition(
        CURRENT_ADVANCED_PIPELINE_SKILLS['run-deepprofiler-profile'],
        key='dp-run-deep-feature-model',
        description='Run the DeepProfiler model against a prepared DeepProfiler project.',
        category='deepprofiler',
        user_summary='Use this skill when you want raw DeepProfiler feature files from a prepared project.',
        agent_summary='Choose this skill when the request is to execute the deep feature model without collecting the tabular outputs yet.',
        composes_with=('dp-collect-deep-features',),
    ),
    'dp-collect-deep-features': _remap_skill_definition(
        CURRENT_ADVANCED_PIPELINE_SKILLS['collect-deepprofiler-features'],
        key='dp-collect-deep-features',
        description='Collect DeepProfiler outputs into single-cell and well-level tabular feature files.',
        category='deepprofiler',
        user_summary='Use this skill when you want DeepProfiler feature files collected into analysis-ready tables.',
        agent_summary='Choose this skill when the request is to convert raw DeepProfiler outputs into tabular features.',
        composes_with=('dp-summarize-deep-features',),
    ),
    'dp-summarize-deep-features': _remap_skill_definition(
        CURRENT_PRIMARY_PIPELINE_SKILLS['summarize-deepprofiler-profiles'],
        key='dp-summarize-deep-features',
        description='Summarize DeepProfiler single-cell and well-level tables into readable metadata, variability, and PCA outputs.',
        category='deepprofiler',
        user_summary='Use this skill when you want a readable summary of DeepProfiler outputs instead of raw embedding tables only.',
        agent_summary='Choose this skill when the request is to explain or summarize DeepProfiler outputs for a human reader.',
    ),
}

ADVANCED_PIPELINE_SKILLS: dict[str, PipelineSkillDefinition] = {
    **{
        key: _remap_skill_definition(
            definition,
            status='advanced',
        )
        for key, definition in CURRENT_PRIMARY_PIPELINE_SKILLS.items()
    },
    **{
        key: _remap_skill_definition(
            definition,
            status='advanced',
        )
        for key, definition in CURRENT_ADVANCED_PIPELINE_SKILLS.items()
    },
}

LEGACY_PIPELINE_SKILLS: dict[str, PipelineSkillDefinition] = {
    'build-image-manifest': _remap_skill_definition(
        CURRENT_LEGACY_PIPELINE_SKILLS['build-image-manifest'],
        replaced_by=('cp-extract-measurements',),
    ),
    'validate-profiling-inputs': _remap_skill_definition(
        CURRENT_LEGACY_PIPELINE_SKILLS['validate-profiling-inputs'],
        replaced_by=('cp-extract-measurements', 'cp-build-single-cell-table'),
    ),
    'export-segmentation-load-data': _remap_skill_definition(
        CURRENT_LEGACY_PIPELINE_SKILLS['export-segmentation-load-data'],
        replaced_by=('cp-prepare-segmentation-inputs',),
    ),
    'plan-data-access': _remap_skill_definition(
        CURRENT_LEGACY_PIPELINE_SKILLS['plan-data-access'],
        replaced_by=('data-inspect-availability', 'data-plan-download'),
    ),
    'download-data': _remap_skill_definition(
        CURRENT_LEGACY_PIPELINE_SKILLS['download-data'],
        replaced_by=('data-download',),
    ),
    'run-classical-profiling': _remap_skill_definition(
        CURRENT_LEGACY_PIPELINE_SKILLS['run-classical-profiling'],
        replaced_by=('cyto-aggregate-profiles', 'cyto-select-profile-features'),
    ),
    'run-segmentation': _remap_skill_definition(
        CURRENT_LEGACY_PIPELINE_SKILLS['run-segmentation'],
        replaced_by=('cp-extract-segmentation-artifacts',),
    ),
    'prepare-deepprofiler-inputs': _remap_skill_definition(
        CURRENT_LEGACY_PIPELINE_SKILLS['prepare-deepprofiler-inputs'],
        replaced_by=('dp-export-deep-feature-inputs',),
    ),
}

ALL_PIPELINE_SKILLS: dict[str, PipelineSkillDefinition] = {
    **PRIMARY_PIPELINE_SKILLS,
    **ADVANCED_PIPELINE_SKILLS,
    **LEGACY_PIPELINE_SKILLS,
}


def available_pipeline_skills(*, include_advanced: bool = False, include_legacy: bool = False) -> list[str]:
    catalog: dict[str, PipelineSkillDefinition] = dict(PRIMARY_PIPELINE_SKILLS)
    if include_advanced:
        catalog.update(ADVANCED_PIPELINE_SKILLS)
    if include_legacy:
        catalog.update(LEGACY_PIPELINE_SKILLS)
    return list(catalog)


def get_pipeline_skill_definition(skill_key: str) -> PipelineSkillDefinition:
    if skill_key not in ALL_PIPELINE_SKILLS:
        available = ', '.join(available_pipeline_skills(include_advanced=True, include_legacy=True))
        raise KeyError(f'Unknown pipeline skill: {skill_key}. Available: {available}')
    return ALL_PIPELINE_SKILLS[skill_key]


# Re-exported for convenience: callers historically imported the serialiser
# next to the catalog helpers.
from cellpaint_pipeline.skills.definitions import (  # noqa: E402  (re-export)
    pipeline_skill_definition_to_dict as pipeline_skill_definition_to_dict,
)
