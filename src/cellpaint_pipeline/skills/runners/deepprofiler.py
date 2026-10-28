from __future__ import annotations

"""DeepProfiler skill runners."""

from typing import Any

from cellpaint_pipeline.skills.context import SkillRuntimeContext
from cellpaint_pipeline.skills.definitions import PipelineSkillResult
from cellpaint_pipeline.skills.finalize import _native, _finalize_skill_result, _json_ready
from cellpaint_pipeline.skills.outputs import (
    DEEPPROFILER_COLLECT_OUTPUTS,
    build_primary_outputs,
)


def _run_export_deepprofiler_inputs(context: SkillRuntimeContext) -> PipelineSkillResult:
    from cellpaint_pipeline.adapters.deepprofiler import export_deepprofiler_input

    if context.workflow_root is not None:
        result = export_deepprofiler_input(
            context.config,
            output_dir=context.run_root,
            image_csv_path=context.workflow_root / 'cellprofiler_masks' / 'Image.csv',
            nuclei_csv_path=context.workflow_root / 'cellprofiler_masks' / 'Nuclei.csv',
            load_data_csv_path=context.workflow_root / 'load_data_for_segmentation.csv',
            source_label='workflow-local-mask-export',
        )
    else:
        result = export_deepprofiler_input(
            context.config,
            output_dir=context.run_root,
            image_csv_path=context.image_csv_path,
            nuclei_csv_path=context.nuclei_csv_path,
            load_data_csv_path=context.load_data_csv_path,
        )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.adapters.deepprofiler',
        primary_outputs={
            'export_root': result.export_root,
            'export_manifest_path': result.manifest_path,
            'field_metadata_path': result.field_metadata_path,
            'locations_root': result.locations_root,
        },
        details=_json_ready(result),
        ok=True,
    )


def _run_prepare_deepprofiler_project(context: SkillRuntimeContext) -> PipelineSkillResult:
    from cellpaint_pipeline.adapters.deepprofiler import export_deepprofiler_input
    from cellpaint_pipeline.adapters.deepprofiler_project import build_deepprofiler_project

    export_result = None
    resolved_export_root = context.export_root
    if resolved_export_root is None:
        if context.workflow_root is not None:
            export_result = export_deepprofiler_input(
                context.config,
                output_dir=context.run_root / 'deepprofiler_export',
                image_csv_path=context.workflow_root / 'cellprofiler_masks' / 'Image.csv',
                nuclei_csv_path=context.workflow_root / 'cellprofiler_masks' / 'Nuclei.csv',
                load_data_csv_path=context.workflow_root / 'load_data_for_segmentation.csv',
                source_label='workflow-local-mask-export',
            )
        else:
            export_result = export_deepprofiler_input(
                context.config,
                output_dir=context.run_root / 'deepprofiler_export',
                image_csv_path=context.image_csv_path,
                nuclei_csv_path=context.nuclei_csv_path,
                load_data_csv_path=context.load_data_csv_path,
            )
        resolved_export_root = export_result.export_root

    project_result = build_deepprofiler_project(
        context.config,
        output_dir=context.run_root / 'deepprofiler_project',
        export_root=resolved_export_root,
        experiment_name=context.experiment_name,
        config_filename=context.config_filename,
        metadata_filename=context.metadata_filename,
    )
    details: dict[str, Any] = {
        'project': _json_ready(project_result),
    }
    if export_result is not None:
        details['export'] = _json_ready(export_result)
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.adapters.deepprofiler_project',
        primary_outputs={
            'project_root': project_result.project_root,
            'project_manifest_path': project_result.manifest_path,
            'config_path': project_result.config_path,
            'metadata_path': project_result.metadata_path,
            'locations_root': project_result.locations_root,
            'export_root': resolved_export_root,
            'export_manifest_path': export_result.manifest_path if export_result is not None else (resolved_export_root / 'manifest.json'),
        },
        details=details,
        ok=True,
    )


def _run_build_deepprofiler_project(context: SkillRuntimeContext) -> PipelineSkillResult:
    from cellpaint_pipeline.adapters.deepprofiler_project import build_deepprofiler_project

    result = build_deepprofiler_project(
        context.config,
        output_dir=context.run_root,
        workflow_root=context.workflow_root,
        export_root=context.export_root,
        experiment_name=context.experiment_name,
        config_filename=context.config_filename,
        metadata_filename=context.metadata_filename,
    )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.adapters.deepprofiler_project',
        primary_outputs={
            'project_root': result.project_root,
            'project_manifest_path': result.manifest_path,
            'config_path': result.config_path,
            'metadata_path': result.metadata_path,
            'locations_root': result.locations_root,
        },
        details=_json_ready(result),
        ok=True,
    )


def _run_deepprofiler(context: SkillRuntimeContext) -> PipelineSkillResult:
    from cellpaint_pipeline.adapters.deepprofiler_features import collect_deepprofiler_features
    from cellpaint_pipeline.adapters.deepprofiler_project import run_deepprofiler_profile
    from cellpaint_pipeline.deepprofiler_pipeline import deepprofiler_pipeline_result_to_dict, run_deepprofiler_pipeline

    if context.project_root is not None:
        profile_result = run_deepprofiler_profile(
            context.config,
            project_root=context.project_root,
            experiment_name=context.experiment_name,
            config_filename=context.config_filename,
            metadata_filename=context.metadata_filename,
            gpu=context.gpu,
        )
        if profile_result.returncode != 0:
            return _finalize_skill_result(
                context,
                implementation='cellpaint_pipeline.adapters.deepprofiler_project',
                primary_outputs={
                    'project_root': profile_result.project_root,
                    'feature_dir': profile_result.feature_dir,
                    'log_path': profile_result.log_path,
                },
                details={'profile': _json_ready(profile_result)},
                ok=False,
            )
        collection_result = collect_deepprofiler_features(
            context.config,
            project_root=context.project_root,
            output_dir=context.run_root / 'deepprofiler_tables',
            experiment_name=context.experiment_name,
        )
        return _finalize_skill_result(
            context,
            implementation='cellpaint_pipeline.adapters.deepprofiler_features',
            primary_outputs={
                'project_root': profile_result.project_root,
                'feature_dir': profile_result.feature_dir,
                'single_cell_parquet_path': collection_result.single_cell_parquet_path,
                'single_cell_csv_gz_path': collection_result.single_cell_csv_gz_path,
                'well_aggregated_parquet_path': collection_result.well_aggregated_parquet_path,
                'well_aggregated_csv_gz_path': collection_result.well_aggregated_csv_gz_path,
                'feature_manifest_path': collection_result.manifest_path,
                'log_path': profile_result.log_path,
            },
            details={
                'profile': _json_ready(profile_result),
                'collection': _json_ready(collection_result),
            },
            ok=True,
        )

    result = run_deepprofiler_pipeline(
        context.config,
        output_dir=context.run_root,
        workflow_root=context.workflow_root,
        image_csv_path=context.image_csv_path,
        nuclei_csv_path=context.nuclei_csv_path,
        load_data_csv_path=context.load_data_csv_path,
        experiment_name=context.experiment_name,
        config_filename=context.config_filename,
        metadata_filename=context.metadata_filename,
        gpu=context.gpu,
    )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.deepprofiler_pipeline',
        primary_outputs={
            'project_root': result.project_root,
            'feature_dir': result.feature_dir,
            'collection_output_dir': result.collection_output_dir,
            'pipeline_manifest_path': result.manifest_path,
            'collection_manifest_path': result.collection_manifest_path,
        },
        details=deepprofiler_pipeline_result_to_dict(result),
        ok=result.ok,
    )


def _run_summarize_deepprofiler_profiles(context: SkillRuntimeContext) -> PipelineSkillResult:
    result = _native('summarize_deepprofiler_profiles')(
        output_dir=context.run_root,
        single_cell_parquet_path=context.single_cell_parquet_path,
        well_aggregated_parquet_path=context.well_aggregated_parquet_path,
        manifest_path=context.manifest_path,
    )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.profile_summaries',
        primary_outputs={
            'summary_path': result.summary_path,
            'well_metadata_summary_path': result.well_metadata_summary_path,
            'top_variable_features_path': result.top_variable_features_path,
            'pca_coordinates_path': result.pca_coordinates_path,
            'pca_plot_path': result.pca_plot_path,
        },
        details=_json_ready(result),
        ok=True,
    )


def _run_deepprofiler_profile(context: SkillRuntimeContext) -> PipelineSkillResult:
    from cellpaint_pipeline.adapters.deepprofiler_project import run_deepprofiler_profile

    resolved_project_root = context.project_root or context.config.deepprofiler_project_root
    result = run_deepprofiler_profile(
        context.config,
        project_root=resolved_project_root,
        experiment_name=context.experiment_name,
        config_filename=context.config_filename,
        metadata_filename=context.metadata_filename,
        gpu=context.gpu,
    )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.adapters.deepprofiler_project',
        primary_outputs={
            'project_root': result.project_root,
            'feature_dir': result.feature_dir,
            'checkpoint_dir': result.checkpoint_dir,
            'log_path': result.log_path,
        },
        details=_json_ready(result),
        ok=result.returncode == 0,
    )


def _run_collect_deepprofiler_features(context: SkillRuntimeContext) -> PipelineSkillResult:
    from cellpaint_pipeline.adapters.deepprofiler_features import collect_deepprofiler_features

    resolved_project_root = context.project_root or context.config.deepprofiler_project_root
    result = collect_deepprofiler_features(
        context.config,
        project_root=resolved_project_root,
        output_dir=context.run_root,
        experiment_name=context.experiment_name,
    )
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.adapters.deepprofiler_features',
        primary_outputs=build_primary_outputs(
            DEEPPROFILER_COLLECT_OUTPUTS,
            {
                'single_cell_parquet_path': result.single_cell_parquet_path,
                'single_cell_csv_gz_path': result.single_cell_csv_gz_path,
                'well_aggregated_parquet_path': result.well_aggregated_parquet_path,
                'well_aggregated_csv_gz_path': result.well_aggregated_csv_gz_path,
                'field_summary_path': result.field_summary_path,
                'feature_manifest_path': result.manifest_path,
            },
        ),
        details=_json_ready(result),
        ok=True,
    )
