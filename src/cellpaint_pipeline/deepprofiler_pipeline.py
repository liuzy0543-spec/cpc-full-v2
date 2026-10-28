from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cellpaint_pipeline.adapters.deepprofiler import export_deepprofiler_input
from cellpaint_pipeline.adapters.deepprofiler import infer_deepprofiler_sources_from_workflow_root
from cellpaint_pipeline.adapters.deepprofiler_features import collect_deepprofiler_features
from cellpaint_pipeline.adapters.deepprofiler_project import build_deepprofiler_project
from cellpaint_pipeline.adapters.deepprofiler_project import run_deepprofiler_profile
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class DeepProfilerPipelineResult:
    output_dir: Path
    manifest_path: Path
    export_root: Path
    export_manifest_path: Path
    project_root: Path
    project_manifest_path: Path
    feature_dir: Path
    collection_output_dir: Path
    collection_manifest_path: Path
    experiment_name: str
    source_label: str
    field_count: int
    location_file_count: int
    total_nuclei: int
    field_file_count: int
    cell_count: int
    feature_count: int
    well_count: int
    profile_returncode: int
    log_path: Path | None
    ok: bool


def run_deepprofiler_pipeline(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    workflow_root: Path | None = None,
    image_csv_path: Path | None = None,
    nuclei_csv_path: Path | None = None,
    load_data_csv_path: Path | None = None,
    source_label: str | None = None,
    experiment_name: str | None = None,
    config_filename: str | None = None,
    metadata_filename: str | None = None,
    gpu: str | None = None,
) -> DeepProfilerPipelineResult:
    run_root = output_dir.resolve() if output_dir is not None else (config.default_output_root / 'deepprofiler_pipeline')
    run_root.mkdir(parents=True, exist_ok=True)

    source_kwargs = _resolve_source_kwargs(
        workflow_root=workflow_root,
        image_csv_path=image_csv_path,
        nuclei_csv_path=nuclei_csv_path,
        load_data_csv_path=load_data_csv_path,
    )
    resolved_source_label = source_label or ('workflow-local-mask-export' if workflow_root is not None else None)

    export_result = export_deepprofiler_input(
        config,
        output_dir=run_root / 'deepprofiler_export',
        source_label=resolved_source_label,
        **source_kwargs,
    )
    project_result = build_deepprofiler_project(
        config,
        output_dir=run_root / 'deepprofiler_project',
        export_root=export_result.export_root,
        experiment_name=experiment_name,
        config_filename=config_filename,
        metadata_filename=metadata_filename,
    )
    profile_result = run_deepprofiler_profile(
        config,
        project_root=project_result.project_root,
        experiment_name=project_result.experiment_name,
        config_filename=config_filename,
        metadata_filename=metadata_filename,
        gpu=gpu,
    )
    if profile_result.returncode != 0:
        raise RuntimeError(
            f'DeepProfiler profile failed with return code {profile_result.returncode}. '
            f'See log: {profile_result.log_path}'
        )
    collection_result = collect_deepprofiler_features(
        config,
        project_root=project_result.project_root,
        output_dir=run_root / 'deepprofiler_tables',
        experiment_name=project_result.experiment_name,
    )

    manifest_path = run_root / 'deepprofiler_pipeline_manifest.json'
    payload = {
        'implementation': 'cellpaint_pipeline.deepprofiler_pipeline',
        'package_version': _package_version(),
        'output_dir': str(run_root),
        'workflow_root': str(workflow_root.expanduser().resolve()) if workflow_root is not None else None,
        'source_image_csv': str(export_result.source_image_csv),
        'source_nuclei_csv': str(export_result.source_nuclei_csv),
        'source_load_data_csv': str(export_result.source_load_data_csv),
        'source_label': export_result.source_label,
        'export_root': str(export_result.export_root),
        'export_manifest_path': str(export_result.manifest_path),
        'project_root': str(project_result.project_root),
        'project_manifest_path': str(project_result.manifest_path),
        'feature_dir': str(profile_result.feature_dir),
        'collection_output_dir': str(collection_result.output_dir),
        'collection_manifest_path': str(collection_result.manifest_path),
        'experiment_name': project_result.experiment_name,
        'field_count': export_result.field_count,
        'location_file_count': export_result.location_file_count,
        'total_nuclei': export_result.total_nuclei,
        'field_file_count': collection_result.field_file_count,
        'cell_count': collection_result.cell_count,
        'feature_count': collection_result.feature_count,
        'well_count': collection_result.well_count,
        'profile_returncode': profile_result.returncode,
        'log_path': str(profile_result.log_path) if profile_result.log_path else None,
        'ok': profile_result.returncode == 0,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + chr(10), encoding='utf-8')

    return DeepProfilerPipelineResult(
        output_dir=run_root,
        manifest_path=manifest_path,
        export_root=export_result.export_root,
        export_manifest_path=export_result.manifest_path,
        project_root=project_result.project_root,
        project_manifest_path=project_result.manifest_path,
        feature_dir=profile_result.feature_dir,
        collection_output_dir=collection_result.output_dir,
        collection_manifest_path=collection_result.manifest_path,
        experiment_name=project_result.experiment_name,
        source_label=export_result.source_label,
        field_count=export_result.field_count,
        location_file_count=export_result.location_file_count,
        total_nuclei=export_result.total_nuclei,
        field_file_count=collection_result.field_file_count,
        cell_count=collection_result.cell_count,
        feature_count=collection_result.feature_count,
        well_count=collection_result.well_count,
        profile_returncode=profile_result.returncode,
        log_path=profile_result.log_path,
        ok=profile_result.returncode == 0,
    )


def deepprofiler_pipeline_result_to_dict(result: DeepProfilerPipelineResult) -> dict[str, object]:
    return {
        'implementation': 'cellpaint_pipeline.deepprofiler_pipeline',
        'output_dir': str(result.output_dir),
        'manifest_path': str(result.manifest_path),
        'export_root': str(result.export_root),
        'export_manifest_path': str(result.export_manifest_path),
        'project_root': str(result.project_root),
        'project_manifest_path': str(result.project_manifest_path),
        'feature_dir': str(result.feature_dir),
        'collection_output_dir': str(result.collection_output_dir),
        'collection_manifest_path': str(result.collection_manifest_path),
        'experiment_name': result.experiment_name,
        'source_label': result.source_label,
        'field_count': result.field_count,
        'location_file_count': result.location_file_count,
        'total_nuclei': result.total_nuclei,
        'field_file_count': result.field_file_count,
        'cell_count': result.cell_count,
        'feature_count': result.feature_count,
        'well_count': result.well_count,
        'profile_returncode': result.profile_returncode,
        'log_path': str(result.log_path) if result.log_path else None,
        'ok': result.ok,
    }


def _resolve_source_kwargs(
    *,
    workflow_root: Path | None,
    image_csv_path: Path | None,
    nuclei_csv_path: Path | None,
    load_data_csv_path: Path | None,
) -> dict[str, Path]:
    source_kwargs: dict[str, Path] = {}
    if workflow_root is not None:
        source_kwargs.update(
            infer_deepprofiler_sources_from_workflow_root(
                workflow_root.expanduser().resolve()
            )
        )
    if image_csv_path is not None:
        source_kwargs['image_csv_path'] = image_csv_path.expanduser().resolve()
    if nuclei_csv_path is not None:
        source_kwargs['nuclei_csv_path'] = nuclei_csv_path.expanduser().resolve()
    if load_data_csv_path is not None:
        source_kwargs['load_data_csv_path'] = load_data_csv_path.expanduser().resolve()
    return source_kwargs


def _package_version() -> str:
    from cellpaint_pipeline import __version__

    return __version__
