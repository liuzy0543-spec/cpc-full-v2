"""CLI commands for the deepprofiler domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.cli.helpers import _impl

from cellpaint_pipeline.cli.helpers import (
    _maybe_resolve_path,
    _resolve_deepprofiler_source_kwargs,
)



def _cmd_export_deepprofiler_input(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            source_kwargs = _resolve_deepprofiler_source_kwargs(args)
            export_result = _impl("export_deepprofiler_input")(
                config,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                source_label=args.source_label,
                **source_kwargs,
            )
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.deepprofiler_export',
                'export_root': str(export_result.export_root),
                'manifest_path': str(export_result.manifest_path),
                'field_metadata_path': str(export_result.field_metadata_path),
                'locations_root': str(export_result.locations_root),
                'field_count': export_result.field_count,
                'location_file_count': export_result.location_file_count,
                'total_nuclei': export_result.total_nuclei,
                'source_image_csv': str(export_result.source_image_csv),
                'source_nuclei_csv': str(export_result.source_nuclei_csv),
                'source_load_data_csv': str(export_result.source_load_data_csv),
                'source_label': export_result.source_label,
            }, indent=2, ensure_ascii=False))
            return 0



def _cmd_build_deepprofiler_project(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            project_result = _impl("build_deepprofiler_project")(
                config,
                output_dir=_maybe_resolve_path(args.output_dir),
                workflow_root=_maybe_resolve_path(args.workflow_root),
                export_root=_maybe_resolve_path(args.export_root),
                experiment_name=args.experiment_name,
                config_filename=args.config_name,
                metadata_filename=args.metadata_name,
            )
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.deepprofiler_project',
                'project_root': str(project_result.project_root),
                'manifest_path': str(project_result.manifest_path),
                'config_path': str(project_result.config_path),
                'metadata_path': str(project_result.metadata_path),
                'locations_root': str(project_result.locations_root),
                'field_count': project_result.field_count,
                'location_file_count': project_result.location_file_count,
                'image_width': project_result.image_width,
                'image_height': project_result.image_height,
                'image_bits': project_result.image_bits,
                'image_format': project_result.image_format,
                'experiment_name': project_result.experiment_name,
                'label_field': project_result.label_field,
                'control_value': project_result.control_value,
                'export_root': str(project_result.export_root),
                'source_label': project_result.source_label,
            }, indent=2, ensure_ascii=False))
            return 0



def _cmd_run_deepprofiler_profile(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            profile_result = _impl("run_deepprofiler_profile")(
                config,
                project_root=Path(args.project_root).expanduser().resolve(),
                experiment_name=args.experiment_name,
                config_filename=args.config_name,
                metadata_filename=args.metadata_name,
                gpu=args.gpu,
            )
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.deepprofiler_profile',
                'project_root': str(profile_result.project_root),
                'manifest_path': str(profile_result.manifest_path),
                'config_path': str(profile_result.config_path),
                'metadata_path': str(profile_result.metadata_path),
                'experiment_name': profile_result.experiment_name,
                'feature_dir': str(profile_result.feature_dir),
                'checkpoint_dir': str(profile_result.checkpoint_dir),
                'log_path': str(profile_result.log_path) if profile_result.log_path else None,
                'command': profile_result.command,
                'returncode': profile_result.returncode,
            }, indent=2, ensure_ascii=False))
            return 0



def _cmd_collect_deepprofiler_features(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            collection_result = _impl("collect_deepprofiler_features")(
                config,
                project_root=Path(args.project_root).expanduser().resolve(),
                output_dir=_maybe_resolve_path(args.output_dir),
                experiment_name=args.experiment_name,
            )
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.deepprofiler_feature_collection',
                'project_root': str(collection_result.project_root),
                'feature_dir': str(collection_result.feature_dir),
                'output_dir': str(collection_result.output_dir),
                'manifest_path': str(collection_result.manifest_path),
                'single_cell_parquet_path': str(collection_result.single_cell_parquet_path),
                'single_cell_csv_gz_path': str(collection_result.single_cell_csv_gz_path),
                'well_aggregated_parquet_path': str(collection_result.well_aggregated_parquet_path),
                'well_aggregated_csv_gz_path': str(collection_result.well_aggregated_csv_gz_path),
                'field_summary_path': str(collection_result.field_summary_path),
                'experiment_name': collection_result.experiment_name,
                'field_file_count': collection_result.field_file_count,
                'cell_count': collection_result.cell_count,
                'feature_count': collection_result.feature_count,
                'metadata_column_count': collection_result.metadata_column_count,
                'well_count': collection_result.well_count,
            }, indent=2, ensure_ascii=False))
            return 0



def _cmd_run_deepprofiler_pipeline(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            source_kwargs = _resolve_deepprofiler_source_kwargs(args)
            pipeline_result = _impl("run_deepprofiler_pipeline")(
                config,
                output_dir=_maybe_resolve_path(args.output_dir),
                workflow_root=_maybe_resolve_path(args.workflow_root),
                source_label=args.source_label,
                experiment_name=args.experiment_name,
                config_filename=args.config_name,
                metadata_filename=args.metadata_name,
                gpu=args.gpu,
                **source_kwargs,
            )
            print(json.dumps(_impl("deepprofiler_pipeline_result_to_dict")(pipeline_result), indent=2, ensure_ascii=False))
            return 0 if pipeline_result.ok else 1
def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach this domain's sub-commands, in their original order."""
    export_parser = subparsers.add_parser(
        'export-deepprofiler-input',
        help='Export field metadata and nuclei locations for DeepProfiler-style consumption.',
    )
    export_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    export_parser.add_argument('--output-dir', default=None, help='Optional export directory. Defaults to deepprofiler_export_root in config.')
    export_parser.add_argument('--workflow-root', default=None, help='Optional workflow root containing cellprofiler_masks and load_data_for_segmentation.csv.')
    export_parser.add_argument('--image-csv-path', default=None, help='Optional Image.csv override for DeepProfiler export.')
    export_parser.add_argument('--nuclei-csv-path', default=None, help='Optional Nuclei.csv override for DeepProfiler export.')
    export_parser.add_argument('--load-data-path', default=None, help='Optional segmentation load-data CSV override for DeepProfiler export.')
    export_parser.add_argument('--source-label', default=None, help='Optional free-text label recorded in the DeepProfiler export manifest.')

    project_parser = subparsers.add_parser(
        'build-deepprofiler-project',
        help='Materialize a DeepProfiler project directory from a prepared DeepProfiler export.',
    )
    project_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    project_parser.add_argument('--output-dir', default=None, help='Optional output directory for the DeepProfiler project.')
    project_parser.add_argument('--workflow-root', default=None, help='Optional workflow root containing deepprofiler_export/.')
    project_parser.add_argument('--export-root', default=None, help='Optional DeepProfiler export root override.')
    project_parser.add_argument('--experiment-name', default=None, help='Optional DeepProfiler experiment name override.')
    project_parser.add_argument('--config-name', default=None, help='Optional config filename under inputs/config/.')
    project_parser.add_argument('--metadata-name', default=None, help='Optional metadata filename under inputs/metadata/.')

    profile_parser = subparsers.add_parser(
        'run-deepprofiler-profile',
        help='Run the DeepProfiler profile command inside a prepared DeepProfiler project.',
    )
    profile_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    profile_parser.add_argument('--project-root', required=True, help='Path to the prepared DeepProfiler project root.')
    profile_parser.add_argument('--experiment-name', default=None, help='Optional DeepProfiler experiment name override.')
    profile_parser.add_argument('--config-name', default=None, help='Optional config filename override.')
    profile_parser.add_argument('--metadata-name', default=None, help='Optional metadata filename override.')
    profile_parser.add_argument('--gpu', default=None, help='Optional GPU id forwarded to DeepProfiler.')

    collect_parser = subparsers.add_parser(
        'collect-deepprofiler-features',
        help='Collect DeepProfiler .npz outputs into pycytominer-friendly single-cell and well-level tables.',
    )
    collect_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    collect_parser.add_argument('--project-root', required=True, help='Path to the prepared DeepProfiler project root.')
    collect_parser.add_argument('--output-dir', default=None, help='Optional output directory for collected DeepProfiler tables.')
    collect_parser.add_argument('--experiment-name', default=None, help='Optional DeepProfiler experiment name override.')

    pipeline_parser = subparsers.add_parser(
        'run-deepprofiler-pipeline',
        help='Run the full DeepProfiler export, project build, profile, and feature collection chain.',
    )
    pipeline_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    pipeline_parser.add_argument('--output-dir', default=None, help='Optional output directory for the standardized DeepProfiler run root.')
    pipeline_parser.add_argument('--workflow-root', default=None, help='Optional workflow root used to infer Image.csv, Nuclei.csv, and load_data_for_segmentation.csv.')
    pipeline_parser.add_argument('--image-csv-path', default=None, help='Optional Image.csv override for DeepProfiler pipeline input.')
    pipeline_parser.add_argument('--nuclei-csv-path', default=None, help='Optional Nuclei.csv override for DeepProfiler pipeline input.')
    pipeline_parser.add_argument('--load-data-path', default=None, help='Optional segmentation load-data CSV override for DeepProfiler pipeline input.')
    pipeline_parser.add_argument('--source-label', default=None, help='Optional free-text label recorded in the DeepProfiler export manifest.')
    pipeline_parser.add_argument('--experiment-name', default=None, help='Optional DeepProfiler experiment name override.')
    pipeline_parser.add_argument('--config-name', default=None, help='Optional config filename under inputs/config/.')
    pipeline_parser.add_argument('--metadata-name', default=None, help='Optional metadata filename under inputs/metadata/.')
    pipeline_parser.add_argument('--gpu', default=None, help='Optional GPU id forwarded to DeepProfiler.')


HANDLERS = {
    'export-deepprofiler-input': _cmd_export_deepprofiler_input,
    'build-deepprofiler-project': _cmd_build_deepprofiler_project,
    'run-deepprofiler-profile': _cmd_run_deepprofiler_profile,
    'collect-deepprofiler-features': _cmd_collect_deepprofiler_features,
    'run-deepprofiler-pipeline': _cmd_run_deepprofiler_pipeline,
}
