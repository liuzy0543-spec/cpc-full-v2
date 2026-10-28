from __future__ import annotations

import argparse
from pathlib import Path

from cellpaint_pipeline.cli.lazy import (
    infer_deepprofiler_sources_from_workflow_root,
)


"""Argument and result helpers shared by more than one command."""
def _normalize_extra_args(extra_args: list[str]) -> list[str]:
    if extra_args and extra_args[0] == '--':
        return extra_args[1:]
    return extra_args

def _resolve_deepprofiler_source_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    source_kwargs: dict[str, Path] = {}
    if args.workflow_root:
        source_kwargs.update(
            infer_deepprofiler_sources_from_workflow_root(
                Path(args.workflow_root).expanduser().resolve()
            )
        )
    if args.image_csv_path:
        source_kwargs['image_csv_path'] = Path(args.image_csv_path).expanduser().resolve()
    if args.nuclei_csv_path:
        source_kwargs['nuclei_csv_path'] = Path(args.nuclei_csv_path).expanduser().resolve()
    if args.load_data_path:
        source_kwargs['load_data_csv_path'] = Path(args.load_data_path).expanduser().resolve()
    return source_kwargs

def _maybe_resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()

def _native_result_to_dict(result: object) -> dict:
    if hasattr(result, 'row_count') and hasattr(result, 'unmatched_files'):
        return {
            'implementation': 'native',
            'step': 'build-image-manifest',
            'output_path': str(result.output_path),
            'row_count': result.row_count,
            'unmatched_file_count': len(result.unmatched_files),
            'unmatched_files': result.unmatched_files,
        }
    if hasattr(result, 'feature_selected_path'):
        return {
            'implementation': 'native',
            'step': 'run-pycytominer',
            'aggregated_path': str(result.aggregated_path),
            'annotated_path': str(result.annotated_path),
            'normalized_path': str(result.normalized_path),
            'feature_selected_path': str(result.feature_selected_path),
            'aggregated_row_count': result.aggregated_row_count,
            'aggregated_column_count': result.aggregated_column_count,
            'annotated_row_count': result.annotated_row_count,
            'annotated_column_count': result.annotated_column_count,
            'normalized_row_count': result.normalized_row_count,
            'normalized_column_count': result.normalized_column_count,
            'feature_selected_row_count': result.feature_selected_row_count,
            'feature_selected_column_count': result.feature_selected_column_count,
        }
    if hasattr(result, 'mode'):
        return {
            'implementation': 'native',
            'step': 'export-cellprofiler-to-singlecell',
            'output_path': str(result.output_path),
            'row_count': result.row_count,
            'column_count': result.column_count,
            'object_table': result.object_table,
            'mode': result.mode,
            'shard_count': result.shard_count,
        }
    return {
        'implementation': 'native',
        'step': 'validate-inputs',
        'raw_dir': str(result.raw_dir),
        'raw_file_count': result.raw_file_count,
        'manifest_path': str(result.manifest_path),
        'plate_map_path': str(result.plate_map_path),
        'ok': result.ok,
        'problems': result.problems,
    }

def _native_result_ok(result: object) -> bool:
    if hasattr(result, 'ok'):
        return bool(result.ok)
    if hasattr(result, 'unmatched_files'):
        return len(result.unmatched_files) == 0
    return True

def _native_segmentation_result_to_dict(result: object) -> dict:
    if hasattr(result, 'crop_count') and hasattr(result, 'background_masked'):
        return {
            'implementation': 'native',
            'step': 'extract-single-cell-crops',
            'mode': result.mode,
            'crops_dir': str(result.crops_dir),
            'manifest_path': str(result.manifest_path),
            'crop_count': result.crop_count,
            'worker_count': result.worker_count,
            'background_masked': result.background_masked,
        }
    if hasattr(result, 'preview_count') and hasattr(result, 'manifest_path'):
        return {
            'implementation': 'native',
            'step': 'generate-png-previews',
            'mode': result.mode,
            'manifest_path': str(result.manifest_path),
            'output_dir': str(result.output_dir),
            'preview_count': result.preview_count,
            'worker_count': result.worker_count,
            'chunk_size': result.chunk_size,
        }
    if hasattr(result, 'generated_count'):
        return {
            'implementation': 'native',
            'step': 'generate-sample-previews',
            'output_dir': str(result.output_dir),
            'generated_count': result.generated_count,
            'skipped_existing': result.skipped_existing,
            'field_count': result.field_count,
        }
    if hasattr(result, 'module_count'):
        return {
            'implementation': 'native',
            'step': 'build-mask-export-pipeline',
            'output_path': str(result.output_path),
            'module_count': result.module_count,
            'source_cppipe_path': str(result.source_cppipe_path),
            'selected_via': result.selected_via,
            'execution_mode': result.execution_mode,
        }
    return {
        'implementation': 'native',
        'step': 'prepare-load-data',
        'output_path': str(result.output_path),
        'row_count': result.row_count,
        'plate_count': result.plate_count,
        'well_count': result.well_count,
        'site_count': result.site_count,
    }


def _impl(name: str):
    """Resolve an implementation proxy through the package facade.

    Going through the facade (rather than importing the proxy directly) keeps
    patch('cellpaint_pipeline.cli.<name>') effective, exactly as it was when
    every command lived in one module.  The facade is looked up at call time
    so this module can be imported while cellpaint_pipeline.cli is still
    initialising.
    """
    from cellpaint_pipeline import cli as _cli

    return getattr(_cli, name)
