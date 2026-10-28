from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class NativeManifestResult:
    output_path: Path
    row_count: int
    unmatched_files: list[str]


@dataclass(frozen=True)
class NativeValidationResult:
    raw_dir: Path
    raw_file_count: int
    manifest_path: Path
    plate_map_path: Path
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class NativeSingleCellExportResult:
    output_path: Path
    row_count: int
    column_count: int
    object_table: str
    mode: str
    shard_count: int


@dataclass(frozen=True)
class NativePycytominerResult:
    aggregated_path: Path
    annotated_path: Path
    normalized_path: Path
    feature_selected_path: Path
    aggregated_row_count: int
    aggregated_column_count: int
    annotated_row_count: int
    annotated_column_count: int
    normalized_row_count: int
    normalized_column_count: int
    feature_selected_row_count: int
    feature_selected_column_count: int


@dataclass(frozen=True)
class NativeProfileTableResult:
    output_path: Path
    row_count: int
    column_count: int


def build_image_manifest_native(
    config: ProjectConfig,
    output_path: Path | None = None,
) -> NativeManifestResult:
    backend_payload = config.load_profiling_backend_payload()
    paths_payload = backend_payload['paths']
    raw_config = backend_payload['raw_images']

    raw_dir = config.resolve_profiling_backend_path(paths_payload['raw_images_dir'])
    manifest_path = output_path.resolve() if output_path is not None else config.resolve_profiling_backend_path(paths_payload['manifest_csv'])
    valid_extensions = {suffix.lower() for suffix in raw_config['valid_extensions']}
    filename_pattern = re.compile(raw_config['filename_regex'])
    layout = raw_config.get('layout', 'generic')
    plate_dir_pattern = re.compile(raw_config.get('plate_dir_regex', '^$'))
    channel_map = raw_config.get('channel_map', {})

    rows: list[dict[str, str]] = []
    unmatched: list[str] = []
    for path in sorted(raw_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in valid_extensions:
            continue
        if layout == 'cpjump1':
            row = parse_cpjump1_path(
                path=path,
                filename_pattern=filename_pattern,
                plate_dir_pattern=plate_dir_pattern,
                channel_map=channel_map,
            )
        else:
            row = parse_generic_path(path=path, filename_pattern=filename_pattern)
        if row is None:
            unmatched.append(path.name)
            continue
        rows.append(row)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'Metadata_Plate',
        'Metadata_Well',
        'Metadata_Site',
        'Metadata_Channel',
        'FileName',
        'FilePath',
    ]
    if any('Metadata_ChannelNumber' in row for row in rows):
        fieldnames.insert(3, 'Metadata_ChannelNumber')

    with manifest_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return NativeManifestResult(
        output_path=manifest_path,
        row_count=len(rows),
        unmatched_files=sorted(unmatched),
    )


def validate_inputs_native(
    config: ProjectConfig,
    manifest_path: Path | None = None,
) -> NativeValidationResult:
    backend_payload = config.load_profiling_backend_payload()
    paths_payload = backend_payload['paths']
    metadata_payload = backend_payload['metadata']

    raw_dir = config.resolve_profiling_backend_path(paths_payload['raw_images_dir'])
    resolved_manifest_path = manifest_path.resolve() if manifest_path is not None else config.resolve_profiling_backend_path(paths_payload['manifest_csv'])
    plate_map_path = config.resolve_profiling_backend_path(paths_payload['plate_map_csv'])

    problems: list[str] = []
    raw_files = [path for path in raw_dir.rglob('*') if path.is_file()]

    if not raw_dir.exists():
        problems.append(f'Raw image directory is missing: {raw_dir}')
    if not raw_files:
        problems.append(f'No raw image files found under: {raw_dir}')
    if not resolved_manifest_path.exists():
        problems.append(f'Image manifest is missing: {resolved_manifest_path}. Run native or script manifest build first.')
    if not plate_map_path.exists():
        problems.append(f'Plate map is missing: {plate_map_path}. Copy the example and fill in metadata.')

    manifest_rows: list[dict[str, str]] = []
    if resolved_manifest_path.exists():
        manifest_rows = read_csv_rows(resolved_manifest_path)
        missing = missing_columns(manifest_rows, metadata_payload['required_manifest_columns'])
        if missing:
            problems.append(f"Manifest is missing required columns: {', '.join(missing)}")

    plate_rows: list[dict[str, str]] = []
    if plate_map_path.exists():
        plate_rows = read_csv_rows(plate_map_path)
        missing = missing_columns(plate_rows, metadata_payload['required_plate_map_columns'])
        if missing:
            problems.append(f"Plate map is missing required columns: {', '.join(missing)}")

    if manifest_rows and plate_rows:
        manifest_pairs = {
            (row['Metadata_Plate'], row['Metadata_Well'])
            for row in manifest_rows
            if row.get('Metadata_Plate') and row.get('Metadata_Well')
        }
        plate_pairs = {
            (row['Metadata_Plate'], row['Metadata_Well'])
            for row in plate_rows
            if row.get('Metadata_Plate') and row.get('Metadata_Well')
        }
        if not (manifest_pairs & plate_pairs):
            problems.append('No overlapping (Metadata_Plate, Metadata_Well) pairs between manifest and plate map.')

    return NativeValidationResult(
        raw_dir=raw_dir,
        raw_file_count=len(raw_files),
        manifest_path=resolved_manifest_path,
        plate_map_path=plate_map_path,
        problems=problems,
    )


def export_cellprofiler_to_singlecell_native(
    config: ProjectConfig,
    *,
    object_table: str | None = None,
    image_table_path: Path | None = None,
    object_table_path: Path | None = None,
    output_path: Path | None = None,
) -> NativeSingleCellExportResult:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('pandas is required in the active CellPainting-Claw runtime for native single-cell export.') from exc

    backend_payload = config.load_profiling_backend_payload()
    paths_payload = backend_payload['paths']
    cellprofiler_payload = backend_payload['cellprofiler']

    table_name = object_table or cellprofiler_payload['default_object_table']
    merge_key = cellprofiler_payload['image_join_key']
    resolved_output_path = output_path.resolve() if output_path is not None else config.resolve_profiling_backend_path(paths_payload['single_cell_output_csv_gz'])

    if image_table_path is not None and object_table_path is not None:
        merged_df = _export_single_cell_frame(
            pd=pd,
            image_table_path=image_table_path.resolve(),
            object_table_path=object_table_path.resolve(),
            table_name=table_name,
            merge_key=merge_key,
            metadata_columns_from_image=cellprofiler_payload['metadata_columns_from_image'],
        )
        mode = 'explicit'
        shard_count = 1
    else:
        default_image_path = config.resolve_profiling_backend_path(paths_payload['image_table_csv'])
        default_object_path = config.resolve_profiling_backend_path(paths_payload[f'{table_name.lower()}_table_csv'])
        if default_image_path.exists() and default_object_path.exists():
            merged_df = _export_single_cell_frame(
                pd=pd,
                image_table_path=default_image_path,
                object_table_path=default_object_path,
                table_name=table_name,
                merge_key=merge_key,
                metadata_columns_from_image=cellprofiler_payload['metadata_columns_from_image'],
            )
            mode = 'single'
            shard_count = 1
        else:
            sharded_root = config.profiling_backend_root / 'outputs' / 'sharded' / 'cellprofiler'
            shard_dirs = sorted(path for path in sharded_root.glob('shard_*') if path.is_dir())
            if not shard_dirs:
                raise FileNotFoundError(
                    f'Could not find single CellProfiler tables or shard tables for {table_name}. '
                    f'Checked: {default_image_path} and {sharded_root}'
                )
            shard_frames = []
            for shard_dir in shard_dirs:
                shard_image = shard_dir / 'Image.csv'
                shard_object = shard_dir / f'{table_name}.csv'
                if not shard_image.exists() or not shard_object.exists():
                    continue
                shard_frame = _export_single_cell_frame(
                    pd=pd,
                    image_table_path=shard_image,
                    object_table_path=shard_object,
                    table_name=table_name,
                    merge_key=merge_key,
                    metadata_columns_from_image=cellprofiler_payload['metadata_columns_from_image'],
                )
                # Match the historical sharded backend exactly: each shard is written
                # to single-cell CSV.gz and read back before the final concatenation.
                shard_frames.append(_roundtrip_single_cell_frame(pd=pd, dataframe=shard_frame))
            if not shard_frames:
                raise FileNotFoundError(f'No usable shard tables found under: {sharded_root}')
            merged_df = pd.concat(shard_frames, ignore_index=True)
            mode = 'sharded'
            shard_count = len(shard_frames)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(resolved_output_path, index=False, compression='gzip')
    return NativeSingleCellExportResult(
        output_path=resolved_output_path,
        row_count=int(len(merged_df)),
        column_count=int(len(merged_df.columns)),
        object_table=table_name,
        mode=mode,
        shard_count=shard_count,
    )


def run_pycytominer_native(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    single_cell_path: Path | None = None,
    plate_map_path: Path | None = None,
) -> NativePycytominerResult:
    resolved_paths = _resolve_pycytominer_stage_paths(config, output_dir=output_dir)
    aggregate_result = run_pycytominer_aggregate_native(
        config,
        output_path=resolved_paths['aggregated_path'],
        single_cell_path=single_cell_path,
    )
    annotate_result = run_pycytominer_annotate_native(
        config,
        output_path=resolved_paths['annotated_path'],
        aggregated_path=aggregate_result.output_path,
        plate_map_path=plate_map_path,
    )
    normalize_result = run_pycytominer_normalize_native(
        config,
        output_path=resolved_paths['normalized_path'],
        annotated_path=annotate_result.output_path,
    )
    feature_select_result = run_pycytominer_feature_select_native(
        config,
        output_path=resolved_paths['feature_selected_path'],
        normalized_path=normalize_result.output_path,
    )
    return NativePycytominerResult(
        aggregated_path=aggregate_result.output_path,
        annotated_path=annotate_result.output_path,
        normalized_path=normalize_result.output_path,
        feature_selected_path=feature_select_result.output_path,
        aggregated_row_count=aggregate_result.row_count,
        aggregated_column_count=aggregate_result.column_count,
        annotated_row_count=annotate_result.row_count,
        annotated_column_count=annotate_result.column_count,
        normalized_row_count=normalize_result.row_count,
        normalized_column_count=normalize_result.column_count,
        feature_selected_row_count=feature_select_result.row_count,
        feature_selected_column_count=feature_select_result.column_count,
    )


def _import_pycytominer_runtime():
    try:
        import numpy as np
        import numpy.typing as np_typing
        import pandas as pd

        np.typing = np_typing
        from pycytominer import aggregate, feature_select, normalize
    except ImportError as exc:
        raise RuntimeError(
            'pandas, pyarrow, numpy, scipy, and pycytominer are required in the active CellPainting-Claw runtime for native run-pycytominer.'
        ) from exc
    return pd, aggregate, normalize, feature_select


def _resolve_pycytominer_stage_paths(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    paths_payload = config.load_profiling_backend_payload()['paths']
    if output_dir is not None:
        resolved_output_dir = output_dir.resolve()
        return {
            'aggregated_path': resolved_output_dir / Path(paths_payload['aggregated_output_parquet']).name,
            'annotated_path': resolved_output_dir / Path(paths_payload['annotated_output_parquet']).name,
            'normalized_path': resolved_output_dir / Path(paths_payload['normalized_output_parquet']).name,
            'feature_selected_path': resolved_output_dir / Path(paths_payload['feature_selected_output_parquet']).name,
        }
    return {
        'aggregated_path': config.resolve_profiling_backend_path(paths_payload['aggregated_output_parquet']),
        'annotated_path': config.resolve_profiling_backend_path(paths_payload['annotated_output_parquet']),
        'normalized_path': config.resolve_profiling_backend_path(paths_payload['normalized_output_parquet']),
        'feature_selected_path': config.resolve_profiling_backend_path(paths_payload['feature_selected_output_parquet']),
    }


def run_pycytominer_aggregate_native(
    config: ProjectConfig,
    *,
    output_path: Path | None = None,
    single_cell_path: Path | None = None,
) -> NativeProfileTableResult:
    pd, aggregate, _, _ = _import_pycytominer_runtime()

    backend_payload = config.load_profiling_backend_payload()
    paths_payload = backend_payload['paths']
    pycytominer_payload = backend_payload['pycytominer']
    resolved_single_cell_path = single_cell_path.resolve() if single_cell_path is not None else config.resolve_profiling_backend_path(paths_payload['single_cell_output_csv_gz'])
    aggregated_path = output_path.resolve() if output_path is not None else _resolve_pycytominer_stage_paths(config)['aggregated_path']
    if not resolved_single_cell_path.exists():
        raise FileNotFoundError(f'Missing required input for native run-pycytominer aggregate: {resolved_single_cell_path}')
    aggregated_path.parent.mkdir(parents=True, exist_ok=True)
    single_cell_df = pd.read_csv(resolved_single_cell_path)
    aggregate(
        population_df=single_cell_df,
        strata=pycytominer_payload['aggregate']['strata'],
        features='infer',
        operation=pycytominer_payload['aggregate']['operation'],
        output_file=str(aggregated_path),
        output_type='parquet',
        compute_object_count=pycytominer_payload['aggregate']['compute_object_count'],
        object_feature=pycytominer_payload['aggregate']['object_feature'],
    )
    aggregate_df = pd.read_parquet(aggregated_path)
    return NativeProfileTableResult(
        output_path=aggregated_path,
        row_count=int(len(aggregate_df)),
        column_count=int(len(aggregate_df.columns)),
    )


def run_pycytominer_annotate_native(
    config: ProjectConfig,
    *,
    output_path: Path | None = None,
    aggregated_path: Path | None = None,
    plate_map_path: Path | None = None,
) -> NativeProfileTableResult:
    pd, _, _, _ = _import_pycytominer_runtime()
    backend_payload = config.load_profiling_backend_payload()
    paths_payload = backend_payload['paths']
    pycytominer_payload = backend_payload['pycytominer']
    resolved_aggregated_path = aggregated_path.resolve() if aggregated_path is not None else _resolve_pycytominer_stage_paths(config)['aggregated_path']
    resolved_plate_map_path = plate_map_path.resolve() if plate_map_path is not None else config.resolve_profiling_backend_path(paths_payload['plate_map_csv'])
    annotated_path = output_path.resolve() if output_path is not None else _resolve_pycytominer_stage_paths(config)['annotated_path']
    for required in [resolved_aggregated_path, resolved_plate_map_path]:
        if not required.exists():
            raise FileNotFoundError(f'Missing required input for native run-pycytominer annotate: {required}')
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_df = pd.read_parquet(resolved_aggregated_path)
    plate_map_df = pd.read_csv(resolved_plate_map_path)
    join_columns = pycytominer_payload['annotate']['join_on_columns']
    annotated_df = aggregate_df.merge(plate_map_df, on=join_columns, how='left', validate='many_to_one')
    annotated_df.to_parquet(annotated_path, index=False)
    return NativeProfileTableResult(
        output_path=annotated_path,
        row_count=int(len(annotated_df)),
        column_count=int(len(annotated_df.columns)),
    )


def run_pycytominer_normalize_native(
    config: ProjectConfig,
    *,
    output_path: Path | None = None,
    annotated_path: Path | None = None,
) -> NativeProfileTableResult:
    pd, _, normalize, _ = _import_pycytominer_runtime()
    backend_payload = config.load_profiling_backend_payload()
    pycytominer_payload = backend_payload['pycytominer']
    resolved_annotated_path = annotated_path.resolve() if annotated_path is not None else _resolve_pycytominer_stage_paths(config)['annotated_path']
    normalized_path = output_path.resolve() if output_path is not None else _resolve_pycytominer_stage_paths(config)['normalized_path']
    if not resolved_annotated_path.exists():
        raise FileNotFoundError(f'Missing required input for native run-pycytominer normalize: {resolved_annotated_path}')
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalize(
        profiles=str(resolved_annotated_path),
        features='infer',
        meta_features='infer',
        samples=pycytominer_payload['normalize']['samples'],
        method=pycytominer_payload['normalize']['method'],
        output_file=str(normalized_path),
        output_type='parquet',
    )
    normalized_df = pd.read_parquet(normalized_path)
    return NativeProfileTableResult(
        output_path=normalized_path,
        row_count=int(len(normalized_df)),
        column_count=int(len(normalized_df.columns)),
    )


def run_pycytominer_feature_select_native(
    config: ProjectConfig,
    *,
    output_path: Path | None = None,
    normalized_path: Path | None = None,
) -> NativeProfileTableResult:
    pd, _, _, feature_select = _import_pycytominer_runtime()
    backend_payload = config.load_profiling_backend_payload()
    pycytominer_payload = backend_payload['pycytominer']
    resolved_normalized_path = normalized_path.resolve() if normalized_path is not None else _resolve_pycytominer_stage_paths(config)['normalized_path']
    feature_selected_path = output_path.resolve() if output_path is not None else _resolve_pycytominer_stage_paths(config)['feature_selected_path']
    if not resolved_normalized_path.exists():
        raise FileNotFoundError(f'Missing required input for native run-pycytominer feature selection: {resolved_normalized_path}')
    feature_selected_path.parent.mkdir(parents=True, exist_ok=True)
    feature_select(
        profiles=str(resolved_normalized_path),
        features='infer',
        operation=pycytominer_payload['feature_select']['operations'],
        na_cutoff=pycytominer_payload['feature_select']['na_cutoff'],
        corr_threshold=pycytominer_payload['feature_select']['corr_threshold'],
        output_file=str(feature_selected_path),
        output_type='parquet',
    )
    feature_selected_df = pd.read_parquet(feature_selected_path)
    return NativeProfileTableResult(
        output_path=feature_selected_path,
        row_count=int(len(feature_selected_df)),
        column_count=int(len(feature_selected_df.columns)),
    )


def _roundtrip_single_cell_frame(*, pd, dataframe):
    buffer = io.BytesIO()
    dataframe.to_csv(buffer, index=False, compression='gzip')
    buffer.seek(0)
    return pd.read_csv(buffer, compression='gzip')


def _export_single_cell_frame(
    *,
    pd,
    image_table_path: Path,
    object_table_path: Path,
    table_name: str,
    merge_key: str,
    metadata_columns_from_image: list[str],
):
    image_df = pd.read_csv(image_table_path)
    object_df = pd.read_csv(object_table_path)
    object_df = add_compartment_prefixes(object_df, table_name, merge_key)

    metadata_columns = [column for column in metadata_columns_from_image if column in image_df.columns]
    if merge_key not in metadata_columns:
        metadata_columns.insert(0, merge_key)

    merged_df = object_df.merge(
        image_df[metadata_columns].drop_duplicates(),
        on=merge_key,
        how='left',
        validate='many_to_one',
    )
    if 'ObjectNumber' in merged_df.columns and 'Metadata_ObjectNumber' not in merged_df.columns:
        merged_df = merged_df.rename(columns={'ObjectNumber': 'Metadata_ObjectNumber'})
    return merged_df


def add_compartment_prefixes(dataframe, table_name: str, merge_key: str):
    known_prefixes = ('Cells_', 'Cytoplasm_', 'Nuclei_', 'Metadata_')
    renamed_columns: dict[str, str] = {}
    for column in dataframe.columns:
        if column in {merge_key, 'TableNumber', 'ObjectNumber', 'Metadata_ObjectNumber'}:
            continue
        if column.startswith(known_prefixes):
            continue
        renamed_columns[column] = f'{table_name}_{column}'
    return dataframe.rename(columns=renamed_columns)


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def missing_columns(rows: list[dict[str, str]], required_columns: Iterable[str]) -> list[str]:
    if not rows:
        return list(required_columns)
    fieldnames = set(rows[0].keys())
    return [column for column in required_columns if column not in fieldnames]


def row_index_to_letter(row_index: int) -> str:
    if row_index < 1:
        raise ValueError(f'Row index must be >= 1, received {row_index}')
    return chr(ord('A') + row_index - 1)


def find_plate_name(path: Path, plate_dir_pattern: re.Pattern[str]) -> str | None:
    for part in path.parts:
        if plate_dir_pattern.match(part):
            return part
    return None


def parse_cpjump1_path(
    path: Path,
    filename_pattern: re.Pattern[str],
    plate_dir_pattern: re.Pattern[str],
    channel_map: dict[str, str],
) -> dict[str, str] | None:
    match = filename_pattern.match(path.name)
    if match is None:
        return None
    plate = find_plate_name(path, plate_dir_pattern)
    if plate is None:
        return None
    groups = match.groupdict()
    row_index = int(groups['row'])
    column_index = int(groups['column'])
    well = f"{row_index_to_letter(row_index)}{column_index:02d}"
    channel_number = groups['channel_number']
    return {
        'Metadata_Plate': plate,
        'Metadata_Well': well,
        'Metadata_Site': str(int(groups['site'])),
        'Metadata_ChannelNumber': channel_number,
        'Metadata_Channel': channel_map.get(channel_number, f'ch{channel_number}'),
        'FileName': path.name,
        'FilePath': str(path.resolve()),
    }


def parse_generic_path(path: Path, filename_pattern: re.Pattern[str]) -> dict[str, str] | None:
    match = filename_pattern.match(path.name)
    if match is None:
        return None
    groups = match.groupdict()
    return {
        'Metadata_Plate': groups['plate'],
        'Metadata_Well': groups['well'],
        'Metadata_Site': groups['site'],
        'Metadata_Channel': groups['channel'],
        'FileName': path.name,
        'FilePath': str(path.resolve()),
    }
