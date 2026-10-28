from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tifffile

from cellpaint_pipeline.ports import build_pythonpath_env
from cellpaint_pipeline.runner import run_command


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


CHANNEL_METADATA_MAP = {
    'DNA': 'dna_path',
    'RNA': 'rna_path',
    'ER': 'er_path',
    'AGP': 'agp_path',
    'Mito': 'mito_path',
}
DEFAULT_LABEL_CANDIDATES = [
    'Metadata_pert_iname',
    'Metadata_pert_name',
    'Metadata_gene',
    'Metadata_Treatment',
    'Treatment',
    'pert_name',
    'Metadata_Well',
]
DEFAULT_REPLICATE_CANDIDATES = [
    'Metadata_pert_name_replicate',
    'Metadata_replicate',
    'Replicate',
    'Metadata_Well',
]
DEFAULT_CONTROL_CANDIDATES = ['dmso', 'negcon', 'negative_control', 'mock', 'control']
PROJECT_MANIFEST_NAME = 'project_manifest.json'
DEFAULT_DEEPPROFILER_CHECKPOINT = 'Cell_Painting_CNN_v1.hdf5'
DEFAULT_DEEPPROFILER_CHECKPOINT_URL = 'https://zenodo.org/records/7114558/files/Cell_Painting_CNN_v1.hdf5?download=1'


@dataclass(frozen=True)
class DeepProfilerProjectResult:
    project_root: Path
    manifest_path: Path
    config_path: Path
    metadata_path: Path
    locations_root: Path
    field_count: int
    location_file_count: int
    image_width: int
    image_height: int
    image_bits: int
    image_format: str
    experiment_name: str
    label_field: str
    control_value: str
    export_root: Path
    source_label: str
    checkpoint_filename: str


@dataclass(frozen=True)
class DeepProfilerProfileResult:
    project_root: Path
    manifest_path: Path
    config_path: Path
    metadata_path: Path
    experiment_name: str
    feature_dir: Path
    checkpoint_dir: Path
    checkpoint_path: Path | None
    log_path: Path | None
    command: list[str]
    returncode: int


def build_deepprofiler_project(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    workflow_root: Path | None = None,
    export_root: Path | None = None,
    experiment_name: str | None = None,
    config_filename: str | None = None,
    metadata_filename: str | None = None,
) -> DeepProfilerProjectResult:
    runtime = dict(config.deepprofiler_runtime)
    resolved_project_root = (output_dir or config.deepprofiler_project_root).expanduser().resolve()
    resolved_export_root = _resolve_export_root(config, workflow_root=workflow_root, export_root=export_root)
    export_manifest_path = resolved_export_root / 'manifest.json'
    if not export_manifest_path.exists():
        raise FileNotFoundError(f'DeepProfiler export manifest not found: {export_manifest_path}')

    export_manifest = json.loads(export_manifest_path.read_text(encoding='utf-8'))
    field_metadata_path = Path(export_manifest['field_metadata_csv']).expanduser().resolve()
    source_load_data_path = Path(export_manifest['source_load_data_csv']).expanduser().resolve()
    source_label = str(export_manifest.get('source_label', 'deepprofiler-export'))

    field_rows = _read_csv_rows(field_metadata_path)
    if not field_rows:
        raise ValueError(f'No field rows found in {field_metadata_path}')

    load_data_lookup = _load_data_lookup(source_load_data_path)
    merged_rows = _merge_field_and_load_data_rows(field_rows, load_data_lookup)

    first_image_path = _first_image_path(merged_rows[0])
    image_height, image_width, image_bits, image_format = _infer_image_properties(first_image_path)

    resolved_experiment_name = str(runtime.get('experiment_name') or experiment_name or 'imagenet_pretrained_cellpainting')
    resolved_config_filename = str(runtime.get('config_filename') or config_filename or 'profile_config.json')
    resolved_metadata_filename = str(runtime.get('metadata_filename') or metadata_filename or 'index.csv')
    box_size = int(runtime.get('box_size', 128))
    mask_objects = bool(runtime.get('mask_objects', False))
    if mask_objects:
        raise NotImplementedError('mask_objects=true is not implemented yet for DeepProfiler project generation. Use the current unmasked project builder.')

    inputs_root = resolved_project_root / 'inputs'
    config_root = inputs_root / 'config'
    metadata_root = inputs_root / 'metadata'
    locations_root = inputs_root / 'locations'
    images_root = inputs_root / 'images'
    outputs_root = resolved_project_root / 'outputs' / resolved_experiment_name
    features_root = outputs_root / 'features'
    checkpoint_root = outputs_root / 'checkpoint'

    for path in [config_root, metadata_root, locations_root, images_root, features_root, checkpoint_root]:
        path.mkdir(parents=True, exist_ok=True)

    _ensure_deepprofiler_plugins_link(resolved_project_root)

    metadata_rows = []
    extra_columns: set[str] = set()
    location_file_count = 0
    for row in merged_rows:
        plate = row['Metadata_Plate']
        well = row['Metadata_Well']
        site = _normalize_site_token(row['Metadata_Site'])

        destination_location_path = locations_root / plate / f'{well}-{site}-Nuclei.csv'
        destination_location_path.parent.mkdir(parents=True, exist_ok=True)
        _write_deepprofiler_locations_csv(Path(row['nuclei_locations_csv']).expanduser().resolve(), destination_location_path)
        location_file_count += 1

        metadata_row = {
            'Metadata_Plate': plate,
            'Metadata_Well': well,
            'Metadata_Site': site,
        }
        for channel_name, source_column in CHANNEL_METADATA_MAP.items():
            metadata_row[channel_name] = row[source_column]
        for key, value in row.items():
            if key in metadata_row or key in CHANNEL_METADATA_MAP.values():
                continue
            if key in {'outline_path', 'nuclei_locations_csv', 'ImageNumber'}:
                continue
            if value in {'', None}:
                continue
            metadata_row[key] = value
            extra_columns.add(key)
        metadata_rows.append(metadata_row)

    label_field = str(runtime.get('label_field') or _infer_first_available_column(metadata_rows, DEFAULT_LABEL_CANDIDATES))
    control_value = str(runtime.get('control_value') or _infer_control_value(metadata_rows, label_field))
    replicate_field = str(runtime.get('replicate_field') or _infer_first_available_column(metadata_rows, DEFAULT_REPLICATE_CANDIDATES))

    metadata_path = metadata_root / resolved_metadata_filename
    metadata_columns = [
        'Metadata_Plate',
        'Metadata_Well',
        'Metadata_Site',
        'DNA',
        'RNA',
        'ER',
        'AGP',
        'Mito',
    ]
    extra_columns_ordered = [
        column
        for column in sorted(extra_columns)
        if column not in metadata_columns
    ]
    _write_csv(metadata_path, metadata_columns + extra_columns_ordered, metadata_rows)

    config_path = config_root / resolved_config_filename
    config_payload = _build_deepprofiler_config(
        runtime=runtime,
        label_field=label_field,
        control_value=control_value,
        image_width=image_width,
        image_height=image_height,
        image_bits=image_bits,
        image_format=image_format,
        box_size=box_size,
        mask_objects=mask_objects,
        replicate_field=replicate_field,
    )
    config_path.write_text(json.dumps(config_payload, indent=2) + chr(10), encoding='utf-8')

    manifest_path = resolved_project_root / PROJECT_MANIFEST_NAME
    manifest = {
        'implementation': 'cellpaint_pipeline.deepprofiler_project',
        'project_root': str(resolved_project_root),
        'export_root': str(resolved_export_root),
        'export_manifest_path': str(export_manifest_path),
        'config_path': str(config_path),
        'metadata_path': str(metadata_path),
        'locations_root': str(locations_root),
        'experiment_name': resolved_experiment_name,
        'field_count': len(metadata_rows),
        'location_file_count': location_file_count,
        'image_width': image_width,
        'image_height': image_height,
        'image_bits': image_bits,
        'image_format': image_format,
        'label_field': label_field,
        'control_value': control_value,
        'replicate_field': replicate_field,
        'checkpoint_filename': _checkpoint_filename_for_project(runtime),
        'checkpoint_url': str(runtime.get('checkpoint_url', DEFAULT_DEEPPROFILER_CHECKPOINT_URL)),
        'source_label': source_label,
        'channels': list(CHANNEL_METADATA_MAP),
        'notes': [
            'Images are referenced directly from the source dataset; no image copies are made.',
            'Locations are converted to the DeepProfiler two-column nuclei centroid format.',
            'This project is configured for unmasked single-cell-in-context profiling by default.',
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + chr(10), encoding='utf-8')

    return DeepProfilerProjectResult(
        project_root=resolved_project_root,
        manifest_path=manifest_path,
        config_path=config_path,
        metadata_path=metadata_path,
        locations_root=locations_root,
        field_count=len(metadata_rows),
        location_file_count=location_file_count,
        image_width=image_width,
        image_height=image_height,
        image_bits=image_bits,
        image_format=image_format,
        experiment_name=resolved_experiment_name,
        label_field=label_field,
        control_value=control_value,
        export_root=resolved_export_root,
        source_label=source_label,
        checkpoint_filename=_checkpoint_filename_for_project(runtime),
    )


def run_deepprofiler_profile(
    config: ProjectConfig,
    *,
    project_root: Path,
    experiment_name: str | None = None,
    config_filename: str | None = None,
    metadata_filename: str | None = None,
    gpu: str | None = None,
) -> DeepProfilerProfileResult:
    resolved_project_root = project_root.expanduser().resolve()
    manifest_path = resolved_project_root / PROJECT_MANIFEST_NAME
    manifest_payload: dict[str, Any] = {}
    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding='utf-8'))

    runtime = dict(config.deepprofiler_runtime)
    resolved_experiment_name = str(
        experiment_name
        or manifest_payload.get('experiment_name')
        or runtime.get('experiment_name')
        or 'imagenet_pretrained_cellpainting'
    )
    resolved_config_filename = str(
        config_filename
        or runtime.get('config_filename')
        or Path(str(manifest_payload.get('config_path', 'profile_config.json'))).name
    )
    resolved_metadata_filename = str(
        metadata_filename
        or runtime.get('metadata_filename')
        or Path(str(manifest_payload.get('metadata_path', 'index.csv'))).name
    )
    deepprofiler_executable = runtime.get('deepprofiler_executable')
    command = _resolve_deepprofiler_command(config, deepprofiler_executable)
    command.extend([
        f'--root={resolved_project_root}',
        f'--config={resolved_config_filename}',
        f'--metadata={resolved_metadata_filename}',
        f'--exp={resolved_experiment_name}',
    ])
    resolved_gpu = gpu if gpu is not None else runtime.get('gpu')
    if resolved_gpu not in {None, '', False}:
        command.append(f'--gpu={resolved_gpu}')
    command.append('profile')

    checkpoint_path = _stage_deepprofiler_checkpoint(
        project_root=resolved_project_root,
        runtime=runtime,
        experiment_name=resolved_experiment_name,
    )

    execution = run_command(
        command,
        cwd=resolved_project_root,
        log_dir=config.log_root / 'deepprofiler',
        label='deepprofiler_profile',
        env=_build_deepprofiler_runtime_env(command),
    )

    return DeepProfilerProfileResult(
        project_root=resolved_project_root,
        manifest_path=manifest_path,
        config_path=resolved_project_root / 'inputs' / 'config' / resolved_config_filename,
        metadata_path=resolved_project_root / 'inputs' / 'metadata' / resolved_metadata_filename,
        experiment_name=resolved_experiment_name,
        feature_dir=resolved_project_root / 'outputs' / resolved_experiment_name / 'features',
        checkpoint_dir=resolved_project_root / 'outputs' / resolved_experiment_name / 'checkpoint',
        checkpoint_path=checkpoint_path,
        log_path=execution.log_path,
        command=execution.command,
        returncode=execution.returncode,
    )


def infer_deepprofiler_export_root_from_workflow_root(workflow_root: Path) -> Path:
    return workflow_root.expanduser().resolve() / 'deepprofiler_export'


def _resolve_export_root(
    config: ProjectConfig,
    *,
    workflow_root: Path | None,
    export_root: Path | None,
) -> Path:
    if export_root is not None:
        return export_root.expanduser().resolve()
    if workflow_root is not None:
        return infer_deepprofiler_export_root_from_workflow_root(workflow_root)
    return config.deepprofiler_export_root.expanduser().resolve()


def _resolve_deepprofiler_command(config: ProjectConfig, deepprofiler_executable: Any) -> list[str]:
    if deepprofiler_executable:
        candidate = Path(str(deepprofiler_executable)).expanduser()
        if ('/' in str(deepprofiler_executable) or str(deepprofiler_executable).startswith('.')):
            if candidate.exists():
                return [str(candidate.resolve())]
            raise FileNotFoundError(
                f'DeepProfiler executable not found: {candidate}. Install DeepProfiler in the active CellPainting-Claw runtime or clear deepprofiler_runtime.deepprofiler_executable to use python -m deepprofiler.'
            )
        return [str(deepprofiler_executable)]

    python_path = Path(config.python_executable).expanduser().resolve()
    sibling_binary = python_path.with_name('deepprofiler')
    if sibling_binary.exists():
        return [str(sibling_binary)]
    return [str(python_path), '-m', 'deepprofiler']


def _build_deepprofiler_runtime_env(command: list[str]) -> dict[str, str] | None:
    if len(command) < 3 or command[1] != '-m' or command[2] != 'deepprofiler':
        return None

    spec = importlib.util.find_spec('deepprofiler')
    origin = getattr(spec, 'origin', None) if spec is not None else None
    if not origin:
        return None

    package_root = Path(str(origin)).expanduser().resolve().parents[1]
    return build_pythonpath_env(package_root)


def _ensure_deepprofiler_plugins_link(project_root: Path) -> None:
    try:
        import deepprofiler  # type: ignore
    except ImportError:
        return

    source_plugins = Path(deepprofiler.__file__).resolve().parents[1] / 'plugins'
    if not source_plugins.exists():
        return

    target_plugins = project_root / 'plugins'
    if target_plugins.exists() or target_plugins.is_symlink():
        return
    target_plugins.symlink_to(source_plugins, target_is_directory=True)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _load_data_lookup(load_data_path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not load_data_path.exists():
        return {}
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in _read_csv_rows(load_data_path):
        key = (
            row.get('Metadata_Plate', ''),
            row.get('Metadata_Well', ''),
            _normalize_site_token(row.get('Metadata_Site', '')),
        )
        lookup[key] = row
    return lookup


def _merge_field_and_load_data_rows(
    field_rows: list[dict[str, str]],
    load_data_lookup: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, str]]:
    merged_rows = []
    for row in field_rows:
        key = (
            row.get('Metadata_Plate', ''),
            row.get('Metadata_Well', ''),
            _normalize_site_token(row.get('Metadata_Site', '')),
        )
        merged = dict(load_data_lookup.get(key, {}))
        merged.update(row)
        merged['Metadata_Site'] = key[2]
        merged_rows.append(merged)
    return merged_rows


def _first_image_path(row: dict[str, str]) -> Path:
    for column in CHANNEL_METADATA_MAP.values():
        value = row.get(column)
        if value:
            return Path(value).expanduser().resolve()
    raise ValueError('No image path found in DeepProfiler field metadata row.')


def _infer_image_properties(image_path: Path) -> tuple[int, int, int, str]:
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')
    with tifffile.TiffFile(str(image_path)) as handle:
        page = handle.pages[0]
        shape = page.shape
        dtype = page.dtype
    if len(shape) < 2:
        raise ValueError(f'Unsupported image shape for DeepProfiler project: {shape}')
    height = int(shape[-2])
    width = int(shape[-1])
    bits = int(dtype.itemsize * 8)
    image_format = image_path.suffix.lstrip('.').lower() or 'tif'
    return height, width, bits, image_format


def _write_deepprofiler_locations_csv(source_path: Path, destination_path: Path) -> None:
    with source_path.open('r', encoding='utf-8', newline='') as source_handle:
        reader = csv.DictReader(source_handle)
        rows = list(reader)

    with destination_path.open('w', encoding='utf-8', newline='') as destination_handle:
        writer = csv.DictWriter(
            destination_handle,
            fieldnames=['Nuclei_Location_Center_X', 'Nuclei_Location_Center_Y'],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'Nuclei_Location_Center_X': row.get('Nuclei_Location_Center_X') or row.get('Location_Center_X') or '',
                'Nuclei_Location_Center_Y': row.get('Nuclei_Location_Center_Y') or row.get('Location_Center_Y') or '',
            })


def _infer_first_available_column(rows: list[dict[str, str]], candidates: list[str]) -> str:
    for candidate in candidates:
        if all(candidate in row for row in rows):
            return candidate
    return candidates[-1]


def _infer_control_value(rows: list[dict[str, str]], label_field: str) -> str:
    values = [str(row.get(label_field, '')).strip() for row in rows if str(row.get(label_field, '')).strip()]
    if not values:
        return 'unknown_control'
    lowercase_map = {value.lower(): value for value in values}
    for candidate in DEFAULT_CONTROL_CANDIDATES:
        if candidate in lowercase_map:
            return lowercase_map[candidate]
    return sorted(set(values))[0]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, '') for fieldname in fieldnames})


def _build_deepprofiler_config(
    *,
    runtime: dict[str, Any],
    label_field: str,
    control_value: str,
    image_width: int,
    image_height: int,
    image_bits: int,
    image_format: str,
    box_size: int,
    mask_objects: bool,
    replicate_field: str,
) -> dict[str, Any]:
    batch_size = int(runtime.get('batch_size', 64))
    checkpoint = _checkpoint_filename_for_project(runtime)
    checkpoint_mode = checkpoint != 'None'
    feature_layer = str(runtime.get('feature_layer', 'block6a_activation' if checkpoint_mode else 'avg_pool'))
    model_name = str(runtime.get('model_name', 'efficientnet'))
    crop_generator = str(runtime.get('crop_generator', 'crop_generator' if checkpoint_mode else 'repeat_channel_crop_generator'))
    use_pretrained_input_size_value = runtime.get('use_pretrained_input_size', False if checkpoint_mode else 224)
    conv_blocks = int(runtime.get('conv_blocks', 0))
    learning_rate = float(runtime.get('learning_rate', 0.0001))
    label_smoothing = float(runtime.get('label_smoothing', 0.0))
    validation_batch_size = int(runtime.get('validation_batch_size', batch_size))
    initialization = str(runtime.get('initialization', 'random' if checkpoint_mode else 'ImageNet'))

    profile_payload = {
        'feature_layer': feature_layer,
        'checkpoint': checkpoint,
        'batch_size': batch_size,
    }
    if use_pretrained_input_size_value not in {False, None, '', 0}:
        profile_payload['use_pretrained_input_size'] = int(use_pretrained_input_size_value)

    return {
        'dataset': {
            'metadata': {
                'label_field': label_field,
                'control_value': control_value,
                'control_id': control_value,
                'replicate_field': replicate_field,
            },
            'images': {
                'channels': list(CHANNEL_METADATA_MAP),
                'file_format': image_format,
                'bits': image_bits,
                'width': image_width,
                'height': image_height,
            },
            'locations': {
                'mode': 'single_cells',
                'box_size': box_size,
                'view_size': box_size,
                'mask_objects': mask_objects,
            },
        },
        'prepare': {
            'compression': {
                'implement': False,
                'scaling_factor': 1.0,
            },
        },
        'profile': profile_payload,
        'train': {
            'partition': {
                'targets': [label_field],
                'split_field': replicate_field,
                'training': [],
                'validation': [],
            },
            'model': {
                'name': model_name,
                'crop_generator': crop_generator,
                'augmentations': False,
                'initialization': initialization,
                'params': {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'conv_blocks': conv_blocks,
                    'label_smoothing': label_smoothing,
                },
            },
            'sampling': {
                'factor': 1.0,
                'cache_size': batch_size,
                'workers': 1,
                'alpha': 0.2,
            },
            'validation': {
                'batch_size': validation_batch_size,
                'top_k': 1,
                'sample_first_crops': False,
            },
        },
    }


def _checkpoint_filename_for_project(runtime: dict[str, Any]) -> str:
    checkpoint_value = str(runtime.get('checkpoint', DEFAULT_DEEPPROFILER_CHECKPOINT)).strip()
    if checkpoint_value in {'', 'None'}:
        return 'None'
    return Path(checkpoint_value).name


def _stage_deepprofiler_checkpoint(
    *,
    project_root: Path,
    runtime: dict[str, Any],
    experiment_name: str,
) -> Path | None:
    checkpoint_filename = _checkpoint_filename_for_project(runtime)
    if checkpoint_filename == 'None':
        return None

    checkpoint_dir = project_root / 'outputs' / experiment_name / 'checkpoint'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / checkpoint_filename

    source_path_value = str(runtime.get('checkpoint_source_path', '')).strip()
    if source_path_value:
        source_path = Path(source_path_value).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f'DeepProfiler checkpoint_source_path not found: {source_path}')
        shutil.copy2(source_path, destination)
        return destination

    direct_path_value = str(runtime.get('checkpoint', '')).strip()
    if direct_path_value and direct_path_value not in {'', 'None'}:
        direct_path = Path(direct_path_value).expanduser()
        if direct_path.exists():
            shutil.copy2(direct_path.resolve(), destination)
            return destination

    checkpoint_url = str(runtime.get('checkpoint_url', DEFAULT_DEEPPROFILER_CHECKPOINT_URL)).strip()
    if not checkpoint_url:
        if destination.exists():
            return destination
        raise FileNotFoundError(
            f'DeepProfiler checkpoint missing at {destination} and no checkpoint_source_path/checkpoint_url was provided.'
        )

    expected_size = _fetch_remote_content_length(checkpoint_url)
    if destination.exists() and _checkpoint_matches_size(destination, expected_size):
        return destination

    last_error: Exception | None = None
    for attempt in range(1, 4):
        temporary_destination = destination.with_suffix(destination.suffix + '.part')
        if temporary_destination.exists():
            temporary_destination.unlink()
        try:
            with urllib.request.urlopen(checkpoint_url) as response, temporary_destination.open('wb') as handle:
                shutil.copyfileobj(response, handle)
            _validate_checkpoint_size(temporary_destination, expected_size, checkpoint_url)
            temporary_destination.replace(destination)
            _validate_checkpoint_size(destination, expected_size, checkpoint_url)
            return destination
        except Exception as exc:
            last_error = exc
            if temporary_destination.exists():
                temporary_destination.unlink()
            if destination.exists() and not _checkpoint_matches_size(destination, expected_size):
                destination.unlink()
            if attempt < 3:
                time.sleep(attempt)

    assert last_error is not None
    raise last_error


def _fetch_remote_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method='HEAD')
    try:
        with urllib.request.urlopen(request) as response:
            value = response.headers.get('Content-Length')
    except Exception:
        return None
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _checkpoint_matches_size(path: Path, expected_size: int | None) -> bool:
    if not path.exists():
        return False
    if expected_size is None:
        return path.stat().st_size > 0
    return path.stat().st_size == expected_size


def _validate_checkpoint_size(path: Path, expected_size: int | None, checkpoint_url: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f'DeepProfiler checkpoint download did not create file: {path}')
    actual_size = path.stat().st_size
    if actual_size <= 0:
        raise OSError(f'DeepProfiler checkpoint download produced an empty file: {path}')
    if expected_size is not None and actual_size != expected_size:
        raise OSError(
            f'DeepProfiler checkpoint download size mismatch for {checkpoint_url}: expected {expected_size} bytes, got {actual_size} bytes.'
        )


def _normalize_site_token(value: Any) -> str:
    token = str(value).strip()
    if not token:
        return token
    try:
        return str(int(float(token)))
    except ValueError:
        return token
