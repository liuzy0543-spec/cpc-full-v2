from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from cellpaint_pipeline.cppipe import resolve_cppipe_selection
from cellpaint_pipeline.ports import BaseDirFileLocator, FileLocatorPort, validate_local_files
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


ARTICLE_PSEUDOCOLORS = {
    'OrigMito': (1.0, 0.0, 0.0),
    'OrigAGP': (1.0, 0.55, 0.0),
    'OrigRNA': (1.0, 1.0, 0.0),
    'OrigER': (0.0, 1.0, 0.0),
    'OrigDNA': (0.0, 1.0, 1.0),
}

REQUIRED_CHANNELS = [
    'OrigMito',
    'OrigAGP',
    'OrigRNA',
    'OrigER',
    'OrigDNA',
    'OrigBrightfield',
    'OrigHighZBF',
    'OrigLowZBF',
]


MODULE_INSERTION = """
ConvertObjectsToImage:[module_num:32|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select the input objects:Nuclei
    Name the output image:NucleiLabels
    Select the color format:uint16
    Select the colormap:Default

ConvertObjectsToImage:[module_num:33|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select the input objects:Cells
    Name the output image:CellLabels
    Select the color format:uint16
    Select the colormap:Default

SaveImages:[module_num:34|svn_version:'Unknown'|variable_revision_number:15|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select the type of image to save:Image
    Select the image to save:NucleiLabels
    Select method for constructing file names:Single name
    Select image name for file prefix:OrigDNA
    Enter single file name:\g<Plate>_\g<Well>_s\g<Site>--nuclei_labels
    Number of digits:4
    Append a suffix to the image file name?:No
    Text to append to the image name:
    Saved file format:tiff
    Output file location:Default Output Folder sub-folder|labels
    Image bit depth:16-bit integer
    Overwrite existing files without warning?:Yes
    When to save:Every cycle
    Record the file and path information to the saved image?:Yes
    Create subfolders in the output folder?:No
    Base image folder:Default Input Folder
    How to save the series:T (Time)

SaveImages:[module_num:35|svn_version:'Unknown'|variable_revision_number:15|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Select the type of image to save:Image
    Select the image to save:CellLabels
    Select method for constructing file names:Single name
    Select image name for file prefix:OrigDNA
    Enter single file name:\g<Plate>_\g<Well>_s\g<Site>--cell_labels
    Number of digits:4
    Append a suffix to the image file name?:No
    Text to append to the image name:
    Saved file format:tiff
    Output file location:Default Output Folder sub-folder|labels
    Image bit depth:16-bit integer
    Overwrite existing files without warning?:Yes
    When to save:Every cycle
    Record the file and path information to the saved image?:Yes
    Create subfolders in the output folder?:No
    Base image folder:Default Input Folder
    How to save the series:T (Time)
""".strip()


@dataclass(frozen=True)
class NativeSegmentationLoadDataResult:
    output_path: Path
    row_count: int
    plate_count: int
    well_count: int
    site_count: int


@dataclass(frozen=True)
class NativeMaskExportPipelineResult:
    output_path: Path
    module_count: int
    source_cppipe_path: Path
    selected_via: str
    execution_mode: str


@dataclass(frozen=True)
class NativeSamplePreviewsResult:
    output_dir: Path
    generated_count: int
    skipped_existing: int
    field_count: int


@dataclass(frozen=True)
class NativePNGPreviewsResult:
    mode: str
    manifest_path: Path
    output_dir: Path
    preview_count: int
    worker_count: int
    chunk_size: int


@dataclass(frozen=True)
class NativeSingleCellCropsResult:
    mode: str
    crops_dir: Path
    manifest_path: Path
    crop_count: int
    worker_count: int
    background_masked: bool


@dataclass(frozen=True)
class SegmentationSummaryResult:
    load_data_path: Path
    load_data_rows: int
    mask_output_dir: Path
    image_table_path: Path
    image_row_count: int
    cells_table_path: Path
    cell_row_count: int
    nuclei_table_path: Path
    nuclei_row_count: int
    label_dir: Path
    label_file_count: int
    outline_dir: Path
    outline_file_count: int
    sample_previews_dir: Path
    sample_preview_count: int
    masked_manifest_path: Path
    masked_manifest_rows: int
    masked_image_stack_count: int
    masked_cell_mask_count: int
    masked_nuclei_mask_count: int
    masked_preview_count: int
    unmasked_manifest_path: Path
    unmasked_manifest_rows: int
    unmasked_image_stack_count: int
    unmasked_cell_mask_count: int
    unmasked_nuclei_mask_count: int
    unmasked_preview_count: int
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def prepare_segmentation_load_data_native(
    config: ProjectConfig,
    output_path: Path | None = None,
    *,
    source_base_dir: Path | None = None,
) -> NativeSegmentationLoadDataResult:
    """Build the load-data table used by the segmentation pipeline.

    ``source_base_dir`` only affects how the ``PathName_*`` columns of the
    source table are turned into filesystem paths.  ``None`` keeps the
    historical behaviour of resolving them against the process working
    directory, so existing callers see no change.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('pandas is required in the active CellPainting-Claw runtime for native segmentation load-data preparation.') from exc

    backend_payload = config.load_segmentation_backend_payload()
    paths_payload = backend_payload['paths']
    subset_payload = backend_payload['subset']

    source_csv = config.resolve_segmentation_backend_path(paths_payload['source_load_data_csv'])
    resolved_output_path = output_path.resolve() if output_path is not None else config.resolve_segmentation_backend_path(paths_payload['load_data_csv'])

    source_df = pd.read_csv(source_csv)
    subset_df = _filter_segmentation_rows(source_df, subset_payload).drop_duplicates().copy()
    subset_df = subset_df.sort_values(['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']).reset_index(drop=True)
    if subset_df.empty:
        raise ValueError('The configured segmentation subset did not match any rows.')

    _validate_local_files(
        subset_df,
        locator=BaseDirFileLocator(source_base_dir) if source_base_dir is not None else None,
    )

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(resolved_output_path, index=False)
    return NativeSegmentationLoadDataResult(
        output_path=resolved_output_path,
        row_count=int(len(subset_df)),
        plate_count=int(subset_df['Metadata_Plate'].nunique()),
        well_count=int(subset_df[['Metadata_Plate', 'Metadata_Well']].drop_duplicates().shape[0]),
        site_count=int(subset_df[['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']].drop_duplicates().shape[0]),
    )


def build_mask_export_pipeline_native(
    config: ProjectConfig,
    output_path: Path | None = None,
) -> NativeMaskExportPipelineResult:
    backend_payload = config.load_segmentation_backend_payload()
    paths_payload = backend_payload['paths']

    selection = resolve_cppipe_selection(config, 'segmentation')
    resolved_output_path = output_path.resolve() if output_path is not None else config.resolve_segmentation_backend_path(paths_payload['mask_export_pipeline'])

    if not selection.exists:
        raise FileNotFoundError(f'Selected segmentation .cppipe does not exist: {selection.cppipe_path}')

    pipeline_text = selection.cppipe_path.read_text(encoding='utf-8')
    if selection.execution_mode == 'derive-mask-export':
        pipeline_text = pipeline_text.replace('ModuleCount:33', 'ModuleCount:37', 1)
        pipeline_text = re.sub(r'module_num:(\d+)', _renumber_module, pipeline_text)

        anchor = 'ExportToSpreadsheet:[module_num:36|'
        if anchor not in pipeline_text:
            raise ValueError('Could not find the ExportToSpreadsheet anchor after renumbering.')
        pipeline_text = pipeline_text.replace(anchor, f'{MODULE_INSERTION}\n\n{anchor}', 1)

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(pipeline_text, encoding='utf-8')
    return NativeMaskExportPipelineResult(
        output_path=resolved_output_path,
        module_count=_extract_module_count(pipeline_text),
        source_cppipe_path=selection.cppipe_path,
        selected_via=selection.selected_via,
        execution_mode=selection.execution_mode,
    )


def generate_sample_previews_native(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> NativeSamplePreviewsResult:
    try:
        import numpy as np
        import pandas as pd
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('numpy, pandas, and Pillow are required in the active CellPainting-Claw runtime for native sample previews.') from exc

    backend_payload = config.load_segmentation_backend_payload()
    paths_payload = backend_payload['paths']
    load_data_path = config.resolve_segmentation_backend_path(paths_payload['load_data_csv'])
    previews_dir = output_dir.resolve() if output_dir is not None else config.resolve_segmentation_backend_path(paths_payload['sample_previews_dir'])
    previews_dir.mkdir(parents=True, exist_ok=True)

    load_df = pd.read_csv(load_data_path)
    field_df = load_df.sort_values(['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']).drop_duplicates(
        subset=['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']
    )

    generated_count = 0
    skipped_existing = 0
    for row in field_df.itertuples(index=False):
        plate = str(row.Metadata_Plate)
        well = str(row.Metadata_Well)
        site = int(row.Metadata_Site)
        preview_path = previews_dir / f'{plate}_{well}_s{site}_sample.png'
        if preview_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        channel_images = {
            channel_name: np.array(
                Image.open(Path(str(getattr(row, f'PathName_{channel_name}'))) / str(getattr(row, f'FileName_{channel_name}')))
            )
            for channel_name in REQUIRED_CHANNELS
        }
        rgb = _compose_field_article_preview(channel_images)
        Image.fromarray((rgb * 255).astype(np.uint8), mode='RGB').save(preview_path)
        generated_count += 1

    return NativeSamplePreviewsResult(
        output_dir=previews_dir,
        generated_count=generated_count,
        skipped_existing=skipped_existing,
        field_count=int(len(field_df)),
    )


def generate_png_previews_native(
    config: ProjectConfig,
    *,
    mode: str = 'masked',
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
    workers: int = 0,
    chunk_size: int = 64,
) -> NativePNGPreviewsResult:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('pandas is required in the active CellPainting-Claw runtime for native single-cell PNG previews.') from exc

    resolved_manifest_path = manifest_path.resolve() if manifest_path is not None else _get_crop_manifest_path(config, mode)
    manifest_df = pd.read_csv(resolved_manifest_path)
    if manifest_df.empty:
        raise ValueError(f'No crops found in manifest: {resolved_manifest_path}')

    previews_dir = output_dir.resolve() if output_dir is not None else resolved_manifest_path.parent / 'previews_png'
    previews_dir.mkdir(parents=True, exist_ok=True)

    backend_payload = config.load_segmentation_backend_payload()
    channel_order = list(backend_payload['crop_extraction']['channel_order'])
    channel_index = {name: idx for idx, name in enumerate(channel_order)}

    records = manifest_df[['CropID', 'ImageStackPath']].copy()
    records['RowIndex'] = records.index
    record_list = records.to_dict(orient='records')

    effective_chunk_size = max(1, int(chunk_size))
    worker_count = int(workers) if int(workers) > 0 else min(
        os.cpu_count() or 1,
        max(1, len(record_list) // effective_chunk_size + 1),
    )
    worker_count = max(1, worker_count)

    preview_paths = [''] * len(record_list)
    if worker_count == 1:
        chunk_results = _process_png_preview_chunk(
            {
                'rows': record_list,
                'channel_index': channel_index,
                'previews_dir': str(previews_dir),
            }
        )
        for row_index, preview_path in chunk_results:
            preview_paths[row_index] = preview_path
    else:
        from concurrent.futures import ProcessPoolExecutor

        tasks = [
            {
                'rows': chunk,
                'channel_index': channel_index,
                'previews_dir': str(previews_dir),
            }
            for chunk in _chunk_records(record_list, effective_chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for chunk_results in executor.map(_process_png_preview_chunk, tasks):
                for row_index, preview_path in chunk_results:
                    preview_paths[row_index] = preview_path

    manifest_df['PreviewPNGPath'] = preview_paths
    manifest_df.to_csv(resolved_manifest_path, index=False)
    return NativePNGPreviewsResult(
        mode=mode,
        manifest_path=resolved_manifest_path,
        output_dir=previews_dir,
        preview_count=int(len(preview_paths)),
        worker_count=worker_count,
        chunk_size=effective_chunk_size,
    )


def extract_single_cell_crops_native(
    config: ProjectConfig,
    *,
    mode: str = 'masked',
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
    workers: int = 0,
) -> NativeSingleCellCropsResult:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('pandas is required in the active CellPainting-Claw runtime for native single-cell crop extraction.') from exc

    backend_payload = config.load_segmentation_backend_payload()
    paths_payload = backend_payload['paths']

    mask_output_dir = config.resolve_segmentation_backend_path(paths_payload['cellprofiler_output_dir'])
    load_data_path = config.resolve_segmentation_backend_path(paths_payload['load_data_csv'])
    default_crops_dir = _get_crop_output_dir(config, mode)
    resolved_crops_dir = output_dir.resolve() if output_dir is not None else default_crops_dir
    resolved_manifest_path = manifest_path.resolve() if manifest_path is not None else (
        resolved_crops_dir / 'single_cell_manifest.csv' if output_dir is not None else _get_crop_manifest_path(config, mode)
    )

    crop_size = int(backend_payload['crop_extraction']['crop_size_pixels'])
    channel_order = list(backend_payload['crop_extraction']['channel_order'])
    mask_target_cell = mode == 'masked'

    image_df = pd.read_csv(mask_output_dir / 'Image.csv')
    cells_df = pd.read_csv(mask_output_dir / 'Cells.csv')
    load_data_df = pd.read_csv(load_data_path)
    load_data_lookup = _build_load_data_lookup(load_data_df)

    labels_dir = mask_output_dir / 'labels'
    image_crops_dir = resolved_crops_dir / 'image_stacks'
    cell_masks_dir = resolved_crops_dir / 'cell_masks'
    nuclei_masks_dir = resolved_crops_dir / 'nuclei_masks'
    for path in [image_crops_dir, cell_masks_dir, nuclei_masks_dir]:
        path.mkdir(parents=True, exist_ok=True)

    image_lookup = {
        int(row['ImageNumber']): row
        for row in image_df.to_dict(orient='records')
    }
    tasks = []
    for image_number, image_cells_df in cells_df.groupby('ImageNumber', sort=True):
        image_number = int(image_number)
        image_row = image_lookup[image_number]
        plate = str(image_row['Metadata_Plate'])
        well = str(image_row['Metadata_Well'])
        site = int(image_row['Metadata_Site'])
        tasks.append(
            {
                'image_row': image_row,
                'cell_rows': image_cells_df.to_dict(orient='records'),
                'load_row': load_data_lookup[(plate, well, site)],
                'labels_dir': str(labels_dir),
                'image_crops_dir': str(image_crops_dir),
                'cell_masks_dir': str(cell_masks_dir),
                'nuclei_masks_dir': str(nuclei_masks_dir),
                'crop_size': crop_size,
                'channel_order': channel_order,
                'mask_target_cell': mask_target_cell,
            }
        )

    if not tasks:
        raise ValueError('No image groups found in Cells.csv.')

    worker_count = int(workers) if int(workers) > 0 else min(len(tasks), os.cpu_count() or 1)
    worker_count = max(1, worker_count)

    manifest_rows: list[dict] = []
    if worker_count == 1:
        for task in tasks:
            manifest_rows.extend(_process_single_cell_image_task(task))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for rows in executor.map(_process_single_cell_image_task, tasks):
                manifest_rows.extend(rows)

    manifest_df = pd.DataFrame(manifest_rows).sort_values(
        ['Metadata_Plate', 'Metadata_Well', 'Metadata_Site', 'ObjectNumber']
    ).reset_index(drop=True)
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(resolved_manifest_path, index=False)
    return NativeSingleCellCropsResult(
        mode=mode,
        crops_dir=resolved_crops_dir,
        manifest_path=resolved_manifest_path,
        crop_count=int(len(manifest_df)),
        worker_count=worker_count,
        background_masked=mask_target_cell,
    )


def cellprofiler_executable() -> Path | None:
    """Locate the CellProfiler CLI.

    Resolution order, first hit wins:

    1. ``CELLPROFILER_EXECUTABLE`` -- explicit, so a deployment never has to
       guess and a test can pin an exact build;
    2. a ``cellprofiler`` next to the running interpreter (conda/pip installs);
    3. ``cellprofiler`` on ``PATH``;
    4. the Windows one-folder installs, including this machine's
       ``C:\\CellProfiler\\CellProfiler.exe`` (4.2.7).

    Returning ``None`` means "no CellProfiler here"; callers then fall back to
    the in-process segmenter instead of failing.
    """
    import os
    import shutil
    import sys

    override = os.environ.get('CELLPROFILER_EXECUTABLE')
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.exists() else None

    sibling = Path(sys.executable).resolve().with_name('cellprofiler')
    for candidate in (sibling.with_suffix('.exe'), sibling):
        if candidate.exists():
            return candidate

    found = shutil.which('cellprofiler')
    if found:
        return Path(found)

    for candidate in (
        Path(r'C:\CellProfiler\CellProfiler.exe'),
        Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'CellProfiler' / 'CellProfiler.exe',
        Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'CellProfiler' / 'CellProfiler.exe',
    ):
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class NativeMaskExportResult:
    """Outcome of the in-process mask export."""

    output_dir: Path
    image_table_path: Path
    cells_table_path: Path
    nuclei_table_path: Path
    cytoplasm_table_path: Path
    experiment_table_path: Path
    labels_dir: Path
    outlines_dir: Path
    log_path: Path
    image_count: int
    nucleus_count: int
    cell_count: int
    mode: str


#: Channels measured for every object, in the order CellProfiler writes them.
_MEASURED_CHANNELS = ('OrigDNA', 'OrigER', 'OrigRNA', 'OrigAGP', 'OrigMito')


def _resolve_channel_path(path_name: str, file_name: str, *, bases: list[Path]) -> Path:
    candidate = Path(path_name) / file_name
    if candidate.is_absolute():
        return candidate
    for base in bases:
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return (bases[0] / candidate) if bases else candidate


def _segment_field(dna, *, nuclei_diameter: int | None = None) -> tuple:
    """Segment one field into (nuclei labels, cell labels).

    Deterministic classical pipeline: Otsu threshold on the DNA channel, distance
    transform, one peak per nucleus, watershed to split touching nuclei, then
    propagate each nucleus outwards to obtain the cell body.  This is the same
    family of operations the CellProfiler ``IdentifyPrimaryObjects`` /
    ``IdentifySecondaryObjects`` pair performs, implemented in-process so the
    demo does not need a CellProfiler install.
    """
    import numpy as np
    from scipy import ndimage as ndi
    from skimage import filters, measure, morphology, segmentation

    image = np.asarray(dna, dtype='float32')
    if image.ndim > 2:
        image = image[..., 0]
    # Object size scales with the field size: the demo ships 64 px crops while a
    # real Cell Painting field is ~1000 px, so a fixed diameter cannot serve both.
    if nuclei_diameter is None:
        nuclei_diameter = max(5, int(round(min(image.shape) * 0.08)))
    if image.size == 0 or float(image.max()) <= float(image.min()):
        empty = np.zeros(image.shape, dtype='uint16')
        return empty, empty

    smoothed = filters.gaussian(image, sigma=1.5, preserve_range=True)
    threshold = filters.threshold_otsu(smoothed) if smoothed.max() > smoothed.min() else smoothed.max()
    foreground = smoothed > threshold
    foreground = morphology.remove_small_objects(
        foreground, min_size=max(12, int(0.0015 * image.size))
    )
    foreground = morphology.binary_opening(foreground, morphology.disk(2))
    foreground = ndi.binary_fill_holes(foreground)
    if not foreground.any():
        empty = np.zeros(image.shape, dtype='uint16')
        return empty, empty

    min_distance = max(3, int(round(nuclei_diameter / 2)))
    distance = ndi.distance_transform_edt(foreground)
    from skimage.feature import peak_local_max

    peaks = peak_local_max(
        distance, min_distance=min_distance, labels=foreground, exclude_border=False
    )
    markers = np.zeros(distance.shape, dtype='int32')
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    nuclei = segmentation.watershed(-distance, markers, mask=foreground)
    # drop nuclei smaller than a plausible object, then renumber 1..N
    min_area = max(12, int(0.0006 * image.size))
    sizes = np.bincount(nuclei.ravel(), minlength=2)
    keep = np.flatnonzero(sizes >= min_area)
    keep = keep[keep != 0]
    if keep.size == 0:
        empty = np.zeros(image.shape, dtype='uint16')
        return empty, empty
    remap = np.zeros(sizes.shape, dtype='uint16')
    remap[keep] = np.arange(1, keep.size + 1, dtype='uint16')
    nuclei = remap[nuclei]
    if nuclei.max() == 0:
        empty = np.zeros(image.shape, dtype='uint16')
        return empty, empty

    cells = segmentation.expand_labels(nuclei, distance=max(4, int(nuclei_diameter / 2)))
    cells = cells.astype('uint16')
    return nuclei.astype('uint16'), cells


def _object_rows(labels, channels: dict, *, image_number: int, row: dict,
                 parent: dict | None = None, prefix: str = '') -> list[dict]:
    import numpy as np
    from skimage import measure

    records: list[dict] = []
    if labels is None or int(np.max(labels)) == 0:
        return records
    for region in measure.regionprops(labels):
        min_y, min_x, max_y, max_x = region.bbox
        record = {
            'ImageNumber': image_number,
            'ObjectNumber': int(region.label),
            'Metadata_Plate': row['Metadata_Plate'],
            'Metadata_Well': row['Metadata_Well'],
            'Metadata_Site': int(row['Metadata_Site']),
            'AreaShape_Area': float(region.area),
            'AreaShape_Perimeter': float(region.perimeter),
            'AreaShape_Center_X': float(region.centroid[1]),
            'AreaShape_Center_Y': float(region.centroid[0]),
            'Location_Center_X': float(region.centroid[1]),
            'Location_Center_Y': float(region.centroid[0]),
            'AreaShape_BoundingBoxMinimum_X': float(min_x),
            'AreaShape_BoundingBoxMaximum_X': float(max_x),
            'AreaShape_BoundingBoxMinimum_Y': float(min_y),
            'AreaShape_BoundingBoxMaximum_Y': float(max_y),
        }
        if parent is not None:
            record['Parent_Nuclei'] = parent.get(int(region.label))
        for channel_name, image in channels.items():
            values = image[labels == region.label]
            record[f'Intensity_MeanIntensity_{channel_name}'] = float(values.mean()) if values.size else 0.0
        records.append(record)
    return records


def run_mask_export_native(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    log_path: Path | None = None,
) -> NativeMaskExportResult:
    """Segment the configured fields in-process and write CellProfiler-shaped tables.

    This is the fallback used when the CellProfiler backend script is not
    packaged or not installed.  It writes the same artifact contract the
    CellProfiler path produces (``Image.csv``, ``Cells.csv``, ``Nuclei.csv``,
    ``Cytoplasm.csv``, ``Experiment.csv``, ``labels/``, ``outlines/``) so every
    downstream skill keeps working unchanged.
    """
    import numpy as np
    import pandas as pd
    import tifffile
    from PIL import Image

    backend_payload = config.load_segmentation_backend_payload()
    paths_payload = backend_payload['paths']
    load_data_path = config.resolve_segmentation_backend_path(paths_payload['load_data_csv'])
    resolved_output_dir = (
        output_dir.resolve() if output_dir is not None
        else config.resolve_segmentation_backend_path(paths_payload['cellprofiler_output_dir'])
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = resolved_output_dir / 'labels'
    outlines_dir = resolved_output_dir / 'outlines'
    labels_dir.mkdir(parents=True, exist_ok=True)
    outlines_dir.mkdir(parents=True, exist_ok=True)

    resolved_log_path = (log_path or (resolved_output_dir / 'native_mask_export.log')).resolve()

    load_data = pd.read_csv(load_data_path)
    if load_data.empty:
        raise ValueError(f'Segmentation load-data table is empty: {load_data_path}')

    # PathName_* in the shipped demo tables is repo-relative; accept cwd first,
    # then the two layouts a project config implies (<repo>/demo/workspace/...).
    bases = [Path.cwd()]
    for anchor in (config.workspace_root, config.segmentation_backend_root):
        for parent in list(anchor.parents)[:3]:
            if parent not in bases:
                bases.append(parent)
    available_channels = [
        name for name in _MEASURED_CHANNELS
        if f'FileName_{name}' in load_data.columns and f'PathName_{name}' in load_data.columns
    ]
    if 'OrigDNA' not in available_channels:
        raise ValueError('The load-data table has no OrigDNA channel; cannot segment.')

    image_rows: list[dict] = []
    nuclei_rows: list[dict] = []
    cell_rows: list[dict] = []
    cytoplasm_rows: list[dict] = []
    log_lines: list[str] = []

    for image_number, (_, row) in enumerate(load_data.iterrows(), start=1):
        channels = {}
        for name in available_channels:
            path = _resolve_channel_path(
                str(row[f'PathName_{name}']), str(row[f'FileName_{name}']), bases=bases
            )
            channels[name] = np.asarray(Image.open(path))
        nuclei, cells = _segment_field(channels['OrigDNA'])
        prefix = f"{row['Metadata_Plate']}_{row['Metadata_Well']}_s{int(row['Metadata_Site'])}"
        tifffile.imwrite(labels_dir / f'{prefix}--nuclei_labels.tiff', nuclei.astype('uint16'))
        tifffile.imwrite(labels_dir / f'{prefix}--cell_labels.tiff', cells.astype('uint16'))

        from skimage import segmentation as _seg
        for kind, labels in (('nuclei', nuclei), ('cells', cells)):
            outline = _seg.find_boundaries(labels, mode='outer').astype('uint8') * 255
            Image.fromarray(outline).save(outlines_dir / f'{prefix}--{kind}_outlines.png')

        parents = {}
        if int(cells.max()) > 0 and int(nuclei.max()) > 0:
            for cell_label in range(1, int(cells.max()) + 1):
                nucleus_labels = nuclei[cells == cell_label]
                nucleus_labels = nucleus_labels[nucleus_labels > 0]
                if nucleus_labels.size:
                    parents[cell_label] = int(np.bincount(nucleus_labels).argmax())

        nucleus_count = int(len(np.unique(nuclei))) - (1 if int(nuclei.max()) > 0 else 0)
        cell_count = int(len(np.unique(cells))) - (1 if int(cells.max()) > 0 else 0)
        image_record = {
            'ImageNumber': image_number,
            'Metadata_Plate': row['Metadata_Plate'],
            'Metadata_Well': row['Metadata_Well'],
            'Metadata_Site': int(row['Metadata_Site']),
            'Count_Nuclei': nucleus_count,
            'Count_Cells': cell_count,
        }
        for name in available_channels:
            image_record[f'FileName_{name}'] = str(row[f'FileName_{name}'])
            image_record[f'PathName_{name}'] = str(row[f'PathName_{name}'])
        image_rows.append(image_record)

        nuclei_rows.extend(_object_rows(nuclei, channels, image_number=image_number, row=row))
        cell_rows.extend(_object_rows(cells, channels, image_number=image_number, row=row, parent=parents))
        # cytoplasm = cell body minus its nucleus
        cytoplasm = np.zeros_like(cells)
        for cell_label, nucleus_label in parents.items():
            region = cells == cell_label
            cytoplasm[region & (nuclei != nucleus_label)] = cell_label
        cytoplasm_rows.extend(
            _object_rows(cytoplasm, channels, image_number=image_number, row=row)
        )
        log_lines.append(
            f'field {prefix}: nuclei={nucleus_count} cells={cell_count}'
        )

    resolved_log_path.write_text('\n'.join(log_lines) + '\n', encoding='utf-8')

    image_table_path = resolved_output_dir / 'Image.csv'
    cells_table_path = resolved_output_dir / 'Cells.csv'
    nuclei_table_path = resolved_output_dir / 'Nuclei.csv'
    cytoplasm_table_path = resolved_output_dir / 'Cytoplasm.csv'
    experiment_table_path = resolved_output_dir / 'Experiment.csv'

    pd.DataFrame(image_rows).to_csv(image_table_path, index=False)
    pd.DataFrame(cell_rows).to_csv(cells_table_path, index=False)
    pd.DataFrame(nuclei_rows).to_csv(nuclei_table_path, index=False)
    pd.DataFrame(cytoplasm_rows).to_csv(cytoplasm_table_path, index=False)
    pd.DataFrame([{'ExperimentNumber': 1, 'Metadata_Plate': '-', 'Metadata_Well': '-'}]).to_csv(
        experiment_table_path, index=False
    )

    return NativeMaskExportResult(
        output_dir=resolved_output_dir,
        image_table_path=image_table_path,
        cells_table_path=cells_table_path,
        nuclei_table_path=nuclei_table_path,
        cytoplasm_table_path=cytoplasm_table_path,
        experiment_table_path=experiment_table_path,
        labels_dir=labels_dir,
        outlines_dir=outlines_dir,
        log_path=resolved_log_path,
        image_count=len(image_rows),
        nucleus_count=len(nuclei_rows),
        cell_count=len(cell_rows),
        mode='native-fallback',
    )


def summarize_segmentation_outputs(config: ProjectConfig) -> SegmentationSummaryResult:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('pandas is required in the active CellPainting-Claw runtime for segmentation summary.') from exc

    backend_payload = config.load_segmentation_backend_payload()
    paths_payload = backend_payload['paths']

    load_data_path = config.resolve_segmentation_backend_path(paths_payload['load_data_csv'])
    mask_output_dir = config.resolve_segmentation_backend_path(paths_payload['cellprofiler_output_dir'])
    image_table_path = mask_output_dir / 'Image.csv'
    cells_table_path = mask_output_dir / 'Cells.csv'
    nuclei_table_path = mask_output_dir / 'Nuclei.csv'
    label_dir = mask_output_dir / 'labels'
    outline_dir = mask_output_dir / 'outlines'
    sample_previews_dir = config.resolve_segmentation_backend_path(paths_payload['sample_previews_dir'])
    masked_manifest_path = config.resolve_segmentation_backend_path(paths_payload['masked_manifest_csv'])
    unmasked_manifest_path = config.resolve_segmentation_backend_path(paths_payload['unmasked_manifest_csv'])

    problems: list[str] = []

    load_data_rows = _csv_row_count(pd, load_data_path, problems, 'load_data_csv')
    image_row_count = _csv_row_count(pd, image_table_path, problems, 'Image.csv')
    cell_row_count = _csv_row_count(pd, cells_table_path, problems, 'Cells.csv')
    nuclei_row_count = _csv_row_count(pd, nuclei_table_path, problems, 'Nuclei.csv')
    masked_manifest_rows = _csv_row_count(pd, masked_manifest_path, problems, 'masked_manifest_csv')
    unmasked_manifest_rows = _csv_row_count(pd, unmasked_manifest_path, problems, 'unmasked_manifest_csv')

    label_file_count = _count_files(label_dir)
    outline_file_count = _count_files(outline_dir)
    sample_preview_count = _count_files(sample_previews_dir, '*.png')

    masked_root = masked_manifest_path.parent
    masked_image_stack_count = _count_files(masked_root / 'image_stacks', '*.tiff')
    masked_cell_mask_count = _count_files(masked_root / 'cell_masks', '*.tiff')
    masked_nuclei_mask_count = _count_files(masked_root / 'nuclei_masks', '*.tiff')
    masked_preview_count = _count_files(masked_root / 'previews_png', '*.png')

    unmasked_root = unmasked_manifest_path.parent
    unmasked_image_stack_count = _count_files(unmasked_root / 'image_stacks', '*.tiff')
    unmasked_cell_mask_count = _count_files(unmasked_root / 'cell_masks', '*.tiff')
    unmasked_nuclei_mask_count = _count_files(unmasked_root / 'nuclei_masks', '*.tiff')
    unmasked_preview_count = _count_files(unmasked_root / 'previews_png', '*.png')

    if image_row_count > 0 and label_file_count not in {0, image_row_count * 2}:
        problems.append(
            f'Expected label file count to be 2x Image.csv rows, got labels={label_file_count}, images={image_row_count}'
        )
    if masked_manifest_rows > 0:
        _check_manifest_counts(
            manifest_name='masked',
            manifest_rows=masked_manifest_rows,
            image_stack_count=masked_image_stack_count,
            cell_mask_count=masked_cell_mask_count,
            nuclei_mask_count=masked_nuclei_mask_count,
            preview_count=masked_preview_count,
            problems=problems,
        )
    if unmasked_manifest_rows > 0:
        _check_manifest_counts(
            manifest_name='unmasked',
            manifest_rows=unmasked_manifest_rows,
            image_stack_count=unmasked_image_stack_count,
            cell_mask_count=unmasked_cell_mask_count,
            nuclei_mask_count=unmasked_nuclei_mask_count,
            preview_count=unmasked_preview_count,
            problems=problems,
        )
    if masked_manifest_rows > 0 and unmasked_manifest_rows > 0 and masked_manifest_rows != unmasked_manifest_rows:
        problems.append(
            f'Masked and unmasked manifest row counts differ: masked={masked_manifest_rows}, unmasked={unmasked_manifest_rows}'
        )

    return SegmentationSummaryResult(
        load_data_path=load_data_path,
        load_data_rows=load_data_rows,
        mask_output_dir=mask_output_dir,
        image_table_path=image_table_path,
        image_row_count=image_row_count,
        cells_table_path=cells_table_path,
        cell_row_count=cell_row_count,
        nuclei_table_path=nuclei_table_path,
        nuclei_row_count=nuclei_row_count,
        label_dir=label_dir,
        label_file_count=label_file_count,
        outline_dir=outline_dir,
        outline_file_count=outline_file_count,
        sample_previews_dir=sample_previews_dir,
        sample_preview_count=sample_preview_count,
        masked_manifest_path=masked_manifest_path,
        masked_manifest_rows=masked_manifest_rows,
        masked_image_stack_count=masked_image_stack_count,
        masked_cell_mask_count=masked_cell_mask_count,
        masked_nuclei_mask_count=masked_nuclei_mask_count,
        masked_preview_count=masked_preview_count,
        unmasked_manifest_path=unmasked_manifest_path,
        unmasked_manifest_rows=unmasked_manifest_rows,
        unmasked_image_stack_count=unmasked_image_stack_count,
        unmasked_cell_mask_count=unmasked_cell_mask_count,
        unmasked_nuclei_mask_count=unmasked_nuclei_mask_count,
        unmasked_preview_count=unmasked_preview_count,
        problems=problems,
    )


def segmentation_summary_to_dict(result: SegmentationSummaryResult) -> dict[str, object]:
    return {
        'implementation': 'native',
        'step': 'summarize-segmentation',
        'ok': result.ok,
        'load_data_path': str(result.load_data_path),
        'load_data_rows': result.load_data_rows,
        'mask_output_dir': str(result.mask_output_dir),
        'image_table_path': str(result.image_table_path),
        'image_row_count': result.image_row_count,
        'cells_table_path': str(result.cells_table_path),
        'cell_row_count': result.cell_row_count,
        'nuclei_table_path': str(result.nuclei_table_path),
        'nuclei_row_count': result.nuclei_row_count,
        'label_dir': str(result.label_dir),
        'label_file_count': result.label_file_count,
        'outline_dir': str(result.outline_dir),
        'outline_file_count': result.outline_file_count,
        'sample_previews_dir': str(result.sample_previews_dir),
        'sample_preview_count': result.sample_preview_count,
        'masked_manifest_path': str(result.masked_manifest_path),
        'masked_manifest_rows': result.masked_manifest_rows,
        'masked_image_stack_count': result.masked_image_stack_count,
        'masked_cell_mask_count': result.masked_cell_mask_count,
        'masked_nuclei_mask_count': result.masked_nuclei_mask_count,
        'masked_preview_count': result.masked_preview_count,
        'unmasked_manifest_path': str(result.unmasked_manifest_path),
        'unmasked_manifest_rows': result.unmasked_manifest_rows,
        'unmasked_image_stack_count': result.unmasked_image_stack_count,
        'unmasked_cell_mask_count': result.unmasked_cell_mask_count,
        'unmasked_nuclei_mask_count': result.unmasked_nuclei_mask_count,
        'unmasked_preview_count': result.unmasked_preview_count,
        'problems': result.problems,
    }


def write_segmentation_summary(result: SegmentationSummaryResult, output_path: Path) -> Path:
    resolved_output_path = output_path.resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(segmentation_summary_to_dict(result), indent=2) + chr(10), encoding='utf-8')
    return resolved_output_path


def _normalize_channel(channel):
    import numpy as np

    channel = channel.astype(np.float32, copy=False)
    low = float(np.percentile(channel, 1))
    high = float(np.percentile(channel, 99.5))
    if high <= low:
        high = low + 1.0
    return np.clip((channel - low) / (high - low), 0.0, 1.0)


def _compose_field_article_preview(channel_images):
    import numpy as np

    first_channel = next(iter(channel_images.values()))
    rgb = np.zeros((*first_channel.shape, 3), dtype=np.float32)
    for channel_name, color in ARTICLE_PSEUDOCOLORS.items():
        normalized = _normalize_channel(channel_images[channel_name])
        rgb = np.maximum(rgb, normalized[..., None] * np.array(color, dtype=np.float32))
    return rgb


def _compose_stack_article_preview(stack, channel_index):
    import numpy as np

    rgb = np.zeros((*stack.shape[1:], 3), dtype=np.float32)
    for channel_name, color in ARTICLE_PSEUDOCOLORS.items():
        normalized = _normalize_channel(stack[channel_index[channel_name]])
        rgb = np.maximum(rgb, normalized[..., None] * np.array(color, dtype=np.float32))
    return rgb


def _filter_segmentation_rows(dataframe, subset_cfg: dict):
    filtered = dataframe.copy()
    plates = subset_cfg.get('plates', [])
    wells = subset_cfg.get('wells', [])
    sites = subset_cfg.get('sites', [])
    if plates:
        filtered = filtered[filtered['Metadata_Plate'].isin(plates)]
    if wells:
        filtered = filtered[filtered['Metadata_Well'].isin(wells)]
    if sites:
        filtered = filtered[filtered['Metadata_Site'].isin(sites)]
    return filtered


def _get_crop_manifest_path(config: ProjectConfig, mode: str) -> Path:
    backend_payload = config.load_segmentation_backend_payload()
    paths_payload = backend_payload['paths']
    if mode == 'masked':
        return config.resolve_segmentation_backend_path(paths_payload['masked_manifest_csv'])
    if mode == 'unmasked':
        return config.resolve_segmentation_backend_path(paths_payload['unmasked_manifest_csv'])
    raise ValueError(f'Unsupported crop preview mode: {mode}')


def _get_crop_output_dir(config: ProjectConfig, mode: str) -> Path:
    backend_payload = config.load_segmentation_backend_payload()
    paths_payload = backend_payload['paths']
    if mode == 'masked':
        return config.resolve_segmentation_backend_path(paths_payload['masked_crops_dir'])
    if mode == 'unmasked':
        return config.resolve_segmentation_backend_path(paths_payload['unmasked_crops_dir'])
    raise ValueError(f'Unsupported crop preview mode: {mode}')


def _validate_local_files(dataframe, *, locator: FileLocatorPort | None = None) -> None:
    """Check that every file referenced by the load-data table exists.

    ``locator`` decides how a ``PathName_*`` entry becomes a concrete path.
    The default keeps the historical behaviour of resolving against the
    process working directory; pass a
    :class:`~cellpaint_pipeline.ports.BaseDirFileLocator` to make the base
    directory explicit instead of implicit.
    """
    validate_local_files(dataframe, locator=locator)


def _centered_crop_bounds(center: float, full_size: int, crop_size: int) -> tuple[int, int, int, int]:
    start = int(round(center)) - (crop_size // 2)
    end = start + crop_size
    src_start = max(0, start)
    src_end = min(full_size, end)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    return src_start, src_end, dst_start, dst_end


def _build_load_data_lookup(load_data_df) -> dict[tuple[str, str, int], dict]:
    lookup: dict[tuple[str, str, int], dict] = {}
    for row in load_data_df.to_dict(orient='records'):
        key = (
            str(row['Metadata_Plate']),
            str(row['Metadata_Well']),
            int(row['Metadata_Site']),
        )
        lookup[key] = row
    return lookup


def _chunk_records(records: list[dict], chunk_size: int) -> list[list[dict]]:
    return [records[index:index + chunk_size] for index in range(0, len(records), chunk_size)]


def _process_single_cell_image_task(task: dict) -> list[dict]:
    import numpy as np
    import tifffile
    from PIL import Image

    image_row = task['image_row']
    cell_rows = task['cell_rows']
    load_row = task['load_row']
    labels_dir = Path(task['labels_dir'])
    image_crops_dir = Path(task['image_crops_dir'])
    cell_masks_dir = Path(task['cell_masks_dir'])
    nuclei_masks_dir = Path(task['nuclei_masks_dir'])
    crop_size = int(task['crop_size'])
    channel_order = list(task['channel_order'])
    mask_target_cell = bool(task['mask_target_cell'])

    plate = str(image_row['Metadata_Plate'])
    well = str(image_row['Metadata_Well'])
    site = int(image_row['Metadata_Site'])
    image_number = int(image_row['ImageNumber'])
    label_prefix = f'{plate}_{well}_s{site}'

    nuclei_labels = tifffile.imread(labels_dir / f'{label_prefix}--nuclei_labels.tiff')
    cell_labels = tifffile.imread(labels_dir / f'{label_prefix}--cell_labels.tiff')

    channel_images = {}
    for channel_name in channel_order:
        filename = load_row[f'FileName_{channel_name}']
        pathname = load_row[f'PathName_{channel_name}']
        channel_images[channel_name] = np.array(Image.open(Path(str(pathname)) / str(filename)))

    height, width = cell_labels.shape
    channel_dtype = channel_images[channel_order[0]].dtype
    manifest_rows: list[dict] = []

    for cell_row in cell_rows:
        object_number = int(cell_row['ObjectNumber'])
        parent_nucleus = int(cell_row['Parent_Nuclei'])
        bbox_min_x = int(cell_row['AreaShape_BoundingBoxMinimum_X'])
        bbox_max_x = int(cell_row['AreaShape_BoundingBoxMaximum_X'])
        bbox_min_y = int(cell_row['AreaShape_BoundingBoxMinimum_Y'])
        bbox_max_y = int(cell_row['AreaShape_BoundingBoxMaximum_Y'])
        center_x = float(cell_row['AreaShape_Center_X'])
        center_y = float(cell_row['AreaShape_Center_Y'])

        min_y, max_y, dst_min_y, dst_max_y = _centered_crop_bounds(center_y, height, crop_size)
        min_x, max_x, dst_min_x, dst_max_x = _centered_crop_bounds(center_x, width, crop_size)

        crop_id = f'{plate}_{well}_s{site}_cell{object_number:04d}'
        stack_path = image_crops_dir / f'{crop_id}.tiff'
        cell_mask_path = cell_masks_dir / f'{crop_id}.tiff'
        nuclei_mask_path = nuclei_masks_dir / f'{crop_id}.tiff'

        cropped_stack = np.zeros((len(channel_order), crop_size, crop_size), dtype=channel_dtype)
        for channel_idx, channel_name in enumerate(channel_order):
            cropped_stack[channel_idx, dst_min_y:dst_max_y, dst_min_x:dst_max_x] = (
                channel_images[channel_name][min_y:max_y, min_x:max_x]
            )

        cropped_cell_mask = np.zeros((crop_size, crop_size), dtype='uint8')
        cropped_cell_mask[dst_min_y:dst_max_y, dst_min_x:dst_max_x] = (
            cell_labels[min_y:max_y, min_x:max_x] == object_number
        ).astype('uint8')
        cropped_nuclei_mask = np.zeros((crop_size, crop_size), dtype='uint8')
        cropped_nuclei_mask[dst_min_y:dst_max_y, dst_min_x:dst_max_x] = (
            nuclei_labels[min_y:max_y, min_x:max_x] == parent_nucleus
        ).astype('uint8')

        if mask_target_cell:
            cropped_stack *= cropped_cell_mask[None, :, :].astype(channel_dtype)

        is_clipped = (
            bbox_min_y < min_y
            or bbox_max_y > max_y
            or bbox_min_x < min_x
            or bbox_max_x > max_x
        )

        tifffile.imwrite(stack_path, cropped_stack)
        tifffile.imwrite(cell_mask_path, cropped_cell_mask)
        tifffile.imwrite(nuclei_mask_path, cropped_nuclei_mask)

        manifest_rows.append(
            {
                'ImageNumber': image_number,
                'Metadata_Plate': plate,
                'Metadata_Well': well,
                'Metadata_Site': site,
                'ObjectNumber': object_number,
                'Parent_Nuclei': parent_nucleus,
                'CropID': crop_id,
                'CenterY': center_y,
                'CenterX': center_x,
                'MinY': min_y,
                'MaxY': max_y,
                'MinX': min_x,
                'MaxX': max_x,
                'PadTop': dst_min_y,
                'PadBottom': crop_size - dst_max_y,
                'PadLeft': dst_min_x,
                'PadRight': crop_size - dst_max_x,
                'Height': crop_size,
                'Width': crop_size,
                'CropWindowClippedCell': is_clipped,
                'BackgroundMasked': mask_target_cell,
                'ImageStackPath': str(stack_path),
                'CellMaskPath': str(cell_mask_path),
                'NucleiMaskPath': str(nuclei_mask_path),
            }
        )
    return manifest_rows


def _process_png_preview_chunk(task: dict) -> list[tuple[int, str]]:
    import numpy as np
    import tifffile
    from PIL import Image

    channel_index = task['channel_index']
    previews_dir = Path(task['previews_dir'])
    results: list[tuple[int, str]] = []
    for row in task['rows']:
        stack = tifffile.imread(row['ImageStackPath'])
        rgb = _compose_stack_article_preview(stack=stack, channel_index=channel_index)
        preview_path = previews_dir / f"{row['CropID']}.png"
        Image.fromarray((rgb * 255).astype(np.uint8), mode='RGB').save(preview_path)
        results.append((int(row['RowIndex']), str(preview_path)))
    return results


def _extract_module_count(pipeline_text: str) -> int:
    match = re.search(r'ModuleCount:(\d+)', pipeline_text)
    if match is None:
        raise ValueError('Could not determine ModuleCount from the generated .cppipe text.')
    return int(match.group(1))


def _renumber_module(match: re.Match[str]) -> str:
    current = int(match.group(1))
    if current >= 32:
        current += 4
    return f'module_num:{current}'


def _csv_row_count(pd, path: Path, problems: list[str], label: str) -> int:
    if not path.exists():
        problems.append(f'Missing required segmentation artifact: {label} -> {path}')
        return 0
    return int(len(pd.read_csv(path)))


def _count_files(path: Path, pattern: str = '*') -> int:
    if not path.exists():
        return 0
    return sum(1 for child in path.glob(pattern) if child.is_file())


def _check_manifest_counts(
    *,
    manifest_name: str,
    manifest_rows: int,
    image_stack_count: int,
    cell_mask_count: int,
    nuclei_mask_count: int,
    preview_count: int,
    problems: list[str],
) -> None:
    counts = {
        'image_stacks': image_stack_count,
        'cell_masks': cell_mask_count,
        'nuclei_masks': nuclei_mask_count,
        'previews_png': preview_count,
    }
    for artifact_name, artifact_count in counts.items():
        if artifact_count != manifest_rows:
            problems.append(
                f'{manifest_name} manifest rows do not match {artifact_name}: rows={manifest_rows}, files={artifact_count}'
            )
