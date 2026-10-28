"""Turn BBBC022 into a single-cell benchmark dataset shared by all four methods.

WHAT THIS PRODUCES
------------------
For every cell it writes the four input representations each method needs, all
derived from the *same* segmentation of the *same* field, so the comparison is
not confounded by different cells:

  deepprofiler   (N, 32, 32, 5)  float32, per-crop min-max onto [0,1], BHWC
  ca_mae         (N, 5, 256, 256) uint8, raw 0..255, BCHW
  cellclip       (N, 224, 224, 5) uint8, the crops DINOv2 will embed
  o2vae          (N, 1, 64, 64)  float32 binary nucleus silhouette in [0,1]

plus `meta.csv` carrying the row identity and the BBBC022 labels
(compound, concentration, well role, plate, site) needed for scoring.

WHY THE CROP SIZES DIFFER (and why that is still fair)
-----------------------------------------------------
Each method is fed at the scale it was trained on: DeepProfiler on 32x32 crops,
OpenPhenom on 256x256, DINOv2 on 224x224.  Forcing one size would handicap
whichever method does not match.  The *cells* are identical; only the field of
view around each cell follows each method's contract.

KNOWN ASYMMETRY, STATED UP FRONT
--------------------------------
O2-VAE is a shape model trained on MEF cell silhouettes; here it receives a
*nucleus* silhouette, because BBBC022 has no cytoplasm stain that segments
reliably.  A nucleus mask is not a whole-cell mask, so O2-VAE's input is a close
but not exact analogue of its training distribution.  That is a property of the
method's modality, not a shortcut, and it must be mentioned wherever its scores
are compared against the intensity models.

SEGMENTATION
------------
Nuclei are found on the Hoechst channel with Otsu thresholding + distance-
watershed, then filtered by area.  This is deliberately simple and identical for
every cell.  The upstream DeepProfiler protocol uses CellProfiler masks; using a
lighter segmenter changes the *input* but not the *comparison*, and keeps the
whole dataset reproducible without a frozen CellProfiler install.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

# BBBC022 channel columns, mapped onto the SEMANTIC names.
# Hoechst = DNA, Syto = RNA, Ph_golgi = AGP, and the ER/Mito columns are theirs.
CHANNEL_COLUMNS = {
    'DNA': 'Image_FileName_OrigHoechst',
    'RNA': 'Image_FileName_OrigSyto',
    'ER': 'Image_FileName_OrigER',
    'AGP': 'Image_FileName_OrigPh_golgi',
    'Mito': 'Image_FileName_OrigMito',
}

#: Channel order written into the saved arrays.
#:
#: CORRECTION.  This used to follow the dict order above, i.e.
#: ``(DNA, RNA, ER, AGP, Mito)``.  The Cell Painting CNN v1 was trained on
#: ``(DNA, ER, RNA, AGP, Mito)`` (upstream README.md:74 and
#: tests/deepprofiler/profiling/test_cell_painting_cnn_v1.py:74), so RNA and ER
#: reached the model transposed.  The dict above stays as the column lookup; this
#: tuple is the order that is actually written.
CHANNELS = ('DNA', 'ER', 'RNA', 'AGP', 'Mito')

LABEL_COLUMNS = {
    'plate': 'Image_Metadata_PlateID',
    'well': 'Image_Metadata_CPD_WELL_POSITION',
    'site': 'Image_Metadata_Site',
    'compound': 'Image_Metadata_SOURCE_COMPOUND_NAME',
    'concentration': 'Image_Metadata_CPD_MMOL_CONC',
    'well_role': 'Image_Metadata_ASSAY_WELL_ROLE',
    'broad_id': 'Image_Metadata_BROAD_ID',
}


# ----------------------------------------------------------------- indexing ---

def index_zips(zip_paths: list[Path]) -> dict[str, Path]:
    """Map every member file name to the zip that holds it."""
    index: dict[str, Path] = {}
    for path in zip_paths:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.lower().endswith(('.tif', '.tiff')):
                    index[Path(name).name] = path
    return index


def index_directory(root: Path) -> dict[str, Path]:
    """Map every image file name under ``root`` to its path.

    Needed because the BBBC022 mirror degraded to ~19 KB/s mid-download: the
    archives are truncated, so the complete entries were salvaged out of them and
    the benchmark runs from that extracted tree instead of the zips.
    """
    index: dict[str, Path] = {}
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in ('.tif', '.tiff'):
            index[path.name] = path
    return index


def load_tiff(index: dict[str, Path], name: str) -> np.ndarray:
    """Read one TIFF from a zip archive or from a plain file."""
    import tifffile

    archive_path = index.get(name)
    if archive_path is None:
        raise FileNotFoundError(name)
    if archive_path.suffix.lower() in ('.tif', '.tiff'):
        return tifffile.imread(str(archive_path))
    with zipfile.ZipFile(archive_path) as archive:
        member = next(n for n in archive.namelist() if Path(n).name == name)
        payload = archive.read(member)
    return tifffile.imread(io.BytesIO(payload))


# -------------------------------------------------------------- segmentation --

def segment_nuclei(dna: np.ndarray, *, min_area: int = 80, max_area: int = 20000):
    """Otsu + distance-watershed on one Hoechst image.

    Returns a label image (0 = background).  Kept dependency-light on purpose:
    `skimage` is already required by the pipeline, so no CellProfiler run is
    needed to regenerate the dataset.
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed

    image = dna.astype(np.float64)
    low, high = np.percentile(image, [1, 99])
    image = np.clip((image - low) / max(high - low, 1e-9), 0, 1)

    try:
        threshold = threshold_otsu(image)
    except ValueError:
        return np.zeros(image.shape, dtype=np.int32)

    binary = remove_small_objects(image > threshold, min_size=max(8, min_area // 4))
    if not binary.any():
        return np.zeros(image.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=7, labels=binary, exclude_border=False)
    markers = np.zeros(image.shape, dtype=np.int32)
    for label_id, (row, col) in enumerate(coords, start=1):
        markers[row, col] = label_id

    labelled = watershed(-distance, markers, mask=binary)
    cleaned = np.zeros_like(labelled)
    next_id = 1
    for region in regionprops(labelled):
        if min_area <= region.area <= max_area:
            cleaned[labelled == region.label] = next_id
            next_id += 1
    return cleaned


def crop_pad(image: np.ndarray, centre_y: float, centre_x: float, size: int) -> np.ndarray:
    """Square crop centred on a float coordinate, zero-padded at the border."""
    half = size // 2
    top = int(round(centre_y)) - half
    left = int(round(centre_x)) - half
    height, width = image.shape[:2]
    out = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    src_top, src_left = max(top, 0), max(left, 0)
    dst_top, dst_left = max(-top, 0), max(-left, 0)
    take_h = min(size - dst_top, height - src_top)
    take_w = min(size - dst_left, width - src_left)
    if take_h > 0 and take_w > 0:
        out[dst_top:dst_top + take_h, dst_left:dst_left + take_w] = image[
            src_top:src_top + take_h, src_left:src_left + take_w
        ]
    return out


def resize_to(image: np.ndarray, size: int) -> np.ndarray:
    from skimage.transform import resize

    if image.shape[0] == size and image.shape[1] == size:
        return image
    # Resize per channel so a uint8 stack keeps its values and dtype intent.
    channels = [
        resize(image[..., c], (size, size), order=1, preserve_range=True,
               anti_aliasing=True)
        for c in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1)


# ------------------------------------------------------------------- build ----

def build(
    source: list[Path],
    labels_csv: Path,
    output_dir: Path,
    *,
    images_dir: Path | None = None,
    fields_per_well: int = 1,
    max_cells: int = 3000,
    seed: int = 0,
) -> dict:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_csv, engine='python', on_bad_lines='skip')

    if images_dir is not None:
        index = index_directory(images_dir)
        print(f'indexed {len(index)} images under {images_dir}')
    else:
        index = index_zips(source)
        print(f'indexed {len(index)} images across {len(source)} archives')

    # One row per (well, site) is what we need; keep the first field of each well
    # so the benchmark is not dominated by a handful of wells.
    usable = labels.dropna(subset=list(CHANNEL_COLUMNS.values()))
    usable = usable[usable['Image_Metadata_Site'].astype(str).str.isdigit()]
    rows = usable.groupby(
        [LABEL_COLUMNS['well'], LABEL_COLUMNS['site']], as_index=False
    ).first()
    if fields_per_well > 1:
        keep = (
            rows.groupby(LABEL_COLUMNS['well'], as_index=False)
            .head(fields_per_well)
        )
        rows = keep
    else:
        rows = rows.groupby(LABEL_COLUMNS['well'], as_index=False).first()

    rng = np.random.default_rng(seed)
    rows = rows.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    buckets = {name: [] for name in ('dp', 'ca', 'clip', 'o2')}
    records = []
    total = 0
    per_field_cap = max(1, max_cells // max(len(rows), 1))

    for _, row in rows.iterrows():
        names = {ch: row[col] for ch, col in CHANNEL_COLUMNS.items()}
        if not all(name in index for name in names.values()):
            continue
        planes = [load_tiff(index, names[ch]) for ch in CHANNELS]
        field_stack = np.stack(
            [p.astype(np.float32) for p in planes], axis=-1
        )  # (H, W, 5)

        nuclei = segment_nuclei(field_stack[..., 0])
        count = int(nuclei.max())
        if count == 0:
            continue
        centres = []
        for label_id in range(1, count + 1):
            mask = nuclei == label_id
            ys, xs = np.nonzero(mask)
            centres.append((ys.mean(), xs.mean()))

        # Cap per FIELD, not globally.  A global cap silently consumed the first
        # few fields and dropped the rest, which cut 65 wells down to 26 and
        # 54 compounds down to 22 -- destroying the benchmark's label diversity
        # while leaving the cell count looking reasonable.
        taken = 0
        for cell_index, (cy, cx) in enumerate(centres):
            if taken >= per_field_cap or total >= max_cells:
                break

            dp_crop = crop_pad(field_stack, cy, cx, 32)
            # PER-CHANNEL min-max, matching the official crop graph exactly.
            # `cropping.py:65-67` reduces `axis=[1, 2]` on a `(N, H, W, C)`
            # tensor, which leaves the channel axis intact -- so each channel is
            # normalised independently.  This used to be a joint min/max over the
            # whole crop, which squashed four channels into a narrow band
            # whenever one channel was much brighter.
            lo = dp_crop.min(axis=(0, 1), keepdims=True)
            hi = dp_crop.max(axis=(0, 1), keepdims=True)
            buckets['dp'].append(
                ((dp_crop - lo) / np.maximum(hi - lo, 1e-6)).astype(np.float32)
            )

            ca_crop = resize_to(crop_pad(field_stack, cy, cx, 256), 256)
            scaled = np.clip(ca_crop / max(ca_crop.max(), 1e-6) * 255.0, 0, 255)
            buckets['ca'].append(
                scaled.round().astype(np.uint8).transpose(2, 0, 1)
            )

            clip_crop = resize_to(crop_pad(field_stack, cy, cx, 224), 224)
            scaled_clip = np.clip(clip_crop / max(clip_crop.max(), 1e-6) * 255.0, 0, 255)
            buckets['clip'].append(scaled_clip.round().astype(np.uint8))

            # Nucleus silhouette: take the labelled nucleus, crop, then resample.
            singular = (nuclei == (cell_index + 1)).astype(np.float32)[..., None]
            mask64 = resize_to(crop_pad(singular, cy, cx, 64), 64)
            buckets['o2'].append(
                (mask64[..., 0] > 0.5).astype(np.float32)[None, ...]
            )

            records.append({
                'cell_id': f"{row[LABEL_COLUMNS['plate']]}_{row[LABEL_COLUMNS['well']]}_"
                           f"{row[LABEL_COLUMNS['site']]}_{cell_index}",
                'plate': str(row[LABEL_COLUMNS['plate']]),
                'well': str(row[LABEL_COLUMNS['well']]),
                'site': str(row[LABEL_COLUMNS['site']]),
                'compound': str(row[LABEL_COLUMNS['compound']]),
                'concentration': row[LABEL_COLUMNS['concentration']],
                'well_role': str(row[LABEL_COLUMNS['well_role']]),
                'broad_id': str(row[LABEL_COLUMNS['broad_id']]),
            })
            total += 1
            taken += 1
        if total >= max_cells:
            print(f'reached max_cells={max_cells}')
            break

    if total == 0:
        raise RuntimeError('no cells produced; check the archives and labels')

    arrays = {
        'dp': np.stack(buckets['dp']),
        'ca': np.stack(buckets['ca']),
        'clip': np.stack(buckets['clip']),
        'o2': np.stack(buckets['o2']),
    }
    for name, array in arrays.items():
        path = output_dir / f'cells_{name}.npy'
        np.save(path, array)
        print(f'  {name:<5} {str(array.shape):<24} {array.dtype}  -> {path.name}')

    meta = pd.DataFrame(records)
    meta_path = output_dir / 'meta.csv'
    meta.to_csv(meta_path, index=False)
    print(f'  meta  {meta.shape}  -> {meta_path.name}')

    summary = {
        'cells': int(total),
        'fields_used': int(len(rows)),
        'arrays': {k: list(v.shape) for k, v in arrays.items()},
        'channels': list(CHANNELS),
        'source': [p.name for p in source] if images_dir is None else str(images_dir),
        'segmentation': 'otsu + distance watershed on Hoechst, area-filtered',
        'crop_sizes': {'dp': 32, 'ca': 256, 'clip': 224, 'o2': 64},
        'note': ('O2-VAE receives a NUCLEUS silhouette, not a whole-cell mask; '
                 'BBBC022 has no reliably segmentable cytoplasm stain.'),
    }
    (output_dir / 'dataset_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zips', nargs='*', default=[])
    parser.add_argument('--images-dir', default=None,
                        help='use an already-extracted image tree instead of zips')
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-cells', type=int, default=3000)
    parser.add_argument('--fields-per-well', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not args.zips and not args.images_dir:
        parser.error('pass --zips or --images-dir')

    summary = build(
        [Path(p) for p in args.zips],
        Path(args.labels),
        Path(args.output_dir),
        images_dir=Path(args.images_dir) if args.images_dir else None,
        fields_per_well=args.fields_per_well,
        max_cells=args.max_cells,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
