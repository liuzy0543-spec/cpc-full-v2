from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parents[4]
RAW_ROOT = ROOT / "demo" / "backend" / "profiling_backend" / "data" / "raw_core" / "BR00000001"
LABEL_ROOT = ROOT / "demo" / "backend" / "segmentation_backend" / "outputs" / "cellprofiler_masks" / "labels"


SITE_TO_WELL = {
    "r01c01f01p01": "A01",
    "r01c02f01p01": "A02",
}


CHANNEL_SPECS = {
    1: ("OrigMito", 900, 18000),
    2: ("OrigAGP", 850, 20000),
    3: ("OrigRNA", 950, 26000),
    4: ("OrigER", 900, 17000),
    5: ("OrigDNA", 1100, 52000),
    6: ("OrigBrightfield", 1400, 6000),
    7: ("OrigHighZBF", 1500, 5500),
    8: ("OrigLowZBF", 1450, 5000),
}


def _soft_mask(mask: np.ndarray, *, sigma: float) -> np.ndarray:
    softened = ndi.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    peak = float(softened.max())
    if peak <= 0:
        return softened
    return softened / peak


def _ring_mask(cell_mask: np.ndarray, nuclei_mask: np.ndarray) -> np.ndarray:
    cytoplasm = np.logical_and(cell_mask, ~nuclei_mask)
    ring = ndi.binary_dilation(nuclei_mask, iterations=4) & cytoplasm
    return _soft_mask(ring, sigma=1.2)


def _structured_background(height: int, width: int, *, seed: int, base: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    gradient = (0.8 * yy + 1.2 * xx) / float(height + width)
    noise = ndi.gaussian_filter(rng.normal(0.0, 1.0, size=(height, width)), sigma=3.0)
    noise = (noise - noise.min()) / max(noise.max() - noise.min(), 1e-6)
    return base + 180.0 * gradient + 120.0 * noise


def _build_channel(
    channel_index: int,
    *,
    background: np.ndarray,
    nuclei_soft: np.ndarray,
    cell_soft: np.ndarray,
    ring_soft: np.ndarray,
    seed: int,
) -> np.ndarray:
    _, base_level, object_scale = CHANNEL_SPECS[channel_index]
    rng = np.random.default_rng(seed)
    signal = np.array(background, copy=True)

    if channel_index == 5:
        signal += object_scale * nuclei_soft
        signal += 3500.0 * ring_soft
    elif channel_index == 3:
        signal += object_scale * cell_soft
        signal += 3000.0 * nuclei_soft
    elif channel_index == 2:
        signal += object_scale * ring_soft
        signal += 9000.0 * cell_soft
    elif channel_index == 4:
        signal += object_scale * ring_soft
        signal += 5000.0 * cell_soft
    elif channel_index == 1:
        signal += object_scale * ring_soft
        signal += 4500.0 * cell_soft
    else:
        signal += object_scale * (0.35 * cell_soft + 0.15 * nuclei_soft)

    signal += rng.normal(0.0, 140.0, size=signal.shape)
    signal = ndi.gaussian_filter(signal, sigma=0.6)
    signal = np.clip(signal, base_level, 60000.0)
    return signal.astype(np.uint16)


def regenerate_demo_raw_images() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    for site_prefix, well in SITE_TO_WELL.items():
        cell_labels = tifffile.imread(LABEL_ROOT / f"BR00000001_{well}_s1--cell_labels.tiff")
        nuclei_labels = tifffile.imread(LABEL_ROOT / f"BR00000001_{well}_s1--nuclei_labels.tiff")

        cell_mask = cell_labels > 0
        nuclei_mask = nuclei_labels > 0
        nuclei_soft = _soft_mask(nuclei_mask, sigma=1.0)
        cell_soft = _soft_mask(cell_mask, sigma=2.2)
        ring_soft = _ring_mask(cell_mask, nuclei_mask)

        height, width = cell_labels.shape
        for channel_index in range(1, 9):
            _, base_level, _ = CHANNEL_SPECS[channel_index]
            background = _structured_background(
                height,
                width,
                seed=hash((site_prefix, channel_index, "background")) & 0xFFFFFFFF,
                base=base_level,
            )
            image = _build_channel(
                channel_index,
                background=background,
                nuclei_soft=nuclei_soft,
                cell_soft=cell_soft,
                ring_soft=ring_soft,
                seed=hash((site_prefix, channel_index, "channel")) & 0xFFFFFFFF,
            )
            filename = RAW_ROOT / f"{site_prefix}-ch{channel_index}sk1fk1fl1.tiff"
            tifffile.imwrite(filename, image)
            print(f"wrote {filename.relative_to(ROOT)} shape={image.shape} dtype={image.dtype} min={int(image.min())} max={int(image.max())}")


if __name__ == "__main__":
    regenerate_demo_raw_images()
