"""Compare this project's DeepProfiler adapter against the official pipeline.

Why this is not a one-line assertion
------------------------------------
"Same as the original DeepProfiler" has two different meanings here, and they
disagree:

* **Implementation fidelity** -- does our crop/normalise/forward/reduce chain
  reproduce exactly what the official `deepprofiler` package produced on the same
  data?  This is checkable bitwise against the official run's own `.npz`.
* **Upstream training semantics** -- does the channel order match what the
  published checkpoint was TRAINED on?  Upstream says `DNA, ER, RNA, AGP, Mito`
  (`D:\\official_src\\1-deepprofiler\\README.md:74`,
  `tests\\deepprofiler\\profiling\\test_cell_painting_cnn_v1.py:44-48,74`).

The catch: the demo run in this project used `DNA, RNA, ER, AGP, Mito` -- i.e. RNA
and ER swapped. So a bitwise match against that demo proves fidelity of the
*code path*, while the corrected order necessarily *differs* from it. Reporting
only one of the two would be misleading, so both are measured.

Usage:
    python parity_vs_official_dp.py
    python parity_vs_official_dp.py --official-root D:\\other\\demo --wells A01
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import tifffile  # noqa: E402
import tensorflow as tf  # noqa: E402

from cellpaint_pipeline import models  # noqa: E402

#: Order the checkpoint was TRAINED on (upstream README + test fixtures).
UPSTREAM_CHANNELS = ('DNA', 'ER', 'RNA', 'AGP', 'Mito')
#: Order this project's demo run used, which swapped RNA and ER.
DEMO_CHANNELS = ('DNA', 'RNA', 'ER', 'AGP', 'Mito')
#: channel name -> file suffix, as the demo config maps them.
DEMO_SUFFIX = {'DNA': 'ch5', 'RNA': 'ch3', 'ER': 'ch4', 'AGP': 'ch2', 'Mito': 'ch1'}

BOX = 32
IMAGE_SIZE = 64
DEFAULT_OFFICIAL_ROOT = Path(
    r'D:\cpc_c_official\demo\workspace\outputs\dp_run\deepprofiler_project'
)
DEFAULT_RAW_ROOT = Path(
    r'D:\cpc_c_official\demo\backend\profiling_backend\data\raw_core\BR00000001'
)


def load_field(raw_root: Path, well: str, channels: tuple[str, ...]) -> np.ndarray:
    """Stack one field's channels -> (H, W, C) float32, in the given order."""
    planes = []
    for name in channels:
        suffix = DEMO_SUFFIX[name]
        candidates = sorted(raw_root.glob(f'*c{well[1:3]}f01p01-{suffix}*.tiff'))
        if not candidates:
            raise FileNotFoundError(f'no {name} TIFF for well {well}')
        planes.append(tifffile.imread(str(candidates[0])).astype(np.float32))
    return np.stack(planes, axis=-1)


def read_locations(project: Path, well: str) -> np.ndarray:
    path = project / 'inputs' / 'locations' / 'BR00000001' / f'{well}-1-Nuclei.csv'
    xs, ys = [], []
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row['Nuclei_Location_Center_X']))
            ys.append(float(row['Nuclei_Location_Center_Y']))
    return np.array(list(zip(xs, ys)), dtype=np.float64)


def crop_with_official_op(image: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Crop with TensorFlow's `crop_and_resize`, the op the official code uses.

    Reimplementing the sampling grid in NumPy was tried and abandoned: it
    disagreed with this op by ~1.4e4 on real data. Use the same op.
    """
    half = (BOX / 2) / IMAGE_SIZE
    boxes = np.array([
        [
            (cy / IMAGE_SIZE) - half,
            (cx / IMAGE_SIZE) - half,
            (cy / IMAGE_SIZE) + half,
            (cx / IMAGE_SIZE) + half,
        ]
        for cx, cy in centres
    ], dtype=np.float32)
    with tf.compat.v1.Session() as session:
        return session.run(
            tf.image.crop_and_resize(
                tf.constant(image[np.newaxis, ...].astype(np.float32)),
                tf.constant(boxes),
                tf.constant(np.zeros(len(boxes), np.int32)),
                [BOX, BOX],
                method='bilinear',
            )
        )


def per_crop_minmax(crops: np.ndarray) -> np.ndarray:
    """Official `cropping.py:65-67`: per-crop, per-channel min-max onto [0, 1]."""
    mini = crops.min(axis=(1, 2), keepdims=True)
    maxi = crops.max(axis=(1, 2), keepdims=True)
    return (crops - mini) / (maxi - mini + 1e-7)


def compare(got: np.ndarray, expected: np.ndarray) -> tuple[bool, float]:
    if got.shape != expected.shape:
        return False, float('nan')
    return bool(np.array_equal(got, expected)), float(np.abs(got - expected).max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-root', default=str(DEFAULT_OFFICIAL_ROOT))
    parser.add_argument('--raw-root', default=str(DEFAULT_RAW_ROOT))
    parser.add_argument('--wells', nargs='*', default=['A01', 'A02'])
    args = parser.parse_args()

    project = Path(args.official_root)
    raw_root = Path(args.raw_root)

    official, crops_demo, crops_upstream = {}, {}, {}
    for well in args.wells:
        npz_path = (
            project / 'outputs' / 'cell_painting_cnn_demo' / 'features'
            / 'BR00000001' / well / '1.npz'
        )
        with np.load(npz_path, allow_pickle=True) as payload:
            official[well] = np.asarray(payload['features'])

        centres = read_locations(project, well)
        crops_demo[well] = crop_with_official_op(
            load_field(raw_root, well, DEMO_CHANNELS), centres
        )
        crops_upstream[well] = crop_with_official_op(
            load_field(raw_root, well, UPSTREAM_CHANNELS), centres
        )
        print(f'  {well}: cells={len(centres)} crops={crops_demo[well].shape} '
              f'official_features={official[well].shape}')

    expected = np.concatenate([official[w] for w in args.wells], axis=0)
    demo = np.concatenate([crops_demo[w] for w in args.wells], axis=0).astype('float32')
    upstream = np.concatenate(
        [crops_upstream[w] for w in args.wells], axis=0
    ).astype('float32')

    print(f'\n  official features : {expected.shape}')
    print(f'  crop value range  : [{demo.min():.1f}, {demo.max():.1f}] (raw 16-bit)')

    cases = [
        (
            'A. demo order + per-crop min-max',
            per_crop_minmax(demo), DEMO_CHANNELS, True,
            'must match the official run bitwise: same code path, same order',
        ),
        (
            'B. demo order + raw pixels',
            demo, DEMO_CHANNELS, True,
            'shows the size of the normalisation mistake (contract blocks this)',
        ),
        (
            'C. upstream order + per-crop min-max',
            per_crop_minmax(upstream), UPSTREAM_CHANNELS, False,
            'differs from the old demo BY DESIGN: RNA/ER are no longer swapped',
        ),
    ]

    print('\n  === does our adapter reproduce the official pipeline? ===')
    rows = []
    for label, payload, channels, skip, _note in cases:
        result = models.embed(
            'deepprofiler', payload, unit='single_cell_crop',
            channels=channels, skip_contract_check=skip,
        )
        got = result.as_numpy()
        equal, maxdiff = compare(got, expected)
        rows.append((label, got.shape, equal, maxdiff))
        print(f'  {label:<38} shape={got.shape} '
              f'bitwise_equal={equal} max|diff|={maxdiff:.6e}')

    print('\n  what each case means:')
    for (label, _shape, _equal, _maxdiff), (_l, _p, _c, _s, note) in zip(rows, cases):
        print(f'    {label}: {note}')

    # ---- contract behaviour -------------------------------------------------
    print('\n  === contract guards ===')
    guards = []
    try:
        models.embed('deepprofiler', demo, unit='single_cell_crop',
                     channels=DEMO_CHANNELS)
        guards.append(('old channel order rejected', False, 'accepted'))
    except Exception as exc:  # noqa: BLE001 - the point is that it raises
        guards.append(('old channel order rejected', True, type(exc).__name__))

    try:
        models.embed('deepprofiler', demo, unit='single_cell_crop',
                     channels=UPSTREAM_CHANNELS)
        guards.append(('16-bit raw input rejected', False, 'accepted'))
    except Exception as exc:  # noqa: BLE001
        guards.append(('16-bit raw input rejected', True, type(exc).__name__))

    for name, ok, detail in guards:
        print(f'  {name:<32} {"PASS" if ok else "FAIL"} ({detail})')

    verdict = {
        'implementation_fidelity': rows[0][2],
        'fidelity_max_abs_diff': rows[0][3],
        'raw_pixel_error': rows[1][3],
        'upstream_order_differs_from_demo': not rows[2][2],
    }
    print('\n  === verdict ===')
    for key, value in verdict.items():
        print(f'    {key}: {value}')

    ok = rows[0][2] and all(g[1] for g in guards)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())