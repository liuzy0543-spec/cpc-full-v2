"""Build embeddings for the three surrogate-backed methods (``v3`` profile).

The four official-weights methods are produced by ``embed_dataset.py`` from
arrays that were made once and verified.  The three surrogate methods need their
inputs assembled differently, and pretending otherwise would have meant bending
``embed_dataset.py`` around three special cases:

* **unidino** takes 5-channel crops and does its own 16-bit rescale, so a raw
  uint16 crop is enough -- ``assume='raw_crop'`` runs the declared preprocessing.
* **dinocells** consumes the five image channels as a stack and **discards the
  mask**, which is what the official dataset does: it folds its stored tape
  back open and drops the mask column before the network sees it
  (``data_utils/cellpainting_dataset.py:23-47``, ``:86``).  A mask is therefore
  optional here, and a batch is fine -- the official model batches like any
  image stack.
* **morphogenie** declares ``matlab_centring``, which ``prepare`` refuses because
  the real step is not in the repository.  Its surrogate therefore takes a
  plainly rescaled 3-channel crop.  That this is *not* the official
  preprocessing is the whole reason the method was cancelled, and it is why its
  scores are not comparable to anything.

Everything goes through ``models.embed`` -- the same entry point a user calls --
so this script also serves as the end-to-end proof that the dispatch layer drives
all seven methods.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src'))

from cellpaint_pipeline import models  # noqa: E402
from cellpaint_pipeline.models import registry  # noqa: E402

__all__ = ['main']

#: Channels morphogenie's surrogate is given, in the order its manifest declares.
MORPHOGENIE_CHANNELS = ('C0', 'C1', 'C2')


def _otsu_threshold(plane: np.ndarray) -> float:
    """Otsu threshold for one plane (see the note in ``train_surrogates.py``)."""
    values = plane.ravel().astype(np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=256)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(edges[0])
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cumulative = np.cumsum(hist * centres)
    mean_bg = np.divide(cumulative, weight_bg, out=np.zeros_like(cumulative),
                        where=weight_bg > 0)
    mean_fg = np.divide(cumulative[-1] - cumulative, weight_fg,
                        out=np.zeros_like(cumulative), where=weight_fg > 0)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~np.isfinite(between)] = -1.0
    return float(centres[int(np.argmax(between))])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells-dir', default=r'E:\bbbc022\cells')
    parser.add_argument('--methods', nargs='*',
                        default=['unidino', 'dinocells', 'morphogenie'])
    parser.add_argument('--chunk', type=int, default=125)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    registry.set_profile('v3')
    cells = Path(args.cells_dir)
    crops = np.load(cells / 'cells_ca.npy', mmap_mode='r')   # (N, 5, 256, 256) uint8
    total = crops.shape[0]
    print(f'cells {total}  methods {args.methods}  profile {registry.active_profile()}')

    report = {}
    for name in args.methods:
        manifest = registry.manifest(name)
        target = cells / f'emb_{name}.npy'
        if target.exists() and not args.force:
            print(f'--- {name}: {target.name} exists, skipped (use --force)')
            continue

        print(f'--- {name}  [SURROGATE]  unit={manifest.input_unit}')
        pieces = []
        first_result = None
        for start in range(0, total, args.chunk):
            block = np.asarray(crops[start:start + args.chunk])
            stop = start + len(block)

            if name == 'unidino':
                payload = np.moveaxis(block, 1, -1).astype('uint16') * 257
                result = models.embed(name, payload, assume='raw_crop')
                first_result = first_result or result
                values = result.as_numpy()

            elif name == 'dinocells':
                # The official model input is a channel stack with the mask
                # dropped, so the whole block goes through in one call -- no
                # per-cell loop, because a tape is no longer the input format.
                payload = np.moveaxis(block, 1, -1).astype('uint16') * 257
                result = models.embed(name, payload, assume='raw_crop')
                first_result = first_result or result
                values = result.as_numpy()

            elif name == 'morphogenie':
                payload = block[:, :len(MORPHOGENIE_CHANNELS)].astype('float32') / 255.0
                result = models.embed(
                    name, payload, channels=MORPHOGENIE_CHANNELS
                )
                first_result = first_result or result
                values = result.as_numpy()

            else:
                raise SystemExit(f'{name} is not a surrogate-backed method')

            pieces.append(np.asarray(values, dtype='float32'))
            print(f'    {stop:5d}/{total}  {values.shape}')

        matrix = np.concatenate(pieces, axis=0)
        if matrix.shape[0] != total:
            raise SystemExit(
                f'{name}: produced {matrix.shape[0]} rows for {total} cells'
            )
        np.save(target, matrix)
        print(f'    wrote {target.name} {matrix.shape}')

        # Provenance read off a result object rather than restated here, so a
        # drift between the manifest and what `embed` actually returned would
        # show up in this file instead of being papered over.
        report[name] = {
            'shape': list(matrix.shape),
            'weights_provenance': first_result.metadata.get('weights_provenance'),
            'weights_sha256': first_result.weights_sha256,
            'determinism': first_result.determinism,
            'license': first_result.license,
            'input_unit': str(first_result.input_unit),
        }

    if report:
        out = cells / 'v3_surrogate_embeddings.json'
        existing = {}
        if out.exists():
            existing = json.loads(out.read_text(encoding='utf-8'))
        existing.update(report)
        out.write_text(json.dumps(existing, indent=2), encoding='utf-8')
        print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())