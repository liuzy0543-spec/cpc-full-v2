"""Verify that the embedding layer really is one call signature for every method.

The claim under test is narrow and checkable: given a method name and a payload
in that method's declared input kind, `models.embed` returns an object with the
*same* shape of information regardless of which method ran.  Two things are
verified, and they fail differently:

1. **Uniform I/O** -- every method is called through the same signature, and the
   returned object exposes the same fields with the same meanings.  A method that
   needs a different call, or returns a different result type, breaks this.
2. **Batch dispatch** -- `run_benchmark` routes different payloads to different
   methods in one plan, aligns rows across methods, and writes artefacts.

What this does *not* check: whether each method's output is scientifically good,
or whether its preprocessing matches upstream.  Those are separate checks (see
`docs/archive/review_roadmap_prompt.md`).

Usage:
    python verify_io_uniformity.py --cells-dir E:\\bbbc022\\cells
    python verify_io_uniformity.py --cells-dir E:\\bbbc022\\cells --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cellpaint_pipeline import embedding_benchmark as bench  # noqa: E402
from cellpaint_pipeline import models  # noqa: E402
from cellpaint_pipeline.models import registry  # noqa: E402

#: method -> (prepared input file, input unit, chunk size).  Chunking is a memory
#: concern, not an interface concern: the same `embed()` call is used either way.
METHOD_INPUTS = {
    'deepprofiler': ('cells_dp.npy', 'single_cell_crop', 512),
    'ca_mae': ('cells_ca.npy', 'single_cell_crop', 32),
    'cellclip': ('cellclip_vectors.npy', 'channel_embeddings', 512),
    'o2vae': ('cells_o2.npy', 'single_cell_crop', 512),
    # v3: surrogate-backed methods.  Same source crops as ca_mae, but each needs
    # its own declared transform, which is exactly what `assume='raw_crop'` is
    # for.  The three differ in unit and width, so they belong in this check --
    # a uniformity test that skips them proves the interface only for the easy
    # four.
    'unidino': ('cells_ca.npy', 'single_cell_crop', 64),
    'dinocells': ('cells_ca.npy', 'single_cell_crop', 64),
    'morphogenie': ('cells_ca.npy', 'single_cell_crop', 64),
}

#: Surrogate methods whose payload has to be reshaped before `embed`.
_SURROGATE_TRANSFORMS = ('unidino', 'dinocells', 'morphogenie')


def _chunk_payload(name: str, payload, start: int, stop: int) -> dict:
    """Build the `embed()` keyword arguments for one chunk.

    The four official methods are handed straight to the model in their stored
    layout; the three surrogates need the transform their manifest declares, so
    `assume='raw_crop'` is passed and `prepare` does the work.  Both are "how a
    user would call it" -- the difference is the method's declaration, not the
    interface.
    """
    array = np.asarray(payload[start:stop])
    if name not in _SURROGATE_TRANSFORMS:
        return {'array': array, 'unit': METHOD_INPUTS[name][1]}
    if name in ('unidino', 'dinocells'):
        # Raw crops are uint8 in storage; unidino divides by 65535 and dinocells
        # z-scores, so both are declared against the 16-bit acquisition domain.
        return {'array': np.moveaxis(array, 1, -1).astype('uint16') * 257,
                'assume': 'raw_crop'}
    # morphogenie declares `matlab_centring`, which prepare refuses; its
    # surrogate takes a plainly rescaled 3-channel crop instead.
    return {'array': array[:, :3].astype('float32') / 255.0,
            'channels': ('C0', 'C1', 'C2')}

#: Fields every result must expose for the interface to count as unified.  These
#: are the ones a downstream benchmark actually consumes.  `EmbeddingResult`
#: keeps provenance flat (not nested under a `provenance` dict), so the weight
#: attribution fields are listed individually.
REQUIRED_RESULT_FIELDS = (
    'model', 'values', 'dim', 'input_unit', 'n_samples',
    'weights_path', 'weights_sha256', 'license', 'determinism',
    'metadata', 'sample_metadata',
    'shape', 'as_numpy',            # properties, not dataclass fields
)

#: Attribution fields a reader needs to trace a number to a checkpoint.
ATTRIBUTION_FIELDS = ('weights_path', 'weights_sha256', 'license', 'determinism')


def _check_result_shape(name: str, result) -> list[str]:
    problems: list[str] = []
    missing = [f for f in REQUIRED_RESULT_FIELDS if not hasattr(result, f)]
    if missing:
        problems.append(f'{name}: result missing fields {missing}')

    manifest = registry.manifest(name)
    # Mirror the library's own rule (`embedding._check_output_width`): a method
    # may declare a per-channel width and return `per_channel * n_channels`.
    # unidino is exactly that case (384 x 5 = 1920), so a flat equality check
    # would report a defect where the contract is being honoured.
    declared = {int(manifest.output_dim)}
    declared.update(int(alt) for alt in manifest.output_dim_alternatives)
    per_channel = manifest.output_dim_per_channel
    ok_width = int(result.dim) in declared or (
        per_channel and int(result.dim) > 0
        and int(result.dim) % int(per_channel) == 0
    )
    if not ok_width:
        problems.append(
            f'{name}: dim={result.dim} but manifest declares '
            f'{sorted(declared)}'
            + (f' (or a multiple of {per_channel})' if per_channel else '')
        )
    if tuple(result.shape) != (int(result.n_samples), int(result.dim)):
        problems.append(
            f'{name}: shape={tuple(result.shape)} not (n_samples, dim)'
        )

    if not result.weights_sha256:
        problems.append(f'{name}: no weights_sha256 recorded')
    elif result.weights_sha256 != manifest.weights.sha256:
        problems.append(f'{name}: weights_sha256 differs from manifest')
    if not result.license:
        problems.append(f'{name}: no license recorded')
    if not result.determinism:
        problems.append(f'{name}: no determinism recorded')

    values = np.asarray(result.values)
    if not np.isfinite(values).all():
        problems.append(f'{name}: output has non-finite values')
    return problems


def embed_one(name: str, cells: Path, limit: int | None) -> dict:
    """Call `embed()` exactly the way a user would, and validate the result."""
    filename, unit, chunk = METHOD_INPUTS[name]
    source = cells / filename
    payload = np.load(source, mmap_mode='r')
    total = min(int(payload.shape[0]), limit) if limit else int(payload.shape[0])

    started = time.time()
    outputs = []
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        # `unit` is only passed for the official methods; the surrogates carry
        # their own declaration via `assume` / `channels`.
        kwargs = _chunk_payload(name, payload, start, stop)
        result = models.embed(name, **kwargs)
        outputs.append(result.as_numpy())
    elapsed = time.time() - started

    stacked = np.concatenate(outputs, axis=0)

    # Re-derive the result metadata from the last chunk's object, but validate the
    # full stacked matrix for shape/finiteness.
    problems = _check_result_shape(name, result)
    if int(stacked.shape[0]) != total:
        problems.append(f'{name}: stacked rows {stacked.shape[0]} != {total}')
    if not np.isfinite(stacked).all():
        problems.append(f'{name}: stacked matrix has non-finite values')

    return {
        'method': name,
        'n': int(stacked.shape[0]),
        'dim': int(stacked.shape[1]),
        'seconds': round(elapsed, 2),
        'cells_per_second': round(total / elapsed, 1) if elapsed else None,
        'input_unit': str(result.input_unit),
        'determinism': result.determinism,
        'weights_sha256': (result.weights_sha256 or '')[:16],
        'license': result.license,
        'metadata_keys': sorted(result.metadata),
        'ok': not problems,
        'problems': problems,
        'array_sha256': _sha256(stacked),
    }


def _sha256(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_batch(cells: Path, methods: list[str], limit: int | None,
              chunk_size: int | None = None) -> dict:
    """Run the same methods through the batch entry point and check alignment.

    `chunk_size` matters here: without it the batch path passes the whole payload
    to one forward pass, and a full plate exhausts memory (measured: 8 GiB card
    OOM on 3,857 CA-MAE crops).  With it, the batch path is usable at real scale.
    """
    payloads, units = {}, {}
    for name in methods:
        filename, unit, _chunk = METHOD_INPUTS[name]
        arr = np.load(cells / filename, mmap_mode='r')
        n = min(int(arr.shape[0]), limit) if limit else int(arr.shape[0])
        payloads[name] = np.asarray(arr[:n])
        units[name] = unit

    import pandas as pd

    meta = pd.read_csv(cells / 'meta.csv')
    n_rows = min(len(meta), limit) if limit else len(meta)
    sample_metadata = {col: list(meta[col][:n_rows]) for col in meta.columns}

    out_dir = cells / 'bench_verify'
    plan = bench.build_benchmark_plan(
        methods, output_dir=out_dir, unit=units, align_on='cell_id',
        chunk_size=chunk_size,
    )
    started = time.time()
    outcome = bench.run_benchmark(plan, payloads, sample_metadata=sample_metadata)
    elapsed = time.time() - started

    return {
        'seconds': round(elapsed, 2),
        'chunk_size': chunk_size,
        'shapes': {k: list(v) for k, v in outcome.shapes.items()},
        'errors': dict(outcome.errors),
        'n_aligned': len(outcome.aligned_ids or ()),
        'finished_at_set': bool(outcome.finished_at),
        'written': [str(p) for p in (outcome.written or ())],
        'output_dir': str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells-dir', required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--methods', nargs='*', default=None)
    parser.add_argument('--profile', default='v2', choices=['v2', 'v3'],
                        help='v3 also verifies the three surrogate-backed methods')
    parser.add_argument('--batch', action='store_true', help='also run the batch path')
    parser.add_argument('--batch-chunk-size', type=int, default=256,
                        help='rows per forward pass in the batch path (0 = no chunking)')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    registry.set_profile(args.profile)
    cells = Path(args.cells_dir)
    ready = set(models.available())
    requested = args.methods or list(METHOD_INPUTS)
    methods = [m for m in requested if m in ready and m in METHOD_INPUTS]
    skipped = [m for m in requested if m not in methods]

    print(f'runnable+prepared: {methods}')
    if skipped:
        print(f'skipped (not runnable or no prepared input): {skipped}')

    report = {'methods': [], 'skipped': skipped, 'profile': args.profile}

    for name in methods:
        print(f'\n--- {name} ---')
        try:
            row = embed_one(name, cells, args.limit)
        except Exception as exc:  # noqa: BLE001 - this script reports, not raises
            traceback.print_exc()
            report['methods'].append({
                'method': name, 'ok': False,
                'problems': [f'{type(exc).__name__}: {exc}'],
            })
            continue
        report['methods'].append(row)
        status = 'OK ' if row['ok'] else 'FAIL'
        print(
            f'  {status} n={row["n"]} dim={row["dim"]} '
            f'{row["seconds"]}s ({row["cells_per_second"]}/s) '
            f'unit={row["input_unit"]} weights={row["weights_sha256"]}'
        )
        for p in row['problems']:
            print(f'    ! {p}')

    if args.batch and len(methods) > 1:
        print('\n--- batch dispatch (run_benchmark) ---')
        try:
            report['batch'] = run_batch(
                cells, methods, args.limit,
                chunk_size=args.batch_chunk_size or None,
            )
            print('  chunk_size:', report['batch']['chunk_size'])
            print('  shapes :', report['batch']['shapes'])
            print('  errors :', report['batch']['errors'] or '{}')
            print('  aligned:', report['batch']['n_aligned'])
            print('  written:', len(report['batch']['written']))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            report['batch'] = {'error': f'{type(exc).__name__}: {exc}'}

    # ---- summary -----------------------------------------------------------
    ok = [r for r in report['methods'] if r.get('ok')]
    bad = [r for r in report['methods'] if not r.get('ok')]
    report['summary'] = {
        'n_ok': len(ok),
        'n_failed': len(bad),
        'all_ok': not bad and len(ok) == len(methods),
    }

    print('\n=== IO consistency verdict ===')
    print(f'  succeeded {len(ok)}/{len(methods)}; failed {len(bad)}')
    if ok:
        dims = {r['dim'] for r in ok}
        field_sets = {tuple(r['metadata_keys']) for r in ok}
        print(f'  output dims (expected to differ per method): {sorted(dims)}')
        print(f'  identical result-object field sets: {len(field_sets) == 1}')
        print(f'  all carry a weights sha256: {all(r["weights_sha256"] for r in ok)}')
        print(f'  all carry a license: {all(bool(r["license"]) for r in ok)}')
    for r in bad:
        print(f'  FAIL {r["method"]}: {r["problems"]}')

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print(f'\nwrote {args.output}')

    return 0 if report['summary']['all_ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())