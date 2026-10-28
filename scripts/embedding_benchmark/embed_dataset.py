"""Embed the BBBC022 single-cell dataset with every runnable method.

Runs one method at a time and in chunks.  Both are lessons from a failed first
attempt: DINOv2-Giant and CA-MAE together exhausted the 8 GB GPU, and CA-MAE's
adapter was handed all 3,000 cells at once (256x256x5 uint8 = 983 MB before any
activations).  Chunking keeps peak memory bounded; sequencing keeps the two big
models from competing for VRAM.

The dispatcher caches loaded adapters, so chunking does not reload weights per
chunk -- verified separately at ~7 s for the first call and ~0.03 s after.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Resolve `src/` from this file's location (scripts/embedding_benchmark/<file>)
# rather than hardcoding an absolute path, so the script runs from any checkout.
_SRC = Path(__file__).resolve().parents[2] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cellpaint_pipeline import models  # noqa: E402

#: method -> (array file, input unit, chunk size)
METHOD_INPUTS = {
    'deepprofiler': ('cells_dp.npy', 'single_cell_crop', 256),
    'ca_mae': ('cells_ca.npy', 'single_cell_crop', 32),
    'cellclip': ('cellclip_vectors.npy', 'channel_embeddings', 512),
    'o2vae': ('cells_o2.npy', 'single_cell_crop', 256),
}


def embed_in_chunks(model_name: str, array_path: Path, unit: str, chunk: int):
    payload = np.load(array_path, mmap_mode='r')
    total = payload.shape[0]
    outputs = []
    started = time.time()
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        block = np.asarray(payload[start:stop])
        result = models.embed(model_name, block, unit=unit)
        outputs.append(result.as_numpy())
    elapsed = time.time() - started
    stacked = np.concatenate(outputs, axis=0)
    return stacked, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells-dir', required=True)
    parser.add_argument('--methods', nargs='*', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    cells = Path(args.cells_dir)
    wanted = args.methods or list(METHOD_INPUTS)

    summary = {}
    for name in wanted:
        if name not in METHOD_INPUTS:
            print(f'skip {name}: no prepared input')
            continue
        filename, unit, chunk = METHOD_INPUTS[name]
        source = cells / filename
        target = cells / f'emb_{name}.npy'
        if not source.exists():
            print(f'skip {name}: {source} missing')
            continue
        if target.exists() and not args.force:
            existing = np.load(target, mmap_mode='r')
            print(f'skip {name}: {target.name} already exists {existing.shape}')
            summary[name] = list(existing.shape)
            continue

        print(f'--- {name} <- {filename} (unit={unit}, chunk={chunk})')
        try:
            embeddings, elapsed = embed_in_chunks(name, source, unit, chunk)
            np.save(target, embeddings)
            print(f'    {embeddings.shape} in {elapsed:.1f} s '
                  f'({embeddings.shape[0]/max(elapsed,1e-9):.1f} cells/s)')
            summary[name] = list(embeddings.shape)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f'    FAILED {type(exc).__name__}: {str(exc)[:200]}')
            summary[name] = f'error: {type(exc).__name__}'

    (cells / 'embedding_run.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
