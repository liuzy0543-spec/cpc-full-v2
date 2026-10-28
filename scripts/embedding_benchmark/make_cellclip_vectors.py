"""Produce CellCLIP's real input: per-channel DINOv2-Giant embeddings.

CellCLIP does not consume pixels.  Its contract is ``(N, 5, 1536)``: for each
cell, one 1536-dim vector per channel, produced by DINOv2-Giant (ViT-g/14,
hidden 1536) run on each channel independently.  Without those vectors CellCLIP
can only be handed fabricated numbers, which is what earlier runs did.

This reproduces the upstream recipe from
`CellCLIP/preprocessing/convert_npz_to_avg_emb.py`:
  * crop size comes from the processor (`crop_size.height` = 224) and
    `processor.do_resize` is set False (line 520), so no further resize happens
  * each channel is repeated across RGB (lines 136-138) before the processor
  * the embedding is `model(batch).pooler_output` (line 357), the CLS token
  * one crop per cell is used (`n_crops=1`): the upstream default of 5 crops is
    then averaged, and with a single deterministic crop that reduces to the crop
    itself.  Random crop sampling would make the benchmark non-reproducible,
    which is the opposite of what it is for.

Output dtype is float32 and the shape is (N, 5, 1536), matching what
`CellClipAdapter` validates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Resolve `src/` from this file's location so the script imports the package from
# any checkout instead of relying on PYTHONPATH or a hardcoded absolute path.
_SRC = Path(__file__).resolve().parents[2] / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def load_dinov2(model_dir: Path, device: str = 'cpu'):
    """Load DINOv2-Giant plus its image processor from a local directory.

    Loaded from disk rather than the hub so the run does not depend on network
    access at benchmark time, and so the exact checkpoint is attributable.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(str(model_dir))
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    # Upstream disables the processor's resize (convert_npz_to_avg_emb.py:520);
    # the crops are already the right size, so resizing would only blur them.
    processor.do_resize = False
    model.eval().to(device)
    return model, processor


def embed_crops(
    crops: np.ndarray,
    model,
    processor,
    *,
    device: str = 'cpu',
    batch_size: int = 8,
) -> np.ndarray:
    """``(N, H, W, 5)`` uint8 crops -> ``(N, 5, 1536)`` float32 vectors."""
    import torch

    cells, height, width, channels = crops.shape
    vectors = np.zeros((cells, channels, 1536), dtype=np.float32)

    for start in range(0, cells, batch_size):
        stop = min(start + batch_size, cells)
        batch = crops[start:stop]

        # (cells, 5, H, W, 3): replicate each channel across RGB, as upstream does.
        rgb = np.repeat(batch.astype(np.float32)[..., None], 3, axis=-1)
        rgb = rgb.transpose(0, 3, 1, 2, 4)  # -> (cells, channels, H, W, 3)

        flat = rgb.reshape(-1, height, width, 3)
        inputs = processor(
            list(flat), return_tensors='pt', do_rescale=True
        )['pixel_values'].to(device)

        with torch.no_grad():
            pooled = model(inputs).pooler_output  # (cells*channels, 1536)

        vectors[start:stop] = (
            pooled.detach().cpu().numpy().reshape(stop - start, channels, 1536)
        )

    return vectors


#: Channel order of the five slots in the crop array.
#:
#: It is the same order the dataset builder writes (``CHANNELS`` in
#: ``build_bbbc022_cells.py``) and the same order CellCLIP's own preprocessing
#: documents (``preprocess_images_jumpcp.py:78-83``: DNA, ER, RNA, AGP, Mito).
#: It has to be recorded explicitly: the *count* alone does not say which slot
#: is which, and this project has been bitten by an ER/RNA swap before.
CHANNEL_ORDER = ('DNA', 'ER', 'RNA', 'AGP', 'Mito')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crops', required=True,
                        help='cells_clip.npy: (N, 224, 224, 5) uint8')
    parser.add_argument('--model-dir', required=True,
                        help='directory holding DINOv2-Giant config + weights')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    # Memory-mapped: the crop array is 752 MB for 3,000 cells, and loading it
    # while a 4.5 GB model is resident was enough to exhaust RAM alongside a
    # second embedding job.  Reading slices keeps only the current batch hot.
    crops = np.load(args.crops, mmap_mode='r')
    print(f'crops {crops.shape} {crops.dtype}')
    if crops.dtype != np.uint8:
        raise SystemExit('expected uint8 crops')

    model, processor = load_dinov2(Path(args.model_dir), args.device)
    print(f"processor crop_size={processor.crop_size} do_resize={processor.do_resize}")

    vectors = embed_crops(
        crops, model, processor, device=args.device, batch_size=args.batch_size
    )
    print(f'vectors {vectors.shape} {vectors.dtype}')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vectors)

    # Hash the upstream model this artefact depends on.  Without it the recorded
    # provenance starts at CellCLIP and silently omits the 4.5 GB model that
    # actually produced these vectors, so the chain cannot be re-created.
    from cellpaint_pipeline.models.weights import verify_sha256

    weights_file = Path(args.model_dir) / 'model.safetensors'
    model_sha256 = (
        verify_sha256(weights_file) if weights_file.is_file() else None
    )

    # The sidecar is the ONLY record of where these vectors came from, so it
    # has to identify every link:
    #
    #   * the model that produced them (by digest, not by path -- a path can
    #     still point at a different file tomorrow);
    #   * the crops they were computed from (by digest);
    #   * the channel order the five slots are in (the count alone does not
    #     say which slot is which, and this project has been bitten by an
    #     ER/RNA swap before);
    #   * the vectors themselves (by digest) -- so a later run can tell
    #     whether it is looking at the same artefact.
    #
    # Without the last two, a changed input file changes every embedding and
    # nothing notices.  CellCLIP is the top-scoring method here, so a silent
    # change in its input is a silent change in the headline number.
    from cellpaint_pipeline.models.weights import verify_sha256 as _verify

    def _digest(path: Path) -> str | None:
        return _verify(path, use_cache=False) if path.is_file() else None

    out.with_suffix('.json').write_text(
        json.dumps({
            'shape': list(vectors.shape),
            'dtype': str(vectors.dtype),
            'vectors_sha256': _digest(out),
            'model_dir': str(args.model_dir),
            'model_sha256': model_sha256,
            'model_name': 'facebook/dinov2-giant',
            'source_crops': str(args.crops),
            'source_crops_sha256': _digest(Path(args.crops)),
            'channel_count': int(vectors.shape[1]),
            'channel_order': list(CHANNEL_ORDER),
            'crop_size': int(processor.crop_size['height']) if isinstance(
                processor.crop_size, dict) else int(processor.crop_size),
            'do_resize': bool(processor.do_resize),
            'embedding': 'model(x).pooler_output (CLS token)',
            'note': 'one deterministic crop per cell; upstream averages 5 random crops',
        }, indent=2),
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
