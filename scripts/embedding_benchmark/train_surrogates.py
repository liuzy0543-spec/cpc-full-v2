"""Train the three surrogate checkpoints used by the ``v3`` profile.

Read ``src/cellpaint_pipeline/models/adapters/_surrogate.py`` first: it explains
why a surrogate exists at all, and what it is not.

Design decision that matters more than the hyperparameters
---------------------------------------------------------
Training inputs are built with :func:`cellpaint_pipeline.models.preprocessing.prepare`
using the *same* manifests the adapters use at inference time.  It would have been
shorter to re-implement the transforms here, but then a surrogate would be trained
on one tensor layout and served another -- and the failure mode is silent: the
model still returns an embedding of the right width, it is just meaningless.

Labels are **never** used.  The objective is self-supervised (VICReg: invariance
between two augmented views, plus variance and covariance terms that stop the
embedding from collapsing to a constant).  A supervised objective on `compound`
would make the benchmark a memorisation test -- the surrogate would retrieve
compounds because it was told the answer.

Training cells are **mock/control only** (``well_role == 'mock'``).  Since the
benchmark's task is compound retrieval, a surrogate that has never seen a compound
cannot cheat: whatever it scores is genuine generalisation, not recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src'))

from cellpaint_pipeline.models import preprocessing, registry  # noqa: E402
from cellpaint_pipeline.models.adapters._surrogate import (  # noqa: E402
    SPECS,
    sha256_file,
    tiny_vit,
)

__all__ = ['main']


# ------------------------------------------------------------------ helpers --

def _otsu_threshold(plane: np.ndarray) -> float:
    """Return Otsu's threshold for a 2-D image.

    A Cell Painting crop arrives with a segmentation mask when the upstream
    pipeline ran CellProfiler.  This dataset was built without one, so the
    surrogate's mask is derived from the DNA channel instead.  It is a
    surrogate *input* for a surrogate *model*: documented here so nobody mistakes
    it for real segmentation.
    """
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


def _masks_from_dna(crops: np.ndarray, dna_index: int = 0) -> np.ndarray:
    """Derive a binary silhouette per cell from the DNA channel (Otsu).

    **Not used by the training pipeline any more.**  It existed because
    dinocells' tape carried a mask column; the official model input turned out
    to be a channel stack with the mask *dropped* (the official dataset folds
    the stored tape back open and discards the mask column before the network
    sees it -- ``data_utils/cellpainting_dataset.py:23-47``, ``:86``), so no
    surrogate needs a mask.  Kept because it documents how a silhouette was
    obtained, and because a future method whose contract really does take a
    mask would need exactly this.
    """
    """Derive one binary silhouette per crop from its DNA channel."""
    # crops is (N, C, H, W): the mask is one plane per cell, not per channel.
    masks = np.zeros((crops.shape[0], crops.shape[2], crops.shape[3]),
                     dtype='uint8')
    for i in range(crops.shape[0]):
        plane = crops[i, dna_index]
        threshold = _otsu_threshold(plane)
        masks[i] = (plane > threshold).astype('uint8')
    return masks


def _augment(batch, generator, *, lo, hi):
    """Two cheap views: flips plus a brightness/contrast jitter.

    Deliberately conservative.  Aggressive crops on 96x96 surrogate data destroy
    the little signal there is, and the point is a non-degenerate embedding, not a
    state-of-the-art representation.
    """
    views = []
    for _ in range(2):
        view = batch.copy()
        if generator.random() < 0.5:
            view = np.flip(view, axis=-1).copy()
        if generator.random() < 0.5:
            view = np.flip(view, axis=-2).copy()
        gain = generator.uniform(0.85, 1.15)
        span = float(hi - lo) if hi > lo else 1.0
        bias = generator.uniform(-0.05, 0.05) * span
        view = view * gain + bias
        # Clip to the data's own range, not to [0, 1]: dinocells' input is
        # per-sample standardised and legitimately carries negative values, so a
        # fixed 0..1 clip would flatten it.
        views.append(np.clip(view, lo, hi))
    return views


def _vicreg(z1, z2, torch, *, sim=25.0, var=25.0, cov=1.0, eps=1e-4):
    """VICReg: invariance + variance + covariance, without negatives."""
    import torch.nn.functional as F

    invariance = F.mse_loss(z1, z2)

    def variance_term(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    def covariance_term(z):
        n, d = z.shape
        centred = z - z.mean(dim=0)
        matrix = (centred.T @ centred) / max(n - 1, 1)
        off_diagonal = matrix - torch.diag_embed(torch.diag(matrix))
        return (off_diagonal ** 2).sum() / d

    return (
        sim * invariance
        + var * (variance_term(z1) + variance_term(z2))
        + cov * (covariance_term(z1) + covariance_term(z2))
    )


# ---------------------------------------------------------------- per method --

def _build_inputs(name: str, crops: np.ndarray, masks: np.ndarray | None) -> np.ndarray:
    """Return the exact tensor the adapter will see at inference time."""
    manifest = registry.manifest(name)

    if name == 'morphogenie':
        # `prepare` refuses matlab_centring and must keep refusing: the real
        # centring step is not in the repository.  The surrogate therefore takes
        # a plainly-resized 3-channel crop, and the fact that this is *not* the
        # official preprocessing is recorded in the manifest notes.
        from torch.nn.functional import interpolate
        import torch

        tensor = torch.as_tensor(crops[:, :3].astype('float32')) / 255.0
        # Already BCHW: interpolate takes (N, C, H, W), so no transpose here.
        tensor = interpolate(tensor, size=SPECS[name].spatial,
                             mode='bilinear', align_corners=False)
        return tensor.numpy().astype('float32')

    if name == 'unidino':
        # div65535 is declared for 16-bit data; feeding 0..255 would scale every
        # value to ~0.004 and the model would see a black image.
        # `prepare` takes BHWC on the way in; `batch_layout` describes the
        # tensor it returns, which is BCHW here.
        payload = np.moveaxis(crops.astype('uint16') * 257, 1, -1)
        prepared = preprocessing.prepare(
            manifest, payload, channels=manifest.channel_names
        )
        # Mirror `forward_inference`: each channel goes through the ViT on its
        # own, so training must see the same (N*C, 1, H, W) view the adapter
        # builds at inference.  Training on the 5-channel stack would learn a
        # different first convolution than the one that gets served.
        samples, channels, height, width = prepared.shape
        return prepared.reshape(samples * channels, 1, height, width)

    if name == 'dinocells':
        # The official model input is a CHANNEL STACK with the mask dropped:
        # the official dataset folds its stored tape back open and discards the
        # mask column before the network sees it
        # (data_utils/cellpainting_dataset.py:23-47, :86).  So `prepare` is
        # given the five image channels and returns (N, C, H, W) -- and a batch
        # is fine, because the official model batches like any image stack.
        payload = np.moveaxis(crops.astype('uint16') * 257, 1, -1)
        prepared = preprocessing.prepare(
            manifest,
            payload,
            masks=masks,
            channels=manifest.channel_names,
        )
        return prepared.astype('float32')

    raise ValueError(f'no surrogate recipe for {name}')


def _make_model(name: str):
    spec = SPECS[name]
    if name == 'morphogenie':
        import torch
        from torch import nn

        z_dim = spec.embed_dim

        class _Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(spec.in_chans, 32, 4, 2, 1), nn.ReLU(True),
                    nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
                    nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
                    nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),
                )
                self.head = nn.Linear(256, z_dim * 2)

            def forward(self, x):
                # Must match the adapter exactly -- head is z_dim*2 wide (mu and
                # logvar) but only mu is exported, so the state_dict is shared
                # with the served model without any key surgery.
                h = self.features(x).mean(dim=(2, 3))
                return self.head(h)[:, :z_dim]

        return _Encoder()

    spatial = spec.spatial

    return tiny_vit(
        spatial=spatial,
        patch_size=16,
        in_chans=spec.in_chans,
        embed_dim=spec.embed_dim,
        depth=4,
        num_heads=6,
    )


def train_one(name: str, inputs: np.ndarray, *, steps: int, batch: int,
              seed: int, device: str, log) -> dict:
    """Train one surrogate and return its checkpoint payload."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = __import__('random').Random(seed)

    model = _make_model(name).to(device)
    # A projection head is standard for VICReg: the objective is applied to the
    # projection, the *encoder output* is what gets exported as the embedding.
    projector = torch.nn.Sequential(
        torch.nn.Linear(SPECS[name].embed_dim, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 256),
    ).to(device)
    params = list(model.parameters()) + list(projector.parameters())
    optimiser = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=1e-3, total_steps=steps
    )

    n = inputs.shape[0]
    lo, hi = float(inputs.min()), float(inputs.max())
    losses = []

    for step in range(steps):
        index = rng.integers(0, n, size=min(batch, n))
        raw = inputs[index]
        views = _augment(raw, generator, lo=lo, hi=hi)

        tensors = [
            torch.as_tensor(v, dtype=torch.float32).to(device) for v in views
        ]
        embeddings = [model(t) for t in tensors]
        projections = [projector(e) for e in embeddings]
        loss = _vicreg(projections[0], projections[1], torch)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        schedule.step()
        losses.append(float(loss))

        if (step + 1) % 50 == 0 or step == 0:
            log(f'    step {step + 1:4d}/{steps}  loss={np.mean(losses[-50:]):8.4f}')

    model.eval()
    with torch.no_grad():
        probe = torch.as_tensor(inputs[:64], dtype=torch.float32).to(device)
        out = model(probe).cpu().numpy()

    return {
        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'meta': {
            'method': name,
            'provenance': 'surrogate',
            'objective': 'vicreg-self-supervised',
            'trained_on': 'BBBC022 mock/control crops only (no compound labels)',
            'n_train': int(n),
            'steps': steps,
            'seed': seed,
            'embedding_std': float(np.mean(out.std(axis=0))),
            'embedding_dim': int(out.shape[1]),
        },
    }


# --------------------------------------------------------------------- main --

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells-dir', default=r'E:\bbbc022\cells')
    parser.add_argument('--out-dir', default=r'E:\cpc_weights')
    parser.add_argument('--methods', nargs='*',
                        default=['unidino', 'dinocells', 'morphogenie'])
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    def log(message: str) -> None:
        print(message, flush=True)

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'device={device}')

    cells_dir = Path(args.cells_dir)
    crops = np.load(cells_dir / 'cells_ca.npy', mmap_mode='r')
    # `csv`, not `np.genfromtxt`: compound names contain commas and are quoted, so
    # a naive split reports nine columns on some rows.  Reading it as CSV keeps
    # the mock/compound split -- the thing this script is careful about -- correct.
    with open(cells_dir / 'meta.csv', encoding='utf-8', newline='') as handle:
        meta_rows = list(csv.DictReader(handle))
    roles = np.asarray([row['well_role'] for row in meta_rows])
    if roles.size != crops.shape[0]:
        raise SystemExit(
            f'meta has {roles.size} rows but crops has {crops.shape[0]}; refusing '
            f'to guess which cell is which'
        )
    mock = np.flatnonzero(roles == 'mock')
    log(f'crops={crops.shape}  mock={mock.size}  compounds never seen in training')

    registry.set_profile('v3')
    selected = np.asarray(crops[mock])
    # No masks: since the F2 fix, dinocells' official model input is a channel
    # stack with the mask DROPPED (the official dataset folds its stored tape
    # back open and discards the mask column), so nothing in the training
    # pipeline needs one any more.  The Otsu helper below is kept only as
    # documentation of how a mask used to be derived.

    out_dir = Path(args.out_dir)
    report = {}
    for name in args.methods:
        log(f'--- {name} ---')
        inputs = _build_inputs(name, selected, None)
        log(f'    inputs {inputs.shape} dtype={inputs.dtype} '
            f'range=[{inputs.min():.3f}, {inputs.max():.3f}]')

        payload = train_one(
            name, inputs, steps=args.steps, batch=args.batch,
            seed=args.seed, device=device, log=log,
        )

        target_dir = out_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = target_dir / f'surrogate_{name}.pt'
        torch.save(payload, checkpoint)
        digest = sha256_file(checkpoint)
        info = {
            **payload['meta'],
            'checkpoint': str(checkpoint),
            'bytes': checkpoint.stat().st_size,
            'sha256': digest,
        }
        (target_dir / f'surrogate_{name}.json').write_text(
            json.dumps(info, indent=2), encoding='utf-8'
        )
        report[name] = info
        log(f'    saved {checkpoint} ({info["bytes"]:,} B) sha256={digest[:16]} '
            f'emb_std={info["embedding_std"]:.4f}')

    (out_dir / 'surrogates_manifest.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'
    )
    log('')
    log('done.  These are SURROGATES: same architecture, trained here.')
    log('Their scores are not any method\'s published performance.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())