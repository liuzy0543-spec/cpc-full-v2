"""Shared building blocks for the *surrogate* checkpoints (v3 profile).

Why this module exists
----------------------
Three methods (``dinocells``, ``unidino``, ``morphogenie``) publish weights that
cannot be obtained here: dinocells ships only inside a 3.71 GB tarball at
~9 KB/s (~110 h), unidino is a 1.38 GB Zenodo archive at the same rate, and
morphogenie's Google Drive copy is unreachable and needs a MATLAB preprocessing
step whose runtime (MATLAB, Cellpose, raw data) is not available here -- the
MATLAB source itself is in the official repository.

Cancelling them entirely (the v2 decision) hides a real question: *does the
dispatch path actually work for them?*  Getting that answer needs a checkpoint
that produces the right kind of tensor.  So v3 pairs each of those methods with a
**surrogate**: the published architecture, trained briefly on this repository's
own demo crops.

What a surrogate is and is not
------------------------------
It **is**:
  * the published architecture (ViT-S/16 for unidino and dinocells, the dVAE
    encoder for morphogenie) -- so the tensor shapes, the channel handling and
    the tape layout are exercised for real;
  * a real training run: weights come from gradient descent on local crops, not
    from a random seed, so embeddings are non-degenerate and a retrieval
    benchmark produces something other than noise;
  * labelled ``weights_provenance='surrogate'`` in the manifest, in every
    ``EmbeddingResult`` and in every benchmark row.

It **is not**:
  * the authors' method.  No number produced with a surrogate may be reported as
    that method's performance, compared against a paper, or mixed into a table
    that also holds official-weights results without a visible marking.

Keeping training here (rather than in each adapter) means one script, one set of
seeds and one reproducibility story for all three.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    'SurrogateSpec',
    'SPECS',
    'describe_checkpoint',
    'sha256_file',
    'tiny_vit',
]


# --------------------------------------------------------------------- specs --

@dataclass(frozen=True)
class SurrogateSpec:
    """What one surrogate has to look like to be usable by its adapter."""

    name: str
    #: Input the adapter will hand the network.
    spatial: tuple[int, int]
    in_chans: int
    #: Embedding width the manifest declares.
    embed_dim: int
    #: Why the official checkpoint could not be used, in one line.
    reason: str


#: Mirrors the architecture choices in the official code.
#:
#: * unidino: ``inference.py:97`` selects ``vit_small`` with ``patch_size=16``;
#:   each channel goes through the network separately, so ``in_chans=1`` and the
#:   adapter concatenates ``C`` vectors of width 384.
#: * dinocells: ``run_get_features.py:87-91`` builds ``vits[arch](..., in_chans=
#:   num_channels)``; the official Cell Painting dataset *folds the tape back
#:   into a channel stack and drops the mask* before the model sees it
#:   (``data_utils/cellpainting_dataset.py:23-47``, ``mode="drop"``), so the
#:   model input is ``(C, H, W)`` with ``in_chans=num_channels`` -- not a
#:   one-channel wide image.
#: * morphogenie: ``gan_training/models/dvae.py`` ends the encoder in
#:   ``Linear(256, c_dim*2)`` with ``c_dim=10``; the published extraction path
#:   persists the reparameterised ``c``, so the surrogate keeps the same 10-wide
#:   bottleneck.
SPECS: dict[str, SurrogateSpec] = {
    'unidino': SurrogateSpec(
        name='unidino',
        spatial=(224, 224),
        in_chans=1,
        embed_dim=384,
        reason='Zenodo checkpoints.zip is 1.38 GB; measured ~9 KB/s here (~42 h).',
    ),
    'dinocells': SurrogateSpec(
        name='dinocells',
        spatial=(96, 96),
        # The official network is a multi-channel ViT
        # (`in_chans=config["model"]["num_channels"]`) over the folded,
        # mask-stripped stack -- five channels for Cell Painting.
        in_chans=5,
        embed_dim=384,
        reason=(
            'Weights are open (Zenodo 8061392, CC-BY-4.0) but ship only inside a '
            '3.71 GB tarball; measured ~9 KB/s here (~110 h).'
        ),
    ),
    'morphogenie': SurrogateSpec(
        name='morphogenie',
        spatial=(256, 256),
        in_chans=3,
        embed_dim=10,
        reason=(
            'Weights only on Google Drive (unreachable) and the centring step '
            'runs in MATLAB -- source is in the official repository '
            '(ImagePreProcessing/), but the MATLAB runtime, Cellpose and the raw '
            'data are not available here.'
        ),
    ),
}


# ------------------------------------------------------------------ utilities --

def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def describe_checkpoint(path: str | Path) -> dict[str, Any]:
    """Return size and digest for a checkpoint, for embedding in a manifest."""
    file = Path(path)
    return {
        'path': str(file),
        'bytes': file.stat().st_size,
        'sha256': sha256_file(file),
    }


def tiny_vit(
    *,
    spatial: tuple[int, int],
    patch_size: int,
    in_chans: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
) -> Any:
    """Build the DINO-style ViT the official code uses, at a small depth.

    Imported lazily so this module stays importable without torch -- the adapter
    contract checks run on machines with no deep-learning framework at all, and a
    module-level ``import torch`` would break them.
    """
    import torch
    from torch import nn

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )

        def forward(self, x):
            h = self.norm1(x)
            x = x + self.attn(h, h, h, need_weights=False)[0]
            return x + self.mlp(self.norm2(x))

    class _ViT(nn.Module):
        """ViT with the same patch embedding and CLS token as ``vits.vit_small``."""

        def __init__(self) -> None:
            super().__init__()
            self.patch_size = patch_size
            self.embed_dim = embed_dim
            self.patch_embed = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            grid = (spatial[0] // patch_size) * (spatial[1] // patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, grid + 1, embed_dim))
            self.blocks = nn.Sequential(*[_Block() for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        def forward(self, x):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, x), dim=1) + self.pos_embed
            return self.norm(self.blocks(x))[:, 0]

    return _ViT()