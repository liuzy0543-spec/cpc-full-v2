"""uniDINO adapter -- official contract, *surrogate* checkpoint (v3 profile).

What the official code does
---------------------------
``inference.py:97`` builds ``vits.vit_small(patch_size=16)``.  Cells are *not*
fed in as a channel stack: ``inference_utils.py:forward_inference`` rearranges
``(batch, ch, crops, 1, h, w)`` to ``(batch*ch*crops, 1, h, w)`` and runs each
channel through the ViT **separately**, then flattens back to
``(batch, crops, ch * h)``.  So one cell's embedding is ``C`` Vit-S vectors
concatenated: ``5 * 384 = 1920`` wide.  That is why the manifest declares
``output_dim_per_channel=384`` and why channel order matters -- concatenation is
not symmetric.

The checkpoint
--------------
The published checkpoint cannot be obtained here (Zenodo ``checkpoints.zip`` is
1.38 GB; measured ~9 KB/s from this host, ~42 h).  This adapter therefore loads a
**surrogate**: the same architecture, trained on this repository's demo crops by
``scripts/embedding_benchmark/train_surrogates.py``.  The manifest says
``weights_provenance='surrogate'`` and so does every result it returns.

Do not report its scores as uniDINO's.  The interface is real; the numbers are
not the paper's.
"""

from __future__ import annotations

from typing import Any

from cellpaint_pipeline.models.adapters._surrogate import (
    SPECS,
    sha256_file,
    tiny_vit,
)
from cellpaint_pipeline.models.adapters.base import BaseAdapter

__all__ = ['UniDinoSurrogateAdapter']


class UniDinoSurrogateAdapter(BaseAdapter):
    """ViT-S/16 applied per channel, concatenated."""

    requires_frameworks = ('torch',)
    backend = 'inprocess-surrogate'

    #: Depth/width of the surrogate.  Deliberately smaller than the official
    #: ``vit_small`` (depth 12): the point is to exercise the per-channel
    #: contract, and a 12-layer model on 3.8k crops would only overfit.
    depth = 4
    num_heads = 6

    def check_dependencies(self) -> None:
        self.require_frameworks()

    def _build(self) -> Any:
        import torch

        spec = SPECS[self.manifest.name]
        model = tiny_vit(
            spatial=spec.spatial,
            patch_size=16,
            in_chans=spec.in_chans,
            embed_dim=spec.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
        )
        checkpoint = self.resolve_weights()
        state = torch.load(str(checkpoint), map_location='cpu')
        model.load_state_dict(state['model'] if 'model' in state else state)
        # Digest of the artefact actually loaded.  The manifest declares no hash
        # for a surrogate (it is trained locally, not distributed), so without
        # this the result would carry an empty `weights_sha256` and the
        # provenance chain would stop at the last link that happens to be hashed.
        self.weights_sha256 = self.weights_sha256 or sha256_file(checkpoint)
        self._device = self.select_device(torch)
        model.eval().to(self._device)
        return model

    def _forward(self, array: Any, **kwargs: Any) -> Any:
        import numpy as np
        import torch

        # Declared layout is BCHW; one vector per channel, concatenated.
        tensor = torch.as_tensor(np.asarray(array), dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        samples, channels, height, width = tensor.shape

        # (B, C, H, W) -> (B*C, 1, H, W): the official flatten, per channel.
        flat = tensor.reshape(samples * channels, 1, height, width)
        pieces = []
        with torch.no_grad():
            for start in range(0, flat.shape[0], self.batch_size):
                block = flat[start:start + self.batch_size].to(self._device)
                pieces.append(self._model(block).cpu())
        vectors = torch.cat(pieces)                      # (B*C, 384)
        return vectors.reshape(samples, channels * vectors.shape[-1]).numpy()

    #: Channels per forward pass.  Kept modest so a 3.8k-cell run fits in memory
    #: on a machine that is busy with something else.
    batch_size = 64