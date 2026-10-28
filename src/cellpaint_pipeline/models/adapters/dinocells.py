"""DinoCells adapter -- official contract, *surrogate* checkpoint (v3 profile).

What the official code does
---------------------------
The official Cell Painting dataset stores each cell as a **tape**: the five
image channels laid out side by side in the width dimension, with a mask column
riding along, so a cell is ``(H, W * (C + 1))`` on disk.

That is a **storage** format.  The model does not see it.  The official dataset
class folds the tape back open before the transform pipeline runs:

    data_utils/cellpainting_dataset.py:23-47
        def fold_channels(image, channel_width, mode="drop"):
            # Expected input image shape: (h, w * c)
            # Output image shape: (h, w, c)
            output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")
            ...
            elif mode == "drop":
                output = output[:, :, 0:-1]     # Drop mask channel (last)
    data_utils/cellpainting_dataset.py:86
        image = fold_channels(image, channel_width)    # default mode="drop"

and the network is built for a multi-channel stack:

    run_get_features.py:87-91
        model = vits.__dict__[config["model"]["arch"]](
            img_size=[224], patch_size=..., num_classes=0,
            in_chans=config["model"]["num_channels"],
        )

So the official model input is ``(C, H, W)`` with **the mask discarded**, and
it batches like any other image stack (``batch_size_per_gpu: 24``).

An earlier version of this adapter implemented the *inverse* of the official
fold and fed the resulting ``(H, W * (C + 1))`` wide single-channel image to the
network.  The fold was implemented correctly -- which is exactly the problem:
it reproduced the upstream **storage** format and then passed it off as the
**model input** format.  The mask was not even an official input channel; the
official code throws it away.

The normalisation the official pipeline applies to the folded stack is
``self_normalize`` (``utils/augmentations.py:460-468``): a per-channel z-score
over the spatial axes.  The upstream ``norm_loader`` that runs before it is a
whole-image min-max onto 0..255 (``data_utils/file_dataset.py:77-82``) -- and
because that is a single affine map applied to every channel alike, the
per-channel z-score cancels it exactly.  ``per_sample_selfnorm`` therefore
reproduces the official chain, with only uint8 quantisation left over.

The checkpoint
--------------
The published weights are genuinely open (Zenodo record 8061392, CC-BY-4.0) but
are distributed *only* inside a single 3.71 GB ``HPA_FOV_data.tar.gz`` that also
carries the whole HPA FOV image set; the checkpoint cannot be fetched alone.
Measured at ~9 KB/s from this host that is ~110 h.  This adapter therefore loads a
**surrogate** trained locally by ``scripts/embedding_benchmark/train_surrogates.py``.

Do not report its scores as DinoCells'.  The interface -- including the input
geometry, which is the part that is easy to get wrong -- is real.
"""

from __future__ import annotations

from typing import Any

from cellpaint_pipeline.models.adapters._surrogate import (
    SPECS,
    sha256_file,
    tiny_vit,
)
from cellpaint_pipeline.models.adapters.base import BaseAdapter

__all__ = ['DinoCellsSurrogateAdapter']


class DinoCellsSurrogateAdapter(BaseAdapter):
    """ViT over the channel stack, CLS token as the embedding."""

    requires_frameworks = ('torch',)
    backend = 'inprocess-surrogate'

    depth = 4
    num_heads = 6
    #: The official model input is a channel stack, so a batch is a real batch
    #: here -- unlike the *input* contract of the tape era, where a batch had
    #: to be refused because the width carried the channel index.
    batch_size = 32

    def check_dependencies(self) -> None:
        self.require_frameworks()

    def _build(self) -> Any:
        import torch

        spec = SPECS[self.manifest.name]
        # The official network is a multi-channel ViT: `in_chans=num_channels`
        # over the folded (and mask-stripped) stack, at the crop's own size.
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

        # Declared layout is a channel stack: (N, C, H, W), or a single
        # (C, H, W) crop.
        stack = np.asarray(array, dtype='float32')
        if stack.ndim == 3:
            stack = stack[None, ...]
        if stack.ndim != 4:
            raise ValueError(
                f'dinocells expects a (C, H, W) crop or a batch of them, '
                f'got {stack.shape}'
            )
        tensor = torch.as_tensor(stack)
        pieces = []
        with torch.no_grad():
            for start in range(0, tensor.shape[0], self.batch_size):
                block = tensor[start:start + self.batch_size].to(self._device)
                pieces.append(self._model(block).cpu())
        return torch.cat(pieces).numpy()
