"""MorphoGenie adapter -- official contract, *surrogate* checkpoint (v3 profile).

What the official code does
---------------------------
``gan_training/models/dvae.py`` builds a small convolutional variational
autoencoder whose encoder ends in ``Linear(256, c_dim * 2)`` with ``c_dim = 10``;
``stats`` is then split into ``mu = stats[:, :c_dim]`` and
``logvar = stats[:, c_dim:]``, and ``forward`` returns ``(c, mu, logvar)``.

(``dVAE/model.py`` is a *different*, legacy training script: it uses ``z_dim``
rather than ``c_dim`` and returns a four-tuple
``(x_recon, mu, logvar, z.squeeze())``.  It is not on the inference path --
``gan_training/models/__init__.py`` maps ``fvae_dict['FactorVAE256']`` to
``gan_training/models/dvae.py`` -- so quoting it as the official extraction
path misdescribes the method.)

Two measured details are worth keeping in view, because both were found by
reading the official caller rather than the model:

* ``MorphoGenie_Test.py:67-68`` sets ``c_dim = 10`` and ``z_dim = 256``; only
  ``c_dim`` reaches the embedding.  256 is the GAN generator's latent, not the
  embedding width.  The manifest declares ``output_dim=10`` for that reason.
* the official caller unpacks ``mu_real, logvar_real, c_real = fvae(...)`` while
  ``dvae.forward`` returns ``(c, mu, logvar)`` -- so what it persists as the
  embedding is the **reparameterised sample**, which draws noise.  That is why
  the published path is not reproducible run to run.

The checkpoint
--------------
Weights are only on Google Drive, and the per-crop centring runs in MATLAB
(``preprocess.source_map='matlab_centring'``, which
:func:`~cellpaint_pipeline.models.preprocessing.prepare` refuses rather than
guesses).  The MATLAB source is in the official repository --
``ImagePreProcessing/MultipleCells/CPA/GetCentImages.m`` -- and crops to
200x200 there (:26) before the Python side resizes to 256x256
(``gan_training/utils.py:352-357``); what is missing here is the MATLAB
runtime, Cellpose and the raw data.  This adapter loads a **surrogate**
trained locally.

It returns the encoder **mean**, not a sample: the surrogate is deterministic,
and its manifest says ``determinism='deterministic'`` so a benchmark does not
average draws that never vary.  That is a deliberate difference from the
official path, recorded here rather than hidden.
"""

from __future__ import annotations

from typing import Any

from cellpaint_pipeline.models.adapters._surrogate import SPECS, sha256_file
from cellpaint_pipeline.models.adapters.base import BaseAdapter

__all__ = ['MorphoGenieSurrogateAdapter']


class MorphoGenieSurrogateAdapter(BaseAdapter):
    """Convolutional encoder with the published 10-wide bottleneck."""

    requires_frameworks = ('torch',)
    backend = 'inprocess-surrogate'

    batch_size = 32

    def check_dependencies(self) -> None:
        self.require_frameworks()

    def _build(self) -> Any:
        import torch
        from torch import nn

        spec = SPECS[self.manifest.name]
        z_dim = spec.embed_dim

        class _Encoder(nn.Module):
            """Same 4-stage downsampling as ``dVAE``, 256x256 -> 16x16 -> Linear."""

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
                h = self.features(x).mean(dim=(2, 3))
                return self.head(h)[:, :z_dim]      # mu only -- see module docstring

        model = _Encoder()
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

        tensor = torch.as_tensor(np.asarray(array), dtype=torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        pieces = []
        with torch.no_grad():
            for start in range(0, tensor.shape[0], self.batch_size):
                block = tensor[start:start + self.batch_size].to(self._device)
                pieces.append(self._model(block).cpu())
        return torch.cat(pieces).numpy()