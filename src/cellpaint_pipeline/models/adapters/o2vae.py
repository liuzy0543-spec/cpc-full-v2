"""O2-VAE adapter -- O(2)-invariant VAE, running on the official MEF weights.

THE PAPER
---------
Burgess J, Nirschl JJ, Zanellati M-C, Lozano A, Cohen S, Yeung-Levy S.
*Orientation-invariant autoencoders learn robust representations for shape
profiling of cells and organelles.* **Nature Communications 15, 1022 (2024)**,
doi:10.1038/s41467-024-45362-4.  Code: https://github.com/jmhb0/o2vae
(Zenodo 10.5281/zenodo.10206848).  The name "mefs" in the upstream artefact
names **Mouse Embryonic Fibroblasts**, not an acronym.

WHICH WEIGHTS
-------------
The pretrained checkpoints ARE released; an earlier version of this docstring
claimed otherwise and loaded a locally self-trained o2-mnist model instead.  They
are tracked in the upstream repository at
``wandb/pretrained_models/model_mefs.pt`` (commit 4c70d58) and were trained on
**MEF** cells -- real mouse embryonic fibroblasts, 25,000 train / 6,598 test
single cells at 128x128 (``data/mefs/README.md``).

The architecture must match that checkpoint, not the demo: ``zdim=512`` and a
six-layer decoder reach 2**6 = 64 px (``configs/config_mefs.py:46,65``).  The
earlier five-layer, ``zdim=64`` configuration matched only the o2-mnist model and
cannot load the published one.

WHAT THE METHOD DOES
--------------------
O2-VAE is an **O(2)-invariant** variational autoencoder: invariance to rotation
and flip is enforced by the architecture (e2cnn steerable convolutions with
``N=-3``) plus an optimal-alignment reconstruction loss.  The embedding is the
latent mean ``z_mu``, not a reconstruction and not the log-variance.

INPUT SURFACE
-------------
A single-channel binary silhouette of a *centred* cell, resized to 64x64 and in
``[0, 1]`` (``do_sigmoid=True``; the manifest declares the range and the contract
enforces it).  It is a **shape** model: on Cell Painting data it consumes a mask
derived from the DNA channel rather than five-channel intensity.  That modality
difference is inherent to the method and has to be stated wherever its numbers
are compared against intensity models.

DOMAIN SHIFT -- READ THIS BEFORE QUOTING AN O2-VAE NUMBER
---------------------------------------------------------
The official checkpoint was trained on **cell** silhouettes: the upstream
configuration points at ``data/mefs/scaled/cell``
(``configs/config_mefs.py:19``), and no separate nucleus checkpoint is
published.  What this project feeds it is a **nucleus** silhouette -- the DNA
channel thresholded, cropped and resampled
(``scripts/embedding_benchmark/build_bbbc022_cells.py``).

So the inference is **out of domain**, and the embedding distribution does not
match the training distribution.  This is not fixable by construction: BBBC022
has no cytoplasm stain that segments reliably, so there is no cell silhouette to
feed instead.  The honest options are to declare the out-of-domain inference
wherever the number appears, or not to report the number at all -- and the
second is preferable for a performance claim.

A second, smaller discrepancy, for the same reason: the paper's Methods say
"embedding dimensions d = 128 for all models trained in this paper", while the
released checkpoint measures ``zdim=512``
(``q_net.fully_net.3.bias`` is (1024,); ``p_net.fc.0.weight`` is (48, 512)).
The code and the weights are authoritative; the discrepancy belongs in the
supplementary material, or a reviewer will assume the computation is wrong.
============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cellpaint_pipeline.models.adapters.base import BaseAdapter
from cellpaint_pipeline.models.errors import EmbeddingError
from cellpaint_pipeline.models.contract import ModelManifest

__all__ = ['O2VaeAdapter']


@dataclass
class O2VaeAdapter(BaseAdapter):
    """Adapter for a self-trained O2-VAE checkpoint."""

    manifest: ModelManifest = None  # type: ignore[assignment]
    #: Upstream checkout the architecture is imported from.  ``None`` -> the
    #: directory named by the manifest's ``weights.repo_source`` under the same
    #: weights root that produced the checkpoint.
    repo_root: Path | None = None
    #: Requested compute device (``None`` -> the adapter decides).  Declared as a
    #: field so the generated ``__init__`` accepts it instead of shadowing
    #: :meth:`BaseAdapter.__init__`, which is what made ``device=`` a TypeError.
    device: str | None = None
    #: Additional roots searched for the checkpoint before the defaults.
    weights_roots: tuple[Any, ...] | None = None

    requires_frameworks: tuple[str, ...] = ('torch', 'e2cnn')
    dependency_hint: str | None = (
        'O2-VAE needs torch plus e2cnn (steerable convolutions). On this machine '
        'they live in the torch environment.'
    )

    def __post_init__(self) -> None:
        if self.manifest is None:
            from cellpaint_pipeline.models.registry import manifest as manifest_for

            self.manifest = manifest_for('o2vae')
        BaseAdapter.__init__(
            self,
            manifest=self.manifest,
            weights_roots=self.weights_roots,
            device=self.device,
        )
        self._torch = None

    # ------------------------------------------------------------ protocol --

    def check_dependencies(self) -> None:
        self.require_frameworks()

    def _build(self) -> Any:
        import sys

        # torchgeometry 0.1.2 (pulled in by e2cnn) still calls ``numpy.float``,
        # which numpy removed in 1.24.  Restoring the alias is the narrowest fix:
        # it keeps the upstream module untouched and only affects this process.
        import numpy as _np

        if not hasattr(_np, 'float'):
            _np.float = float  # type: ignore[attr-defined]

        import torch

        self._torch = torch

        checkpoint = self.resolve_weights()

        # The manifest declares where the upstream checkout lives relative to the
        # weights root, because that cannot be inferred from the checkpoint path.
        if self.repo_root is not None:
            repo = self.repo_root
        else:
            source_parts = [
                p for p in self.manifest.weights.source.replace('\\', '/').split('/') if p
            ]
            repo = checkpoint.parent
            for _ in range(len(source_parts) - 1):
                repo = repo.parent
            repo = repo / (self.manifest.weights.repo_source or '')

        if not (repo / 'models').is_dir():
            raise EmbeddingError(
                f'upstream O2-VAE repository not found at {repo}; pass repo_root= '
                f'pointing at the checkout that owns the checkpoint, or set '
                f'weights.repo_source in the manifest',
                reason='repo-missing',
                details=[f'looked for {repo / "models"}'],
            )

        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        # The upstream modules are imported rather than vendored, so the
        # published architecture stays the single source of truth.  Note this
        # adds a top-level `models` package to sys.path -- see the module
        # docstring for why that is safe for the namespaced
        # `cellpaint_pipeline.models` but shadows a bare `import models`.
        from models.decoders.cnn_decoder import CnnDecoder  # type: ignore
        from models.encoders_o2.e2scnn import E2SFCNN  # type: ignore
        from models.vae import VAE  # type: ignore

        zdim = self.manifest.output_dim
        encoder = E2SFCNN(
            n_classes=zdim * 2,
            n_channels=1,
            cnn_dims=[6, 9, 12, 12, 19, 25],
            layer_type='inducedgated_norm',
            N=-3,
        )
        # Decoder depth comes from `configs/config_mefs.py:65`: six entries, so
        # the decoder upsamples to 2^6 = 64 px.  An earlier version hard-coded
        # five entries, which matched only the self-trained o2-mnist model and
        # cannot load the published MEF checkpoint.
        decoder = CnnDecoder(
            zdim=zdim,
            cnn_dims=[192, 96, 96, 48, 48, 48],
            out_channels=1,
        )
        model = VAE(
            name='vae',
            zdim=zdim,
            channels=1,
            do_sigmoid=True,
            vanilla=False,
            q_net=encoder,
            p_net=decoder,
        )

        payload = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        state = payload.get('model_state_dict', payload)

        # e2cnn registers its basis-expansion index buffers as zero-stride
        # expanded views (`strides=(0,)`, one element of real storage presented
        # as N).  torch's `load_state_dict` copies into buffers with `copy_`, and
        # copy_ refuses a destination whose elements all alias one memory
        # location -- "more than one element of the written-to tensor refers to a
        # single memory location".  Measured: exactly 3 buffers are affected, all
        # of them integer index tables whose checkpoint value is constant
        # (`unique == [0]`).
        #
        # Replacing the buffer with a contiguous tensor is correct rather than a
        # workaround: the destination holds no information beyond "these indices
        # are all zero", which the contiguous replacement preserves exactly.  The
        # alternative, skipping the copy, would silently keep whatever the
        # freshly-built model happened to hold.
        _materialise_expanded_buffers(model, state, torch)

        # `strict=False` is unavoidable here: 188 of the checkpoint's keys map to
        # buffers that e2cnn recomputes from the architecture, so a strict load
        # can never succeed.  But a silent `strict=False` would also accept a
        # checkpoint that shares *no* learned weights with this architecture and
        # run happily on random initialisation.  So the tolerance is narrowed to
        # exactly what is justified: missing keys may only be buffers, and
        # nothing unexpected may appear.
        load_result = model.load_state_dict(state, strict=False)
        parameter_names = set(dict(model.named_parameters()))
        missing_parameters = [
            key for key in load_result.missing_keys if key in parameter_names
        ]
        if missing_parameters or load_result.unexpected_keys:
            raise EmbeddingError(
                f'O2-VAE checkpoint does not match this architecture: '
                f'{len(missing_parameters)} learnable parameters missing, '
                f'{len(load_result.unexpected_keys)} unexpected keys',
                reason='weight-mismatch',
                details=(
                    [f'missing parameter: {k}' for k in missing_parameters[:5]]
                    + [f'unexpected key: {k}' for k in load_result.unexpected_keys[:5]]
                ),
            )
        self._loaded_key_count = len(state) - len(load_result.missing_keys)
        model.eval()

        # Pinned to CPU, and this is a measured requirement rather than a
        # preference: `model.to('cuda')` itself succeeds, but the first forward
        # pass then raises
        #   RuntimeError: Expected all tensors to be on the same device, but
        #   found at least two devices, cuda:0 and cpu!
        # because e2cnn keeps its steerable-kernel basis tensors as plain
        # attributes rather than registered buffers, so `.to()` never moves them.
        # Verified both ways: device='cpu' returns (N, 64); device='cuda' fails.
        #
        # An automatic CUDA selection therefore falls back to CPU; only an
        # explicit device='cuda' request is refused, because by then the caller
        # has named something that cannot work.
        if self.requested_device is not None and self.requested_device != 'cpu':
            raise EmbeddingError(
                f'O2-VAE cannot run on {self.requested_device!r}: e2cnn leaves '
                f'its steerable basis tensors on the CPU, so the forward pass '
                f'fails with a device mismatch. Use device="cpu".',
                reason='device-unsupported',
            )
        self._device = 'cpu'
        model.to(self._device)
        return model

    def _forward(self, array: Any, **kwargs: Any) -> Any:
        torch = self._torch
        tensor = array if isinstance(array, torch.Tensor) else torch.as_tensor(array)
        tensor = tensor.to(self._device).float()
        # A one-row batch takes a different path through the e2cnn encoder:
        # ``NormMaxPool`` gathers fields by ``argmax`` index, so torch's
        # batch-dependent reduction order (a ~3e-6 perturbation in the pre-pool
        # activations) can flip an argmax and move the embedding by 1.9e-2,
        # which is 2.6% of its own std.  Measured, not theorised.  Padding to
        # two rows makes a caller's single-cell call agree with the same cell
        # inside a batch; the duplicate is discarded.
        padded = tensor.shape[0] == 1
        if padded:
            tensor = tensor.repeat(2, 1, 1, 1)
        with torch.no_grad():
            latent = self._model.q_net(tensor)
            # The encoder emits mean and log-variance concatenated; keep the mean.
            mean, _logvar = torch.split(latent, self.manifest.output_dim, dim=1)
        rows = mean.detach().to('cpu').numpy()
        return rows[:1] if padded else rows


def _materialise_expanded_buffers(model: Any, state: dict, torch: Any) -> None:
    """Replace zero-stride buffers that the checkpoint needs to write into.

    A zero-stride ("expanded") tensor aliases one memory location across all its
    elements, so ``copy_`` into it is refused by torch.  Only buffers that the
    checkpoint actually targets are touched, and each is replaced by a contiguous
    tensor holding the checkpoint's values, so nothing is invented and nothing is
    skipped.

    Factored out of ``_build`` so the reasoning above sits next to the code that
    depends on it, and so it can be tested without loading the ~9 MB checkpoint.
    """
    targets = {
        name
        for name, buffer in model.named_buffers()
        if any(stride == 0 for stride in buffer.stride())
    }
    if not targets:
        return

    buffers = dict(model.named_buffers())
    for name in sorted(targets & set(state)):
        replacement = state[name].clone().reshape(buffers[name].shape).contiguous()
        owner = model
        parts = name.split('.')
        for part in parts[:-1]:
            owner = getattr(owner, part)
        # Assigning into `_buffers` rather than using copy_ is the point: the
        # destination has to be a real storage, not an alias.
        owner._buffers[parts[-1]] = replacement

