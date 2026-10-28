"""CA-MAE / OpenPhenom adapter.

IDENTITY
--------
CA-MAE is the **Channel-Agnostic Masked Autoencoder**, from Recursion.  Two
papers carry the name and they must not be swapped:

* arXiv **2309.16064** -- *Masked Autoencoders are Scalable Learners of
  **Cellular Morphology*** (NeurIPS 2023).
* arXiv **2404.10242** -- *Masked Autoencoders **for Microscopy** are Scalable
  Learners of **Cellular Biology*** (CVPR 2024).

The upstream repository lists them separately (``README.md:3-9``).  The
authors' affiliation is Recursion; "Valence Labs", which earlier revisions of
this file mentioned, appears in neither the repository nor either paper and
has been removed.

Its public weights are published as ``OpenPhenom-S/16``, a ViT-S/16 encoder.  It
is *not* "Cross-Architecture" or "Context-Aware" MAE -- both expansions are
wrong and circulate widely.

WHY THIS ONE IS THE EASIEST TO GET RIGHT
----------------------------------------
It is the only method in this layer whose preprocessing is **inside the
checkpoint**: the module applies ``Normalizer()`` (pixels / 255) followed by a
non-affine ``InstanceNorm2d``.  So the contract is simply "hand me raw uint8
0..255 at (N, C, 256, 256)" and the model does the rest -- the manifest says so,
and :func:`check_compatible` enforces both the range and the dtype.

The official normalisation is exactly one line -- ``pixels.float() / 255.0``
(``normalizer.py:7``, ``vit_encoder.py:57-60``) -- and this project now does
exactly that and nothing else (``source_map='identity'``: the uint8 crop reaches
the model untouched, the model divides by 255 internally).

How a **16-bit** crop becomes 0..255 *is* unspecified upstream, because there is
no 16-to-8 conversion code in the repository.  This project used to stretch each
crop by its own global maximum (``source_map='per_crop_max255'``) and justified it
by claiming the checkpoint's ``InstanceNorm2d`` cancels the stretch.  Measured,
that claim is **half right, and the wrong half was load-bearing** (4 crops,
384-d output, stretch disabled, input scaled):

| input scale | cosine | ``max|diff|`` | as % of output std |
|---|---|---|---|
| x0.8 | 0.99970 | 0.0568 | 21.0% |
| x0.5 | 0.99948 | 0.0697 | 25.8% |

So InstanceNorm cancels a global scale in **direction** (cosine is essentially 1)
but not in **magnitude** -- individual coordinates move by up to a quarter of the
output's standard deviation.  For a cosine/kNN benchmark that is nearly harmless;
for anything using raw distances it is not, and the earlier "residual ~0.7% of
std" figure understated it by roughly an order of magnitude.

The step only looked free because every one of the 3,857 BBBC022 crops already
has per-crop max 255, so the stretch was an exact identity there (removing it
leaves the stored embeddings bit-for-bit unchanged).  It is now removed, so the
path is official for *any* input rather than only for this dataset.

Removing it makes the path official for *any* input, not just this dataset.  The
16-to-8 question is still unresolved upstream and the dtype guard rejects a
16-bit array rather than inventing a conversion.

Channel-agnosticism is architectural (channelwise cross-attention over patch
tokens), which is why the manifest sets ``names_are_labels=False``.  That is a
claim backed by measurement: the reference run exercised C in {1, 4, 5, 6, 11}.
The *checkpoint* still has a ceiling -- ``max_in_chans=11``
(``huggingface_mae.py:75``), with the positional embedding generated for
exactly that many modalities -- so the manifest declares ``max_channels=11``
and the contract refuses a twelfth channel instead of letting it fail inside
the model.

TWO FORMAT TRAPS, BOTH PAID FOR ONCE
------------------------------------
1. The file is named ``model.safetensors`` but its magic bytes are
   ``50 4b 03 04`` -- a ZIP, i.e. a ``torch.save`` artefact.
   ``safetensors.torch.load_file`` fails on it; the manifest records
   ``format='torch-save-zip'`` so this is declared rather than discovered.
2. The upstream layout is ``save_pretrained``-style: the checkpoint **directory
   is itself an importable package** holding the encoder code
   (``huggingface_mae.py``, ``vit.py`` ...) *and* ``model.safetensors``, and the
   encoder uses **relative** imports.  So the package's *parent* goes on
   ``sys.path`` and the import is ``openphenom.huggingface_mae``; putting the
   checkpoint directory itself on the path fails with
   "attempted relative import with no known parent package".
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

from cellpaint_pipeline.models.adapters.base import BaseAdapter
from cellpaint_pipeline.models.contract import ModelManifest
from cellpaint_pipeline.models.errors import MissingDependency

__all__ = ['CaMaeAdapter']

#: Encoder module inside the published package.
_ENCODER_MODULE = 'huggingface_mae'


@dataclass
class CaMaeAdapter(BaseAdapter):
    """Adapter for OpenPhenom-S/16 (CA-MAE)."""

    manifest: ModelManifest = None  # type: ignore[assignment]
    #: ``384 * C`` instead of ``384`` when True.  Fixed at load time because the
    #: upstream module exposes it as a mutable attribute read during forward.
    return_channelwise: bool = False
    #: Requested compute device (``None`` -> the adapter decides).  Declared as a
    #: field so the generated ``__init__`` accepts it instead of shadowing
    #: :meth:`BaseAdapter.__init__`, which is what made ``device=`` a TypeError.
    device: str | None = None
    #: Additional roots searched for the checkpoint before the defaults.
    weights_roots: tuple[Any, ...] | None = None

    requires_frameworks: tuple[str, ...] = ('torch',)
    dependency_hint: str | None = (
        'OpenPhenom needs PyTorch. On this machine that is the torch 2.5.1+cu124 '
        'environment, not the TensorFlow one.'
    )

    def __post_init__(self) -> None:
        if self.manifest is None:
            from cellpaint_pipeline.models.registry import manifest as manifest_for

            self.manifest = manifest_for('ca_mae')
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
        import torch

        self._torch = torch

        checkpoint = self.resolve_weights()
        package_dir = checkpoint.parent
        package_name = package_dir.name
        package_parent = package_dir.parent

        if str(package_parent) not in sys.path:
            sys.path.insert(0, str(package_parent))

        try:
            encoder = importlib.import_module(f'{package_name}.{_ENCODER_MODULE}')
        except ImportError as exc:
            raise MissingDependency(
                f'{package_name}.{_ENCODER_MODULE}',
                model=self.manifest.name,
                hint=(
                    f'the published checkpoint ships its own encoder code, which '
                    f'must sit in {package_dir} next to model.safetensors, with '
                    f'{package_parent} importable. ({type(exc).__name__}: {exc})'
                ),
            ) from exc

        config = encoder.MAEConfig.from_pretrained(str(package_dir))
        payload = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
        state = (
            payload['state_dict']
            if isinstance(payload, dict) and 'state_dict' in payload
            else payload
        )

        model = encoder.MAEModel(config)
        # strict=True: a silent partial load here would mean random weights and
        # plausible-looking embeddings, which is worse than a failure.
        model.load_state_dict(state, strict=True)
        model.eval()
        model.return_channelwise_embeddings = bool(self.return_channelwise)

        self._device = self.select_device(torch)
        model.to(self._device)
        return model

    def _forward(self, array: Any, **kwargs: Any) -> Any:
        torch = self._torch
        channelwise = kwargs.get('return_channelwise')
        if channelwise is not None and bool(channelwise) != self.return_channelwise:
            from cellpaint_pipeline.models.errors import EmbeddingError

            raise EmbeddingError(
                'return_channelwise cannot change after load(); construct a new '
                'adapter instead',
                reason='adapter-reuse',
            )

        tensor = array if isinstance(array, torch.Tensor) else torch.as_tensor(array)
        tensor = tensor.to(self._device)
        with torch.no_grad():
            embeddings = self._model.predict(tensor)
        return embeddings.detach().to('cpu').numpy()
