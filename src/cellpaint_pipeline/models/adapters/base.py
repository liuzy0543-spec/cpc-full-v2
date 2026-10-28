"""Shared adapter scaffolding.

WHY THIS MODULE EXISTS
----------------------
Each adapter has to do the same three things before it can embed anything:

1. resolve its weights locally, without ever downloading;
2. import a framework and turn its absence into a typed error;
3. pick a device and move a model onto it.

Written per adapter, that is three near-identical blocks, and the third or
fourth method added by someone else will get one of them subtly wrong -- most
likely the "never download" rule, which is exactly the one whose absence is
expensive (the upstream default silently fetches ~19 MB from Zenodo at ~9 KB/s).

:class:`BaseAdapter` provides those three, plus the small amount of bookkeeping
the dispatcher reads back (``backend``, ``weights_path``, ``feature_layer``).
Subclasses implement only what is genuinely method-specific:

``check_dependencies()``
    Usually one line delegating to :meth:`BaseAdapter.require_frameworks`.
``_build()`
    Construct the network.  Called once, under :meth:`load`.
``_forward()``
    Run the network on an already-validated array.

``load()`` is implemented here and is idempotent, so an adapter author cannot
forget the "already loaded?" guard and accidentally reload 700 MB per call.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from cellpaint_pipeline.models.errors import MissingDependency
from cellpaint_pipeline.models.contract import ModelManifest
from cellpaint_pipeline.models.weights import resolve_weights

__all__ = ['BaseAdapter']


class BaseAdapter(ABC):
    """Common scaffolding for embedding adapters.

    Subclasses must set :attr:`manifest` (the registry normally injects it) and
    implement :meth:`_build` and :meth:`_forward`.
    """

    #: Declared contract.  The registry passes this to the constructor.
    manifest: ModelManifest

    #: Reported in :class:`~cellpaint_pipeline.models.embedding.EmbeddingResult`.
    backend: str = 'inprocess'

    def __init__(
        self,
        *,
        manifest: ModelManifest,
        weights_roots: tuple[Path | str, ...] | None = None,
        device: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.weights_roots = weights_roots
        self.requested_device = device
        self.weights_path: Path | None = None
        #: Digest actually observed (or declared) for the checkpoint.
        self.weights_sha256: str | None = None
        self._model: Any = None
        self._device: str = 'cpu'

    # ------------------------------------------------------------- hooks ----

    @abstractmethod
    def check_dependencies(self) -> None:
        """Raise :class:`MissingDependency` when a framework is unavailable."""

    @abstractmethod
    def _build(self) -> Any:
        """Construct and return the network.  Called once by :meth:`load`."""

    @abstractmethod
    def _forward(self, array: Any, **kwargs: Any) -> Any:
        """Run the network and return an ``(N, output_dim)`` array."""

    #: Framework modules this adapter needs, general-to-specific.  Used by the
    #: default :meth:`check_dependencies`.
    requires_frameworks: tuple[str, ...] = ()

    # ------------------------------------------------------------ helpers ---

    def require_frameworks(self, *modules: str) -> None:
        """Assert that each module is importable, naming the first that is not.

        ``importlib.util.find_spec`` is used rather than ``import`` so the check
        does not pay the import cost (seconds to minutes for TensorFlow) and does
        not partially initialise a framework on a machine that turns out not to
        need it.
        """
        import importlib.util

        for module in modules or self.requires_frameworks:
            if importlib.util.find_spec(module) is None:
                raise MissingDependency(
                    module,
                    model=self.manifest.name,
                    hint=self.dependency_hint,
                )

    #: Overridden by adapters that can name the environment to use.
    dependency_hint: str | None = None

    def resolve_weights(self) -> Path:
        """Locate the checkpoint, and verify it when a hash is declared.

        Verification is on by default.  A benchmark number is only meaningful if
        it can be attributed to a specific artefact, and the layer already
        carries ``weights_sha256`` into every result -- reporting a hash that was
        never checked against the file would be provenance theatre rather than
        provenance.

        ``weights.sha256`` may be recorded as unavailable for a self-trained
        checkpoint, in which case there is nothing to verify and the resolver is
        left in locate-only mode.
        """
        declared = self.manifest.weights.sha256
        resolved = resolve_weights(
            self.manifest.name,
            self.manifest.weights.source,
            roots=self.weights_roots,
            expected_sha256=declared,
            verify=declared is not None,
        )
        self.weights_path = resolved.path
        self.weights_sha256 = getattr(resolved, 'sha256', None) or declared
        return resolved.path

    def select_device(self, torch_module: Any) -> str:
        """Return the device string, honouring an explicit request."""
        if self.requested_device:
            return self.requested_device
        return 'cuda' if torch_module.cuda.is_available() else 'cpu'

    # ------------------------------------------------------------ protocol ---

    def load(self) -> None:
        """Build the model.  Idempotent: repeated calls are free."""
        if self._model is not None:
            return
        self._model = self._build()

    def embed(self, array: Any, **kwargs: Any) -> Any:
        """Embed ``array``, loading the model on first use."""
        self.load()
        return self._forward(array, **kwargs)

    def embed_many(self, arrays: Any, **kwargs: Any) -> Any:
        """Embed several payloads, reusing the loaded model.

        The default implementation concatenates per-payload results.  Adapters
        whose framework batches better on its own may override this.
        """
        import numpy as np

        return np.concatenate(
            [np.asarray(self._forward(array, **kwargs)) for array in arrays], axis=0
        )
