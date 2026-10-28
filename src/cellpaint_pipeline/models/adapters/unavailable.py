"""Placeholder adapter for methods that are declared but cannot run yet.

WHY THIS EXISTS
---------------
Three of the seven methods have no usable weights on this machine, all for the
same reason: the only published copies sit behind hosts that are unreachable or
unusably slow here (Zenodo throttling to ~10 KB/s, Google Drive timing out).
The other four -- including O2-VAE, whose official MEF checkpoint *is* published
in its upstream repository -- are implemented in sibling modules.

Two designs were possible:

1. omit them from the registry, so ``models.available()`` lists only what runs;
2. declare them, and make the failure typed and explanatory.

(2) was chosen.  A caller asking for ``unidino`` should learn *what* is missing
-- a 1.38 GB download at ~43 hours, or an absent dependency -- and be able to
inspect the method's real input contract.  Declaring the manifest also keeps the
contracts testable now, so when a weight arrives only the adapter changes.

The error raised is always :class:`WeightsUnavailable` or
:class:`MissingDependency`, never a bare ``ImportError`` or ``KeyError``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cellpaint_pipeline.models.errors import WeightsUnavailable
from cellpaint_pipeline.models.contract import ModelManifest

__all__ = ['UnavailableAdapter', 'make_unavailable_adapter']


@dataclass
class UnavailableAdapter:
    """Declared method whose weights or dependencies are not available here.

    Implements the full :class:`~cellpaint_pipeline.models.embedding.ModelAdapter`
    protocol so the dispatcher needs no special case; the only difference is
    that :meth:`load` explains itself instead of proceeding.
    """

    manifest: ModelManifest
    weights_path: Any = None
    backend: str = 'unavailable'
    _reason: str = field(default='', init=False, repr=False)

    def check_dependencies(self) -> None:
        """Report missing dependencies before the weight problem.

        Dependency status is genuinely useful here: a user who intends to fetch
        the weights wants to know whether the environment can run them at all.
        """
        import importlib.util

        from cellpaint_pipeline.models.errors import MissingDependency

        for module in self.manifest.requires:
            if importlib.util.find_spec(module) is None:
                raise MissingDependency(
                    module,
                    model=self.manifest.name,
                    hint=(
                        f'{self.manifest.name} is declared but not runnable here: '
                        f'the module {module!r} is absent. Its weights are also '
                        f'not available locally '
                        f'({self.manifest.weights.host}, '
                        f'downloadable={self.manifest.weights.downloadable}).'
                    ),
                )

    def load(self) -> None:
        """Always fails, with the manifest's own explanation of why."""
        host = self.manifest.weights.host
        hint = self.manifest.notes or (
            f'weights host={host}, downloadable='
            f'{self.manifest.weights.downloadable}'
        )
        raise WeightsUnavailable(
            self.manifest.name,
            probed=(self.manifest.weights.source,),
            hint=hint,
        )

    def embed(self, array: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """Unreachable: :meth:`load` raises first."""
        self.load()
        raise AssertionError('unreachable')


def make_unavailable_adapter(manifest: ModelManifest) -> UnavailableAdapter:
    """Factory used by the registry for declared-but-unrunnable methods."""
    return UnavailableAdapter(manifest=manifest)
