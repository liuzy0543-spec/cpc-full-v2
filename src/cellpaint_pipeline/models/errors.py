"""Error types for the embedding layer.

WHY THIS MODULE EXISTS
----------------------
The embedding layer has three genuinely different failure modes that a caller
must be able to tell apart, because the remedy differs:

* :class:`IncompatibleInput` -- *your input is wrong* (shape, channel order,
  value range).  Fix the call.  Raised **before** any weight is loaded.
* :class:`MissingDependency` -- *this machine cannot run this method* (torch or
  TensorFlow absent).  Fix the environment; the call was fine.
* :class:`WeightsUnavailable` -- *the weights are not here*.  Fetch them; the
  environment and the call were fine.

Collapsing these into one exception (or letting a bare ``ImportError`` escape)
is what makes embedding pipelines hard to operate: the operator cannot tell
whether to change the data, the env, or the disk.

All three derive from :class:`cellpaint_pipeline.errors.PipelineError`, so
existing ``except PipelineError`` handlers keep working and the new layer joins
the project's single error hierarchy rather than starting a second one.
"""
from __future__ import annotations

from cellpaint_pipeline.errors import PipelineError

__all__ = [
    'EmbeddingError',
    'IncompatibleInput',
    'MissingDependency',
    'UnknownModel',
    'WeightsUnavailable',
]


class EmbeddingError(PipelineError):
    """Base class for every embedding-layer failure."""


class UnknownModel(EmbeddingError):
    """No adapter is registered under the requested name."""

    def __init__(self, name: str, available: tuple[str, ...]) -> None:
        self.name = name
        self.available = tuple(available)
        super().__init__(
            f"Unknown embedding model {name!r}. Available: "
            f"{', '.join(self.available) if self.available else '(none)'}",
            reason='unknown-model',
            details=[f'available: {list(self.available)}'],
        )


class IncompatibleInput(EmbeddingError):
    """The supplied input does not satisfy the model's declared contract.

    Always raised by :meth:`CompatibilityReport.raise_for_status` before the
    adapter touches a weight file, so a contract violation never costs a model
    load.
    """


class MissingDependency(EmbeddingError):
    """A framework the adapter needs is not importable in this interpreter.

    Carries the exact ``import`` name and a copy-pasteable hint, so the operator
    is not left guessing which of several environments to use.
    """

    def __init__(self, module: str, *, model: str, hint: str | None = None) -> None:
        self.module = module
        self.model = model
        details = [f"model: {model}", f"missing import: {module}"]
        if hint:
            details.append(f"hint: {hint}")
        super().__init__(
            f"Cannot run embedding model {model!r}: the module {module!r} is not "
            f"installed in this interpreter.",
            reason='missing-dependency',
            details=details,
        )


class WeightsUnavailable(EmbeddingError):
    """The model's weights cannot be located, and implicit download is refused.

    This layer never downloads: a silent 600 MB fetch (or a 40-minute Zenodo
    transfer) inside an ``embed()`` call is not acceptable behaviour.  The error
    carries the paths that were probed and an explicit instruction.
    """

    def __init__(self, model: str, *, probed: tuple[str, ...], hint: str | None = None) -> None:
        self.model = model
        self.probed = tuple(probed)
        # Kept as an attribute as well as folded into `details`: a caller that
        # wants to branch on the reason (or surface it in a UI) should not have
        # to parse the message text.
        self.hint = hint
        details = [f"model: {model}"] + [f"probed: {p}" for p in self.probed]
        if hint:
            details.append(f"hint: {hint}")
        super().__init__(
            f"Weights for embedding model {model!r} were not found locally. "
            f"Implicit download is disabled.",
            reason='weights-unavailable',
            details=details,
        )
