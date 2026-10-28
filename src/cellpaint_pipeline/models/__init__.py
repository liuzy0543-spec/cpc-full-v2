"""Unified input/output interface for Cell Painting embedding methods.

WHAT THIS LAYER IS FOR
----------------------
One call, one shape, whichever method::

    from cellpaint_pipeline import models

    models.available()                # ['deepprofiler', 'ca_mae', 'cellclip', ...]
    models.runnable()                 # the subset that can embed on this machine
    models.describe('deepprofiler')   # the full contract as a dict
    models.probe('ca_mae')            # can this machine run it, and why not
    models.embed('deepprofiler', crops)

The methods disagree about almost everything -- array layout, channel count,
dtype, value range, and even whether the input is pixels at all (CellCLIP
consumes precomputed channel embeddings).  The unification therefore lives in
the **contract**, not in a single tensor shape:

``manifest``
    Declares what a method eats and produces.  Importable with no framework.
``check_compatible``
    A pure function that decides, before anything is loaded, whether a call can
    work and precisely why not.
``embed``
    Validates, then dispatches to an adapter, then wraps the array with
    provenance.

ADDING A METHOD
---------------
Append one :class:`~cellpaint_pipeline.models.registry.AdapterRegistration` to
``registry.ADAPTERS`` and write the adapter module.  Nothing else changes --
``available``, ``describe``, ``manifest`` and ``embed`` all read that table.
Downstream code may also call
:func:`~cellpaint_pipeline.models.registry.register` at runtime.

WHY THE CONTRACT MODULE IS CALLED ``contract`` AND NOT ``manifest``
------------------------------------------------------------------
``models.manifest`` is the function callers want, but a submodule of that name
would win the attribute on the package as soon as anything imported it, so
``models.manifest('deepprofiler')`` would raise ``TypeError: 'module' object is
not callable``.  The contract module is therefore named
:mod:`~cellpaint_pipeline.models.contract`; ``manifest`` stays a function.  The
registry functions are additionally wrapped below rather than re-exported, so
the same trap cannot reappear if another submodule is added later.
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    'ADAPTERS',
    'AdapterRegistration',
    'BatchResult',
    'ChannelSpec',
    'CompatibilityIssue',
    'CompatibilityReport',
    'EmbeddingError',
    'EmbeddingResult',
    'InputUnit',
    'IncompatibleInput',
    'MissingDependency',
    'ModelAdapter',
    'ModelManifest',
    'ParityMode',
    'PreprocessSpec',
    'ResolvedWeights',
    'UnknownModel',
    'WeightsSpec',
    'WeightsUnavailable',
    'active_profile',
    'available',
    'check_compatible',
    'declared',
    'describe',
    'doctor',
    'embed',
    'embed_all',
    'embed_many',
    'manifests',
    'plan',
    'prepare',
    'probe',
    'register',
    'resolve_weights',
    'runnable',
    'profile_names',
    'set_profile',
]

_SELF = 'cellpaint_pipeline.models.'

#: Facade name -> defining module, for names resolved lazily on first access.
_LAZY_SOURCES: dict[str, str] = {}


def _declare(module: str, *names: str) -> None:
    for name in names:
        _LAZY_SOURCES[name] = module


_declare(_SELF + 'contract',
         'ChannelSpec', 'CompatibilityIssue', 'CompatibilityReport',
         'InputUnit', 'ModelManifest', 'ParityMode', 'PreprocessSpec',
         'WeightsSpec', 'check_compatible')
_declare(_SELF + 'errors',
         'EmbeddingError', 'IncompatibleInput', 'MissingDependency',
         'UnknownModel', 'WeightsUnavailable')
_declare(_SELF + 'weights', 'ResolvedWeights', 'resolve_weights')
_declare(_SELF + 'embedding',
         'BatchResult', 'EmbeddingResult', 'ModelAdapter', 'embed', 'embed_all',
         'embed_many', 'probe', 'clear_adapter_cache')
_declare(_SELF + 'preprocessing', 'PrepareStep', 'plan', 'prepare')


def _registry() -> Any:
    """Import the registry on call.

    Deferred so that the registry submodule is not bound as a package attribute
    during init, which would shadow the ``manifest`` function below.
    """
    return importlib.import_module(_SELF + 'registry')


# --------------------------------------------------------------------------
# Thin wrappers.  Each imports its implementation on call rather than at import
# time, for the shadowing reason documented above.  The cost is one dict lookup
# per call; `importlib.import_module` returns the cached module.
# --------------------------------------------------------------------------

def declared() -> list[str]:
    """Return every declared method name, in declaration order.

    This is the full table -- including methods that are registered only so
    their contract is visible while their weights are unavailable.  For "what
    can actually run on this machine", use :func:`runnable`.
    """
    return _registry().declared()


def available() -> list[str]:
    """Deprecated alias of :func:`declared`.

    The name promised more than it delivered: it listed every declared method,
    so a caller could reasonably read it as "these seven all work", when three of
    them have no weights and no implementation.  Use :func:`runnable` to ask the
    question this name appears to ask.
    """
    return declared()


def runnable() -> list[str]:
    """Return the declared methods that can actually embed on this machine.

    Equivalent to the ``runnable`` flag in :func:`doctor`, without the rest of
    the report.  This imports no framework: :func:`probe` inspects the
    environment via ``importlib.util.find_spec`` and the filesystem.

    Note that this is a property of the *interpreter and the weight cache*, not
    of the project: the same checkout answers differently on a machine with
    torch installed and the checkpoints downloaded.
    """
    probe = __getattr__('probe')
    return [name for name in declared() if probe(name)['runnable']]


def manifests() -> list[Any]:
    """Return every declared manifest, in declaration order."""
    return _registry().manifests()


def pending() -> list[Any]:
    """Return the contracts of methods that were cancelled (not registered).

    They cannot be embedded -- no weights are obtainable here -- but the
    contracts stay reviewable so restoring one later is a small change rather
    than a re-read of the upstream sources.
    """
    return _registry().pending()


def active_profile() -> str:
    """Return the profile in force: ``'v2'`` (official weights) or ``'v3'``."""
    return _registry().active_profile()


def profile_names() -> list[str]:
    """Return the selectable profiles."""
    return _registry().profile_names()


def set_profile(name: str) -> str:
    """Switch profile, returning the previous one.

    ``v2`` lists the four methods that can run with the authors' weights.
    ``v3`` also lists three methods paired with locally trained **surrogate**
    checkpoints -- their interface is real, their scores are not the published
    methods' and must never be reported as such.  Every result from one of them
    carries ``metadata['weights_provenance'] == 'surrogate'``.
    """
    return _registry().set_profile(name)


def manifest(name: str) -> Any:
    """Return the :class:`ModelManifest` for ``name``.

    Imports nothing heavier than the contract module, so this works on a machine
    with neither TensorFlow nor torch installed.
    """
    return _registry().manifest(name)


def describe(name: str) -> dict:
    """Return a JSON-serialisable description of one method."""
    return _registry().describe(name)


def register(registration: Any, *, replace: bool = False) -> None:
    """Add an embedding method at runtime."""
    return _registry().register(registration, replace=replace)


def clear_adapter_cache() -> None:
    """Release every loaded model held by the dispatcher.

    ``embed`` keeps a loaded adapter per method so repeat calls do not reload the
    checkpoint -- measured, that took a per-sample loop from ~7 s per call to
    ~0.03 s.  The price is that each cached adapter keeps its model resident, so
    this exists for a caller that is done with a method and wants the memory back.
    """
    from cellpaint_pipeline.models.embedding import clear_adapter_cache as _clear

    return _clear()


def __getattr__(name: str) -> Any:
    """Resolve a remaining facade name on first use, then cache it.

    ``ADAPTERS`` and ``AdapterRegistration`` are fetched through here rather than
    wrapped, because neither is shadowed by a submodule name.
    """
    if name in {'ADAPTERS', 'AdapterRegistration'}:
        value = getattr(_registry(), name)
        globals()[name] = value
        return value

    module_name = _LAZY_SOURCES.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def doctor() -> list[dict]:
    """Report, for every declared method, whether it can run here.

    Never raises and never imports a framework: the first command a new operator
    should run on a fresh machine.
    """
    probe = __getattr__('probe')
    return [probe(name) for name in declared()]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SOURCES) | {'ADAPTERS', 'AdapterRegistration'})
