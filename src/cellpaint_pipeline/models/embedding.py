"""The single entry point: ``models.embed(name, x)``.

WHY THIS MODULE EXISTS
----------------------
The goal is that a caller who knows *which* method they want does not have to
know how it is loaded, which environment it needs, or what shape it eats::

    from cellpaint_pipeline import models
    result = models.embed('deepprofiler', crops)

Everything that differs between methods is pushed behind two seams:

* :class:`ModelAdapter` -- the protocol an adapter implements.  It has three
  responsibilities and nothing else: check dependencies, load weights, embed.
* :func:`embed` -- resolve, validate, dispatch, wrap.

VALIDATION HAPPENS BEFORE LOADING
---------------------------------
:func:`embed` always runs :func:`~cellpaint_pipeline.models.contract.check_compatible`
first.  A contract violation therefore never costs a model load (seconds) or a
dependency import (seconds to minutes on a cold cache).  It also means the same
contract can be exercised by tests on a machine with no frameworks installed.

DEPENDENCY ERRORS ARE TYPED
---------------------------
An adapter that needs torch but finds it missing raises
:class:`~cellpaint_pipeline.models.errors.MissingDependency` naming the module,
rather than letting a bare ``ImportError`` escape from deep inside a third-party
package.  The distinction matters operationally: "install X" and "your array is
the wrong shape" are different problems.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from cellpaint_pipeline.models import registry
from cellpaint_pipeline.models.errors import (
    EmbeddingError,
    IncompatibleInput,
    MissingDependency,
    UnknownModel,
)
from cellpaint_pipeline.models.contract import (
    CompatibilityIssue,
    CompatibilityReport,
    InputUnit,
    ModelManifest,
    check_compatible,
)

__all__ = [
    'BatchResult',
    'EmbeddingResult',
    'ModelAdapter',
    'embed',
    'embed_all',
    'embed_many',
    'probe',
]


@dataclass(frozen=True)
class EmbeddingResult:
    """The uniform output of every embedding method.

    ``values`` keeps whatever array type the adapter produced (numpy in
    practice) so the layer does not dictate a tensor library.  Everything else
    is plain data, so a result can be logged or serialised without importing the
    framework that generated it.

    Row identity: an ``(N, D)`` array alone cannot say *which* cell each row
    belongs to, and every downstream comparison needs that.  Adapters pass
    whatever they know through ``sample_metadata``; the layer does not invent it.
    """

    model: str
    values: Any
    dim: int
    input_unit: InputUnit
    #: ``inprocess`` or ``cli`` -- for DeepProfiler, which has both.
    backend: str = 'inprocess'
    feature_layer: str | None = None
    n_samples: int | None = None
    weights_path: str | None = None
    weights_sha256: str | None = None
    license: str = 'UNKNOWN'
    #: ``deterministic`` or ``stochastic``, copied from the manifest.
    #:
    #: Surfaced on every result on purpose: a consumer comparing methods must
    #: know whether a row is a measurement or a single random draw.
    determinism: str = 'deterministic'
    metadata: dict = field(default_factory=dict)
    #: Per-row annotations, e.g. ``{'cell_id': [...], 'plate': [...]}``.  Each
    #: value must have ``n_samples`` entries.
    sample_metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.n_samples if self.n_samples is not None else 0

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(getattr(self.values, 'shape', ()))

    # ------------------------------------------------------------- export ---

    def as_numpy(self, *, dtype: str | None = None) -> Any:
        """Return the embeddings as a numpy array.

        The common case for benchmarking, and the one export that must never
        fail on a machine that ran an embedding at all -- numpy is already a
        transitive dependency.
        """
        import numpy as np

        array = np.asarray(self.values)
        return array.astype(dtype, copy=False) if dtype else array

    def as_torch(self, *, device: str | None = None) -> Any:
        """Return the embeddings as a torch tensor.

        Imported lazily: a TensorFlow-only environment can still consume a
        result as numpy without torch installed.
        """
        from cellpaint_pipeline.models.errors import MissingDependency

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise MissingDependency('torch', model=self.model) from exc

        tensor = torch.as_tensor(self.as_numpy())
        return tensor.to(device) if device else tensor

    def to_frame(self, *, prefix: str | None = None) -> Any:
        """Return a pandas DataFrame of ``sample_metadata`` plus feature columns.

        Column naming follows the project's existing DeepProfiler tables
        (``Cells_DeepProfiler_0001``...), so a frame from this layer can be
        concatenated with the pipeline's own output without renaming.  The
        prefix is derived from the model name when not supplied.
        """
        import pandas as pd

        tag = prefix or f'Cells_{self.model}'
        columns = [f'{tag}_{index:04d}' for index in range(1, self.dim + 1)]
        frame = pd.DataFrame(self.as_numpy(), columns=columns)
        if self.sample_metadata:
            meta = pd.DataFrame(self.sample_metadata)
            if len(meta) != len(frame):
                from cellpaint_pipeline.models.errors import EmbeddingError

                raise EmbeddingError(
                    f'sample_metadata has {len(meta)} rows but the embeddings have '
                    f'{len(frame)}; they must correspond one-to-one',
                    reason='metadata-length',
                )
            frame = pd.concat([meta.reset_index(drop=True), frame], axis=1)
        return frame

    def save_npz(self, path: Any, *, compressed: bool = True) -> Any:
        """Write ``values``, provenance and per-row identity to a ``.npz``.

        The layout mirrors the upstream DeepProfiler artefact (``features`` plus a
        ``metadata`` mapping) so existing consumers keep working.

        Per-row annotations are written as separate ``sample_<key>`` arrays.
        Without them an artefact cannot be aligned to anything: a benchmark that
        writes only `(N, D)` loses which cell each row belongs to, and no later
        step can recover it.  ``sample_metadata`` was being dropped here, which
        made every benchmark ``.npz`` unusable for cross-method comparison.
        """
        import json
        from pathlib import Path as _Path

        import numpy as np

        destination = _Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'features': self.as_numpy(),
            'metadata': np.array(
                json.dumps(
                    {
                        'model': self.model,
                        'backend': self.backend,
                        'feature_layer': self.feature_layer,
                        'dim': self.dim,
                        'n_samples': self.n_samples,
                        'weights_sha256': self.weights_sha256,
                        'license': self.license,
                        'determinism': self.determinism,
                        'sample_metadata_keys': sorted(self.sample_metadata),
                        'sample_metadata': self.sample_metadata,
                    }
                )
            ),
        }
        for key, values in self.sample_metadata.items():
            array = np.asarray(values)
            if array.shape[0] != payload['features'].shape[0]:
                raise ValueError(
                    f"sample_metadata[{key!r}] has {array.shape[0]} entries but "
                    f"the embeddings have {payload['features'].shape[0]}; they "
                    f"must correspond one-to-one"
                )
            payload[f'sample_{key}'] = array
        (np.savez_compressed if compressed else np.savez)(destination, **payload)
        return destination

    def to_dict(self) -> dict:
        """JSON-serialisable summary.  ``values`` is described, not inlined."""
        return {
            'model': self.model,
            'backend': self.backend,
            'input_unit': self.input_unit.value,
            'feature_layer': self.feature_layer,
            'dim': self.dim,
            'n_samples': self.n_samples,
            'shape': list(self.shape),
            'weights_path': self.weights_path,
            'weights_sha256': self.weights_sha256,
            'license': self.license,
            'determinism': self.determinism,
            'metadata': dict(self.metadata),
        }


@runtime_checkable
class ModelAdapter(Protocol):
    """What every embedding adapter must implement.

    Deliberately three methods.  An adapter that needs more (a tokenizer, a
    two-stage pipeline) does that work inside :meth:`embed` rather than widening
    this protocol, so the dispatcher stays method-agnostic.
    """

    manifest: ModelManifest

    def check_dependencies(self) -> None:
        """Raise :class:`MissingDependency` when a framework is absent."""
        ...

    def load(self) -> None:
        """Materialise weights.  Idempotent; cheap on the second call."""
        ...

    def embed(self, array: Any, **kwargs: Any) -> Any:
        """Return an ``(N, output_dim)`` array for ``array``."""
        ...


def _import_guard(manifest: ModelManifest) -> None:
    """Fail with a typed error when a declared dependency is missing.

    Only the *first* declared requirement is probed: ``requires`` is ordered
    general-to-specific, and reporting one concrete missing module is more
    actionable than listing all of them.
    """
    import importlib.util

    for module in manifest.requires:
        if importlib.util.find_spec(module) is None:
            raise MissingDependency(module, model=manifest.name)


def probe(name: str) -> dict:
    """Report whether a method can run **in this interpreter**.

    Used by ``models.doctor()`` and by tests.  Never imports a framework.

    READ THE RESULT CAREFULLY
    -------------------------
    Runnable-ness is a property of an *interpreter*, not of the project.  This
    layer deliberately supports several environments -- the TensorFlow stack for
    DeepProfiler, the torch stack for the others -- so ``runnable=False`` here
    can simply mean "you are asking the wrong interpreter", not "this method is
    broken".  That is why the result carries :data:`requires` and
    :data:`missing_dependencies` separately: the caller can tell "install the
    missing module" apart from "run this under the other environment".
    """
    manifest_ = registry.manifest(name)
    import importlib.util

    # Ask the adapter that will actually run, not the manifest.  Under the v3
    # profile the manifest still describes the *official* implementation (its
    # dependency list is part of the published contract), while the surrogate
    # that runs needs a strictly smaller set.  Reading the manifest here made
    # `probe('dinocells')` report "missing oyaml" and therefore
    # "not runnable" on a machine where `embed('dinocells', ...)` worked --
    # the diagnostic told the operator the method was broken when it was not.
    required = registry.entry(name).requires_frameworks
    missing = [m for m in required if importlib.util.find_spec(m) is None]

    from cellpaint_pipeline.models.weights import (
        DEFAULT_WEIGHTS_ROOTS,
        WEIGHTS_ROOT_ENV,
        resolve_weights,
    )

    import os

    weights_ok, weights_detail = False, None
    if manifest_.weights.format != 'none':
        try:
            resolved = resolve_weights(name, manifest_.weights.source)
            weights_ok = True
            weights_detail = str(resolved.path)
        except Exception as exc:  # noqa: BLE001 - probe never raises
            roots = (
                [os.environ[WEIGHTS_ROOT_ENV]]
                if os.environ.get(WEIGHTS_ROOT_ENV, '').strip()
                else [str(r) for r in DEFAULT_WEIGHTS_ROOTS]
            )
            weights_detail = f'{type(exc).__name__}: searched {roots}'

    # An adapter that imports the published architecture from a repository --
    # rather than vendoring it -- is not runnable on its weights alone.  The
    # repository location is declared in the manifest because it cannot be
    # inferred from the checkpoint path.
    repo_ok: bool | None = None
    repo_detail: str | None = None
    if manifest_.requires_upstream_repo:
        from pathlib import Path

        repo_rel = manifest_.weights.repo_source
        if repo_rel is None:
            repo_ok = False
            repo_detail = 'manifest sets requires_upstream_repo but no repo_source'
        else:
            # Resolve through the same root that produced the weights, so a
            # relocated weights store stays consistent.  For a source one level
            # deep (``o2vae/o2vae_50ep.pt``) the root is the checkpoint's parent.
            repo_root = None
            if weights_detail:
                source_parts = [
                    p for p in manifest_.weights.source.replace('\\', '/').split('/') if p
                ]
                repo_root = Path(weights_detail).parent
                for _ in range(len(source_parts) - 1):
                    repo_root = repo_root.parent
            if repo_root is None:
                repo_ok = False
                repo_detail = 'weights not found, so the repository cannot be located'
            else:
                candidate = repo_root / repo_rel
                if (candidate / 'models').is_dir():
                    repo_ok = True
                    repo_detail = str(candidate)
                else:
                    repo_ok = False
                    repo_detail = f'no importable `models` package under {candidate}'

    runnable = not missing and weights_ok and (repo_ok is not False)

    if runnable:
        status = 'ready'
    elif repo_ok is False:
        status = 'needs-upstream-repo'
    elif missing and not weights_ok:
        status = 'needs-deps-and-weights'
    elif missing:
        # The common case in a multi-environment project: weights are on disk,
        # this interpreter just lacks the framework.
        status = 'wrong-interpreter'
    else:
        status = 'needs-weights'

    return {
        'model': name,
        'runnable': runnable,
        'status': status,
        # The *effective* requirement list -- what the adapter that will run
        # needs -- not the manifest's official-requirement list.
        'requires': list(required),
        'manifest_requires': list(manifest_.requires),
        'missing_dependencies': missing,
        'weights_found': weights_ok,
        'weights_detail': weights_detail,
        'repo_found': repo_ok,
        'repo_detail': repo_detail,
        'license': manifest_.license,
    }


#: Loaded adapters, keyed by (method, declared weights digest).
#:
#: Why this exists: ``embed`` builds an adapter per call, and building one loads
#: the checkpoint.  For CA-MAE that is 712 MB of disk plus torch initialisation;
#: measured, a per-sample loop spent essentially all of its time reloading rather
#: than computing.  A benchmark that embeds a dataset is exactly that loop, so
#: without a cache the dispatcher is unusable for the thing it was built for.
#:
#: Safety: an adapter is read-only after ``load()`` -- ``_forward`` only reads
#: ``self._model`` -- so sharing one across calls cannot leak state between
#: embeddings.  The key includes the declared weights digest, so replacing a
#: checkpoint produces a different key instead of silently reusing the old model.
#: Memory is the cost: each entry keeps a model resident.  Call
#: :func:`clear_adapter_cache` to release them.
_ADAPTER_CACHE: dict[tuple, ModelAdapter] = {}

#: Fingerprint of the weights file each cached adapter was built from.
#:
#: Why this exists: the cache key contains the **declared** digest, which comes
#: from the manifest and therefore never changes when the file on disk does.
#: So a checkpoint replaced mid-session kept being served from the cache, and
#: the ``weights_sha256`` reported alongside the embedding still described the
#: *declared* artefact, not the file that was actually loaded -- a provenance
#: statement that had quietly become false.
#:
#: The fingerprint is ``(path, size, mtime_ns)``: one ``stat`` per call, which
#: is nothing next to a forward pass, and it detects a replacement without
#: re-hashing a multi-gigabyte checkpoint.  A change invalidates the entry and
#: the load is redone, which re-verifies the digest -- so a swapped file is
#: rejected exactly as a fresh load would reject it.
_ADAPTER_FINGERPRINTS: dict[tuple, tuple] = {}


def _weights_fingerprint(adapter: ModelAdapter) -> tuple | None:
    """Return ``(path, size, mtime_ns)`` of the file an adapter was built from."""
    path = getattr(adapter, 'weights_path', None)
    if path is None:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (str(path), stat.st_size, stat.st_mtime_ns)


def clear_adapter_cache() -> None:
    """Drop every cached adapter, releasing the loaded models."""
    _ADAPTER_CACHE.clear()
    _ADAPTER_FINGERPRINTS.clear()


def _build_adapter(name: str, **adapter_kwargs: Any) -> ModelAdapter:
    """Instantiate the adapter for ``name`` (imports its module).

    The manifest is injected rather than left for the adapter to look up, so an
    adapter has exactly one source for its contract and cannot drift from the
    registry.  ``make_unavailable_adapter`` is the one factory that takes it
    positionally.  ``adapter_kwargs`` reach the adapter's constructor, which is
    how ``device``, ``weights_roots`` and ``feature_layer`` become settable from
    the public API instead of requiring a hand-built adapter.
    """
    registration = registry.entry(name)
    factory = registration.load_factory()
    manifest = registration.manifest
    if registration.factory == 'make_unavailable_adapter':
        return factory(manifest)
    return factory(manifest=manifest, **adapter_kwargs)


def _cached_adapter(
    name: str,
    *,
    use_cache: bool = True,
    device: str | None = None,
    weights_roots: Any = None,
    adapter_kwargs: dict | None = None,
) -> ModelAdapter:
    """Return a loaded adapter for ``name``, building it only when needed.

    The cache key includes everything that changes the loaded object -- the
    weights digest, the requested device, and the extra constructor arguments --
    so asking for the same method on a different device builds a separate
    instance rather than silently returning the CPU one.
    """
    manifest_ = registry.manifest(name)
    merged = dict(adapter_kwargs or {})
    if device is not None:
        merged['device'] = device
    if weights_roots is not None:
        merged['weights_roots'] = (
            tuple(str(root) for root in weights_roots)
            if isinstance(weights_roots, (list, tuple))
            else (str(weights_roots),)
        )

    key = (
        name,
        manifest_.weights.sha256,
        merged.get('device'),
        merged.get('weights_roots'),
        tuple(sorted((k, str(v)) for k, v in merged.items())),
    )

    if use_cache:
        cached = _ADAPTER_CACHE.get(key)
        if cached is not None:
            recorded = _ADAPTER_FINGERPRINTS.get(key)
            current = _weights_fingerprint(cached)
            if recorded is None or current is None or current == recorded:
                return cached
            # The weights file changed under us.  Drop the entry and rebuild:
            # the rebuild re-verifies the digest, so a swapped checkpoint is
            # rejected exactly as a fresh load would reject it, instead of the
            # cache quietly serving the old weights under the old digest.
            _ADAPTER_CACHE.pop(key, None)
            _ADAPTER_FINGERPRINTS.pop(key, None)

    adapter = _build_adapter(name, **merged)
    adapter.check_dependencies()
    adapter.load()
    if use_cache:
        _ADAPTER_CACHE[key] = adapter
        fingerprint = _weights_fingerprint(adapter)
        if fingerprint is not None:
            _ADAPTER_FINGERPRINTS[key] = fingerprint
    return adapter


def _promote_single_sample(array: Any, manifest_: ModelManifest) -> Any:
    """Give a single crop the batch axis every adapter expects.

    ``prepare`` is a pure array transform and preserves rank, so a ``(H, W, C)``
    crop stays 3-D.  The adapters feed a framework placeholder that wants a batch,
    which is why a bare 3-D crop used to die inside Keras with "expected input_2
    to have 4 dimensions".

    The promotion is deliberately conservative: it fires only when the shape
    *provably* describes one sample in the manifest's own layout.  A 3-D array of
    shape ``(N, C, H)`` -- a batch of torch layouts -- must not be mistaken for a
    single ``(C, H, W)`` crop, so when the shape does not match the declaration
    the array is passed through unchanged and the adapter/contract reports the
    real problem.
    """
    shape = tuple(getattr(array, 'shape', ()) or ())
    if len(shape) != 3:
        return array
    if manifest_.input_unit is InputUnit.CHANNEL_EMBEDDINGS:
        # (N, C, D) is already batched.
        return array

    spec = manifest_.preprocess
    channels = len(manifest_.channel_names)
    spatial = manifest_.spatial
    if spec.batch_layout == 'BHWC':
        single = (
            shape[-1] == channels
            if spatial is None
            else (tuple(shape[0:2]) == tuple(spatial) and shape[2] == channels)
        )
    else:
        single = (
            shape[0] == channels
            if spatial is None
            else (tuple(shape[1:3]) == tuple(spatial) and shape[0] == channels)
        )
    return array[None, ...] if single else array


def _check_output_width(
    *, name: str, manifest_: ModelManifest, values: Any
) -> None:
    """Refuse an embedding whose width contradicts the manifest.

    The input side of the contract was checked; the output side was not, so a
    manifest that said 672 while the adapter returned 1280 produced a result
    object whose ``dim`` silently disagreed with ``manifest.output_dim``.  A
    downstream benchmark then aligned those embeddings against arrays sized from
    the declaration, which is the kind of mismatch that survives all the way into
    a published number.

    Only the last axis is checked: that is the feature width for every method
    here, and the per-channel shapes are covered by ``output_dim_per_channel``.
    """
    shape = getattr(values, 'shape', None)
    if shape is None or len(shape) < 1:
        return
    width = int(shape[-1])

    allowed = {int(manifest_.output_dim)}
    allowed.update(int(alt) for alt in manifest_.output_dim_alternatives)
    per_channel = manifest_.output_dim_per_channel

    legal = width in allowed
    if not legal and per_channel:
        legal = width > 0 and width % int(per_channel) == 0
    if legal:
        return

    expected = ', '.join(str(value) for value in sorted(allowed))
    if per_channel:
        expected += f', or a multiple of {per_channel} (per-channel output)'
    raise EmbeddingError(
        f"'{name}' returned width {width} but its manifest declares {expected}; "
        'the loaded checkpoint or feature layer does not match the declared '
        'contract',
        reason='output-width',
        details=[f'observed width: {width}', f'declared widths: {expected}'],
    )


def _wrap_result(
    *,
    name: str,
    manifest_: ModelManifest,
    adapter: ModelAdapter,
    values: Any,
    report: CompatibilityReport,
    sample_metadata: dict | None = None,
    extra_metadata: dict | None = None,
) -> EmbeddingResult:
    """Assemble an :class:`EmbeddingResult` from an adapter's raw output.

    Shared by :func:`embed` and :func:`embed_many` so provenance fields are
    populated identically on both paths -- a divergence there would show up as a
    subtly different artefact depending on which entry point was used.
    """
    _check_output_width(name=name, manifest_=manifest_, values=values)

    shape = tuple(getattr(values, 'shape', ()))
    n_samples = int(shape[0]) if len(shape) >= 1 else None
    dim = int(shape[-1]) if len(shape) >= 1 else manifest_.output_dim
    weights_path = getattr(adapter, 'weights_path', None)

    metadata = {
        'channels': list(manifest_.channel_names),
        'preprocess_scale': manifest_.preprocess.scale,
        'contract_codes': list(report.codes),
        # Carried through so a consumer sees every model in the chain, not just
        # the final checkpoint.
        'chain_weights': [list(pair) for pair in manifest_.chain_weights],
        # 'official' or 'surrogate'.  Present in every result, not only the
        # surrogate ones: a consumer must be able to check the provenance of the
        # numbers it is about to compare, and an absent key is easy to read as
        # "official".  A surrogate's scores are not the published method's.
        'weights_provenance': manifest_.weights_provenance,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return EmbeddingResult(
        model=name,
        values=values,
        dim=dim,
        input_unit=manifest_.input_unit,
        backend=getattr(adapter, 'backend', 'inprocess'),
        feature_layer=getattr(adapter, 'feature_layer', manifest_.feature_layer),
        n_samples=n_samples,
        weights_path=str(weights_path) if weights_path else None,
        # The digest of the artefact that was actually loaded, which is what a
        # provenance chain needs.  Falling back only to the declaration meant a
        # correctly-loaded checkpoint could report no hash at all -- the case for
        # a surrogate, which is trained locally and has no published digest.
        weights_sha256=(
            getattr(adapter, 'weights_sha256', None) or manifest_.weights.sha256
        ),
        license=manifest_.license,
        determinism=manifest_.determinism,
        metadata=metadata,
        sample_metadata=dict(sample_metadata or {}),
    )


def _normalise_input(array: Any) -> Any:
    """Make a duck-typed payload visible to the contract.

    ``check_compatible`` is deliberately numpy-free: it reads ``shape``,
    ``dtype``, ``min`` and ``max`` straight off whatever object it is handed.
    A plain Python list exposes none of those, so it satisfies *every* check --
    shape as well as dtype and value range -- while the identical numbers as an
    ndarray are rejected.  The adapter's own ``np.asarray`` then converts the
    list silently and the caller gets a plausible but wrong embedding, which is
    the exact failure mode this layer exists to prevent.

    Normalising here, at the boundary that :func:`embed` and :func:`embed_many`
    share, restores parity without giving the contract a numpy dependency.
    Objects that already advertise a ``shape`` (ndarray, tensor, HDF5 dataset,
    pandas frame, test double) are returned untouched, so duck-typed payloads
    keep working.

    One more thing happens here, and it is not cosmetic: an integer payload is
    narrowed to the **narrowest type that can hold its values**.  A plain
    Python list has no dtype, so ``np.asarray`` gives it ``int64`` -- and with
    that, a list of 8-bit values looks like a legitimate 64-bit array to every
    dtype guard, while the *numbers* are 257x too dark for a method that
    divides by 2**16-1.  Measured: ``embed('unidino', uint8_array.tolist(),
    assume='raw_crop')`` was accepted and embedded a nearly black image,
    producing a different embedding from the same numbers given as an ndarray.
    Narrowing keeps the guards honest for list payloads, and it is also the
    more faithful reading of what the caller passed.
    """
    if array is None or hasattr(array, 'shape'):
        return array
    import numpy as np

    try:
        arr = np.asarray(array)
    except (TypeError, ValueError) as exc:
        raise IncompatibleInput(
            'payload is not array-like and cannot be read as an array: '
            f'{exc}',
            reason='not-array-like',
        ) from exc

    if arr.dtype.kind in 'iu' and arr.size:
        # Only integers are narrowed: a float payload's width carries
        # information (float64 vs float32), and silently demoting it would
        # throw away precision the caller asked for.
        #
        # `np.min_scalar_type` is deliberately conservative about signedness:
        # for the values 0..255 it answers int16, because int8 tops out at 127.
        # That would keep the payload signed, and the guards that look for an
        # 8-bit array would not recognise it.  So when every value is
        # non-negative the *unsigned* width is used, which is both narrower
        # and the truthful reading of the values.
        low = int(arr.min())
        high = int(arr.max())
        if low >= 0:
            narrow = np.dtype(
                'uint8' if high <= 0xFF else
                'uint16' if high <= 0xFFFF else
                'uint32' if high <= 0xFFFFFFFF else
                'uint64'
            )
        else:
            narrow = np.min_scalar_type(arr)
        if narrow.itemsize < arr.dtype.itemsize:
            arr = arr.astype(narrow, copy=False)
    return arr


def embed(
    name: str,
    array: Any,
    *,
    unit: InputUnit | str | None = None,
    channels: Sequence[str] | None = None,
    assume: str | None = None,
    masks: Any = None,
    runtime_root: str | None = None,
    sample_metadata: dict | None = None,
    device: str | None = None,
    weights_roots: Any = None,
    adapter_kwargs: dict | None = None,
    skip_contract_check: bool = False,
    **kwargs: Any,
) -> EmbeddingResult:
    """Produce an embedding with the named method.

    Args:
        name: a key of :data:`~cellpaint_pipeline.models.registry.ADAPTERS`.
        array: the payload.  Pixels for most methods; per-channel embeddings
            for CellCLIP.
        unit: the kind of input; validated against the manifest when supplied.
        channels: channel names in the order they appear in ``array``.
        assume: ``'raw_crop'`` to run :func:`models.prepare` first, so the caller
            can hand in raw 16-bit pixels instead of an array that already
            satisfies the manifest.  ``None`` (the default) keeps the previous
            behaviour unchanged.
        masks: binary nucleus mask(s) for methods whose input is a silhouette
            (O2-VAE) or whose encoding carries a mask channel (tape).
        runtime_root: where the method will run, for the ASCII-path check.
        sample_metadata: per-row annotations carried onto the result (cell ids,
            well, plate ...).  Must have one entry per row of the output.
        device: ``'cpu'``/``'cuda'`` for torch-backed methods.  Reaches the
            adapter constructor instead of being swallowed by ``**kwargs``.
        weights_roots: explicit checkpoint roots, overriding the environment and
            the defaults.  Same purpose: reach the constructor.
        adapter_kwargs: any other constructor argument, e.g.
            ``{'feature_layer': 'pool5'}`` for DeepProfiler or
            ``{'return_channelwise': True}`` for CA-MAE.  Kept as an explicit
            dict because those settings belong to a *loaded model*, not to a
            single forward pass, and mixing them into ``**kwargs`` made them
            silently ignorable.
        skip_contract_check: bypass validation.  Only for callers that have
            already validated and are embedding in a tight loop.
        **kwargs: forwarded to the adapter's forward pass (e.g. ``batch_size``).

    Returns:
        :class:`EmbeddingResult` carrying the array plus provenance.

    Raises:
        UnknownModel: ``name`` is not registered.
        IncompatibleInput: the input violates the manifest contract.
        MissingDependency: a required framework is not installed.
        WeightsUnavailable: the weights are absent and will not be downloaded.
    """
    manifest_ = registry.manifest(name)

    # Normalise FIRST, so a payload without a dtype (a plain list) becomes an
    # array *before* `prepare` looks at it.  Otherwise the list's values reach
    # the dtype guards as int64 -- every guard sees a legitimate 64-bit array,
    # and 8-bit numbers slip through 257x too dark.  Measured before the fix:
    # `embed('unidino', uint8_array.tolist(), assume='raw_crop')` was accepted
    # and embedded a nearly black image, silently differing from the same
    # numbers given as an ndarray (which was correctly rejected).
    array = _normalise_input(array)

    if assume is not None:
        if assume != 'raw_crop':
            raise ValueError(
                f"assume={assume!r} is not understood; the only supported value "
                "is 'raw_crop', which runs models.prepare() on raw pixel values"
            )
        from cellpaint_pipeline.models.preprocessing import prepare as _prepare

        array = _prepare(manifest_, array, channels=channels, masks=masks)
        # The channels have been permuted into the manifest's order already, so
        # re-declaring them would reorder twice.
        channels = None

    array = _promote_single_sample(array, manifest_)

    if not skip_contract_check:
        report: CompatibilityReport = check_compatible(
            manifest_,
            unit=unit,
            channels=tuple(channels) if channels is not None else None,
            array=array,
            runtime_root=runtime_root,
        )
        report.raise_for_status()
    else:
        report = CompatibilityReport(manifest_name=name)

    adapter = _cached_adapter(
        name, device=device, weights_roots=weights_roots,
        adapter_kwargs=adapter_kwargs,
    )
    values = adapter.embed(array, **kwargs)

    return _wrap_result(
        name=name,
        manifest_=manifest_,
        adapter=adapter,
        values=values,
        report=report,
        sample_metadata=sample_metadata,
    )


def embed_many(
    name: str,
    arrays: Sequence[Any],
    *,
    unit: InputUnit | str | None = None,
    channels: Sequence[str] | None = None,
    assume: str | None = None,
    masks: Any = None,
    sample_metadata: dict | None = None,
    runtime_root: str | None = None,
    device: str | None = None,
    weights_roots: Any = None,
    adapter_kwargs: dict | None = None,
    **kwargs: Any,
) -> EmbeddingResult:
    """Embed several payloads, reusing one loaded model.

    Adapters cache their loaded model, so this avoids re-loading weights per
    chunk.  Kept as a separate function rather than a flag on :func:`embed` so
    the common single-payload path stays a single call.
    """
    manifest_ = registry.manifest(name)
    if not arrays:
        raise IncompatibleInput(
            f'embed_many({name!r}) received no arrays', reason='empty-input'
        )
    if assume is not None:
        if assume != 'raw_crop':
            raise ValueError(
                f"assume={assume!r} is not understood; the only supported value "
                "is 'raw_crop', which runs models.prepare() on raw pixel values"
            )
        from cellpaint_pipeline.models.preprocessing import prepare as _prepare

        arrays = [
            _prepare(manifest_, _normalise_input(array), channels=channels, masks=masks)
            for array in arrays
        ]
        channels = None
    arrays = [
        _promote_single_sample(_normalise_input(array), manifest_)
        for array in arrays
    ]

    # Validate every chunk *before* building or loading anything: a bulk call
    # with one bad payload used to pay for the checkpoint load first and only
    # then reject the batch.
    report = CompatibilityReport(manifest_name=name)
    for index, array in enumerate(arrays):
        report = check_compatible(
            manifest_, unit=unit,
            channels=tuple(channels) if channels is not None else None,
            array=array,
            runtime_root=runtime_root,
        )
        if not report.ok:
            # Re-label the offending chunk so a bulk call says *which* payload
            # was wrong; the individual issue messages are preserved verbatim.
            report = CompatibilityReport(
                manifest_name=report.manifest_name,
                issues=tuple(
                    CompatibilityIssue(issue.code, f'chunk[{index}]: {issue.message}')
                    for issue in report.issues
                ),
            )
            report.raise_for_status()

    adapter = _cached_adapter(
        name, device=device, weights_roots=weights_roots,
        adapter_kwargs=adapter_kwargs,
    )

    if hasattr(adapter, 'embed_many'):
        values = adapter.embed_many(arrays, **kwargs)
    else:
        values = _concat(adapter.embed(array, **kwargs) for array in arrays)

    return _wrap_result(
        name=name,
        manifest_=manifest_,
        adapter=adapter,
        values=values,
        report=report,
        sample_metadata=sample_metadata,
        extra_metadata={'chunks': len(arrays)},
    )


def _concat(parts: Any) -> Any:
    """Concatenate chunk results along the sample axis, without importing numpy."""
    import numpy as np

    return np.concatenate([np.asarray(p) for p in parts], axis=0)


def _split_into_blocks(payload: Any, chunk_size: int | None) -> list:
    """Split a batch payload into blocks of at most ``chunk_size`` rows.

    Returns a one-element list when no chunking is asked for, so callers can use
    a single code path.  A payload whose first dimension is unknown or not a
    batch axis is returned unsplit rather than guessed at.
    """
    if not chunk_size or chunk_size <= 0:
        return [payload]

    shape = getattr(payload, 'shape', None)
    if shape is None and hasattr(payload, '__len__'):
        try:
            shape = (len(payload),)
        except TypeError:
            return [payload]
    if not shape:
        return [payload]

    total = int(shape[0])
    if total <= chunk_size:
        return [payload]

    # Slicing keeps the batch axis first for every array-like we accept
    # (numpy arrays, torch tensors, and nested sequences all slice the same way).
    return [
        payload[start:min(start + chunk_size, total)]
        for start in range(0, total, chunk_size)
    ]


def _merge_result_parts(
    name: str,
    parts: Sequence[EmbeddingResult],
    *,
    sample_metadata: dict | None = None,
) -> EmbeddingResult:
    """Concatenate already-wrapped results from several blocks of one method.

    Used only for the stochastic + chunked path, where each block has gone
    through its own repeat-and-average cycle.  Deterministic chunked runs go
    through :func:`embed_many` instead, which does this internally.
    """
    import numpy as np

    if not parts:
        raise EmbeddingError(f'{name}: no blocks to merge')
    dims = {int(p.dim) for p in parts}
    if len(dims) != 1:
        raise EmbeddingError(
            f'{name}: blocks disagree on output width {sorted(dims)}; the method '
            'changed shape mid-run, so the concatenation would be wrong'
        )

    values = np.concatenate([np.asarray(p.values) for p in parts], axis=0)
    metadata = dict(parts[0].metadata)
    metadata['chunks'] = len(parts)

    return replace(
        parts[0],
        values=values,
        n_samples=int(values.shape[0]),
        metadata=metadata,
        sample_metadata=sample_metadata if sample_metadata is not None else {},
    )


@dataclass(frozen=True)
class BatchResult:
    """Outcome of :func:`embed_all`: the results plus per-method failures.

    Why a type instead of a dict with an ``'__errors__'`` key: that key was a
    string, so it type-checked as an ``EmbeddingResult`` and would break any
    caller that iterated the mapping expecting embeddings.  Keeping failures in a
    separate field means ``results`` is always exactly ``{method: result}``, and
    a lookup for a method that failed raises ``KeyError`` the way a dict
    should -- rather than returning a message string.

    The mapping interface is kept so existing ``result[name]`` call sites and
    ``len(results)``/iteration keep working.
    """

    results: dict[str, EmbeddingResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    # -- mapping-ish convenience over `results` --------------------------------
    def __getitem__(self, name: str) -> EmbeddingResult:
        return self.results[name]

    def __contains__(self, name: object) -> bool:
        return name in self.results

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def keys(self):
        return self.results.keys()

    def values(self):
        return self.results.values()

    def items(self):
        return self.results.items()

    def get(self, name: str, default: Any = None) -> Any:
        return self.results.get(name, default)

    def methods(self) -> tuple[str, ...]:
        """Names that produced an embedding, in run order."""
        return tuple(self.results)

    def failed(self) -> tuple[str, ...]:
        """Names that failed, in run order."""
        return tuple(self.errors)

    def to_dict(self) -> dict:
        return {
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'errors': dict(self.errors),
        }


def embed_all(
    names: Sequence[str],
    inputs: Mapping[str, Any] | Any,
    *,
    unit: InputUnit | str | Mapping[str, InputUnit | str] | None = None,
    channels: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    assume: str | Mapping[str, str] | None = None,
    masks: Any = None,
    sample_metadata: dict | None = None,
    runtime_root: str | None = None,
    on_error: str = 'raise',
    device: str | None = None,
    weights_roots: Any = None,
    adapter_kwargs: dict | None = None,
    chunk_size: int | None = None,
    **kwargs: Any,
) -> BatchResult:
    """Run several methods over one batch and return results keyed by method.

    This is the dispatch entry point.  It exists because a benchmark needs the
    same call shape for every method while the methods themselves do *not* share
    one input type: CellCLIP consumes per-channel embeddings, everything else
    consumes pixel crops.  ``inputs`` therefore maps method name to payload.

    A single payload may be passed instead, in which case it is used for every
    method whose declared ``input_unit`` matches ``unit``.  That shorthand is a
    convenience for the common case and is deliberately strict: if a method's
    unit does not match, it is skipped with a stated reason rather than being
    handed an array that would produce a confidently wrong embedding.

    Stochastic methods are repeated and aggregated, because a single draw from a
    stochastic method is not comparable with a deterministic one.

    Args:
        names: methods to run, in order.
        inputs: ``{method: array}``, or one array used for all matching methods.
        unit: declared input kind, validated per method.  May also be a
            ``{method: unit}`` mapping, which is required to run methods with
            different input kinds in one call -- CellCLIP consumes channel
            embeddings while the rest consume pixel crops, so a single scalar
            unit cannot describe the batch.
        channels: channel names in ``array`` order.  Like ``unit``, this may be
            a ``{method: names}`` mapping: the surrogates read three channels
            where the official four read five, so one shared list cannot
            describe a seven-method batch.
        assume: how to read the payload, per method.  ``'raw_crop'`` runs the
            declared pipeline, ``None`` takes the payload as already prepared.
            This *must* be per-method capable: the same call mixes methods whose
            input is a raw acquisition crop (``assume='raw_crop'``) with methods
            whose input is a precomputed vector (``assume=None``).
        masks: nucleus masks for the shape-based method.  Accepts one array or a
            ``{method: array}`` mapping; methods that do not consume masks ignore
            it.  Without this, a caller had to call :func:`prepare` itself before
            dispatching, which is exactly the per-method branching that
            ``embed_all`` exists to remove.
        sample_metadata: per-row annotations, carried onto every result.
        runtime_root: filesystem root, for the ASCII-path check.
        on_error: ``'raise'`` (default) or ``'collect'``.  Collect mode is for
            benchmark sweeps, where one unavailable method should not discard
            the results of the others that did run.
        chunk_size: maximum rows per forward pass.  ``None`` passes the payload
            through in one call.  Without this, a realistic dataset cannot be
            benchmarked at all: 3,857 CA-MAE crops (1.26 GB uint8) become a
            5 GB float32 tensor in one go and exhaust an 8 GiB card.  Chunking
            is an interface concern, not a caller concern -- otherwise every
            caller re-implements the same loop and gets the memory bound wrong.
        **kwargs: forwarded to each adapter.

    Returns:
        :class:`BatchResult`.  ``results`` holds every method that ran;
        ``errors`` explains each one that did not.  In ``'raise'`` mode the
        first failure propagates instead of being collected.
    """
    per_method = not _is_single_payload(inputs)
    errors: dict[str, str] = {}
    results: dict[str, EmbeddingResult] = {}

    def _for(name: str, value: Any) -> Any:
        """Resolve a possibly-per-method argument for one method."""
        if isinstance(value, Mapping):
            return value.get(name)
        return value

    for name in names:
        # Resolved inside a guard, not before it: an unknown name is precisely
        # the kind of failure ``on_error='collect'`` exists to collect, and a
        # sweep must not die because one method is absent from this profile.
        try:
            manifest_ = registry.manifest(name)
        except UnknownModel:
            # An unknown *name* is a caller error, not an unavailable method: it
            # cannot be distinguished from a typo, and silently collecting it
            # would let a benchmark report success over a method it never ran.
            # ``on_error='collect'`` is for methods that exist but cannot run
            # here; ``build_benchmark_plan`` already rejects unknown names.
            raise
        # `unit` used to have its own Mapping check here, separate from the other
        # three arguments.  One resolver for all four is what keeps the
        # "per-method or shared" rule a single thing to understand.
        method_unit = _for(name, unit)
        method_channels = _for(name, channels)
        method_assume = _for(name, assume)
        method_masks = _for(name, masks)
        try:
            payload = inputs[name] if per_method else inputs
        except (KeyError, TypeError) as exc:
            if on_error == 'raise':
                raise IncompatibleInput(
                    f'embed_all received no payload for {name!r}; pass a mapping '
                    f'keyed by method name, or a single array with unit=',
                    reason='missing-payload',
                ) from exc
            errors[name] = 'no payload supplied for this method'
            continue

        if not per_method and method_unit is not None:
            supplied = (
                method_unit.value
                if isinstance(method_unit, InputUnit)
                else str(method_unit)
            )
            if supplied != manifest_.input_unit.value:
                # Skipping beats guessing: the alternative is feeding a pixel
                # array to a method that wants vectors, which does not raise but
                # does produce a meaningless embedding.
                message = (
                    f'{name!r} consumes {manifest_.input_unit.value!r}, but the '
                    f'shared payload was declared as {supplied!r}'
                )
                if on_error == 'raise':
                    raise IncompatibleInput(message, reason='input-unit')
                errors[name] = message
                continue

        try:
            blocks = _split_into_blocks(payload, chunk_size)
            if len(blocks) > 1:
                # One forward pass per block, same loaded model.  Stored as the
                # `chunks` metadata count so a reader can tell a chunked run from
                # a single-shot run when comparing timings.
                if manifest_.determinism == 'stochastic':
                    parts = [
                        _aggregate_stochastic(
                            name, block, unit=method_unit,
                            channels=method_channels,
                            assume=method_assume, masks=method_masks,
                            runtime_root=runtime_root, sample_metadata=None,
                            device=device, weights_roots=weights_roots,
                            adapter_kwargs=adapter_kwargs,
                            **kwargs
                        )
                        for block in blocks
                    ]
                    result = _merge_result_parts(
                        name, parts, sample_metadata=sample_metadata,
                    )
                else:
                    result = embed_many(
                        name, blocks,
                        unit=method_unit,
                        channels=method_channels,
                        assume=method_assume,
                        masks=method_masks,
                        runtime_root=runtime_root,
                        sample_metadata=sample_metadata,
                        device=device,
                        weights_roots=weights_roots,
                        adapter_kwargs=adapter_kwargs,
                        **kwargs,
                    )
            else:
                result = embed(
                    name,
                    payload,
                    unit=method_unit,
                    channels=method_channels,
                    assume=method_assume,
                    masks=method_masks,
                    runtime_root=runtime_root,
                    sample_metadata=sample_metadata,
                    device=device,
                    weights_roots=weights_roots,
                    adapter_kwargs=adapter_kwargs,
                    **kwargs,
                )
                if manifest_.determinism == 'stochastic':
                    result = _aggregate_stochastic(
                        name, payload, unit=method_unit,
                        channels=method_channels,
                        assume=method_assume, masks=method_masks,
                        runtime_root=runtime_root, sample_metadata=sample_metadata,
                        device=device, weights_roots=weights_roots,
                        adapter_kwargs=adapter_kwargs,
                        **kwargs
                )
            results[name] = result
        except Exception as exc:  # noqa: BLE001 - collect mode wants them all
            if on_error == 'raise':
                raise
            errors[name] = f'{type(exc).__name__}: {exc}'

    return BatchResult(results=results, errors=errors)


def _aggregate_stochastic(
    name: str, payload: Any, *, repeats: int = 3, **kwargs: Any
) -> EmbeddingResult:
    """Run a stochastic method ``repeats`` times and report spread.

    Reporting one draw as "the" embedding of a stochastic method is how a
    benchmark ends up comparing noise against signal.  The mean is kept as the
    value and the observed spread is recorded, so a downstream reader can see
    whether the method is stable enough to compare at all.
    """
    import numpy as np

    runs = [embed(name, payload, **kwargs).values for _ in range(max(2, repeats))]
    stacked = np.stack([np.asarray(r) for r in runs])
    spread = float(np.max(np.abs(stacked - stacked.mean(axis=0))))

    result = embed(name, payload, **kwargs)
    return replace(
        result,
        values=stacked.mean(axis=0),
        metadata={
            **result.metadata,
            'stochastic_repeats': len(runs),
            'stochastic_max_spread': spread,
        },
    )


def _is_single_payload(inputs: Any) -> bool:
    """Distinguish ``{method: array}`` from a bare array.

    Checked structurally rather than by ``isinstance(dict)`` because a caller
    may reasonably pass any mapping, and an ndarray is not a Mapping.
    """
    return not isinstance(inputs, Mapping)
