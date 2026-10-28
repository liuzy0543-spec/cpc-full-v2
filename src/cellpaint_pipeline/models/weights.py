"""Weight location and verification.  Local first; never downloads implicitly.

WHY THIS MODULE EXISTS
----------------------
The upstream DeepProfiler adapter falls back to a hard-coded Zenodo URL when a
checkpoint is missing.  On this machine that transfer runs at ~9 KB/s, so a
19 MB checkpoint takes ~40 minutes and a 1.4 GB one takes ~43 hours -- and the
failure is not immediate.  An ``embed()`` call must never do that.

So this module makes weight acquisition **explicit and inspectable**:

* :func:`resolve_weights` probes an ordered list of roots and returns the first
  hit, or raises :class:`~cellpaint_pipeline.models.errors.WeightsUnavailable`
  listing every path it tried.
* :func:`verify_sha256` lets a caller assert the file is the artefact the
  manifest describes.
* :func:`ascii_safe_root` guards the CellProfiler-backed methods, which fail
  with an opaque ``OSError`` on non-ASCII paths.

The module is pure: no numpy, no framework imports, no network.  That keeps
``models.weights`` usable from environment-setup scripts and tests without
pulling in TensorFlow or torch.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from cellpaint_pipeline.models.errors import WeightsUnavailable

__all__ = [
    'DEFAULT_WEIGHTS_ROOTS',
    'WEIGHTS_ROOT_ENV',
    'ResolvedWeights',
    'ascii_safe_root',
    'resolve_weights',
    'verify_sha256',
]

#: Environment variable that overrides the search path entirely.
WEIGHTS_ROOT_ENV = 'CPC_WEIGHTS_ROOT'

#: Roots probed, in order, when the caller passes no ``roots``.
#:
#: ``E:\\cpc_weights`` is the project's canonical store (pure ASCII: it holds
#: multi-GB model files and some CellProfiler-adjacent steps are path-fragile).
#: The remaining entries keep older layouts working so an existing checkout does
#: not have to be re-arranged to be usable.
DEFAULT_WEIGHTS_ROOTS: tuple[Path, ...] = (
    Path(r'E:\cpc_weights'),
    Path(r'D:\cpc_weights'),
    Path(r'D:\模型权重'),
    Path.home() / '.cellpainting-weights',
)


@dataclass(frozen=True)
class ResolvedWeights:
    """A weight file that exists on disk, plus how it was found."""

    path: Path
    model: str
    probed: tuple[str, ...]
    sha256: str | None = None

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def to_dict(self) -> dict:
        return {
            'model': self.model,
            'path': str(self.path),
            'exists': self.exists,
            'size_bytes': self.size_bytes,
            'sha256': self.sha256,
        }


def _candidate_names(filename: str, model: str) -> tuple[str, ...]:
    """Names to try for a model, most specific first.

    A method may publish more than one artefact name across versions, so the
    manifest can pass a filename and we also try a ``<model>/`` subdirectory
    layout.
    """
    names = [filename]
    # `deep/cellclip/model.safetensors` and `cellclip/model.safetensors` should
    # both resolve from the same root without the manifest spelling out both.
    if '/' not in filename and os.sep not in filename:
        names.append(f'{model}/{filename}')
    return tuple(names)


def resolve_weights(
    model: str,
    filename: str,
    *,
    roots: Sequence[Path | str] | None = None,
    expected_sha256: str | None = None,
    verify: bool = False,
    hint: str | None = None,
) -> ResolvedWeights:
    """Locate ``filename`` for ``model`` without ever downloading.

    Search order:

    1. ``roots`` when given (explicit always wins);
    2. else ``$CPC_WEIGHTS_ROOT`` when set;
    3. else :data:`DEFAULT_WEIGHTS_ROOTS`.

    Args:
        model: registry name, used for subdirectory layout and messages.
        filename: the artefact's file name (or a relative path).
        roots: explicit roots to probe, in order.
        expected_sha256: recorded in the result even when ``verify`` is False.
        verify: when True, hash the file and refuse a mismatch.
        hint: extra guidance appended to the error.

    Raises:
        WeightsUnavailable: no root contained the file, or verification failed.
    """
    if roots is not None:
        search_roots = [Path(r) for r in roots]
    elif (env_root := os.environ.get(WEIGHTS_ROOT_ENV, '').strip()):
        search_roots = [Path(env_root)]
    else:
        search_roots = list(DEFAULT_WEIGHTS_ROOTS)

    probed: list[str] = []
    for root in search_roots:
        for candidate_name in _candidate_names(filename, model):
            candidate = root / candidate_name
            probed.append(str(candidate))
            if candidate.is_file():
                resolved = ResolvedWeights(
                    path=candidate,
                    model=model,
                    probed=tuple(probed),
                    sha256=expected_sha256,
                )
                if verify and expected_sha256:
                    actual = verify_sha256(candidate)
                    if actual != expected_sha256.lower():
                        raise WeightsUnavailable(
                            model,
                            probed=tuple(probed),
                            hint=(
                                f'sha256 mismatch for {candidate}: expected '
                                f'{expected_sha256}, got {actual}'
                            ),
                        )
                return resolved

    raise WeightsUnavailable(
        model,
        probed=tuple(probed),
        hint=hint
        or (
            f'Place the file under one of {[str(r) for r in search_roots]} '
            f'or set {WEIGHTS_ROOT_ENV}.'
        ),
    )


#: Memo of digests keyed by (path, size, mtime_ns).
#:
#: Hashing is the only expensive step in resolving a checkpoint -- CellCLIP's is
#: 1.8 GB, roughly 10 s from cold storage -- and a benchmark embeds many batches
#: against the same file.  Keying on size and mtime means the cache is
#: invalidated automatically when a checkpoint is replaced, so this shortens the
#: common path without ever reporting a digest for different bytes.
_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def verify_sha256(
    path: Path | str, *, chunk_bytes: int = 1 << 20, use_cache: bool = True
) -> str:
    """Return the lowercase hex sha256 of ``path``, streamed.

    Results are memoised on (path, size, mtime) so repeated embeddings against
    one checkpoint hash it once.  Pass ``use_cache=False`` to force a re-read,
    which a verification tool should do when auditing a file it did not write.
    """
    file_path = Path(path)
    if use_cache:
        stat = file_path.stat()
        key = (str(file_path.resolve()), stat.st_size, stat.st_mtime_ns)
        cached = _DIGEST_CACHE.get(key)
        if cached is not None:
            return cached

    digest = hashlib.sha256()
    with open(file_path, 'rb') as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    result = digest.hexdigest()

    if use_cache:
        _DIGEST_CACHE[key] = result
    return result


def ascii_safe_root(root: Path | str, *, model: str) -> Path:
    """Return ``root`` as a :class:`Path`, refusing non-ASCII locations.

    CellProfiler's frozen Windows build fails on non-ASCII paths with
    ``OSError: Test for access to directory failed``, which is reported tens of
    seconds later and names the wrong component.  Methods that shell out to it
    call this first so the rejection is immediate and correct.

    Raises:
        WeightsUnavailable: never -- guarded by the caller.  Here we raise
            :class:`ValueError` so the contract check (which produces the
            user-facing error) stays in :mod:`manifest`.
    """
    resolved = Path(root)
    text = str(resolved)
    if not text.isascii():
        raise ValueError(
            f"Embedding model {model!r} requires an ASCII runtime root; "
            f"got {text!r}"
        )
    return resolved


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return tuple(seen)
