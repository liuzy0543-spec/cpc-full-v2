"""The executable half of the preprocessing contract.

:class:`~cellpaint_pipeline.models.contract.PreprocessSpec` *declares* what each
model wants; until now the code that actually produced it lived in dataset
specific scripts (``scripts/embedding_benchmark/build_bbbc022_cells.py``), so
"ask for a method name and hand it pixels" was impossible from the framework.

This module closes that gap for the transforms the manifest can express, and
refuses -- with a typed error naming what is missing -- for the ones it cannot.
Refusing is deliberate: guessing at a segmentation threshold or a MATLAB
centring step would produce exactly the confident-but-wrong embedding this
layer exists to prevent.

Design notes:

* numpy and skimage are imported lazily.  ``import cellpaint_pipeline.models``
  stays cheap and framework-free, and ``contract.py`` in particular must never
  acquire a numpy dependency -- that is why the transforms live here.
* Every step is per-sample and deterministic.  A batch of one and a batch of
  many therefore agree, which is what lets a benchmark chunk its input freely.
* The order of operations is fixed and documented by :func:`plan`, because
  resizing before or after rescaling is a real, observable difference.

Coverage of the declared transforms (one test per row in
``tests/test_models_preprocessing.py``):

=====================  ===============  ==================================================
source_map              methods          status
=====================  ===============  ==================================================
per_crop_minmax_bhwc    deepprofiler     implemented; bitwise-identical to upstream
                                         ``cropping.py:65-67``
identity                ca_mae           implemented
per_crop_max255         (unused)         implemented
div65535                unidino *        implemented
div257                  --               implemented (16-bit to 8-bit rescale)
per_sample_selfnorm     dinocells *      implemented
mask_to_silhouette      o2vae            implemented; resizes the mask to the declared
                                         spatial and keeps it binary
matlab_centring         morphogenie *    **refused** (``unsupported-preprocess``)
=====================  ===============  ==================================================

``*`` cancelled on 2026-10-21: no longer registered, because their weights cannot
be obtained here.  The contracts survive in ``registry.PENDING``, so the
transforms stay implemented and tested; ``prepare``/``plan`` still accept those
manifests when passed directly instead of by name.

``morphogenie`` is refused rather than approximated on purpose.  Upstream centres
each crop in MATLAB, using a step this repository does not run: the MATLAB
*source* is in the official repository
(``ImagePreProcessing/MultipleCells/CPA/GetCentImages.m`` -- a 200x200 crop,
min-max stretched to uint8, at :26 and :38), and the 256x256 resize happens on
the Python side (``gan_training/utils.py:352-357``), but what is missing here is
the MATLAB runtime, Cellpose and the raw data, so the step cannot be executed
end to end.  Any reproduction here would be a guess, and a guess changes the
embedding without raising.  The manifest still describes the method so its
contract stays visible, and :func:`plan` reports ``unsupported-preprocess`` so a
caller finds out before loading anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from cellpaint_pipeline.models.contract import InputUnit, ModelManifest
from cellpaint_pipeline.models.errors import IncompatibleInput

__all__ = ['PrepareStep', 'plan', 'prepare']

#: Guards divisions by a constant crop (all-zero input is legal and must not
#: become NaN/inf: a black image is a real, if degenerate, sample).
_EPS = 1e-12


def _numpy():
    import numpy as np

    return np


def _manifest_of(source: Any) -> ModelManifest:
    """Accept a manifest or a registered method name."""
    if isinstance(source, ModelManifest):
        return source
    if isinstance(source, str):
        from cellpaint_pipeline.models import registry

        return registry.manifest(source)
    raise TypeError(
        f'prepare() expects a method name or a ModelManifest, got {type(source)!r}'
    )


@dataclass(frozen=True)
class PrepareStep:
    """One transformation :func:`prepare` applies, in order.

    Returned by :func:`plan` so a caller can print what will happen to an array
    instead of reading the source.  ``name`` is stable enough to assert on;
    ``detail`` is prose.
    """

    name: str
    detail: str


def plan(source: Any) -> tuple[PrepareStep, ...]:
    """Describe the pipeline for one method, without touching any data."""
    manifest = _manifest_of(source)
    spec = manifest.preprocess
    steps: list[PrepareStep] = []

    if manifest.input_unit is InputUnit.CHANNEL_EMBEDDINGS:
        return (
            PrepareStep(
                'chain-required',
                f"'{manifest.name}' consumes precomputed channel embeddings "
                f"(width {manifest.input_feature_dim}), not pixels; a pixel crop "
                'cannot be converted here. Produce the features first '
                f"(chain_weights={manifest.chain_weights}).",
            ),
        )
    if spec.source_map is None:
        return (
            PrepareStep(
                'identity',
                f"'{manifest.name}' declares no source_map: the caller must "
                f'already supply value_range={spec.value_range!r} '
                f'to_dtype={spec.to_dtype!r}.',
            ),
        )
    if spec.source_map == 'matlab_centring':
        return (
            PrepareStep(
                'unsupported-preprocess',
                f"'{manifest.name}' is centred in MATLAB upstream (the source "
                'is in the official repository, but the MATLAB runtime, Cellpose '
                'and the raw data are not available here); prepare() will not '
                'guess it.',
            ),
        )

    steps.append(
        PrepareStep('coerce', 'treat input as (N, H, W, C) in the named channel order')
    )
    if manifest.channels.order_matters:
        steps.append(
            PrepareStep(
                'reorder-channels',
                'permute to ' + ', '.join(manifest.channels.names),
            )
        )
    if manifest.spatial:
        if manifest.preprocess.resize_policy == 'refuse':
            steps.append(
                PrepareStep(
                    'check-size',
                    f'require {manifest.spatial[0]}x{manifest.spatial[1]}; '
                    f'resize is refused because the upstream operation is a '
                    f'crop, not a resize',
                )
            )
        else:
            steps.append(
                PrepareStep(
                    'resize',
                    f'bilinear resize to {manifest.spatial[0]}x{manifest.spatial[1]}',
                )
            )
    steps.append(PrepareStep('source-map', spec.source_map))
    if spec.input_encoding == 'tape':
        steps.append(
            PrepareStep(
                'unsupported-encoding',
                'the tape layout is not implemented: it was the *storage* '
                'format of the official DINO4Cells dataset, not its model '
                'input -- see the mask_mode field',
            )
        )
    steps.append(PrepareStep('layout', f"transpose to {spec.batch_layout}"))
    steps.append(PrepareStep('cast', f"cast to {spec.to_dtype}"))
    return tuple(steps)


# ----------------------------------------------------------------- transforms --


def _as_bhwc(array: Any) -> tuple[Any, bool]:
    """Coerce to ``(N, H, W, C)``; report whether a batch axis was synthesised."""
    np = _numpy()
    try:
        arr = np.asarray(array)
    except (TypeError, ValueError) as exc:
        raise IncompatibleInput(
            f'cannot read the payload as an array: {exc}',
            reason='not-array-like',
        ) from exc
    if arr.ndim == 3:
        return arr[None, ...], True
    if arr.ndim == 4:
        return arr, False
    raise IncompatibleInput(
        f'prepare() expects a single crop (H, W, C) or a batch (N, H, W, C), '
        f'got shape {arr.shape}',
        reason='not-bhwc',
    )


def _reorder(arr: Any, source: Sequence[str], target: Sequence[str]) -> Any:
    """Permute the channel axis so it matches ``target``."""
    missing = [name for name in target if name not in source]
    if missing:
        raise IncompatibleInput(
            f'the payload is missing channel(s) {missing}; it declares '
            f'{list(source)} but the method needs {list(target)}',
            reason='channel-count',
        )
    index = [list(source).index(name) for name in target]
    if index == list(range(len(index))):
        return arr
    return arr[..., index]


def _resize(arr: Any, spatial: tuple[int, int]) -> Any:
    """Bilinear resize to ``spatial``, one channel at a time.

    Per-channel rather than ``channel_axis=-1`` because skimage's signature for
    that changed across versions and this project supports 0.19-0.25.
    """
    if tuple(arr.shape[1:3]) == tuple(spatial):
        return arr
    try:
        from skimage.transform import resize
    except ImportError as exc:  # pragma: no cover - depends on environment
        from cellpaint_pipeline.models.errors import MissingDependency

        raise MissingDependency('skimage', model='prepare()') from exc

    np = _numpy()
    out = np.empty(arr.shape[:1] + tuple(spatial) + arr.shape[3:], dtype=np.float64)
    for n in range(arr.shape[0]):
        for c in range(arr.shape[3]):
            out[n, :, :, c] = resize(
                arr[n, :, :, c],
                tuple(spatial),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            )
    return out


def _check_scale_dtype(manifest: Any, arr: Any) -> None:
    """Refuse an input whose dtype cannot carry the scale's domain.

    ``div65535`` divides by ``2**16 - 1``, so the input has to be able to hold
    65535.  Handed a ``uint8`` crop it produces a picture that is essentially
    black -- measured, the prepared maximum is ``0.0039`` instead of ``0.98``
    -- and the model still returns a well-formed 1920-vector.  Nothing
    downstream can tell that the embedding is wrong.

    The check is on the **dtype**, not the values: a uint8 array is numerically
    inside ``(0, 65535)`` while carrying values 257x too dark, so a value-range
    test would wave it through.
    """
    if manifest.preprocess.source_map != 'div65535':
        return
    dtype = getattr(arr, 'dtype', None)
    kind = getattr(dtype, 'kind', None)

    if kind == 'u':
        if int(getattr(dtype, 'itemsize', 8)) >= 2:
            return
        raise IncompatibleInput(
            f"the scale 'div65535' needs a 16-bit input: an {dtype} array "
            f"holds values 257x smaller than the 16-bit counts this method "
            f"was trained on, so dividing it by 65535 would produce a black "
            f"image and a silently wrong embedding. Pass uint16 crops, or "
            f"float values already spanning 0..65535",
            reason='value-range',
        )

    if kind == 'f':
        # A float array is ambiguous: it may hold raw 16-bit-domain counts, or
        # it may already have been divided by 65535 -- which is exactly what
        # `prepare` is about to do again.  The two are told apart by the
        # scale's own output domain: `div65535` maps [0, 65535] onto [0, 1],
        # so an input whose maximum is inside [0, 1] is indistinguishable from
        # an already-scaled one.  Dividing it again would produce values of
        # order 1e-5 -- a black image with a perfectly well-formed embedding.
        #
        # Refusing is the honest answer: the alternative is to guess, and a
        # guess here is invisible.  (A raw crop whose maximum is at most 1.0
        # is degenerate anyway -- it is black to begin with.)
        top = getattr(arr, 'max', None)
        try:
            maximum = float(top()) if callable(top) else None
        except (TypeError, ValueError):
            maximum = None
        if maximum is not None and maximum <= 1.0:
            raise IncompatibleInput(
                f"the scale 'div65535' would divide this array a second time: "
                f"its maximum is {maximum:g}, which is already inside the "
                f"output domain [0, 1] of the scale. Dividing again yields "
                f"values of order 1e-5 -- a black image with a well-formed but "
                f"meaningless embedding. Pass the raw uint16 counts, or hand "
                f"the already-scaled array straight to embed() without "
                f"assume='raw_crop'",
                reason='value-range',
            )


def _apply_source_map(arr: Any, source_map: str) -> Any:
    """Map raw acquisition values onto the method's declared range."""
    np = _numpy()
    if source_map == 'identity':
        # The array already carries the values the model expects.  This exists so
        # a method that needs *no* mapping can still declare one explicitly: the
        # alternative (`source_map=None`) means "the caller has prepared this by
        # hand", which `prepare()` refuses because it cannot know the array is
        # ready.
        return arr
    if source_map == 'per_crop_minmax_bhwc':
        lo = arr.min(axis=(1, 2), keepdims=True)
        hi = arr.max(axis=(1, 2), keepdims=True)
        return (arr - lo) / np.maximum(hi - lo, _EPS)
    if source_map == 'per_crop_max255':
        hi = arr.max(axis=(1, 2, 3), keepdims=True)
        return np.clip(arr / np.maximum(hi, _EPS) * 255.0, 0.0, 255.0)
    if source_map == 'div65535':
        # The divisor encodes the source depth, so the input must be able to
        # carry it.  An 8-bit array divides into an essentially black image
        # (measured: prepared max 0.0039 instead of 0.98) and the model still
        # returns a well-formed embedding -- a silent failure with no way to
        # notice it downstream.  Refuse it here, where the scale is applied,
        # and let the contract refuse it on the direct path.
        if getattr(arr.dtype, 'kind', None) == 'u' and arr.dtype.itemsize < 2:
            raise IncompatibleInput(
                f"the scale 'div65535' needs a 16-bit input: an "
                f"{arr.dtype} array holds values 257x smaller than the 16-bit "
                f"counts this method was trained on, so dividing it by 65535 "
                f"would produce a black image and a silently wrong embedding. "
                f"Pass uint16 crops, or float values already spanning "
                f"0..65535",
                reason='value-range',
            )
        return arr / 65535.0
    if source_map == 'div257':
        return np.clip(arr / 257.0, 0.0, 255.0)
    if source_map == 'per_sample_selfnorm':
        mean = arr.mean(axis=(1, 2), keepdims=True)
        std = arr.std(axis=(1, 2), keepdims=True)
        return (arr - mean) / np.maximum(std, _EPS)
    raise IncompatibleInput(
        f'unknown source_map {source_map!r}', reason='unknown-source-map'
    )


def _cast(arr: Any, to_dtype: str) -> Any:
    np = _numpy()
    target = np.dtype(to_dtype)
    if target.kind in ('u', 'i'):
        info = np.iinfo(target)
        arr = np.clip(arr, info.min, info.max)
    return arr.astype(target, copy=False)


def _apply_mask_mode(arr: Any, masks: Any, mask_mode: str) -> Any:
    """Apply the declared mask policy to a stacked crop.

    Mirrors the ``mode`` argument of the official DINO4Cells ``fold_channels``
    (``data_utils/cellpainting_dataset.py:23-47``), which the official Cell
    Painting dataset calls with its default ``mode="drop"``:

    ``'drop'``  -- the mask is discarded.  This is what the official dataset
                   does, so this is the only setting that reproduces the
                   published model.  A mask may still be supplied (the
                   upstream pipeline needs one to build the tape on disk) and
                   it is simply not part of the model's input.
    ``'apply'`` -- the mask zeroes everything outside the cell, which is the
                   official ``mode="apply"``.
    ``'none'``  -- no mask is part of the input at all.

    Note what this is *not*: the official code folds the **stored** tape back
    into a stack and drops the mask column.  Feeding the mask to the network
    instead would give the model one more channel than the published
    checkpoint was trained on.
    """
    np = _numpy()
    if mask_mode == 'none':
        return arr
    if masks is None:
        if mask_mode == 'apply':
            raise IncompatibleInput(
                "the mask policy is 'apply', so a mask is required; pass "
                "masks=(H, W) or (N, H, W)",
                reason='mask-required',
            )
        # 'drop': nothing to do.  The mask is not part of the model's input.
        return arr

    mask = np.asarray(masks)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        raise IncompatibleInput(
            f'mask must be (H, W) or (1, H, W); got shape {mask.shape}',
            reason='mask-shape',
        )
    if mask_mode == 'drop':
        # The mask is accepted and discarded, exactly as the official
        # `mode="drop"` discards the tape's mask column.
        return arr

    target = tuple(arr.shape[1:3])
    if mask.shape != target:
        # The plan resizes the crop to the method's declared spatial size, so
        # the mask has to follow it.
        resized = _resize(
            (mask > 0).astype(np.float32)[None, ..., None], target
        )[0, ..., 0]
        mask = (resized > 0.5).astype(np.float32)
    return arr * mask[..., None]


# --------------------------------------------------------------------- public --


def prepare(
    source: Any,
    array: Any,
    *,
    channels: Sequence[str] | None = None,
    masks: Any = None,
) -> Any:
    """Turn a raw acquisition array into the array the named method expects.

    Args:
        source: method name or the :class:`ModelManifest` itself.
        array: a single crop ``(H, W, C)`` or a batch ``(N, H, W, C)``, holding
            raw pixel values (16-bit counts by default) in ``channels`` order.
        channels: channel names *in the order they appear in* ``array``.  When
            omitted the array is assumed to already be in the manifest's order.
        masks: binary nucleus mask(s), required by methods whose input is a
            silhouette (O2-VAE) or by the tape encoding.

    Returns:
        The prepared array, with the same batch rank it was given (``3-D`` in
        gives ``3-D`` out; a tape result is always 2-D because it has no batch
        axis).

    Raises:
        IncompatibleInput: the method cannot consume pixels (CellCLIP's chained
            features), its preprocessing is not reproducible here (MorphoGenie's
            MATLAB centring), or a required ``masks`` argument was not supplied.

    Example:
        >>> crop = np.zeros((32, 32, 5), dtype='uint16')   # doctest: +SKIP
        >>> pixels = models.prepare('deepprofiler', crop,
        ...                         channels=('DNA', 'ER', 'RNA', 'AGP', 'Mito'))
        >>> models.embed('deepprofiler', pixels).shape
        (1, 672)
    """
    manifest = _manifest_of(source)
    spec = manifest.preprocess

    if spec.input_encoding == 'tape':
        raise IncompatibleInput(
            f"'{manifest.name}' declares input_encoding='tape', which this "
            f"framework no longer implements. The tape -- channels laid out "
            f"side by side in the width dimension with a mask column -- is the "
            f"*storage* format of the official DINO4Cells dataset, not its "
            f"model input: the official dataset folds the tape back open and "
            f"drops the mask before the network sees it "
            f"(data_utils/cellpainting_dataset.py:23-47, :86). Declare "
            f"input_encoding='stack' with the matching mask_mode instead",
            reason='unsupported-encoding',
        )
    if manifest.input_unit is InputUnit.CHANNEL_EMBEDDINGS:
        raise IncompatibleInput(
            f"'{manifest.name}' consumes precomputed channel embeddings "
            f'(width {manifest.input_feature_dim}), not pixels; produce those '
            f'features first (chain_weights={manifest.chain_weights})',
            reason='chain-required',
        )
    if spec.source_map is None:
        raise IncompatibleInput(
            f"'{manifest.name}' declares no source_map, so prepare() has nothing "
            f'to apply: supply value_range={spec.value_range!r} '
            f'to_dtype={spec.to_dtype!r} yourself',
            reason='no-source-map',
        )
    if spec.source_map == 'matlab_centring':
        raise IncompatibleInput(
            f"'{manifest.name}' centres its 256x256 crops in MATLAB upstream, "
            'which this repository does not reproduce; prepare() refuses rather '
            'than guessing',
            reason='unsupported-preprocess',
        )

    if spec.source_map == 'mask_to_silhouette':
        if masks is None:
            raise IncompatibleInput(
                f"'{manifest.name}' consumes a nucleus silhouette, not "
                'intensities; pass masks=<binary mask> (one per cell)',
                reason='mask-required',
            )
        np = _numpy()
        silhouette = (np.asarray(masks) > 0).astype(np.float32)
        if silhouette.ndim == 2:
            silhouette = silhouette[None, ..., None]
        elif silhouette.ndim == 3:
            silhouette = silhouette[..., None]
        if manifest.spatial:
            # The plan advertises a resize step, so this branch must honour it
            # too: an earlier version returned the mask at whatever size it
            # arrived, which contradicted the declared spatial and produced a
            # tensor the model cannot consume.
            silhouette = _resize(silhouette, manifest.spatial)
            # Resizing a binary mask bilinearly produces soft edges; the method
            # consumes a silhouette, so restore the binary form.
            silhouette = (silhouette > 0.5).astype(np.float32)
        was_single = silhouette.shape[0] == 1
        out = _cast(np.transpose(silhouette, (0, 3, 1, 2)), spec.to_dtype)
        return out[0] if was_single else out

    arr, was_3d = _as_bhwc(array)

    # The dtype guard has to run on the **incoming** array, before any
    # transform.  A resize would launder it: `_resize` allocates float64, so by
    # the time the scale ran, a uint8 crop would look like a float one and the
    # check below would have nothing to catch.  This is also why the check is
    # not folded into the value-range test -- a uint8 array is numerically
    # inside (0, 65535) while being 257x too dark.
    _check_scale_dtype(manifest, arr)

    if manifest.channels.order_matters and channels is not None:
        arr = _reorder(arr, channels, manifest.channels.names)
    elif channels is not None and len(channels) != arr.shape[-1]:
        raise IncompatibleInput(
            f'{len(channels)} channel names were given but the array has '
            f'{arr.shape[-1]} channels',
            reason='channel-count',
        )

    if manifest.spatial:
        if manifest.preprocess.resize_policy == 'refuse':
            # The upstream operation on a crop is a *crop* (`crop_and_resize`),
            # not a resize, so resizing here would invent a preprocessing step
            # the official pipeline does not have -- and would do it
            # differently.  Refuse and let the caller resample deliberately.
            observed = (int(arr.shape[1]), int(arr.shape[2]))
            if observed != tuple(manifest.spatial):
                raise IncompatibleInput(
                    f"'{manifest.name}' expects {manifest.spatial} but got "
                    f"{observed}; resizing is refused because the upstream "
                    f"operation on a crop is a crop, not a resize. Resample the "
                    f"input yourself if that is what the upstream configuration "
                    f"asks for",
                    reason='spatial-size',
                )
        else:
            arr = _resize(arr, manifest.spatial)

    arr = _apply_source_map(arr, spec.source_map)

    # The mask policy is applied to the *stacked* crop, before the layout
    # transpose, so 'apply' multiplies the image channels by the mask and
    # 'drop' simply discards it.  The official DINO4Cells dataset folds the
    # stored tape back open and drops the mask column before the model sees
    # it; the mask is never an input channel.
    arr = _apply_mask_mode(arr, masks, spec.mask_mode)

    if spec.batch_layout == 'BCHW':
        arr = _numpy().transpose(arr, (0, 3, 1, 2))

    arr = _cast(arr, spec.to_dtype)
    return arr[0] if was_3d else arr