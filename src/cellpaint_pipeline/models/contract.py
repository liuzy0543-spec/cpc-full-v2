"""Declarative contracts for the embedding layer.

WHY THIS MODULE EXISTS
----------------------
Several Cell Painting embedding methods have to be callable through one shape --
``models.embed('deepprofiler', x)`` -- while their real input/output contracts
disagree on almost everything:

======================  ==================  =================  ===============
method                  input unit          layout             dtype / range
======================  ==================  =================  ===============
DeepProfiler            single-cell crop    ``(B,H,W,C)``      16-bit raw
CA-MAE / OpenPhenom     crop                ``(B,C,H,W)``      uint8 0..255
CellCLIP                **not pixels**      ``(B,C,D)``        float32 channel
                                            channel embeddings  embeddings
======================  ==================  =================  ===============

Forcing one tensor shape onto all of them would either lose information or push
model-specific work into callers.  So the unification happens **here**, in a
declarative contract, not in the tensor layer:

* :class:`ModelManifest` states what a method eats and produces.
* :func:`check_compatible` is a **pure function** that decides -- before any
  weight is loaded and before any heavy framework is imported -- whether a given
  call can succeed, and why not when it cannot.

This module deliberately imports nothing heavy.  ``models.contract`` must stay
importable with no TensorFlow, no torch, and no numpy, so that capability
discovery and contract validation never drag a deep-learning stack into memory.
Consequently ``check_compatible`` inspects arrays through their *duck-typed*
attributes (``ndim``, ``shape``, ``dtype``) and never imports numpy.

The module is named ``contract`` rather than ``manifest`` because
``models.manifest`` is the *function* callers want; a submodule of that name
would shadow it, and ``models.manifest('deepprofiler')`` would raise
``TypeError: 'module' object is not callable``.

MEASURED FACTS THIS CONTRACT ENCODES
------------------------------------
1. **Input scaling is not optional metadata.**  Feeding the same DeepProfiler
   checkpoint the same shape but a different value scale changes the embedding
   by ~25,000x with no error.  The model graph contains no normalisation layer,
   so :class:`PreprocessSpec` is a first-class, *checked* part of the contract.
2. **Channel order is a silent failure mode.**  The segmentation crop order
   (``Mito, AGP, RNA, ER, DNA``) differs from the model order
   (``DNA, RNA, ER, AGP, Mito``) and nothing raises.
3. **CellCLIP consumes precomputed per-channel vectors**, not images, so
   :class:`InputUnit` has to be able to say so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

__all__ = [
    'CompatibilityIssue',
    'CompatibilityReport',
    'ChannelSpec',
    'InputUnit',
    'ModelManifest',
    'ParityMode',
    'PreprocessSpec',
    'SourceMap',
    'WeightsSpec',
    'check_compatible',
    'is_ascii_path',
]


class InputUnit(str, Enum):
    """What a method consumes.  Not every method eats pixels.

    Subclasses ``str`` as well as ``Enum`` (rather than using ``enum.StrEnum``,
    which is Python 3.11+) because this project pins Python 3.10.12.  Members
    therefore compare equal to their string values, so both ``InputUnit.FIELD``
    and ``'field'`` work at call sites.
    """

    #: One cell per sample, cut from a field by a segmentation mask.
    SINGLE_CELL_CROP = 'single_cell_crop'
    #: A whole field of view.
    FIELD = 'field'
    #: Precomputed per-channel embeddings -- CellCLIP consumes these, not images.
    CHANNEL_EMBEDDINGS = 'channel_embeddings'


#: How a method's output may be compared against the official implementation.
#: ``bitwise``  - arrays must match exactly.
#: ``normalized`` - arrays must match exactly, but surrounding metadata (paths,
#:   timestamps, filenames) is only required to match after normalisation.
ParityMode = Literal['bitwise', 'normalized']


@dataclass(frozen=True)
class ChannelSpec:
    """Which channels a method expects, and how load-bearing their identity is.

    Two independent questions, which earlier drafts conflated:

    ``names_are_labels``
        Do the names carry *semantics*?  For DeepProfiler, ``DNA`` means the DNA
        stain -- feeding Mito under that name silently corrupts the embedding, so
        the set is enforced.  For a channel-agnostic architecture the names are
        pure annotations; a caller may supply anything and only the *count* is
        checked.
    ``order_matters``
        Does the *order* of the channels change the result?  Independently
        meaningful: a model could be order-sensitive without caring which stain
        is which.

    ``names`` is a tuple, not a set, because order matters for most methods.
    """

    names: tuple[str, ...]
    order_matters: bool = True
    #: True when ``names`` are meaningful labels that must match; False when they
    #: are placeholders for an architecture that is agnostic to channel identity.
    names_are_labels: bool = True
    #: Does the number of channels have to equal ``len(names)``?
    #:
    #: Separate from ``names_are_labels`` because the two are genuinely
    #: independent.  CA-MAE is channel-agnostic *architecturally* -- channelwise
    #: cross-attention over patch tokens -- and runs with any C, so for it the
    #: names are placeholders AND the count is free.  A model could instead be
    #: identity-agnostic but still shape-bound.  ``names`` then documents the
    #: count this project normally feeds, not a requirement.
    channel_count_fixed: bool = True
    #: Hard upper bound on the number of input channels, when the *checkpoint*
    #: imposes one even though the architecture is channel-agnostic.
    #:
    #: CA-MAE is the motivating case: the published model is channel-agnostic
    #: (``ChannelAgnosticPatchEmbed`` shares one projection across channels),
    #: but its positional embedding is generated for exactly ``max_in_chans``
    #: modalities (``huggingface_mae.py:75``, ``vit.py:138-143``).  Feeding a
    #: twelfth modality is therefore not "an unusual input" but a shape error
    #: raised from inside the checkpoint -- with no ``reason`` code, in the
    #: framework's own vocabulary, at the worst possible moment.
    #:
    #: ``None`` means no bound is known.  This is deliberately *not* folded into
    #: ``channel_count_fixed``: that field says the count must be exactly
    #: ``len(names)``; this one says the count may vary but not beyond a limit.
    max_channels: int | None = None

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError('ChannelSpec.names must not be empty')
        if len(set(self.names)) != len(self.names):
            raise ValueError(f'ChannelSpec.names must be unique, got {self.names}')
        if not self.names_are_labels and self.order_matters:
            raise ValueError(
                'names_are_labels=False with order_matters=True is contradictory: '
                'if channel identity is meaningless then order cannot matter'
            )
        if not self.channel_count_fixed and self.order_matters:
            raise ValueError(
                'channel_count_fixed=False with order_matters=True is '
                'contradictory: if the count is free then the declared order '
                'cannot be a requirement'
            )
        if self.max_channels is not None and self.max_channels < 1:
            raise ValueError(
                f'ChannelSpec.max_channels must be >= 1, got {self.max_channels}'
            )


#: Scale keywords understood by :class:`PreprocessSpec`.
#:
#: ``per_crop_minmax`` is the value that matters most here, and the one that was
#: hardest to get right.  DeepProfiler's crop graph normalises *each crop
#: independently* onto [0, 1] (`imaging/cropping.py:65-67`) before the network
#: sees it, so the model's true input contract is [0, 1] even though the files on
#: disk hold 16-bit values.  Measured by driving the official crop graph directly:
#: its output spans exactly [0.0000, 1.0000] per crop.  Feeding raw 16-bit values
#: instead produces embeddings off by up to 8.8e4.
Scale = Literal[
    'raw', 'div255', 'div65535', 'per_image_selfnorm', 'per_crop_minmax', 'none'
]

#: How a *raw acquisition array* becomes the array declared by ``value_range``.
#:
#: ``scale`` alone cannot express this: it describes what the model does to its
#: input (or what the caller must already have done), not how to get there from
#: the 16-bit values that come off the microscope.  Every value here is a
#: mechanical, deterministic transform that :func:`models.prepare` executes, so
#: the conversion stops living in dataset-specific scripts.
#:
#: ``per_crop_minmax_bhwc`` - per-sample per-channel min-max onto ``value_range``.
#: ``identity``             - the array already holds the values the model wants
#: ``per_crop_max255``      - per-sample global max stretch onto ``0..255``
#:                            (what the BBBC022 builder does for CA-MAE).
#: ``div65535``             - 16-bit counts onto ``[0, 1]``.
#: ``div257``               - 16-bit counts onto ``0..255`` (exact 16->8 bit).
#: ``per_sample_selfnorm``  - per-sample per-channel z-score.
#: ``mask_to_silhouette``   - a binary nucleus mask rendered at ``spatial``.
#: ``matlab_centring``      - declared but *not* executable here: MorphoGenie's
#:                            256x256 centring lives in MATLAB upstream, so
#:                            ``prepare`` refuses rather than guessing.
SourceMap = Literal[
    'identity',
    'per_crop_minmax_bhwc',
    'per_crop_max255',
    'div65535',
    'div257',
    'per_sample_selfnorm',
    'mask_to_silhouette',
    'matlab_centring',
]
#: Whether a method's embedding is reproducible across runs (see ModelManifest).
Determinism = Literal['deterministic', 'stochastic']
#: Where the checkpoint being loaded came from (see ModelManifest).
#:
#: ``surrogate`` exists so the interface can be exercised end to end for methods
#: whose published weights cannot be obtained.  It is never a claim about the
#: method: the same architecture runs, the numbers are not the paper's.
WeightsProvenance = Literal['official', 'surrogate']


@dataclass(frozen=True)
class PreprocessSpec:
    """The numeric contract of the input, which shape checks cannot express.

    Every field here exists because getting it wrong produces a *plausible but
    wrong* embedding rather than an error.  ``scale`` says how pixel values must
    be mapped before the model sees them; ``value_range`` says what range the
    incoming array is expected to occupy so the scale can be verified rather
    than assumed.
    """

    scale: Scale
    #: Inclusive range the *incoming* array is expected to lie in, e.g. (0, 65535).
    value_range: tuple[float, float] | None = None
    #: dtype the model's input placeholder wants.
    to_dtype: str = 'float32'
    #: Memory layout expected by the framework: ``BHWC`` (Keras) or ``BCHW`` (torch).
    batch_layout: Literal['BHWC', 'BCHW'] = 'BHWC'
    #: Whether the method requires illumination-corrected input.
    illum_corrected: bool = False
    #: Physical arrangement of the channels *before* ``scale`` is applied.
    #:
    #: ``'stack'`` is the ordinary ``(N, C, H, W)`` (or ``(N, H, W, C)``) tensor
    #: and is what every method here consumes.
    #:
    #: ``'tape'`` -- channels laid out side by side in the width dimension with
    #: a trailing mask column, i.e. ``(H, W * (C + 1))`` -- is **not
    #: implemented** and is refused with ``reason='unsupported-encoding'``.
    #: It was written for the DINO4Cells Cell Painting dataset, and it turned
    #: out to describe that dataset's **storage** format, not its model input:
    #: the official dataset folds the stored tape back open and drops the mask
    #: column before the network sees it
    #: (``data_utils/cellpainting_dataset.py:23-47``, ``:86``).  Keeping the
    #: value in the type but refusing it loudly is deliberate -- a future
    #: method that genuinely consumes a wide image can implement it, and until
    #: then nobody can mistake the field for a working one.
    input_encoding: Literal['stack', 'tape'] = 'stack'
    #: What to do with the segmentation mask when the input carries one.
    #:
    #: Corresponds to the ``mode`` argument of the official DINO4Cells
    #: ``fold_channels`` (``data_utils/cellpainting_dataset.py:23-47``):
    #:
    #: ``'drop'``  -- the mask channel is discarded and the model sees only the
    #:                image channels.  This is what the official Cell Painting
    #:                dataset does (``mode="drop"`` is the default, and
    #:                ``SingleCellDataset.__getitem__`` calls it without an
    #:                argument), so it is the only setting that reproduces the
    #:                published model.
    #: ``'apply'`` -- the mask is used to zero out everything outside the cell
    #:                (official ``mode="apply"``).
    #: ``'none'``  -- no mask is part of the input at all (the default; every
    #:                method except DINO4Cells).
    #:
    #: Declared explicitly rather than inferred, because the choice changes the
    #: model's input geometry and therefore the embedding.
    mask_mode: Literal['drop', 'apply', 'none'] = 'none'
    #: Whether :func:`cellpaint_pipeline.models.prepare` may resize the input to
    #: :attr:`ModelManifest.spatial`.
    #:
    #: ``'auto'``   -- resize (the default; correct when the method documents
    #:                its own input size and any crop of another size is
    #:                acceptable to it).
    #: ``'refuse'`` -- do **not** resize.  The array is passed through
    #:                unchanged and a size mismatch is reported by
    #:                :func:`check_compatible` as ``reason='spatial-size'``.
    #:
    #: Why it exists: DeepProfiler's crop size is a *configuration* value
    #: upstream (``imaging/cropping.py:125``, ``box_size``), not a constant of
    #: the model, and the upstream operation on a crop is ``crop_and_resize`` --
    #: a crop, not a resize.  Silently resizing to satisfy the contract would
    #: invent a preprocessing step the official pipeline does not have, and
    #: would do it differently.  Refusing is the honest answer.
    resize_policy: Literal['auto', 'refuse'] = 'auto'
    #: How a raw acquisition array is mapped onto :attr:`value_range`.  ``None``
    #: means the caller must already supply the declared range (no conversion is
    #: attempted).  Executed by :func:`cellpaint_pipeline.models.prepare`; kept
    #: separate from :attr:`scale` because "what the model does to its input" and
    #: "how to get from microscope counts to that input" are different facts.
    source_map: SourceMap | None = None


@dataclass(frozen=True)
class WeightsSpec:
    """Where the weights live and how they must be loaded.

    ``host``/``downloadable`` are descriptive and drive reporting; this layer
    never downloads.  ``format`` encodes *observed* facts, e.g. OpenPhenom ships
    a file named ``.safetensors`` that is actually a ``torch.save`` zip, so a
    naive ``safetensors.torch.load_file`` fails.
    """

    source: str
    format: Literal[
        'keras-hdf5', 'torch-pth', 'torch-save-zip', 'safetensors',
        'lightning-ckpt', 'onnx', 'none',
    ]
    sha256: str | None = None
    host: Literal['local', 'hf-mirror', 'zenodo', 'gdrive', 'none'] = 'local'
    downloadable: bool = True
    approx_bytes: int | None = None
    #: Present only for Keras HDF5 checkpoints that store the training graph.
    weight_file_layer_count: int | None = None
    #: Layers in the graph the adapter actually *builds* (differs from the file).
    rebuilt_layer_count: int | None = None
    #: For adapters that import the published architecture from an upstream
    #: checkout: the directory, relative to the same weights root as ``source``,
    #: that must contain an importable ``models`` package.  Declared explicitly
    #: because it cannot be inferred from the checkpoint path -- a checkpoint at
    #: ``<root>/<model>/<file>.pt`` says nothing about where the code lives.
    repo_source: str | None = None


@dataclass(frozen=True)
class ModelManifest:
    """Everything a caller must know about one embedding method.

    This is the published contract.  A caller can inspect it, and
    :func:`check_compatible` can validate against it, without importing the
    framework the method runs on.
    """

    name: str
    version: str
    input_unit: InputUnit
    channels: ChannelSpec
    preprocess: PreprocessSpec
    weights: WeightsSpec
    output_dim: int
    #: Other widths this method may legally return, chosen by a load-time option.
    #:
    #: DeepProfiler is the motivating case: ``block6a_activation`` yields 672 but
    #: ``pool5`` yields 1280, and which one you get is an adapter argument, not a
    #: property of the method.  Declaring the alternatives is what lets the output
    #: check below stay strict without rejecting a legitimate layer choice.
    output_dim_alternatives: tuple[int, ...] = ()
    #: Width of one channel's vector when the method returns ``k`` of them.
    #:
    #: uniDINO returns ``384 * C`` (one DINOv2 vector per input channel), and
    #: CA-MAE's ``return_channelwise`` does the same.  Without this the only
    #: honest declaration would be a width that is wrong by a factor of five.
    #:
    #: The invariant both fields must satisfy together: **``output_dim`` is the
    #: width returned for the declared channel count**.  ``output_dim_per_channel``
    #: then says the total scales as ``per_channel * C`` when the caller asks for
    #: k vectors (or when the method always does).  So uniDINO, which only has
    #: the per-channel mode, declares ``1920`` (384 * 5) and not a bare ``384``;
    #: CA-MAE pools by default and so declares its pooled ``384`` while listing
    #: ``per_channel=384`` for the ``return_channelwise`` path.
    output_dim_per_channel: int | None = None
    #: Human-readable summary of the output (e.g. ``'block6a_activation'``).
    feature_layer: str | None = None
    #: Spatial size the method expects, ``(height, width)``; ``None`` = flexible.
    spatial: tuple[int, int] | None = None
    #: For ``InputUnit.CHANNEL_EMBEDDINGS``: the width of each per-channel vector.
    #:
    #: Without this the layer had no way to tell a ``(N, 5, 1536)`` vector stack
    #: from a ``(N, 5, 256, 256)`` pixel stack, and `check_compatible` accepted
    #: the pixel stack -- so a caller who forgot ``unit=`` got past validation and
    #: only failed later, inside the adapter.  Declaring the width lets the
    #: contract reject it before any model is loaded, which is the whole point.
    input_feature_dim: int | None = None
    #: SPDX-ish licence string, reported with every result.
    license: str = 'UNKNOWN'
    #: Import requirements, e.g. ``('tensorflow', 'efficientnet')``.
    requires: tuple[str, ...] = ()
    #: CellProfiler-backed or otherwise path-fragile methods set this True.
    ascii_path_required: bool = False
    #: True when running this method also needs its upstream *repository*, not
    #: just the checkpoint: the published architecture is imported from there
    #: rather than vendored, so the code stays the single source of truth.
    #: Declared, rather than inferred from a path, so ``probe()`` can report an
    #: honest status -- a method with weights but no code is not runnable.
    requires_upstream_repo: bool = False
    #: Whether repeated calls on identical input return identical embeddings.
    #:
    #: ``stochastic`` is not a defect to be hidden: a benchmark that compares
    #: a stochastic method once against a deterministic one is comparing noise
    #: against signal.  Declaring it lets downstream code know it must repeat
    #: runs and aggregate, instead of silently reporting one draw as the answer.
    #: This is measured, not assumed -- the reason it exists is that
    #: MorphoGenie's published extraction path returns a reparameterised sample
    #: (`reparametrize(mu, logvar)`), so two calls on one input differ.
    determinism: Determinism = 'deterministic'
    parity_mode: ParityMode = 'normalized'
    #: Upstream stages that must have run for parity comparison to be meaningful.
    parity_prerequisites: tuple[str, ...] = ()
    #: Additional models whose *outputs* this method consumes, as
    #: ``(name, sha256)`` pairs.
    #:
    #: CellCLIP does not read pixels.  Its input is per-channel DINOv2-Giant
    #: embeddings, so a run is only reproducible if the model that produced them
    #: is identified too.  Without this field the provenance covered only the
    #: 1.8 GB CellCLIP checkpoint and silently omitted the 4.5 GB model that
    #: actually generated its input -- a chain is only as traceable as its
    #: weakest link.
    chain_weights: tuple[tuple[str, str], ...] = ()
    #: Where the checkpoint that will actually be loaded came from.
    #:
    #: ``official``  -- the authors' published weights.  Parities and scores are
    #:                  meaningful.
    #: ``surrogate`` -- weights trained here on local data, standing in for a
    #:                  checkpoint that could not be obtained (see the v3 profile).
    #:                  The architecture and the input/output contract are the
    #:                  published ones, so the *interface* is exercised end to
    #:                  end; the *numbers* are not the paper's, and a benchmark
    #:                  that mixes the two without saying so is lying.
    #:
    #: This is a manifest field rather than a convention because it has to reach
    #: the caller: it is copied into every ``EmbeddingResult`` and every benchmark
    #: row, so a surrogate cannot be mistaken for the real method downstream.
    weights_provenance: WeightsProvenance = 'official'
    notes: str = ''

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self.channels.names


@dataclass(frozen=True)
class CompatibilityIssue:
    """One reason a call cannot proceed, with an actionable message."""

    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f'[{self.code}] {self.message}'


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of :func:`check_compatible`.  Truthy when compatible."""

    manifest_name: str
    issues: tuple[CompatibilityIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def raise_for_status(self) -> None:
        """Raise the embedding layer's input error when not compatible.

        Imported lazily so this module never depends on the error module at
        import time (keeps the import graph one-directional).
        """
        if self.ok:
            return
        from cellpaint_pipeline.models.errors import IncompatibleInput

        raise IncompatibleInput(
            f"Input is not compatible with model '{self.manifest_name}': "
            + '; '.join(str(issue) for issue in self.issues),
            reason=self.issues[0].code,
            details=[str(issue) for issue in self.issues],
        )

    def to_dict(self) -> dict:
        return {
            'model': self.manifest_name,
            'ok': self.ok,
            'codes': list(self.codes),
            'issues': [{'code': i.code, 'message': i.message} for i in self.issues],
        }


def is_ascii_path(value: str | object) -> bool:
    """Return True when ``value`` can be represented in pure ASCII.

    CellProfiler (including the frozen Windows build) fails on non-ASCII paths
    with an opaque ``OSError``; catching it here turns a late crash into an
    early, explainable rejection.
    """
    return str(value).isascii()


def _dtype_kind(dtype: object) -> str | None:
    """Normalise a dtype to numpy's single-character ``kind`` (``f``, ``u``...).

    numpy dtypes expose ``.kind``; a string like ``'float32'`` or ``'uint8'``
    (from a test double or a serialised payload) is mapped by prefix so both
    spellings compare equal downstream.
    """
    if dtype is None:
        return None
    explicit = getattr(dtype, 'kind', None)
    if isinstance(explicit, str) and len(explicit) == 1:
        return explicit
    text = str(dtype).lower()
    if text.startswith('float') or text.startswith('f'):
        return 'f'
    if text.startswith('uint') or text.startswith('u'):
        return 'u'
    if text.startswith('int') or text.startswith('i'):
        return 'i'
    if text.startswith('bool') or text.startswith('b'):
        return 'b'
    return None


def _array_facts(array: object) -> dict:
    """Read shape/dtype off any array-like without importing numpy.

    ``dtype_kind`` is consulted as a fallback for payloads that expose a kind
    without a dtype object, so the contract stays duck-typed.
    """
    shape = getattr(array, 'shape', None)
    dtype = getattr(array, 'dtype', None)
    ndim = getattr(array, 'ndim', None)
    if ndim is None and shape is not None:
        try:
            ndim = len(shape)
        except TypeError:
            ndim = None

    kind = _dtype_kind(dtype)
    if kind is None:
        kind = getattr(array, 'dtype_kind', None)

    return {
        'shape': tuple(shape) if shape is not None else None,
        'ndim': ndim,
        'dtype': str(dtype) if dtype is not None else None,
        'kind': kind,
        'min': getattr(array, 'min', None),
        'max': getattr(array, 'max', None),
    }


def _check_scale_dtype(manifest: ModelManifest, facts: dict, issues: list) -> None:
    """Reject an input whose *dtype* cannot carry the scale's domain.

    WHY THIS IS SEPARATE FROM THE VALUE-RANGE CHECK
    -----------------------------------------------
    The value-range check compares numbers, and numbers lie about their own
    encoding.  A ``uint8`` array whose maximum is 255 sits comfortably inside a
    declared ``(0, 65535)`` range -- but every one of those pixels is 257
    times darker than the 16-bit value it is pretending to be.

    uniDINO's preprocessing is ``img / (2**16 - 1)``
    (``source/image_ops.py:14-20``).  Handed a ``uint8`` crop it produces a
    picture that is essentially black: measured, the prepared maximum is
    ``0.0039`` instead of ``0.98``.  The model still runs, still returns 1920
    numbers, and nothing anywhere raises.  That is precisely the failure this
    layer exists to prevent, so the dtype is checked *before* the value range:
    an 8-bit array cannot express the domain the scale divides by.

    The same reasoning rules out a "span" heuristic (observed span < declared
    span): a legitimately dim field is not a contract violation.
    """
    if manifest.preprocess.scale != 'div65535':
        return
    observed_kind = facts.get('kind')
    observed_dtype = facts.get('dtype')
    if observed_kind != 'u':
        # Only unsigned integer input is at risk; a float array is either
        # already scaled (and then the value-range check speaks) or an
        # unsigned-valued array in disguise (and then the caller chose that).
        return
    # ``uint8`` is the only unsigned dtype that cannot hold 65535.
    if observed_dtype is not None and 'uint8' in str(observed_dtype):
        issues.append(
            CompatibilityIssue(
                'value-range',
                f"'{manifest.name}' divides by 2**16-1, so its input must be "
                f"uint16 (or float values already spanning 0..65535); got "
                f"{observed_dtype}. An 8-bit array holds values 257x smaller "
                f"than the 16-bit counts this method was trained on, so "
                f"prepare() would produce an essentially black image and the "
                f"embedding would be silently wrong.",
            )
        )


def _check_value_range(manifest: ModelManifest, array: object, facts: dict, issues: list) -> None:
    """Verify the array's dtype and range against the declared contract.

    Two distinct failures are caught here, and they need different checks:

    1. **Values outside the declared range** -- e.g. a 16-bit array handed to a
       method that declares ``(0, 255)``.
    2. **Wrong scale within the declared range** -- the common "I divided by 255
       already" mistake.  ``[0, 1]`` is a *subset* of ``[0, 255]``, so a range
       check alone cannot see it.  The unambiguous signal is the **dtype**: a
       method that declares ``to_dtype='uint8'`` is telling us its input is
       quantised, so a float array is a contract violation regardless of the
       values it happens to hold.

    A span heuristic (``observed_span >= k * declared_span``) was rejected: a
    legitimately dark field of view is a false positive, and false positives here
    would train operators to ignore the check.
    """
    declared_range = manifest.preprocess.value_range
    declared_dtype = manifest.preprocess.to_dtype

    observed_kind = facts.get('kind')
    observed_dtype = facts.get('dtype')

    _check_scale_dtype(manifest, facts, issues)

    if declared_dtype == 'uint8' and observed_kind == 'f':
        issues.append(
            CompatibilityIssue(
                'dtype',
                f"'{manifest.name}' declares to_dtype='uint8' "
                f"(scale={manifest.preprocess.scale!r}) but received "
                f"{observed_dtype}. A float array here usually means the input "
                f"was already divided by 255; the model normalises internally, "
                f"so that division would be applied twice.",
            )
        )

    if declared_range is None:
        return
    lo_fn = getattr(array, 'min', None)
    hi_fn = getattr(array, 'max', None)
    if not callable(lo_fn) or not callable(hi_fn):
        return
    try:
        actual_lo = float(lo_fn())
        actual_hi = float(hi_fn())
    except (TypeError, ValueError):
        return
    low, high = declared_range
    if actual_lo < low - 1e-9 or actual_hi > high + 1e-9:
        issues.append(
            CompatibilityIssue(
                'value-range',
                f"the array handed to the contract spans "
                f"[{actual_lo:g}, {actual_hi:g}] but '{manifest.name}' "
                f"expects {manifest.preprocess.value_range} "
                f"(scale={manifest.preprocess.scale!r}). "
                f"Note this range is the one the MODEL consumes: if you are "
                f"passing raw acquisition values, run them through "
                f"models.prepare() (or use assume='raw_crop') first -- "
                f"feeding the wrong scale produces a plausible but wrong "
                f"embedding with no error.",
            )
        )


def check_compatible(
    manifest: ModelManifest,
    *,
    unit: InputUnit | str | None = None,
    channels: tuple[str, ...] | list[str] | None = None,
    array: object | None = None,
    runtime_root: str | None = None,
) -> CompatibilityReport:
    """Decide whether a call is compatible.  Pure: no IO, no heavy imports.

    Every check below is falsifiable and produces a distinct ``code`` so callers
    (and tests) can assert on the specific reason rather than on message text.

    Args:
        manifest: the method's declared contract.
        unit: the kind of input being supplied.
        channels: channel names, in the order they appear in ``array``.
        array: the payload; only its shape/dtype/range are inspected.
        runtime_root: filesystem root the method will run in; checked for ASCII
            only when the manifest requires it.
    """
    issues: list[CompatibilityIssue] = []

    if unit is not None:
        unit_value = unit.value if isinstance(unit, InputUnit) else str(unit)
        if unit_value != manifest.input_unit.value:
            issues.append(
                CompatibilityIssue(
                    'input-unit',
                    f"'{manifest.name}' consumes {manifest.input_unit.value!r}, "
                    f"got {unit_value!r}",
                )
            )

    if channels is not None:
        supplied = tuple(channels)
        expected = manifest.channel_names
        spec = manifest.channels

        if not spec.names_are_labels:
            # Channel-agnostic model: any labels are acceptable.  Whether the
            # count is also free is a separate declaration, because a model can
            # be identity-agnostic while still being shape-bound.
            if spec.channel_count_fixed and len(supplied) != len(expected):
                issues.append(
                    CompatibilityIssue(
                        'channel-count',
                        f"'{manifest.name}' is channel-agnostic but shape-bound: "
                        f"it expects {len(expected)} channels, got {len(supplied)} "
                        f"({supplied}). Names are free-form for this model, so only "
                        f"the count is checked.",
                    )
                )
            if spec.max_channels is not None and len(supplied) > spec.max_channels:
                issues.append(
                    CompatibilityIssue(
                        'channel-count',
                        f"'{manifest.name}' accepts at most {spec.max_channels} "
                        f"channels, got {len(supplied)}. The published checkpoint "
                        f"cannot represent more, so it would fail with a shape "
                        f"error from inside the model rather than here.",
                    )
                )
        elif set(supplied) != set(expected):
            issues.append(
                CompatibilityIssue(
                    'channel-set',
                    f"channel names {supplied} do not match the model's {expected}",
                )
            )
        elif spec.order_matters and supplied != expected:
            issues.append(
                CompatibilityIssue(
                    'channel-order',
                    f"channel order {supplied} differs from the model's {expected}; "
                    f"this is a silent failure mode -- the embedding changes but "
                    f"nothing raises",
                )
            )

    if array is not None:
        facts = _array_facts(array)
        # A tape array is legitimately 2-D; channel embeddings are 3-D
        # (N, channels, features); everything else is a batched image stack.
        if manifest.input_unit is InputUnit.CHANNEL_EMBEDDINGS:
            expected_ndims = (3,)
        else:
            expected_ndims = (3, 4)
        if facts['ndim'] is not None and facts['ndim'] not in expected_ndims:
            expected_text = {
                (3,): 'a 3-D (N, channels, features) array of channel embeddings',
                (3, 4): 'a 3-D or 4-D array',
            }[expected_ndims]
            issues.append(
                CompatibilityIssue(
                    'ndim',
                    f"expected {expected_text}, got {facts['ndim']}-D "
                    f"(shape={facts['shape']})",
                )
            )
        if facts['shape'] is not None and facts['ndim'] in expected_ndims:
            layout = manifest.preprocess.batch_layout
            if manifest.input_unit is InputUnit.CHANNEL_EMBEDDINGS:
                # (N, C, D): check the channel count and the feature width.
                # Without the width check a 4-D pixel stack was already rejected
                # by the rank test above, but a *3-D* pixel stack at (N, 5, 256)
                # would still slip through, and the adapter would then fail deep
                # inside the forward pass instead of here.
                _, observed_channels, observed_width = facts['shape']
                if manifest.channels.channel_count_fixed and observed_channels != len(
                    manifest.channel_names
                ):
                    issues.append(
                        CompatibilityIssue(
                            'channel-count',
                            f"'{manifest.name}' expects {len(manifest.channel_names)} "
                            f"channel vectors, got {observed_channels} "
                            f"(shape={facts['shape']})",
                        )
                    )
                if (
                    manifest.input_feature_dim is not None
                    and observed_width != manifest.input_feature_dim
                ):
                    issues.append(
                        CompatibilityIssue(
                            'feature-width',
                            f"'{manifest.name}' expects each channel vector to be "
                            f"{manifest.input_feature_dim}-d, got {observed_width} "
                            f"(shape={facts['shape']}). Pixel data has the wrong "
                            f"width, so this is not a channel-embedding input.",
                        )
                    )
            else:
                channel_axis = -1 if layout == 'BHWC' else 1
                observed = facts['shape'][channel_axis]
                # A channel-agnostic, count-free method accepts whatever width
                # the caller's data has; only the axes that do not scale with the
                # channel count stay checkable.
                if manifest.channels.channel_count_fixed and observed != len(
                    manifest.channel_names
                ):
                    issues.append(
                        CompatibilityIssue(
                            'channel-count',
                            f"'{manifest.name}' expects "
                            f"{len(manifest.channel_names)} channels in {layout} "
                            f"layout, got {observed} (shape={facts['shape']})",
                        )
                    )
                if (
                    manifest.channels.max_channels is not None
                    and observed > manifest.channels.max_channels
                ):
                    issues.append(
                        CompatibilityIssue(
                            'channel-count',
                            f"'{manifest.name}' accepts at most "
                            f"{manifest.channels.max_channels} channels, got "
                            f"{observed} (shape={facts['shape']}). The published "
                            f"checkpoint cannot represent more, so it would fail "
                            f"with a shape error from inside the model rather "
                            f"than here.",
                        )
                    )
                if manifest.spatial is not None and facts['ndim'] in (3, 4):
                    # A 4-D array is (N, H, W, C) in BHWC layout or
                    # (N, C, H, W) in BCHW layout; a 3-D one is the same without
                    # the batch axis.  Checking only the batched case would let
                    # a single wrongly-sized crop through -- and single crops
                    # are the normal unit of work for this layer.
                    if facts['ndim'] == 4:
                        spatial_axes = (1, 2) if layout == 'BHWC' else (2, 3)
                    else:
                        spatial_axes = (0, 1) if layout == 'BHWC' else (1, 2)
                    observed_spatial = (
                        facts['shape'][spatial_axes[0]],
                        facts['shape'][spatial_axes[1]],
                    )
                    if tuple(observed_spatial) != tuple(manifest.spatial):
                        issues.append(
                            CompatibilityIssue(
                                'spatial-size',
                                f"'{manifest.name}' expects {manifest.spatial}, got "
                                f"{tuple(observed_spatial)}",
                            )
                        )
        _check_value_range(manifest, array, facts, issues)

    if manifest.ascii_path_required and runtime_root is not None:
        if not is_ascii_path(runtime_root):
            issues.append(
                CompatibilityIssue(
                    'ascii-path',
                    f"'{manifest.name}' requires an ASCII runtime path but got "
                    f"{runtime_root!r}; CellProfiler fails on non-ASCII paths "
                    f"with an opaque OSError",
                )
            )

    return CompatibilityReport(manifest_name=manifest.name, issues=tuple(issues))
