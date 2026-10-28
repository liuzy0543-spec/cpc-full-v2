"""Tests for `preprocessing.prepare`: does the contract actually get executed?

Why this file exists
--------------------
`preprocessing.py` turns raw acquisition values into the tensor a method expects.
It is the one module where a mistake does not raise -- it silently changes the
embedding.  Measured on real data, feeding un-normalised pixels instead of
normalised ones moved the DeepProfiler embedding by 4.5e6, so "looks fine" is not
a useful signal here.

Every test therefore compares against the **official formula** written out
independently, rather than against a stored snapshot: a snapshot would only pin
whatever the code currently does, including its bugs.

Each `source_map` variant has exactly one test; there are six, plus the tape
encoding and the refusals.
"""
from __future__ import annotations

import numpy as np
import pytest

from cellpaint_pipeline.models import preprocessing, registry
from cellpaint_pipeline.models.errors import IncompatibleInput, UnknownModel

#: The epsilon `_apply_source_map` uses to avoid dividing by zero.
EPS = 1e-12


def source(name: str):
    """Resolve a method to something ``prepare``/``plan`` accept.

    `unidino`, `dinocells` and `morphogenie` were cancelled on 2026-10-21 (their
    weights cannot be obtained here) and no longer resolve by name.  Their
    contracts are kept in ``registry.PENDING``, so the transforms they declare can
    still be tested -- and should be, because a wrong `source_map` is exactly the
    kind of mistake that changes an embedding without raising.
    """
    try:
        return registry.manifest(name)
    except UnknownModel:
        for registration in registry.PENDING:
            if registration.manifest.name == name:
                return registration.manifest
        raise


def raw_crop(n: int = 3, size: int = 32, channels: int = 5, seed: int = 0) -> np.ndarray:
    """A plausible 16-bit acquisition batch: values well above 1, never zero."""
    rng = np.random.RandomState(seed)
    return (rng.rand(n, size, size, channels) * 52000 + 850).astype('float32')


# --------------------------------------------------------------- source maps --

class TestSourceMaps:
    """One test per `source_map`, each against the formula it claims to implement."""

    def test_per_crop_minmax_bhwc_matches_official_cropping(self):
        """DeepProfiler: per-crop, per-channel min-max onto [0, 1].

        This is `cropping.py:65-67` upstream.  It is also the branch the parity
        harness proved bitwise-identical to the official run, so it is the
        reference the others are written to look like.
        """
        crops = raw_crop(n=4, size=32)

        lo = crops.min(axis=(1, 2), keepdims=True)
        hi = crops.max(axis=(1, 2), keepdims=True)
        expected = (crops - lo) / np.maximum(hi - lo, EPS)

        got = preprocessing.prepare('deepprofiler', crops)

        assert got.shape == expected.shape
        assert np.array_equal(got, expected.astype(got.dtype))

    def test_ca_mae_is_identity_normalisation(self):
        """CA-MAE: the official normalisation is ``pixels/255`` inside the model
        (``normalizer.py:7``), so ``prepare`` must hand the uint8 crop through
        with no value change of its own.

        This used to assert a per-crop max stretch.  That step was removed
        because it is not official: measured, a global 0.5x rescale moves the
        embedding by 22% of its std, so the checkpoint's InstanceNorm does *not*
        cancel it -- the stretch only looked harmless because every BBBC022 crop
        already has per-crop max 255.
        """
        # ca_mae declares uint8 / 0..255, so the identity claim is only
        # meaningful on an 8-bit batch (a 16-bit one has no declared conversion,
        # which is exactly the gap the docs say is unresolved upstream).
        rng = np.random.RandomState(0)
        crops = (rng.rand(3, 256, 256, 5) * 255.0).astype('uint8')

        got = preprocessing.prepare('ca_mae', crops)

        assert got.dtype == np.uint8
        assert got.shape == (3, 5, 256, 256), 'declared BCHW layout not applied'
        # Compare through the BCHW transpose, since layout is the only difference.
        assert np.array_equal(np.transpose(got, (0, 2, 3, 1)), crops)

    def test_per_crop_max255_map_itself(self):
        """The ``per_crop_max255`` primitive still works, though no method now
        declares it: per-sample global max stretch, clipped into 0..255."""
        crops = raw_crop(n=3, size=256)

        hi = crops.max(axis=(1, 2, 3), keepdims=True)
        expected = np.clip(crops / np.maximum(hi, EPS) * 255.0, 0.0, 255.0)

        got = preprocessing._apply_source_map(crops, 'per_crop_max255')

        assert np.array_equal(
            np.asarray(got).astype(np.uint8), expected.astype(np.uint8)
        )

    def test_div65535_matches_declared_range(self):
        """uniDINO: a straight 16-bit rescale to [0, 1]."""
        crops = raw_crop(n=3, size=224)

        expected = crops / 65535.0

        got = preprocessing.prepare(source('unidino'), crops)

        assert got.shape == (3, 5, 224, 224)
        assert np.allclose(np.transpose(got, (0, 2, 3, 1)), expected, atol=1e-7)

    def test_div257_matches_declared_range(self):
        """The 16-bit -> 8-bit rescale some pipelines use (255 * 257 = 65535)."""
        crops = raw_crop(n=2, size=32)

        expected = np.clip(crops / 257.0, 0.0, 255.0)

        got = preprocessing._apply_source_map(crops, 'div257')

        assert np.allclose(got, expected, atol=1e-6)

    def test_per_sample_selfnorm_standardises_per_channel(self):
        """dinocells: subtract the per-crop, per-channel mean and divide by std."""
        crops = raw_crop(n=2, size=96)

        mean = crops.mean(axis=(1, 2), keepdims=True)
        std = crops.std(axis=(1, 2), keepdims=True)
        expected = (crops - mean) / np.maximum(std, EPS)

        got = preprocessing._apply_source_map(crops, 'per_sample_selfnorm')

        assert np.allclose(got, expected, atol=1e-6)
        # Standardised input must be centred and unit-scale per channel.
        assert np.allclose(got.mean(axis=(1, 2)), 0.0, atol=1e-5)
        assert np.allclose(got.std(axis=(1, 2)), 1.0, atol=1e-5)

    def test_unknown_source_map_is_refused(self):
        """An unhandled name must raise rather than pass the array through."""
        with pytest.raises(IncompatibleInput):
            preprocessing._apply_source_map(raw_crop(), 'invented_map')


# ------------------------------------------------------------ mask silhouette --

class TestMaskSilhouette:
    """O2-VAE consumes a nucleus silhouette, not intensities."""

    def test_mask_becomes_a_single_channel_silhouette(self):
        masks = np.zeros((4, 64, 64), dtype='uint8')
        masks[:, 16:48, 16:48] = 1

        got = preprocessing.prepare(
            'o2vae', np.zeros((4, 1, 64, 64), dtype='float32'), masks=masks
        )

        assert got.shape == (4, 1, 64, 64)
        assert set(np.unique(got)) <= {0.0, 1.0}, 'silhouette must stay binary'
        assert got.sum() == masks.sum()

    def test_mask_is_resized_to_the_declared_spatial(self):
        """The plan advertises a resize step, so this branch must honour it.

        Regression: an earlier version returned the mask at whatever size it
        arrived, producing a tensor that contradicted the declared `(64, 64)`.
        """
        masks = np.zeros((3, 128, 128), dtype='uint8')
        masks[:, 32:96, 32:96] = 1

        got = preprocessing.prepare(
            'o2vae', np.zeros((3, 1, 64, 64), dtype='float32'), masks=masks
        )

        assert got.shape == (3, 1, 64, 64), 'mask was not resized to manifest.spatial'
        assert set(np.unique(got)) <= {0.0, 1.0}

    def test_mask_is_required(self):
        with pytest.raises(IncompatibleInput) as excinfo:
            preprocessing.prepare('o2vae', np.zeros((1, 64, 64), dtype='float32'))

        assert getattr(excinfo.value, 'reason', None) == 'mask-required'

    def test_single_mask_keeps_single_rank(self):
        got = preprocessing.prepare(
            'o2vae', np.zeros((1, 64, 64), dtype='float32'),
            masks=np.ones((64, 64), dtype='uint8'),
        )

        assert got.shape == (1, 64, 64), 'a 2-D mask should give a 3-D silhouette'


# ------------------------------------------------------------------- layout --

class TestLayoutAndRank:
    """Batch rank and batch layout are part of the contract, not a detail."""

    def test_bhwc_method_keeps_channels_last(self):
        crops = raw_crop(n=2, size=32)

        got = preprocessing.prepare('deepprofiler', crops)

        assert got.shape == (2, 32, 32, 5)

    def test_bchw_method_moves_channels_first(self):
        crops = raw_crop(n=2, size=256)

        got = preprocessing.prepare('ca_mae', crops)

        assert got.shape == (2, 5, 256, 256)

    def test_single_crop_keeps_single_rank(self):
        """3-D in must give 3-D out; a batch axis is not silently added."""
        crop = raw_crop(n=1, size=32)[0]

        single = preprocessing.prepare('deepprofiler', crop)
        batched = preprocessing.prepare('deepprofiler', crop[None])

        assert single.shape == (32, 32, 5)
        assert batched.shape == (1, 32, 32, 5)
        assert np.array_equal(single, batched[0])

    def test_declared_spatial_is_applied(self):
        crops = raw_crop(n=2, size=128)

        got = preprocessing.prepare(source('unidino'), crops)

        assert got.shape == (2, 5, 224, 224), 'declared spatial was not honoured'


# ----------------------------------------------------------------- channels --

class TestChannelOrder:
    """A swapped channel order must be corrected by name, not by position."""

    def test_channels_are_reordered_into_manifest_order(self):
        names = ('DNA', 'ER', 'RNA', 'AGP', 'Mito')
        crops = raw_crop(n=2, size=32)
        permutation = [2, 0, 1, 3, 4]
        shuffled = crops[..., permutation]
        shuffled_names = tuple(names[i] for i in permutation)

        reference = preprocessing.prepare('deepprofiler', crops)
        got = preprocessing.prepare(
            'deepprofiler', shuffled, channels=shuffled_names
        )

        assert np.array_equal(got, reference), (
            'reordering by name did not restore the manifest order'
        )

    def test_channel_count_mismatch_is_refused(self):
        with pytest.raises(IncompatibleInput) as excinfo:
            preprocessing.prepare(
                'deepprofiler', raw_crop(n=2, size=32),
                channels=('DNA', 'ER', 'RNA'),
            )

        assert getattr(excinfo.value, 'reason', None) == 'channel-count'


# ------------------------------------------------------------- channel stack --

class TestChannelStackEncoding:
    """dinocells consumes a channel stack and drops the mask.

    The official Cell Painting dataset stores each cell as a "tape" -- the
    channels side by side in the width dimension with a mask column -- but
    that is a *storage* format.  The official dataset class folds the tape
    back open and **drops the mask column** before the network sees it
    (``data_utils/cellpainting_dataset.py:23-47``, ``:86``), and the network
    is built for ``in_chans=num_channels``
    (``run_get_features.py:87-91``).

    An earlier contract implemented the *inverse* of the official fold and fed
    the resulting wide single-channel image to the network.  The fold was
    implemented correctly -- which is exactly the problem: it reproduced the
    upstream storage format and passed it off as the model input format.
    """

    @staticmethod
    def _expected(crop: np.ndarray) -> np.ndarray:
        """The official chain: per-channel z-score, mask discarded.

        The result is in the declared layout -- BCHW, i.e. (C, H, W).
        """
        mean = crop.mean(axis=(0, 1), keepdims=True)
        std = crop.std(axis=(0, 1), keepdims=True)
        normalised = (crop - mean) / np.maximum(std, EPS)
        return np.transpose(normalised, (2, 0, 1))

    def test_the_model_input_is_a_stack_not_a_tape(self):
        crop = raw_crop(n=1, size=96)[0]

        got = preprocessing.prepare(source('dinocells'), crop)

        assert got.shape == (5, 96, 96), 'expected (C, H, W)'
        assert got.ndim == 3, 'a single crop keeps its rank'
        assert np.allclose(got, self._expected(crop), atol=1e-5)

    def test_a_batch_is_allowed(self):
        """The official model batches like any image stack.

        The old tape encoding refused a batch because the width carried the
        channel index; that refusal was a property of the local encoding, not
        of the method.
        """
        got = preprocessing.prepare(
            source('dinocells'), raw_crop(n=3, size=96),
            masks=np.ones((3, 96, 96), dtype='uint8'),
        )

        assert got.shape == (3, 5, 96, 96)

    def test_the_mask_is_discarded_not_fed_to_the_model(self):
        """Official `mode="drop"`: the mask column is thrown away."""
        crop = raw_crop(n=1, size=96)[0]
        mask = np.zeros((96, 96), dtype='uint8')
        mask[24:72, 24:72] = 1

        with_mask = preprocessing.prepare(source('dinocells'), crop, masks=mask)
        without = preprocessing.prepare(source('dinocells'), crop)

        assert with_mask.shape == without.shape
        assert np.array_equal(with_mask, without)

    def test_the_mask_is_not_required(self):
        """The model input is five image channels; the mask is not one of them."""
        got = preprocessing.prepare(source('dinocells'), raw_crop(n=1, size=96)[0])

        assert got.shape == (5, 96, 96)

    def test_a_mask_of_the_wrong_rank_is_refused(self):
        with pytest.raises(IncompatibleInput) as excinfo:
            preprocessing.prepare(
                source('dinocells'), raw_crop(n=1, size=96)[0],
                masks=np.ones(96, dtype='uint8'),
            )

        assert getattr(excinfo.value, 'reason', None) == 'mask-shape'

    def test_the_tape_encoding_is_refused_outright(self):
        """The framework no longer implements the tape layout.

        It is refused loudly rather than left half-implemented, because a
        half-implemented encoding is worse than a missing one: it produces a
        number without producing an error.
        """
        manifest = source('dinocells')
        assert manifest.preprocess.input_encoding == 'stack'
        assert manifest.preprocess.mask_mode == 'drop'


    def test_mask_mode_apply_zeroes_the_region_outside_the_mask(self):
        """The official `mode="apply"` branch: mask the cell, drop the rest.

        It is the sibling of `mode="drop"` (which discards the mask), so it
        needs its own test -- otherwise the branch is only ever exercised by
        nobody.
        """
        from dataclasses import replace

        manifest = source('dinocells')
        applying = replace(
            manifest, preprocess=replace(manifest.preprocess, mask_mode='apply')
        )
        crop = raw_crop(n=1, size=96)[0]
        mask = np.zeros((96, 96), dtype='uint8')
        mask[24:72, 24:72] = 1

        got = preprocessing.prepare(applying, crop, masks=mask)

        assert got.shape == (5, 96, 96)
        # Outside the mask the crop is zeroed; inside it, untouched.
        assert np.allclose(got[:, :10, :10], 0.0)
        inside = np.abs(got[:, 40:50, 40:50]).max()
        assert inside > 0.0

    def test_mask_mode_apply_needs_a_mask(self):
        from dataclasses import replace

        manifest = source('dinocells')
        applying = replace(
            manifest, preprocess=replace(manifest.preprocess, mask_mode='apply')
        )
        with pytest.raises(IncompatibleInput) as excinfo:
            preprocessing.prepare(applying, raw_crop(n=1, size=96)[0])

        assert getattr(excinfo.value, 'reason', None) == 'mask-required'


# ---------------------------------------------------------------- refusals --

class TestRefusals:
    """The cases where guessing would be worse than failing."""

    def test_cellclip_refuses_pixels(self):
        """CellCLIP eats DINOv2 channel embeddings; a pixel crop cannot be converted."""
        with pytest.raises(IncompatibleInput) as excinfo:
            preprocessing.prepare('cellclip', raw_crop())

        assert getattr(excinfo.value, 'reason', None) == 'chain-required'

    def test_morphogenie_refuses_matlab_centring(self):
        """Upstream centres in MATLAB, outside this repository -- do not guess."""
        with pytest.raises(IncompatibleInput) as excinfo:
            preprocessing.prepare(source('morphogenie'), raw_crop())

        assert getattr(excinfo.value, 'reason', None) == 'unsupported-preprocess'
        assert 'MATLAB' in str(excinfo.value)

    def test_unidino_refuses_an_eight_bit_crop(self):
        """`div65535` on uint8 is a black image, not a dark one.

        The scale divides by 2**16-1, so an 8-bit crop lands 257x too dark and
        the model still returns a well-formed embedding.  That is the failure
        this layer exists to prevent, so it must be refused -- with a reason
        code, and before any transform has laundered the dtype away.
        """
        crop8 = raw_crop().astype('uint8')

        with pytest.raises(IncompatibleInput) as excinfo:
            preprocessing.prepare(source('unidino'), crop8)

        assert getattr(excinfo.value, 'reason', None) == 'value-range'
        assert '16-bit' in str(excinfo.value)

    def test_unidino_refuses_an_already_scaled_float_crop(self):
        """The scale must not be applied twice.

        ``div65535`` maps [0, 65535] onto [0, 1], so an array whose maximum is
        already inside [0, 1] is indistinguishable from one that has been
        scaled.  Dividing it again yields values of order 1e-5 -- a black
        image with a perfectly well-formed embedding.  Refusing is the only
        honest answer.
        """
        already_scaled = raw_crop().astype('float32') / 65535.0

        with pytest.raises(IncompatibleInput) as excinfo:
            preprocessing.prepare(source('unidino'), already_scaled)

        assert getattr(excinfo.value, 'reason', None) == 'value-range'
        assert 'second time' in str(excinfo.value)

    def test_unidino_accepts_float_counts_in_the_sixteen_bit_domain(self):
        """The refusal must not catch raw counts that happen to be float.

        A float array spanning the 16-bit domain is exactly what the scale
        expects; only an array already inside the *output* domain is refused.
        """
        counts = raw_crop().astype('float32')

        prepared = preprocessing.prepare(source('unidino'), counts)

        assert prepared.max() > 0.5

    def test_unidino_accepts_a_sixteen_bit_crop(self):
        """The refusal must not be a blanket ban: uint16 is the real input."""
        crop16 = raw_crop().astype('uint16')

        prepared = preprocessing.prepare(source('unidino'), crop16)

        assert prepared.dtype == np.float32
        assert 0.0 <= prepared.max() <= 1.0
        # A uint16 crop divided by 65535 lands near the top of [0, 1].
        assert prepared.max() > 0.5

    def test_deepprofiler_refuses_a_crop_of_the_wrong_size(self):
        """The checkpoint wants 32x32; the upstream operation is a crop, not a resize.

        Silently resizing would invent a preprocessing step the official
        pipeline does not have, so the wrong size has to be rejected with a
        typed error instead.
        """
        with pytest.raises(IncompatibleInput) as excinfo:
            preprocessing.prepare(source('deepprofiler'), raw_crop(size=64))

        assert getattr(excinfo.value, 'reason', None) == 'spatial-size'
        assert '(32, 32)' in str(excinfo.value)

    def test_deepprofiler_accepts_the_size_its_checkpoint_wants(self):
        manifest = source('deepprofiler')
        prepared = preprocessing.prepare(manifest, raw_crop(size=32))

        # DeepProfiler is a Keras method: channels last, so the spatial axes
        # are 1 and 2, not the last two.
        assert prepared.shape[1:3] == (32, 32)


# -------------------------------------------------------------------- plan --

class TestPlan:
    """`plan()` describes without touching data, and must match `prepare()`."""

    def test_plan_needs_no_array(self):
        steps = preprocessing.plan('deepprofiler')

        # DeepProfiler declares the crop size its checkpoint expects and
        # refuses to resize (the upstream operation on a crop is a crop, not a
        # resize), so the plan says 'check-size' rather than 'resize'.
        assert [step.name for step in steps] == [
            'coerce', 'reorder-channels', 'check-size', 'source-map', 'layout',
            'cast',
        ]

    def test_plan_distinguishes_resize_from_refuse(self):
        """A refused size must not be advertised as a resize.

        The plan is what a caller reads to learn what will happen to their
        array.  If it says 'resize' and the code refuses, the plan is lying in
        the one direction that matters -- it promises a transformation that
        does not happen.
        """
        refusing = [s.name for s in preprocessing.plan('deepprofiler')]
        resizing = [s.name for s in preprocessing.plan('ca_mae')]

        assert 'check-size' in refusing and 'resize' not in refusing
        assert 'resize' in resizing and 'check-size' not in resizing

    def test_plan_reports_the_two_unrunnable_pipelines(self):
        assert [s.name for s in preprocessing.plan('cellclip')] == ['chain-required']
        assert [s.name for s in preprocessing.plan(source('morphogenie'))] == [
            'unsupported-preprocess'
        ]

    def test_plan_resize_step_is_backed_by_behaviour(self):
        """Guards the plan/behaviour drift that the mask bug came from.

        If the plan advertises `resize`, both the pixel path and the silhouette
        path must actually resize.
        """
        for name in ('ca_mae', 'o2vae', 'unidino', 'dinocells'):
            resolved = source(name)
            names = [step.name for step in preprocessing.plan(resolved)]
            if 'resize' not in names:
                continue
            spatial = resolved.spatial
            assert spatial is not None

            oversized = (spatial[0] * 2, spatial[1] * 2)
            n = 2
            mask = np.ones((n, *oversized), dtype='uint8')
            got = preprocessing.prepare(
                resolved, raw_crop(n=n, size=oversized[0]), masks=mask
            )
            # Every method that advertises a resize now consumes a channel
            # stack, so the spatial axes are the last two.
            assert got.shape[-2:] == tuple(spatial), (
                f'{name} advertises resize but produced {got.shape}'
            )


# ------------------------------------------------------- embed(assume=...) --

class TestAssumeRawCrop:
    """`embed(..., assume='raw_crop')` must be exactly `prepare` then `embed`."""

    def test_only_raw_crop_is_understood(self):
        from cellpaint_pipeline import models

        with pytest.raises(ValueError, match='raw_crop'):
            models.embed('deepprofiler', raw_crop(), assume='already_done')

    def test_refusals_happen_before_the_model_is_loaded(self):
        """A method that cannot eat pixels must say so without loading anything.

        This is what makes the error useful on a machine with no weights: the
        answer does not depend on the checkpoint being present.

        morphogenie used to be the second case here; it was cancelled, so asking
        for it by name now fails earlier, at resolution.  Both paths are checked
        because "not registered" and "registered but unable to preprocess" are
        different answers and a caller benefits from telling them apart.
        """
        from cellpaint_pipeline import models

        with pytest.raises(IncompatibleInput) as excinfo:
            models.embed('cellclip', raw_crop(), assume='raw_crop')
        assert getattr(excinfo.value, 'reason', None) == 'chain-required'

        with pytest.raises(UnknownModel):
            models.embed('morphogenie', raw_crop(), assume='raw_crop')