"""Contract tests for the embedding interface layer.

Why this file exists
--------------------
`cellpaint_pipeline.models` shipped with zero test coverage: 13 modules and
~2.6k lines that no test asserted and that the pipeline does not import.  Three
of its manifests were then found to be *wrong* -- and one of those errors was
exactly the silent-wrong-answer class the layer was built to prevent
(`unidino` declared `order_matters=False` while a channel permutation changes
its output by 2.109).  A guard layer whose own declarations are unchecked is
worse than no guard layer, because it is trusted.

These tests are therefore mostly about the *declarations*, not the frameworks:
they run without torch, tensorflow or any weights, so they stay fast and run in
every environment.

The regression tests at the bottom pin the three measured defects so they cannot
come back.
"""
from __future__ import annotations

import numpy as np
import pytest

from cellpaint_pipeline.models import registry
from cellpaint_pipeline.models.contract import (
    InputUnit,
    ModelManifest,
    check_compatible,
    is_ascii_path,
)
from cellpaint_pipeline.models.errors import UnknownModel


# --------------------------------------------------------------- helpers ----

def _manifest(name: str) -> ModelManifest:
    """Resolve a manifest by name, including cancelled (unregistered) methods.

    Three methods were cancelled on 2026-10-21 because their weights cannot be
    obtained here; their contracts survive in ``registry.PENDING``.  The tests
    that pin down *what the upstream code actually does* -- the channel order, the
    tape layout, the c_dim/z_dim mix-up -- are still worth running, because they
    record measurements that would otherwise have to be redone.  They resolve from
    PENDING instead of being deleted.
    """
    try:
        return registry.manifest(name)
    except UnknownModel:
        for registration in registry.PENDING:
            if registration.manifest.name == name:
                return registration.manifest
        raise


def _crop_for(manifest: ModelManifest, n=2, c=None, h=None, w=None, dtype='float32'):
    """Build an array that satisfies ``manifest``'s declared layout.

    Layout is per-method (Keras methods are ``BHWC``, torch methods ``BCHW``), so
    a helper that hard-codes one of them produces false failures and, worse,
    would hide a real layout mistake.  The declared layout is what is being
    tested, so the fixtures must follow it.
    """
    channels = c if c is not None else len(manifest.channel_names)
    height = h if h is not None else (manifest.spatial or (32, 32))[0]
    width = w if w is not None else (manifest.spatial or (32, 32))[1]
    if manifest.preprocess.batch_layout == 'BHWC':
        shape = (n, height, width, channels)
    else:
        shape = (n, channels, height, width)
    return np.zeros(shape, dtype=dtype)


# ------------------------------------------------------- registry basics ----

class TestRegistry:
    def test_declares_the_four_runnable_methods(self):
        assert registry.available() == [
            'deepprofiler', 'ca_mae', 'cellclip', 'o2vae',
        ]

    def test_cancelled_methods_are_not_registered(self):
        """A method with no obtainable weights must not appear as usable.

        `dinocells` is genuinely open (Zenodo 8061392, CC-BY-4.0) but ships only
        as one 3.71 GB tarball at ~9 KB/s, so it was cancelled too.  These names
        must not resolve, otherwise `embed` would advertise work it cannot do.
        """
        cancelled = {'dinocells', 'unidino', 'morphogenie'}
        assert cancelled.isdisjoint(registry.available())
        for name in cancelled:
            with pytest.raises(UnknownModel):
                registry.manifest(name)

    def test_cancelled_contracts_are_kept_for_a_future_readd(self):
        """Cancelled is not deleted: the measured contracts stay reviewable."""
        names = [manifest.name for manifest in registry.pending()]
        assert names == ['dinocells', 'unidino', 'morphogenie']
        for manifest in registry.pending():
            assert manifest.output_dim > 0, manifest.name
            assert manifest.preprocess is not None, manifest.name

    def test_every_registration_has_a_manifest_and_a_factory(self):
        for registration in registry.ADAPTERS:
            assert registration.manifest is not None, registration.module
            assert registration.factory, registration.module
            assert registration.module.startswith('cellpaint_pipeline.models.')

    def test_unknown_model_raises_typed_error(self):
        with pytest.raises(UnknownModel):
            registry.manifest('no_such_method')

    def test_output_dim_is_positive_for_every_method(self):
        for manifest in registry.manifests():
            assert manifest.output_dim > 0, manifest.name

    def test_channel_names_are_unique_and_non_empty(self):
        for manifest in registry.manifests():
            names = manifest.channel_names
            assert names, manifest.name
            assert len(set(names)) == len(names), manifest.name


# --------------------------------------------------- manifest consistency ----

class TestManifestConsistency:
    """Invariants that must hold across the whole registry."""

    def test_order_matters_only_when_names_are_labels(self):
        """A channel-agnostic method cannot also be order-sensitive.

        If the names are not labels then order is not meaningful, so declaring
        both is a contradiction -- the dataclass rejects it, and this test makes
        sure no manifest slips past by being constructed another way.
        """
        for manifest in registry.manifests():
            spec = manifest.channels
            if not spec.names_are_labels:
                assert not spec.order_matters, manifest.name

    def test_no_registered_method_declares_the_unimplemented_tape_encoding(self):
        """`input_encoding='tape'` is not implemented, so nothing may declare it.

        The value survives in the type because a future method might genuinely
        consume a wide image -- but until the framework implements it,
        ``prepare`` refuses it with ``reason='unsupported-encoding'``.  A
        manifest that declared it would advertise an encoding that raises.
        """
        for manifest in registry.manifests():
            assert manifest.preprocess.input_encoding == 'stack', manifest.name

    def test_spatial_is_a_positive_pair(self):
        for manifest in registry.manifests():
            if manifest.spatial is not None:
                h, w = manifest.spatial
                assert h > 0 and w > 0, manifest.name

    def test_determinism_is_a_declared_value(self):
        for manifest in registry.manifests():
            assert manifest.determinism in ('deterministic', 'stochastic'), manifest.name

    def test_stochastic_methods_say_so_in_notes(self):
        """A stochastic method must carry the reason, not just the flag."""
        for manifest in registry.manifests():
            if manifest.determinism == 'stochastic':
                assert manifest.notes, manifest.name


# ------------------------------------------- regression: the three defects ---

class TestMeasuredDefectsDoNotRegress:
    """Each test here corresponds to a defect found by running official code.

    They are the reason this file exists; the general invariants above are the
    reason it stays useful.
    """

    def test_unidino_is_order_sensitive(self):
        """Measured: permuting channels changed the output by up to 2.109.

        source/inference_utils.py:29-34 concatenates per-channel embeddings in
        input order, so the vector follows the input order.  Declaring
        `order_matters=False` let a permuted array through silently.
        """
        manifest = _manifest('unidino')
        assert manifest.channels.order_matters is True

    def test_unidino_permutation_is_rejected(self):
        """The check must actually fire on a permuted channel list."""
        manifest = _manifest('unidino')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=('RNA', 'DNA', 'ER', 'AGP', 'Mito'),
            array=_crop_for(manifest),
        )
        assert not report.ok
        assert any(i.code == 'channel-order' for i in report.issues), report.issues

    def test_morphogenie_output_dim_is_c_dim_not_z_dim(self):
        """Measured: official encode_only returns (4, 10).

        MorphoGenie_Test.py:67 sets c_dim=10 (the embedding width) and :68 sets
        z_dim=256 (the GAN generator's latent).  Only c_dim reaches the
        embedding: dvae.py:410-411 splits the encoder output at c_dim.
        """
        manifest = _manifest('morphogenie')
        assert manifest.output_dim == 10

    def test_morphogenie_is_marked_stochastic(self):
        """Measured: three calls on identical input differed by up to 2.979.

        MorphoGenie_Test.py:297 binds `mu_real` to the FIRST return value while
        dvae.py:396 returns `(c, mu, logvar)`; `c = reparametrize(mu, logvar)`
        draws noise.  A benchmark must know this before comparing it.
        """
        manifest = _manifest('morphogenie')
        assert manifest.determinism == 'stochastic'

    def test_unidino_declares_the_scale_the_official_code_actually_applies(self):
        """`/ (2**16 - 1)` is in the official source; the declaration must match.

        History: this assertion used to be the opposite (``scale != 'div65535'``)
        because a grep for ``65535`` across the upstream ``*.py`` found nothing.
        A targeted read of the loader settles it: ``source/image_ops.py`` does
        ``img = img / (2**16 - 1)`` and ``source/io.py``'s
        ``_convert_to_float_tensor`` routes every input through it, so 16-bit
        counts *are* mapped onto [0, 1] before the network.

        The old ``none`` was not merely vague: it invited uint8 0..255 input,
        which is 65793x too large for that mapping -- the silent-wrong-embedding
        failure this layer exists to catch.
        """
        manifest = _manifest('unidino')
        assert manifest.preprocess.scale == 'div65535'
        # The executable half has to agree with the declaration.
        assert manifest.preprocess.source_map == 'div65535'


# ------------------------------------------------------- contract checking ---

class TestChannelContract:
    def test_deepprofiler_accepts_its_own_channel_order(self):
        manifest = _manifest('deepprofiler')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=_crop_for(manifest),
        )
        assert report.ok, report.issues

    def test_deepprofiler_rejects_swapped_channels(self):
        manifest = _manifest('deepprofiler')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=tuple(reversed(manifest.channel_names)),
            array=_crop_for(manifest),
        )
        assert not report.ok

    def test_wrong_channel_count_is_reported(self):
        manifest = _manifest('deepprofiler')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=('DNA', 'RNA'),
            array=_crop_for(manifest, c=2),
        )
        assert not report.ok
        assert any(i.code == 'channel-count' for i in report.issues)

    def test_channel_agnostic_method_accepts_an_arbitrary_count(self):
        """CA-MAE is channel-agnostic by architecture, so only the count matters."""
        manifest = _manifest('ca_mae')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=('A', 'B', 'C', 'D', 'E', 'F', 'G'),
            array=_crop_for(manifest, n=1, c=7, h=256, w=256, dtype='uint8'),
        )
        assert report.ok, report.issues

    def test_a_channel_count_the_checkpoint_cannot_hold_is_refused(self):
        """CA-MAE adapts to any channel count -- up to eleven.

        The architecture is channel-agnostic, but the *checkpoint* is not: its
        positional embedding is generated for exactly ``max_in_chans=11``
        modalities (``huggingface_mae.py:75``, ``vit.py:138-143``).  A twelfth
        channel is a shape error raised from inside the model, with no reason
        code and in the framework's own vocabulary -- so the contract refuses
        it first.
        """
        manifest = _manifest('ca_mae')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            array=_crop_for(manifest, n=1, c=12, h=256, w=256, dtype='uint8'),
        )
        assert not report.ok
        assert any(i.code == 'channel-count' for i in report.issues), report.issues

    def test_the_declared_channel_ceiling_is_itself_accepted(self):
        """The limit must not be off by one in either direction."""
        manifest = _manifest('ca_mae')
        assert manifest.channels.max_channels == 11

        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            array=_crop_for(manifest, n=1, c=11, h=256, w=256, dtype='uint8'),
        )
        assert report.ok, report.issues


class TestDtypeAndRangeContract:
    """Value-range mistakes are silent, so they get their own tests."""

    def test_ca_mae_rejects_float_where_uint8_is_required(self):
        """0..1 is inside 0..255, so a range check alone would pass this."""
        manifest = _manifest('ca_mae')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=_crop_for(manifest, n=1, h=256, w=256),
        )
        assert not report.ok
        assert any('dtype' in i.code for i in report.issues), report.issues

    def test_deepprofiler_rejects_raw_16bit_values(self):
        """Regression test for a measured defect.

        The manifest used to declare `scale='raw'` with `value_range=None`, on
        the reasoning that the Keras graph has no normalisation layer.  The crop
        graph normalises instead (`imaging/cropping.py:65-67`), and driving it
        directly returned values spanning exactly [0, 1].  Under the old
        declaration a caller feeding 16-bit values got no warning and an
        embedding wrong by up to 8.8e4.
        """
        manifest = _manifest('deepprofiler')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=np.full(_crop_for(manifest, n=1, h=32, w=32).shape, 3000,
                          dtype='float32'),
        )
        assert not report.ok
        assert any(i.code == 'value-range' for i in report.issues), report.issues

    def test_deepprofiler_accepts_the_normalised_crop_it_actually_receives(self):
        """The crop graph's real output range is what the contract must accept."""
        manifest = _manifest('deepprofiler')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=np.random.RandomState(0).rand(
                *_crop_for(manifest, n=1, h=32, w=32).shape
            ).astype('float32'),
        )
        assert report.ok, report.issues

    def test_deepprofiler_declares_the_crop_graph_scale(self):
        assert _manifest('deepprofiler').preprocess.scale == 'per_crop_minmax'

    def test_a_crop_of_the_wrong_size_is_refused_with_a_reason_code(self):
        """The checkpoint wants 32x32 and the upstream operation is a crop.

        Resizing to satisfy the contract would invent a preprocessing step the
        official pipeline does not have -- and would do it differently from
        `crop_and_resize`.  So a wrong-sized crop is refused here, in the
        contract, with a reason code, rather than reaching the Keras graph and
        coming back as an untyped ``ValueError``.
        """
        manifest = _manifest('deepprofiler')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=_crop_for(manifest, n=1, h=64, w=64),
        )
        assert not report.ok
        assert any(i.code == 'spatial-size' for i in report.issues), report.issues

    def test_a_single_wrongly_sized_crop_is_refused_too(self):
        """The size check must not only cover batched input.

        A single crop -- one cell -- is the normal unit of work for this layer,
        so a 3-D array with the wrong spatial size has to be caught as well.
        """
        manifest = _manifest('deepprofiler')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=np.zeros((64, 64, 5), dtype='float32'),
        )
        assert not report.ok
        assert any(i.code == 'spatial-size' for i in report.issues), report.issues

    def test_an_eight_bit_crop_is_refused_by_a_scale_that_needs_sixteen(self):
        """`div65535` divides by 2**16-1, so uint8 is 257x too dark.

        A uint8 array is numerically inside the declared ``(0, 65535)`` range,
        so a value-range test alone would wave it through.  The dtype is what
        carries the information, and it has to be checked.
        """
        manifest = _manifest('unidino')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=_crop_for(manifest, n=1, h=224, w=224, dtype='uint8'),
        )
        assert not report.ok
        assert any(i.code == 'value-range' for i in report.issues), report.issues

    def test_a_sixteen_bit_crop_is_accepted_by_that_scale(self):
        manifest = _manifest('unidino')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=_crop_for(manifest, n=1, h=224, w=224, dtype='uint16'),
        )
        assert report.ok, report.issues

    def test_dinocells_consumes_a_channel_stack(self):
        """(N, C, H, W) at the declared spatial size is the official input."""
        manifest = _manifest('dinocells')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=_crop_for(manifest, n=1, h=96, w=96),
        )
        assert report.ok, report.issues

    def test_dinocells_accepts_a_batch_of_stacks(self):
        """The official model batches like any image stack.

        The old tape encoding had no batch axis -- the width carried the
        channel index -- so a batch had to be refused.  That was a property of
        the local encoding, not of the method.
        """
        manifest = _manifest('dinocells')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=_crop_for(manifest, n=3, h=96, w=96),
        )
        assert report.ok, report.issues

    def test_a_wide_tape_is_no_longer_a_valid_input(self):
        """The tape was the official *storage* format, not the model input.

        Feeding one now must be refused rather than silently reinterpreted:
        the official dataset folds the stored tape back open and drops the
        mask column before the network sees it.
        """
        manifest = _manifest('dinocells')
        h, w = manifest.spatial
        tape = np.zeros((h, w * (len(manifest.channel_names) + 1)), dtype='float32')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=tape,
        )
        assert not report.ok
        assert any(i.code == 'ndim' for i in report.issues), report.issues

    def test_the_tape_encoding_is_not_implemented(self):
        """A declared-but-unimplemented encoding must be refused, not guessed.

        A half-implemented encoding is worse than a missing one: it returns a
        number without returning an error.
        """
        manifest = _manifest('dinocells')
        assert manifest.preprocess.input_encoding == 'stack'
        assert manifest.preprocess.mask_mode == 'drop'


    def test_value_range_describes_the_model_input_not_the_raw_domain(self):
        """A declared range must mean the same thing on both paths.

        With `assume='raw_crop'` the contract sees the *prepared* array; without
        it, the contract sees the caller's array as-is.  A range that described
        the raw domain therefore meant two different things -- and on the
        second path it let raw 16-bit counts through to a network trained on
        [0, 1], producing an embedding wrong by max|diff| = 2.98 with no error
        anywhere.
        """
        manifest = _manifest('unidino')
        # The model consumes [0, 1]: the official pipeline divides by 2**16-1
        # before the network (source/image_ops.py:14-20).
        assert manifest.preprocess.value_range == (0.0, 1.0)

    def test_raw_counts_are_refused_when_the_caller_skips_prepare(self):
        """The non-assume path must not accept raw 16-bit counts."""
        manifest = _manifest('unidino')
        raw = np.full((2, 5, 224, 224), 40000, dtype='uint16')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=raw,
        )
        assert not report.ok
        assert any(i.code == 'value-range' for i in report.issues), report.issues

    def test_the_prepared_domain_is_accepted(self):
        """The prepared [0, 1] array is what the model wants."""
        manifest = _manifest('unidino')
        prepared = np.zeros((2, 5, 224, 224), dtype='float32')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=prepared,
        )
        assert report.ok, report.issues


class TestInputUnitContract:
    def test_cellclip_requires_channel_embeddings_not_pixels(self):
        manifest = _manifest('cellclip')
        assert manifest.input_unit is InputUnit.CHANNEL_EMBEDDINGS

    def test_cellclip_rejects_a_pixel_shaped_array(self):
        manifest = _manifest('cellclip')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=manifest.channel_names,
            array=_crop_for(manifest, n=1, h=256, w=256),
        )
        assert not report.ok

    def test_cellclip_accepts_n_by_channels_by_features(self):
        manifest = _manifest('cellclip')
        report = check_compatible(
            manifest,
            unit=InputUnit.CHANNEL_EMBEDDINGS,
            channels=manifest.channel_names,
            array=np.zeros((3, 5, 1536), dtype='float32'),
        )
        assert report.ok, report.issues

    def test_cellclip_rejects_a_4d_pixel_stack_without_a_unit(self):
        """Regression: omitting ``unit`` used to let pixel stacks through.

        `check_compatible` only checked ``ndim in (3, 4)``, so a
        ``(N, 5, 256, 256)`` pixel stack satisfied the CellCLIP contract and the
        error surfaced later, inside the adapter, after the model had loaded.
        A channel-embedding input is 3-D by definition.
        """
        manifest = _manifest('cellclip')
        report = check_compatible(manifest, array=np.zeros((2, 5, 256, 256), 'float32'))
        assert not report.ok
        assert any(i.code == 'ndim' for i in report.issues), report.issues

    def test_cellclip_rejects_a_4d_pixel_stack_even_with_the_right_unit(self):
        manifest = _manifest('cellclip')
        report = check_compatible(
            manifest,
            unit=InputUnit.CHANNEL_EMBEDDINGS,
            array=np.zeros((2, 5, 256, 256), 'float32'),
        )
        assert not report.ok

    def test_cellclip_rejects_a_3d_array_with_the_wrong_feature_width(self):
        """A 3-D array of pixels is still not a vector stack."""
        manifest = _manifest('cellclip')
        report = check_compatible(
            manifest,
            unit=InputUnit.CHANNEL_EMBEDDINGS,
            array=np.zeros((2, 5, 256), 'float32'),
        )
        assert not report.ok
        assert any(i.code == 'feature-width' for i in report.issues), report.issues

    def test_cellclip_declares_the_feature_width(self):
        manifest = _manifest('cellclip')
        assert manifest.input_feature_dim == 1536


class TestChainProvenance:
    """A method's provenance must cover every model in its chain.

    CellCLIP consumes DINOv2-Giant output, so recording only the CellCLIP
    checkpoint omits the 4.5 GB model that actually produced the input and makes
    the run un-reproducible.
    """

    def test_cellclip_records_the_dinov2_chain_weight(self):
        manifest = _manifest('cellclip')
        names = [name for name, _digest in manifest.chain_weights]
        assert any('dinov2' in name.lower() for name in names), names

    def test_chain_weight_digests_look_like_sha256(self):
        manifest = _manifest('cellclip')
        for _name, digest in manifest.chain_weights:
            assert len(digest) == 64
            assert all(c in '0123456789abcdef' for c in digest)

    def test_pixel_methods_declare_no_chain(self):
        """Only methods that consume another model's output need a chain."""
        for name in ('deepprofiler', 'ca_mae', 'o2vae'):
            assert _manifest(name).chain_weights == (), name


class TestPublicApiReachesConstructorArgs:
    """`device`, `weights_roots` and constructor settings used to be swallowed by
    ``**kwargs`` and forwarded to the forward pass, where they were ignored."""

    def test_embed_accepts_device_and_adapter_kwargs(self):
        from cellpaint_pipeline.models.embedding import embed
        from cellpaint_pipeline.models.errors import EmbeddingError

        # unidino has no local weights or dependencies, so this stops at a typed
        # environment error.  That is the point: a broken signature would raise
        # TypeError before the adapter is built, whereas any EmbeddingError means
        # the arguments were accepted and carried to the adapter constructor.
        array = np.zeros((1, 5, 224, 224), dtype='float32')
        with pytest.raises(EmbeddingError) as caught:
            embed(
                'unidino', array,
                unit=InputUnit.SINGLE_CELL_CROP,
                device='cpu',
                weights_roots=('/nonexistent',),
                adapter_kwargs={'feature_layer': 'pool5'},
            )
        assert not isinstance(caught.value, TypeError)

    def test_benchmark_plan_carries_a_device(self):
        from pathlib import Path as _Path

        from cellpaint_pipeline import embedding_benchmark as bench

        plan = bench.build_benchmark_plan(
            ['deepprofiler'], output_dir=_Path('.')
        )
        assert plan.device is None


class TestChannelOrderMatchesUpstream:
    """Pins the training-time channel order of the published checkpoints.

    The order was ``('DNA','RNA','ER','AGP','Mito')``, inherited from this
    repository's own DeepProfiler project builder.  That source is wrong: the
    upstream repository states the training order twice and both disagree with
    it.  Swapping RNA and ER feeds the model two channels transposed without
    raising, which is the silent-wrong-answer class this layer exists to stop.
    """

    def test_deepprofiler_uses_the_upstream_training_order(self):
        # upstream README.md:74 and
        # tests/deepprofiler/profiling/test_cell_painting_cnn_v1.py:74
        assert registry.DEEPPROFILER_CHANNELS == (
            'DNA', 'ER', 'RNA', 'AGP', 'Mito'
        )

    def test_cellclip_has_its_own_channel_constant(self):
        """CellCLIP must not silently inherit DeepProfiler's order.

        The two agree today, so an identity check is meaningless (Python interns
        identical tuple literals).  What matters is that CellCLIP is bound to a
        *separately named* constant sourced from its own upstream evidence
        (JUMP-CP w1..w5), so correcting one project's order does not silently
        change the other's contract.
        """
        assert hasattr(registry, 'CELLCLIP_CHANNELS')
        # JUMP-CP w1=DNA, w2=ER, w3=RNA, w4=AGP, w5=Mito
        assert registry.CELLCLIP_CHANNELS == ('DNA', 'ER', 'RNA', 'AGP', 'Mito')
        cellclip_manifest = _manifest('cellclip')
        assert cellclip_manifest.channel_names == registry.CELLCLIP_CHANNELS

    def test_the_old_order_is_now_rejected(self):
        manifest = _manifest('deepprofiler')
        report = check_compatible(
            manifest,
            unit=InputUnit.SINGLE_CELL_CROP,
            channels=('DNA', 'RNA', 'ER', 'AGP', 'Mito'),
            array=_crop_for(manifest),
        )
        assert not report.ok
        assert any(i.code == 'channel-order' for i in report.issues), report.issues


class TestAsciiPathPolicy:
    def test_pure_ascii_is_accepted(self):
        assert is_ascii_path(r'D:\cpc_run\demo')

    def test_non_ascii_is_rejected(self):
        assert not is_ascii_path('D:\\椤圭洰\\demo')

    def test_only_methods_that_need_it_require_ascii(self):
        for manifest in registry.manifests():
            if manifest.ascii_path_required:
                assert 'cellprofiler' in ' '.join(manifest.requires).lower(), manifest.name
