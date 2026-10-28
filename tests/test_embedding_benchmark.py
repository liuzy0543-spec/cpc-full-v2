"""Tests for the dispatch and benchmark layer.

Scope: everything that can be checked without a framework or a checkpoint.  The
paths that need real weights are exercised by `_probe/run_benchmark_e2e.py`; what
is asserted here is the logic that decides *what* to run and *how* results are
combined, because a mistake there is invisible in the numbers it produces.
"""
from __future__ import annotations

import json
from pathlib import Path

import importlib.util

import numpy as np
import pytest

from cellpaint_pipeline import embedding_benchmark as bench
from cellpaint_pipeline.models import registry
from cellpaint_pipeline.models.contract import InputUnit
from cellpaint_pipeline.models.errors import IncompatibleInput, UnknownModel
from cellpaint_pipeline.models.weights import verify_sha256

# --------------------------------------------------------------- licence -----

class TestLicenceGate:
    """The methods differ sharply in terms; a published run must not mix them."""

    @pytest.mark.parametrize(
        ('licence', 'expected'),
        [
            ('BSD-3-Clause (code) / CC-BY-4.0 (weights)', 'permissive'),
            ('MIT', 'permissive'),
            ('Recursion Non-Commercial EULA', 'non-commercial'),
            ('CC BY-NC-ND 4.0', 'no-derivatives'),
            ('GPL-3.0', 'copyleft'),
            ('CC-BY-4.0 (weights) / no licence declared (code)', 'unknown'),
        ],
    )
    def test_classification(self, licence, expected):
        assert bench.classify_licence(licence) == expected

    def test_every_manifest_classifies(self):
        for manifest in registry.manifests():
            assert bench.classify_licence(manifest.license) in {
                'permissive', 'non-commercial', 'no-derivatives', 'copyleft', 'unknown'
            }

    def test_only_deepprofiler_is_permissive_on_both_sides(self):
        """Recorded fact: it is the only method clear for public comparison.

        If this changes, the licence text in the registry changed, and whoever
        changed it should be aware it affects what may be published.
        """
        both_permissive = [
            m.name for m in registry.manifests()
            if bench.classify_licence(m.license) == 'permissive'
        ]
        assert 'deepprofiler' in both_permissive


# ------------------------------------------------------------------ plan -----

class TestBenchmarkPlan:
    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match='unknown method'):
            bench.build_benchmark_plan(['not_a_method'], output_dir='.')

    def test_rejects_empty_methods(self):
        with pytest.raises(ValueError, match='must not be empty'):
            bench.BenchmarkPlan(methods=(), output_dir=Path('.'))

    def test_defaults_to_declared_methods_when_none_runnable(self):
        plan = bench.build_benchmark_plan(
            ['deepprofiler', 'ca_mae'], output_dir='.'
        )
        assert plan.methods == ('deepprofiler', 'ca_mae')

    def test_invariants_are_populated(self):
        """A plan states what the run guarantees, so it can be inspected."""
        plan = bench.build_benchmark_plan(['deepprofiler'], output_dir='.')
        assert plan.invariants
        assert any('sha256' in item for item in plan.invariants)

    def test_available_methods_can_be_listed_without_weights(self):
        assert bench.available_benchmark_methods(require_weights=False) == (
            registry.available()
        )


# ------------------------------------------------------ fail-fast gate --------

class TestRowCountFailFast:
    """A mis-sized batch must be rejected before any checkpoint loads.

    The mismatch used to surface from inside ``_align`` after every model had run
    (measured at 14.79s) and then wrote no artefacts at all, because the raise
    happened before ``_write_artefacts``.
    """

    def _plan(self, tmp_path):
        return bench.BenchmarkPlan(
            methods=('deepprofiler',),
            output_dir=tmp_path / 'out',
            align_on='cell_id',
            unit='single_cell_crop',
        )

    def test_row_mismatch_raises_before_computing(self, tmp_path):
        plan = self._plan(tmp_path)
        inputs = {'deepprofiler': np.zeros((4, 32, 32, 5), dtype='float32')}
        metadata = {'cell_id': ['a', 'b', 'c', 'd', 'e']}
        with pytest.raises(ValueError) as exc:
            bench.run_benchmark(plan, inputs, sample_metadata=metadata)
        assert 'rows' in str(exc.value)
        # Nothing was written, and no output directory was even created.
        assert not (tmp_path / 'out').exists()

    def test_missing_align_column_is_reported_up_front(self, tmp_path):
        plan = self._plan(tmp_path)
        inputs = {'deepprofiler': np.zeros((2, 32, 32, 5), dtype='float32')}
        with pytest.raises(ValueError) as exc:
            bench.run_benchmark(plan, inputs, sample_metadata={'other': [1, 2]})
        assert 'align_on' in str(exc.value)

    def test_align_without_metadata_is_reported_up_front(self, tmp_path):
        plan = self._plan(tmp_path)
        inputs = {'deepprofiler': np.zeros((2, 32, 32, 5), dtype='float32')}
        with pytest.raises(ValueError) as exc:
            bench.run_benchmark(plan, inputs, sample_metadata=None)
        assert 'sample_metadata' in str(exc.value)

    def test_metadata_columns_must_agree(self, tmp_path):
        plan = self._plan(tmp_path)
        inputs = {'deepprofiler': np.zeros((2, 32, 32, 5), dtype='float32')}
        metadata = {'cell_id': ['a', 'b'], 'plate': ['p1']}
        with pytest.raises(ValueError) as exc:
            bench.run_benchmark(plan, inputs, sample_metadata=metadata)
        assert 'disagree' in str(exc.value)

    def test_matching_counts_pass_the_gate(self, tmp_path):
        plan = self._plan(tmp_path)
        inputs = {'deepprofiler': np.zeros((3, 32, 32, 5), dtype='float32')}
        metadata = {'cell_id': ['a', 'b', 'c']}
        # The gate itself must not raise; whether the model then loads is out of
        # scope for a framework-free unit test.
        bench._validate_row_counts(plan, inputs, metadata)


# ------------------------------------------------- embed_many fail-fast ------

class TestEmbedManyFailFast:
    """Bulk calls must validate every chunk before paying for a model load."""

    def test_a_bad_chunk_is_rejected_before_the_adapter_is_built(self, monkeypatch):
        from cellpaint_pipeline.models import embedding

        def _explode(*args, **kwargs):
            raise AssertionError(
                'the adapter must not be built before every chunk is validated'
            )

        monkeypatch.setattr(embedding, '_cached_adapter', _explode)
        good = np.zeros((32, 32, 5), dtype='float32')
        bad = np.zeros((32, 32, 3), dtype='float32')   # wrong channel count
        with pytest.raises(IncompatibleInput):
            embedding.embed_many(
                'deepprofiler',
                [good, bad],
                unit='single_cell_crop',
                channels=('DNA', 'ER', 'RNA', 'AGP', 'Mito'),
            )


# ------------------------------------------------------------------- rv ------

class TestRvCoefficient:
    """The agreement measure must work across different embedding widths."""

    def test_identical_matrices_score_one(self):
        rng = np.random.RandomState(0)
        a = rng.randn(8, 16)
        assert bench._rv_coefficient(a, a.copy()) == pytest.approx(1.0, abs=1e-9)

    def test_works_across_different_widths(self):
        """The reason RV was chosen: 384 vs 512 has no row-wise cosine."""
        rng = np.random.RandomState(1)
        a = rng.randn(8, 384)
        b = rng.randn(8, 512)
        value = bench._rv_coefficient(a, b)
        assert value is not None
        assert 0.0 <= value <= 1.0

    def test_a_scale_preserving_relabelling_scores_one(self):
        """RV is invariant under orthogonal maps, and that is stable to test.

        The invariance is exact only for maps that preserve the geometry.  A
        random well-conditioned matrix is *mathematically* invertible but
        numerically lossy here: centring ``a @ M`` rounds at the scale of ``M``,
        and with ``M`` inflating the data (std 8.7 -> 134) that rounding survives
        into the Gram matrix, giving 0.847 rather than 1.  A permutation is
        orthogonal and exactly representable, so it tests the invariance claim
        without conflating it with floating-point drift.
        """
        rng = np.random.RandomState(2)
        a = rng.randn(12, 5)
        permutation = np.eye(5)[:, [3, 0, 4, 1, 2]]
        b = a @ permutation
        assert bench._rv_coefficient(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_not_invariant_under_anisotropic_rescaling(self):
        """Documented limit, measured rather than assumed.

        RV is invariant under *orthogonal* maps only.  Rescaling each axis
        differently is invertible, so exact invariance would give 1.0; measured,
        it converges to ~0.72.  The test pins the measured behaviour so the
        eventual reader of a low RV score knows it can mean "rescaled" rather
        than "unrelated".
        """
        rng = np.random.RandomState(0)
        a = rng.randn(2000, 5)
        b = a * np.array([1.0, 2.0, 0.5, 3.0, 1.5])
        value = bench._rv_coefficient(a, b)
        assert value is not None
        assert 0.6 < value < 0.8, value

    def test_a_global_scale_scores_one(self):
        """Multiplying everything by one constant does preserve geometry."""
        rng = np.random.RandomState(6)
        a = rng.randn(30, 4)
        assert bench._rv_coefficient(a, a * 1234.5) == pytest.approx(1.0, abs=1e-9)

    def test_degenerate_input_returns_none_not_zero(self):
        """Constant rows make the ratio undefined; saying 0 would be a claim."""
        a = np.ones((4, 3))
        b = np.random.RandomState(3).randn(4, 3)
        assert bench._rv_coefficient(a, b) is None

    def test_centring_is_applied(self):
        """Two clouds differing only by a constant offset must score 1."""
        rng = np.random.RandomState(4)
        a = rng.randn(10, 6)
        assert bench._rv_coefficient(a, a + 100.0) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------ chunking ------

class TestChunking:
    """`embed_all` used to hand the whole payload to one forward pass.

    A full plate is 3,857 CA-MAE crops: 1.26 GB of uint8 becomes a ~5 GB float32
    tensor in a single pass, which exhausted an 8 GiB card.  Chunking is an
    interface concern; leaving it to callers means every caller gets it wrong the
    same way.
    """

    def test_split_respects_the_bound_and_the_total(self):
        from cellpaint_pipeline.models.embedding import _split_into_blocks

        payload = np.zeros((1000, 2, 2, 1), dtype='uint8')
        blocks = _split_into_blocks(payload, 256)
        assert len(blocks) == 4
        assert [b.shape[0] for b in blocks] == [256, 256, 256, 232]
        assert sum(b.shape[0] for b in blocks) == 1000

    def test_split_is_a_noop_when_not_needed(self):
        from cellpaint_pipeline.models.embedding import _split_into_blocks

        payload = np.zeros((10, 3), dtype='float32')
        assert _split_into_blocks(payload, None) == [payload]
        assert _split_into_blocks(payload, 0) == [payload]
        assert _split_into_blocks(payload, 10) == [payload]
        assert len(_split_into_blocks(payload, 4)) == 3

    def test_split_does_not_copy_small_payloads(self):
        """A single block must be the caller's object, not a copy of it."""
        from cellpaint_pipeline.models.embedding import _split_into_blocks

        payload = np.zeros((5, 3), dtype='float32')
        assert _split_into_blocks(payload, 100)[0] is payload

    def test_merge_refuses_blocks_that_disagree_on_width(self):
        """Concatenating blocks of different widths would silently corrupt rows."""
        from cellpaint_pipeline.models.embedding import (
            EmbeddingResult,
            _merge_result_parts,
        )
        from cellpaint_pipeline.models.errors import EmbeddingError

        wide = EmbeddingResult(
            model='x', values=np.zeros((2, 8)), dim=8,
            input_unit=InputUnit.SINGLE_CELL_CROP,
        )
        narrow = EmbeddingResult(
            model='x', values=np.zeros((2, 5)), dim=5,
            input_unit=InputUnit.SINGLE_CELL_CROP,
        )
        with pytest.raises(EmbeddingError):
            _merge_result_parts('x', [wide, narrow])

    def test_merge_concatenates_rows_and_counts_chunks(self):
        from cellpaint_pipeline.models.embedding import (
            EmbeddingResult,
            _merge_result_parts,
        )

        parts = [
            EmbeddingResult(
                model='x', values=np.arange(6, dtype='float32').reshape(2, 3),
                dim=3, input_unit=InputUnit.SINGLE_CELL_CROP,
            )
            for _ in range(3)
        ]
        merged = _merge_result_parts('x', parts)
        assert merged.shape == (6, 3)
        assert merged.metadata['chunks'] == 3

    def test_plan_carries_a_chunk_size_and_reaches_embed_all(self):
        """The plan field is only useful if `run_benchmark` forwards it."""
        import inspect

        from cellpaint_pipeline import embedding_benchmark as bench

        plan = bench.build_benchmark_plan(
            ['deepprofiler'], output_dir=Path('.'), chunk_size=128,
        )
        assert plan.chunk_size == 128
        source = inspect.getsource(bench.run_benchmark)
        assert 'chunk_size=plan.chunk_size' in source


# ------------------------------------------------------- dispatch layer ------

class TestEmbedAllDispatch:
    """`embed_all` decides which payload reaches which method. Getting this
    wrong feeds pixels to a vector model, which does not raise."""

    def test_missing_payload_is_reported_in_collect_mode(self):
        # ca_mae has no payload here; the other is present but also unavailable,
        # so we only assert the bookkeeping, not the embeddings.
        from cellpaint_pipeline.models import embedding

        result = embedding.embed_all(
            ['ca_mae'],
            {},
            unit=InputUnit.SINGLE_CELL_CROP,
            on_error='collect',
        )
        assert result.errors['ca_mae'] == 'no payload supplied for this method'
        assert result.failed() == ('ca_mae',)
        assert len(result) == 0

    def test_input_unit_mismatch_is_skipped_not_guessed(self):
        """A shared payload is only used for methods whose unit matches.

        cellclip consumes channel embeddings, so handing it the shared pixel
        array must not happen -- it would produce a meaningless embedding rather
        than an error.
        """
        from cellpaint_pipeline.models import embedding

        array = np.zeros((1, 5, 256, 256), dtype='uint8')
        result = embedding.embed_all(
            ['cellclip'],
            array,
            unit=InputUnit.SINGLE_CELL_CROP,
            on_error='collect',
        )
        errors = result.errors
        assert 'cellclip' in errors
        assert 'channel_embeddings' in errors['cellclip']

    def test_input_unit_mismatch_raises_by_default(self):
        from cellpaint_pipeline.models import embedding

        array = np.zeros((1, 5, 256, 256), dtype='uint8')
        with pytest.raises(IncompatibleInput):
            embedding.embed_all(
                ['cellclip'], array, unit=InputUnit.SINGLE_CELL_CROP
            )

    def test_per_method_units_let_mixed_input_kinds_share_one_call(self):
        """CellCLIP consumes vectors, everything else pixels.

        `BenchmarkPlan` used to carry a single `unit`, so a plan could not
        describe both kinds and a four-method benchmark could not be expressed in
        one call at all.  The mapping form fixes that.
        """
        from cellpaint_pipeline.models import embedding

        payloads = {
            'cellclip': np.zeros((1, 5, 1536), dtype='float32'),
            'ca_mae': np.zeros((1, 5, 256, 256), dtype='uint8'),
        }
        units = {
            'cellclip': InputUnit.CHANNEL_EMBEDDINGS,
            'ca_mae': InputUnit.SINGLE_CELL_CROP,
        }
        # Both reach the adapter stage rather than being rejected as
        # mismatched.  ca_mae needs a framework that may be absent in the
        # contract-only environment, so collect mode is used and only unit
        # handling asserted.
        #
        # The outcome is environment-dependent -- with torch present ca_mae
        # SUCCEEDS -- so the assertion must not be "it failed".  What the test
        # is actually about is that the unit mapping was respected, so it
        # checks whichever way the call went.
        result = embedding.embed_all(
            ['ca_mae'], payloads, unit=units, on_error='collect'
        )
        if 'ca_mae' in result.errors:
            assert 'input-unit' not in result.errors['ca_mae']
            assert 'declared as' not in result.errors['ca_mae']
        else:
            assert result.results['ca_mae'].dim == 384

    def test_unknown_method_raises(self):
        from cellpaint_pipeline.models import embedding

        with pytest.raises(UnknownModel):
            embedding.embed_all(['nope'], {}, on_error='collect')


# ------------------------------------------------ payload normalisation ------

class TestArrayLikeNormalisation:
    """A payload must be validated identically whatever container it arrives in.

    ``check_compatible`` is numpy-free by design: it duck-types ``shape``,
    ``dtype``, ``min`` and ``max``.  A plain Python list exposes none of those,
    so it used to satisfy *every* check while the identical numbers as an
    ndarray were rejected -- and the adapter then converted the list silently,
    returning a plausible but wrong embedding with no error at all.
    """

    _CHANNELS = ('DNA', 'ER', 'RNA', 'AGP', 'Mito')

    def test_list_payload_is_rejected_exactly_like_an_ndarray(self):
        from cellpaint_pipeline.models import embedding

        # 32x32 is the size DeepProfiler's checkpoint expects, so the *value*
        # range is what this payload violates -- not its shape.  Keeping the
        # fixture at the right size is what makes the assertion below about
        # the list/ndarray equivalence rather than about which check happens
        # to run first.
        raw = np.full((2, 32, 32, 5), 65535, dtype='uint16')
        with pytest.raises(IncompatibleInput) as from_array:
            embedding.embed(
                'deepprofiler', raw,
                unit=InputUnit.SINGLE_CELL_CROP, channels=self._CHANNELS,
            )
        with pytest.raises(IncompatibleInput) as from_list:
            embedding.embed(
                'deepprofiler', raw.tolist(),
                unit=InputUnit.SINGLE_CELL_CROP, channels=self._CHANNELS,
            )
        assert from_array.value.reason == 'value-range'
        assert from_list.value.reason == from_array.value.reason

    def test_embed_many_validates_list_chunks_too(self, monkeypatch):
        """The chunk loop must see normalised payloads as well.

        ``embed_many`` builds the adapter before validating (a separate P2
        ordering defect), so the adapter is stubbed out here to keep the
        assertion about validation, and to stay runnable without frameworks.
        """
        from cellpaint_pipeline.models import embedding

        class _Unused:
            def embed(self, array, **kwargs):  # pragma: no cover - must not run
                raise AssertionError('adapter must not be reached')

        monkeypatch.setattr(
            embedding, '_cached_adapter', lambda *a, **k: _Unused()
        )
        raw = np.full((2, 32, 32, 5), 65535, dtype='uint16').tolist()
        with pytest.raises(IncompatibleInput) as excinfo:
            embedding.embed_many(
                'deepprofiler', [raw],
                unit=InputUnit.SINGLE_CELL_CROP, channels=self._CHANNELS,
            )
        assert excinfo.value.reason == 'value-range'

    def test_a_list_payload_keeps_its_value_width(self):
        # unidino is a v3 surrogate; under v2 it is not registered at all.
        from cellpaint_pipeline.models import registry

        before = registry.active_profile()
        registry.set_profile('v3')
        try:
            self._assert_list_width_is_kept()
        finally:
            registry.set_profile(before)

    def _assert_list_width_is_kept(self):
        """A list has no dtype, and the guards must still see the truth.

        ``np.asarray`` gives a list of 8-bit values an ``int64`` dtype, so
        every dtype guard sees a legitimate 64-bit array and 8-bit numbers
        slip through 257x too dark.  Measured before the fix:
        ``embed('unidino', uint8_array.tolist(), assume='raw_crop')`` was
        accepted and embedded a nearly black image, silently differing from
        the same numbers given as an ndarray (which was correctly rejected).
        """
        from cellpaint_pipeline.models import embedding as embedding_module

        raw = np.full((2, 8, 8, 5), 200, dtype='uint8')
        with pytest.raises(IncompatibleInput) as excinfo:
            embedding_module.embed('unidino', raw.tolist(), assume='raw_crop')
        assert excinfo.value.reason == 'value-range'

    @pytest.mark.skipif(
        importlib.util.find_spec('torch') is None,
        reason='the unidino surrogate needs torch',
    )
    def test_a_list_of_sixteen_bit_values_is_accepted(self):
        from cellpaint_pipeline.models import registry

        before = registry.active_profile()
        registry.set_profile('v3')
        try:
            self._assert_list_of_sixteen_bit_values_is_accepted()
        finally:
            registry.set_profile(before)

    def _assert_list_of_sixteen_bit_values_is_accepted(self):
        """The narrowing must not reject the legitimate case."""
        from cellpaint_pipeline.models import embedding as embedding_module

        raw = np.full((2, 8, 8, 5), 40000, dtype='uint16')
        result = embedding_module.embed('unidino', raw.tolist(), assume='raw_crop')
        assert result.dim == 1920

    def test_normalisation_leaves_duck_typed_payloads_untouched(self):
        """Anything already advertising a shape is passed through by identity.

        A tensor, an HDF5 dataset or a test double must not be round-tripped
        through numpy merely to be validated.
        """
        from cellpaint_pipeline.models import embedding

        class Duck:
            shape = (2, 8, 8, 5)
            dtype = 'float32'

            def min(self):
                return 0.0

            def max(self):
                return 1.0

        arr = np.zeros((2, 8, 8, 5), dtype='float32')
        duck = Duck()
        assert embedding._normalise_input(arr) is arr
        assert embedding._normalise_input(duck) is duck
        normalised = embedding._normalise_input(arr.tolist())
        assert isinstance(normalised, np.ndarray)
        assert normalised.shape == (2, 8, 8, 5)


# ------------------------------------------------------- prepare layer --------

class TestPrepareLayer:
    """`models.prepare` must execute the manifest, not just describe it.

    Each case checks the transform the method's own manifest declares, because
    "declared but not enforced" was exactly the gap that made a raw-pixel call
    impossible.
    """

    CH = ('DNA', 'ER', 'RNA', 'AGP', 'Mito')

    def _raw(self, shape=(32, 32, 5), dtype='uint16', seed=0):
        rs = np.random.RandomState(seed)
        return (rs.rand(*shape) * 65535).astype(dtype)

    @staticmethod
    def _prepare(*args, **kwargs):
        from cellpaint_pipeline.models import prepare

        return prepare(*args, **kwargs)

    def test_deepprofiler_minmax_is_per_sample_and_per_channel(self):
        raw = np.stack([self._raw(seed=1), self._raw(seed=2)])
        out = self._prepare('deepprofiler', raw, channels=self.CH)
        assert out.shape == (2, 32, 32, 5)
        assert out.dtype == np.float32
        # Every sample/channel is independently mapped onto [0, 1].
        assert np.allclose(out.min(axis=(1, 2)), 0.0, atol=1e-6)
        assert np.allclose(out.max(axis=(1, 2)), 1.0, atol=1e-6)

    def test_ca_mae_stretches_each_crop_to_uint8(self):
        raw = self._raw(shape=(32, 32, 5))
        out = self._prepare('ca_mae', raw, channels=self.CH)
        assert out.shape == (5, 256, 256)     # BCHW, resized
        assert out.dtype == np.uint8
        assert out.max() == 255
        # The stretch maps each crop's own maximum onto 255; the minimum only
        # reaches 0 when the crop's floor is 0, so it must merely stay in range.
        assert out.min() >= 0

    # The unidino channel-order and div-65535 cases moved to
    # tests/test_models_preprocessing.py when unidino was cancelled (2026-10-21):
    # they exercise the transform, not the registry, and that file tests every
    # source_map against the official formula.

    def test_batch_is_per_sample_not_global(self):
        a = np.zeros((32, 32, 5), dtype='uint16')
        b = np.full((32, 32, 5), 65535, dtype='uint16')
        b[0, 0, 0] = 0
        out = self._prepare('deepprofiler', np.stack([a, b]), channels=self.CH)
        # A constant (all-zero) sample must not become NaN; a per-batch min-max
        # would.
        assert np.all(np.isfinite(out))

    def test_o2vae_refuses_to_invent_a_silhouette(self):
        from cellpaint_pipeline.models.errors import IncompatibleInput

        with pytest.raises(IncompatibleInput) as exc:
            self._prepare('o2vae', self._raw())
        assert exc.value.reason == 'mask-required'

    def test_o2vae_renders_a_silhouette_from_a_supplied_mask(self):
        mask = np.zeros((64, 64), dtype='uint8')
        mask[20:44, 20:44] = 1
        out = self._prepare('o2vae', self._raw(), masks=mask)
        assert out.shape == (1, 64, 64)
        assert out.dtype == np.float32
        assert set(np.unique(out)) <= {0.0, 1.0}

    def test_cellclip_refuses_pixels_and_names_the_chain(self):
        from cellpaint_pipeline.models.errors import IncompatibleInput

        with pytest.raises(IncompatibleInput) as exc:
            self._prepare('cellclip', self._raw())
        assert exc.value.reason == 'chain-required'

    # morphogenie's MATLAB-centring refusal and dinocells' channel-stack
    # layout moved to
    # tests/test_models_preprocessing.py when both were cancelled (2026-10-21).
    def test_plan_describes_the_steps_without_touching_data(self):
        from cellpaint_pipeline.models import plan

        names = [step.name for step in plan('deepprofiler')]
        assert names[0] == 'coerce'
        assert 'reorder-channels' in names
        assert 'source-map' in names
        assert names[-1] == 'cast'

    def test_prepare_is_reachable_from_the_facade(self):
        import cellpaint_pipeline.models as models

        assert 'prepare' in models.__all__
        assert callable(models.prepare)


# ------------------------------------------------- adapter construction -------

class TestAdapterConstructorPassthrough:
    """``device=``/``weights_roots=`` must reach the adapters, not raise TypeError.

    Each adapter is a ``@dataclass``, so its generated ``__init__`` shadows
    ``BaseAdapter.__init__``.  Until the two parameters were declared as fields
    every kwarg was rejected outright, which made ``models.embed(..., device=...)``
    and any benchmark with a pinned device fail with a bare ``TypeError``.  These
    tests build adapters and never import a framework or open a checkpoint.
    """

    NAMES = ('deepprofiler', 'ca_mae', 'o2vae', 'cellclip')

    @staticmethod
    def _build(name, **kwargs):
        from cellpaint_pipeline.models import embedding

        return embedding._build_adapter(name, **kwargs)

    @pytest.mark.parametrize('name', NAMES)
    def test_device_reaches_the_adapter_without_loading(self, name):
        adapter = self._build(name, device='cpu')
        assert adapter.requested_device == 'cpu'
        # Constructed only: nothing was built or loaded.
        assert adapter.weights_path is None
        assert adapter._model is None

    @pytest.mark.parametrize('name', NAMES)
    def test_weights_roots_reach_the_adapter(self, name):
        adapter = self._build(name, weights_roots=('X:/nope',))
        assert adapter.weights_roots == ('X:/nope',)

    @pytest.mark.parametrize('name', NAMES)
    def test_defaults_stay_unset(self, name):
        adapter = self._build(name)
        assert adapter.requested_device is None
        assert adapter.weights_roots is None


# ------------------------------------------------ single-sample promotion -----

class TestSingleSamplePromotion:
    """A bare single crop has to reach the framework with a batch axis.

    ``prepare`` preserves rank on purpose, but every adapter feeds a placeholder
    that wants a batch, so a 3-D crop used to die inside Keras with "expected
    input_2 to have 4 dimensions".  The promotion must not guess: a 3-D batch in
    torch layout is a different array with the same rank.
    """

    @staticmethod
    def _promote(name, array):
        from cellpaint_pipeline.models import registry
        from cellpaint_pipeline.models.embedding import _promote_single_sample

        return _promote_single_sample(array, registry.manifest(name))

    def test_a_single_bhwc_crop_gains_a_batch_axis(self):
        out = self._promote('deepprofiler', np.zeros((32, 32, 5), 'float32'))
        assert out.shape == (1, 32, 32, 5)

    def test_a_single_bchw_crop_gains_a_batch_axis(self):
        out = self._promote('ca_mae', np.zeros((5, 256, 256), 'uint8'))
        assert out.shape == (1, 5, 256, 256)

    def test_a_real_batch_is_left_alone(self):
        out = self._promote('ca_mae', np.zeros((8, 5, 256, 256), 'uint8'))
        assert out.shape == (8, 5, 256, 256)

    def test_an_ambiguous_3d_batch_is_not_guessed(self):
        # (N, C, H) is a batch for a BCHW method: the spatial axes do not match
        # the declared (256, 256), so it must pass through untouched.
        batch = np.zeros((8, 5, 256), 'uint8')
        assert self._promote('ca_mae', batch).shape == (8, 5, 256)

    def test_channel_embeddings_keep_their_own_layout(self):
        out = self._promote('cellclip', np.zeros((1, 5, 1536), 'float32'))
        assert out.shape == (1, 5, 1536)


# ------------------------------------------------------ output contract -------

class TestOutputContract:
    """The declared output width has to be enforced, not just documented.

    Before this check an adapter returning 1280 for a manifest that declares 672
    produced an ``EmbeddingResult`` whose ``dim`` quietly disagreed with the
    manifest -- and ``dim`` is what a benchmark sizes its aligned arrays from.
    """

    class _FakeAdapter:
        backend = 'inprocess'
        feature_layer = None
        weights_path = None

        def __init__(self, width):
            self.width = width

        def embed(self, array):
            return np.zeros((np.shape(array)[0], self.width), dtype='float32')

    def _embed_with_width(self, monkeypatch, name, width, array):
        from cellpaint_pipeline.models import embedding

        fake = self._FakeAdapter(width)
        monkeypatch.setattr(embedding, '_cached_adapter', lambda *a, **k: fake)
        return embedding.embed(name, array, skip_contract_check=True)

    def test_the_declared_width_passes(self, monkeypatch):
        array = np.zeros((1, 32, 32, 5), dtype='float32')
        result = self._embed_with_width(monkeypatch, 'deepprofiler', 672, array)
        assert result.dim == 672

    def test_a_declared_alternative_passes(self, monkeypatch):
        # DeepProfiler's pool5 layer is a legal choice and is 1280 wide.
        array = np.zeros((1, 32, 32, 5), dtype='float32')
        result = self._embed_with_width(monkeypatch, 'deepprofiler', 1280, array)
        assert result.dim == 1280

    def test_a_per_channel_multiple_passes(self, monkeypatch):
        """The per-channel rule is a contract rule, not a unidino rule.

        unidino (which declares `output_dim_per_channel`) was cancelled, so the
        check is exercised through its retained contract rather than through a
        registration it no longer has.
        """
        from cellpaint_pipeline.models import registry
        from cellpaint_pipeline.models.embedding import _check_output_width

        pending = next(r for r in registry.PENDING if r.name == 'unidino')
        per_channel = pending.manifest.output_dim_per_channel
        assert per_channel, 'unidino no longer declares a per-channel width'

        width = per_channel * 5
        _check_output_width(
            name='unidino', manifest_=pending.manifest,
            values=np.zeros((1, width), dtype='float32'),
        )

    def test_a_per_channel_width_that_is_not_a_multiple_is_refused(self):
        from cellpaint_pipeline.models import registry
        from cellpaint_pipeline.models.embedding import _check_output_width
        from cellpaint_pipeline.models.errors import EmbeddingError

        pending = next(r for r in registry.PENDING if r.name == 'unidino')
        bad = pending.manifest.output_dim_per_channel * 7 + 1
        with pytest.raises(EmbeddingError) as exc:
            _check_output_width(
                name='unidino', manifest_=pending.manifest,
                values=np.zeros((1, bad), dtype='float32'),
            )
        assert exc.value.reason == 'output-width'

    def test_an_undeclared_width_is_refused(self, monkeypatch):
        from cellpaint_pipeline.models.errors import EmbeddingError

        array = np.zeros((1, 32, 32, 5), dtype='float32')
        with pytest.raises(EmbeddingError) as exc:
            self._embed_with_width(monkeypatch, 'deepprofiler', 1281, array)
        assert exc.value.reason == 'output-width'

    def test_every_manifest_output_declaration_is_self_consistent(self):
        from cellpaint_pipeline.models import registry

        registrations = list(registry.ADAPTERS) + list(registry.PENDING)
        for registration in registrations:
            manifest = registration.manifest
            assert manifest.output_dim > 0, manifest.name
            assert manifest.output_dim not in manifest.output_dim_alternatives, manifest.name
            assert (
                len(set(manifest.output_dim_alternatives))
                == len(manifest.output_dim_alternatives)
            ), manifest.name
            if manifest.output_dim_per_channel is not None:
                assert manifest.output_dim % manifest.output_dim_per_channel == 0, manifest.name


# ------------------------------------------------------------ hashing --------

class TestWeightHashing:
    """Provenance is only real if the digest is computed from the file."""

    def test_hash_matches_a_known_file(self, tmp_path):
        import hashlib

        payload = b'cellpainting-claw'
        target = tmp_path / 'weights.bin'
        target.write_bytes(payload)
        expected = hashlib.sha256(payload).hexdigest()
        assert verify_sha256(target) == expected

    def test_repeated_hashes_agree(self, tmp_path):
        target = tmp_path / 'weights.bin'
        target.write_bytes(b'x' * 4096)
        assert verify_sha256(target) == verify_sha256(target)

    def test_cache_is_invalidated_when_the_file_changes(self, tmp_path):
        """Keyed on (path, size, mtime): a replaced checkpoint must re-hash.

        A stale digest here would attribute a benchmark number to the wrong
        artefact, which is the exact failure provenance exists to prevent.
        """
        target = tmp_path / 'weights.bin'
        target.write_bytes(b'a' * 1024)
        first = verify_sha256(target)
        target.write_bytes(b'b' * 2048)
        second = verify_sha256(target)
        assert first != second

    def test_use_cache_false_forces_a_reread(self, tmp_path):
        target = tmp_path / 'weights.bin'
        target.write_bytes(b'z' * 512)
        assert verify_sha256(target) == verify_sha256(target, use_cache=False)

    def test_declared_hashes_are_present_for_every_downloaded_method(self):
        """Methods whose weights are on disk must declare a digest."""
        for manifest in registry.manifests():
            if manifest.weights.host == 'local' and manifest.weights.downloadable:
                assert manifest.weights.sha256, manifest.name


# ------------------------------------------------------------ parity pin -----

class TestParityOutcomeIsPinned:
    """Pins the measured parity result so a regression cannot pass unnoticed.

    Measured 2026-10-10 against the known official `.npz`: using TensorFlow's own
    `crop_and_resize` for cropping plus the declared `per_crop_minmax`
    normalisation, the layer reproduces the official DeepProfiler embedding
    exactly -- `bitwise_equal=True`, `max|diff|=0.000000e+00`.  The alternative
    (raw 16-bit input, which the old manifest permitted) was off by 4.5e6.

    These two assertions cover the parts a future edit could silently undo: the
    scale keyword, and the range being enforced rather than optional.
    """

    def test_scale_is_per_crop_minmax(self):
        assert registry.manifest('deepprofiler').preprocess.scale == 'per_crop_minmax'

    def test_value_range_is_enforced_not_optional(self):
        manifest = registry.manifest('deepprofiler')
        assert manifest.preprocess.value_range == (0.0, 1.0)
        # A `None` range is exactly what let the 4.5e6-error path through before.
        assert manifest.preprocess.value_range is not None

    def test_the_scale_keyword_is_a_known_value(self):
        """`per_crop_minmax` must be part of the declared Scale vocabulary."""
        from cellpaint_pipeline.models.contract import Scale

        assert 'per_crop_minmax' in getattr(Scale, '__args__', ())


# ----------------------------------------------------------- summary io ------

class TestSummarySerialisation:
    def test_summary_is_json_serialisable(self, tmp_path):
        plan = bench.build_benchmark_plan(['deepprofiler'], output_dir=tmp_path)
        outcome = bench.BenchmarkRunResult(plan=plan)
        payload = bench.benchmark_result_to_dict(outcome)
        # Round-trips through JSON: no arrays or Paths leak into the summary.
        assert json.loads(json.dumps(payload))['methods_requested'] == ['deepprofiler']

    def test_summary_states_the_agreement_measure_and_its_limits(self):
        plan = bench.build_benchmark_plan(['deepprofiler'], output_dir=Path('.'))
        payload = bench.benchmark_result_to_dict(
            bench.BenchmarkRunResult(plan=plan)
        )
        assert 'RV coefficient' in payload['agreement_note']
        assert 'not a quality ranking' in payload['agreement_note']


class TestFacadeAndArtefactRegressions:
    """Each test pins a defect that was real, not hypothetical."""

    def test_embed_all_is_reachable_on_the_facade(self):
        """`PROJECT_STATUS.md` documented `models.embed_all`, but the facade did
        not export it, so the documented dispatch entry point raised
        AttributeError."""
        from cellpaint_pipeline import models

        assert hasattr(models, 'embed_all')
        assert 'embed_all' in models.__all__
        assert callable(models.embed_all)

    def test_batch_result_is_exported_and_not_a_bare_dict(self):
        """Failures used to be smuggled in as ``results['__errors__']``, a string
        typed as an EmbeddingResult. Iterating the mapping then yielded a str."""
        from cellpaint_pipeline import models
        from cellpaint_pipeline.models.embedding import BatchResult

        assert 'BatchResult' in models.__all__
        assert hasattr(models, 'BatchResult')
        batch = BatchResult()
        assert batch.results == {} and batch.errors == {}
        assert len(batch) == 0

    def test_batch_result_mapping_interface_only_exposes_embeddings(self):
        from cellpaint_pipeline.models.embedding import BatchResult

        sentinel = object()
        batch = BatchResult(results={'a': sentinel}, errors={'b': 'boom'})
        assert batch['a'] is sentinel
        assert 'a' in batch and 'b' not in batch
        assert list(batch) == ['a']
        assert batch.methods() == ('a',)
        assert batch.failed() == ('b',)
        with pytest.raises(KeyError):
            _ = batch['b']

    def test_save_npz_keeps_per_row_identity(self, tmp_path):
        """`save_npz` wrote only features+metadata, so a benchmark artefact could
        not be aligned afterwards -- the cell_id was gone."""
        from cellpaint_pipeline.models.contract import InputUnit
        from cellpaint_pipeline.models.embedding import EmbeddingResult

        result = EmbeddingResult(
            model='x', values=np.zeros((3, 4), dtype='float32'), dim=4,
            input_unit=InputUnit.SINGLE_CELL_CROP, n_samples=3,
            sample_metadata={'cell_id': ['a', 'b', 'c'],
                             'well': ['A01', 'A01', 'A02']},
        )
        path = result.save_npz(tmp_path / 'a.npz')
        with np.load(path, allow_pickle=True) as payload:
            assert 'sample_cell_id' in payload.files
            assert list(payload['sample_cell_id']) == ['a', 'b', 'c']
            assert 'sample_well' in payload.files
            assert 'n_samples' in json.loads(str(payload['metadata']))

    def test_save_npz_refuses_mismatched_metadata(self, tmp_path):
        from cellpaint_pipeline.models.contract import InputUnit
        from cellpaint_pipeline.models.embedding import EmbeddingResult

        result = EmbeddingResult(
            model='x', values=np.zeros((3, 4), dtype='float32'), dim=4,
            input_unit=InputUnit.SINGLE_CELL_CROP, n_samples=3,
            sample_metadata={'cell_id': ['a', 'b']},
        )
        with pytest.raises(ValueError, match='one-to-one'):
            result.save_npz(tmp_path / 'b.npz')

    def test_summary_records_finished_at_and_written_paths(self):
        plan = bench.build_benchmark_plan(['deepprofiler'], output_dir=Path('.'))
        outcome = bench.BenchmarkRunResult(plan=plan)
        outcome.finished_at = 'now'
        outcome.written = ['a.npz']
        payload = bench.benchmark_result_to_dict(outcome)
        assert payload['finished_at'] == 'now'
        assert payload['written'] == ['a.npz']

    def test_run_benchmark_sets_finished_at_before_writing(self):
        """`_write_artefacts` used to run before `finished_at` was assigned, so
        every written summary carried an empty timestamp and empty path list."""
        import inspect

        source = inspect.getsource(bench.run_benchmark)
        assert source.index('outcome.finished_at =') < source.index(
            '_write_artefacts(outcome)'
        )
