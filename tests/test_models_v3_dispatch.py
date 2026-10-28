"""End-to-end dispatch test for the ``v3`` profile (surrogate-backed methods).

What this proves, and what it deliberately does not
--------------------------------------------------
It proves the **dispatch layer** drives all seven methods: each produces a result
object with the same fields, and `models.embed_all` runs them from one call.

It does not prove anything about the three surrogate methods' *quality*.  Their
checkpoints were trained here, so a score from them is an interface check.  The
assertions look at shapes, provenance and field parity -- never at how good an
embedding is.

The tests skip when the checkpoints, the dataset or torch are absent, because a
fresh checkout has none of those and their absence is not a code failure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cellpaint_pipeline import models
from cellpaint_pipeline.models import preprocessing, registry

WEIGHTS_ROOT = Path(r'E:\cpc_weights')
CELLS_DIR = Path(r'E:\bbbc022\cells')

SURROGATES = ('unidino', 'dinocells', 'morphogenie')
SURROGATE_FILES = {
    name: WEIGHTS_ROOT / name / f'surrogate_{name}.pt' for name in SURROGATES
}

#: Arrays every method's payload is built from.  Named so a missing one produces
#: a skip that says which, rather than a stack trace from deep inside an adapter.
ARRAYS = {
    'deepprofiler': 'cells_dp.npy',
    'ca_mae': 'cells_ca.npy',
    'cellclip': 'cellclip_vectors.npy',
    'o2vae': 'cells_o2.npy',
    'unidino': 'cells_ca.npy',
    'dinocells': 'cells_ca.npy',
    'morphogenie': 'cells_ca.npy',
}

_MISSING = [str(p) for p in SURROGATE_FILES.values() if not p.exists()]
_MISSING += [
    str(CELLS_DIR / filename)
    for filename in sorted(set(ARRAYS.values()))
    if not (CELLS_DIR / filename).exists()
]
if importlib.util.find_spec('torch') is None:
    _MISSING.append('torch')

pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason='missing prerequisites: ' + ', '.join(_MISSING),
)


@pytest.fixture(autouse=True)
def _v3_profile():
    before = registry.active_profile()
    registry.set_profile('v3')
    yield
    registry.set_profile(before)


@pytest.fixture
def payloads():
    """One prepared payload per method, built with the library's own transforms.

    Preparing outside `embed_all` is not a workaround: `embed_all` forwards shared
    keyword arguments (``masks``, ``channels``, ``assume``) to *every* method, so a
    benchmark prepares each method's tensor first and then dispatches.  Doing it
    here keeps the test honest about what the interface actually promises.
    """
    block = np.asarray(np.load(CELLS_DIR / 'cells_ca.npy', mmap_mode='r')[:2])
    dp = np.asarray(np.load(CELLS_DIR / 'cells_dp.npy', mmap_mode='r')[:2])
    vectors = np.asarray(np.load(CELLS_DIR / 'cellclip_vectors.npy', mmap_mode='r')[:2])
    silhouettes = np.asarray(np.load(CELLS_DIR / 'cells_o2.npy', mmap_mode='r')[:2])
    bhwc = np.moveaxis(block, 1, -1).astype('uint16') * 257
    dna = block[:, 0]
    masks = (dna > dna.mean(axis=(1, 2), keepdims=True)).astype('uint8')

    def prepared(name, array, **kwargs):
        manifest = registry.manifest(name)
        return preprocessing.prepare(
            manifest, array, channels=manifest.channel_names, **kwargs
        )

    return {
        'deepprofiler': prepared('deepprofiler', dp),
        'ca_mae': prepared('ca_mae', bhwc),
        'cellclip': vectors,
        'o2vae': silhouettes,
        'unidino': prepared('unidino', bhwc),
        'dinocells': np.stack([
            prepared('dinocells', np.moveaxis(block[i], 0, -1) * 257,
                     masks=masks[i])
            for i in range(len(block))
        ]),
        'morphogenie': block[:, :3].astype('float32') / 255.0,
    }


class TestSurrogateDispatch:
    def test_each_surrogate_embeds_through_the_public_entry_point(self, payloads):
        for name in SURROGATES:
            result = models.embed(name, payloads[name])
            assert result.dim > 0, name
            assert result.metadata['weights_provenance'] == 'surrogate', name
            assert result.weights_sha256, name
            assert np.all(np.isfinite(result.as_numpy())), name

    def test_output_widths_match_the_declared_contract(self, payloads):
        """unidino concatenates one vector per channel; the others are flat."""
        unidino = models.embed('unidino', payloads['unidino'])
        manifest = registry.manifest('unidino')
        # `output_dim` is the width for the declared channel count, so it is
        # already the 1920 a five-channel call returns.
        assert unidino.dim == manifest.output_dim == 1920

        for name in ('dinocells', 'morphogenie'):
            result = models.embed(name, payloads[name])
            assert result.dim == registry.manifest(name).output_dim, name

    def test_all_seven_dispatch_in_one_call(self, payloads):
        """The dispatch requirement, stated as a test.

        Four input unit kinds and seven output widths go in; seven result objects
        come out, and every one exposes the same fields.  That sameness is what a
        benchmark scheduler needs -- not a shared array shape.
        """
        names = list(payloads)
        batch = models.embed_all(
            names, payloads, on_error='raise', weights_roots=(str(WEIGHTS_ROOT),)
        )

        assert set(batch.results) == set(names)
        assert batch.errors == {}

        field_sets = {n: set(r.to_dict()) for n, r in batch.results.items()}
        reference = field_sets['deepprofiler']
        for name, fields in field_sets.items():
            assert fields == reference, f'{name} exposes different fields'

        provenance = {
            n: r.metadata['weights_provenance'] for n, r in batch.results.items()
        }
        assert provenance == {
            'deepprofiler': 'official',
            'ca_mae': 'official',
            'cellclip': 'official',
            'o2vae': 'official',
            'unidino': 'surrogate',
            'dinocells': 'surrogate',
            'morphogenie': 'surrogate',
        }

    def test_a_surrogate_is_not_resolvable_under_v2(self, payloads):
        """The safety property: the default profile cannot reach a surrogate."""
        registry.set_profile('v2')
        with pytest.raises(models.UnknownModel):
            models.embed('unidino', payloads['unidino'])


class TestAdapterCacheDoesNotMaskAReplacedCheckpoint:
    """A checkpoint swapped mid-session must not be served from the cache.

    The cache key contains the **declared** digest, which comes from the
    manifest and therefore does not change when the file on disk does.  So a
    replaced checkpoint kept being served from the cache, and the
    ``weights_sha256`` reported next to the embedding still described the
    declared artefact rather than the file that was loaded -- a provenance
    statement that had quietly become false.

    The fix invalidates the entry when the file's fingerprint changes, so the
    load is redone and the digest is re-verified: a swapped file is rejected
    exactly as a fresh load would reject it.
    """

    def test_replacing_the_checkpoint_is_detected(self, tmp_path):
        import shutil

        from cellpaint_pipeline.models import embedding as embedding_module
        from cellpaint_pipeline.models.errors import WeightsUnavailable

        target = WEIGHTS_ROOT / 'unidino' / 'surrogate_unidino.pt'
        other = WEIGHTS_ROOT / 'morphogenie' / 'surrogate_morphogenie.pt'
        if not target.exists() or not other.exists():
            pytest.skip('surrogate checkpoints not present')

        payload = np.moveaxis(
            np.asarray(np.load(CELLS_DIR / 'cells_ca.npy', mmap_mode='r')[:2]),
            1, -1,
        ).astype('uint16') * 257

        embedding_module.clear_adapter_cache()
        models.embed('unidino', payload, assume='raw_crop')   # populates the cache

        backup = tmp_path / 'surrogate_unidino.pt'
        shutil.copy2(target, backup)
        try:
            shutil.copy2(other, target)      # same path, different content
            with pytest.raises(WeightsUnavailable):
                models.embed('unidino', payload, assume='raw_crop')
        finally:
            shutil.copy2(backup, target)
            embedding_module.clear_adapter_cache()

        # And after restoring, the method works again.
        result = models.embed('unidino', payload, assume='raw_crop')
        assert result.dim == 1920


class TestOneCallDrivesEveryMethod:
    """`embed_all` must be the single dispatch entry point, not a convenience.

    Before this, `channels`/`assume`/`masks` were broadcast to every method, so a
    seven-method call was impossible without preparing each payload by hand --
    exactly the per-method branching `embed_all` exists to remove.  These tests
    pin the uniform form.
    """

    def test_per_method_assume_and_channels(self):
        import numpy as np

        from cellpaint_pipeline import models
        from cellpaint_pipeline.models import registry

        if not (CELLS_DIR / 'cells_ca.npy').exists():
            pytest.skip('BBBC022 cells not present')

        registry.set_profile('v3')
        ca = np.asarray(np.load(CELLS_DIR / 'cells_ca.npy', mmap_mode='r')[:2])
        bhwc = np.moveaxis(ca, 1, -1).astype('uint16') * 257
        inputs = {
            'deepprofiler': np.asarray(np.load(CELLS_DIR / 'cells_dp.npy', mmap_mode='r')[:2]),
            'ca_mae': bhwc,
            'cellclip': np.asarray(np.load(CELLS_DIR / 'cellclip_vectors.npy', mmap_mode='r')[:2]),
            'o2vae': np.asarray(np.load(CELLS_DIR / 'cells_o2.npy', mmap_mode='r')[:2]),
            'unidino': bhwc,
            'dinocells': bhwc,
            'morphogenie': ca[:, :3].astype('float32') / 255.0,
        }
        names = list(inputs)

        batch = models.embed_all(
            names, inputs,
            assume={n: 'raw_crop' for n in
                    ('deepprofiler', 'ca_mae', 'unidino', 'dinocells')},
            channels={'morphogenie': ('C0', 'C1', 'C2')},
            on_error='raise',
            weights_roots=(str(WEIGHTS_ROOT),),
        )

        assert batch.errors == {}
        assert set(batch.results) == set(names)
        assert {n: r.dim for n, r in batch.results.items()} == {
            'deepprofiler': 672, 'ca_mae': 384, 'cellclip': 512, 'o2vae': 512,
            'unidino': 1920, 'dinocells': 384, 'morphogenie': 10,
        }

    def test_mapping_argument_absent_for_a_method_is_none(self):
        """A method missing from a per-method mapping gets `None`, not the dict."""
        import numpy as np

        from cellpaint_pipeline import models
        from cellpaint_pipeline.models import registry

        if not (CELLS_DIR / 'cells_ca.npy').exists():
            pytest.skip('BBBC022 cells not present')

        registry.set_profile('v3')
        ca = np.asarray(np.load(CELLS_DIR / 'cells_ca.npy', mmap_mode='r')[:1])
        payload = np.moveaxis(ca, 1, -1).astype('uint16') * 257

        # unidino is absent from `assume`, so it must be read as already
        # prepared -- and a raw uint16 crop then fails the value-range guard
        # instead of silently being divided twice.
        with pytest.raises(Exception):
            models.embed_all(
                ['unidino'], {'unidino': payload},
                assume={'ca_mae': 'raw_crop'},
                on_error='raise',
                weights_roots=(str(WEIGHTS_ROOT),),
            )
