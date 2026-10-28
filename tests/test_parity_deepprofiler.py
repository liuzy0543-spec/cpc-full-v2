"""Parity smoke test: does our adapter still reproduce the official pipeline?

This is the acceptance test for the whole layer's functional claim.  It runs the
real checkpoints and the real TIFFs, so it takes ~40 s and needs the official
demo tree on disk; when any prerequisite is missing it skips rather than failing,
because a machine without the weights cannot answer the question.

What it pins down
-----------------
The official DeepProfiler run's `.npz` is a known result.  Rebuilding the same
crops and pushing them through our adapter must reproduce it **bitwise** -- not
"approximately", since the same graph on the same input is deterministic.

It deliberately does *not* assert that the corrected channel order matches the
old demo: it does not, and that divergence is the fix, not a regression.  See
`scripts/embedding_benchmark/parity_vs_official_dp.py` case C.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts' / 'embedding_benchmark' / 'parity_vs_official_dp.py'

#: Where the official demo run left its outputs.  Override with CPC_DP_OFFICIAL.
OFFICIAL_ROOT = Path(
    r'D:\cpc_c_official\demo\workspace\outputs\dp_run\deepprofiler_project'
)
RAW_ROOT = Path(
    r'D:\cpc_c_official\demo\backend\profiling_backend\data\raw_core\BR00000001'
)


def _load_parity_module():
    spec = importlib.util.spec_from_file_location('cpc_parity', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prerequisites() -> str | None:
    """Return a skip reason, or None when the parity check can actually run."""
    if not SCRIPT.is_file():
        return f'parity script missing at {SCRIPT}'
    if not (OFFICIAL_ROOT / 'outputs').is_dir():
        return f'official demo tree missing at {OFFICIAL_ROOT}'
    if not RAW_ROOT.is_dir():
        return f'raw TIFFs missing at {RAW_ROOT}'
    try:
        importlib.import_module('tensorflow')
    except Exception:  # noqa: BLE001 - any import failure means "cannot run here"
        return 'tensorflow is not installed'
    return None


_SKIP = _prerequisites()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))


@pytest.fixture(scope='module')
def parity():
    return _load_parity_module()


@pytest.fixture(scope='module')
def rebuilt(parity):
    """Rebuild the official crops and load the official features once."""
    official, crops = {}, {}
    for well in ('A01', 'A02'):
        npz_path = (
            parity.DEFAULT_OFFICIAL_ROOT
            / 'outputs' / 'cell_painting_cnn_demo' / 'features'
            / 'BR00000001' / well / '1.npz'
        )
        if not npz_path.is_file():
            pytest.skip(f'official features missing at {npz_path}')
        with np.load(npz_path, allow_pickle=True) as payload:
            official[well] = np.asarray(payload['features'])

        centres = parity.read_locations(parity.DEFAULT_OFFICIAL_ROOT, well)
        image = parity.load_field(parity.DEFAULT_RAW_ROOT, well, parity.DEMO_CHANNELS)
        crops[well] = parity.crop_with_official_op(image, centres)

    expected = np.concatenate([official[w] for w in official], axis=0)
    stacked = np.concatenate([crops[w] for w in crops], axis=0).astype('float32')
    return expected, parity.per_crop_minmax(stacked)


class TestOfficialPipelineParity:
    """The functional claim: same code path in, same numbers out."""

    def test_reproduces_official_embedding_bitwise(self, parity, rebuilt):
        expected, prepared = rebuilt
        from cellpaint_pipeline import models

        got = models.embed(
            'deepprofiler', prepared,
            unit='single_cell_crop', channels=parity.DEMO_CHANNELS,
            # The demo order is the swapped one, so the contract rejects it by
            # design (see test_contract_rejects_the_swapped_channel_order).  The
            # point of THIS test is the numerical path, so the guard is bypassed
            # explicitly rather than the assertion being weakened.
            skip_contract_check=True,
        ).as_numpy()

        assert got.shape == expected.shape, (
            f'embedding width drifted: {got.shape} vs official {expected.shape}'
        )
        # Bitwise, not allclose: the same graph on the same input is deterministic,
        # so any tolerance here would hide a real change in the pipeline.
        assert np.array_equal(got, expected), (
            f'max|diff| = {np.abs(got - expected).max():.6e}'
        )

    def test_prepare_matches_the_official_normalisation(self, parity, rebuilt):
        """`preprocessing.prepare` must equal `cropping.py:65-67`, not merely be close."""
        _expected, prepared = rebuilt
        from cellpaint_pipeline.models import preprocessing

        raw = parity.load_field(
            parity.DEFAULT_RAW_ROOT, 'A01', parity.DEMO_CHANNELS
        )
        centres = parity.read_locations(parity.DEFAULT_OFFICIAL_ROOT, 'A01')
        crops = parity.crop_with_official_op(raw, centres)

        assert np.array_equal(
            preprocessing.prepare('deepprofiler', crops.astype('float32')),
            parity.per_crop_minmax(crops),
        ), 'preprocessing.prepare drifted from the official normalisation'

    def test_contract_rejects_the_swapped_channel_order(self, parity, rebuilt):
        """The old demo's RNA/ER swap must stay rejected, not silently accepted."""
        from cellpaint_pipeline import models
        from cellpaint_pipeline.models.errors import IncompatibleInput

        _expected, prepared = rebuilt
        with pytest.raises(IncompatibleInput):
            models.embed(
                'deepprofiler', prepared,
                unit='single_cell_crop', channels=parity.DEMO_CHANNELS,
            )

    def test_contract_rejects_raw_16bit_pixels(self, parity, rebuilt):
        """Feeding un-normalised pixels must fail loudly (it is off by ~4.5e6)."""
        from cellpaint_pipeline import models
        from cellpaint_pipeline.models.errors import IncompatibleInput

        _expected, _prepared = rebuilt
        raw = parity.load_field(
            parity.DEFAULT_RAW_ROOT, 'A01', parity.UPSTREAM_CHANNELS
        )
        centres = parity.read_locations(parity.DEFAULT_OFFICIAL_ROOT, 'A01')
        crops = parity.crop_with_official_op(raw, centres).astype('float32')

        with pytest.raises(IncompatibleInput):
            models.embed(
                'deepprofiler', crops,
                unit='single_cell_crop', channels=parity.UPSTREAM_CHANNELS,
            )

    def test_assume_raw_crop_equals_prepare_then_embed(self, parity, rebuilt):
        """The convenience path must not be a second, subtly different pipeline.

        `assume='raw_crop'` exists so a caller can hand over acquisition values
        and let the layer normalise them.  If it drifted from
        `prepare` + `embed`, the two paths would disagree and there would be no
        way to tell which number was authoritative.
        """
        from cellpaint_pipeline import models

        _expected, _prepared = rebuilt
        raw = parity.load_field(
            parity.DEFAULT_RAW_ROOT, 'A01', parity.DEMO_CHANNELS
        )
        centres = parity.read_locations(parity.DEFAULT_OFFICIAL_ROOT, 'A01')
        crops = parity.crop_with_official_op(raw, centres).astype('float32')

        direct = models.embed(
            'deepprofiler', crops, assume='raw_crop', channels=parity.DEMO_CHANNELS
        ).as_numpy()

        from cellpaint_pipeline.models import preprocessing
        prepared = preprocessing.prepare(
            'deepprofiler', crops, channels=parity.DEMO_CHANNELS
        )
        manual = models.embed(
            'deepprofiler', prepared, unit='single_cell_crop'
        ).as_numpy()

        assert np.array_equal(direct, manual), (
            f'max|diff| = {np.abs(direct - manual).max():.6e}'
        )