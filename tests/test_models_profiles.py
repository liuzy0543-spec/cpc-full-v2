"""Tests for the v2/v3 profile split and the surrogate provenance marker.

The property under test is not "does the surrogate produce good embeddings" --
it cannot, and no test should pretend otherwise.  What must hold is:

1. the default answer to "what can I run" stays the four official methods, so
   nobody accidentally publishes a surrogate;
2. switching to v3 adds exactly the three surrogate-backed methods and nothing
   else;
3. every surrogate result says so, in the manifest *and* in the result object,
   because a marking that only exists in a document will not survive contact
   with a results table;
4. a surrogate's checkpoint digest is reported even though no digest is
   published for it.
"""

from __future__ import annotations

import pytest

from cellpaint_pipeline.models import registry
from cellpaint_pipeline.models.errors import UnknownModel


OFFICIAL = ['deepprofiler', 'ca_mae', 'cellclip', 'o2vae']
SURROGATES = ['unidino', 'dinocells', 'morphogenie']


@pytest.fixture(autouse=True)
def _restore_profile():
    """Leave the profile exactly as it was, whatever a test does to it."""
    before = registry.active_profile()
    yield
    registry.set_profile(before)


class TestProfiles:
    def test_v2_is_the_default(self):
        """A surrogate must never appear unless it was asked for by name."""
        assert registry.active_profile() == 'v2'
        assert registry.declared() == OFFICIAL

    def test_v3_adds_exactly_the_surrogates(self):
        registry.set_profile('v3')
        assert registry.declared() == OFFICIAL + SURROGATES

    def test_switching_back_removes_them_again(self):
        registry.set_profile('v3')
        registry.set_profile('v2')
        assert registry.declared() == OFFICIAL
        for name in SURROGATES:
            with pytest.raises(UnknownModel):
                registry.manifest(name)

    def test_an_unknown_profile_is_refused(self):
        with pytest.raises(ValueError, match='v9'):
            registry.set_profile('v9')

    def test_the_profile_is_reported(self):
        assert set(registry.profile_names()) == {'v2', 'v3'}

    def test_pending_is_not_the_same_as_v3(self):
        """`pending()` is contracts-only; v3 is contracts *plus* an adapter.

        Conflating them would mean a cancelled method with no checkpoint looked
        runnable, which is the exact confusion the split exists to prevent.
        """
        assert {m.name for m in registry.pending()} == set(SURROGATES)
        assert all(
            registration.factory != 'make_unavailable_adapter'
            for registration in registry.SURROGATES
        )


class TestSurrogateMarking:
    def test_every_surrogate_manifest_says_surrogate(self):
        for registration in registry.SURROGATES:
            manifest = registration.manifest
            assert manifest.weights_provenance == 'surrogate', manifest.name
            assert '[SURROGATE]' in manifest.notes, manifest.name

    def test_official_manifests_stay_official(self):
        for registration in registry.ADAPTERS:
            assert registration.manifest.weights_provenance == 'official'

    def test_surrogate_contracts_are_the_published_ones(self):
        """Only provenance changed -- the interface must be untouched.

        If a surrogate also had its own input rules, comparing it against an
        official method would be meaningless even as an interface exercise.
        """
        pending = {m.name: m for m in registry.pending()}
        for registration in registry.SURROGATES:
            manifest = registration.manifest
            original = pending[manifest.name]
            assert manifest.input_unit == original.input_unit, manifest.name
            assert manifest.channels == original.channels, manifest.name
            assert manifest.spatial == original.spatial, manifest.name
            assert manifest.output_dim == original.output_dim, manifest.name
            assert manifest.preprocess == original.preprocess, manifest.name

    def test_surrogates_declare_no_download(self):
        """Claiming a download would send a user after a release that is not there."""
        for registration in registry.SURROGATES:
            assert registration.manifest.weights.downloadable is False
            assert registration.manifest.weights.host == 'local'

    def test_morphogenie_surrogate_is_deterministic(self):
        """The official path samples; the surrogate returns the mean and says so.

        Declaring `stochastic` while always returning the mean would make a
        benchmark average draws that never vary -- a false precision.
        """
        registry.set_profile('v3')
        surrogate = registry.manifest('morphogenie')
        assert surrogate.determinism == 'deterministic'
        official_contract = next(
            m for m in registry.pending() if m.name == 'morphogenie'
        )
        assert official_contract.determinism == 'stochastic'


class TestDiagnosticsTellTheTruth:
    """`probe()` must describe what will actually run.

    The manifest's ``requires`` describes the *official* implementation -- it
    is part of the published contract, and it is what a reader wants when the
    official weights finally arrive.  Under the v3 profile, though, the
    adapter that runs is the surrogate, and a surrogate needs a strictly
    smaller set: the official DINO4Cells code imports ``kornia`` and
    ``oyaml``; the surrogate is plain torch.

    Reading the manifest made ``probe('dinocells')`` report "missing oyaml"
    and therefore "not runnable" on a machine where
    ``embed('dinocells', ...)`` succeeded -- the diagnostic told the operator
    the method was broken when it was not.
    """

    def test_probe_asks_the_adapter_not_the_manifest(self):
        registry.set_profile('v3')
        entry = registry.entry('dinocells')

        assert entry.requires_frameworks == ('torch',)
        # The manifest still carries the official contract, untouched.
        assert entry.manifest.requires == ('torch', 'kornia', 'oyaml')

    def test_probe_reports_the_effective_requirement_list(self):
        from cellpaint_pipeline.models import embedding

        registry.set_profile('v3')
        report = embedding.probe('dinocells')

        assert report['requires'] == ['torch']
        assert report['manifest_requires'] == ['torch', 'kornia', 'oyaml']

    def test_a_surrogate_is_not_penalised_for_the_official_dependencies(self):
        """The point of the fix: no false negative.

        Whether the surrogate is runnable here depends on the machine, so the
        assertion is that the *official-only* dependencies no longer appear
        in the missing list -- they are not what the surrogate needs.
        """
        from cellpaint_pipeline.models import embedding

        registry.set_profile('v3')
        report = embedding.probe('dinocells')

        assert 'kornia' not in report['missing_dependencies']
        assert 'oyaml' not in report['missing_dependencies']