"""The output contract is declared once, so the two views cannot drift.

WHY THIS TEST EXISTS
--------------------
A black-box audit found that cp-extract-segmentation-artifacts wrote
Cytoplasm.csv, Experiment.csv, labels/ and outlines/ without enumerating them,
and that crop-export-single-cell-crops wrote three sub-directories its manifest
never mentioned.  Both were caused by the same file list living in two places.

These tests pin the fix: one declaration projects to the manifest keys and to
the agent-facing catalog, and a runner cannot smuggle in an undeclared key.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cellpaint_pipeline.skills.catalog import get_pipeline_skill_definition
from cellpaint_pipeline.skills.outputs import (
    DEEPPROFILER_COLLECT_OUTPUTS,
    SEGMENTATION_ARTIFACT_OUTPUTS,
    SEGMENTATION_MASK_OUTPUTS,
    SINGLE_CELL_CROP_OUTPUTS,
    SKILL_OUTPUTS,
    OutputEntry,
    advertised_outputs,
    build_primary_outputs,
)


class OutputContractTests(unittest.TestCase):
    def test_catalog_matches_the_declaration(self) -> None:
        """Every registered skill advertises exactly what it declares."""
        for key, entries in SKILL_OUTPUTS.items():
            with self.subTest(skill=key):
                definition = get_pipeline_skill_definition(key)
                self.assertEqual(
                    tuple(definition.typical_outputs),
                    advertised_outputs(entries),
                )

    def test_public_and_advanced_aliases_agree(self) -> None:
        """A public key and the advanced key it wraps share one declaration."""
        for public, advanced in (
            ('crop-export-single-cell-crops', 'export-single-cell-crops'),
            ('dp-collect-deep-features', 'collect-deepprofiler-features'),
        ):
            with self.subTest(skill=public):
                self.assertIs(SKILL_OUTPUTS[public], SKILL_OUTPUTS[advanced])

    def test_public_segmentation_key_omits_previews(self) -> None:
        """cp-extract-segmentation-artifacts must not advertise previews.

        It wraps run-segmentation-masks but stops before the preview step, so
        its contract is a strict prefix of the mask-export contract rather than
        the same declaration.
        """
        public = advertised_outputs(SKILL_OUTPUTS['cp-extract-segmentation-artifacts'])
        advanced = advertised_outputs(SKILL_OUTPUTS['run-segmentation-masks'])
        self.assertEqual(advanced[: len(public)], public)
        self.assertIn('sample_previews_png/', advanced)
        self.assertNotIn('sample_previews_png/', public)

    def test_undeclared_key_is_rejected(self) -> None:
        """A runner cannot add a manifest key the contract does not know."""
        with self.assertRaises(ValueError) as ctx:
            build_primary_outputs(
                SINGLE_CELL_CROP_OUTPUTS,
                {'crops_dir': Path('x'), 'surprise_path': Path('y')},
            )
        self.assertIn('surprise_path', str(ctx.exception))

    def test_optional_entries_normalise_missing_paths_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            present = Path(td) / "present.csv"
            present.write_text("a\n", encoding="utf-8")
            entries = (
                OutputEntry('required_missing', optional=False),
                OutputEntry('optional_missing', optional=True),
                OutputEntry('optional_present', optional=True),
            )
            projected = build_primary_outputs(entries, {
                'required_missing': Path(td) / 'nope.csv',
                'optional_missing': Path(td) / 'nope.csv',
                'optional_present': present,
            })
            self.assertEqual(projected['required_missing'], Path(td) / 'nope.csv')
            self.assertIsNone(projected['optional_missing'])
            self.assertEqual(projected['optional_present'], present)

    def test_declaration_order_is_preserved(self) -> None:
        """Manifest key order follows the declaration, not the caller."""
        entries = SINGLE_CELL_CROP_OUTPUTS
        values = {entry.key: Path(entry.key) for entry in reversed(entries)}
        self.assertEqual(list(build_primary_outputs(entries, values)), [e.key for e in entries])

    def test_every_declared_display_is_unique(self) -> None:
        for name, entries in (
            ('segmentation-artifacts', SEGMENTATION_ARTIFACT_OUTPUTS),
            ('segmentation-masks', SEGMENTATION_MASK_OUTPUTS),
            ('single-cell-crops', SINGLE_CELL_CROP_OUTPUTS),
            ('deepprofiler-collect', DEEPPROFILER_COLLECT_OUTPUTS),
        ):
            with self.subTest(contract=name):
                displays = advertised_outputs(entries)
                self.assertEqual(len(displays), len(set(displays)))

    def test_segmentation_masks_extends_artifacts(self) -> None:
        """The mask-export contract is the artifact contract plus previews."""
        self.assertEqual(
            SEGMENTATION_MASK_OUTPUTS[: len(SEGMENTATION_ARTIFACT_OUTPUTS)],
            SEGMENTATION_ARTIFACT_OUTPUTS,
        )


if __name__ == "__main__":
    unittest.main()
