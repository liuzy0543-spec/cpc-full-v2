"""What each skill promises to leave on disk, declared exactly once.

WHY THIS MODULE EXISTS
----------------------
The same fact - "which files does this skill produce" - used to be written
down twice: as the primary_outputs dict a runner hands to
_finalize_skill_result, and as the typical_outputs tuple the catalog shows an
agent.  Nothing compared the two, so they drifted.  A black-box audit against
the published documentation found that cp-extract-segmentation-artifacts wrote
Cytoplasm.csv, Experiment.csv, labels/ and outlines/ without ever enumerating
them, and that crop-export-single-cell-crops wrote three sub-directories its
manifest never mentioned.

Declaring the outputs here removes the second copy.  A runner binds values
through build_primary_outputs; the catalog derives its agent-facing list
through advertised_outputs.  Because both views are projected from one tuple,
they cannot disagree - and tests/test_skill_output_contract.py fails if a
future edit tries.

The module is a leaf: it imports nothing from the pipeline, so any layer may
read an output contract without pulling in a runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    'OutputEntry',
    'SEGMENTATION_ARTIFACT_OUTPUTS',
    'SEGMENTATION_MASK_OUTPUTS',
    'SINGLE_CELL_CROP_OUTPUTS',
    'DEEPPROFILER_COLLECT_OUTPUTS',
    'SKILL_OUTPUTS',
    'build_primary_outputs',
    'advertised_outputs',
]


@dataclass(frozen=True)
class OutputEntry:
    """One artefact a skill writes.

    key
        Manifest key inside primary_outputs.
    display
        Entry the agent-facing catalog shows, or None for an artefact that is
        an implementation detail: still enumerated in the manifest (so a
        consumer walking the manifest sees the whole output directory) but
        deliberately not advertised as part of the skill contract.
    optional
        The artefact may legitimately be absent.  A runner that hands over a
        path which does not exist then gets None in the manifest instead of a
        dangling path, matching the convention already used by
        _run_cellprofiler_profiling for cytoplasm_table_path.
    """

    key: str
    display: str | None = None
    optional: bool = False


# --------------------------------------------------------------------------
# cp-extract-segmentation-artifacts
# --------------------------------------------------------------------------
# The CellProfiler mask-export pipeline writes one CSV per configured object.
# Cytoplasm and Experiment are produced by the pipeline but were never
# enumerated; Experiment.csv is CellProfiler per-run table and Cytoplasm.csv
# appears only when a cytoplasm object is configured, hence optional.
SEGMENTATION_ARTIFACT_OUTPUTS: tuple[OutputEntry, ...] = (
    OutputEntry('load_data_path', 'load_data_for_segmentation.csv'),
    OutputEntry('pipeline_path', 'CPJUMP1_analysis_mask_export.cppipe'),
    OutputEntry('cellprofiler_output_dir', 'cellprofiler_masks/'),
    OutputEntry('image_table_path', 'Image.csv'),
    OutputEntry('cells_table_path', 'Cells.csv'),
    OutputEntry('cytoplasm_table_path', 'Cytoplasm.csv', optional=True),
    OutputEntry('nuclei_table_path', 'Nuclei.csv'),
    OutputEntry('experiment_table_path', 'Experiment.csv', optional=True),
    OutputEntry('labels_dir', 'labels/'),
    OutputEntry('outlines_dir', 'outlines/'),
    # Implementation details of the CellProfiler backend: enumerated so the
    # manifest describes the whole output directory, not advertised because a
    # consumer must not start depending on them.
    OutputEntry('illumination_runtime_dir', None, optional=True),
    OutputEntry('load_data_absolute_path', None, optional=True),
    # The isolated backend config this skill hands to CellProfiler.  Always
    # written next to the artefacts, so the manifest describes it too.
    OutputEntry('workflow_config_path', None),
    OutputEntry('summary_path', 'segmentation_summary.json'),
    OutputEntry('log_path', 'CellProfiler log', optional=True),
)


# run-segmentation-masks is the same pipeline plus the sample-preview step, so
# it declares the identical contract with one extra directory instead of
# repeating the twelve entries above.
SEGMENTATION_MASK_OUTPUTS: tuple[OutputEntry, ...] = SEGMENTATION_ARTIFACT_OUTPUTS + (
    OutputEntry('sample_previews_dir', 'sample_previews_png/'),
)


# --------------------------------------------------------------------------
# crop-export-single-cell-crops (and the masked / unmasked variants)
# --------------------------------------------------------------------------
# extract_single_cell_crops_native fixes this layout under the crop root; the
# manifest used to record only the root, so a manifest-driven consumer could
# not find the crops themselves.
SINGLE_CELL_CROP_OUTPUTS: tuple[OutputEntry, ...] = (
    OutputEntry('crops_dir', 'masked/ or unmasked/'),
    OutputEntry('image_stacks_dir', 'image_stacks/'),
    OutputEntry('cell_masks_dir', 'cell_masks/'),
    OutputEntry('nuclei_masks_dir', 'nuclei_masks/'),
    OutputEntry('manifest_path', 'single_cell_manifest.csv'),
    # Written only when a workflow root is supplied, hence optional.
    OutputEntry('source_config_path', None, optional=True),
)


# --------------------------------------------------------------------------
# dp-collect-deep-features
# --------------------------------------------------------------------------
# The runner already enumerated all six files; only the catalog lagged behind
# and hid deepprofiler_field_summary.csv from agents.
DEEPPROFILER_COLLECT_OUTPUTS: tuple[OutputEntry, ...] = (
    OutputEntry('single_cell_parquet_path', 'deepprofiler_single_cell.parquet'),
    OutputEntry('single_cell_csv_gz_path', 'deepprofiler_single_cell.csv.gz'),
    OutputEntry('well_aggregated_parquet_path', 'deepprofiler_well_aggregated.parquet'),
    OutputEntry('well_aggregated_csv_gz_path', 'deepprofiler_well_aggregated.csv.gz'),
    OutputEntry('field_summary_path', 'deepprofiler_field_summary.csv'),
    OutputEntry('feature_manifest_path', 'deepprofiler_feature_manifest.json'),
)


#: Skills whose catalog wording is generated from the declaration above, so
#: that definition.typical_outputs == advertised_outputs(SKILL_OUTPUTS[key])
#: holds for every entry.  A skill missing from this mapping keeps its
#: historical hand-written primary_outputs / typical_outputs.
#:
#: export-masked-single-cell-crops and export-unmasked-single-cell-crops are
#: deliberately absent: they run the same crop contract but advertise it with a
#: mode-prefixed wording (masked/image_stacks/ rather than masked/ or unmasked/),
#: which is a documentation choice rather than a different contract.
SKILL_OUTPUTS: dict[str, tuple[OutputEntry, ...]] = {
    'cp-extract-segmentation-artifacts': SEGMENTATION_ARTIFACT_OUTPUTS,
    'run-segmentation-masks': SEGMENTATION_MASK_OUTPUTS,
    'crop-export-single-cell-crops': SINGLE_CELL_CROP_OUTPUTS,
    'export-single-cell-crops': SINGLE_CELL_CROP_OUTPUTS,
    'dp-collect-deep-features': DEEPPROFILER_COLLECT_OUTPUTS,
    'collect-deepprofiler-features': DEEPPROFILER_COLLECT_OUTPUTS,
}


def build_primary_outputs(
    entries: tuple[OutputEntry, ...],
    values: Mapping[str, Path | None],
) -> dict[str, Path | None]:
    """Project runner-supplied paths onto the declared contract.

    The declaration decides which keys exist and in which order; the runner
    only supplies the paths it computed.  A key the declaration does not know
    is a programming error and is reported as one, because silently accepting
    it would let the manifest drift away from the contract again.
    """
    declared = {entry.key for entry in entries}
    undeclared = set(values) - declared
    if undeclared:
        raise ValueError(
            'primary_outputs key(s) missing from the output contract: '
            + ', '.join(sorted(undeclared))
        )
    projected: dict[str, Path | None] = {}
    unproduced: list[str] = []
    for entry in entries:
        value = values.get(entry.key)
        if value is None:
            # The mirror of an undeclared key: a required entry the runner never
            # produced.  Recording null would advertise an output that does not
            # exist, which is the drift this contract exists to prevent.
            if not entry.optional:
                unproduced.append(entry.key)
            projected[entry.key] = None
            continue
        if entry.optional and not Path(value).exists():
            value = None
        projected[entry.key] = value
    if unproduced:
        raise ValueError(
            'required output(s) were not produced: ' + ', '.join(sorted(unproduced))
        )
    return projected


def advertised_outputs(entries: tuple[OutputEntry, ...]) -> tuple[str, ...]:
    """Agent-facing view: the contract without the implementation details."""
    return tuple(entry.display for entry in entries if entry.display)
