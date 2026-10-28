"""What the library can do, expressed without depending on how it does it.

WHY THIS MODULE EXISTS
-----------------------
Three consumers need the same three lists: which packaged workflows exist,
which native profiling steps exist, and which native segmentation steps exist.
Those lists used to live next to the machinery that implements them
(:mod:`cellpaint_pipeline.workflows.orchestration`,
:mod:`cellpaint_pipeline.workflows.profiling` and
:mod:`cellpaint_pipeline.workflows.segmentation`), which forced
:mod:`cellpaint_pipeline.reporting` - a module that only assembles a report from
what it finds on disk - to import the whole workflow layer, and with it the
adapters, the evaluation module and both native layers.

That is a dependency pointing the wrong way: the reporting layer sat below the
orchestration layer yet depended on it.  The lists are plain metadata, so they
are declared here, in a module with no intra-package imports at all, and the
three implementation modules re-export them.  The resulting graph is::

    reporting  ──▶  capabilities  ◀──  workflows.*

Every existing import path keeps working, because each former owner still
re-exports the names it used to define.

The values themselves are unchanged: the order of every list is part of the
JSON payloads written by the validation report and by ``--help``.
"""
from __future__ import annotations

__all__ = [
    'NATIVE_PROFILING_KEYS',
    'NATIVE_SEGMENTATION_KEYS',
    'WORKFLOW_KEYS',
    'available_native_profiling_keys',
    'available_native_segmentation_keys',
    'available_workflows',
]


#: Keys of the packaged multi-step workflows, in the order the CLI prints them.
WORKFLOW_KEYS: list[str] = [
    'full-post-mvp-with-script-eval',
    'full-post-mvp-with-native-eval',
    'post-cellprofiler-native-profiling-with-native-eval',
    'post-cellprofiler-native-segmentation-suite',
    'mask-export-script-with-native-postprocessing',
    'segmentation-and-deepprofiler-export',
    'segmentation-and-deepprofiler-full-stack',
]

#: Steps of the profiling backend that run in-process rather than as a script.
NATIVE_PROFILING_KEYS: list[str] = [
    'build-image-manifest',
    'validate-inputs',
    'export-cellprofiler-to-singlecell',
    'run-pycytominer',
]

#: Steps of the segmentation backend that run in-process rather than as a script.
NATIVE_SEGMENTATION_KEYS: list[str] = [
    'prepare-load-data',
    'build-mask-export-pipeline',
    'extract-single-cell-crops',
    'generate-png-previews',
    'generate-sample-previews',
]


def available_workflows() -> list[str]:
    """Return the packaged workflow keys, in their declared order."""
    return list(WORKFLOW_KEYS)


def available_native_profiling_keys() -> list[str]:
    """Return the native profiling step keys, in their declared order."""
    return list(NATIVE_PROFILING_KEYS)


def available_native_segmentation_keys() -> list[str]:
    """Return the native segmentation step keys, in their declared order."""
    return list(NATIVE_SEGMENTATION_KEYS)
