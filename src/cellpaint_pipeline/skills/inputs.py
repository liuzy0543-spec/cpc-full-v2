"""Parameter object and argument assembly for the pipeline skill runner.

WHY THIS MODULE EXISTS
-----------------------
``run_pipeline_skill`` used to expose every optional input as a flat keyword
parameter.  With seven accepted-but-ignored parameters mixed in, callers could
not tell which knobs actually did something, and the function body was a long
sequence of ``del`` statements whose only purpose was to tolerate those
parameters.

Two things are fixed here without breaking any existing caller:

* :class:`SkillInputs` groups the real inputs into one immutable object, so the
  runner receives a single value instead of dozens of keywords.
* :func:`assemble_skill_inputs` accepts **both** shapes - a ready-made
  ``SkillInputs`` or the historical flat keywords - and produces the same
  result.  The seven ignored parameters are still accepted (removing them would
  break callers) but they now emit :class:`DeprecationWarning` instead of
  being silently dropped.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from cellpaint_pipeline.config import ProjectConfig

if TYPE_CHECKING:  # annotations only - keeps "import ...skills" from pulling data_access
    from cellpaint_pipeline.data_access import DataDownloadPlan, DataRequest

__all__ = [
    'DEPRECATED_SKILL_PARAMETERS',
    'SkillInputs',
    'assemble_skill_inputs',
    'skill_inputs_from_mapping',
    'skill_inputs_to_mapping',
]


@dataclass(frozen=True)
class SkillInputs:
    """Every input a pipeline skill can be started with.

    Field names, order and defaults mirror the historical keyword parameters of
    ``run_pipeline_skill`` exactly, so a mapping built from the old keywords is
    interchangeable with this object.
    """

    output_dir: Path | None = None
    data_request: DataRequest | None = None
    download_plan: DataDownloadPlan | None = None
    workflow_root: Path | None = None
    export_root: Path | None = None
    project_root: Path | None = None
    image_csv_path: Path | None = None
    nuclei_csv_path: Path | None = None
    load_data_csv_path: Path | None = None
    manifest_path: Path | None = None
    object_table_path: Path | None = None
    single_cell_path: Path | None = None
    aggregated_path: Path | None = None
    annotated_path: Path | None = None
    normalized_path: Path | None = None
    feature_selected_path: Path | None = None
    single_cell_parquet_path: Path | None = None
    well_aggregated_parquet_path: Path | None = None
    object_table: str | None = None
    crop_mode: str | None = None
    gpu: str | None = None
    experiment_name: str | None = None
    config_filename: str | None = None
    metadata_filename: str | None = None
    workers: int = 0
    chunk_size: int = 64
    overwrite: bool = False

    def resolved(self) -> SkillInputs:
        """Return a copy with every path expanded and resolved.

        This is the single place where the historical per-parameter
        ``expanduser().resolve()`` calls live.  ``output_dir`` keeps the
        original treatment - it is resolved but not expanded - because the
        runner used ``output_dir.resolve()`` for exactly that parameter.
        """

        def resolve(value: Path | None) -> Path | None:
            return value.expanduser().resolve() if value is not None else None

        def resolve_root(value: Path | None) -> Path | None:
            return value.resolve() if value is not None else None

        return replace(
            self,
            output_dir=resolve_root(self.output_dir),
            workflow_root=resolve(self.workflow_root),
            export_root=resolve(self.export_root),
            project_root=resolve(self.project_root),
            image_csv_path=resolve(self.image_csv_path),
            nuclei_csv_path=resolve(self.nuclei_csv_path),
            load_data_csv_path=resolve(self.load_data_csv_path),
            manifest_path=resolve(self.manifest_path),
            object_table_path=resolve(self.object_table_path),
            single_cell_path=resolve(self.single_cell_path),
            aggregated_path=resolve(self.aggregated_path),
            annotated_path=resolve(self.annotated_path),
            normalized_path=resolve(self.normalized_path),
            feature_selected_path=resolve(self.feature_selected_path),
            single_cell_parquet_path=resolve(self.single_cell_parquet_path),
            well_aggregated_parquet_path=resolve(self.well_aggregated_parquet_path),
        )


#: Parameters that were accepted but never had any effect.  They are still
#: accepted so that existing callers keep working, but they are reported.
DEPRECATED_SKILL_PARAMETERS: tuple[str, ...] = (
    'profiling_suite',
    'segmentation_suite',
    'deepprofiler_mode',
    'include_validation_report',
    'include_data_access_summary',
    'plan_data_download',
    'execute_data_download_step',
)

_FIELD_NAMES = tuple(item.name for item in fields(SkillInputs))

#: Sentinel used by callers that need to forward every parameter verbatim.
UNSET: Any = object()

#: Parameters whose declared default is not ``None``.  When the object form is
#: used together with a flat keyword that still carries the documented default
#: value, the keyword is treated as "not supplied" - otherwise every call such
#: as ``inputs=SkillInputs(...)`` combined with the default ``workers=0`` would
#: be reported as a duplicate.
_NON_NULL_DEFAULTS: dict[str, Any] = {
    field.name: field.default
    for field in fields(SkillInputs)
    if field.default is not None
}

#: Aliases accepted by the automation layers for the same underlying input.
_INPUT_ALIASES: dict[str, str] = {
    'request': 'data_request',
    'plan': 'download_plan',
}


def _drop_defaulted(values: Mapping[str, Any]) -> dict[str, Any]:
    """Remove flat keywords that carry no information.

    A keyword is dropped when it is ``None`` (the caller simply did not supply
    it) or when it still equals the parameter's documented default (the caller
    repeated the default rather than overriding anything).  Both cases must
    stay out of the duplicate check, otherwise the object form could never be
    combined with the flat one.
    """
    return {
        name: value
        for name, value in values.items()
        if value is not None
        and not (name in _NON_NULL_DEFAULTS and value == _NON_NULL_DEFAULTS[name])
    }


def _warn_for_deprecated(values: Mapping[str, Any]) -> None:
    supplied = [name for name in DEPRECATED_SKILL_PARAMETERS if values.get(name) is not None]
    if not supplied:
        return
    listed = ', '.join(supplied)
    warnings.warn(
        f'run_pipeline_skill received ignored parameter(s): {listed}. '
        'They never affected the result and will be removed in a future release; '
        'use the skill that matches your goal instead.',
        DeprecationWarning,
        stacklevel=3,
    )


def _split_deprecated(values: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate the real inputs from the ignored compatibility parameters."""
    deprecated = {name: value for name, value in values.items()
                  if name in DEPRECATED_SKILL_PARAMETERS}
    real = {name: value for name, value in values.items()
            if name not in DEPRECATED_SKILL_PARAMETERS}
    return real, deprecated


def skill_inputs_from_mapping(values: Mapping[str, Any]) -> SkillInputs:
    """Build :class:`SkillInputs` from a flat mapping of keyword arguments.

    Unknown keys raise :class:`TypeError` so that a typo fails immediately
    instead of being ignored.  The ``request``/``plan`` aliases used by the
    automation layers are accepted as synonyms of ``data_request`` and
    ``download_plan``; the historically ignored parameters are dropped after
    reporting them.
    """
    real, deprecated = _split_deprecated(values)
    _warn_for_deprecated(deprecated)
    normalized: dict[str, Any] = {}
    for key, value in real.items():
        normalized[_INPUT_ALIASES.get(key, key)] = value
    known = {name: value for name, value in normalized.items()
             if name in _FIELD_NAMES}
    unexpected = sorted(name for name in normalized if name not in _FIELD_NAMES)
    if unexpected:
        raise TypeError(
            'Unexpected keyword argument for run_pipeline_skill: '
            + ', '.join(unexpected)
        )
    return SkillInputs(**known)


def skill_inputs_to_mapping(inputs: SkillInputs) -> dict[str, Any]:
    """Serialise a :class:`SkillInputs` back into a flat mapping."""
    return {name: getattr(inputs, name) for name in _FIELD_NAMES}


def assemble_skill_inputs(
    config: ProjectConfig,
    inputs: SkillInputs | Mapping[str, Any] | None = None,
    **flat_kwargs: Any,
) -> SkillInputs:
    """Normalise the two accepted call shapes into one :class:`SkillInputs`.

    * ``inputs=SkillInputs(...)`` - the object form.
    * ``inputs=None`` plus flat keywords - the historical form.
    * ``inputs=<mapping>`` - a mapping is treated like flat keywords.

    Supplying both an ``inputs`` object and the same field as a keyword is a
    programming error and is reported as such; everything else that the
    historical signature tolerated keeps working.
    """
    real_supplied, deprecated = _split_deprecated(
        {name: value for name, value in flat_kwargs.items() if value is not UNSET}
    )
    _warn_for_deprecated(deprecated)
    if inputs is None:
        if not real_supplied:
            return SkillInputs()
        return skill_inputs_from_mapping(real_supplied).resolved()

    if isinstance(inputs, Mapping):
        merged = dict(inputs)
    elif isinstance(inputs, SkillInputs):
        merged = {name: value for name, value in skill_inputs_to_mapping(inputs).items()
                  if value is not None}
    else:
        raise TypeError(
            'run_pipeline_skill inputs must be a SkillInputs, a mapping or None, '
            f'got {type(inputs).__name__}.'
        )

    # Only values the caller actually set take part in the duplicate check, and
    # a flat keyword that merely repeats a parameter default never overrides the
    # object form.
    effective = _drop_defaulted(real_supplied)
    duplicates = sorted(set(merged) & set(effective))
    if duplicates:
        raise TypeError(
            'run_pipeline_skill received the same input twice: ' + ', '.join(duplicates)
        )
    merged.update(effective)
    return skill_inputs_from_mapping(merged).resolved()
