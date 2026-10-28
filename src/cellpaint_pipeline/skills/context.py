"""The runtime context handed to every skill runner.

WHY THE FIELD GROUPS
--------------------
:class:`SkillRuntimeContext` is a flat record whose fields are only meaningful
for particular skills - ``crop_mode`` is used by the crop exporters,
``gpu`` by the DeepProfiler runner, and a profiling skill never looks at any of
them.  The record itself is part of the internal runner contract, so it keeps
its flat fields, but the same information is also exposed as four grouped
views (:attr:`SkillRuntimeContext.data_access`,
:attr:`~SkillRuntimeContext.profiling`,
:attr:`~SkillRuntimeContext.segmentation` and
:attr:`~SkillRuntimeContext.deepprofiler`) plus
:attr:`~SkillRuntimeContext.execution`.

A runner that only needs its own stage reads the matching group, which documents
which inputs that stage actually consumes.  The grouped views are computed from
the flat fields, so the two can never disagree.

Each group also declares the catalog ``input_keys`` it serves, which lets the
package check at import time that a skill definition and its runner cannot drift
apart silently.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.skills.definitions import PipelineSkillDefinition

if TYPE_CHECKING:  # annotations only - keeps "import ...skills" from pulling data_access
    from cellpaint_pipeline.data_access import DataDownloadPlan, DataRequest


@dataclass(frozen=True)
class DataAccessInputs:
    """Inputs consumed by the data-access skills."""

    data_request: DataRequest | None = None
    download_plan: DataDownloadPlan | None = None


@dataclass(frozen=True)
class ProfilingInputs:
    """Inputs consumed by the CellProfiler / pycytominer skills."""

    image_csv_path: Path | None = None
    object_table_path: Path | None = None
    object_table: str | None = None
    single_cell_path: Path | None = None
    aggregated_path: Path | None = None
    annotated_path: Path | None = None
    normalized_path: Path | None = None
    feature_selected_path: Path | None = None
    manifest_path: Path | None = None


@dataclass(frozen=True)
class SegmentationInputs:
    """Inputs consumed by the segmentation skills."""

    workflow_root: Path | None = None
    manifest_path: Path | None = None
    crop_mode: str | None = None


@dataclass(frozen=True)
class DeepProfilerInputs:
    """Inputs consumed by the DeepProfiler skills."""

    workflow_root: Path | None = None
    export_root: Path | None = None
    project_root: Path | None = None
    image_csv_path: Path | None = None
    nuclei_csv_path: Path | None = None
    load_data_csv_path: Path | None = None
    single_cell_parquet_path: Path | None = None
    well_aggregated_parquet_path: Path | None = None
    manifest_path: Path | None = None
    gpu: str | None = None
    experiment_name: str | None = None
    config_filename: str | None = None
    metadata_filename: str | None = None


@dataclass(frozen=True)
class ExecutionOptions:
    """Knobs that apply to any stage."""

    workers: int = 0
    chunk_size: int = 64
    overwrite: bool = False


@dataclass(frozen=True)
class SkillRuntimeContext:
    config: ProjectConfig
    definition: PipelineSkillDefinition
    run_root: Path
    data_request: DataRequest | None
    download_plan: DataDownloadPlan | None
    workflow_root: Path | None
    export_root: Path | None
    project_root: Path | None
    image_csv_path: Path | None
    nuclei_csv_path: Path | None
    load_data_csv_path: Path | None
    manifest_path: Path | None
    object_table_path: Path | None
    single_cell_path: Path | None
    aggregated_path: Path | None
    annotated_path: Path | None
    normalized_path: Path | None
    feature_selected_path: Path | None
    single_cell_parquet_path: Path | None
    well_aggregated_parquet_path: Path | None
    object_table: str | None
    crop_mode: str | None
    gpu: str | None
    experiment_name: str | None
    config_filename: str | None
    metadata_filename: str | None
    workers: int
    chunk_size: int
    overwrite: bool

    # -- grouped views ----------------------------------------------------
    @property
    def data_access(self) -> DataAccessInputs:
        return DataAccessInputs(data_request=self.data_request,
                                download_plan=self.download_plan)

    @property
    def profiling(self) -> ProfilingInputs:
        return ProfilingInputs(
            image_csv_path=self.image_csv_path,
            object_table_path=self.object_table_path,
            object_table=self.object_table,
            single_cell_path=self.single_cell_path,
            aggregated_path=self.aggregated_path,
            annotated_path=self.annotated_path,
            normalized_path=self.normalized_path,
            feature_selected_path=self.feature_selected_path,
            manifest_path=self.manifest_path,
        )

    @property
    def segmentation(self) -> SegmentationInputs:
        return SegmentationInputs(workflow_root=self.workflow_root,
                                  manifest_path=self.manifest_path,
                                  crop_mode=self.crop_mode)

    @property
    def deepprofiler(self) -> DeepProfilerInputs:
        return DeepProfilerInputs(
            workflow_root=self.workflow_root,
            export_root=self.export_root,
            project_root=self.project_root,
            image_csv_path=self.image_csv_path,
            nuclei_csv_path=self.nuclei_csv_path,
            load_data_csv_path=self.load_data_csv_path,
            single_cell_parquet_path=self.single_cell_parquet_path,
            well_aggregated_parquet_path=self.well_aggregated_parquet_path,
            manifest_path=self.manifest_path,
            gpu=self.gpu,
            experiment_name=self.experiment_name,
            config_filename=self.config_filename,
            metadata_filename=self.metadata_filename,
        )

    @property
    def execution(self) -> ExecutionOptions:
        return ExecutionOptions(workers=self.workers, chunk_size=self.chunk_size,
                                overwrite=self.overwrite)

    def declared_input_keys(self) -> frozenset[str]:
        """Return the input keys the catalog declares for this skill."""
        return frozenset(self.definition.input_keys)

    #: Catalog key -> context attribute.  ``output_dir`` becomes ``run_root``
    #: because the runner resolves it into the skill's own run directory.
    INPUT_KEY_TO_FIELD: ClassVar[dict[str, str]] = {'output_dir': 'run_root'}

    def undeclared_inputs(self) -> frozenset[str]:
        """Return the declared input keys this skill can never receive.

        A definition that advertises a key the dispatcher does not know about
        would silently receive ``None`` forever, so it is reported here instead
        of failing at the end of a long run.
        """
        known = {item.name for item in fields(self) if item.init}
        return frozenset(
            key for key in self.declared_input_keys()
            if self.INPUT_KEY_TO_FIELD.get(key, key) not in known
        )


__all__ = [
    'DataAccessInputs',
    'DeepProfilerInputs',
    'ExecutionOptions',
    'ProfilingInputs',
    'SegmentationInputs',
    'SkillRuntimeContext',
]
