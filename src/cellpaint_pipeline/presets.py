from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cellpaint_pipeline.data_access import DataDownloadPlan, DataRequest
from cellpaint_pipeline.orchestration import EndToEndPipelineResult, run_end_to_end_pipeline
from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class PipelinePresetDefinition:
    key: str
    description: str
    options: dict[str, Any]


PIPELINE_PRESETS: dict[str, PipelinePresetDefinition] = {
    'data-access-plan': PipelinePresetDefinition(
        key='data-access-plan',
        description='Build a data-access summary and gallery download plan without running profiling or segmentation.',
        options={
            'include_data_access_summary': True,
            'plan_data_download': True,
            'execute_data_download_step': False,
            'run_profiling': False,
            'run_segmentation': False,
            'include_validation_report': False,
            'deepprofiler_mode': 'off',
        },
    ),
    'data-access-download': PipelinePresetDefinition(
        key='data-access-download',
        description='Build and execute the data download path without running profiling or segmentation.',
        options={
            'include_data_access_summary': False,
            'plan_data_download': True,
            'execute_data_download_step': True,
            'run_profiling': False,
            'run_segmentation': False,
            'include_validation_report': False,
            'deepprofiler_mode': 'off',
        },
    ),
    'profiling-only': PipelinePresetDefinition(
        key='profiling-only',
        description='Run the profiling suite with validation reporting, without the segmentation branch.',
        options={
            'run_profiling': True,
            'run_segmentation': False,
            'include_validation_report': True,
            'deepprofiler_mode': 'off',
        },
    ),
    'segmentation-only': PipelinePresetDefinition(
        key='segmentation-only',
        description='Run the segmentation suite without profiling, keeping the standard mask-export route.',
        options={
            'run_profiling': False,
            'run_segmentation': True,
            'include_validation_report': True,
            'deepprofiler_mode': 'off',
            'segmentation_suite': 'mask-export',
        },
    ),
    'deepprofiler-export': PipelinePresetDefinition(
        key='deepprofiler-export',
        description='Run the segmentation branch in DeepProfiler export mode.',
        options={
            'run_profiling': False,
            'run_segmentation': True,
            'include_validation_report': True,
            'deepprofiler_mode': 'export',
        },
    ),
    'deepprofiler-full': PipelinePresetDefinition(
        key='deepprofiler-full',
        description='Run the segmentation branch in DeepProfiler full-stack mode.',
        options={
            'run_profiling': False,
            'run_segmentation': True,
            'include_validation_report': True,
            'deepprofiler_mode': 'full',
        },
    ),
    'full-pipeline': PipelinePresetDefinition(
        key='full-pipeline',
        description='Run profiling and segmentation together with validation reporting.',
        options={
            'run_profiling': True,
            'run_segmentation': True,
            'include_validation_report': True,
            'deepprofiler_mode': 'off',
        },
    ),
    'full-pipeline-with-data-plan': PipelinePresetDefinition(
        key='full-pipeline-with-data-plan',
        description='Build a data-access summary and download plan, then run the standard profiling and segmentation branches.',
        options={
            'include_data_access_summary': True,
            'plan_data_download': True,
            'run_profiling': True,
            'run_segmentation': True,
            'include_validation_report': True,
            'deepprofiler_mode': 'off',
        },
    ),
}


def available_pipeline_presets() -> list[str]:
    return list(PIPELINE_PRESETS)


def get_pipeline_preset_definition(preset_key: str) -> PipelinePresetDefinition:
    if preset_key not in PIPELINE_PRESETS:
        available = ', '.join(available_pipeline_presets())
        raise KeyError(f'Unknown pipeline preset: {preset_key}. Available: {available}')
    return PIPELINE_PRESETS[preset_key]


def pipeline_preset_definition_to_dict(definition: PipelinePresetDefinition) -> dict[str, Any]:
    return {
        'key': definition.key,
        'description': definition.description,
        'options': dict(definition.options),
    }


def run_pipeline_preset(
    config: ProjectConfig,
    preset_key: str,
    *,
    output_dir: Path | None = None,
    data_request: DataRequest | None = None,
    download_plan: DataDownloadPlan | None = None,
    profiling_suite: str | None = None,
    segmentation_suite: str | None = None,
    deepprofiler_mode: str | None = None,
    include_validation_report: bool | None = None,
    include_data_access_summary: bool | None = None,
    plan_data_download: bool | None = None,
    execute_data_download_step: bool | None = None,
) -> EndToEndPipelineResult:
    definition = get_pipeline_preset_definition(preset_key)
    options = dict(definition.options)

    if profiling_suite is not None:
        options['profiling_suite'] = profiling_suite
    if segmentation_suite is not None:
        options['segmentation_suite'] = segmentation_suite
    if deepprofiler_mode is not None:
        options['deepprofiler_mode'] = deepprofiler_mode
    if include_validation_report is not None:
        options['include_validation_report'] = include_validation_report
    if include_data_access_summary is not None:
        options['include_data_access_summary'] = include_data_access_summary
    if plan_data_download is not None:
        options['plan_data_download'] = plan_data_download
    if execute_data_download_step is not None:
        options['execute_data_download_step'] = execute_data_download_step

    resolved_output_dir = output_dir.resolve() if output_dir is not None else (config.default_output_root / 'presets' / preset_key)
    return run_end_to_end_pipeline(
        config,
        output_dir=resolved_output_dir,
        data_request=data_request,
        download_plan=download_plan,
        **options,
    )
