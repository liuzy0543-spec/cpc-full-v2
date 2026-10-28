from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# The native step catalogue lives in ``cellpaint_pipeline.capabilities`` so that
# lower layers can read it without importing this module; re-exported here
# because these names have always been importable from this path.
from cellpaint_pipeline.capabilities import (
    NATIVE_PROFILING_KEYS as _NATIVE_PROFILING_KEYS,
)
from cellpaint_pipeline.capabilities import (
    available_native_profiling_keys as _available_native_profiling_keys,
)
from cellpaint_pipeline.profiling_native import (
    NativeManifestResult,
    NativePycytominerResult,
    NativeSingleCellExportResult,
    NativeValidationResult,
    build_image_manifest_native,
    export_cellprofiler_to_singlecell_native,
    run_pycytominer_native,
    validate_inputs_native,
)
from cellpaint_pipeline.runner import ExecutionResult, run_python_script
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class ProfilingScript:
    filename: str
    pass_config: bool = True


PROFILING_SCRIPT_MAP = {
    'validate-inputs': ProfilingScript('01_validate_inputs.py'),
    'build-image-manifest': ProfilingScript('02_build_image_manifest.py'),
    'export-cellprofiler-to-singlecell': ProfilingScript('03_export_cellprofiler_to_singlecell.py'),
    'run-pycytominer': ProfilingScript('04_run_pycytominer.py'),
    'prepare-cpjump1-subset': ProfilingScript('05_prepare_cpjump1_subset.py'),
    'prepare-cellprofiler-load-data': ProfilingScript('06_prepare_cellprofiler_load_data.py'),
    'run-official-cellprofiler': ProfilingScript('07_run_official_cellprofiler.py'),
    'full-pipeline': ProfilingScript('08_run_full_pipeline.py'),
    'cellprofiler-sharded': ProfilingScript('09_run_cellprofiler_sharded.py'),
    'evaluation': ProfilingScript('10_run_profile_evaluation.py', pass_config=False),
    'outlines-only-sharded': ProfilingScript('11_run_outlines_only_sharded.py'),
}

PROFILING_TASK_MAP = {
    'full-post-mvp': 'full-pipeline',
    'evaluation-only': 'evaluation',
    'cellprofiler-only': 'run-official-cellprofiler',
}

NATIVE_PROFILING_KEYS = _NATIVE_PROFILING_KEYS


def available_profiling_scripts() -> list[str]:
    return sorted(PROFILING_SCRIPT_MAP)


def available_native_profiling_keys() -> list[str]:
    return _available_native_profiling_keys()


def available_profiling_tasks() -> list[str]:
    return sorted(PROFILING_TASK_MAP)


def run_profiling_script(
    config: ProjectConfig,
    script_key: str,
    extra_args: list[str] | None = None,
) -> ExecutionResult:
    if script_key not in PROFILING_SCRIPT_MAP:
        raise KeyError(
            f"Unknown profiling script key: {script_key}. "
            f"Available: {', '.join(available_profiling_scripts())}"
        )

    script = PROFILING_SCRIPT_MAP[script_key]
    script_path = config.profiling_backend_root / 'scripts' / script.filename
    final_args = _with_config_arg(config.profiling_backend_config, extra_args, enabled=script.pass_config)
    return run_python_script(
        config.python_executable,
        Path(script_path),
        cwd=config.profiling_backend_root,
        extra_args=final_args,
        log_dir=config.log_root / 'profiling',
        label=f'profiling_{script_key}',
    )


def run_profiling_native(
    config: ProjectConfig,
    step_key: str,
    *,
    output_path: Path | None = None,
    output_dir: Path | None = None,
    manifest_path: Path | None = None,
    image_table_path: Path | None = None,
    object_table_path: Path | None = None,
    object_table: str | None = None,
) -> NativeManifestResult | NativeValidationResult | NativeSingleCellExportResult | NativePycytominerResult:
    if step_key == 'build-image-manifest':
        return build_image_manifest_native(config, output_path=output_path)
    if step_key == 'validate-inputs':
        return validate_inputs_native(config, manifest_path=manifest_path)
    if step_key == 'export-cellprofiler-to-singlecell':
        return export_cellprofiler_to_singlecell_native(
            config,
            object_table=object_table,
            image_table_path=image_table_path,
            object_table_path=object_table_path,
            output_path=output_path,
        )
    if step_key == 'run-pycytominer':
        return run_pycytominer_native(config, output_dir=output_dir)
    raise KeyError(
        f"Native profiling is not available for: {step_key}. "
        f"Available native steps: {', '.join(available_native_profiling_keys())}"
    )


def run_profiling_task(
    config: ProjectConfig,
    task_key: str,
    extra_args: list[str] | None = None,
) -> ExecutionResult:
    if task_key not in PROFILING_TASK_MAP:
        raise KeyError(
            f"Unknown profiling task: {task_key}. "
            f"Available: {', '.join(available_profiling_tasks())}"
        )
    return run_profiling_script(config, PROFILING_TASK_MAP[task_key], extra_args)


def _with_config_arg(config_path: Path, extra_args: list[str] | None, *, enabled: bool) -> list[str]:
    args = list(extra_args or [])
    if not enabled:
        return args
    if '--config' in args:
        return args
    return ['--config', str(config_path), *args]
