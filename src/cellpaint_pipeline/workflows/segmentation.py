from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# The native step catalogue lives in ``cellpaint_pipeline.capabilities`` so that
# lower layers can read it without importing this module; re-exported here
# because these names have always been importable from this path.
from cellpaint_pipeline.capabilities import (
    NATIVE_SEGMENTATION_KEYS as _NATIVE_SEGMENTATION_KEYS,
)
from cellpaint_pipeline.capabilities import (
    available_native_segmentation_keys as _available_native_segmentation_keys,
)
from cellpaint_pipeline.runner import ExecutionResult, run_python_script
from cellpaint_pipeline.segmentation_native import (
    NativeMaskExportPipelineResult,
    NativePNGPreviewsResult,
    NativeSamplePreviewsResult,
    NativeSegmentationLoadDataResult,
    NativeSingleCellCropsResult,
    build_mask_export_pipeline_native,
    extract_single_cell_crops_native,
    generate_png_previews_native,
    generate_sample_previews_native,
    prepare_segmentation_load_data_native,
)
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


@dataclass(frozen=True)
class SegmentationScript:
    filename: str
    pass_config: bool = True


SEGMENTATION_SCRIPT_MAP = {
    "prepare-load-data": SegmentationScript("01_prepare_segmentation_load_data.py"),
    "build-mask-export-pipeline": SegmentationScript("02_build_mask_export_pipeline.py"),
    "run-mask-export": SegmentationScript("03_run_mask_export.py"),
    "extract-single-cell-crops": SegmentationScript("04_extract_single_cell_crops.py"),
    "generate-png-previews": SegmentationScript("05_generate_png_previews.py"),
    "generate-sample-previews": SegmentationScript("06_generate_sample_previews.py"),
    "full-segmentation": SegmentationScript("07_run_full_segmentation_branch.py"),
}

SEGMENTATION_TASK_MAP = {
    "full-post-mvp-segmentation": "full-segmentation",
    "mask-export-only": "run-mask-export",
    "single-cell-crops-only": "extract-single-cell-crops",
}

NATIVE_SEGMENTATION_KEYS = _NATIVE_SEGMENTATION_KEYS


def available_segmentation_scripts() -> list[str]:
    return sorted(SEGMENTATION_SCRIPT_MAP)


def available_segmentation_tasks() -> list[str]:
    return sorted(SEGMENTATION_TASK_MAP)


def available_native_segmentation_keys() -> list[str]:
    return _available_native_segmentation_keys()


def run_segmentation_script(
    config: ProjectConfig,
    script_key: str,
    extra_args: list[str] | None = None,
) -> ExecutionResult:
    if script_key not in SEGMENTATION_SCRIPT_MAP:
        raise KeyError(
            f"Unknown segmentation script key: {script_key}. "
            f"Available: {', '.join(available_segmentation_scripts())}"
        )

    script = SEGMENTATION_SCRIPT_MAP[script_key]
    script_path = config.segmentation_backend_root / "scripts" / script.filename
    final_args = _with_config_arg(config.segmentation_backend_config, extra_args, enabled=script.pass_config)
    return run_python_script(
        config.python_executable,
        Path(script_path),
        cwd=config.segmentation_backend_root,
        extra_args=final_args,
        log_dir=config.log_root / 'segmentation',
        label=f'segmentation_{script_key}',
    )


def run_segmentation_task(
    config: ProjectConfig,
    task_key: str,
    extra_args: list[str] | None = None,
) -> ExecutionResult:
    if task_key not in SEGMENTATION_TASK_MAP:
        raise KeyError(
            f"Unknown segmentation task: {task_key}. "
            f"Available: {', '.join(available_segmentation_tasks())}"
        )
    return run_segmentation_script(config, SEGMENTATION_TASK_MAP[task_key], extra_args)


def run_segmentation_native(
    config: ProjectConfig,
    step_key: str,
    *,
    output_path: Path | None = None,
    manifest_path: Path | None = None,
    mode: str = 'masked',
    workers: int = 0,
    chunk_size: int = 64,
    overwrite: bool = False,
) -> NativeSegmentationLoadDataResult | NativeMaskExportPipelineResult | NativeSingleCellCropsResult | NativePNGPreviewsResult | NativeSamplePreviewsResult:
    if step_key == 'prepare-load-data':
        return prepare_segmentation_load_data_native(config, output_path=output_path)
    if step_key == 'build-mask-export-pipeline':
        return build_mask_export_pipeline_native(config, output_path=output_path)
    if step_key == 'extract-single-cell-crops':
        return extract_single_cell_crops_native(
            config,
            mode=mode,
            output_dir=output_path,
            manifest_path=manifest_path,
            workers=workers,
        )
    if step_key == 'generate-png-previews':
        return generate_png_previews_native(
            config,
            mode=mode,
            output_dir=output_path,
            manifest_path=manifest_path,
            workers=workers,
            chunk_size=chunk_size,
        )
    if step_key == 'generate-sample-previews':
        return generate_sample_previews_native(config, output_dir=output_path, overwrite=overwrite)
    raise KeyError(
        f"Native segmentation is not available for: {step_key}. "
        f"Available native steps: {', '.join(available_native_segmentation_keys())}"
    )


def _with_config_arg(config_path: Path, extra_args: list[str] | None, *, enabled: bool) -> list[str]:
    args = list(extra_args or [])
    if not enabled:
        return args
    if '--config' in args:
        return args
    return ['--config', str(config_path), *args]
