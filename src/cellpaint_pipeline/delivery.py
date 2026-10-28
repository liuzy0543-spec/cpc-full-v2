from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cellpaint_pipeline import __version__
from cellpaint_pipeline.reporting import build_validation_report_payload
from cellpaint_pipeline.workflows.orchestration import WorkflowExecutionError, WorkflowResult, run_workflow
from typing import TYPE_CHECKING
from cellpaint_pipeline.config import ProjectConfig


PROFILING_SUITE_WORKFLOWS = {
    'native': 'post-cellprofiler-native-profiling-with-native-eval',
    'legacy-script': 'full-post-mvp-with-native-eval',
}

SEGMENTATION_SUITE_WORKFLOWS = {
    'native-post-cellprofiler': 'post-cellprofiler-native-segmentation-suite',
    'mask-export': 'mask-export-script-with-native-postprocessing',
    'deepprofiler-export': 'segmentation-and-deepprofiler-export',
    'deepprofiler-full': 'segmentation-and-deepprofiler-full-stack',
}

DEFAULT_SEGMENTATION_BUNDLE_SUITE = 'mask-export'
DEFAULT_DEEPPROFILER_FULL_STACK_SUITE = 'deepprofiler-full'


@dataclass(frozen=True)
class SuiteRunResult:
    suite_key: str
    workflow_key: str
    output_dir: Path
    manifest_path: Path | None
    step_count: int


@dataclass(frozen=True)
class FullPipelineResult:
    output_dir: Path
    profiling_suite: str
    profiling_manifest_path: Path | None
    segmentation_suite: str
    segmentation_manifest_path: Path | None
    validation_report_path: Path | None
    manifest_path: Path


@dataclass(frozen=True)
class SmokeTestResult:
    output_path: Path
    ok: bool
    check_count: int
    failed_checks: list[str]
    validation_ok: bool


class SuiteExecutionError(RuntimeError):
    """Raised when a named delivery suite cannot complete successfully."""

    def __init__(
        self,
        *,
        suite_label: str,
        suite_key: str,
        workflow_key: str,
        output_dir: Path,
        reason: str,
    ) -> None:
        self.suite_label = suite_label
        self.suite_key = suite_key
        self.workflow_key = workflow_key
        self.output_dir = output_dir
        self.reason = reason
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return '\n'.join([
            f"{self.suite_label.capitalize()} suite '{self.suite_key}' failed.",
            f"Workflow: {self.workflow_key}",
            f"Output dir: {self.output_dir}",
            f"Reason: {self.reason}",
        ])


def available_profiling_suites() -> list[str]:
    return list(PROFILING_SUITE_WORKFLOWS)


def available_segmentation_suites() -> list[str]:
    return list(SEGMENTATION_SUITE_WORKFLOWS)


def run_profiling_suite(
    config: ProjectConfig,
    suite_key: str,
    *,
    output_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> SuiteRunResult:
    workflow_key = _resolve_suite_workflow(PROFILING_SUITE_WORKFLOWS, suite_key, 'profiling suite')
    suite_root = output_dir.resolve() if output_dir is not None else (config.default_output_root / 'deliveries' / f'profiling_suite_{suite_key}')
    try:
        workflow_result = run_workflow(config, workflow_key, extra_args=extra_args, export_output_dir=suite_root)
    except WorkflowExecutionError as exc:
        raise SuiteExecutionError(
            suite_label='profiling',
            suite_key=suite_key,
            workflow_key=workflow_key,
            output_dir=suite_root,
            reason=str(exc),
        ) from exc
    return _suite_result_from_workflow(suite_key, workflow_result, suite_root)


def run_segmentation_suite(
    config: ProjectConfig,
    suite_key: str,
    *,
    output_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> SuiteRunResult:
    workflow_key = _resolve_suite_workflow(SEGMENTATION_SUITE_WORKFLOWS, suite_key, 'segmentation suite')
    suite_root = output_dir.resolve() if output_dir is not None else (config.default_output_root / 'deliveries' / f'segmentation_suite_{suite_key}')
    try:
        workflow_result = run_workflow(config, workflow_key, extra_args=extra_args, export_output_dir=suite_root)
    except WorkflowExecutionError as exc:
        raise SuiteExecutionError(
            suite_label='segmentation',
            suite_key=suite_key,
            workflow_key=workflow_key,
            output_dir=suite_root,
            reason=str(exc),
        ) from exc
    return _suite_result_from_workflow(suite_key, workflow_result, suite_root)


def run_segmentation_bundle(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> SuiteRunResult:
    return run_segmentation_suite(
        config,
        DEFAULT_SEGMENTATION_BUNDLE_SUITE,
        output_dir=output_dir,
        extra_args=extra_args,
    )


def run_deepprofiler_full_stack(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> SuiteRunResult:
    return run_segmentation_suite(
        config,
        DEFAULT_DEEPPROFILER_FULL_STACK_SUITE,
        output_dir=output_dir,
        extra_args=extra_args,
    )


def run_full_pipeline(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    profiling_suite: str = 'native',
    segmentation_suite: str = 'mask-export',
    include_validation_report: bool = True,
) -> FullPipelineResult:
    run_root = output_dir.resolve() if output_dir is not None else (config.default_output_root / 'deliveries' / 'full_pipeline')
    run_root.mkdir(parents=True, exist_ok=True)

    profiling_result = run_profiling_suite(
        config,
        profiling_suite,
        output_dir=run_root / 'profiling',
    )
    segmentation_result = run_segmentation_suite(
        config,
        segmentation_suite,
        output_dir=run_root / 'segmentation',
    )

    validation_report_path = None
    validation_payload = None
    if include_validation_report:
        validation_payload = build_validation_report_payload(config)
        validation_report_path = run_root / 'validation_report.json'
        validation_report_path.write_text(json.dumps(validation_payload, indent=2) + chr(10), encoding='utf-8')

    manifest_path = run_root / 'full_pipeline_manifest.json'
    payload = {
        'implementation': 'cellpaint_pipeline.full_pipeline',
        'generated_utc': _utc_now(),
        'package_version': __version__,
        'project_name': config.project_name,
        'output_dir': str(run_root),
        'profiling_suite': profiling_result.suite_key,
        'profiling_workflow_key': profiling_result.workflow_key,
        'profiling_manifest_path': str(profiling_result.manifest_path) if profiling_result.manifest_path else None,
        'segmentation_suite': segmentation_result.suite_key,
        'segmentation_workflow_key': segmentation_result.workflow_key,
        'segmentation_manifest_path': str(segmentation_result.manifest_path) if segmentation_result.manifest_path else None,
        'validation_report_path': str(validation_report_path) if validation_report_path else None,
        'validation_ok': bool(validation_payload['ok']) if validation_payload is not None else None,
        'ok': True,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + chr(10), encoding='utf-8')

    return FullPipelineResult(
        output_dir=run_root,
        profiling_suite=profiling_result.suite_key,
        profiling_manifest_path=profiling_result.manifest_path,
        segmentation_suite=segmentation_result.suite_key,
        segmentation_manifest_path=segmentation_result.manifest_path,
        validation_report_path=validation_report_path,
        manifest_path=manifest_path,
    )


def run_smoke_test(
    config: ProjectConfig,
    *,
    output_path: Path | None = None,
) -> SmokeTestResult:
    resolved_output_path = output_path.resolve() if output_path is not None else (config.default_output_root / 'smoke_test_report.json')
    validation_payload = build_validation_report_payload(config)
    checks = {
        'profiling_backend_root_exists': config.profiling_backend_root.exists(),
        'profiling_backend_config_exists': config.profiling_backend_config.exists(),
        'segmentation_backend_root_exists': config.segmentation_backend_root.exists(),
        'segmentation_backend_config_exists': config.segmentation_backend_config.exists(),
        'workspace_root_exists': config.workspace_root.exists(),
        'default_output_root_exists': config.default_output_root.exists(),
        'deepprofiler_export_root_parent_exists': config.deepprofiler_export_root.parent.exists(),
        'validation_report_ok': bool(validation_payload['ok']),
        'known_validation_artifact_count_nonzero': int(validation_payload['artifact_counts']['total']) > 0,
        'workflow_manifest_count_nonzero': int(validation_payload['workflow_manifest_count']) > 0,
    }
    failed_checks = [name for name, ok in checks.items() if not ok]
    payload = {
        'implementation': 'cellpaint_pipeline.smoke_test',
        'generated_utc': _utc_now(),
        'package_version': __version__,
        'project_name': config.project_name,
        'config': config.as_dict(),
        'checks': checks,
        'failed_checks': failed_checks,
        'validation_summary': {
            'ok': bool(validation_payload['ok']),
            'artifact_counts': validation_payload['artifact_counts'],
            'workflow_manifest_count': validation_payload['workflow_manifest_count'],
        },
        'ok': len(failed_checks) == 0,
    }
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(payload, indent=2) + chr(10), encoding='utf-8')
    return SmokeTestResult(
        output_path=resolved_output_path,
        ok=payload['ok'],
        check_count=len(checks),
        failed_checks=failed_checks,
        validation_ok=bool(validation_payload['ok']),
    )


def _resolve_suite_workflow(mapping: dict[str, str], suite_key: str, label: str) -> str:
    if suite_key not in mapping:
        available = ', '.join(mapping)
        raise KeyError(f'Unknown {label}: {suite_key}. Available: {available}')
    return mapping[suite_key]


def _suite_result_from_workflow(suite_key: str, workflow_result: WorkflowResult, output_dir: Path) -> SuiteRunResult:
    return SuiteRunResult(
        suite_key=suite_key,
        workflow_key=workflow_result.workflow_key,
        output_dir=output_dir,
        manifest_path=workflow_result.manifest_path,
        step_count=len(workflow_result.steps),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
