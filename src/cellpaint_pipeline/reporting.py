from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cellpaint_pipeline import __version__
from cellpaint_pipeline.capabilities import (
    available_native_profiling_keys,
    available_native_segmentation_keys,
    available_workflows,
)
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from cellpaint_pipeline.config import ProjectConfig


KNOWN_VALIDATION_ARTIFACTS = {
    'native_manifest_validation': 'native_manifest_validation_summary.json',
    'native_single_cell_validation': 'native_single_cell_validation_summary.json',
    'native_pycytominer_validation': 'native_pycytominer_validation_summary.json',
    'native_evaluation_validation': 'native_evaluation_validation_summary.json',
    'native_segmentation_load_data_validation': 'native_segmentation_validation_summary.json',
    'native_sample_previews_validation': 'native_sample_previews_validation_summary.json',
    'native_png_previews_validation': 'native_png_previews_validation_summary.json',
    'native_single_cell_crops_validation': 'native_single_cell_crops_validation_summary.json',
    'segmentation_summary_validation': 'segmentation_summary_validation.json',
    'release_validation_v2': 'release_validation_v2.json',
}


@dataclass(frozen=True)
class ValidationReportResult:
    output_path: Path
    ok: bool
    artifact_count: int
    ok_count: int
    missing_count: int
    failed_count: int


def collect_validation_report(
    config: ProjectConfig,
    *,
    output_path: Path | None = None,
) -> ValidationReportResult:
    resolved_output_path = output_path.resolve() if output_path is not None else (config.default_output_root / 'validation_report.json')
    payload = build_validation_report_payload(config)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(payload, indent=2) + chr(10), encoding='utf-8')
    counts = payload['artifact_counts']
    return ValidationReportResult(
        output_path=resolved_output_path,
        ok=bool(payload['ok']),
        artifact_count=int(counts['total']),
        ok_count=int(counts['ok']),
        missing_count=int(counts['missing']),
        failed_count=int(counts['failed']),
    )


def build_validation_report_payload(config: ProjectConfig) -> dict[str, object]:
    outputs_root = config.default_output_root
    artifacts: list[dict[str, object]] = []
    for artifact_name, filename in KNOWN_VALIDATION_ARTIFACTS.items():
        artifacts.append(_summarize_validation_artifact(artifact_name, outputs_root / filename))

    workflow_root = outputs_root / 'workflows'
    workflow_manifests = []
    if workflow_root.exists():
        for manifest_path in sorted(workflow_root.glob('**/workflow_manifest.json')):
            workflow_manifests.append(_summarize_workflow_manifest(manifest_path))

    ok_count = sum(1 for artifact in artifacts if artifact['status'] == 'ok')
    missing_count = sum(1 for artifact in artifacts if artifact['status'] == 'missing')
    failed_count = sum(1 for artifact in artifacts if artifact['status'] == 'failed')
    unknown_count = sum(1 for artifact in artifacts if artifact['status'] == 'unknown')

    payload = {
        'implementation': 'cellpaint_pipeline.validation_report',
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'package_version': __version__,
        'project_name': config.project_name,
        'default_output_root': str(outputs_root),
        'native_capabilities': {
            'profiling': available_native_profiling_keys(),
            'segmentation': available_native_segmentation_keys(),
        },
        'available_workflows': available_workflows(),
        'artifact_counts': {
            'total': len(artifacts),
            'ok': ok_count,
            'missing': missing_count,
            'failed': failed_count,
            'unknown': unknown_count,
        },
        'workflow_manifest_count': len(workflow_manifests),
        'artifacts': artifacts,
        'workflow_manifests': workflow_manifests,
    }
    payload['ok'] = bool(failed_count == 0 and missing_count == 0 and unknown_count == 0)
    return payload


def _summarize_validation_artifact(name: str, path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            'name': name,
            'path': str(path),
            'exists': False,
            'status': 'missing',
            'ok': False,
        }

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return {
            'name': name,
            'path': str(path),
            'exists': True,
            'status': 'failed',
            'ok': False,
            'error': str(exc),
        }

    ok = _infer_payload_ok(payload)
    status = 'ok' if ok is True else 'failed' if ok is False else 'unknown'
    return {
        'name': name,
        'path': str(path),
        'exists': True,
        'status': status,
        'ok': ok,
        'summary': _extract_payload_summary(payload),
    }


def _summarize_workflow_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {
        'workflow_key': payload.get('workflow_key'),
        'path': str(path),
        'step_count': len(payload.get('steps', [])),
        'workflow_root': payload.get('workflow_root'),
        'export_root': payload.get('export_root'),
    }


def _infer_payload_ok(payload: object) -> bool | None:
    if isinstance(payload, dict):
        ok_value = payload.get('ok')
        if isinstance(ok_value, bool):
            return ok_value

        if 'same_rows' in payload:
            same_rows = bool(payload.get('same_rows'))
            diff_value_count = payload.get('diff_value_count')
            if isinstance(diff_value_count, int):
                return same_rows and diff_value_count == 0
            same_headers = payload.get('same_headers')
            if isinstance(same_headers, bool):
                return same_rows and same_headers
            return same_rows

        if 'same_text' in payload:
            return bool(payload.get('same_text'))

        if 'byte_identical' in payload:
            return bool(payload.get('byte_identical'))

        cases_value = payload.get('cases')
        if isinstance(cases_value, dict) and cases_value:
            inferred_cases = [_infer_payload_ok(value) for value in cases_value.values()]
            if inferred_cases and all(value is not None for value in inferred_cases):
                return all(bool(value) for value in inferred_cases)

        nested_dicts = [value for value in payload.values() if isinstance(value, dict)]
        if nested_dicts:
            inferred_nested = [_infer_payload_ok(value) for value in nested_dicts]
            if inferred_nested and all(value is not None for value in inferred_nested):
                return all(bool(value) for value in inferred_nested)

        bool_fields = [
            value
            for key, value in payload.items()
            if isinstance(value, bool) and (key.endswith('_validated') or key.endswith('_exists'))
        ]
        if bool_fields:
            return all(bool_fields)
    return None


def _extract_payload_summary(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}

    summary: dict[str, object] = {}
    for key in [
        'ok',
        'row_count',
        'field_count',
        'generated_count',
        'preview_count',
        'crop_count',
        'native_count',
        'reference_count',
        'same_rows',
        'same_headers',
        'same_text',
        'diff_value_count',
        'byte_identical',
        'byte_mismatch_count',
        'n_wells',
        'n_feature_columns',
    ]:
        if key in payload:
            summary[key] = payload[key]

    cases_value = payload.get('cases')
    if isinstance(cases_value, dict):
        case_summary = {}
        for case_name, case_payload in cases_value.items():
            case_summary[case_name] = {
                'ok': _infer_payload_ok(case_payload),
            }
            if isinstance(case_payload, dict):
                for field in ['native_count', 'reference_count', 'preview_count', 'crop_count', 'byte_mismatch_count']:
                    if field in case_payload:
                        case_summary[case_name][field] = case_payload[field]
            
        summary['cases'] = case_summary

    return summary
