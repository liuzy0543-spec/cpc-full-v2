from __future__ import annotations

"""Data-access skill runners."""

import json
from dataclasses import replace

from cellpaint_pipeline.runner import ExecutionResult
from cellpaint_pipeline.skills.context import SkillRuntimeContext
from cellpaint_pipeline.skills.definitions import PipelineSkillResult
from cellpaint_pipeline.skills.finalize import _finalize_skill_result, _json_ready


def _resolve_download_request_and_plan(context: SkillRuntimeContext):
    from cellpaint_pipeline.data_access import build_data_request, build_download_plan

    request = context.data_request
    if context.download_plan is not None:
        return request, context.download_plan
    if request is None:
        request = build_data_request(
            dataset_id=context.config.data_access.default_dataset_id,
            source_id=context.config.data_access.default_source_id,
            dry_run=False,
            output_dir=context.run_root / 'downloads',
            manifest_path=context.run_root / 'downloads' / 'download_manifest.json',
        )
    elif request.output_dir is None or request.manifest_path is None:
        request = replace(
            request,
            output_dir=request.output_dir or (context.run_root / 'downloads'),
            manifest_path=request.manifest_path or (context.run_root / 'downloads' / 'download_manifest.json'),
        )
    return request, build_download_plan(context.config, request)


def _run_data_plan_download(context: SkillRuntimeContext) -> PipelineSkillResult:
    from cellpaint_pipeline.data_access import data_download_plan_to_dict, write_download_plan

    _, plan = _resolve_download_request_and_plan(context)
    plan_path = write_download_plan(plan, context.run_root / 'download_plan.json')
    plan_payload = data_download_plan_to_dict(plan)
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.data_access',
        primary_outputs={'download_plan_path': plan_path},
        details=plan_payload,
        ok=True,
    )


def _run_download_cellpainting_data(context: SkillRuntimeContext) -> PipelineSkillResult:
    from cellpaint_pipeline.data_access import (
        execute_download_plan,
        write_download_plan,
    )

    _, plan = _resolve_download_request_and_plan(context)
    plan_path = write_download_plan(plan, context.run_root / 'download_plan.json')
    execution = execute_download_plan(context.config, plan)
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.data_access',
        primary_outputs={
            'download_plan_path': plan_path,
            'download_output_dir': execution.step_results[0].step.output_dir if execution.step_results else None,
            'download_manifest_path': execution.step_results[0].step.manifest_path if execution.step_results else None,
        },
        details=_json_ready(execution),
        ok=execution.ok,
    )


def _run_inspect_cellpainting_data(context: SkillRuntimeContext) -> PipelineSkillResult:
    from cellpaint_pipeline.data_access import data_access_summary_to_dict, summarize_data_access

    summary = summarize_data_access(context.config)
    summary_payload = data_access_summary_to_dict(summary)
    summary_path = context.run_root / 'data_access_summary.json'
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return _finalize_skill_result(
        context,
        implementation='cellpaint_pipeline.data_access',
        primary_outputs={'data_access_summary_path': summary_path},
        details=summary_payload,
        ok=bool(summary_payload.get('ok', True)),
    )
