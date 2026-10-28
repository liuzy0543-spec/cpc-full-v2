"""CLI commands for the pipeline domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.cli.helpers import _impl



def _cmd_run_end_to_end_pipeline(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            download_plan = _impl("load_download_plan")(Path(args.plan_path).expanduser().resolve()) if args.plan_path else None
            should_build_request = bool(download_plan is None and any([
                args.include_data_access_summary,
                args.plan_data_download,
                args.execute_data_download_step,
                args.request_mode != 'gallery-source',
                args.dataset_id is not None,
                args.source_id is not None,
                args.prefix is not None,
                args.subprefix != '',
                args.bucket is not None,
                bool(args.include_substring),
                bool(args.exclude_substring),
                args.max_files is not None,
                args.overwrite,
                args.dry_run,
            ]))
            data_request = None
            if should_build_request:
                data_request = _impl("build_data_request")(
                    mode=args.request_mode,
                    dataset_id=args.dataset_id,
                    source_id=args.source_id,
                    prefix=args.prefix,
                    subprefix=args.subprefix,
                    bucket=args.bucket,
                    include_substrings=args.include_substring or [],
                    exclude_substrings=args.exclude_substring or [],
                    max_files=args.max_files,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
            result = _impl("run_end_to_end_pipeline")(
                config,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                data_request=data_request,
                download_plan=download_plan,
                include_data_access_summary=args.include_data_access_summary,
                plan_data_download=args.plan_data_download,
                execute_data_download_step=args.execute_data_download_step,
                data_summary_max_keys=args.data_summary_max_keys,
                profiling_suite=args.profiling_suite,
                segmentation_suite=args.segmentation_suite,
                run_profiling=not args.skip_profiling,
                run_segmentation=not args.skip_segmentation,
                include_validation_report=not args.skip_validation_report,
                deepprofiler_mode=args.deepprofiler_mode,
            )
            print(json.dumps(_impl("end_to_end_pipeline_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0 if result.ok else 1



def _cmd_list_pipeline_presets(args: argparse.Namespace) -> int:
            payload = [
                _impl("pipeline_preset_definition_to_dict")(_impl("get_pipeline_preset_definition")(key))
                for key in _impl("available_pipeline_presets")()
            ]
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0



def _cmd_run_pipeline_preset(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            download_plan = _impl("load_download_plan")(Path(args.plan_path).expanduser().resolve()) if args.plan_path else None
            should_build_request = bool(download_plan is None and any([
                args.request_mode != 'gallery-source',
                args.dataset_id is not None,
                args.source_id is not None,
                args.prefix is not None,
                args.subprefix != '',
                args.bucket is not None,
                bool(args.include_substring),
                bool(args.exclude_substring),
                args.max_files is not None,
                args.overwrite,
                args.dry_run,
            ]))
            data_request = None
            if should_build_request:
                data_request = _impl("build_data_request")(
                    mode=args.request_mode,
                    dataset_id=args.dataset_id,
                    source_id=args.source_id,
                    prefix=args.prefix,
                    subprefix=args.subprefix,
                    bucket=args.bucket,
                    include_substrings=args.include_substring or [],
                    exclude_substrings=args.exclude_substring or [],
                    max_files=args.max_files,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
            result = _impl("run_pipeline_preset")(
                config,
                args.preset,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                data_request=data_request,
                download_plan=download_plan,
                profiling_suite=args.profiling_suite,
                segmentation_suite=args.segmentation_suite,
                deepprofiler_mode=args.deepprofiler_mode,
            )
            print(json.dumps(_impl("end_to_end_pipeline_result_to_dict")(result), indent=2, ensure_ascii=False))
            return 0 if result.ok else 1
def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach this domain's sub-commands, in their original order."""
    orchestrated_pipeline_parser = subparsers.add_parser('run-end-to-end-pipeline', help='Run the top-level orchestration entrypoint across data access, profiling, segmentation, and optional DeepProfiler.')
    orchestrated_pipeline_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    orchestrated_pipeline_parser.add_argument('--output-dir', default=None, help='Optional output directory for the orchestrated pipeline run.')
    orchestrated_pipeline_parser.add_argument('--profiling-suite', default='native', choices=_impl("available_profiling_suites")(), help='Profiling suite alias used when profiling is enabled.')
    orchestrated_pipeline_parser.add_argument('--segmentation-suite', default='mask-export', choices=_impl("available_segmentation_suites")(), help='Segmentation suite alias used when DeepProfiler mode is off.')
    orchestrated_pipeline_parser.add_argument('--deepprofiler-mode', default='off', choices=_impl("available_deepprofiler_modes")(), help='Force the segmentation branch into DeepProfiler export or full-stack mode.')
    orchestrated_pipeline_parser.add_argument('--skip-profiling', action='store_true', help='Skip the profiling stage.')
    orchestrated_pipeline_parser.add_argument('--skip-segmentation', action='store_true', help='Skip the segmentation stage.')
    orchestrated_pipeline_parser.add_argument('--skip-validation-report', action='store_true', help='Skip writing a validation report snapshot.')
    orchestrated_pipeline_parser.add_argument('--include-data-access-summary', action='store_true', help='Write a combined data-access summary snapshot into the orchestration output.')
    orchestrated_pipeline_parser.add_argument('--plan-data-download', action='store_true', help='Build and persist a gallery download plan before workflow execution.')
    orchestrated_pipeline_parser.add_argument('--execute-data-download-step', action='store_true', help='Execute the planned gallery download step as part of the orchestration run.')
    orchestrated_pipeline_parser.add_argument('--request-mode', default='gallery-source', choices=['gallery-source', 'gallery-prefix'], help='How to interpret the optional data request inputs.')
    orchestrated_pipeline_parser.add_argument('--dataset-id', default=None, help='Optional dataset id override for gallery-source requests.')
    orchestrated_pipeline_parser.add_argument('--source-id', default=None, help='Optional source id override for gallery-source requests.')
    orchestrated_pipeline_parser.add_argument('--prefix', default=None, help='Raw gallery prefix used by gallery-prefix requests.')
    orchestrated_pipeline_parser.add_argument('--subprefix', default='', help='Optional nested subprefix under a dataset/source root.')
    orchestrated_pipeline_parser.add_argument('--bucket', default=None, help='Optional bucket override for the data request.')
    orchestrated_pipeline_parser.add_argument('--include-substring', action='append', default=None, help='Only keep object keys containing this substring. May be repeated.')
    orchestrated_pipeline_parser.add_argument('--exclude-substring', action='append', default=None, help='Skip object keys containing this substring. May be repeated.')
    orchestrated_pipeline_parser.add_argument('--max-files', type=int, default=None, help='Optional max file count for the data request.')
    orchestrated_pipeline_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing local files during download execution.')
    orchestrated_pipeline_parser.add_argument('--dry-run', action='store_true', help='Mark the embedded data request as dry-run.')
    orchestrated_pipeline_parser.add_argument('--plan-path', default=None, help='Optional path to a previously saved download plan JSON. If provided, request-building flags are ignored for execution.')
    orchestrated_pipeline_parser.add_argument('--data-summary-max-keys', type=int, default=1000, help='Maximum number of gallery prefixes inspected during summary-backed planning.')

    list_presets_parser = subparsers.add_parser('list-pipeline-presets', help='List the available high-level pipeline presets.')

    run_preset_parser = subparsers.add_parser('run-pipeline-preset', help='Run a named high-level pipeline preset.')
    run_preset_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    run_preset_parser.add_argument('--preset', required=True, choices=_impl("available_pipeline_presets")(), help='Named pipeline preset.')
    run_preset_parser.add_argument('--output-dir', default=None, help='Optional output directory for the preset run.')
    run_preset_parser.add_argument('--plan-path', default=None, help='Optional path to a previously saved download plan JSON.')
    run_preset_parser.add_argument('--profiling-suite', default=None, choices=_impl("available_profiling_suites")(), help='Optional profiling suite override.')
    run_preset_parser.add_argument('--segmentation-suite', default=None, choices=_impl("available_segmentation_suites")(), help='Optional segmentation suite override.')
    run_preset_parser.add_argument('--deepprofiler-mode', default=None, choices=_impl("available_deepprofiler_modes")(), help='Optional DeepProfiler mode override.')
    run_preset_parser.add_argument('--request-mode', default='gallery-source', choices=['gallery-source', 'gallery-prefix'], help='How to interpret optional request inputs when no plan-path is supplied.')
    run_preset_parser.add_argument('--dataset-id', default=None, help='Optional dataset id override for gallery-source requests.')
    run_preset_parser.add_argument('--source-id', default=None, help='Optional source id override for gallery-source requests.')
    run_preset_parser.add_argument('--prefix', default=None, help='Raw gallery prefix used by gallery-prefix requests.')
    run_preset_parser.add_argument('--subprefix', default='', help='Optional nested subprefix under a dataset/source root.')
    run_preset_parser.add_argument('--bucket', default=None, help='Optional bucket override for the data request.')
    run_preset_parser.add_argument('--include-substring', action='append', default=None, help='Only keep object keys containing this substring. May be repeated.')
    run_preset_parser.add_argument('--exclude-substring', action='append', default=None, help='Skip object keys containing this substring. May be repeated.')
    run_preset_parser.add_argument('--max-files', type=int, default=None, help='Optional max file count for the data request.')
    run_preset_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing local files during download execution.')
    run_preset_parser.add_argument('--dry-run', action='store_true', help='Mark the embedded data request as dry-run.')


HANDLERS = {
    'run-end-to-end-pipeline': _cmd_run_end_to_end_pipeline,
    'list-pipeline-presets': _cmd_list_pipeline_presets,
    'run-pipeline-preset': _cmd_run_pipeline_preset,
}
