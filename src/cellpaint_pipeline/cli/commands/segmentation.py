"""CLI commands for the segmentation domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.cli.helpers import _impl

from cellpaint_pipeline.cli.helpers import (
    _native_segmentation_result_to_dict,
    _normalize_extra_args,
)



def _cmd_run_segmentation(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            if args.backend == 'script':
                extra_args = _normalize_extra_args(args.extra_args)
                if args.overwrite:
                    extra_args = ['--overwrite', *extra_args]
                if args.script_key in {'extract-single-cell-crops', 'generate-png-previews'}:
                    extra_args = ['--mode', args.mode, *extra_args]
                if args.script_key in {'extract-single-cell-crops', 'generate-png-previews'} and args.workers > 0:
                    extra_args = ['--workers', str(args.workers), *extra_args]
                if args.script_key == 'generate-png-previews' and args.chunk_size > 0:
                    extra_args = ['--chunk-size', str(args.chunk_size), *extra_args]
                result = _impl("run_segmentation_script")(config, args.script_key, extra_args)
                print(f'[cellpaint_pipeline] completed: {result.label}')
                return result.returncode
            native_result = _impl("run_segmentation_native")(
                config,
                args.script_key,
                output_path=Path(args.output_path).expanduser().resolve() if args.output_path else None,
                manifest_path=Path(args.manifest_path).expanduser().resolve() if args.manifest_path else None,
                mode=args.mode,
                workers=args.workers,
                chunk_size=args.chunk_size,
                overwrite=args.overwrite,
            )
            print(json.dumps(_native_segmentation_result_to_dict(native_result), indent=2, ensure_ascii=False))
            return 0



def _cmd_run_segmentation_task(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("run_segmentation_task")(config, args.task, _normalize_extra_args(args.extra_args))
            print(f'[cellpaint_pipeline] completed: {result.label}')
            return result.returncode



def _cmd_run_segmentation_suite(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("run_segmentation_suite")(
                config,
                args.suite,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                extra_args=_normalize_extra_args(args.extra_args),
            )
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.delivery',
                'suite_type': 'segmentation',
                'suite_key': result.suite_key,
                'workflow_key': result.workflow_key,
                'output_dir': str(result.output_dir),
                'manifest_path': str(result.manifest_path) if result.manifest_path else None,
                'step_count': result.step_count,
            }, indent=2, ensure_ascii=False))
            return 0



def _cmd_run_segmentation_bundle(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("run_segmentation_bundle")(
                config,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                extra_args=_normalize_extra_args(args.extra_args),
            )
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.delivery',
                'suite_type': 'segmentation',
                'suite_key': result.suite_key,
                'workflow_key': result.workflow_key,
                'output_dir': str(result.output_dir),
                'manifest_path': str(result.manifest_path) if result.manifest_path else None,
                'step_count': result.step_count,
            }, indent=2, ensure_ascii=False))
            return 0



def _cmd_run_deepprofiler_full_stack(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("run_deepprofiler_full_stack")(
                config,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                extra_args=_normalize_extra_args(args.extra_args),
            )
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.delivery',
                'suite_type': 'segmentation',
                'suite_key': result.suite_key,
                'workflow_key': result.workflow_key,
                'output_dir': str(result.output_dir),
                'manifest_path': str(result.manifest_path) if result.manifest_path else None,
                'step_count': result.step_count,
            }, indent=2, ensure_ascii=False))
            return 0



def _cmd_run_full_pipeline(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("run_full_pipeline")(
                config,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                profiling_suite=args.profiling_suite,
                segmentation_suite=args.segmentation_suite,
                include_validation_report=not args.skip_validation_report,
            )
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.full_pipeline',
                'output_dir': str(result.output_dir),
                'profiling_suite': result.profiling_suite,
                'profiling_manifest_path': str(result.profiling_manifest_path) if result.profiling_manifest_path else None,
                'segmentation_suite': result.segmentation_suite,
                'segmentation_manifest_path': str(result.segmentation_manifest_path) if result.segmentation_manifest_path else None,
                'validation_report_path': str(result.validation_report_path) if result.validation_report_path else None,
                'manifest_path': str(result.manifest_path),
            }, indent=2, ensure_ascii=False))
            return 0
def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach this domain's sub-commands, in their original order."""
    segmentation_parser = subparsers.add_parser('run-segmentation', help='Run a segmentation step.')
    segmentation_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    segmentation_parser.add_argument(
        '--script-key',
        default='full-segmentation',
        choices=_impl("available_segmentation_scripts")(),
        help='Segmentation step alias.',
    )
    segmentation_parser.add_argument(
        '--backend',
        default='script',
        choices=['script', 'native'],
        help='Use the validated backend script or a native in-library implementation when available.',
    )
    segmentation_parser.add_argument('--output-path', default=None, help='Optional output path for native segmentation outputs.')
    segmentation_parser.add_argument('--manifest-path', default=None, help='Optional manifest path override for native single-cell preview generation.')
    segmentation_parser.add_argument('--mode', default='masked', choices=['masked', 'unmasked'], help='Crop mode for segmentation preview steps.')
    segmentation_parser.add_argument('--workers', type=int, default=0, help='Worker count for segmentation steps that support parallel execution.')
    segmentation_parser.add_argument('--chunk-size', type=int, default=64, help='Chunk size for native/script single-cell PNG preview generation.')
    segmentation_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing preview outputs for native/sample-preview execution.')
    segmentation_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to the backend script.')

    segmentation_task_parser = subparsers.add_parser('run-segmentation-task', help='Run a packaged segmentation task.')
    segmentation_task_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    segmentation_task_parser.add_argument(
        '--task',
        default='full-post-mvp-segmentation',
        choices=_impl("available_segmentation_tasks")(),
        help='Task-level segmentation alias.',
    )
    segmentation_task_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to the backend script.')

    segmentation_suite_parser = subparsers.add_parser('run-segmentation-suite', help='Run a packaged segmentation delivery suite.')
    segmentation_suite_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    segmentation_suite_parser.add_argument(
        '--suite',
        default='mask-export',
        choices=_impl("available_segmentation_suites")(),
        help='High-level segmentation suite alias.',
    )
    segmentation_suite_parser.add_argument('--output-dir', default=None, help='Optional output directory for the segmentation suite.')
    segmentation_suite_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to the underlying workflow backend when supported.')

    segmentation_bundle_parser = subparsers.add_parser('run-segmentation-bundle', help='Run the default high-level segmentation bundle.')
    segmentation_bundle_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    segmentation_bundle_parser.add_argument('--output-dir', default=None, help='Optional output directory for the segmentation bundle.')
    segmentation_bundle_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to the underlying workflow backend when supported.')

    deepprofiler_full_stack_parser = subparsers.add_parser('run-deepprofiler-full-stack', help='Run the default DeepProfiler full-stack segmentation bundle.')
    deepprofiler_full_stack_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    deepprofiler_full_stack_parser.add_argument('--output-dir', default=None, help='Optional output directory for the DeepProfiler full-stack bundle.')
    deepprofiler_full_stack_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to the underlying workflow backend when supported.')

    full_pipeline_parser = subparsers.add_parser('run-full-pipeline', help='Run the packaged profiling and segmentation suites together.')
    full_pipeline_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    full_pipeline_parser.add_argument(
        '--profiling-suite',
        default='native',
        choices=_impl("available_profiling_suites")(),
        help='Profiling suite alias used by the full pipeline.',
    )
    full_pipeline_parser.add_argument(
        '--segmentation-suite',
        default='mask-export',
        choices=_impl("available_segmentation_suites")(),
        help='Segmentation suite alias used by the full pipeline.',
    )
    full_pipeline_parser.add_argument('--output-dir', default=None, help='Optional output directory for the full pipeline delivery.')
    full_pipeline_parser.add_argument('--skip-validation-report', action='store_true', help='Skip writing a delivery-local validation report snapshot.')


HANDLERS = {
    'run-segmentation': _cmd_run_segmentation,
    'run-segmentation-task': _cmd_run_segmentation_task,
    'run-segmentation-suite': _cmd_run_segmentation_suite,
    'run-segmentation-bundle': _cmd_run_segmentation_bundle,
    'run-deepprofiler-full-stack': _cmd_run_deepprofiler_full_stack,
    'run-full-pipeline': _cmd_run_full_pipeline,
}
