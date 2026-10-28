"""CLI commands for the profiling domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.cli.helpers import _impl

from cellpaint_pipeline.cli.helpers import (
    _native_result_ok,
    _native_result_to_dict,
    _normalize_extra_args,
)



def _cmd_run_profiling(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            if args.backend == 'script':
                result = _impl("run_profiling_script")(config, args.script_key, _normalize_extra_args(args.extra_args))
                print(f'[cellpaint_pipeline] completed: {result.label}')
                return result.returncode
            native_result = _impl("run_profiling_native")(
                config,
                args.script_key,
                output_path=Path(args.output_path).expanduser().resolve() if args.output_path else None,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                manifest_path=Path(args.manifest_path).expanduser().resolve() if args.manifest_path else None,
                image_table_path=Path(args.image_table_path).expanduser().resolve() if args.image_table_path else None,
                object_table_path=Path(args.object_table_path).expanduser().resolve() if args.object_table_path else None,
                object_table=args.object_table,
            )
            print(json.dumps(_native_result_to_dict(native_result), indent=2, ensure_ascii=False))
            return 0 if _native_result_ok(native_result) else 1



def _cmd_run_profiling_task(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("run_profiling_task")(config, args.task, _normalize_extra_args(args.extra_args))
            print(f'[cellpaint_pipeline] completed: {result.label}')
            return result.returncode



def _cmd_run_profiling_suite(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("run_profiling_suite")(
                config,
                args.suite,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
                extra_args=_normalize_extra_args(args.extra_args),
            )
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.delivery',
                'suite_type': 'profiling',
                'suite_key': result.suite_key,
                'workflow_key': result.workflow_key,
                'output_dir': str(result.output_dir),
                'manifest_path': str(result.manifest_path) if result.manifest_path else None,
                'step_count': result.step_count,
            }, indent=2, ensure_ascii=False))
            return 0



def _cmd_run_evaluation(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            if args.backend == 'script':
                result = _impl("run_profiling_script")(config, 'evaluation', _normalize_extra_args(args.extra_args))
                print(f'[cellpaint_pipeline] completed: {result.label}')
                return result.returncode
            native_result = _impl("run_native_evaluation")(
                config,
                output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
            )
            print(json.dumps({
                'implementation': 'native',
                'output_dir': str(native_result.output_dir),
                'n_wells': native_result.n_wells,
                'n_feature_columns': native_result.n_feature_columns,
                'sample_id_column': native_result.sample_id_column,
                'run_manifest_path': str(native_result.run_manifest_path),
            }, indent=2, ensure_ascii=False))
            return 0
def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach this domain's sub-commands, in their original order."""
    profiling_parser = subparsers.add_parser('run-profiling', help='Run a profiling step.')
    profiling_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    profiling_parser.add_argument(
        '--script-key',
        default='full-pipeline',
        choices=_impl("available_profiling_scripts")(),
        help='Profiling step alias.',
    )
    profiling_parser.add_argument(
        '--backend',
        default='script',
        choices=['script', 'native'],
        help='Use the validated backend script or a native in-library implementation when available.',
    )
    profiling_parser.add_argument('--output-path', default=None, help='Optional output path for native profiling outputs.')
    profiling_parser.add_argument('--output-dir', default=None, help='Optional output directory for native profiling steps that emit multiple files.')
    profiling_parser.add_argument('--manifest-path', default=None, help='Optional manifest path for native input validation.')
    profiling_parser.add_argument('--image-table-path', default=None, help='Optional image table path for native single-cell export.')
    profiling_parser.add_argument('--object-table-path', default=None, help='Optional object table path for native single-cell export.')
    profiling_parser.add_argument('--object-table', default=None, choices=['Cells', 'Cytoplasm', 'Nuclei'], help='Object table for native single-cell export.')
    profiling_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to the backend script.')

    profiling_task_parser = subparsers.add_parser('run-profiling-task', help='Run a packaged profiling task.')
    profiling_task_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    profiling_task_parser.add_argument(
        '--task',
        default='full-post-mvp',
        choices=_impl("available_profiling_tasks")(),
        help='Task-level profiling alias.',
    )
    profiling_task_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to the backend script.')

    profiling_suite_parser = subparsers.add_parser('run-profiling-suite', help='Run a packaged profiling delivery suite.')
    profiling_suite_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    profiling_suite_parser.add_argument(
        '--suite',
        default='native',
        choices=_impl("available_profiling_suites")(),
        help='High-level profiling suite alias.',
    )
    profiling_suite_parser.add_argument('--output-dir', default=None, help='Optional output directory for the profiling suite.')
    profiling_suite_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to the underlying workflow backend when supported.')

    evaluation_parser = subparsers.add_parser('run-evaluation', help='Run the evaluation step.')
    evaluation_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    evaluation_parser.add_argument(
        '--backend',
        default='native',
        choices=['native', 'script'],
        help='Use the native in-library implementation or fall back to the validated backend script.',
    )
    evaluation_parser.add_argument('--output-dir', default=None, help='Optional output directory for native evaluation.')
    evaluation_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to the script backend.')


HANDLERS = {
    'run-profiling': _cmd_run_profiling,
    'run-profiling-task': _cmd_run_profiling_task,
    'run-profiling-suite': _cmd_run_profiling_suite,
    'run-evaluation': _cmd_run_evaluation,
}
