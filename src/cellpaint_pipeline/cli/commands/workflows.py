"""CLI commands for the workflows domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.cli.helpers import _impl

from cellpaint_pipeline.cli.helpers import (
    _normalize_extra_args,
)



def _cmd_run_workflow(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            workflow_result = _impl("run_workflow")(
                config,
                args.workflow,
                extra_args=_normalize_extra_args(args.extra_args),
                export_output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
            )
            print(json.dumps({
                'workflow_key': workflow_result.workflow_key,
                'step_count': len(workflow_result.steps),
                'steps': workflow_result.steps,
                'export_root': str(workflow_result.export_root) if workflow_result.export_root else None,
                'manifest_path': str(workflow_result.manifest_path) if workflow_result.manifest_path else None,
            }, indent=2, ensure_ascii=False))
            return 0
def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach this domain's sub-commands, in their original order."""
    workflow_parser = subparsers.add_parser('run-workflow', help='Run a packaged multi-step workflow.')
    workflow_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    workflow_parser.add_argument(
        '--workflow',
        required=True,
        choices=_impl("available_workflows")(),
        help='Workflow alias.',
    )
    workflow_parser.add_argument('--output-dir', default=None, help='Optional export directory for workflows that create exports.')
    workflow_parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to the main backend step.')


HANDLERS = {
    'run-workflow': _cmd_run_workflow,
}
