from __future__ import annotations

import argparse

from cellpaint_pipeline.cli.lazy import (
    available_deepprofiler_modes,
    available_mcp_tools,
    available_pipeline_presets,
    available_profiling_scripts,
    available_profiling_suites,
    available_profiling_tasks,
    available_public_api_entrypoints,
    available_segmentation_scripts,
    available_segmentation_suites,
    available_segmentation_tasks,
    available_workflows,
)

"""Parser assembly and command dispatch.

``build_parser`` attaches the sub-command groups in their original order;
``main`` looks the command up in the merged table.
"""
from collections.abc import Callable

from cellpaint_pipeline.cli import commands
from cellpaint_pipeline.cli.commands import DOMAIN_ORDER


def build_parser(
    *,
    prog: str = 'cellpaint_pipeline',
    description: str = 'Standardized wrapper CLI for validated Cell Painting workflows.',
) -> argparse.ArgumentParser:
    """Build the full argument parser.

    The sub-commands are attached domain by domain, in the order the
    original single-module parser used, so ``--help`` output is
    unchanged.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    commands.config.register(subparsers)
    commands.transfer.register(subparsers)
    commands.profiling.register(subparsers)
    commands.segmentation.register(subparsers)
    commands.pipeline.register(subparsers)
    commands.skills.register(subparsers)
    commands.automation.register(subparsers)
    commands.workflows.register(subparsers)
    commands.deepprofiler.register(subparsers)
    return parser


#: Every CLI command, in the order they appear in ``--help``.
COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    **commands.config.HANDLERS,
    **commands.transfer.HANDLERS,
    **commands.profiling.HANDLERS,
    **commands.segmentation.HANDLERS,
    **commands.pipeline.HANDLERS,
    **commands.skills.HANDLERS,
    **commands.automation.HANDLERS,
    **commands.workflows.HANDLERS,
    **commands.deepprofiler.HANDLERS,
}


def main(
    argv: list[str] | None = None,
    *,
    prog: str = 'cellpaint_pipeline',
    description: str = 'Standardized wrapper CLI for validated Cell Painting workflows.',
) -> int:
    """Parse ``argv`` and run the requested sub-command.

    Each sub-command is implemented by one ``_cmd_*`` function registered in
    :data:`COMMAND_HANDLERS`, so adding a command means adding a function plus
    one table entry.  The error contract is unchanged: a failing command
    prints ``[cellpaint_pipeline] error: ...`` and returns 1, an unknown
    command is reported by argparse and exits with 2.
    """
    parser = build_parser(prog=prog, description=description)
    args = parser.parse_args(argv)

    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.error(f'Unknown command: {args.command}')
        return 2

    try:
        return handler(args)
    except Exception as exc:
        print(f'[cellpaint_pipeline] error: {exc}')
        return 1
        print(f'[cellpaint_pipeline] error: {exc}')
        return 1
