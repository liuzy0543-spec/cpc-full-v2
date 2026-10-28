"""CLI commands for the automation domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.cli.helpers import _impl



def _cmd_collect_validation_report(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            report = _impl("collect_validation_report")(
                config,
                output_path=Path(args.output_path).expanduser().resolve() if args.output_path else None,
            )
            print(json.dumps({
                'implementation': 'native',
                'step': 'collect-validation-report',
                'output_path': str(report.output_path),
                'ok': report.ok,
                'artifact_count': report.artifact_count,
                'ok_count': report.ok_count,
                'missing_count': report.missing_count,
                'failed_count': report.failed_count,
            }, indent=2, ensure_ascii=False))
            return 0 if report.ok else 1



def _cmd_list_mcp_tools(args: argparse.Namespace) -> int:
            print(json.dumps(_impl("mcp_tool_catalog")(), indent=2, ensure_ascii=False))
            return 0



def _cmd_show_mcp_tool_catalog(args: argparse.Namespace) -> int:
            print(json.dumps(_impl("mcp_tool_catalog")(), indent=2, ensure_ascii=False))
            return 0



def _cmd_run_mcp_tool(args: argparse.Namespace) -> int:
            params = json.loads(args.params_json)
            if not isinstance(params, dict):
                raise ValueError('--params-json must decode to a JSON object.')
            config = ProjectConfig.from_json(args.config) if args.config else None
            payload = _impl("run_mcp_tool_to_dict")(
                args.tool,
                config=config,
                **params,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0



def _cmd_list_public_api_entrypoints(args: argparse.Namespace) -> int:
            payload = [
                _impl("public_api_entrypoint_to_dict")(_impl("get_public_api_entrypoint")(name))
                for name in _impl("available_public_api_entrypoints")()
            ]
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0



def _cmd_show_public_api_contract(args: argparse.Namespace) -> int:
            print(json.dumps(_impl("public_api_contract_summary")(), indent=2, ensure_ascii=False))
            return 0



def _cmd_run_public_api_entrypoint(args: argparse.Namespace) -> int:
            params = json.loads(args.params_json)
            if not isinstance(params, dict):
                raise ValueError('--params-json must decode to a JSON object.')
            config = ProjectConfig.from_json(args.config) if args.config else None
            payload = _impl("run_public_api_entrypoint_to_dict")(
                args.entrypoint,
                config=config,
                **params,
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0



def _cmd_serve_mcp(args: argparse.Namespace) -> int:
            _impl("run_mcp_server")(
                transport=args.transport,
                host=args.host,
                port=args.port,
                path=args.path,
            )
            return 0



def _cmd_smoke_test(args: argparse.Namespace) -> int:
            config = ProjectConfig.from_json(args.config)
            result = _impl("run_smoke_test")(
                config,
                output_path=Path(args.output_path).expanduser().resolve() if args.output_path else None,
            )
            print(json.dumps({
                'implementation': 'cellpaint_pipeline.smoke_test',
                'output_path': str(result.output_path),
                'ok': result.ok,
                'check_count': result.check_count,
                'failed_checks': result.failed_checks,
                'validation_ok': result.validation_ok,
            }, indent=2, ensure_ascii=False))
            return 0 if result.ok else 1
def register(subparsers: argparse._SubParsersAction) -> None:
    """Attach this domain's sub-commands, in their original order."""
    validation_report_parser = subparsers.add_parser('collect-validation-report', help='Collect known validation artifacts into one report.')
    validation_report_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    validation_report_parser.add_argument('--output-path', default=None, help='Optional JSON path for the aggregated validation report.')

    list_mcp_tools_parser = subparsers.add_parser('list-mcp-tools', help='List the MCP tool wrappers exposed by the library.')

    show_mcp_catalog_parser = subparsers.add_parser('show-mcp-tool-catalog', help='Print the MCP tool catalog with schemas and routing metadata.')

    run_mcp_tool_parser = subparsers.add_parser('run-mcp-tool', help='Run one MCP wrapper tool through a single dispatcher.')
    run_mcp_tool_parser.add_argument('--tool', required=True, choices=_impl("available_mcp_tools")(), help='MCP tool name.')
    run_mcp_tool_parser.add_argument('--config', default=None, help='Optional project config JSON. Required for config-backed MCP tools.')
    run_mcp_tool_parser.add_argument('--params-json', default='{}', help='JSON object of keyword arguments passed to the selected MCP tool.')

    list_public_api_parser = subparsers.add_parser('list-public-api-entrypoints', help='List the recommended machine-readable public API entrypoints.')

    show_public_api_parser = subparsers.add_parser('show-public-api-contract', help='Print the grouped public API contract summary.')

    run_public_api_parser = subparsers.add_parser('run-public-api-entrypoint', help='Dispatch one public API entrypoint through a single automation-friendly wrapper.')
    run_public_api_parser.add_argument('--entrypoint', required=True, choices=_impl("available_public_api_entrypoints")(), help='Public API entrypoint name.')
    run_public_api_parser.add_argument('--config', default=None, help='Optional project config JSON. Required for config-backed entrypoints.')
    run_public_api_parser.add_argument('--params-json', default='{}', help='JSON object of keyword arguments passed to the selected entrypoint.')

    serve_mcp_parser = subparsers.add_parser('serve-mcp', help='Run the optional MCP server wrapper for OpenClaw or other MCP clients.')
    serve_mcp_parser.add_argument('--transport', default='stdio', choices=['stdio', 'streamable-http'], help='MCP transport mode.')
    serve_mcp_parser.add_argument('--host', default=None, help='Optional host override for HTTP transport.')
    serve_mcp_parser.add_argument('--port', type=int, default=None, help='Optional port override for HTTP transport.')
    serve_mcp_parser.add_argument('--path', default=None, help='Optional HTTP mount path override for streamable-http transport.')

    smoke_test_parser = subparsers.add_parser('smoke-test', help='Run a lightweight delivery smoke test and write a JSON report.')
    smoke_test_parser.add_argument('--config', required=True, help='Path to project config JSON.')
    smoke_test_parser.add_argument('--output-path', default=None, help='Optional JSON path for the smoke test report.')


HANDLERS = {
    'collect-validation-report': _cmd_collect_validation_report,
    'list-mcp-tools': _cmd_list_mcp_tools,
    'show-mcp-tool-catalog': _cmd_show_mcp_tool_catalog,
    'run-mcp-tool': _cmd_run_mcp_tool,
    'list-public-api-entrypoints': _cmd_list_public_api_entrypoints,
    'show-public-api-contract': _cmd_show_public_api_contract,
    'run-public-api-entrypoint': _cmd_run_public_api_entrypoint,
    'serve-mcp': _cmd_serve_mcp,
    'smoke-test': _cmd_smoke_test,
}
