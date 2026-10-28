from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cellpaint_pipeline.config import ProjectConfig
from cellpaint_pipeline.mcp_tools import mcp_tool_catalog, run_mcp_tool_to_dict
from cellpaint_pipeline.public_api import public_api_contract_summary


@dataclass
class _ServerSettingsProxy:
    host: str | None = None
    port: int | None = None
    streamable_http_path: str | None = None
    mount_path: str | None = None


def create_mcp_server() -> Any:
    FastMCP = _load_fastmcp_class()
    mcp = FastMCP(
        'CellPaint Pipeline',
        instructions=(
            'Expose the validated Cell Painting pipeline library as MCP tools. '
            'Prefer discovery tools before running execution tools when task intent is ambiguous.'
        ),
        json_response=True,
    )

    @mcp.tool()
    def list_mcp_tools() -> list[dict[str, Any]]:
        """List MCP tools exposed by this server."""
        return mcp_tool_catalog()

    @mcp.tool()
    def show_mcp_tool_catalog() -> list[dict[str, Any]]:
        """Show the full MCP tool catalog including input schema metadata."""
        return mcp_tool_catalog()

    @mcp.tool()
    def list_public_api_entrypoints() -> list[dict[str, Any]]:
        """List recommended stable public API entrypoints."""
        return run_mcp_tool_to_dict('list_public_api_entrypoints')['result']

    @mcp.tool()
    def show_public_api_contract() -> dict[str, list[dict[str, Any]]]:
        """Show the grouped public API contract."""
        return public_api_contract_summary()

    @mcp.tool()
    def list_pipeline_skills() -> list[dict[str, Any]]:
        """List task-oriented pipeline skills."""
        return run_mcp_tool_to_dict('list_pipeline_skills')['result']

    @mcp.tool()
    def list_pipeline_presets() -> list[dict[str, Any]]:
        """List named orchestration presets."""
        return run_mcp_tool_to_dict('list_pipeline_presets')['result']

    @mcp.tool()
    def run_public_api_entrypoint(
        entrypoint: str,
        params_json: str = '{}',
        config_path: str | None = None,
    ) -> dict[str, Any]:
        """Run one stable public API entrypoint through the unified dispatcher."""
        params = _parse_params_json(params_json)
        config = _load_config(config_path)
        return run_mcp_tool_to_dict(
            'run_public_api_entrypoint',
            config=config,
            entrypoint=entrypoint,
            params=params,
        )

    @mcp.tool()
    def run_pipeline_skill(
        skill_key: str,
        config_path: str,
        params_json: str = '{}',
    ) -> dict[str, Any]:
        """Run one task-oriented pipeline skill."""
        params = _parse_params_json(params_json)
        config = _require_config(config_path)
        return run_mcp_tool_to_dict(
            'run_pipeline_skill',
            config=config,
            skill_key=skill_key,
            **params,
        )

    @mcp.tool()
    def run_pipeline_preset(
        preset_key: str,
        config_path: str,
        params_json: str = '{}',
    ) -> dict[str, Any]:
        """Run one named orchestration preset."""
        params = _parse_params_json(params_json)
        config = _require_config(config_path)
        return run_mcp_tool_to_dict(
            'run_pipeline_preset',
            config=config,
            preset_key=preset_key,
            **params,
        )

    return mcp


def run_mcp_server(
    *,
    transport: str = 'stdio',
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
) -> None:
    mcp = create_mcp_server()
    _apply_server_settings(mcp, host=host, port=port, path=path)
    try:
        if transport == 'streamable-http':
            mount_path = path or getattr(getattr(mcp, 'settings', None), 'mount_path', None)
            mcp.run(transport='streamable-http', mount_path=mount_path)
            return
        if transport == 'stdio':
            mcp.run()
            return
        raise ValueError(f'Unsupported MCP transport: {transport}')
    except (KeyboardInterrupt, asyncio.CancelledError):
        return


def _load_fastmcp_class() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            'MCP server support requires the optional dependency group `mcp`. '
            'Install it in the active CellPainting-Claw runtime with `pip install -e .[mcp]` or equivalent.'
        ) from exc
    return FastMCP


def _load_config(config_path: str | None) -> ProjectConfig | None:
    if not config_path:
        return None
    return _require_config(config_path)


def _require_config(config_path: str) -> ProjectConfig:
    return ProjectConfig.from_json(Path(config_path).expanduser().resolve())


def _parse_params_json(params_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(params_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid params_json: {exc}') from exc
    if not isinstance(payload, dict):
        raise ValueError('params_json must decode to a JSON object.')
    return payload


def _apply_server_settings(
    mcp: Any,
    *,
    host: str | None,
    port: int | None,
    path: str | None,
) -> None:
    settings = getattr(mcp, 'settings', None)
    if settings is None:
        settings = _ServerSettingsProxy()
        setattr(mcp, 'settings', settings)
    if host:
        setattr(settings, 'host', host)
    if port is not None:
        setattr(settings, 'port', port)
    if path:
        setattr(settings, 'streamable_http_path', path)
        setattr(settings, 'mount_path', path)
