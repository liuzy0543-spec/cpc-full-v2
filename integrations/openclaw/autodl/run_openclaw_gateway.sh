#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPENCLAW_BIN="${HOME}/.openclaw/bin/openclaw"
if [ -x "$DEFAULT_OPENCLAW_BIN" ]; then
  export PATH="${HOME}/.openclaw/bin:${PATH}"
fi
OPENCLAW_BIN="${OPENCLAW_BIN:-$(command -v openclaw || true)}"

if [ -z "$OPENCLAW_BIN" ]; then
  echo "openclaw is not installed." >&2
  echo "Run ./install_openclaw_cli.sh first." >&2
  exit 1
fi

export OPENCLAW_HOME="${OPENCLAW_HOME:-$SCRIPT_DIR/state}"
export OPENCLAW_WORKSPACE="${OPENCLAW_WORKSPACE:-$SCRIPT_DIR/workspace}"
export OPENCLAW_RUNTIME_CONFIG_PATH="${OPENCLAW_RUNTIME_CONFIG_PATH:-$OPENCLAW_HOME/openclaw.runtime.json}"
export OPENCLAW_CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-$OPENCLAW_RUNTIME_CONFIG_PATH}"
export CELLPAINT_MCP_HOST="${CELLPAINT_MCP_HOST:-127.0.0.1}"
export CELLPAINT_MCP_PORT="${CELLPAINT_MCP_PORT:-8768}"
export CELLPAINT_MCP_PATH="${CELLPAINT_MCP_PATH:-/mcp}"

"$SCRIPT_DIR/bootstrap_openclaw_workspace.sh"

if [ ! -f "$OPENCLAW_CONFIG_PATH" ]; then
  if [ -n "${OPENCLAW_API_KEY:-}" ] && [ -n "${OPENCLAW_BASE_URL:-}" ] && [ -n "${OPENCLAW_PRIMARY_MODEL:-}" ]; then
    "$SCRIPT_DIR/configure_openai_compatible_provider.sh"
  else
    echo "Runtime OpenClaw config is missing: $OPENCLAW_CONFIG_PATH" >&2
    echo "Run ./configure_openai_compatible_provider.sh first, or export OPENCLAW_BASE_URL, OPENCLAW_API_KEY, and OPENCLAW_PRIMARY_MODEL before launching." >&2
    exit 1
  fi
fi

mkdir -p "$OPENCLAW_HOME/logs"

echo "OPENCLAW_BIN=$OPENCLAW_BIN"
echo "OPENCLAW_HOME=$OPENCLAW_HOME"
echo "OPENCLAW_CONFIG_PATH=$OPENCLAW_CONFIG_PATH"
echo "Gateway URL: http://127.0.0.1:18789/"

"$SCRIPT_DIR/start_cellpaint_mcp_http.sh" &
CELLPAINT_MCP_PID=$!

cleanup() {
  kill "$CELLPAINT_MCP_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd "$OPENCLAW_WORKSPACE"
exec "$OPENCLAW_BIN" gateway run
