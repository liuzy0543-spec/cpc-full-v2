#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTEGRATIONS_ROOT="$ROOT/integrations/nanobot"
PYTHON_BIN="${CELLPAINT_PYTHON_BIN:-${PYTHON_BIN:-python}}"
NANOBOT_BIN="${NANOBOT_BIN:-/home/linuxbrew/.linuxbrew/bin/nanobot}"
MCP_HOST="127.0.0.1"
MCP_PORT="8768"
MCP_PATH="/mcp"
NANOBOT_LISTEN_ADDRESS="${NANOBOT_LISTEN_ADDRESS:-127.0.0.1:8090}"
MCP_LOG="${CELLPAINT_MCP_LOG:-/tmp/cellpaint_mcp_http.log}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo 'OPENAI_API_KEY is not set.' >&2
  echo 'Export it first, then rerun this script.' >&2
  exit 1
fi

if [ ! -x "$NANOBOT_BIN" ]; then
  echo "NanoBot binary not found: $NANOBOT_BIN" >&2
  exit 1
fi

cleanup() {
  if [ -n "${MCP_PID:-}" ]; then
    kill "$MCP_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

cd "$ROOT"
"$PYTHON_BIN" -m cellpainting_claw serve-mcp --transport streamable-http --host "$MCP_HOST" --port "$MCP_PORT" --path "$MCP_PATH" >"$MCP_LOG" 2>&1 &
MCP_PID=$!
sleep 2

cd "$INTEGRATIONS_ROOT"
echo "CellPaint MCP server: http://$MCP_HOST:$MCP_PORT$MCP_PATH"
echo "NanoBot UI: http://$NANOBOT_LISTEN_ADDRESS"
echo "MCP log: $MCP_LOG"
exec "$NANOBOT_BIN" run --config ./local_config --listen-address "$NANOBOT_LISTEN_ADDRESS"
