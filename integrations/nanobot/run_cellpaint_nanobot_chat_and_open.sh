#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INTEGRATIONS_ROOT="$ROOT/integrations/nanobot"
PYTHON_BIN="${CELLPAINT_PYTHON_BIN:-${PYTHON_BIN:-python}}"
NANOBOT_BIN="${NANOBOT_BIN:-/home/linuxbrew/.linuxbrew/bin/nanobot}"
MCP_HOST="127.0.0.1"
MCP_PORT="8768"
MCP_PATH="/mcp"
NANOBOT_HOST="127.0.0.1"
NANOBOT_PORT="8090"
NANOBOT_LISTEN_ADDRESS="${NANOBOT_LISTEN_ADDRESS:-$NANOBOT_HOST:$NANOBOT_PORT}"
NANOBOT_URL="http://$NANOBOT_HOST:$NANOBOT_PORT"
MCP_LOG="${CELLPAINT_MCP_LOG:-/tmp/cellpaint_mcp_http.log}"
NANOBOT_LOG="${CELLPAINT_NANOBOT_LOG:-/tmp/cellpaint_nanobot.log}"

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
  if [ -n "${NANOBOT_PID:-}" ]; then
    kill "$NANOBOT_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${MCP_PID:-}" ]; then
    kill "$MCP_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

open_browser() {
  local url="$1"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 &
    return 0
  fi
  if command -v gio >/dev/null 2>&1; then
    gio open "$url" >/dev/null 2>&1 &
    return 0
  fi
  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 &
    return 0
  fi
  "$PYTHON_BIN" -m webbrowser "$url" >/dev/null 2>&1 &
}

cd "$ROOT"
"$PYTHON_BIN" -m cellpainting_claw serve-mcp --transport streamable-http --host "$MCP_HOST" --port "$MCP_PORT" --path "$MCP_PATH" >"$MCP_LOG" 2>&1 &
MCP_PID=$!
sleep 2

cd "$INTEGRATIONS_ROOT"
"$NANOBOT_BIN" run --config ./local_config --listen-address "$NANOBOT_LISTEN_ADDRESS" >"$NANOBOT_LOG" 2>&1 &
NANOBOT_PID=$!
sleep 3

open_browser "$NANOBOT_URL" || true

echo "CellPaint MCP server: http://$MCP_HOST:$MCP_PORT$MCP_PATH"
echo "NanoBot UI: $NANOBOT_URL"
echo "MCP log: $MCP_LOG"
echo "NanoBot log: $NANOBOT_LOG"
wait "$NANOBOT_PID"
