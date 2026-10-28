#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${CELLPAINT_PYTHON_BIN:-${PYTHON_BIN:-python}}"
HOST="${CELLPAINT_MCP_HOST:-127.0.0.1}"
PORT="${CELLPAINT_MCP_PORT:-8768}"
PATH_MOUNT="${CELLPAINT_MCP_PATH:-/mcp}"
LOG_DIR="${CELLPAINT_OPENCLAW_LOG_DIR:-$ROOT/integrations/openclaw/state/logs}"
LOG_FILE="${CELLPAINT_MCP_LOG:-$LOG_DIR/cellpaint_mcp_http.log}"

mkdir -p "$LOG_DIR"

cd "$ROOT"
exec "$PYTHON_BIN" -m cellpainting_claw serve-mcp \
  --transport streamable-http \
  --host "$HOST" \
  --port "$PORT" \
  --path "$PATH_MOUNT" \
  >>"$LOG_FILE" 2>&1
