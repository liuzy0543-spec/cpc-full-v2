#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/state"
PID_FILE="$STATE_DIR/openclaw_gateway.pid"
GATEWAY_HOST="127.0.0.1"
GATEWAY_PORT="18789"
MCP_HOST="127.0.0.1"
MCP_PORT="8768"

check_port() {
  local host="$1"
  local port="$2"
  python - <<PY
import socket
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect(("$host", int("$port")))
except OSError:
    raise SystemExit(1)
else:
    s.close()
    raise SystemExit(0)
PY
}

if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "gateway_pid=$pid alive=yes"
  else
    echo "gateway_pid=${pid:-missing} alive=no"
  fi
else
  echo "gateway_pid=missing alive=no"
fi

if check_port "$GATEWAY_HOST" "$GATEWAY_PORT"; then
  echo "gateway_port=$GATEWAY_PORT listening=yes"
else
  echo "gateway_port=$GATEWAY_PORT listening=no"
fi

if check_port "$MCP_HOST" "$MCP_PORT"; then
  echo "mcp_port=$MCP_PORT listening=yes"
else
  echo "mcp_port=$MCP_PORT listening=no"
fi
