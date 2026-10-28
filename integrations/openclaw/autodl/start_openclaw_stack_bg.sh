#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/state"
LOG_DIR="$STATE_DIR/logs"
PID_FILE="$STATE_DIR/openclaw_gateway.pid"
GATEWAY_LOG="$LOG_DIR/gateway.stdout.log"
HOST="${CELLPAINT_MCP_HOST:-127.0.0.1}"
PORT="${OPENCLAW_GATEWAY_PORT:-18789}"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "OpenClaw gateway already running with PID $old_pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

nohup "$SCRIPT_DIR/run_openclaw_gateway.sh" >"$GATEWAY_LOG" 2>&1 &
new_pid=$!
echo "$new_pid" > "$PID_FILE"

echo "Started OpenClaw gateway in background. PID=$new_pid"
echo "Log: $GATEWAY_LOG"

for _ in $(seq 1 30); do
  if python - <<PY
import socket
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect(("$HOST", $PORT))
except OSError:
    raise SystemExit(1)
else:
    s.close()
    raise SystemExit(0)
PY
  then
    echo "Gateway is listening on ws://$HOST:$PORT"
    exit 0
  fi
  sleep 1
  if ! kill -0 "$new_pid" 2>/dev/null; then
    echo "Gateway process exited early. Check log: $GATEWAY_LOG" >&2
    exit 1
  fi
done

echo "Gateway did not become ready within 30 seconds. Check log: $GATEWAY_LOG" >&2
exit 1
