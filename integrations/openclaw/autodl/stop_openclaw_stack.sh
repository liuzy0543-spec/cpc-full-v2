#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/state"
PID_FILE="$STATE_DIR/openclaw_gateway.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found. Nothing to stop."
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [ -z "$pid" ]; then
  rm -f "$PID_FILE"
  echo "PID file was empty. Removed it."
  exit 0
fi

if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Sent SIGTERM to gateway PID $pid"
else
  echo "Process $pid is not running"
fi

rm -f "$PID_FILE"
