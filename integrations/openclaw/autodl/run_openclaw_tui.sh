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

"$SCRIPT_DIR/bootstrap_openclaw_workspace.sh" >/dev/null
if [ ! -f "$OPENCLAW_CONFIG_PATH" ]; then
  echo "Runtime OpenClaw config is missing: $OPENCLAW_CONFIG_PATH" >&2
  echo "Run ./configure_openai_compatible_provider.sh first." >&2
  exit 1
fi
cd "$OPENCLAW_WORKSPACE"
exec "$OPENCLAW_BIN" tui
