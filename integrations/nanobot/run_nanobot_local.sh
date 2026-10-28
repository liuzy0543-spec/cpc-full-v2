#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
NANOBOT_BIN="${NANOBOT_BIN:-/home/linuxbrew/.linuxbrew/bin/nanobot}"
if [ ! -x "$NANOBOT_BIN" ]; then
  if command -v nanobot >/dev/null 2>&1; then
    NANOBOT_BIN="$(command -v nanobot)"
  else
    echo 'nanobot is not installed. Install it first, for example:' >&2
    echo '  brew install nanobot' >&2
    exit 1
  fi
fi
exec "$NANOBOT_BIN" run --config ./local_config
