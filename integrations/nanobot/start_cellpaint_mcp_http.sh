#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${CELLPAINT_PYTHON_BIN:-${PYTHON_BIN:-python}}"
cd "$ROOT"
exec "$PYTHON_BIN" -m cellpainting_claw serve-mcp --transport streamable-http --host 127.0.0.1 --port 8768 --path /mcp
