#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec docker compose exec cellpaint-openclaw /opt/CellPainting-Claw/integrations/openclaw/docker/entrypoint.sh tui "$@"
