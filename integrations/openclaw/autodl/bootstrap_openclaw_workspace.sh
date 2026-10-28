#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_WORKSPACE="$ROOT/integrations/openclaw/workspace"
STATE_DIR="${OPENCLAW_HOME:-$SCRIPT_DIR/state}"
WORKSPACE_DIR="${OPENCLAW_WORKSPACE:-$SCRIPT_DIR/workspace}"

mkdir -p "$STATE_DIR" "$STATE_DIR/logs" "$WORKSPACE_DIR" "$WORKSPACE_DIR/skills"

for name in AGENTS.md SOUL.md TOOLS.md USER.md; do
  if [ ! -f "$WORKSPACE_DIR/$name" ]; then
    cp "$SRC_WORKSPACE/$name" "$WORKSPACE_DIR/$name"
  fi
done

if [ ! -d "$WORKSPACE_DIR/skills/cellpaint-pipeline" ]; then
  mkdir -p "$WORKSPACE_DIR/skills/cellpaint-pipeline"
  cp "$SRC_WORKSPACE/skills/cellpaint-pipeline/SKILL.md" \
    "$WORKSPACE_DIR/skills/cellpaint-pipeline/SKILL.md"
fi

echo "Prepared OpenClaw workspace at: $WORKSPACE_DIR"
echo "Prepared OpenClaw state at: $STATE_DIR"
