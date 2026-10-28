#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-gateway}"
if [ "$#" -gt 0 ]; then
  shift
fi

export PATH="/home/cellpaint/.local/bin:/opt/conda/envs/cellpainting-claw/bin:/opt/conda/bin:${PATH}"
export OPENCLAW_HOME="${OPENCLAW_HOME:-/workspace/state/openclaw}"
export OPENCLAW_TEMPLATE_CONFIG_PATH="${OPENCLAW_TEMPLATE_CONFIG_PATH:-/opt/CellPainting-Claw/integrations/openclaw/docker/openclaw.container.json}"
export OPENCLAW_RUNTIME_CONFIG_PATH="${OPENCLAW_RUNTIME_CONFIG_PATH:-$OPENCLAW_HOME/openclaw.runtime.json}"
export OPENCLAW_CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-$OPENCLAW_RUNTIME_CONFIG_PATH}"
export CELLPAINT_PIPELINE_ROOT="${CELLPAINT_PIPELINE_ROOT:-/workspace/pipeline}"
export CELLPAINT_CONFIG_PATH="${CELLPAINT_CONFIG_PATH:-/opt/CellPainting-Claw/integrations/openclaw/docker/project_config.docker.json}"
export CELLPAINT_MCP_HOST="${CELLPAINT_MCP_HOST:-127.0.0.1}"
export CELLPAINT_MCP_PORT="${CELLPAINT_MCP_PORT:-8768}"
export CELLPAINT_MCP_PATH="${CELLPAINT_MCP_PATH:-/mcp}"
export OPENCLAW_WORKSPACE="${OPENCLAW_WORKSPACE:-$OPENCLAW_HOME/workspace}"
export OPENCLAW_PROVIDER_ID="${OPENCLAW_PROVIDER_ID:-openai_compatible}"
export OPENCLAW_PROVIDER_API="${OPENCLAW_PROVIDER_API:-openai-completions}"
export OPENCLAW_PRIMARY_MODEL="${OPENCLAW_PRIMARY_MODEL:-gpt-4o}"
export OPENCLAW_FALLBACK_MODEL="${OPENCLAW_FALLBACK_MODEL:-}"

mkdir -p "$OPENCLAW_HOME" "$OPENCLAW_HOME/logs" "$OPENCLAW_WORKSPACE"

bootstrap_workspace() {
  local src="/opt/CellPainting-Claw/integrations/openclaw/workspace"
  local dst="$OPENCLAW_WORKSPACE"
  for name in AGENTS.md SOUL.md TOOLS.md USER.md; do
    if [ ! -f "$dst/$name" ]; then
      cp "$src/$name" "$dst/$name"
    fi
  done
  mkdir -p "$dst/skills"
  if [ ! -d "$dst/skills/cellpaint-pipeline" ]; then
    mkdir -p "$dst/skills/cellpaint-pipeline"
    cp "$src/skills/cellpaint-pipeline/SKILL.md" "$dst/skills/cellpaint-pipeline/SKILL.md"
  fi
}

check_runtime_mounts() {
  if [ ! -d "$CELLPAINT_PIPELINE_ROOT/CellPainting-Claw" ]; then
    echo "Expected pipeline root at $CELLPAINT_PIPELINE_ROOT is missing." >&2
    echo "Mount the host pipeline root into /workspace/pipeline when you start the container." >&2
    exit 1
  fi
}

configure_runtime_config() {
  if [ -f "$OPENCLAW_CONFIG_PATH" ]; then
    return 0
  fi
  if [ -z "${OPENCLAW_BASE_URL:-}" ] || [ -z "${OPENCLAW_API_KEY:-}" ] || [ -z "${OPENCLAW_PRIMARY_MODEL:-}" ]; then
    echo "Missing OpenClaw provider configuration." >&2
    echo "Set OPENCLAW_BASE_URL, OPENCLAW_API_KEY, and OPENCLAW_PRIMARY_MODEL." >&2
    exit 1
  fi

  python3 - <<'PY'
import json
import os
import secrets
from pathlib import Path

template_path = Path(os.environ['OPENCLAW_TEMPLATE_CONFIG_PATH'])
runtime_path = Path(os.environ['OPENCLAW_RUNTIME_CONFIG_PATH'])
provider_id = os.environ['OPENCLAW_PROVIDER_ID']
provider_api = os.environ['OPENCLAW_PROVIDER_API']
base_url = os.environ['OPENCLAW_BASE_URL']
api_key = os.environ['OPENCLAW_API_KEY']
primary_model = os.environ['OPENCLAW_PRIMARY_MODEL']
fallback_model = os.environ.get('OPENCLAW_FALLBACK_MODEL', '').strip()

with template_path.open('r', encoding='utf-8') as fh:
    config = json.load(fh)

models = config.setdefault('models', {})
providers = models.setdefault('providers', {})
provider_models = [{'id': primary_model, 'name': primary_model}]
if fallback_model and fallback_model != primary_model:
    provider_models.append({'id': fallback_model, 'name': fallback_model})
providers[provider_id] = {
    'baseUrl': base_url,
    'apiKey': api_key,
    'api': provider_api,
    'models': provider_models,
}

gateway = config.setdefault('gateway', {})
auth = gateway.setdefault('auth', {})
if auth.get('mode') == 'token' and auth.get('token') in {None, '', 'replace-at-runtime'}:
    auth['token'] = secrets.token_hex(24)

agent_defaults = config.setdefault('agents', {}).setdefault('defaults', {})
model_config = {'primary': f'{provider_id}/{primary_model}'}
if fallback_model:
    model_config['fallbacks'] = [f'{provider_id}/{fallback_model}']
agent_defaults['model'] = model_config

runtime_path.parent.mkdir(parents=True, exist_ok=True)
with runtime_path.open('w', encoding='utf-8') as fh:
    json.dump(config, fh, indent=2)
    fh.write('\n')
PY

  chmod 600 "$OPENCLAW_RUNTIME_CONFIG_PATH"
}

start_mcp() {
  local log_file="$OPENCLAW_HOME/logs/cellpaint_mcp_http.log"
  /opt/conda/envs/cellpainting-claw/bin/python -m cellpaint_pipeline serve-mcp \
    --transport streamable-http \
    --host "$CELLPAINT_MCP_HOST" \
    --port "$CELLPAINT_MCP_PORT" \
    --path "$CELLPAINT_MCP_PATH" \
    >>"$log_file" 2>&1 &
  export CELLPAINT_MCP_PID=$!
}

cleanup() {
  if [ -n "${CELLPAINT_MCP_PID:-}" ]; then
    kill "$CELLPAINT_MCP_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

bootstrap_workspace
check_runtime_mounts
configure_runtime_config
cd "$OPENCLAW_WORKSPACE"

case "$MODE" in
  gateway)
    start_mcp
    exec openclaw gateway start "$@"
    ;;
  tui)
    exec openclaw tui "$@"
    ;;
  acp-client)
    echo "ACP client support is unstable in this OpenClaw release. Launching TUI instead." >&2
    exec openclaw tui "$@"
    ;;
  mcp)
    exec /opt/conda/envs/cellpainting-claw/bin/python -m cellpaint_pipeline serve-mcp \
      --transport streamable-http \
      --host "$CELLPAINT_MCP_HOST" \
      --port "$CELLPAINT_MCP_PORT" \
      --path "$CELLPAINT_MCP_PATH"
    ;;
  shell)
    exec /bin/bash "$@"
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    echo "Supported modes: gateway | tui | acp-client | mcp | shell" >&2
    exit 1
    ;;
esac
