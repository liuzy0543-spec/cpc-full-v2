#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATE_DIR="${OPENCLAW_HOME:-$SCRIPT_DIR/state}"
TEMPLATE_PATH="${OPENCLAW_TEMPLATE_PATH:-$SCRIPT_DIR/openclaw.autodl.json}"
RUNTIME_CONFIG_PATH="${OPENCLAW_RUNTIME_CONFIG_PATH:-$STATE_DIR/openclaw.runtime.json}"
PROVIDER_ENV_PATH="${OPENCLAW_PROVIDER_ENV_PATH:-$SCRIPT_DIR/provider.env}"

mkdir -p "$STATE_DIR"

if [ -f "$PROVIDER_ENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$PROVIDER_ENV_PATH"
fi

: "${OPENCLAW_BASE_URL:?OPENCLAW_BASE_URL is required}"
: "${OPENCLAW_API_KEY:?OPENCLAW_API_KEY is required}"
: "${OPENCLAW_PRIMARY_MODEL:?OPENCLAW_PRIMARY_MODEL is required}"

export TEMPLATE_PATH RUNTIME_CONFIG_PATH
export OPENCLAW_PROVIDER_ID="${OPENCLAW_PROVIDER_ID:-openai_compatible}"
export OPENCLAW_PROVIDER_API="${OPENCLAW_PROVIDER_API:-openai-completions}"
export OPENCLAW_BASE_URL OPENCLAW_API_KEY OPENCLAW_PRIMARY_MODEL
export OPENCLAW_FALLBACK_MODEL="${OPENCLAW_FALLBACK_MODEL:-}"
export REPO_ROOT

python3 - <<'PY'
import json
import os
import secrets
from pathlib import Path

template_path = Path(os.environ['TEMPLATE_PATH'])
runtime_path = Path(os.environ['RUNTIME_CONFIG_PATH'])

with template_path.open('r', encoding='utf-8') as fh:
    config = json.load(fh)

repo_root = os.environ['REPO_ROOT']

def render_placeholders(value):
    if isinstance(value, str):
        return value.replace('__REPO_ROOT__', repo_root)
    if isinstance(value, list):
        return [render_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: render_placeholders(item) for key, item in value.items()}
    return value

config = render_placeholders(config)

provider_id = os.environ['OPENCLAW_PROVIDER_ID']
provider_api = os.environ['OPENCLAW_PROVIDER_API']
base_url = os.environ['OPENCLAW_BASE_URL']
api_key = os.environ['OPENCLAW_API_KEY']
primary_model = os.environ['OPENCLAW_PRIMARY_MODEL']
fallback_model = os.environ.get('OPENCLAW_FALLBACK_MODEL', '').strip()

gateway = config.setdefault('gateway', {})
auth = gateway.setdefault('auth', {})
if auth.get('mode') == 'token' and auth.get('token') in {None, '', 'replace-at-runtime'}:
    auth['token'] = secrets.token_hex(24)

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

chmod 600 "$RUNTIME_CONFIG_PATH"
echo "Wrote runtime OpenClaw config to: $RUNTIME_CONFIG_PATH"
