#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [ -f ./.env ]; then
  echo "Using docker env file: $(pwd)/.env"
else
  echo "No local .env found. Copy .env.example to .env or export OPENCLAW_BASE_URL/OPENCLAW_API_KEY/OPENCLAW_PRIMARY_MODEL."
fi
exec docker compose up --build "$@"
