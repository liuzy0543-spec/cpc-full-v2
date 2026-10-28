#!/usr/bin/env bash
set -euo pipefail

DEFAULT_OPENCLAW_BIN="${HOME}/.openclaw/bin/openclaw"
if command -v openclaw >/dev/null 2>&1; then
  echo "openclaw is already installed: $(command -v openclaw)"
  exit 0
fi
if [ -x "$DEFAULT_OPENCLAW_BIN" ]; then
  echo "openclaw is already installed: $DEFAULT_OPENCLAW_BIN"
  exit 0
fi

echo "Installing OpenClaw CLI with the official installer..."
echo "This installs the CLI on the current machine user, while runtime state remains inside this autodl directory."
curl -fsSL https://openclaw.ai/install-cli.sh | bash -s -- --no-onboard

echo
echo "Installation finished."
echo "Detected binary path (expected): $DEFAULT_OPENCLAW_BIN"
echo "The runtime wrapper scripts add $HOME/.openclaw/bin automatically, so a shell restart is not required."
