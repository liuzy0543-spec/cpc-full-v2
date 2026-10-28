#!/usr/bin/env bash
set -euo pipefail

if command -v openclaw >/dev/null 2>&1; then
  echo "openclaw is already installed: $(command -v openclaw)"
  exit 0
fi

echo "Installing OpenClaw CLI with the official installer..."
echo "This may also install a newer Node.js runtime if required."
curl -fsSL https://openclaw.ai/install-cli.sh | bash -s -- --no-onboard

echo
echo "Installation finished."
echo "If the binary is not on PATH yet, restart the shell or source the profile updates from the installer."
