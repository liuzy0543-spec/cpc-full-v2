#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CELLPAINT_PYTHON_BIN:-${PYTHON_BIN:-python}}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/project_config.example.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/dist/release_smoke}"
RUN_FAST_TESTS="${RUN_FAST_TESTS:-0}"

mkdir -p "$OUTPUT_ROOT"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

cd "$ROOT"

echo "[1/3] show-config"
PYTHONPATH=src "$PYTHON_BIN" -m cellpainting_claw show-config \
  --config "$CONFIG_PATH" \
  > "$OUTPUT_ROOT/show_config.json"

echo "[2/3] smoke-test"
PYTHONPATH=src "$PYTHON_BIN" -m cellpainting_claw smoke-test \
  --config "$CONFIG_PATH" \
  --output-path "$OUTPUT_ROOT/smoke_test_report.json" \
  > "$OUTPUT_ROOT/smoke_test_stdout.json"

if [ "$RUN_FAST_TESTS" = "1" ]; then
  echo "[3/3] fast test suite"
  PYTHONPATH=src "$PYTHON_BIN" -m cellpainting_claw.test_suites fast \
    > "$OUTPUT_ROOT/fast_tests.log" 2>&1
else
  echo "[3/3] fast test suite skipped (set RUN_FAST_TESTS=1 to enable)"
fi

echo "Release smoke-test artifacts: $OUTPUT_ROOT"
