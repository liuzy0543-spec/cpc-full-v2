#!/usr/bin/env bash
set -euo pipefail

DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/tmp/docker-data}"
DOCKER_LOG_FILE="${DOCKER_LOG_FILE:-/tmp/dockerd.log}"

mkdir -p "$DOCKER_DATA_ROOT"

if pgrep -x dockerd >/dev/null 2>&1; then
  echo "dockerd is already running"
  exit 0
fi

setsid sh -c "dockerd --iptables=false --bridge=none --ip-forward=false --ip-masq=false --storage-driver=vfs --data-root='$DOCKER_DATA_ROOT' >'$DOCKER_LOG_FILE' 2>&1" >/dev/null 2>&1 < /dev/null &

sleep 2
if docker version >/dev/null 2>&1; then
  echo "dockerd started"
  echo "data_root=$DOCKER_DATA_ROOT"
  echo "log_file=$DOCKER_LOG_FILE"
  exit 0
fi

echo "dockerd did not become ready" >&2
if [ -f "$DOCKER_LOG_FILE" ]; then
  tail -n 60 "$DOCKER_LOG_FILE" >&2 || true
fi
exit 1
