# OpenClaw Integration

This directory maintains three OpenClaw tracks:

1. `docker/`
   Preferred on standard Linux hosts with real Docker support.
2. `autodl/`
   Preferred on AutoDL-like hosts where nested Docker is not available.
3. host-local wrappers in this directory
   Thin entrypoints that forward to the sanitized `autodl/` runtime.

## Recommended Path

Choose the runtime by host type:

- use `docker/` on standard Linux hosts with Docker
- use `autodl/` on AutoDL-like platforms

## Current Client Recommendation

For current OpenClaw releases, prefer the TUI path instead of the ACP client path.

## Secret Handling

Provider credentials should be supplied through environment variables or local ignored env files.

Repository-managed JSON templates in this tree do not store provider keys.

## Main References

- `autodl/README.md`
- `docker/README.md`
